import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional, Tuple, List # Added for type hinting

# Setup logger for this module
logger = logging.getLogger("lunaris_model")

try:
    from flash_attn import flash_attn_func
    FLASH_ATTENTION_AVAILABLE = True
    # logger.info("flash_attn library found, but will be disabled due to ALiBi incompatibility.")
except ImportError:
    # logger.warning("flash_attn library not found or could not be imported. Using PyTorch standard attention implementation.")
    FLASH_ATTENTION_AVAILABLE = False

class LunarisCodexConfig:
    def __init__(
        self,
        vocab_size=None,
        d_model=768,
        n_layers=10,
        n_heads=12,
        max_seq_len=1024,
        dropout=0.1,
        activation="swiglu",
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.1,
        lora_rank=32,
        use_flash_attention_if_available=False,
        layer_norm_epsilon=1e-5,
        ff_multiplier=4,
        pad_token_id=-100
    ):
        if vocab_size is None:
            logger.warning("vocab_size is None during LunarisCodexConfig init. Parameter estimation might be affected if used before vocab_size is set.")

        if vocab_size is not None:
            d_ff = d_model * ff_multiplier
            if activation == "swiglu":
                ffn_params_per_layer = d_model * (d_ff * 2) + d_ff * d_model
            elif activation == "gelu":
                ffn_params_per_layer = d_model * d_ff + d_ff * d_model
            else:
                ffn_params_per_layer = d_model * d_ff + d_ff * d_model
                logger.warning(f"Unknown activation '{activation}' for param estimation, assuming GELU-like FFN structure.")

            estimated_params = (
                vocab_size * d_model +
                n_layers * (
                    4 * d_model * d_model + # Simplified attention params
                    ffn_params_per_layer +
                    4 * d_model # Approx LayerNorm params
                )
            )

            if estimated_params < 50_000_000:
                original_dropout = dropout
                dropout = min(dropout, 0.05)
                if dropout != original_dropout:
                    logger.info(f"Small model detected (~{estimated_params/1_000_000:.1f}M params), dropout adjusted from {original_dropout} to {dropout}")
        else:
            estimated_params = 0
            logger.info("Cannot estimate model parameters as vocab_size is None. Dropout adjustment skipped.")

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.activation = activation
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.lora_rank = lora_rank
        self.use_flash_attention_if_available = use_flash_attention_if_available # This is a model-level config
        self.layer_norm_epsilon = layer_norm_epsilon
        self.ff_multiplier = ff_multiplier
        self.pad_token_id = pad_token_id


class LoRALinear(nn.Linear):
    def __init__(self, in_features, out_features, rank=32, alpha=1.0, bias=False):
        super().__init__(in_features, out_features, bias=bias)
        self.has_lora = rank is not None and rank > 0
        if self.has_lora:
            self.rank = rank
            self.lora_alpha = alpha
            self.lora_A = nn.Parameter(torch.Tensor(in_features, rank))
            self.lora_B = nn.Parameter(torch.Tensor(rank, out_features))
            nn.init.normal_(self.lora_A, std=0.02)
            nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_output = super().forward(x)
        if self.has_lora:
            lora_adaptation = (x @ self.lora_A @ self.lora_B) * (self.lora_alpha / self.rank)
            return original_output + lora_adaptation
        return original_output

class SelfAttention(nn.Module):
    def __init__(self, config: LunarisCodexConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        if self.head_dim * self.n_heads != self.d_model:
            msg = f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            logger.error(msg)
            raise ValueError(msg)
        self.qkv_proj = LoRALinear(config.d_model, 3 * config.d_model, rank=config.lora_rank, bias=False)
        self.output_proj = LoRALinear(config.d_model, config.d_model, rank=config.lora_rank, bias=False)
        self.attn_dropout_p = config.dropout
        self.use_flash_attention = False

    def forward(self,
                x: torch.Tensor,
                alibi_combined_bias: torch.Tensor,
                padding_additive_mask: Optional[torch.Tensor] = None,
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size, seq_len_q, _ = x.shape

        qkv_new = self.qkv_proj(x)
        qkv_new = qkv_new.view(batch_size, seq_len_q, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k_new, v_new = qkv_new.unbind(0)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k_new], dim=2)
            v = torch.cat([past_v, v_new], dim=2)
        else:
            k = k_new
            v = v_new

        present_key_value = (k, v)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        seq_len_kv_total = k.shape[2]

        effective_alibi_bias = alibi_combined_bias[:, :, -seq_len_q:, :seq_len_kv_total]

        current_bias_for_scores = effective_alibi_bias
        if padding_additive_mask is not None:
            current_bias_for_scores = current_bias_for_scores + padding_additive_mask

        attn_scores = attn_scores + current_bias_for_scores

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.attn_dropout_p, training=self.training)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.d_model)

        return self.output_proj(attn_output), present_key_value


class TransformerDecoderBlock(nn.Module):
    def __init__(self, config: LunarisCodexConfig):
        super().__init__()
        self.config = config
        self.ln_1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.attn = SelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        ffn_intermediate_dim = config.d_model * config.ff_multiplier
        self.ff = FeedForward(config.d_model, ffn_intermediate_dim, config.dropout, config.activation, config.lora_rank)
        self.dropout_res = nn.Dropout(config.dropout)

        if config.vocab_size is not None:
            d_ff = config.d_model * config.ff_multiplier
            if config.activation == "swiglu":
                ffn_params_per_layer = config.d_model * (d_ff * 2) + d_ff * config.d_model
            else:
                ffn_params_per_layer = config.d_model * d_ff + d_ff * config.d_model
            estimated_params_for_layerscale = (
                config.vocab_size * config.d_model +
                config.n_layers * (
                    4 * config.d_model * config.d_model +
                    ffn_params_per_layer +
                    4 * config.d_model
                )
            )
        else:
            estimated_params_for_layerscale = 0
            logger.info("Cannot estimate model parameters for LayerScale decision as vocab_size is None.")

        self.use_layerscale = estimated_params_for_layerscale > 50_000_000
        if self.use_layerscale:
            init_val = 1e-4 if estimated_params_for_layerscale > 100_000_000 else 0.1
            self.ls_gamma_1 = nn.Parameter(torch.ones(config.d_model) * init_val)
            self.ls_gamma_2 = nn.Parameter(torch.ones(config.d_model) * init_val)
            logger.info(f"LayerScale enabled with init_val: {init_val} (model ~{estimated_params_for_layerscale/1_000_000:.1f}M params)")
        else:
            logger.info(f"LayerScale disabled for model (~{estimated_params_for_layerscale/1_000_000:.1f}M params or vocab_size unknown)")

    def forward(self,
                x: torch.Tensor,
                alibi_combined_bias: torch.Tensor,
                padding_additive_mask: Optional[torch.Tensor] = None,
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache: bool = False
                ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:

        residual = x
        x_norm = self.ln_1(x)

        attn_outputs = self.attn(
            x_norm,
            alibi_combined_bias=alibi_combined_bias,
            padding_additive_mask=padding_additive_mask,
            past_key_value=past_key_value
        )
        attn_output = attn_outputs[0]
        present_key_value = attn_outputs[1]

        if self.use_layerscale:
            x = residual + self.dropout_res(self.ls_gamma_1 * attn_output)
        else:
            x = residual + self.dropout_res(attn_output)

        residual = x
        x_norm = self.ln_2(x)
        ff_output = self.ff(x_norm)
        if self.use_layerscale:
            x = residual + self.dropout_res(self.ls_gamma_2 * ff_output)
        else:
            x = residual + self.dropout_res(ff_output)

        if use_cache:
            return x, present_key_value
        else:
            return x, None


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout=0.1, activation="swiglu", lora_rank=32):
        super().__init__()
        self.activation = activation
        fc1_out_dim = d_ff * 2 if activation == "swiglu" else d_ff
        self.fc1 = LoRALinear(d_model, fc1_out_dim, rank=lora_rank, bias=False)
        self.fc2 = LoRALinear(d_ff, d_model, rank=lora_rank, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "swiglu":
            hidden = self.fc1(x)
            gated, activated = hidden.chunk(2, dim=-1)
            x = F.silu(gated) * activated
        elif self.activation == "gelu":
            x = F.gelu(self.fc1(x))
        else:
            msg = f"Unsupported activation: {self.activation}"
            logger.error(msg)
            raise ValueError(msg)
        x = self.dropout(x)
        return self.fc2(x)


class LunarisMind(nn.Module):
    def __init__(self, config: LunarisCodexConfig):
        super().__init__()
        self.config = config
        if self.config.vocab_size is None:
            logger.error("config.vocab_size is None! Model cannot be constructed.")
            raise ValueError("config.vocab_size must be provided for LunarisMind!")

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([
            TransformerDecoderBlock(config) for _ in range(config.n_layers)
        ])
        self.final_layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        self._init_alibi_slopes()
        self.apply(self._init_weights)

    def _init_alibi_slopes(self):
        """Initializes ALiBi slopes, ensuring they are negative for penalization."""
        def get_slopes(n_heads):
            def get_slopes_power_of_2(n):
                start = 2**(-(2**-(math.log2(n)-3)))
                ratio = 2**(1/n)
                return [start/(ratio**i) for i in range(n)]

            if n_heads <= 8:
                return [2 ** (-8.0 / n_heads * (i + 1)) for i in range(n_heads)]
            else:
                if math.log2(n_heads).is_integer():
                    return get_slopes_power_of_2(n_heads)
                else:
                    closest_power = 2 ** round(math.log2(n_heads))
                    base_slopes = get_slopes_power_of_2(closest_power)
                    if n_heads > closest_power:
                        extra_slopes = []
                        for i in range(n_heads - closest_power):
                            ratio = base_slopes[-1] / base_slopes[-2] if len(base_slopes) > 1 else 0.5
                            extra_slopes.append(base_slopes[-1] * ratio)
                        return base_slopes + extra_slopes
                    else:
                        return base_slopes[:n_heads]

        slopes_list_positive = get_slopes(self.config.n_heads)
        slopes_list_negative = [-abs(s) for s in slopes_list_positive]

        slopes = torch.tensor(slopes_list_negative, dtype=torch.float32)
        self.register_buffer("alibi_slopes", slopes)
        logger.info(f"ALiBi slopes (ensured negative) initialized for {self.config.n_heads} heads: {[f'{s:.6f}' for s in slopes_list_negative]}")


    def _init_weights(self, module):
        """Applies custom weight initialization."""
        if isinstance(module, nn.Linear) and not isinstance(module, LoRALinear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None: nn.init.zeros_(module.bias)
        elif isinstance(module, LoRALinear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None: nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

        if isinstance(module, SelfAttention) or isinstance(module, FeedForward):
            output_proj_std = 0.02 / math.sqrt(self.config.n_layers)
            if isinstance(module, SelfAttention):
                nn.init.normal_(module.output_proj.weight, mean=0.0, std=output_proj_std)
                if module.output_proj.bias is not None: nn.init.zeros_(module.output_proj.bias)
            if isinstance(module, FeedForward):
                nn.init.normal_(module.fc2.weight, mean=0.0, std=output_proj_std)
                if module.fc2.bias is not None: nn.init.zeros_(module.fc2.bias)

    def get_alibi_attention_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Constructs the ALiBi attention bias tensor.
        The bias is m_h * (i - j) for query i and key j, where m_h are negative slopes.
        This results in a negative penalty for past positions (j < i), increasing with distance.
        Future positions (j > i) are handled by the additive causal mask.
        """
        query_positions = torch.arange(seq_len, device=device, dtype=self.alibi_slopes.dtype).unsqueeze(1)
        key_positions = torch.arange(seq_len, device=device, dtype=self.alibi_slopes.dtype).unsqueeze(0)
        relative_positions = query_positions - key_positions
        alibi_bias_values = self.alibi_slopes.view(-1, 1, 1) * relative_positions.unsqueeze(0)

        causal_mask_bool = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device), diagonal=1)
        additive_causal_mask = torch.zeros((seq_len, seq_len), dtype=alibi_bias_values.dtype, device=device)
        additive_causal_mask.masked_fill_(causal_mask_bool, float('-inf'))

        combined_bias = alibi_bias_values + additive_causal_mask.unsqueeze(0)
        return combined_bias.unsqueeze(0)

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
                use_cache: bool = False
                ) -> Tuple[torch.Tensor, Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]]] | torch.Tensor:

        batch_size, seq_len_q = input_ids.size()
        device = input_ids.device

        seq_len_kv_past = 0
        if past_key_values is not None and past_key_values[0] is not None and len(past_key_values[0]) == 2:
            seq_len_kv_past = past_key_values[0][0].shape[2]

        total_seq_len_kv = seq_len_kv_past + seq_len_q

        alibi_combined_bias = self.get_alibi_attention_bias(total_seq_len_kv, device)

        padding_additive_mask = None
        if attention_mask is not None:
            if attention_mask.shape[1] != total_seq_len_kv:
                logger.warning(
                    f"Attention mask length ({attention_mask.shape[1]}) in forward() does not match "
                    f"total_seq_len_kv ({total_seq_len_kv}). This might indicate an issue with "
                    f"how attention_mask is managed during KV caching in the generate loop."
                )
            padding_additive_mask = torch.zeros_like(attention_mask, dtype=self.token_embedding.weight.dtype, device=device)
            padding_additive_mask.masked_fill_(attention_mask == 0, float('-inf'))
            padding_additive_mask = padding_additive_mask.unsqueeze(1).unsqueeze(2)

        x = self.token_embedding(input_ids)

        present_key_values_all_layers = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            layer_past_key_value = past_key_values[i] if past_key_values is not None else None

            layer_outputs = layer(
                x,
                alibi_combined_bias=alibi_combined_bias,
                padding_additive_mask=padding_additive_mask,
                past_key_value=layer_past_key_value,
                use_cache=use_cache
            )
            x = layer_outputs[0]
            if use_cache:
                if present_key_values_all_layers is not None :
                    present_key_values_all_layers.append(layer_outputs[1])

        x = self.final_layer_norm(x)
        logits = self.lm_head(x)

        if use_cache:
            return logits, tuple(present_key_values_all_layers) if present_key_values_all_layers is not None else None
        else:
            return logits

    def _apply_repetition_penalty_optimized(self,
                                            logits: torch.Tensor,
                                            # This should now receive only the tokens generated BY THE MODEL so far in this sequence
                                            model_generated_tokens_for_penalty: torch.Tensor,
                                            repetition_penalty: float
                                            ) -> torch.Tensor:
        """Applies repetition penalty to the logits, considering only model-generated tokens."""
        if repetition_penalty == 1.0:
            return logits

        batch_size, vocab_size = logits.shape

        for b_idx in range(batch_size):
            # Use the passed model_generated_tokens_for_penalty for this batch item
            sequence_tokens_to_check = model_generated_tokens_for_penalty[b_idx]

            valid_tokens_for_penalty = sequence_tokens_to_check # Already excludes prompt

            # Still good to exclude padding from the generated part, if any (though less likely here)
            if self.config.pad_token_id is not None and self.config.pad_token_id >= 0:
                is_not_padding = (sequence_tokens_to_check != self.config.pad_token_id)
                valid_tokens_for_penalty = sequence_tokens_to_check[is_not_padding]

            if len(valid_tokens_for_penalty) > 0:
                unique_tokens_in_sequence = torch.unique(valid_tokens_for_penalty)

                for token_id_to_penalize_tensor in unique_tokens_in_sequence:
                    token_id = token_id_to_penalize_tensor.item()
                    if 0 <= token_id < vocab_size:
                        if logits[b_idx, token_id] > 0:
                            logits[b_idx, token_id] /= repetition_penalty
                        else:
                            logits[b_idx, token_id] *= repetition_penalty
        return logits

    @torch.no_grad()
    def generate(self,
                 input_ids: torch.Tensor,
                 max_new_tokens: int = 50,
                 temperature: Optional[float] = None,
                 top_k: Optional[int] = None,
                 top_p: Optional[float] = None,
                 repetition_penalty: Optional[float] = None,
                 eos_token_id: Optional[int] = None,
                 pad_token_id: Optional[int] = None,
                 attention_mask: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        self.eval()
        device = input_ids.device
        batch_size = input_ids.size(0)
        prompt_len = input_ids.shape[1] # Store original prompt length

        current_pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        temp_to_use = temperature if temperature is not None else self.config.temperature
        top_k_to_use = top_k if top_k is not None else self.config.top_k
        top_p_to_use = top_p if top_p is not None else self.config.top_p
        rep_penalty_to_use = repetition_penalty if repetition_penalty is not None else self.config.repetition_penalty

        generated_ids_all = input_ids
        past_key_values = None

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=device, dtype=torch.long)
            if current_pad_token_id is not None and current_pad_token_id >= 0:
                 attention_mask[input_ids == current_pad_token_id] = 0

        is_finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for i in range(max_new_tokens):
            if past_key_values is None:
                current_input_ids = generated_ids_all
                current_attention_mask_for_forward = attention_mask
            else:
                current_input_ids = generated_ids_all[:, -1:]
                new_token_mask = torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=device)
                current_attention_mask_for_forward = torch.cat([attention_mask, new_token_mask], dim=1)
                attention_mask = current_attention_mask_for_forward


            model_outputs = self.forward(
                input_ids=current_input_ids,
                attention_mask=current_attention_mask_for_forward,
                past_key_values=past_key_values,
                use_cache=True
            )

            logits = model_outputs[0]
            past_key_values = model_outputs[1]

            next_token_logits = logits[:, -1, :]

            next_token_logits_processed = next_token_logits.clone()
            next_token_logits_processed = next_token_logits_processed / max(temp_to_use, 1e-5)

            if rep_penalty_to_use != 1.0:
                # CRITICAL FIX: Only pass model-generated tokens for repetition penalty context
                if generated_ids_all.shape[1] > prompt_len: # Only apply if there are generated tokens
                    model_generated_tokens = generated_ids_all[:, prompt_len:]
                    next_token_logits_processed = self._apply_repetition_penalty_optimized(
                        next_token_logits_processed, model_generated_tokens, rep_penalty_to_use
                    )
                # If only prompt, no model-generated tokens yet, so no repetition penalty from model's output.

            if top_k_to_use > 0:
                top_k_values, _ = torch.topk(next_token_logits_processed, top_k_to_use)
                kth_value = top_k_values[:, -1].unsqueeze(-1)
                next_token_logits_processed[next_token_logits_processed < kth_value] = float('-inf')

            if 0.0 < top_p_to_use < 1.0:
                sorted_logits_val, sorted_indices = torch.sort(next_token_logits_processed, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits_val, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p_to_use
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = torch.zeros_like(next_token_logits_processed, dtype=torch.bool).scatter_(
                    dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
                next_token_logits_processed.masked_fill_(indices_to_remove, float('-inf'))

            if eos_token_id is not None and is_finished.any():
                finished_logits_mask_fill = torch.ones_like(next_token_logits_processed[0], dtype=next_token_logits_processed.dtype) * float('-inf')
                fill_token_for_finished = current_pad_token_id
                if fill_token_for_finished is None or fill_token_for_finished < 0 or fill_token_for_finished == eos_token_id:
                    fill_token_for_finished = eos_token_id
                if fill_token_for_finished is not None and 0 <= fill_token_for_finished < next_token_logits_processed.size(-1):
                    finished_logits_mask_fill[fill_token_for_finished] = 0.0
                next_token_logits_processed[is_finished] = finished_logits_mask_fill

            probs = F.softmax(next_token_logits_processed, dim=-1)
            next_token_candidates = torch.multinomial(probs, num_samples=1)

            generated_ids_all = torch.cat((generated_ids_all, next_token_candidates), dim=-1)

            if eos_token_id is not None:
                just_generated_eos = (next_token_candidates.squeeze(-1) == eos_token_id) & (~is_finished)
                is_finished = is_finished | just_generated_eos

            if is_finished.all():
                break

        self.train()
        return generated_ids_all


def count_parameters(model: nn.Module) -> int:
    """Counts the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # This block is for testing model components.
    # It is recommended to run dedicated test scripts (e.g., using pytest) for thorough testing.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger.info("Testing LunarisCodex Model Components (KV Cache & Corrected ALiBi/RepPenalty Version)...")

    test_pad_token_id = 0
    test_config = LunarisCodexConfig(
        vocab_size=100, d_model=64, n_layers=2, n_heads=2,
        max_seq_len=32, lora_rank=4, pad_token_id=test_pad_token_id, activation="swiglu"
    )
    model = LunarisMind(test_config)
    model.eval()

    logger.info(f"Total model parameters: {count_parameters(model):,}")

    batch_size, prompt_len_test = 2, 5 # Renamed prompt_len to avoid conflict with generate's prompt_len
    dummy_prompt_ids = torch.randint(test_pad_token_id + 1, test_config.vocab_size, (batch_size, prompt_len_test))

    initial_attention_mask = torch.ones_like(dummy_prompt_ids, dtype=torch.long)
    if test_pad_token_id >=0:
        initial_attention_mask[dummy_prompt_ids == test_pad_token_id] = 0

    logger.info(f"\nPrompt IDs shape: {dummy_prompt_ids.shape}")

    try:
        logger.info("\nTesting forward pass (use_cache=False)...")
        logits_no_cache = model(dummy_prompt_ids, attention_mask=initial_attention_mask, use_cache=False)
        logger.info(f"Logits (no_cache) output shape: {logits_no_cache.shape}")
        assert logits_no_cache.shape == (batch_size, prompt_len_test, test_config.vocab_size)
    except Exception as e:
        logger.error(f"❌ ERROR during forward pass (use_cache=False) test: {e}", exc_info=True)

    try:
        logger.info("\nTesting forward pass (use_cache=True, initial step)...")
        logits_cache_init, past_kv_init = model(dummy_prompt_ids, attention_mask=initial_attention_mask, use_cache=True)
        logger.info(f"Logits (cache_init) output shape: {logits_cache_init.shape}")
        assert logits_cache_init.shape == (batch_size, prompt_len_test, test_config.vocab_size)
        logger.info(f"Past KV (init) type: {type(past_kv_init)}, num_layers: {len(past_kv_init) if past_kv_init else 0}")
        if past_kv_init:
            logger.info(f"  Layer 0 KV shapes: K={past_kv_init[0][0].shape}, V={past_kv_init[0][1].shape}")
            assert past_kv_init[0][0].shape == (batch_size, test_config.n_heads, prompt_len_test, test_config.d_model // test_config.n_heads)
    except Exception as e:
        logger.error(f"❌ ERROR during forward pass (use_cache=True, initial) test: {e}", exc_info=True)

    if 'past_kv_init' in locals() and past_kv_init is not None:
        try:
            logger.info("\nTesting forward pass (use_cache=True, next token step)...")
            next_token_input = torch.randint(test_pad_token_id + 1, test_config.vocab_size, (batch_size, 1))

            next_step_attention_mask = torch.cat([initial_attention_mask, torch.ones_like(next_token_input, dtype=torch.long)], dim=1)

            logits_cache_next, past_kv_next = model(
                next_token_input,
                attention_mask=next_step_attention_mask,
                past_key_values=past_kv_init,
                use_cache=True
            )
            logger.info(f"Logits (cache_next) output shape: {logits_cache_next.shape}")
            assert logits_cache_next.shape == (batch_size, 1, test_config.vocab_size)
            logger.info(f"Past KV (next) type: {type(past_kv_next)}, num_layers: {len(past_kv_next) if past_kv_next else 0}")
            if past_kv_next:
                logger.info(f"  Layer 0 KV shapes: K={past_kv_next[0][0].shape}, V={past_kv_next[0][1].shape}")
                assert past_kv_next[0][0].shape == (batch_size, test_config.n_heads, prompt_len_test + 1, test_config.d_model // test_config.n_heads)

        except Exception as e:
            logger.error(f"❌ ERROR during forward pass (use_cache=True, next token) test: {e}", exc_info=True)

    try:
        logger.info("\nTesting generation with KV Caching (and fixes)...")
        gen_prompt_ids = torch.randint(test_pad_token_id + 1, test_config.vocab_size, (batch_size, prompt_len_test))
        gen_initial_attention_mask = torch.ones_like(gen_prompt_ids, dtype=torch.long)
        if test_pad_token_id >=0:
             gen_initial_attention_mask[gen_prompt_ids == test_pad_token_id] = 0

        generated_kv = model.generate(
            gen_prompt_ids,
            max_new_tokens=5,
            eos_token_id=test_pad_token_id + 10,
            pad_token_id=test_config.pad_token_id,
            attention_mask=gen_initial_attention_mask
        )
        logger.info(f"Generated IDs (KV cache) shape: {generated_kv.shape}")
        logger.info(f"Generated IDs (KV cache, first example): {generated_kv[0]}")
        assert generated_kv.shape == (batch_size, prompt_len_test + 5)
        logger.info("✅ Model KV Caching, ALiBi, and Repetition Penalty component tests completed successfully (inspect output for correctness)!")
    except Exception as e:
        logger.error(f"❌ ERROR during KV Caching generation test (with fixes): {e}", exc_info=True)
