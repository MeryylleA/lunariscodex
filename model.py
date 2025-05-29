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
    logger.info("flash_attn library found, but will be disabled due to ALiBi incompatibility.")
except ImportError:
    logger.warning("flash_attn library not found or could not be imported. Using PyTorch standard attention implementation.")
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
        layer_norm_epsilon=1e-6,
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
        self.use_flash_attention_if_available = use_flash_attention_if_available
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
        self.use_flash_attention = False # Explicitly false due to ALiBi

    def forward(self,
                x: torch.Tensor,
                # alibi_combined_bias is the full bias for (total_len, total_len)
                alibi_combined_bias: torch.Tensor,
                # padding_additive_mask is (B, 1, 1, total_len_kv)
                padding_additive_mask: Optional[torch.Tensor] = None,
                # past_key_value is a tuple (past_key, past_value)
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size, seq_len_q, _ = x.shape # seq_len_q is 1 for cached generation after prompt

        # Project Q, K, V from current input x
        qkv_new = self.qkv_proj(x) # (B, S_q, 3*D)
        qkv_new = qkv_new.view(batch_size, seq_len_q, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k_new, v_new = qkv_new.unbind(0) # q, k_new, v_new are (B, H, S_q, D_h)

        if past_key_value is not None:
            # Concatenate past K, V with new K, V
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k_new], dim=2) # dim=2 is the sequence length for K, V (B, H, S, D_h)
            v = torch.cat([past_v, v_new], dim=2)
        else:
            k = k_new
            v = v_new

        present_key_value = (k, v) # Cache these for the next step

        # Attention scores calculation
        # q: (B, H, S_q, D_h), k: (B, H, S_kv_total, D_h)
        # attn_scores: (B, H, S_q, S_kv_total)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        seq_len_kv_total = k.shape[2] # Total length of keys/values including past

        # Apply ALiBi and Padding Bias
        # alibi_combined_bias is (1 or B, H, S_kv_total, S_kv_total)
        # We need the slice corresponding to current queries (S_q) attending to all keys (S_kv_total)
        # If S_q is 1 (generating one token), queries are at the "end" of the full sequence.
        # The bias for query at pos `i` and key at pos `j` is `alibi_combined_bias[..., i, j]`
        # For current queries (last S_q positions) attending to all S_kv_total keys:
        effective_alibi_bias = alibi_combined_bias[:, :, -seq_len_q:, :seq_len_kv_total]

        current_bias_for_scores = effective_alibi_bias
        if padding_additive_mask is not None:
            # padding_additive_mask is (B, 1, 1, S_kv_total)
            current_bias_for_scores = current_bias_for_scores + padding_additive_mask

        attn_scores = attn_scores + current_bias_for_scores

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.attn_dropout_p, training=self.training)

        attn_output = torch.matmul(attn_weights, v) # (B, H, S_q, D_h)
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

        # Self Attention part
        residual = x
        x_norm = self.ln_1(x)

        attn_outputs = self.attn(
            x_norm,
            alibi_combined_bias=alibi_combined_bias,
            padding_additive_mask=padding_additive_mask,
            past_key_value=past_key_value
        )
        attn_output = attn_outputs[0] # The actual attention output
        present_key_value = attn_outputs[1] # The K,V cache for this layer

        if self.use_layerscale:
            x = residual + self.dropout_res(self.ls_gamma_1 * attn_output)
        else:
            x = residual + self.dropout_res(attn_output)

        # Feed Forward part
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
        # ... (ALiBi slope calculation remains the same)
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

        slopes_list = get_slopes(self.config.n_heads)
        slopes = torch.tensor(slopes_list, dtype=torch.float32)
        self.register_buffer("alibi_slopes", slopes)
        logger.info(f"ALiBi slopes initialized for {self.config.n_heads} heads: {[f'{s:.6f}' for s in slopes_list]}")


    def _init_weights(self, module):
        # ... (Weight initialization remains the same)
        if isinstance(module, nn.Linear) and not isinstance(module, LoRALinear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None: nn.init.zeros_(module.bias)
        elif isinstance(module, LoRALinear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None: nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight); nn.init.zeros_(module.bias)

        if isinstance(module, SelfAttention) or isinstance(module, FeedForward):
            output_proj_std = 0.02 / math.sqrt(self.config.n_layers)
            if isinstance(module, SelfAttention):
                nn.init.normal_(module.output_proj.weight, mean=0.0, std=output_proj_std)
                if module.output_proj.bias is not None: nn.init.zeros_(module.output_proj.bias)
            if isinstance(module, FeedForward):
                nn.init.normal_(module.fc2.weight, mean=0.0, std=output_proj_std)
                if module.fc2.bias is not None: nn.init.zeros_(module.fc2.bias)

    def get_alibi_attention_bias(self, seq_len, device):
        # This generates the full (seq_len, seq_len) ALiBi + Causal mask
        causal_mask_bool = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device), diagonal=1)
        pos_indices = torch.arange(seq_len, device=device, dtype=self.alibi_slopes.dtype).unsqueeze(1) - \
                      torch.arange(seq_len, device=device, dtype=self.alibi_slopes.dtype).unsqueeze(0)
        alibi_bias_values = self.alibi_slopes.view(-1, 1, 1) * pos_indices.unsqueeze(0) # (H, S, S)
        additive_causal_mask = torch.zeros((seq_len, seq_len), dtype=alibi_bias_values.dtype, device=device)
        additive_causal_mask.masked_fill_(causal_mask_bool, float('-inf'))
        # alibi_combined_bias is (H, S, S), needs to be (1, H, S, S) or (B, H, S, S) for broadcasting
        return (alibi_bias_values + additive_causal_mask.unsqueeze(0)).unsqueeze(0) # (1, H, S, S)

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None, # Full attention mask (padding)
                past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
                use_cache: bool = False
                ) -> Tuple[torch.Tensor, Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]]]:

        batch_size, seq_len_q = input_ids.size() # seq_len_q is 1 if use_cache and past_key_values is not None
        device = input_ids.device

        # Determine total sequence length for ALiBi and padding mask
        seq_len_kv_past = 0
        if past_key_values is not None and past_key_values[0] is not None:
            seq_len_kv_past = past_key_values[0][0].shape[2] # (B, H, S_past, D_h) -> S_past

        total_seq_len_kv = seq_len_kv_past + seq_len_q

        # 1. ALiBi Bias: Computed for the total current sequence length
        # This alibi_combined_bias will be (1, n_heads, total_seq_len_kv, total_seq_len_kv)
        # SelfAttention will slice it as needed: bias[:, :, -seq_len_q:, :total_seq_len_kv]
        alibi_combined_bias = self.get_alibi_attention_bias(total_seq_len_kv, device)

        # 2. Padding Mask: Based on the full attention_mask argument
        # attention_mask is (B, total_seq_len_kv), needs to be (B, 1, 1, total_seq_len_kv) for attention scores
        padding_additive_mask = None
        if attention_mask is not None:
            # Ensure attention_mask has the correct total_seq_len_kv if caching
            if attention_mask.shape[1] != total_seq_len_kv:
                # This case needs careful handling in `generate` or here.
                # For now, assume `attention_mask` passed to `forward` is always for the full current context.
                # If input_ids is just the new token, attention_mask should still be for the whole sequence.
                logger.warning(f"Attention mask length ({attention_mask.shape[1]}) does not match total_seq_len_kv ({total_seq_len_kv}). This might be an issue with KV caching setup.")

            padding_additive_mask = torch.zeros_like(attention_mask, dtype=self.token_embedding.weight.dtype, device=device) # Use a consistent dtype
            padding_additive_mask.masked_fill_(attention_mask == 0, float('-inf'))
            padding_additive_mask = padding_additive_mask.unsqueeze(1).unsqueeze(2) # (B, 1, 1, total_seq_len_kv)

        # Embedding
        x = self.token_embedding(input_ids) # (B, S_q, D)

        present_key_values_all_layers = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            layer_past_key_value = past_key_values[i] if past_key_values is not None else None

            layer_outputs = layer(
                x,
                alibi_combined_bias=alibi_combined_bias, # Pass the full bias
                padding_additive_mask=padding_additive_mask, # Pass the full padding mask
                past_key_value=layer_past_key_value,
                use_cache=use_cache
            )
            x = layer_outputs[0]
            if use_cache:
                present_key_values_all_layers.append(layer_outputs[1])

        x = self.final_layer_norm(x)
        logits = self.lm_head(x) # (B, S_q, VocabSize)

        if use_cache:
            return logits, tuple(present_key_values_all_layers)
        else:
            return logits

    def _apply_repetition_penalty_optimized(self, logits, generated_ids_for_penalty, repetition_penalty):
        # generated_ids_for_penalty should be the full sequence of generated IDs for the current batch item
        if repetition_penalty == 1.0: return logits
        # logits are (batch_size, vocab_size) for the current token being generated
        # generated_ids_for_penalty is (batch_size, current_total_seq_len)

        batch_size, vocab_size = logits.shape
        for b_idx in range(batch_size):
            sequence_tokens = generated_ids_for_penalty[b_idx]
            valid_tokens_for_penalty = sequence_tokens

            # Exclude padding from penalty calculation
            # Make sure self.config.pad_token_id is correctly set and used
            if self.config.pad_token_id is not None and self.config.pad_token_id >= 0: # Valid pad_token_id
                is_not_padding = (sequence_tokens != self.config.pad_token_id)
                valid_tokens_for_penalty = sequence_tokens[is_not_padding]

            if len(valid_tokens_for_penalty) > 0:
                unique_tokens, counts = torch.unique(valid_tokens_for_penalty, return_counts=True)
                for token_val, count in zip(unique_tokens, counts):
                    token = token_val.item()
                    if 0 <= token < vocab_size:
                        # Original logic had potential issues. Standard approach:
                        # Score_orig / penalty if score > 0 else Score_orig * penalty
                        # This makes repeated tokens less likely.
                        # Using repetition_penalty ** count might be too aggressive.
                        # Let's use a simpler common formulation first.
                        # For now, keeping your original logic but noting it might be too strong.
                        if logits[b_idx, token] > 0: logits[b_idx, token] /= (repetition_penalty ** count)
                        else: logits[b_idx, token] *= (repetition_penalty ** count)
        return logits

    @torch.no_grad() # Ensure no_grad for generation
    def generate(self,
                 input_ids: torch.Tensor, # Prompt
                 max_new_tokens=50,
                 temperature=None,
                 top_k=None,
                 top_p=None,
                 repetition_penalty=None,
                 eos_token_id=None,
                 pad_token_id=None, # pad_token_id from tokenizer for this specific call
                 attention_mask: Optional[torch.Tensor] = None # Initial attention mask for the prompt
                ):
        self.eval()
        device = input_ids.device
        batch_size = input_ids.size(0)

        # Resolve generation parameters
        current_pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        temp_to_use = temperature if temperature is not None else self.config.temperature
        top_k_to_use = top_k if top_k is not None else self.config.top_k
        top_p_to_use = top_p if top_p is not None else self.config.top_p
        rep_penalty_to_use = repetition_penalty if repetition_penalty is not None else self.config.repetition_penalty

        # Initialize containers for generated sequences and KV cache
        generated_ids_all = input_ids # This will store the full sequences (prompt + generated)
        past_key_values = None

        # Initial attention_mask for the prompt
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=device)
            if current_pad_token_id is not None and current_pad_token_id >= 0:
                 attention_mask[input_ids == current_pad_token_id] = 0


        is_finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for i in range(max_new_tokens):
            # Prepare inputs for the current step
            if past_key_values is None: # First step (processing the prompt)
                current_input_ids = generated_ids_all
                current_attention_mask = attention_mask # Use the initial prompt attention mask
            else: # Subsequent steps (generating one token at a time)
                current_input_ids = generated_ids_all[:, -1:] # Only the last generated token
                # Extend attention_mask for the new token
                # New token is never padding, so append 1
                new_token_mask = torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=device)
                current_attention_mask = torch.cat([attention_mask, new_token_mask], dim=1)
                # Update the main attention_mask for the next iteration's full context
                attention_mask = current_attention_mask


            # Forward pass
            # `current_attention_mask` here is the full mask for (prompt + generated_so_far)
            forward_outputs = self.forward(
                current_input_ids,
                attention_mask=current_attention_mask, # Full mask
                past_key_values=past_key_values,
                use_cache=True
            )

            logits = forward_outputs[0] # Logits for the last token(s) in current_input_ids
            past_key_values = forward_outputs[1] # Update KV cache

            # Get logits for the very last token position
            # If processing prompt (S_q > 1), logits is (B, S_q, V), take last.
            # If generating (S_q = 1), logits is (B, 1, V), take that.
            next_token_logits = logits[:, -1, :] # (B, V)

            # Apply sampling strategies
            next_token_logits_processed = next_token_logits.clone()
            next_token_logits_processed = next_token_logits_processed / max(temp_to_use, 1e-5)

            if rep_penalty_to_use != 1.0:
                # Pass the full generated_ids_all for penalty calculation context
                next_token_logits_processed = self._apply_repetition_penalty_optimized(
                    next_token_logits_processed, generated_ids_all, rep_penalty_to_use
                )

            if top_k_to_use > 0:
                top_k_values, _ = torch.topk(next_token_logits_processed, top_k_to_use)
                kth_value = top_k_values[:, -1].unsqueeze(-1)
                next_token_logits_processed[next_token_logits_processed < kth_value] = float('-inf')

            if 0.0 < top_p_to_use < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits_processed, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p_to_use
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = torch.zeros_like(next_token_logits_processed, dtype=torch.bool).scatter_(
                    dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
                next_token_logits_processed.masked_fill_(indices_to_remove, float('-inf'))

            # Handle finished sequences before sampling
            if eos_token_id is not None and is_finished.any():
                # For finished sequences, force sampling of pad_token (or eos if no pad)
                # This ensures they keep "generating" a non-content token
                finished_logits_mask_fill = torch.ones_like(next_token_logits_processed[0], dtype=next_token_logits_processed.dtype) * float('-inf')

                # Determine the token to fill for finished sequences
                # Prefer pad_token_id if valid and different from eos_token_id, otherwise use eos_token_id
                # This helps distinguish true EOS from padding after EOS.
                fill_token_for_finished = current_pad_token_id
                if fill_token_for_finished is None or fill_token_for_finished < 0 or fill_token_for_finished == eos_token_id:
                    fill_token_for_finished = eos_token_id # Fallback to EOS if pad is not suitable

                if fill_token_for_finished is not None and 0 <= fill_token_for_finished < next_token_logits_processed.size(-1):
                    finished_logits_mask_fill[fill_token_for_finished] = 0.0 # Allow sampling of this token

                next_token_logits_processed[is_finished] = finished_logits_mask_fill


            probs = F.softmax(next_token_logits_processed, dim=-1)
            next_token_candidates = torch.multinomial(probs, num_samples=1) # (B, 1)

            # Append new tokens to the sequences that are not yet finished
            # For finished sequences, they will append the fill_token
            generated_ids_all = torch.cat((generated_ids_all, next_token_candidates), dim=-1)

            # Update finished status for sequences that just generated EOS
            # Only update if not already finished to prevent re-finishing
            if eos_token_id is not None:
                just_generated_eos = (next_token_candidates.squeeze(-1) == eos_token_id) & (~is_finished)
                is_finished = is_finished | just_generated_eos

            if is_finished.all():
                break

        self.train()
        return generated_ids_all


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger.info("Testing LunarisCodex Model Components (KV Cache Version)...")

    test_pad_token_id = 0
    test_config = LunarisCodexConfig(
        vocab_size=100, d_model=64, n_layers=2, n_heads=2,
        max_seq_len=32, lora_rank=4, pad_token_id=test_pad_token_id, activation="swiglu"
    )
    model = LunarisMind(test_config)
    model.eval() # Set to eval for testing generation

    logger.info(f"Total model parameters: {count_parameters(model):,}")

    batch_size, prompt_len = 2, 5
    # Ensure prompt tokens are valid (e.g., > pad_token_id if pad_token_id is 0)
    dummy_prompt_ids = torch.randint(test_pad_token_id + 1, test_config.vocab_size, (batch_size, prompt_len))

    # Create an initial attention mask for the prompt
    # Assume prompt itself doesn't have internal padding for this simple test, or handle it if it does
    initial_attention_mask = torch.ones_like(dummy_prompt_ids)
    # Example: if prompt could have padding:
    # initial_attention_mask[dummy_prompt_ids == test_pad_token_id] = 0


    logger.info(f"\nPrompt IDs shape: {dummy_prompt_ids.shape}")

    # Test model.forward with use_cache=False (standard forward pass)
    try:
        logger.info("\nTesting forward pass (use_cache=False)...")
        logits_no_cache, _ = model(dummy_prompt_ids, attention_mask=initial_attention_mask, use_cache=False)
        logger.info(f"Logits (no_cache) output shape: {logits_no_cache.shape}")
        assert logits_no_cache.shape == (batch_size, prompt_len, test_config.vocab_size)
    except Exception as e:
        logger.error(f"❌ ERROR during forward pass (use_cache=False) test: {e}", exc_info=True)


    # Test model.forward with use_cache=True (initial prompt processing)
    try:
        logger.info("\nTesting forward pass (use_cache=True, initial step)...")
        logits_cache_init, past_kv_init = model(dummy_prompt_ids, attention_mask=initial_attention_mask, use_cache=True)
        logger.info(f"Logits (cache_init) output shape: {logits_cache_init.shape}")
        assert logits_cache_init.shape == (batch_size, prompt_len, test_config.vocab_size)
        logger.info(f"Past KV (init) type: {type(past_kv_init)}, num_layers: {len(past_kv_init) if past_kv_init else 0}")
        if past_kv_init:
            logger.info(f"  Layer 0 KV shapes: K={past_kv_init[0][0].shape}, V={past_kv_init[0][1].shape}")
            # Expected K/V shape: (batch_size, n_heads, prompt_len, head_dim)
            assert past_kv_init[0][0].shape == (batch_size, test_config.n_heads, prompt_len, test_config.d_model // test_config.n_heads)
    except Exception as e:
        logger.error(f"❌ ERROR during forward pass (use_cache=True, initial) test: {e}", exc_info=True)

    # Test model.forward with use_cache=True (next token step)
    if 'past_kv_init' in locals() and past_kv_init is not None:
        try:
            logger.info("\nTesting forward pass (use_cache=True, next token step)...")
            next_token_input = torch.randint(test_pad_token_id + 1, test_config.vocab_size, (batch_size, 1)) # Single new token

            # Attention mask for the next step needs to include the prompt and the new token
            # Prompt mask was `initial_attention_mask` (B, S_prompt)
            # New token mask is `ones(B,1)`
            next_step_attention_mask = torch.cat([initial_attention_mask, torch.ones_like(next_token_input)], dim=1)

            logits_cache_next, past_kv_next = model(
                next_token_input,
                attention_mask=next_step_attention_mask, # Full mask
                past_key_values=past_kv_init,
                use_cache=True
            )
            logger.info(f"Logits (cache_next) output shape: {logits_cache_next.shape}") # Should be (B, 1, V)
            assert logits_cache_next.shape == (batch_size, 1, test_config.vocab_size)
            logger.info(f"Past KV (next) type: {type(past_kv_next)}, num_layers: {len(past_kv_next) if past_kv_next else 0}")
            if past_kv_next:
                logger.info(f"  Layer 0 KV shapes: K={past_kv_next[0][0].shape}, V={past_kv_next[0][1].shape}")
                # Expected K/V shape: (batch_size, n_heads, prompt_len + 1, head_dim)
                assert past_kv_next[0][0].shape == (batch_size, test_config.n_heads, prompt_len + 1, test_config.d_model // test_config.n_heads)

        except Exception as e:
            logger.error(f"❌ ERROR during forward pass (use_cache=True, next token) test: {e}", exc_info=True)


    # Test generation with KV Caching
    try:
        logger.info("\nTesting generation with KV Caching...")
        # Use a different prompt for generate to avoid interference from manual forward tests
        gen_prompt_ids = torch.randint(test_pad_token_id + 1, test_config.vocab_size, (batch_size, prompt_len))
        gen_initial_attention_mask = torch.ones_like(gen_prompt_ids)


        generated_kv = model.generate(
            gen_prompt_ids,
            max_new_tokens=5,
            eos_token_id=test_pad_token_id + 10, # Ensure this is a valid token_id
            pad_token_id=test_config.pad_token_id,
            attention_mask=gen_initial_attention_mask # Pass initial mask
        )
        logger.info(f"Generated IDs (KV cache) shape: {generated_kv.shape}")
        logger.info(f"Generated IDs (KV cache, first example): {generated_kv[0]}")
        assert generated_kv.shape == (batch_size, prompt_len + 5)
        logger.info("✅ Model KV Caching component tests completed successfully (inspect output for correctness)!")
    except Exception as e:
        logger.error(f"❌ ERROR during KV Caching generation test: {e}", exc_info=True)
