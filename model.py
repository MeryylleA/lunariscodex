import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging # Import logging

# Setup logger for this module
logger = logging.getLogger("lunaris_model") # Use a specific name for this module's logger

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
        use_flash_attention_if_available=False, # This is a model-level config, not just inference
        layer_norm_epsilon=1e-6,
        ff_multiplier=4,
        pad_token_id=-100
    ):
        if vocab_size is None:
            # This should ideally be caught before config instantiation if it's critical for param estimation
            # but adding a log if it happens.
            logger.warning("vocab_size is None during LunarisCodexConfig init. Parameter estimation might be affected if used before vocab_size is set.")
            # Or raise ValueError("vocab_size must be provided for parameter estimation.") if it's always needed here

        # Parameter estimation (only run if vocab_size is known)
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
                    4 * d_model * d_model +
                    ffn_params_per_layer +
                    4 * d_model
                )
            )

            if estimated_params < 50_000_000:
                original_dropout = dropout
                dropout = min(dropout, 0.05)
                if dropout != original_dropout:
                    logger.info(f"Small model detected (~{estimated_params/1_000_000:.1f}M params), dropout adjusted from {original_dropout} to {dropout}")
        else: # vocab_size is None
            estimated_params = 0 # Cannot estimate
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

class LunarisMind(nn.Module):
    def __init__(self, config: LunarisCodexConfig):
        super().__init__()
        self.config = config
        if self.config.vocab_size is None:
            # This error should be caught by the user of the class ideally.
            # But if it reaches here, it's a critical issue for model construction.
            logger.error("config.vocab_size is None! Model cannot be constructed.")
            raise ValueError("config.vocab_size must be provided for LunarisMind!")

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([
            TransformerDecoderBlock(config) for _ in range(config.n_layers) # LayerScale logs will come from here
        ])
        self.final_layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight # Weight tying

        self._init_alibi_slopes() # ALiBi logs will come from here
        self.apply(self._init_weights)

    def _init_alibi_slopes(self):
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
        if isinstance(module, nn.Linear) and not isinstance(module, LoRALinear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None: nn.init.zeros_(module.bias)
        elif isinstance(module, LoRALinear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02) # Base weights
            if module.bias is not None: nn.init.zeros_(module.bias)
            # LoRA A and B are initialized in LoRALinear.__init__
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight); nn.init.zeros_(module.bias)

        if isinstance(module, SelfAttention) or isinstance(module, FeedForward):
            output_proj_std = 0.02 / math.sqrt(self.config.n_layers) # N_L scaling for stability
            if isinstance(module, SelfAttention):
                nn.init.normal_(module.output_proj.weight, mean=0.0, std=output_proj_std)
                if module.output_proj.bias is not None: nn.init.zeros_(module.output_proj.bias)
            if isinstance(module, FeedForward):
                nn.init.normal_(module.fc2.weight, mean=0.0, std=output_proj_std)
                if module.fc2.bias is not None: nn.init.zeros_(module.fc2.bias)

    def get_alibi_attention_bias(self, seq_len, device):
        causal_mask_bool = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device), diagonal=1)
        pos_indices = torch.arange(seq_len, device=device, dtype=self.alibi_slopes.dtype).unsqueeze(1) - \
                      torch.arange(seq_len, device=device, dtype=self.alibi_slopes.dtype).unsqueeze(0)
        alibi_bias_values = self.alibi_slopes.view(-1, 1, 1) * pos_indices.unsqueeze(0)
        additive_causal_mask = torch.zeros((seq_len, seq_len), dtype=alibi_bias_values.dtype, device=device)
        additive_causal_mask.masked_fill_(causal_mask_bool, float('-inf'))
        return alibi_bias_values + additive_causal_mask.unsqueeze(0)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len = input_ids.size()
        device = input_ids.device
        x = self.token_embedding(input_ids)
        alibi_combined_bias = self.get_alibi_attention_bias(seq_len, device)
        padding_additive_mask = None
        if attention_mask is not None:
            padding_additive_mask = torch.zeros_like(attention_mask, dtype=x.dtype, device=device)
            padding_additive_mask.masked_fill_(attention_mask == 0, float('-inf'))
            padding_additive_mask = padding_additive_mask.unsqueeze(1).unsqueeze(2)
        for layer in self.layers:
            x = layer(x, alibi_combined_bias, padding_additive_mask)
        x = self.final_layer_norm(x)
        logits = self.lm_head(x)
        return logits

    def _apply_repetition_penalty_optimized(self, logits, generated_ids, repetition_penalty):
        if repetition_penalty == 1.0: return logits
        batch_size, vocab_size = logits.shape
        for b_idx in range(batch_size):
            sequence_tokens = generated_ids[b_idx]
            valid_tokens_for_penalty = sequence_tokens
            if self.config.pad_token_id is not None:
                is_not_padding = (sequence_tokens != self.config.pad_token_id)
                valid_tokens_for_penalty = sequence_tokens[is_not_padding]
            if len(valid_tokens_for_penalty) > 0:
                unique_tokens, counts = torch.unique(valid_tokens_for_penalty, return_counts=True)
                for token_val, count in zip(unique_tokens, counts):
                    token = token_val.item()
                    if 0 <= token < vocab_size:
                        if logits[b_idx, token] > 0: logits[b_idx, token] /= (repetition_penalty ** count)
                        else: logits[b_idx, token] *= (repetition_penalty ** count)
        return logits

    def generate(self, input_ids: torch.Tensor, max_new_tokens=50, temperature=None, top_k=None, top_p=None, repetition_penalty=None, eos_token_id=None, pad_token_id=None):
        self.eval()
        device = input_ids.device
        batch_size = input_ids.size(0)

        # Use passed pad_token_id first, then model's config, then tokenizer's (if available via a property)
        # For this method, rely on pad_token_id passed or model.config.pad_token_id
        current_pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id

        temp_to_use = temperature if temperature is not None else self.config.temperature
        top_k_to_use = top_k if top_k is not None else self.config.top_k
        top_p_to_use = top_p if top_p is not None else self.config.top_p
        rep_penalty_to_use = repetition_penalty if repetition_penalty is not None else self.config.repetition_penalty

        generated_ids = input_ids
        is_finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            current_seq_len = generated_ids.size(1)
            current_attention_mask = torch.ones((batch_size, current_seq_len), dtype=torch.long, device=device)
            if current_pad_token_id is not None and current_pad_token_id >= 0 : # Check if pad_token_id is valid
                is_padding = (generated_ids == current_pad_token_id)
                current_attention_mask[is_padding] = 0

            with torch.no_grad():
                logits = self.forward(generated_ids, attention_mask=current_attention_mask)[:, -1, :]

            logits_processed = logits.clone()
            logits_processed = logits_processed / max(temp_to_use, 1e-5) # Avoid division by zero

            if rep_penalty_to_use != 1.0:
                 logits_processed = self._apply_repetition_penalty_optimized(logits_processed, generated_ids, rep_penalty_to_use)

            if top_k_to_use > 0:
                top_k_values, _ = torch.topk(logits_processed, top_k_to_use)
                kth_value = top_k_values[:, -1].unsqueeze(-1)
                logits_processed[logits_processed < kth_value] = float('-inf')

            if 0.0 < top_p_to_use < 1.0: # Valid range for top_p
                sorted_logits, sorted_indices = torch.sort(logits_processed, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p_to_use
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = torch.zeros_like(logits_processed, dtype=torch.bool).scatter_(
                    dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
                logits_processed.masked_fill_(indices_to_remove, float('-inf'))

            if eos_token_id is not None and is_finished.any():
                finished_logits_mask = torch.ones_like(logits_processed[0], dtype=logits_processed.dtype) * float('-inf')
                fill_token = current_pad_token_id if current_pad_token_id is not None and current_pad_token_id != eos_token_id and current_pad_token_id >=0 else eos_token_id
                if fill_token is not None and 0 <= fill_token < logits_processed.size(-1):
                    finished_logits_mask[fill_token] = 0.0
                logits_processed[is_finished] = finished_logits_mask

            probs = F.softmax(logits_processed, dim=-1)
            next_token_candidates = torch.multinomial(probs, num_samples=1)
            generated_ids = torch.cat((generated_ids, next_token_candidates), dim=-1)

            if eos_token_id is not None:
                just_generated_eos = (next_token_candidates.squeeze(-1) == eos_token_id)
                is_finished = is_finished | just_generated_eos
            if is_finished.all(): break

        self.train()
        return generated_ids

class TransformerDecoderBlock(nn.Module):
    def __init__(self, config: LunarisCodexConfig):
        super().__init__()
        self.config = config
        self.ln_1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.attn = SelfAttention(config) # Flash Attention warning will come from here
        self.ln_2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        ffn_intermediate_dim = config.d_model * config.ff_multiplier
        self.ff = FeedForward(config.d_model, ffn_intermediate_dim, config.dropout, config.activation, config.lora_rank)
        self.dropout_res = nn.Dropout(config.dropout)

        # Parameter estimation for LayerScale
        if config.vocab_size is not None: # Only if vocab_size is known
            d_ff = config.d_model * config.ff_multiplier
            if config.activation == "swiglu":
                ffn_params_per_layer = config.d_model * (d_ff * 2) + d_ff * config.d_model
            else: # gelu or other
                ffn_params_per_layer = config.d_model * d_ff + d_ff * config.d_model
            estimated_params_for_layerscale = (
                config.vocab_size * config.d_model +
                config.n_layers * (
                    4 * config.d_model * config.d_model +
                    ffn_params_per_layer +
                    4 * config.d_model
                )
            )
        else: # Cannot estimate if vocab_size is None
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

    def forward(self, x: torch.Tensor, alibi_combined_bias: torch.Tensor, padding_additive_mask: torch.Tensor = None) -> torch.Tensor:
        residual = x
        x_norm = self.ln_1(x)
        attn_output = self.attn(x_norm, alibi_combined_bias, padding_additive_mask)
        if self.use_layerscale: x = residual + self.dropout_res(self.ls_gamma_1 * attn_output)
        else: x = residual + self.dropout_res(attn_output)
        residual = x
        x_norm = self.ln_2(x)
        ff_output = self.ff(x_norm)
        if self.use_layerscale: x = residual + self.dropout_res(self.ls_gamma_2 * ff_output)
        else: x = residual + self.dropout_res(ff_output)
        return x

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
        # Flash Attention check
        self.use_flash_attention = False # Explicitly false due to ALiBi
        if FLASH_ATTENTION_AVAILABLE and config.use_flash_attention_if_available:
            # This message is already logged globally, but can be logged per instance if desired.
            # logger.warning("Flash Attention available but will be disabled in SelfAttention due to ALiBi custom bias incompatibility.")
            pass


    def forward(self, x: torch.Tensor, alibi_combined_bias: torch.Tensor, padding_additive_mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        current_bias = alibi_combined_bias.unsqueeze(0)
        if padding_additive_mask is not None:
            current_bias = current_bias + padding_additive_mask
        attn_scores = attn_scores + current_bias
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.attn_dropout_p, training=self.training)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.output_proj(attn_output)

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

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Configure basic logging for the __main__ block execution if needed
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger.info("Testing LunarisCodex Model Components (Corrected Version with Logging)...")

    test_pad_token_id = 0
    test_config = LunarisCodexConfig(
        vocab_size=1000, d_model=256, n_layers=2, n_heads=4, # Reduced layers for faster test
        max_seq_len=64, lora_rank=8, use_flash_attention_if_available=False,
        pad_token_id=test_pad_token_id, activation="swiglu"
    )
    model = LunarisMind(test_config)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total model parameters: {total_params:,}")

    trainable_params_val = 0
    if test_config.lora_rank > 0:
        for name, param in model.named_parameters():
            if 'lora_' in name or 'ls_gamma' in name:
                param.requires_grad = True; trainable_params_val += param.numel()
            else: param.requires_grad = False
    else:
        for param in model.parameters(): param.requires_grad = True
        trainable_params_val = total_params
    logger.info(f"Trainable parameters (LoRA + LayerScale if active): {trainable_params_val:,} ({trainable_params_val/total_params*100:.2f}%)")

    batch_size, test_seq_len = 2, 32
    dummy_input_ids = torch.randint(test_pad_token_id + 1, test_config.vocab_size, (batch_size, test_seq_len))
    if batch_size > 1 and test_seq_len > 5: dummy_input_ids[1, -5:] = test_pad_token_id
    dummy_attention_mask = (dummy_input_ids != test_pad_token_id).long()

    logger.info(f"\nInput IDs shape: {dummy_input_ids.shape}")
    logger.info(f"Attention Mask shape: {dummy_attention_mask.shape}")

    try:
        logger.info("\nTesting forward pass...")
        logits = model(dummy_input_ids, attention_mask=dummy_attention_mask)
        logger.info(f"Logits output shape: {logits.shape}")

        logger.info("\nTesting generation...")
        prompt_input_ids = dummy_input_ids[:, :5].clone()
        prompt_input_ids[prompt_input_ids == test_pad_token_id] = test_pad_token_id + 1

        # Test generation using the model's generate method
        generated = model.generate(prompt_input_ids, max_new_tokens=5, eos_token_id=test_pad_token_id + 10, pad_token_id=test_config.pad_token_id)
        logger.info(f"Generated IDs shape: {generated.shape}")
        logger.info(f"Generated IDs (first example): {generated[0]}")
        logger.info("✅ Model component tests completed successfully!")
    except Exception as e:
        logger.error(f"❌ ERROR during model component test: {e}", exc_info=True)
