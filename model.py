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
        # Flash Attention check - explicitly disabled if ALiBi is used, as ALiBi adds a custom bias.
        self.use_flash_attention = False
        # The global FLASH_ATTENTION_AVAILABLE check and config.use_flash_attention_if_available
        # are noted at the top level, but ALiBi's custom bias makes standard flash_attn_func incompatible.
        # If a flash attention version compatible with additive bias becomes available, this could be revisited.

    def forward(self,
                x: torch.Tensor,
                alibi_combined_bias: torch.Tensor, # Full bias for (total_len, total_len)
                padding_additive_mask: Optional[torch.Tensor] = None, # (B, 1, 1, total_len_kv)
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None # (past_key, past_value)
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size, seq_len_q, _ = x.shape # seq_len_q is 1 for cached generation after prompt

        qkv_new = self.qkv_proj(x) # (B, S_q, 3*D)
        qkv_new = qkv_new.view(batch_size, seq_len_q, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k_new, v_new = qkv_new.unbind(0) # q, k_new, v_new are (B, H, S_q, D_h)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k_new], dim=2) # dim=2 is the sequence length for K, V (B, H, S_total, D_h)
            v = torch.cat([past_v, v_new], dim=2)
        else:
            k = k_new
            v = v_new

        present_key_value = (k, v) # Cache these for the next step

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        seq_len_kv_total = k.shape[2] # Total length of keys/values including past

        # Slice the ALiBi bias: queries are the last seq_len_q elements, keys are all seq_len_kv_total elements
        # alibi_combined_bias is (1, H, S_total, S_total)
        # We need the bias for queries at positions [-seq_len_q:] attending to keys at positions [:seq_len_kv_total]
        effective_alibi_bias = alibi_combined_bias[:, :, -seq_len_q:, :seq_len_kv_total]

        current_bias_for_scores = effective_alibi_bias
        if padding_additive_mask is not None:
            # padding_additive_mask is (B, 1, 1, S_kv_total)
            # It needs to be broadcastable with effective_alibi_bias (1 or B, H, S_q, S_kv_total)
            # and attn_scores (B, H, S_q, S_kv_total)
            # If padding_additive_mask is (B,1,1,S_kv_total), it will broadcast over H and S_q.
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

        # Parameter estimation for LayerScale decision
        if config.vocab_size is not None:
            d_ff = config.d_model * config.ff_multiplier
            if config.activation == "swiglu":
                ffn_params_per_layer = config.d_model * (d_ff * 2) + d_ff * config.d_model
            else: # gelu or other
                ffn_params_per_layer = config.d_model * d_ff + d_ff * config.d_model
            estimated_params_for_layerscale = (
                config.vocab_size * config.d_model +
                config.n_layers * (
                    4 * config.d_model * config.d_model + # Self-Attention (QKV+Output Proj, simplified)
                    ffn_params_per_layer + # FeedForward
                    4 * config.d_model # Layer Norms (approx.)
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
                use_cache: bool = False # Flag to control KV caching behavior
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
            # For training or when not using cache, only return the hidden states
            return x, None


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout=0.1, activation="swiglu", lora_rank=32):
        super().__init__()
        self.activation = activation
        fc1_out_dim = d_ff * 2 if activation == "swiglu" else d_ff # For SwiGLU, fc1 outputs to 2*d_ff
        self.fc1 = LoRALinear(d_model, fc1_out_dim, rank=lora_rank, bias=False)
        self.fc2 = LoRALinear(d_ff, d_model, rank=lora_rank, bias=False) # fc2 input is d_ff
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "swiglu":
            hidden = self.fc1(x)
            gated, activated = hidden.chunk(2, dim=-1) # Split for SwiGLU
            x = F.silu(gated) * activated # SwiGLU activation
        elif self.activation == "gelu":
            x = F.gelu(self.fc1(x)) # GELU activation
        else:
            msg = f"Unsupported activation: {self.activation}"
            logger.error(msg)
            raise ValueError(msg)
        x = self.dropout(x) # Apply dropout
        return self.fc2(x) # Second linear layer


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
        # Weight Tying
        self.lm_head.weight = self.token_embedding.weight

        self._init_alibi_slopes() # Initialize ALiBi slopes
        self.apply(self._init_weights) # Apply custom weight initialization

    def _init_alibi_slopes(self):
        """Initializes ALiBi slopes, ensuring they are negative for penalization."""
        def get_slopes(n_heads): # Original get_slopes function
            def get_slopes_power_of_2(n):
                # This formula is from the ALiBi paper, designed to generate a geometric sequence
                # The paper uses m = 1/2^(8/N) for N heads, then powers of m.
                # This implementation is slightly different but aims for a similar geometric progression.
                start = 2**(-(2**-(math.log2(n)-3)))
                ratio = 2**(1/n)
                return [start/(ratio**i) for i in range(n)]

            if n_heads <= 8:
                # Original ALiBi paper suggests slopes like m, m^2, ..., m^(n_heads)
                # where m = 2^(-8/n_heads).
                # This implementation is 2^(-8/n_heads * (i+1)) which is m^(i+1).
                return [2 ** (-8.0 / n_heads * (i + 1)) for i in range(n_heads)]
            else: # n_heads > 8
                if math.log2(n_heads).is_integer(): # Power of 2
                    return get_slopes_power_of_2(n_heads)
                else: # Not a power of 2, interpolate/extrapolate
                    closest_power = 2 ** round(math.log2(n_heads))
                    base_slopes = get_slopes_power_of_2(closest_power)
                    if n_heads > closest_power: # Extrapolate
                        extra_slopes = []
                        for i in range(n_heads - closest_power):
                            # Geometric extrapolation
                            ratio = base_slopes[-1] / base_slopes[-2] if len(base_slopes) > 1 else 0.5
                            extra_slopes.append(base_slopes[-1] * ratio)
                        return base_slopes + extra_slopes
                    else: # Truncate
                        return base_slopes[:n_heads]

        slopes_list_positive = get_slopes(self.config.n_heads)
        # CRITICAL FIX: Ensure slopes are negative for ALiBi penalization
        # The ALiBi paper uses negative slopes (m < 1, then m, m^2 etc. applied with a negative sign or directly as m*(j-i))
        # to penalize distance. If get_slopes returns positive decreasing values, make them negative.
        slopes_list_negative = [-abs(s) for s in slopes_list_positive]

        slopes = torch.tensor(slopes_list_negative, dtype=torch.float32)
        self.register_buffer("alibi_slopes", slopes) # Store as a buffer
        logger.info(f"ALiBi slopes (ensured negative) initialized for {self.config.n_heads} heads: {[f'{s:.6f}' for s in slopes_list_negative]}")


    def _init_weights(self, module):
        """Applies custom weight initialization."""
        if isinstance(module, nn.Linear) and not isinstance(module, LoRALinear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None: nn.init.zeros_(module.bias)
        elif isinstance(module, LoRALinear):
            # Base weights of LoRALinear are initialized here
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None: nn.init.zeros_(module.bias)
            # LoRA A and B are initialized in LoRALinear.__init__
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight) # gamma to 1
            nn.init.zeros_(module.bias)  # beta to 0

        # Special initialization for output projections in SelfAttention and FeedForward
        # This helps stabilize training in deep transformers (GPT-2/3 style)
        if isinstance(module, SelfAttention) or isinstance(module, FeedForward):
            # Scale down std for weights of output projection layers
            output_proj_std = 0.02 / math.sqrt(self.config.n_layers)
            if isinstance(module, SelfAttention):
                nn.init.normal_(module.output_proj.weight, mean=0.0, std=output_proj_std)
                if module.output_proj.bias is not None: nn.init.zeros_(module.output_proj.bias)
            if isinstance(module, FeedForward): # Applies to fc2 in FeedForward
                nn.init.normal_(module.fc2.weight, mean=0.0, std=output_proj_std)
                if module.fc2.bias is not None: nn.init.zeros_(module.fc2.bias)

    def get_alibi_attention_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Constructs the ALiBi attention bias tensor.
        The bias is m_h * (i - j) for query i and key j, where m_h are negative slopes.
        This results in a negative penalty for past positions (j < i), increasing with distance.
        Future positions (j > i) are handled by the additive causal mask.
        """
        # self.alibi_slopes are now guaranteed to be negative.

        # query_positions[q_idx] = q_idx
        query_positions = torch.arange(seq_len, device=device, dtype=self.alibi_slopes.dtype).unsqueeze(1)
        # key_positions[k_idx] = k_idx
        key_positions = torch.arange(seq_len, device=device, dtype=self.alibi_slopes.dtype).unsqueeze(0)

        # relative_positions[q_idx, k_idx] = q_idx - k_idx (i.e., i - j)
        # For past keys (k_idx < q_idx), q_idx - k_idx is positive.
        # For current key (k_idx = q_idx), q_idx - k_idx is 0.
        # For future keys (k_idx > q_idx), q_idx - k_idx is negative.
        relative_positions = query_positions - key_positions # Shape (seq_len, seq_len)

        # ALiBi bias: m_h * (i - j)
        # Since self.alibi_slopes (m_h) are negative:
        # - For past keys (i-j > 0): bias is negative (penalty, magnitude increases with distance i-j). Correct.
        # - For current key (i-j = 0): bias is 0. Correct.
        # - For future keys (i-j < 0): bias is positive. These will be masked by causal mask.
        alibi_bias_values = self.alibi_slopes.view(-1, 1, 1) * relative_positions.unsqueeze(0) # Shape (n_heads, seq_len, seq_len)

        # Causal mask: -inf for future positions (j > i), 0 otherwise.
        causal_mask_bool = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device), diagonal=1)
        additive_causal_mask = torch.zeros((seq_len, seq_len), dtype=alibi_bias_values.dtype, device=device)
        additive_causal_mask.masked_fill_(causal_mask_bool, float('-inf'))

        # Combine ALiBi bias with causal mask.
        # For j > i, the -inf from causal_mask dominates.
        # For j <= i, the alibi_bias_values are used.
        combined_bias = alibi_bias_values + additive_causal_mask.unsqueeze(0) # Shape (n_heads, seq_len, seq_len)

        # Add a batch dimension for broadcasting: (1, n_heads, seq_len, seq_len)
        return combined_bias.unsqueeze(0)

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None, # Full attention mask (padding) for total_seq_len_kv
                past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
                use_cache: bool = False
                ) -> Tuple[torch.Tensor, Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]]] | torch.Tensor:

        batch_size, seq_len_q = input_ids.size() # seq_len_q is 1 if use_cache and past_key_values is not None
        device = input_ids.device

        # Determine total sequence length for ALiBi and padding mask based on past_key_values
        seq_len_kv_past = 0
        if past_key_values is not None and past_key_values[0] is not None and len(past_key_values[0]) == 2:
            # past_key_values[layer_idx][0 for key or 1 for value] has shape (B, H, S_past, D_h)
            seq_len_kv_past = past_key_values[0][0].shape[2]

        total_seq_len_kv = seq_len_kv_past + seq_len_q

        # 1. ALiBi Bias: Computed for the total current sequence length (keys and values)
        # This alibi_combined_bias will be (1, n_heads, total_seq_len_kv, total_seq_len_kv)
        alibi_combined_bias = self.get_alibi_attention_bias(total_seq_len_kv, device)

        # 2. Padding Mask: Based on the full attention_mask argument, should also be for total_seq_len_kv
        padding_additive_mask = None
        if attention_mask is not None:
            # Ensure attention_mask has the correct total_seq_len_kv if caching
            if attention_mask.shape[1] != total_seq_len_kv:
                # This can happen if `attention_mask` is not correctly extended in the generate loop
                logger.warning(
                    f"Attention mask length ({attention_mask.shape[1]}) in forward() does not match "
                    f"total_seq_len_kv ({total_seq_len_kv}). This might indicate an issue with "
                    f"how attention_mask is managed during KV caching in the generate loop."
                )
                # Attempt to slice or pad, but this is risky. Best to fix in generate loop.
                # For now, we'll proceed, but SelfAttention will slice the padding_additive_mask
                # based on its k.shape[2] which is total_seq_len_kv.
                # If attention_mask is shorter, this might lead to errors or incorrect masking.

            # Create the additive mask from the boolean/long attention_mask
            # Use a consistent dtype, e.g., from token_embedding or a float type
            padding_additive_mask = torch.zeros_like(attention_mask, dtype=self.token_embedding.weight.dtype, device=device)
            padding_additive_mask.masked_fill_(attention_mask == 0, float('-inf'))
            # Reshape for broadcasting with attention scores: (B, 1, 1, S_total_kv)
            padding_additive_mask = padding_additive_mask.unsqueeze(1).unsqueeze(2)

        # Embedding for the current input_ids (which might be just the new token if caching)
        x = self.token_embedding(input_ids) # Shape: (B, S_q, D)

        present_key_values_all_layers = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            layer_past_key_value = past_key_values[i] if past_key_values is not None else None

            layer_outputs = layer(
                x,
                alibi_combined_bias=alibi_combined_bias, # Pass the full bias for total_seq_len_kv
                padding_additive_mask=padding_additive_mask, # Pass the full padding mask for total_seq_len_kv
                past_key_value=layer_past_key_value,
                use_cache=use_cache
            )
            x = layer_outputs[0] # Output hidden states
            if use_cache:
                if present_key_values_all_layers is not None :
                    present_key_values_all_layers.append(layer_outputs[1]) # Append (K,V) tuple for this layer

        x = self.final_layer_norm(x)
        logits = self.lm_head(x) # Shape: (B, S_q, VocabSize)

        if use_cache:
            # Ensure we return a tuple even if present_key_values_all_layers is empty (e.g. n_layers=0, though not practical)
            return logits, tuple(present_key_values_all_layers) if present_key_values_all_layers is not None else None
        else:
            # For training or when not using cache, return only logits
            return logits

    def _apply_repetition_penalty_optimized(self,
                                            logits: torch.Tensor,
                                            generated_ids_for_penalty: torch.Tensor,
                                            repetition_penalty: float
                                            ) -> torch.Tensor:
        """Applies repetition penalty to the logits."""
        if repetition_penalty == 1.0:
            return logits # No penalty if factor is 1.0

        # logits are (batch_size, vocab_size) for the current token being generated
        # generated_ids_for_penalty is (batch_size, current_total_seq_len)
        batch_size, vocab_size = logits.shape

        for b_idx in range(batch_size):
            # Get tokens from the already generated sequence for this batch item
            sequence_tokens = generated_ids_for_penalty[b_idx]

            # Exclude padding tokens from being considered for penalty
            valid_tokens_for_penalty = sequence_tokens
            if self.config.pad_token_id is not None and self.config.pad_token_id >= 0: # Check for a valid pad_token_id
                is_not_padding = (sequence_tokens != self.config.pad_token_id)
                valid_tokens_for_penalty = sequence_tokens[is_not_padding]

            if len(valid_tokens_for_penalty) > 0:
                # Find unique tokens in the valid part of the sequence that should be penalized
                unique_tokens_in_sequence = torch.unique(valid_tokens_for_penalty)

                for token_id_to_penalize_tensor in unique_tokens_in_sequence:
                    token_id = token_id_to_penalize_tensor.item()
                    if 0 <= token_id < vocab_size: # Ensure token_id is within vocab bounds
                        # Apply linear penalty:
                        # If logit is positive, divide by penalty (making it smaller, less likely)
                        # If logit is negative, multiply by penalty (making it more negative, less likely)
                        if logits[b_idx, token_id] > 0:
                            logits[b_idx, token_id] /= repetition_penalty
                        else:
                            logits[b_idx, token_id] *= repetition_penalty
        return logits

    @torch.no_grad() # Generation should not require gradient calculations
    def generate(self,
                 input_ids: torch.Tensor, # Prompt token IDs
                 max_new_tokens: int = 50,
                 temperature: Optional[float] = None,
                 top_k: Optional[int] = None,
                 top_p: Optional[float] = None,
                 repetition_penalty: Optional[float] = None,
                 eos_token_id: Optional[int] = None,
                 pad_token_id: Optional[int] = None, # pad_token_id from tokenizer for this specific call
                 attention_mask: Optional[torch.Tensor] = None # Initial attention mask for the prompt
                ) -> torch.Tensor:
        """Generates sequences of token ids for models with a language modeling head."""
        self.eval() # Set model to evaluation mode
        device = input_ids.device
        batch_size = input_ids.size(0)

        # Resolve generation parameters, using model config defaults if not provided
        current_pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        temp_to_use = temperature if temperature is not None else self.config.temperature
        top_k_to_use = top_k if top_k is not None else self.config.top_k
        top_p_to_use = top_p if top_p is not None else self.config.top_p
        rep_penalty_to_use = repetition_penalty if repetition_penalty is not None else self.config.repetition_penalty

        # `generated_ids_all` will store the full sequences (prompt + generated tokens)
        generated_ids_all = input_ids
        # `past_key_values` will store the KV cache from previous steps
        past_key_values = None

        # Prepare initial attention_mask for the prompt if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=device, dtype=torch.long)
            if current_pad_token_id is not None and current_pad_token_id >= 0: # Valid pad_token_id
                 attention_mask[input_ids == current_pad_token_id] = 0 # Mask padding in prompt

        # `is_finished` tracks which sequences in the batch have generated EOS
        is_finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for i in range(max_new_tokens):
            # Prepare inputs for the current generation step
            if past_key_values is None: # First step: process the entire prompt
                current_input_ids = generated_ids_all
                current_attention_mask_for_forward = attention_mask # Use the initial prompt attention mask
            else: # Subsequent steps: process only the last generated token
                current_input_ids = generated_ids_all[:, -1:] # Shape: (B, 1)
                # The attention_mask for `model.forward` needs to cover the *entire* sequence so far
                # (prompt + already_generated_tokens + new_token_being_attended_to).
                # The new token is never padding, so append a 1 to the mask.
                new_token_attention_mask_segment = torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=device)
                current_attention_mask_for_forward = torch.cat([attention_mask, new_token_attention_mask_segment], dim=1)
                # Update the main `attention_mask` to be used for the *next* iteration's KV cache context
                attention_mask = current_attention_mask_for_forward


            # Forward pass through the model
            # `current_attention_mask_for_forward` is the full mask for (prompt + generated_so_far)
            model_outputs = self.forward(
                input_ids=current_input_ids,
                attention_mask=current_attention_mask_for_forward, # Full mask for current context
                past_key_values=past_key_values,
                use_cache=True # Enable KV caching
            )

            logits = model_outputs[0] # Logits for the last token(s) in current_input_ids
            past_key_values = model_outputs[1] # Update KV cache for the next step

            # Get logits for the very last token position to be sampled
            # If processing prompt (S_q > 1), logits is (B, S_q, V), take the last one.
            # If generating (S_q = 1), logits is (B, 1, V), so taking [-1] is fine.
            next_token_logits = logits[:, -1, :] # Shape: (B, V)

            # Apply sampling strategies (temperature, repetition penalty, top-k, top-p)
            next_token_logits_processed = next_token_logits.clone()
            next_token_logits_processed = next_token_logits_processed / max(temp_to_use, 1e-5) # Temperature

            if rep_penalty_to_use != 1.0: # Repetition Penalty
                # Pass the full `generated_ids_all` for context of what has been generated so far
                next_token_logits_processed = self._apply_repetition_penalty_optimized(
                    next_token_logits_processed, generated_ids_all, rep_penalty_to_use
                )

            if top_k_to_use > 0: # Top-K
                top_k_values, _ = torch.topk(next_token_logits_processed, top_k_to_use)
                kth_value = top_k_values[:, -1].unsqueeze(-1)
                next_token_logits_processed[next_token_logits_processed < kth_value] = float('-inf')

            if 0.0 < top_p_to_use < 1.0: # Top-P (Nucleus)
                sorted_logits_val, sorted_indices = torch.sort(next_token_logits_processed, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits_val, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p_to_use
                # Keep at least one token, even if its prob is > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = torch.zeros_like(next_token_logits_processed, dtype=torch.bool).scatter_(
                    dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
                next_token_logits_processed.masked_fill_(indices_to_remove, float('-inf'))

            # For sequences that are already finished, force sampling of a specific token (e.g., pad_token)
            # This prevents them from generating further meaningful content.
            if eos_token_id is not None and is_finished.any():
                finished_logits_mask_fill = torch.ones_like(next_token_logits_processed[0], dtype=next_token_logits_processed.dtype) * float('-inf')

                # Determine the token to fill for finished sequences
                fill_token_for_finished = current_pad_token_id
                # If pad_token is invalid or same as EOS, use EOS to avoid issues.
                if fill_token_for_finished is None or fill_token_for_finished < 0 or fill_token_for_finished == eos_token_id:
                    fill_token_for_finished = eos_token_id

                if fill_token_for_finished is not None and 0 <= fill_token_for_finished < next_token_logits_processed.size(-1):
                    finished_logits_mask_fill[fill_token_for_finished] = 0.0 # Allow sampling of this token

                next_token_logits_processed[is_finished] = finished_logits_mask_fill


            # Sample the next token
            probs = F.softmax(next_token_logits_processed, dim=-1)
            next_token_candidates = torch.multinomial(probs, num_samples=1) # Shape: (B, 1)

            # Append the new token to the generated sequences
            generated_ids_all = torch.cat((generated_ids_all, next_token_candidates), dim=-1)

            # Update `is_finished` status for sequences that just generated an EOS token
            # Only update if not already finished to prevent re-finishing
            if eos_token_id is not None:
                # Check if the newly generated token (for non-finished sequences) is EOS
                just_generated_eos = (next_token_candidates.squeeze(-1) == eos_token_id) & (~is_finished)
                is_finished = is_finished | just_generated_eos # Update overall finished status

            # Stop generation if all sequences in the batch are finished
            if is_finished.all():
                break

        self.train() # Set model back to training mode after generation
        return generated_ids_all


def count_parameters(model: nn.Module) -> int:
    """Counts the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger.info("Testing LunarisCodex Model Components (KV Cache & Corrected ALiBi/RepPenalty Version)...")

    test_pad_token_id = 0
    test_config = LunarisCodexConfig(
        vocab_size=100, d_model=64, n_layers=2, n_heads=2,
        max_seq_len=32, lora_rank=4, pad_token_id=test_pad_token_id, activation="swiglu"
    )
    model = LunarisMind(test_config)
    model.eval() # Set to eval for testing generation

    logger.info(f"Total model parameters: {count_parameters(model):,}") # Will count all params as requires_grad=True initially

    batch_size, prompt_len = 2, 5
    # Ensure prompt tokens are valid (e.g., > pad_token_id if pad_token_id is 0)
    dummy_prompt_ids = torch.randint(test_pad_token_id + 1, test_config.vocab_size, (batch_size, prompt_len))

    initial_attention_mask = torch.ones_like(dummy_prompt_ids, dtype=torch.long) # Ensure dtype is long
    if test_pad_token_id >=0: # Ensure pad_token_id is valid before using it for masking
        initial_attention_mask[dummy_prompt_ids == test_pad_token_id] = 0


    logger.info(f"\nPrompt IDs shape: {dummy_prompt_ids.shape}")

    # Test model.forward with use_cache=False (standard forward pass)
    try:
        logger.info("\nTesting forward pass (use_cache=False)...")
        # When use_cache=False, model.forward returns only logits
        logits_no_cache = model(dummy_prompt_ids, attention_mask=initial_attention_mask, use_cache=False)
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
            assert past_kv_init[0][0].shape == (batch_size, test_config.n_heads, prompt_len, test_config.d_model // test_config.n_heads)
    except Exception as e:
        logger.error(f"❌ ERROR during forward pass (use_cache=True, initial) test: {e}", exc_info=True)

    # Test model.forward with use_cache=True (next token step)
    if 'past_kv_init' in locals() and past_kv_init is not None:
        try:
            logger.info("\nTesting forward pass (use_cache=True, next token step)...")
            next_token_input = torch.randint(test_pad_token_id + 1, test_config.vocab_size, (batch_size, 1))

            # Attention mask for the next step needs to include the prompt and the new token
            next_step_attention_mask = torch.cat([initial_attention_mask, torch.ones_like(next_token_input, dtype=torch.long)], dim=1)

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
                assert past_kv_next[0][0].shape == (batch_size, test_config.n_heads, prompt_len + 1, test_config.d_model // test_config.n_heads)

        except Exception as e:
            logger.error(f"❌ ERROR during forward pass (use_cache=True, next token) test: {e}", exc_info=True)


    # Test generation with KV Caching and corrected ALiBi/RepPenalty
    try:
        logger.info("\nTesting generation with KV Caching (and fixes)...")
        gen_prompt_ids = torch.randint(test_pad_token_id + 1, test_config.vocab_size, (batch_size, prompt_len))
        gen_initial_attention_mask = torch.ones_like(gen_prompt_ids, dtype=torch.long) # Ensure dtype is long
        if test_pad_token_id >=0:
             gen_initial_attention_mask[gen_prompt_ids == test_pad_token_id] = 0


        generated_kv = model.generate(
            gen_prompt_ids,
            max_new_tokens=5,
            eos_token_id=test_pad_token_id + 10, # Ensure this is a valid token_id
            pad_token_id=test_config.pad_token_id, # Use pad_token_id from config for consistency
            attention_mask=gen_initial_attention_mask # Pass initial mask
        )
        logger.info(f"Generated IDs (KV cache) shape: {generated_kv.shape}")
        logger.info(f"Generated IDs (KV cache, first example): {generated_kv[0]}")
        assert generated_kv.shape == (batch_size, prompt_len + 5)
        logger.info("✅ Model KV Caching, ALiBi, and Repetition Penalty component tests completed successfully (inspect output for correctness)!")
    except Exception as e:
        logger.error(f"❌ ERROR during KV Caching generation test (with fixes): {e}", exc_info=True)
