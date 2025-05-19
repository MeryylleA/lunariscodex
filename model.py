# model.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Attempt to import flash_attn, but allow fallback
try:
    from flash_attn import flash_attn_func
    FLASH_ATTENTION_AVAILABLE = True
    # Using logger after it's configured in train.py or inference.py
    # For standalone model testing, a simple print is fine.
    print("INFO: flash_attn library found, will be used for attention on CUDA GPU if enabled.")
except ImportError:
    print("WARNING: flash_attn library not found or could not be imported. Using PyTorch standard attention implementation.")
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
        use_flash_attention_if_available=True,
        layer_norm_epsilon=1e-5, # Added for consistency
        ff_multiplier=3,         # Standard multiplier for FFN intermediate dim
        alibi_implementation="original" # or "simplified_slopes"
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len # Max sequence length supported by ALiBi context
        self.dropout = dropout
        self.activation = activation
        self.temperature = temperature # For generation
        self.top_k = top_k # For generation
        self.top_p = top_p # For generation
        self.repetition_penalty = repetition_penalty # For generation
        self.lora_rank = lora_rank # Set to 0 or negative to disable LoRA
        self.use_flash_attention_if_available = use_flash_attention_if_available
        self.layer_norm_epsilon = layer_norm_epsilon
        self.ff_multiplier = ff_multiplier # FFN hidden_dim = d_model * ff_multiplier
        self.alibi_implementation = alibi_implementation


class LoRALinear(nn.Linear):
    """ LoRA (Low-Rank Adaptation) applied to a linear layer. """
    def __init__(self, in_features, out_features, rank=32, alpha=1.0, bias=False):
        super().__init__(in_features, out_features, bias=bias)
        self.has_lora = rank is not None and rank > 0
        if self.has_lora:
            self.rank = rank
            self.lora_alpha = alpha # Typically, alpha is set to rank or 1.

            # LoRA matrices A and B
            self.lora_A = nn.Parameter(torch.Tensor(in_features, rank))
            self.lora_B = nn.Parameter(torch.Tensor(rank, out_features))

            # Initialize LoRA weights
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5)) # Kaiming for A
            nn.init.zeros_(self.lora_B) # Zero for B, so initial adaptation is zero

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_output = super().forward(x)
        if self.has_lora: # LoRA path is active only if lora_rank > 0
            # During training, apply LoRA. For inference, weights can be merged.
            # Here, we always apply if has_lora, assuming inference might also use this dynamic path
            # or a separate merge_lora_weights() method would be called.
            lora_adaptation = (x @ self.lora_A @ self.lora_B) * (self.lora_alpha / self.rank)
            return original_output + lora_adaptation
        return original_output

class LunarisMind(nn.Module):
    """ Main Lunaris Codex Transformer Decoder model. """
    def __init__(self, config: LunarisCodexConfig):
        super().__init__()
        self.config = config
        if self.config.vocab_size is None:
            raise ValueError("config.vocab_size must be provided!")

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([
            TransformerDecoderBlock(config) for _ in range(config.n_layers)
        ])
        self.final_layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

        # Tied language modeling head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight # Tie weights

        self._init_alibi_slopes()
        self.apply(self._init_weights) # Apply custom weight initialization

    def _init_alibi_slopes(self):
        """ Initializes ALiBi slopes based on the number of attention heads. """
        # Calculates slopes m for ALiBi. Each head h gets a slope m_h.
        # The original ALiBi paper uses powers of 2.
        # For n_heads that are powers of 2, it's 2^(-8/n_heads * i)
        # For others, it's an adaptation or using the closest power of 2.
        if self.config.alibi_implementation == "original_paper_style": # More complex, needs careful power-of-2 handling
            def get_slopes_power_of_2(n):
                start = (2**(-2**-(math.log2(n)-3)))
                ratio = start
                return [start*ratio**i for i in range(n)]
            if math.log2(self.config.n_heads).is_integer():
                 slopes_list = get_slopes_power_of_2(self.config.n_heads)
            else: # Fallback for non-power-of-2 n_heads (e.g., geometric progression)
                 closest_power_of_2 = 2**math.floor(math.log2(self.config.n_heads))
                 slopes_list_base = get_slopes_power_of_2(closest_power_of_2)
                 # Simple interpolation or repetition strategy might be needed
                 slopes_list = slopes_list_base * (self.config.n_heads // closest_power_of_2) + \
                               slopes_list_base[:self.config.n_heads % closest_power_of_2]
                 print(f"WARNING: ALiBi slopes adapted for n_heads={self.config.n_heads} (not power of 2).")

        else: # Simplified geometric progression, as in your original code
            slopes_list = [2 ** (-(8.0 / self.config.n_heads) * (i + 1)) for i in range(self.config.n_heads)]

        slopes = torch.tensor(slopes_list, dtype=torch.float)
        self.register_buffer("alibi_slopes", slopes)


    def _init_weights(self, module):
        """ Initializes weights for different module types. """
        if isinstance(module, nn.Linear) and not isinstance(module, LoRALinear): # Standard Linear layers
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, LoRALinear): # Base weights of LoRALinear
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            # LoRA A and B are initialized in LoRALinear.__init__
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

        # Special initialization for output projections in attention and FFN (GPT-2 style)
        if isinstance(module, SelfAttention) or isinstance(module, FeedForward):
            output_proj_std = 0.02 / math.sqrt(2 * self.config.n_layers)
            if isinstance(module, SelfAttention):
                nn.init.normal_(module.output_proj.weight, mean=0.0, std=output_proj_std)
                if module.output_proj.bias is not None: nn.init.zeros_(module.output_proj.bias)
            if isinstance(module, FeedForward):
                nn.init.normal_(module.fc2.weight, mean=0.0, std=output_proj_std)
                if module.fc2.bias is not None: nn.init.zeros_(module.fc2.bias)


    def get_alibi_attention_bias(self, seq_len, device):
        """ Generates the ALiBi attention bias tensor combined with a causal mask. """
        # Causal mask: True where attention should be masked (upper triangle)
        causal_mask_bool = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device), diagonal=1)

        # ALiBi positional bias
        # pos_indices represents (j - i) for query position i and key position j
        pos_indices = torch.arange(seq_len, device=device, dtype=self.alibi_slopes.dtype).unsqueeze(0) - \
                      torch.arange(seq_len, device=device, dtype=self.alibi_slopes.dtype).unsqueeze(1)
        # Shape of pos_indices: (seq_len, seq_len)

        # alibi_slopes: (num_heads)
        # alibi_bias_values: (num_heads, seq_len, seq_len)
        # Each head 'h' has its ALiBi bias: slopes[h] * (j - i)
        alibi_bias_values = self.alibi_slopes.view(-1, 1, 1) * pos_indices.unsqueeze(0)

        # Create an additive causal mask (-infinity for masked positions)
        additive_causal_mask = torch.zeros((seq_len, seq_len), dtype=alibi_bias_values.dtype, device=device)
        additive_causal_mask.masked_fill_(causal_mask_bool, float('-inf'))

        # Combine ALiBi bias with the additive causal mask
        # Broadcasting: (num_heads, seq_len, seq_len) + (1, seq_len, seq_len)
        final_bias = alibi_bias_values + additive_causal_mask.unsqueeze(0)
        return final_bias # Shape: (num_heads, seq_len, seq_len)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len = input_ids.size()
        device = input_ids.device

        x = self.token_embedding(input_ids) # (B, L, D_model)

        # ALiBi bias is independent of batch, depends on seq_len and n_heads
        alibi_combined_bias = self.get_alibi_attention_bias(seq_len, device) # (H, L, L)

        # Prepare padding mask if provided (additive, -inf for padded tokens)
        padding_additive_mask = None
        if attention_mask is not None: # attention_mask: (B, L), 1 for valid, 0 for pad
            padding_additive_mask = torch.zeros_like(attention_mask, dtype=x.dtype, device=device) # (B, L)
            padding_additive_mask.masked_fill_(attention_mask == 0, float('-inf'))
            # Reshape for broadcasting with attention scores (B, H, L, L) or (B, H, L_q, L_k)
            # Mask needs to affect key positions, so expand for key sequence length dimension
            padding_additive_mask = padding_additive_mask.unsqueeze(1).unsqueeze(2) # (B, 1, 1, L)

        for layer in self.layers:
            x = layer(x, alibi_combined_bias, padding_additive_mask)

        x = self.final_layer_norm(x)
        logits = self.lm_head(x)
        return logits

    def generate(self, input_ids: torch.Tensor, max_new_tokens=50, temperature=0.8, top_k=50, top_p=0.95, repetition_penalty=1.1, eos_token_id=None):
        """Generates token sequences autoregressively."""
        self.eval() # Set to evaluation mode
        device = input_ids.device
        batch_size = input_ids.size(0)
        generated_ids = input_ids # (B, current_L)

        for _ in range(max_new_tokens):
            current_seq_len = generated_ids.size(1)
            # Create attention mask for current generated sequence (assumes no internal padding)
            current_attention_mask = torch.ones((batch_size, current_seq_len), dtype=torch.long, device=device)

            # Get logits for the last token only
            with torch.no_grad(): # No need to track gradients during generation
                logits = self.forward(generated_ids, attention_mask=current_attention_mask)[:, -1, :] # (B, VocabSize)

            logits = logits / temperature # Apply temperature scaling

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for b_idx in range(batch_size):
                    seen_tokens = set(generated_ids[b_idx].tolist())
                    for token_id in seen_tokens:
                        if logits[b_idx, token_id] > 0:
                            logits[b_idx, token_id] /= repetition_penalty
                        else:
                            logits[b_idx, token_id] *= repetition_penalty # Makes negative logits more negative

            # Top-k filtering
            if top_k > 0:
                top_k_values, _ = torch.topk(logits, top_k)
                kth_value = top_k_values[:, -1].unsqueeze(-1)
                logits[logits < kth_value] = float('-inf')

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift to keep at least one token if all have cumulative prob > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False # Always keep the most probable token

                indices_to_remove = torch.zeros_like(logits, dtype=torch.bool).scatter_(
                    dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
                logits.masked_fill_(indices_to_remove, float('-inf'))

            probs = F.softmax(logits, dim=-1) # (B, VocabSize)
            next_token = torch.multinomial(probs, num_samples=1) # (B, 1)

            generated_ids = torch.cat((generated_ids, next_token), dim=-1)

            if eos_token_id is not None and (next_token == eos_token_id).any():
                # Check if any sequence in the batch has generated EOS
                # More robustly, stop generation per sequence if EOS is found.
                # For simplicity now, stop if any generates EOS.
                if (next_token.squeeze(-1) == eos_token_id).any(): break

        self.train() # Set back to training mode if it was changed
        return generated_ids

class TransformerDecoderBlock(nn.Module):
    def __init__(self, config: LunarisCodexConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.attn = SelfAttention(config) # Pass the whole config
        self.ln_2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

        # Calculate FFN intermediate dimension
        ffn_intermediate_dim = config.d_model * config.ff_multiplier
        self.ff = FeedForward(config.d_model, ffn_intermediate_dim, config.dropout, config.activation, config.lora_rank)

        self.dropout = nn.Dropout(config.dropout)

        # LayerScale parameters (gamma)
        self.ls_gamma_1 = nn.Parameter(torch.ones(config.d_model) * 0.1)
        self.ls_gamma_2 = nn.Parameter(torch.ones(config.d_model) * 0.1)

    def forward(self, x: torch.Tensor, alibi_combined_bias: torch.Tensor, padding_additive_mask: torch.Tensor = None) -> torch.Tensor:
        # Self-Attention part with pre-LayerNorm and LayerScale
        residual = x
        x_norm = self.ln_1(x)
        attn_output = self.attn(x_norm, alibi_combined_bias, padding_additive_mask)
        x = residual + self.dropout(self.ls_gamma_1 * attn_output) # Apply LayerScale before dropout

        # FeedForward part with pre-LayerNorm and LayerScale
        residual = x
        x_norm = self.ln_2(x)
        ff_output = self.ff(x_norm)
        x = residual + self.dropout(self.ls_gamma_2 * ff_output) # Apply LayerScale before dropout
        return x

class SelfAttention(nn.Module):
    def __init__(self, config: LunarisCodexConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        if self.head_dim * self.n_heads != self.d_model:
            raise ValueError("d_model must be divisible by n_heads")

        self.qkv_proj = LoRALinear(config.d_model, 3 * config.d_model, rank=config.lora_rank, bias=False)
        self.output_proj = LoRALinear(config.d_model, config.d_model, rank=config.lora_rank, bias=False)

        self.attn_dropout_p = config.dropout
        # self.resid_dropout = nn.Dropout(config.dropout) # Residual dropout is applied in TransformerDecoderBlock

        # Determine if FlashAttention can and should be used
        self.use_flash_attention = FLASH_ATTENTION_AVAILABLE and config.use_flash_attention_if_available

    def forward(self, x: torch.Tensor, alibi_combined_bias: torch.Tensor, padding_additive_mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        device = x.device

        qkv = self.qkv_proj(x) # (B, L, 3*D)
        # Reshape and permute for multi-head attention: (3, B, H, L, D_head)
        qkv = qkv.view(batch_size, seq_len, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0) # Each is (B, H, L, D_head)

        if self.use_flash_attention and device.type == 'cuda':
            # NOTE: flash_attn_func with causal=True handles causal masking.
            # The passed alibi_combined_bias and padding_additive_mask are NOT directly used by it.
            # For ALiBi + FlashAttention, a custom kernel or different integration is typically needed.
            # This path currently provides causal attention via FlashAttention without explicit ALiBi/padding mask.
            amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            with torch.cuda.amp.autocast(enabled=True, dtype=amp_dtype):
                attn_output = flash_attn_func(
                    q, k, v,
                    dropout_p=self.attn_dropout_p if self.training else 0.0,
                    causal=True
                ) # (B, H, L, D_head)
        else:
            # Manual attention implementation with ALiBi and padding support
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim) # (B, H, L, L)

            # Apply ALiBi and padding biases
            # alibi_combined_bias is (H, L, L) -> needs to be broadcastable to (B, H, L, L)
            current_bias = alibi_combined_bias.unsqueeze(0) # (1, H, L, L)
            if padding_additive_mask is not None:
                # padding_additive_mask is (B, 1, 1, L)
                current_bias = current_bias + padding_additive_mask # Broadcasts to (B, H, L, L)

            attn_scores = attn_scores + current_bias # Add combined biases

            attn_weights = F.softmax(attn_scores, dim=-1) # (B, H, L, L)
            attn_weights = F.dropout(attn_weights, p=self.attn_dropout_p, training=self.training)
            attn_output = torch.matmul(attn_weights, v) # (B, H, L, D_head)

        # Recombine heads: (B, H, L, D_head) -> (B, L, H, D_head) -> (B, L, D_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        output = self.output_proj(attn_output) # (B, L, D_model)
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout=0.1, activation="swiglu", lora_rank=32):
        super().__init__()
        self.activation = activation

        # For SwiGLU, the first linear layer projects to 2 * d_ff'
        # where d_ff' is the effective intermediate dimension after activation.
        # Here, d_ff is treated as that effective intermediate dimension.
        fc1_out_dim = d_ff * 2 if activation == "swiglu" else d_ff

        self.fc1 = LoRALinear(d_model, fc1_out_dim, rank=lora_rank, bias=False)
        self.fc2 = LoRALinear(d_ff, d_model, rank=lora_rank, bias=False) # Projects back from d_ff
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "swiglu":
            hidden = self.fc1(x)
            gated, activated = hidden.chunk(2, dim=-1)
            x = F.silu(gated) * activated # SwiGLU
        elif self.activation == "gelu":
            x = F.gelu(self.fc1(x))
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")

        x = self.dropout(x) # Dropout is often applied before the second linear layer
        x = self.fc2(x)
        return x

def count_parameters(model: nn.Module) -> int:
    """Counts the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    print("Testing LunarisCodex Model Components...")

    # Example config for a small test model
    test_config = LunarisCodexConfig(
        vocab_size=1000,
        d_model=128,
        n_layers=2,
        n_heads=4,
        max_seq_len=64, # Model's max context for ALiBi
        lora_rank=8,   # Enable LoRA
        use_flash_attention_if_available=False # Force fallback for CPU testing
    )
    print(f"Using Flash Attention if available and enabled: {test_config.use_flash_attention_if_available and FLASH_ATTENTION_AVAILABLE and torch.cuda.is_available()}")

    model = LunarisMind(test_config)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params:,}")

    # Simulate LoRA setup for parameter counting
    trainable_params_val = 0
    if test_config.lora_rank > 0:
        for name, param in model.named_parameters():
            if 'lora_' in name: param.requires_grad = True; trainable_params_val += param.numel()
            else: param.requires_grad = False
    else:
        for param in model.parameters(): param.requires_grad = True
        trainable_params_val = total_params
    print(f"Trainable parameters: {trainable_params_val:,} ({trainable_params_val/total_params*100:.2f}%)")

    # Dummy input for testing forward pass
    batch_size = 2
    test_seq_len = 32 # Must be <= config.max_seq_len
    dummy_input_ids = torch.randint(0, test_config.vocab_size, (batch_size, test_seq_len))
    dummy_attention_mask = torch.ones_like(dummy_input_ids) # No padding
    # Example with padding:
    # dummy_attention_mask[:, test_seq_len//2:] = 0

    print(f"\nInput IDs shape: {dummy_input_ids.shape}")
    print(f"Attention Mask shape: {dummy_attention_mask.shape}")

    try:
        print("\nTesting forward pass...")
        logits = model(dummy_input_ids, attention_mask=dummy_attention_mask)
        print(f"Logits output shape: {logits.shape}") # Expected: (B, L, VocabSize)

        print("\nTesting generation (very basic)...")
        # For a real test, a tokenizer and a proper eos_token_id would be needed
        generated = model.generate(dummy_input_ids[:, :5], max_new_tokens=5, eos_token_id=None)
        print(f"Generated IDs shape: {generated.shape}") # Expected: (B, 5 + 5)
        print(f"Generated IDs (first example): {generated[0]}")
        print("Model component tests completed.")

    except Exception as e:
        print(f"ERROR during model component test: {e}")
        import traceback
        traceback.print_exc()
