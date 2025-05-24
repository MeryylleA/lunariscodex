import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Attempt to import flash_attn, but allow fallback test
try:
    from flash_attn import flash_attn_func
    FLASH_ATTENTION_AVAILABLE = True
    print("INFO: flash_attn library found, but will be disabled due to ALiBi incompatibility.")
except ImportError:
    print("WARNING: flash_attn library not found or could not be imported. Using PyTorch standard attention implementation.")
    FLASH_ATTENTION_AVAILABLE = False

class LunarisCodexConfig:
    """
    Configuration class for the LunarisCodex transformer model.
    Contains all hyperparameters and architectural choices.
    """
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
        use_flash_attention_if_available=False,  # Disabled by default due to ALiBi incompatibility
        layer_norm_epsilon=1e-6,  # More stable than 1e-5
        ff_multiplier=4,  # Increased from 3 to 4 (modern standard)
        alibi_implementation="optimized"
    ):
        # Automatic model size estimation for adjustments
        estimated_params = d_model * n_layers * (d_model * 4 + d_model * ff_multiplier)

        # For small models (<50M parameters), automatically adjust dropout
        if estimated_params < 50_000_000:
            original_dropout = dropout
            dropout = min(dropout, 0.05)  # Reduce dropout for small models
            if dropout != original_dropout:
                print(f"Small model detected (~{estimated_params/1_000_000:.1f}M params), dropout adjusted from {original_dropout} to {dropout}")

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
        self.alibi_implementation = alibi_implementation


class LoRALinear(nn.Linear):
    """
    LoRA (Low-Rank Adaptation) applied to a linear layer.
    Enables efficient fine-tuning by adding low-rank decomposition matrices.
    """
    def __init__(self, in_features, out_features, rank=32, alpha=1.0, bias=False):
        super().__init__(in_features, out_features, bias=bias)
        self.has_lora = rank is not None and rank > 0
        if self.has_lora:
            self.rank = rank
            self.lora_alpha = alpha

            # LoRA matrices A and B for low-rank decomposition
            self.lora_A = nn.Parameter(torch.Tensor(in_features, rank))
            self.lora_B = nn.Parameter(torch.Tensor(rank, out_features))

            # Initialize LoRA weights following best practices
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)  # B starts at zero for stability

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_output = super().forward(x)
        if self.has_lora:
            # Apply LoRA adaptation with proper scaling
            lora_adaptation = (x @ self.lora_A @ self.lora_B) * (self.lora_alpha / self.rank)
            return original_output + lora_adaptation
        return original_output

class LunarisMind(nn.Module):
    """
    Main Lunaris Codex Transformer Decoder model.
    Features ALiBi positional encoding, LoRA adaptation, and optimized attention.
    """
    def __init__(self, config: LunarisCodexConfig):
        super().__init__()
        self.config = config
        if self.config.vocab_size is None:
            raise ValueError("config.vocab_size must be provided!")

        # Core model components
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([
            TransformerDecoderBlock(config) for _ in range(config.n_layers)
        ])
        self.final_layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

        # Tied language modeling head (weight sharing with embeddings)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        # Initialize ALiBi slopes and model weights
        self._init_alibi_slopes()
        self.apply(self._init_weights)

    def _init_alibi_slopes(self):
        """Initialize corrected ALiBi slopes for any number of attention heads."""
        def get_slopes(n_heads):
            def get_slopes_power_of_2(n):
                start = 2**(-(2**-(math.log2(n)-3)))
                ratio = start
                return [start*ratio**i for i in range(n)]

            if n_heads <= 8:
                # For small n_heads, use simplified more stable formula
                return [2 ** (-8.0 / n_heads * (i + 1)) for i in range(n_heads)]
            else:
                # For n_heads > 8, use power-of-2 based approximation
                if math.log2(n_heads).is_integer():
                    return get_slopes_power_of_2(n_heads)
                else:
                    # Approximation for non-power-of-2 numbers
                    closest_power = 2 ** round(math.log2(n_heads))
                    base_slopes = get_slopes_power_of_2(closest_power)

                    if n_heads > closest_power:
                        # Interpolate additional slopes
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
        print(f"ALiBi slopes initialized for {self.config.n_heads} heads: {[f'{s:.6f}' for s in slopes_list]}")

    def _init_weights(self, module):
        """Corrected weight initialization to avoid vanishing gradients."""
        if isinstance(module, nn.Linear) and not isinstance(module, LoRALinear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, LoRALinear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

        # Special less aggressive initialization for output projections
        if isinstance(module, SelfAttention) or isinstance(module, FeedForward):
            # Less aggressive formula - removed factor of 2 that was causing vanishing gradients
            output_proj_std = 0.02 / math.sqrt(self.config.n_layers)
            if isinstance(module, SelfAttention):
                nn.init.normal_(module.output_proj.weight, mean=0.0, std=output_proj_std)
                if module.output_proj.bias is not None:
                    nn.init.zeros_(module.output_proj.bias)
            if isinstance(module, FeedForward):
                nn.init.normal_(module.fc2.weight, mean=0.0, std=output_proj_std)
                if module.fc2.bias is not None:
                    nn.init.zeros_(module.fc2.bias)

    def get_alibi_attention_bias(self, seq_len, device):
        """
        Generates the ALiBi attention bias tensor combined with causal masking.
        ALiBi provides position-dependent attention biases without learned positional embeddings.
        """
        # Causal mask: True where attention should be masked (upper triangle)
        causal_mask_bool = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device), diagonal=1)

        # ALiBi positional bias calculation
        # pos_indices represents (j - i) for query position i and key position j
        pos_indices = torch.arange(seq_len, device=device, dtype=self.alibi_slopes.dtype).unsqueeze(0) - \
                      torch.arange(seq_len, device=device, dtype=self.alibi_slopes.dtype).unsqueeze(1)

        # Apply head-specific slopes to position differences
        # alibi_slopes: (num_heads), alibi_bias_values: (num_heads, seq_len, seq_len)
        alibi_bias_values = self.alibi_slopes.view(-1, 1, 1) * pos_indices.unsqueeze(0)

        # Create additive causal mask (-infinity for masked positions)
        additive_causal_mask = torch.zeros((seq_len, seq_len), dtype=alibi_bias_values.dtype, device=device)
        additive_causal_mask.masked_fill_(causal_mask_bool, float('-inf'))

        # Combine ALiBi bias with causal mask
        final_bias = alibi_bias_values + additive_causal_mask.unsqueeze(0)
        return final_bias

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass through the transformer model.
        
        Args:
            input_ids: Token indices of shape (batch_size, seq_len)
            attention_mask: Padding mask of shape (batch_size, seq_len)
        
        Returns:
            Logits of shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.size()
        device = input_ids.device

        # Token embeddings
        x = self.token_embedding(input_ids)

        # ALiBi bias is batch-independent, depends on seq_len and n_heads
        alibi_combined_bias = self.get_alibi_attention_bias(seq_len, device)

        # Prepare padding mask if provided (additive, -inf for padded tokens)
        padding_additive_mask = None
        if attention_mask is not None:
            padding_additive_mask = torch.zeros_like(attention_mask, dtype=x.dtype, device=device)
            padding_additive_mask.masked_fill_(attention_mask == 0, float('-inf'))
            padding_additive_mask = padding_additive_mask.unsqueeze(1).unsqueeze(2)

        # Pass through transformer blocks
        for layer in self.layers:
            x = layer(x, alibi_combined_bias, padding_additive_mask)

        # Final layer norm and language modeling head
        x = self.final_layer_norm(x)
        logits = self.lm_head(x)
        return logits

    def _apply_repetition_penalty_optimized(self, logits, generated_ids, repetition_penalty):
        """Optimized version of repetition penalty using tensor operations."""
        if repetition_penalty == 1.0:
            return logits

        batch_size, vocab_size = logits.shape

        # Apply penalty more efficiently
        for b_idx in range(batch_size):
            unique_tokens = torch.unique(generated_ids[b_idx])
            if len(unique_tokens) > 0:
                token_logits = logits[b_idx, unique_tokens]
                penalty_mask = token_logits > 0

                # Apply penalty in vectorized manner
                token_logits[penalty_mask] /= repetition_penalty
                token_logits[~penalty_mask] *= repetition_penalty

                logits[b_idx, unique_tokens] = token_logits

        return logits

    def generate(self, input_ids: torch.Tensor, max_new_tokens=50, temperature=0.8, top_k=50, top_p=0.95, repetition_penalty=1.1, eos_token_id=None):
        """
        Generates token sequences autoregressively with various sampling strategies.
        
        Args:
            input_ids: Starting token sequence
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature for randomness control
            top_k: Top-k filtering threshold
            top_p: Nucleus sampling threshold
            repetition_penalty: Penalty factor for repeated tokens
            eos_token_id: End-of-sequence token ID for early stopping
        
        Returns:
            Generated token sequences
        """
        self.eval()
        device = input_ids.device
        batch_size = input_ids.size(0)
        generated_ids = input_ids

        for _ in range(max_new_tokens):
            current_seq_len = generated_ids.size(1)
            current_attention_mask = torch.ones((batch_size, current_seq_len), dtype=torch.long, device=device)

            with torch.no_grad():
                logits = self.forward(generated_ids, attention_mask=current_attention_mask)[:, -1, :]

            # Apply temperature scaling
            logits = logits / temperature

            # Apply repetition penalty (optimized version)
            logits = self._apply_repetition_penalty_optimized(logits, generated_ids, repetition_penalty)

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
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False

                indices_to_remove = torch.zeros_like(logits, dtype=torch.bool).scatter_(
                    dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
                logits.masked_fill_(indices_to_remove, float('-inf'))

            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated_ids = torch.cat((generated_ids, next_token), dim=-1)

            # Check for early stopping
            if eos_token_id is not None and (next_token == eos_token_id).any():
                if (next_token.squeeze(-1) == eos_token_id).any():
                    break

        self.train()
        return generated_ids

class TransformerDecoderBlock(nn.Module):
    """
    Single transformer decoder block with self-attention and feed-forward layers.
    Includes LayerScale for training stability in large models.
    """
    def __init__(self, config: LunarisCodexConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.attn = SelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

        ffn_intermediate_dim = config.d_model * config.ff_multiplier
        self.ff = FeedForward(config.d_model, ffn_intermediate_dim, config.dropout, config.activation, config.lora_rank)
        self.dropout = nn.Dropout(config.dropout)

        # LayerScale only for large models (>50M parameters)
        # Quick parameter count estimation
        estimated_params = config.d_model * config.n_layers * (config.d_model * 4 + ffn_intermediate_dim)
        self.use_layerscale = estimated_params > 50_000_000

        if self.use_layerscale:
            # More conservative initialization based on model size
            init_val = 1e-4 if estimated_params > 100_000_000 else 0.1
            self.ls_gamma_1 = nn.Parameter(torch.ones(config.d_model) * init_val)
            self.ls_gamma_2 = nn.Parameter(torch.ones(config.d_model) * init_val)
            print(f"LayerScale enabled with initial value: {init_val} (model ~{estimated_params/1_000_000:.1f}M params)")
        else:
            print(f"LayerScale disabled for small model (~{estimated_params/1_000_000:.1f}M params)")

    def forward(self, x: torch.Tensor, alibi_combined_bias: torch.Tensor, padding_additive_mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass with pre-norm architecture and optional LayerScale."""
        # Self-Attention block with pre-normalization
        residual = x
        x_norm = self.ln_1(x)
        attn_output = self.attn(x_norm, alibi_combined_bias, padding_additive_mask)

        if self.use_layerscale:
            x = residual + self.dropout(self.ls_gamma_1 * attn_output)
        else:
            x = residual + self.dropout(attn_output)

        # Feed-Forward block with pre-normalization
        residual = x
        x_norm = self.ln_2(x)
        ff_output = self.ff(x_norm)

        if self.use_layerscale:
            x = residual + self.dropout(self.ls_gamma_2 * ff_output)
        else:
            x = residual + self.dropout(ff_output)

        return x

class SelfAttention(nn.Module):
    """
    Multi-head self-attention with ALiBi positional encoding.
    Flash Attention is disabled to ensure ALiBi compatibility.
    """
    def __init__(self, config: LunarisCodexConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        if self.head_dim * self.n_heads != self.d_model:
            raise ValueError("d_model must be divisible by n_heads")

        # Combined QKV projection for efficiency
        self.qkv_proj = LoRALinear(config.d_model, 3 * config.d_model, rank=config.lora_rank, bias=False)
        self.output_proj = LoRALinear(config.d_model, config.d_model, rank=config.lora_rank, bias=False)

        self.attn_dropout_p = config.dropout

        # CRITICAL FIX: Disable Flash Attention to ensure consistent ALiBi behavior
        # Standard Flash Attention doesn't support custom ALiBi bias
        self.use_flash_attention = False
        if FLASH_ATTENTION_AVAILABLE and config.use_flash_attention_if_available:
            print("WARNING: Flash Attention available but disabled due to ALiBi custom bias incompatibility")

    def forward(self, x: torch.Tensor, alibi_combined_bias: torch.Tensor, padding_additive_mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass with manual attention computation to support ALiBi."""
        batch_size, seq_len, _ = x.shape
        device = x.device

        # Compute Q, K, V projections
        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # ALWAYS use manual implementation to ensure correct ALiBi behavior
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply ALiBi, causal, and padding biases
        current_bias = alibi_combined_bias.unsqueeze(0)  # (1, H, L, L)
        if padding_additive_mask is not None:
            current_bias = current_bias + padding_additive_mask

        attn_scores = attn_scores + current_bias
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.attn_dropout_p, training=self.training)
        attn_output = torch.matmul(attn_weights, v)

        # Recombine attention heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.output_proj(attn_output)
        return output

class FeedForward(nn.Module):
    """
    Feed-forward network with configurable activation functions.
    Supports SwiGLU and GELU activations with LoRA adaptation.
    """
    def __init__(self, d_model: int, d_ff: int, dropout=0.1, activation="swiglu", lora_rank=32):
        super().__init__()
        self.activation = activation

        # SwiGLU requires double the intermediate dimension for gating
        fc1_out_dim = d_ff * 2 if activation == "swiglu" else d_ff

        self.fc1 = LoRALinear(d_model, fc1_out_dim, rank=lora_rank, bias=False)
        self.fc2 = LoRALinear(d_ff, d_model, rank=lora_rank, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with activation-specific processing."""
        if self.activation == "swiglu":
            # SwiGLU: Split into gate and activation components
            hidden = self.fc1(x)
            gated, activated = hidden.chunk(2, dim=-1)
            x = F.silu(gated) * activated  # SiLU activation with gating
        elif self.activation == "gelu":
            x = F.gelu(self.fc1(x))
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")

        x = self.dropout(x)
        x = self.fc2(x)
        return x

def count_parameters(model: nn.Module) -> int:
    """Counts the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    print("Testing LunarisCodex Model Components (Corrected Version)...")

    # Example configuration for a small test model
    test_config = LunarisCodexConfig(
        vocab_size=1000,
        d_model=256,   # Size similar to your experiment
        n_layers=6,    # Number of layers from your experiment
        n_heads=4,     # Number of heads from your experiment
        max_seq_len=512, # Context from your experiment
        lora_rank=8,
        use_flash_attention_if_available=False
    )

    model = LunarisMind(test_config)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params:,}")

    # LoRA parameter counting
    trainable_params_val = 0
    if test_config.lora_rank > 0:
        for name, param in model.named_parameters():
            if 'lora_' in name:
                param.requires_grad = True
                trainable_params_val += param.numel()
            else:
                param.requires_grad = False
    else:
        trainable_params_val = total_params

    print(f"Trainable parameters: {trainable_params_val:,} ({trainable_params_val/total_params*100:.2f}%)")

    # Test forward pass
    batch_size = 2
    test_seq_len = 32
    dummy_input_ids = torch.randint(0, test_config.vocab_size, (batch_size, test_seq_len))
    dummy_attention_mask = torch.ones_like(dummy_input_ids)

    print(f"\nInput IDs shape: {dummy_input_ids.shape}")
    print(f"Attention Mask shape: {dummy_attention_mask.shape}")

    try:
        print("\nTesting forward pass...")
        logits = model(dummy_input_ids, attention_mask=dummy_attention_mask)
        print(f"Logits output shape: {logits.shape}")

        print("\nTesting generation...")
        generated = model.generate(dummy_input_ids[:, :5], max_new_tokens=5, eos_token_id=None)
        print(f"Generated IDs shape: {generated.shape}")
        print(f"Generated IDs (first example): {generated[0]}")
        print("✅ Model component tests completed successfully!")

    except Exception as e:
        print(f"❌ ERROR during model component test: {e}")
        import traceback
        traceback.print_exc()
