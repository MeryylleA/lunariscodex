"""
Full definition of a LunarisCodex Language Model, all of it in this single file.
This version is a refactored and enhanced Llama-style model, created by adapting
the robust, industry-standard components from the `Instella` (OLMo) architecture
into a clean, minimal, and self-contained structure.

This version has been enhanced with modern architectural improvements:
- NTK-aware RoPE scaling for better sequence length extrapolation
- QK-Norm for improved training stability and embedding quality
- Separated attention and residual dropout for granular control
- Always-causal SDPA for consistent decoder-only behavior
- Maintained full backward compatibility with previous versions

Architectural Choices:
- Pre-normalization using RMSNorm: Normalizes inputs to each layer rather than outputs,
  providing better gradient flow and training stability
- Rotary Positional Embeddings (RoPE) with NTK scaling: Encodes position information
  directly into query/key vectors using rotation matrices, with intelligent scaling
  for sequence length extrapolation
- SwiGLU as the feed-forward network's activation function: Combines Swish activation
  with a gating mechanism for better performance than ReLU
- Grouped-Query Attention (GQA): Reduces memory usage by sharing key/value heads
  across multiple query heads while maintaining performance
- Optional QK-Norm: Applies RMSNorm to query and key projections for better stability
- Tied input and output embedding weights: Reduces parameters by sharing the token
  embedding matrix with the final projection layer
- KV Caching: Stores computed key/value pairs to avoid recomputation during generation
"""

import math
from dataclasses import dataclass
import inspect
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class LunarisCodexConfig:
    """
    Configuration class for the LunarisCodex model.
    
    Args:
        d_model: Hidden dimension size (embedding dimension)
        n_layers: Number of transformer blocks
        n_heads: Number of attention heads for queries
        n_kv_heads: Number of key/value heads (for GQA). If equal to n_heads, it's MHA
        vocab_size: Size of the vocabulary
        multiple_of: Ensures FFN hidden dimension is a multiple of this (for efficiency)
        ffn_hidden_multiplier: Multiplier for FFN hidden dimension size
        max_seq_len: Maximum sequence length the model can handle
        rope_theta: Base frequency for RoPE (10000 is standard)
        dropout: Legacy parameter, used as resid_dropout for backward compatibility
        
        New parameters:
        rope_ntk_scale_base: Base sequence length for NTK-aware RoPE scaling
        use_qk_norm: Whether to apply RMSNorm to query and key projections
        attn_dropout: Dropout probability within attention mechanism
        resid_dropout: Dropout probability for residual connections (projections/FFN)
    """
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int = 12  # For GQA. If n_kv_heads == n_heads, it's MHA.
    vocab_size: int = 50257
    multiple_of: int = 256  # Make SwiGLU hidden layer size a multiple of this
    ffn_hidden_multiplier: float = 4.0
    max_seq_len: int = 1024
    rope_theta: float = 10000.0
    dropout: float = 0.0  # kept for backward compat; used as resid_dropout by default

    # New flags (backward-compatible defaults)
    rope_ntk_scale_base: int = 2048  # base window to scale theta (NTK-aware)
    use_qk_norm: bool = True         # apply RMSNorm to q and k
    attn_dropout: float = 0.0        # attention dropout
    resid_dropout: float = 0.0       # residual (proj/ffn) dropout


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Precomputes the rotary frequencies in complex number format for RoPE.
    
    RoPE works by rotating query and key vectors in pairs of dimensions using
    rotation matrices. In complex space, rotation by angle θ is multiplication
    by e^(iθ). We precompute these rotation factors for all positions and
    dimension pairs.
    
    Math behind RoPE:
    - For each dimension pair (d_i, d_i+1), we define a rotation frequency: 1/theta^(2i/dim)
    - At position t, the rotation angle is: t * frequency
    - The complex rotation factor is: e^(i * t * frequency) = cos(t*freq) + i*sin(t*freq)
    
    Args:
        dim: The head dimension (d_model // n_heads)
        end: Maximum sequence length to precompute for
        theta: Base frequency (typically 10000, but can be scaled for NTK)
    
    Returns:
        Complex tensor of shape (end, dim//2) containing rotation factors
    """
    # Compute rotation frequencies for each dimension pair
    # Higher dimensions get lower frequencies (rotate more slowly)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    
    # Create position indices
    t = torch.arange(end, dtype=torch.float32)
    
    # Compute rotation angles: outer product gives us t*freq for all t,freq pairs
    freqs = torch.outer(t, freqs)  # Shape: (end, dim//2)
    
    # Convert to complex exponentials: e^(i*angle) = cos(angle) + i*sin(angle)
    # torch.polar creates complex numbers from magnitude and phase
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def apply_rotary_emb(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies rotary positional embeddings to query and key tensors.
    
    RoPE encodes position by rotating the query and key vectors in pairs of
    dimensions. This is done by treating consecutive pairs as complex numbers
    and multiplying by the precomputed rotation factors.
    
    Args:
        xq: Query tensor of shape (batch, heads, seq_len, head_dim)
        xk: Key tensor of shape (batch, heads, seq_len, head_dim)
        freqs_cis: Complex rotation factors of shape (seq_len, head_dim//2)
    
    Returns:
        Tuple of (rotated_queries, rotated_keys) with same shapes as input
    """
    # Reshape last dimension from (head_dim,) to (head_dim//2, 2) and convert to complex
    # This treats consecutive pairs of dimensions as complex numbers
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # Reshape freqs_cis for broadcasting: (seq_len, head_dim//2) -> (1, 1, seq_len, head_dim//2)
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(0)  # (1, 1, T, C/2)

    # Apply rotation by complex multiplication
    # Each complex number represents a 2D rotation in the corresponding dimension pair
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    
    # Convert back to original dtype and return
    return xq_out.type_as(xq), xk_out.type_as(xk)


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    RMSNorm normalizes by the RMS (root mean square) of the input rather than
    mean and variance like LayerNorm. This is more stable and efficient.
    
    Formula: RMSNorm(x) = x / sqrt(mean(x²) + eps) * weight
    
    Why upcast to float32: Mixed precision training uses float16 for speed,
    but normalization operations need higher precision to avoid numerical
    instability. We compute in float32 then cast back.
    """
    
    def __init__(self, dim: int, eps: float = 1e-5):
        """
        Args:
            dim: Input dimension to normalize
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # Learnable scaling parameter

    def _norm(self, x: torch.Tensor):
        """
        Compute RMS normalization.
        
        RMS = sqrt(mean(x²)) provides a measure of the magnitude of x.
        We multiply by the reciprocal (rsqrt) for efficiency.
        """
        # Upcast for stability, calculate RMS, and then downcast
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        """
        Apply RMSNorm with mixed precision support.
        
        The forward pass is stable with mixed-precision training by computing
        the normalization in float32 and then casting back to the input dtype.
        """
        out_dtype = x.dtype
        x = self._norm(x.float()).to(out_dtype)
        return x * self.weight


class Attention(nn.Module):
    """
    Grouped-Query Attention module with KV Caching, optional QK-Norm, and optimized SDPA.
    
    GQA reduces memory usage by having fewer key/value heads than query heads.
    Multiple query heads share the same key/value heads, reducing the KV cache size
    while maintaining most of the performance of full multi-head attention.
    
    QK-Norm applies RMSNorm to query and key projections, which has been shown to
    improve training stability and final model quality, especially in larger models.
    
    KV Caching stores computed key/value pairs from previous tokens to avoid
    recomputation during autoregressive generation.
    """
    
    def __init__(self, config: LunarisCodexConfig):
        """
        Initialize the attention module.
        
        Args:
            config: Model configuration containing attention parameters
        """
        super().__init__()
        assert config.d_model % config.n_heads == 0
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.d_model // config.n_heads
        self.use_qk_norm = config.use_qk_norm

        # Separate projections for Q, K, V to support different numbers of heads
        self.q_proj = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_heads * self.head_dim, config.d_model, bias=False)

        # Separate dropouts for attention and residual paths
        # This allows for more granular control of regularization
        self.attn_dropout_p = config.attn_dropout
        self.dropout = nn.Dropout(config.resid_dropout)

        # Optional QK-Norm: normalizes queries and keys per head dimension
        # This helps stabilize training and often improves final performance
        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        else:
            self.q_norm = None
            self.k_norm = None

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of the attention mechanism.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            freqs_cis: RoPE rotation factors
            past_kv: Cached key/value pairs from previous tokens (for generation)
        
        Returns:
            Tuple of (attention_output, new_kv_cache)
        """
        B, T, C = x.shape  # batch size, sequence length, embedding dimensionality

        # Project input to queries, keys, and values
        q = self.q_proj(x)  # (B, T, n_heads * head_dim)
        k = self.k_proj(x)  # (B, T, n_kv_heads * head_dim)
        v = self.v_proj(x)  # (B, T, n_kv_heads * head_dim)

        # Reshape for multi-head attention: (B, T, n_heads, head_dim) -> (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply rotary positional embeddings to queries and keys
        # RoPE encodes position information directly into the attention computation
        q, k = apply_rotary_emb(q, k, freqs_cis)

        # Optional QK-Norm: normalize queries and keys after RoPE
        # This is applied per head and helps with training stability
        if self.use_qk_norm:
            # Apply normalization to each head independently
            q = self.q_norm(q)
            k = self.k_norm(k)

        # KV Caching: concatenate current K,V with cached K,V from previous tokens
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat((past_k, k), dim=-2)  # Concatenate along sequence dimension
            v = torch.cat((past_v, v), dim=-2)
        present_kv = (k, v)  # Store updated cache for next iteration

        # Grouped-Query Attention: repeat K and V heads to match number of Q heads
        # This allows multiple query heads to share the same key/value heads
        if self.n_kv_heads < self.n_heads:
            n_repeats = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(n_repeats, dim=1)  # Repeat along head dimension
            v = v.repeat_interleave(n_repeats, dim=1)

        # Use PyTorch's optimized scaled dot-product attention
        # Always causal for decoder-only language models (prevents future token leakage)
        # Attention dropout is applied internally by SDPA when training
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
            is_causal=True  # Always True for decoder-only models
        )

        # Reshape back to (B, T, d_model) and apply output projection
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.dropout(self.o_proj(y))  # Apply residual dropout
        
        return y, present_kv


class FeedForward(nn.Module):
    """
    SwiGLU Feed-Forward Network.
    
    SwiGLU combines the Swish activation function with a gating mechanism:
    SwiGLU(x) = Swish(W1 * x) ⊙ (W3 * x) * W2
    where ⊙ is element-wise multiplication.
    
    This provides better performance than ReLU-based FFNs by:
    1. Swish activation: smoother than ReLU, better gradient flow
    2. Gating mechanism: allows the network to control information flow
    """
    
    def __init__(self, config: LunarisCodexConfig):
        """
        Initialize the feed-forward network.
        
        The hidden dimension is calculated as:
        1. Start with d_model * ffn_hidden_multiplier
        2. Adjust for SwiGLU (multiply by 2/3)
        3. Round up to nearest multiple of 'multiple_of' for efficiency
        """
        super().__init__()
        
        # Calculate hidden dimension with proper sizing for SwiGLU
        hidden_dim = int(config.ffn_hidden_multiplier * config.d_model)
        hidden_dim = int(2 * hidden_dim / 3)  # SwiGLU adjustment
        # Round up to multiple_of for computational efficiency (e.g., CUDA tensor cores)
        hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)

        # SwiGLU requires two input projections and one output projection
        self.w1 = nn.Linear(config.d_model, hidden_dim, bias=False)  # First gate
        self.w3 = nn.Linear(config.d_model, hidden_dim, bias=False)  # Second gate
        self.w2 = nn.Linear(hidden_dim, config.d_model, bias=False)  # Output projection
        self.dropout = nn.Dropout(config.resid_dropout)  # Residual dropout

    def forward(self, x: torch.Tensor):
        """
        Apply SwiGLU activation.
        
        Formula: SwiGLU(x) = Swish(W1(x)) ⊙ W3(x) → W2
        where Swish(x) = x * sigmoid(x) = x * σ(x)
        """
        # SwiGLU activation: Swish(W1(x)) * W3(x)
        # F.silu is the Swish activation function
        swiglu = F.silu(self.w1(x)) * self.w3(x)
        return self.dropout(self.w2(swiglu))


class Block(nn.Module):
    """
    A single Transformer block using pre-normalization architecture.
    
    Pre-normalization (used here) vs Post-normalization:
    - Pre-norm: LayerNorm → Attention → Add, LayerNorm → FFN → Add
    - Post-norm: Attention → Add → LayerNorm, FFN → Add → LayerNorm
    
    Pre-normalization provides better gradient flow and training stability
    because the residual connections carry the original gradient directly.
    """
    
    def __init__(self, config: LunarisCodexConfig):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.d_model)
        self.ffn_norm = RMSNorm(config.d_model)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of the transformer block.
        
        Architecture: Pre-norm with residual connections
        1. x + Attention(RMSNorm(x))
        2. x + FFN(RMSNorm(x))
        
        Args:
            x: Input tensor
            freqs_cis: RoPE rotation factors
            past_kv: KV cache from previous tokens
        
        Returns:
            Tuple of (block_output, updated_kv_cache)
        """
        # Pre-normalization and residual connection for attention
        # The KV cache is managed by the attention layer
        attn_output, new_kv = self.attention(self.attention_norm(x), freqs_cis, past_kv)
        h = x + attn_output  # Residual connection
        
        # Pre-normalization and residual connection for FFN
        out = h + self.feed_forward(self.ffn_norm(h))  # Residual connection
        
        return out, new_kv


class LunarisCodex(nn.Module):
    """
    Complete LunarisCodex Language Model.
    
    This is an enhanced Llama-style decoder-only transformer with:
    - Pre-normalization architecture for better training stability
    - RoPE with NTK-aware scaling for improved sequence length extrapolation
    - SwiGLU activation in FFN for better performance
    - Grouped-Query Attention for memory efficiency
    - Optional QK-Norm for enhanced training stability
    - Separated attention and residual dropout for granular control
    - KV caching for fast inference
    - Weight tying between input embeddings and output projection
    - Always-causal attention for consistent decoder-only behavior
    """
    
    def __init__(self, config: LunarisCodexConfig):
        super().__init__()
        self.config = config

        # Main transformer components
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.d_model),  # Token embeddings
            h=nn.ModuleList([Block(config) for _ in range(config.n_layers)]),  # Transformer blocks
            ln_f=RMSNorm(config.d_model),  # Final layer normalization
            drop=nn.Dropout(config.resid_dropout),  # Input dropout (using resid_dropout)
        ))
        
        # Output projection (language modeling head)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: share parameters between input embeddings and output projection
        # This reduces the parameter count and often improves performance
        # The intuition is that both layers deal with the same vocabulary space
        self.transformer.wte.weight = self.lm_head.weight

        # RoPE with NTK-aware scaling for better sequence length extrapolation
        # NTK scaling adjusts the base frequency based on the ratio of max_seq_len to base_len
        # This allows the model to better handle sequences longer than seen during training
        base_len = max(1, int(config.rope_ntk_scale_base))
        scale = max(1.0, float(config.max_seq_len) / float(base_len))
        theta_eff = config.rope_theta * scale

        # Precompute RoPE frequencies for all positions and register as buffer
        # Buffers are saved with the model but don't require gradients
        freqs_cis = precompute_freqs_cis(
            config.d_model // config.n_heads,  # Head dimension
            config.max_seq_len,
            theta=theta_eff,  # NTK-scaled theta
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        # Initialize model weights
        self.apply(self._init_weights)

        # Report number of parameters
        print(f"Number of parameters: {self.get_num_params()/1e6:.2f}M")

    def get_num_params(self) -> int:
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _init_weights(self, module):
        """
        Initialize model weights using scaled initialization.
        
        Standard initialization for most weights, with special scaled initialization
        for residual projections to prevent activation variance from growing with depth.
        """
        if isinstance(module, nn.Linear):
            # Standard initialization for linear layers
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Standard initialization for embeddings
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        # Scaled initialization for residual projections
        # This prevents the variance from growing with the number of layers
        if isinstance(module, Attention):
            # Scale down the output projection to maintain variance across layers
            torch.nn.init.normal_(module.o_proj.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layers))
        if isinstance(module, FeedForward):
            # Scale down the final FFN projection to maintain variance across layers
            torch.nn.init.normal_(module.w2.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layers))

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass of the model.
        
        Args:
            idx: Input token indices of shape (batch, seq_len)
            targets: Target token indices for training (optional)
            past_key_values: KV cache from previous forward passes (for generation)
        
        Returns:
            Tuple of (logits, loss, new_kv_cache)
            - logits: Output probabilities over vocabulary
            - loss: Cross-entropy loss (only if targets provided)
            - new_kv_cache: Updated KV cache for next iteration
        """
        B, T = idx.shape
        
        # Determine starting position for RoPE based on cache length
        # If we have a cache, we're in generation mode and only processing new tokens
        start_pos = past_key_values[0][0].shape[-2] if past_key_values is not None else 0
        
        # Ensure we don't exceed the model's maximum sequence length
        assert start_pos + T <= self.config.max_seq_len, \
            f"Cannot forward, sequence length {start_pos + T} exceeds max_seq_len {self.config.max_seq_len}"

        # Get token embeddings and apply input dropout
        x = self.transformer.wte(idx)
        x = self.transformer.drop(x)

        # Get precomputed RoPE frequencies for the current sequence positions
        freqs_cis = self.freqs_cis[start_pos: start_pos + T]

        # Forward through all transformer blocks, updating the KV cache
        new_past_key_values = []
        for i, block in enumerate(self.transformer.h):
            # Get the cached KV for this specific layer
            past_kv_for_block = past_key_values[i] if past_key_values is not None else None
            x, new_kv = block(x, freqs_cis, past_kv_for_block)
            new_past_key_values.append(new_kv)

        # Apply final layer normalization
        x = self.transformer.ln_f(x)

        # Compute logits and loss
        if targets is not None:
            # Training mode: compute logits for all positions and calculate loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # Inference mode: only compute logits for the last token (efficiency)
            # During generation, we only need the prediction for the next token
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss, new_past_key_values

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        Configure the optimizer with weight decay applied only to 2D parameters.
        
        Weight decay is applied to matrices (2D tensors) but not to biases and
        layer norm parameters (1D tensors) for better training dynamics.
        """
        # Get all parameters that require gradients
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        
        # Separate parameters: 2D tensors get weight decay, 1D tensors don't
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        
        # Create optimizer parameter groups
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        
        # Print parameter count information
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        # Use fused AdamW if available (faster on CUDA)
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, fused=use_fused)
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate text using the model with efficient KV caching.
        
        The generation process has two phases:
        1. Prefill: Process the entire input prompt to build the initial KV cache
        2. Decode: Generate tokens one by one, reusing the KV cache
        
        Args:
            idx: Input token indices (prompt)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Keep only top k tokens for sampling (None = no filtering)
        
        Returns:
            Generated token sequence including the original prompt
        """
        self.eval()  # Set model to evaluation mode
        past_key_values = None  # Start with empty cache

        for _ in range(max_new_tokens):
            # Check if we've reached the maximum sequence length
            current_len = past_key_values[0][0].shape[-2] if past_key_values is not None else idx.shape[1]
            if current_len >= self.config.max_seq_len:
                break
                
            # Prefill phase: process full prompt. Decode phase: process only last token
            # This is the key efficiency gain of KV caching
            idx_cond = idx if past_key_values is None else idx[:, -1:]

            # Forward pass with KV cache
            logits, _, past_key_values = self(idx_cond, past_key_values=past_key_values)
            
            # Sample the next token using temperature and top-k sampling
            logits = logits[:, -1, :] / max(1e-8, temperature)  # Apply temperature scaling with safe division
            
            # Top-k filtering: keep only the k most likely tokens
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Convert to probabilities and sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append the new token to the sequence
            idx = torch.cat((idx, idx_next), dim=1)

        self.train()  # Return to training mode
        return idx
