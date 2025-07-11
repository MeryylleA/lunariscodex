"""
Full definition of a LunarisCodex Language Model, all of it in this single file.
This version is a refactored and simplified Llama-style model, created by adapting
the robust, industry-standard components from the `Instella` (OLMo) architecture
into a clean, minimal, and self-contained structure.

This version has been refactored to include KV Caching for efficient inference.

This model includes modifications for μ-Parametrization (μP), specifically targeting
the output layer's initialization and learning rate to improve training stability
and enable hyperparameter transfer across model sizes, as described in
"Tensor Programs V" (Yang et al., 2023).

This model has been augmented with LayerSkip (Gated Residual Connections) to allow
the model to dynamically learn to skip sub-layers for each token.

This version has been augmented with Split-Head RoPE, allowing each attention
head to process both position-sensitive (RoPE) and position-agnostic features.
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
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int = 12  # For GQA. If n_kv_heads == n_heads, it's MHA.
    vocab_size: int = 50257
    multiple_of: int = 256  # Make SwiGLU hidden layer size a multiple of this
    ffn_hidden_multiplier: float = 4.0
    max_seq_len: int = 1024
    rope_theta: float = 10000.0
    dropout: float = 0.0
    # --- Split-Head RoPE Change: Add head dimension for RoPE ---
    rope_head_dim: int = 64


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Precomputes the rotary frequencies in complex number format.
    Adapted from `instella_model.py`'s `RotaryEmbedding` but simplified
    to use a buffer instead of a custom cache.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def apply_rotary_emb(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies rotary positional embeddings to query and key tensors.
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # Reshape freqs_cis for broadcasting
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(0)  # (1, 1, T, C/2)

    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    Adapted from `instella_model.py`'s `RMSLayerNorm`.
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        # Upcast for stability, calculate RMS, and then downcast
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        # The forward pass is stable with mixed-precision training
        output_dtype = x.dtype
        x = self._norm(x.float()).to(output_dtype)
        return x * self.weight


class Attention(nn.Module):
    """
    Grouped-Query Attention with Split-Head RoPE and KV Caching.
    Logic adapted from `instella_model.py`'s `InstellaLlamaBlock`.
    """
    def __init__(self, config: LunarisCodexConfig):
        super().__init__()
        self.config = config # --- Split-Head RoPE Change: Save config ---
        assert config.d_model % config.n_heads == 0
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.d_model // config.n_heads

        # --- Split-Head RoPE Change: Add assertion ---
        assert config.rope_head_dim <= self.head_dim, \
            "RoPE dimension cannot be larger than head dimension."

        self.q_proj = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_heads * self.head_dim, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, T, C = x.shape  # batch size, sequence length, embedding dimensionality

        # QKV projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for attention calculation
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)    # (B, n_heads, T, head_dim)
        k = k.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2) # (B, n_kv_heads, T, head_dim)
        v = v.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2) # (B, n_kv_heads, T, head_dim)

        # --- Split-Head RoPE Change: Apply RoPE to a subset of the head dimension ---
        rope_dim = self.config.rope_head_dim
        
        # Split the query and key heads into RoPE and non-RoPE parts
        q_rope, q_nope = q.split([rope_dim, self.head_dim - rope_dim], dim=-1)
        k_rope, k_nope = k.split([rope_dim, self.head_dim - rope_dim], dim=-1)

        # Apply rotary embeddings only to the RoPE part
        q_rope, k_rope = apply_rotary_emb(q_rope, k_rope, freqs_cis)
        
        # Concatenate the parts back together
        q = torch.cat([q_rope, q_nope], dim=-1)
        k = torch.cat([k_rope, k_nope], dim=-1)
        # --------------------- End of Split-Head RoPE Change ---------------------

        # KV Caching: if a past cache is provided, concatenate along the sequence dimension
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat((past_k, k), dim=-2)
            v = torch.cat((past_v, v), dim=-2)
        present_kv = (k, v)

        # Grouped-Query Attention: repeat K and V heads
        if self.n_kv_heads < self.n_heads:
            n_repeats = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(n_repeats, dim=1)
            v = v.repeat_interleave(n_repeats, dim=1)

        # Use flash attention. is_causal is True only for the prefill step.
        is_causal = past_kv is None
        y = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

        # Concatenate heads and project back to embedding space
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.dropout(self.o_proj(y))

        # Return attention output and the updated KV cache
        return y, present_kv


class FeedForward(nn.Module):
    """
    SwiGLU Feed-Forward Network.
    Logic adapted from `instella_model.py`'s `SwiGLU` and Llama's FFN structure.
    """
    def __init__(self, config: LunarisCodexConfig):
        super().__init__()
        hidden_dim = int(config.ffn_hidden_multiplier * config.d_model)
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)

        self.w1 = nn.Linear(config.d_model, hidden_dim, bias=False)
        self.w3 = nn.Linear(config.d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor):
        # SwiGLU activation
        swiglu = F.silu(self.w1(x)) * self.w3(x)
        return self.dropout(self.w2(swiglu))


class Block(nn.Module):
    """
    A single Transformer block, Llama-style (pre-normalization), with LayerSkip
    (Gated Residual Connections) and KV cache management.
    """
    def __init__(self, config: LunarisCodexConfig):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.d_model)
        self.ffn_norm = RMSNorm(config.d_model)

        # --- LayerSkip Change: Add gating layers and biases ---
        # These small linear layers learn a token-wise scalar gate.
        self.attention_gate = nn.Linear(config.d_model, 1, bias=False)
        self.ffn_gate = nn.Linear(config.d_model, 1, bias=False)

        # We use a separate learnable bias to initialize the gates to be mostly
        # open at the start of training. sigmoid(2.0) is ~0.88.
        self.attention_gate_bias = nn.Parameter(torch.tensor(2.0))
        self.ffn_gate_bias = nn.Parameter(torch.tensor(2.0))
        # --------------------- End of LayerSkip Change ---------------------

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # --- LayerSkip Change: Apply Gated Residual for Attention ---
        normed_x = self.attention_norm(x)

        # Calculate the attention gate value
        g_attn_logits = self.attention_gate(normed_x) + self.attention_gate_bias
        g_attn = torch.sigmoid(g_attn_logits)

        # Get attention output, scaled by the gate
        attn_output, new_kv = self.attention(normed_x, freqs_cis, past_kv)
        h = x + g_attn * attn_output
        # --------------------- End of LayerSkip Change ---------------------

        # --- LayerSkip Change: Apply Gated Residual for FFN ---
        normed_h = self.ffn_norm(h)

        # Calculate the FFN gate value
        g_ffn_logits = self.ffn_gate(normed_h) + self.ffn_gate_bias
        g_ffn = torch.sigmoid(g_ffn_logits)

        # Get FFN output, scaled by the gate
        ffn_output = self.feed_forward(normed_h)
        out = h + g_ffn * ffn_output
        # --------------------- End of LayerSkip Change ---------------------

        # Return the block's output and the updated cache from the attention layer
        return out, new_kv


class LunarisCodex(nn.Module):
    def __init__(self, config: LunarisCodexConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.d_model),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
            ln_f = RMSNorm(config.d_model),
            drop = nn.Dropout(config.dropout),
        ))
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Tie input and output weights
        self.transformer.wte.weight = self.lm_head.weight

        # --- Split-Head RoPE Change: Precompute frequencies for rope_head_dim ---
        # Precompute RoPE frequencies and register as a buffer
        freqs_cis = precompute_freqs_cis(
            self.config.rope_head_dim,
            self.config.max_seq_len,
            self.config.rope_theta,
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)
        # --------------------- End of Split-Head RoPE Change ---------------------

        # Initialize weights
        self.apply(self._init_weights)

        # Report number of parameters
        print(f"Number of parameters: {self.get_num_params()/1e6:.2f}M")

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # --- μP Change: Special initialization for the output layer ---
            # According to μP, the output layer (lm_head) should be initialized
            # with a much smaller variance, ideally zero, to stabilize training
            # across different model widths.
            if module is self.lm_head:
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.0)
            else:
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            # --------------------- End of μP Change ---------------------
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        # Apply special scaled initialization for residual projections
        if isinstance(module, (Attention, FeedForward)):
            for name, p in module.named_parameters():
                if name.endswith("o_proj.weight") or name.endswith("w2.weight"):
                    torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layers))

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T = idx.shape

        # Determine start position for RoPE from the cache length
        start_pos = past_key_values[0][0].shape[-2] if past_key_values is not None else 0
        assert start_pos + T <= self.config.max_seq_len, \
            f"Cannot forward, sequence length {start_pos + T} exceeds model's max_seq_len {self.config.max_seq_len}"

        # Get token embeddings
        x = self.transformer.wte(idx)
        x = self.transformer.drop(x)

        # Get precomputed RoPE frequencies for the current sequence length and position
        freqs_cis = self.freqs_cis[start_pos : start_pos + T]

        # Forward through the transformer blocks, managing the cache
        new_past_key_values = []
        for i, block in enumerate(self.transformer.h):
            past_kv_for_block = past_key_values[i] if past_key_values is not None else None
            x, new_kv = block(x, freqs_cis, past_kv_for_block)
            new_past_key_values.append(new_kv)

        # Final normalization
        x = self.transformer.ln_f(x)

        if targets is not None:
            # If training, compute loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # At inference, only compute logits for the last token for efficiency
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss, new_past_key_values

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # --- μP Change: Create a separate parameter group for the output layer ---
        # The core idea of μP is to scale the learning rates of different layers
        # differently based on how their dimensions change with model width.
        # The output layer (lm_head) requires a much smaller learning rate
        # than the rest of the model. Here we use a simple fixed scaling factor.

        # Start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        # Isolate the lm_head parameters
        lm_head_params = [p for n, p in param_dict.items() if n.startswith("lm_head")]
        # Create a dictionary of the rest of the parameters
        other_params = {pn: p for pn, p in param_dict.items() if not pn.startswith("lm_head")}

        # Create optimizer groups for the "other" parameters.
        # Any 2D parameter will be weight decayed, otherwise no.
        decay_params = [p for n, p in other_params.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in other_params.items() if p.dim() < 2]

        # The lm_head weight is 2D, so it should be decayed. It has no bias.
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
            # Special group for the output layer with a scaled LR.
            # We use a fixed factor of 0.1 as a simple but effective heuristic.
            {'params': lm_head_params, 'weight_decay': weight_decay, 'lr': learning_rate * 0.1}
        ]
        # --------------------- End of μP Change ---------------------

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        num_lm_head_params = sum(p.numel() for p in lm_head_params)
        print(f"num decayed parameter tensors (excluding lm_head): {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors (excluding lm_head): {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        print(f"num lm_head parameter tensors: {len(lm_head_params)}, with {num_lm_head_params:,} parameters (using scaled LR)")

        # The default learning rate in AdamW is applied to groups that don't have an 'lr' key.
        # The lm_head group overrides this with its own specified learning rate.
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, fused=use_fused)
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generates text by efficiently using the KV cache.
        The first pass (prefill) processes the prompt. Subsequent passes process
        only the most recent token.
        """
        self.eval()
        past_key_values = None

        for _ in range(max_new_tokens):
            # Stop generating if the context window is full
            current_len = past_key_values[0][0].shape[-2] if past_key_values else idx.shape[1]
            if current_len >= self.config.max_seq_len:
                break

            # If a cache exists, use the last token as input. Otherwise, use the full prompt.
            idx_cond = idx if past_key_values is None else idx[:, -1:]

            # Forward pass with the cache
            logits, _, past_key_values = self(idx_cond, past_key_values=past_key_values)

            # Sample the next token
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append the new token for the next iteration
            idx = torch.cat((idx, idx_next), dim=1)

        self.train()
        return idx
