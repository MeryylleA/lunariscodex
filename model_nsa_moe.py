# MODIFIED: The entire file has been refactored to integrate Native Sparse Attention (NSA)
# with the existing Mixture-of-Experts (MoE) implementation.

"""
Full definition of a LunarisCodex Language Model, all of it in this single file.
This version is a refactored and simplified Llama-style model, created by adapting
the robust, industry-standard components from the `Instella` (OLMo) architecture
into a clean, minimal, and self-contained structure.

This version has been adapted to include:
1.  Native Sparse Attention (NSA): Replaces the standard attention mechanism for
    efficient long-sequence processing. It combines coarse-grained compression,
    fine-grained token selection, and a sliding window.
2.  Mixture-of-Experts (MoE): Replaces the standard FeedForward network with a
    sparsely-gated MoE layer (Switch Transformer, k=1 routing) for increased
    parameter capacity with low computational overhead.
"""

import math
from dataclasses import dataclass
import inspect
from typing import Optional, Tuple, List, Union, Dict

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

# --- NEW: Imports required for NativeSparseAttention ---
from __future__ import annotations
import warnings
from einops import rearrange
from fla.models.utils import Cache
from fla.modules import RotaryEmbedding
from native_sparse_attention.ops.parallel import parallel_nsa


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
        dropout: Dropout probability for regularization

        --- MoE CONFIGURATIONS ---
        n_experts: Total number of experts in the MoE layer. If None, uses standard FFN.
        n_experts_per_token: Number of experts to route each token to. We'll use 1 for Switch Transformer.
        aux_loss_weight: Multiplier for the auxiliary load balancing loss.

        --- NEW: NSA Parameters ---
        block_size: Block size for NSA's coarse-grained compression and fine-grained selection.
        block_counts: Number of selected blocks in NSA.
        window_size: Sliding window size for local attention in NSA.
    """
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int = 12
    vocab_size: int = 50257
    multiple_of: int = 256
    ffn_hidden_multiplier: float = 4.0
    max_seq_len: int = 1024
    rope_theta: float = 10000.0
    dropout: float = 0.0
    # --- MoE Params ---
    n_experts: Optional[int] = 8 # Example: 8 experts
    n_experts_per_token: int = 1 # For Switch Transformer, this is 1
    aux_loss_weight: float = 1e-2 # From the Switch Transformer paper
    # --- NEW: NSA Parameters ---
    block_size: int = 64
    block_counts: int = 16
    window_size: int = 512


# --- NEW: NativeSparseAttention class copied from modeling_nsa.py ---
class NativeSparseAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 64,
        num_kv_heads: Optional[int] = 4,
        head_dim: int = 64,
        qkv_bias: bool = False,
        block_size: Optional[int] = 64,
        block_counts: Optional[Union[torch.LongTensor, int]] = 16,
        window_size: Optional[int] = 512,
        rope_theta: Optional[float] = 10000.,
        max_position_embeddings: Optional[int] = None,
        layer_idx: int = None
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        if num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        else:
            self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        self.head_dim = head_dim
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.qkv_bias = qkv_bias

        self.block_size = block_size
        self.block_counts = block_counts
        self.window_size = window_size
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.layer_idx = layer_idx

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=self.qkv_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=self.qkv_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=self.qkv_bias)
        self.g_proj = nn.Linear(self.hidden_size, self.num_heads * 3, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary = RotaryEmbedding(dim=self.head_dim, base=self.rope_theta)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, seq_len, _ = hidden_states.size()

        q = rearrange(self.q_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)
        k = rearrange(self.k_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)
        v = rearrange(self.v_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)
        g = rearrange(self.g_proj(hidden_states), '... (h d) -> ... h d', d=3)
        g_cmp, g_slc, g_swa = g.sigmoid().unbind(-1)

        cu_seqlens = kwargs.get('cu_seqlens', None)

        seqlen_offset, max_seqlen = 0, seq_len
        if past_key_values is not None:
            seqlen_offset = past_key_values.get_seq_length(self.layer_idx)
            max_seqlen = q.shape[1] + seqlen_offset

            if attention_mask is not None:
                # to deliminate the offsets of padding tokens
                seqlen_offset = (seqlen_offset + attention_mask.sum(-1) - attention_mask.shape[-1]).clamp(min=0)
                max_seqlen = q.shape[1] + max(seqlen_offset)

        if self.max_position_embeddings is not None:
            max_seqlen = max(max_seqlen, self.max_position_embeddings)
        q, k = self.rotary(q, k, seqlen_offset=seqlen_offset, max_seqlen=max_seqlen, cu_seqlens=cu_seqlens)

        if use_cache:
            cache_has_content = past_key_values.get_seq_length(self.layer_idx) > 0
            k_cached, v_cached = past_key_values.update(
                attn_state=(k.flatten(-2, -1), v.flatten(-2, -1)),
                layer_idx=self.layer_idx,
                offset=seq_len,
                cache_kwargs=dict(window_size=self.window_size)
            )['attn_state']
            if cache_has_content:
                k, v = k_cached, v_cached
                k = rearrange(k, '... (h d) -> ... h d', d=self.head_dim)
                v = rearrange(v, '... (h d) -> ... h d', d=self.head_dim)

        o = parallel_nsa(
            q=q,
            k=k,
            v=v,
            g_cmp=g_cmp,
            g_slc=g_slc,
            g_swa=g_swa,
            block_size=self.block_size,
            block_counts=self.block_counts,
            window_size=self.window_size,
            cu_seqlens=cu_seqlens,
            head_first=False
        )
        o = o.reshape(batch_size, seq_len, -1)
        o = self.o_proj(o)

        if not output_attentions:
            attentions = None

        return o, attentions, past_key_values


# The original FeedForward class is kept, as it will be used as the "expert" network.
class FeedForward(nn.Module):
    def __init__(self, config: LunarisCodexConfig):
        super().__init__()
        hidden_dim = int(config.ffn_hidden_multiplier * config.d_model)
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)

        self.w13 = nn.Linear(config.d_model, 2 * hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor):
        gate, up = self.w13(x).chunk(2, dim=-1)
        swiglu = F.silu(gate) * up
        return self.dropout(self.w2(swiglu))

# --- NEW: Mixture-of-Experts Layer (Unchanged from original file) ---
class MixtureOfExperts(nn.Module):
    """
    A Sparsely-Gated Mixture-of-Experts layer implementing Switch Transformer routing (k=1).
    This layer replaces the standard FFN in a Transformer block, enabling conditional
    computation. Instead of all tokens passing through the same FFN, each token is
    dynamically routed to one of several "expert" FFNs.

    This sparsity is the source of MoE's computational savings, but it introduces
    a challenge: load balancing. If the gating network learns to always route tokens
    to a few popular experts, the other experts are not trained. To prevent this,
    an auxiliary "load balancing loss" is added, encouraging the gate to distribute
    tokens evenly across all experts.

    Args:
        config: The main model configuration.
    """
    def __init__(self, config: LunarisCodexConfig):
        super().__init__()
        self.n_experts = config.n_experts
        self.n_experts_per_token = config.n_experts_per_token # Should be 1 for Switch Transformer

        # The gating network (router) is a simple linear layer that takes a token's
        # representation and outputs a logit for each expert.
        self.gate = nn.Linear(config.d_model, self.n_experts, bias=False)

        # The experts are a list of standard FeedForward networks.
        self.experts = nn.ModuleList([FeedForward(config) for _ in range(self.n_experts)])

        # The weight for the auxiliary loss, as defined in the Switch Transformer paper.
        # This controls how much we penalize unbalanced expert utilization.
        self.aux_loss_weight = config.aux_loss_weight

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for the MoE layer.

        1.  Reshape input for token-level routing.
        2.  Compute router logits and probabilities using the gating network.
        3.  Calculate the auxiliary load balancing loss. This is crucial for stable training.
            The loss encourages the gating network to distribute tokens evenly, preventing a
            state where only a few experts are ever used.
        4.  Select the top-1 expert for each token based on the highest router probability.
        5.  Dispatch tokens to their selected experts. Only the chosen expert computes for a given token.
        6.  Combine expert outputs, scaling them by the router probabilities. This scaling is
            critical because it makes the discrete routing decision differentiable, allowing
            gradients to flow back to the gating network.
        7.  Return the final output and the auxiliary loss.
        """
        batch_size, seq_len, d_model = x.shape

        # Reshape input from (batch_size, seq_len, d_model) to (total_tokens, d_model)
        # to perform routing for each token independently.
        x_flat = x.view(-1, d_model)
        num_tokens = x_flat.shape[0]

        # 1. Compute router logits using the gating network.
        # Upcast to float32 for numerical stability during softmax and loss calculation,
        # as recommended in the Switch Transformer paper.
        router_logits = self.gate(x_flat.float())
        router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)

        # 2. Calculate the auxiliary load balancing loss.
        # The loss is the product of two terms for each expert, summed over all experts:
        # a) `prob_mass_per_expert`: The average probability assigned to an expert by the router.
        # b) `tokens_per_expert`: The actual fraction of tokens routed to that expert.
        # The loss is minimized when both of these values are equal for all experts (balanced load).
        prob_mass_per_expert = router_probs.mean(dim=0)
        expert_indices = torch.argmax(router_probs, dim=-1)
        expert_one_hot = F.one_hot(expert_indices, num_classes=self.n_experts).float()
        tokens_per_expert = expert_one_hot.mean(dim=0)
        # The loss is scaled by the number of experts and a configurable weight.
        aux_loss = self.aux_loss_weight * self.n_experts * (prob_mass_per_expert * tokens_per_expert).sum()

        # 3. Get the top-1 expert index and its corresponding probability for each token.
        # For Switch Transformer, we only route to one expert (k=1).
        top_k_probs, top_k_indices = torch.topk(router_probs, self.n_experts_per_token, dim=-1)
        top_expert_indices = top_k_indices.squeeze(1) # Shape: (num_tokens,)
        top_expert_probs = top_k_probs.squeeze(1)     # Shape: (num_tokens,)

        # 4. Dispatch tokens to experts and combine results.
        final_output = torch.zeros_like(x_flat)
        # We iterate through each expert and process all tokens routed to it in a single batch.
        for i in range(self.n_experts):
            # Create a boolean mask for tokens routed to the current expert.
            token_mask = (top_expert_indices == i)
            if token_mask.sum() == 0:
                continue # No tokens routed to this expert, skip.

            # Select the input tokens for the current expert.
            expert_inputs = x_flat[token_mask]
            # Compute the output from the expert.
            expert_outputs = self.experts[i](expert_inputs)
            # Place the expert outputs into the correct positions in the final output tensor.
            final_output[token_mask] = expert_outputs.to(final_output.dtype)

        # 5. Scale the output by the gate probability and reshape back.
        # This is a key step to make the routing decision differentiable. The gradient
        # can flow through `top_expert_probs` back to the gating network's weights.
        final_output = final_output * top_expert_probs.unsqueeze(-1)
        final_output = final_output.view(batch_size, seq_len, d_model)

        # Reshape expert_indices back to (batch_size, seq_len) for tracking
        expert_indices_reshaped = expert_indices.view(batch_size, seq_len)

        return final_output, aux_loss, expert_indices_reshaped


# --- MODIFIED: Block to use NativeSparseAttention and support MoE ---
class Block(nn.Module):
    """
    A single Transformer block, now with Native Sparse Attention and a choice
    between a standard FFN and an MoE layer.
    """
    def __init__(self, config: LunarisCodexConfig, layer_idx: int): # MODIFIED: Added layer_idx
        super().__init__()
        # NEW: Pre-attention RMSNorm, as used in the reference NSABlock
        self.attn_norm = nn.RMSNorm(config.d_model, eps=1e-5)
        # NEW: Instantiate NativeSparseAttention instead of standard Attention
        self.attention = NativeSparseAttention(
            hidden_size=config.d_model,
            num_heads=config.n_heads,
            num_kv_heads=config.n_kv_heads,
            head_dim=config.d_model // config.n_heads,
            qkv_bias=False, # Consistent with original wqkv layer
            block_size=config.block_size,
            block_counts=config.block_counts,
            window_size=config.window_size,
            rope_theta=config.rope_theta,
            max_position_embeddings=config.max_seq_len,
            layer_idx=layer_idx
        )
        self.ffn_norm = nn.RMSNorm(config.d_model, eps=1e-5)

        # Conditionally create either a standard FFN or a Mixture-of-Experts layer.
        if config.n_experts is not None and config.n_experts > 0:
            self.feed_forward = MixtureOfExperts(config)
            self.is_moe = True
            print(f"Block {layer_idx} initialized with Mixture-of-Experts ({config.n_experts} experts).")
        else:
            self.feed_forward = FeedForward(config)
            self.is_moe = False
            print(f"Block {layer_idx} initialized with standard FeedForward network.")

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None, # MODIFIED: New argument
        past_kv: Optional[Cache] = None, # MODIFIED: Type is now Cache
        use_cache: bool = False # MODIFIED: New argument
    ) -> Tuple[torch.Tensor, Optional[Cache], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass of the transformer block. Now uses activation checkpointing to save memory during training.
        """
        def _inner_forward(
            x_inner: torch.Tensor,
            attention_mask_inner: Optional[torch.Tensor] = None,
            past_kv_inner: Optional[Cache] = None,
            use_cache_inner: bool = False
        ):
            # Part 1: Native Sparse Attention (MODIFIED)
            residual = x_inner
            hidden_states_norm = self.attn_norm(x_inner)
            attn_output, _, new_past_kv = self.attention(
                hidden_states=hidden_states_norm,
                attention_mask=attention_mask_inner,
                past_key_values=past_kv_inner,
                use_cache=use_cache_inner
            )
            h = residual + attn_output # First residual connection

            # Part 2: FFN / MoE Layer (MODIFIED)
            residual = h # Second residual connection starts from the output of the first
            ffn_input = self.ffn_norm(h)

            aux_loss = None
            expert_indices = None
            if self.is_moe:
                ffn_output, aux_loss, expert_indices = self.feed_forward(ffn_input)
            else:
                ffn_output = self.feed_forward(ffn_input)

            out = residual + ffn_output # Second residual connection

            return out, new_past_kv, aux_loss, expert_indices

        if self.training:
            # During training, use checkpointing to save memory.
            return checkpoint(
                _inner_forward, x, attention_mask, past_kv, use_cache, use_reentrant=False
            )
        else:
            # During evaluation, run the forward pass directly.
            return _inner_forward(x, attention_mask, past_kv, use_cache)


# --- MODIFIED: Main LunarisCodex class to handle NSA and auxiliary loss ---
class LunarisCodex(nn.Module):
    """
    Complete LunarisCodex Language Model, now with Native Sparse Attention and optional MoE support.
    """
    def __init__(self, config: LunarisCodexConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.d_model),
            # MODIFIED: Pass layer_idx to each Block
            h = nn.ModuleList([Block(config, i) for i in range(config.n_layers)]),
            ln_f = nn.RMSNorm(config.d_model, eps=1e-5),
            drop = nn.Dropout(config.dropout),
        ))

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        # MODIFIED: Removed freqs_cis buffer, as NSA handles RoPE internally.

        self.apply(self._init_weights)

        print(f"Number of parameters: {self.get_num_params()/1e6:.2f}M")
        if config.n_experts is not None:
             print("Note: Parameter count includes all experts. Active parameters per forward pass are much lower.")

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        # MODIFIED: Adapted weight initialization for NSA
        if isinstance(module, (NativeSparseAttention, FeedForward, MixtureOfExperts)):
            for name, p in module.named_parameters():
                if name.endswith("o_proj.weight") or name.endswith("w2.weight"):
                    torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layers))

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None, # MODIFIED: Type is now Cache
        attention_mask: Optional[torch.Tensor] = None # MODIFIED: New argument
    ) -> Tuple[torch.Tensor, Optional[tuple], Optional[Cache], Optional[List[torch.Tensor]]]:
        B, T = idx.shape
        # MODIFIED: Infer use_cache and initialize Cache object for generation
        use_cache = (past_key_values is not None) or (targets is None)
        if use_cache and past_key_values is None:
            past_key_values = Cache()

        x = self.transformer.wte(idx)
        x = self.transformer.drop(x)
        # MODIFIED: Removed freqs_cis calculation

        total_aux_loss = 0.0
        expert_indices_list = []

        # MODIFIED: Updated loop to pass the entire cache object through each block
        for block in self.transformer.h:
            x, past_key_values, aux_loss, expert_indices = block(
                x,
                attention_mask=attention_mask,
                past_kv=past_key_values,
                use_cache=use_cache
            )
            if aux_loss is not None:
                total_aux_loss += aux_loss
            if expert_indices is not None:
                expert_indices_list.append(expert_indices)

        x = self.transformer.ln_f(x)

        loss = None
        if targets is not None:
            logits = self.lm_head(x)
            main_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            num_moe_layers = sum(1 for block in self.transformer.h if getattr(block, 'is_moe', False))
            final_aux_loss = total_aux_loss
            if num_moe_layers > 0:
                final_aux_loss /= num_moe_layers
            total_loss = main_loss + final_aux_loss
            loss = (total_loss, main_loss, final_aux_loss)
            return logits, loss, past_key_values, expert_indices_list
        else: # Inference mode
            logits = self.lm_head(x[:, [-1], :])
            loss = None
            return logits, loss, past_key_values, None

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, fused=use_fused)
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        self.eval()
        past_key_values = None # MODIFIED: Starts as None, will become a Cache object
        for _ in range(max_new_tokens):
            # MODIFIED: Logic to get current length from Cache object
            if past_key_values is not None:
                current_len = past_key_values.get_seq_length()
            else:
                current_len = idx.shape[1]

            if current_len >= self.config.max_seq_len:
                break

            idx_cond = idx if past_key_values is None else idx[:, -1:]
            logits, _, past_key_values, _ = self(idx_cond, past_key_values=past_key_values)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        self.train()
        return idx
