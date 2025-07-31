"""
Full definition of a LunarisCodex Language Model, all of it in this single file.
This version is a refactored and simplified Llama-style model, created by adapting
the robust, industry-standard components from the `Instella` (OLMo) architecture
into a clean, minimal, and self-contained structure.

This version has been adapted to include a Mixture-of-Experts (MoE) layer,
specifically following the principles of the Switch Transformer (k=1 routing).
The standard FeedForward network inside each Transformer Block is replaced by a
MixtureOfExperts layer.

This version has been refactored to use the QK-Reorder-LN normalization scheme
as described in the EXAONE 4.0 paper.

Key MoE additions:
- MixtureOfExperts (MoE) Layer: Replaces the FFN. Contains a gating network and multiple FFN "experts".
- Switch Routing (k=1): Each token is routed to only a single expert.
- Auxiliary Load Balancing Loss: A crucial loss term is computed to ensure experts are utilized
  evenly, preventing training collapse. This loss is returned alongside the main model output.
- Configurable number of experts.

Key QK-Reorder-LN changes:
- The pre-attention RMSNorm in the Block is removed.
- RMSNorm is applied individually to the Query (Q) and Key (K) tensors within the Attention
  module, immediately after their projection and before RoPE.
"""

import math
from dataclasses import dataclass
import inspect
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


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

# Pre-existing functions (precompute_freqs_cis, apply_rotary_emb) remain unchanged.
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(0)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class Attention(nn.Module):
    def __init__(self, config: LunarisCodexConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.d_model // config.n_heads

        q_size = self.n_heads * self.head_dim
        kv_size = self.n_kv_heads * self.head_dim

        self.wqkv = nn.Linear(config.d_model, q_size + 2 * kv_size, bias=False)
        self.o_proj = nn.Linear(q_size, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

        # QK-Reorder-LN: Normalization layers for Query and Key
        self.q_norm = nn.RMSNorm(q_size, eps=1e-5)
        self.k_norm = nn.RMSNorm(kv_size, eps=1e-5)


    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, T, C = x.shape

        qkv = self.wqkv(x)
        q, k, v = torch.split(qkv, [self.n_heads * self.head_dim, self.n_kv_heads * self.head_dim, self.n_kv_heads * self.head_dim], dim=-1)

        # QK-Reorder-LN: Apply normalization to Q and K tensors
        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rotary_emb(q, k, freqs_cis)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat((past_k, k), dim=-2)
            v = torch.cat((past_v, v), dim=-2)

        present_kv = (k, v)

        if self.n_kv_heads < self.n_heads:
            n_repeats = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(n_repeats, dim=1)
            v = v.repeat_interleave(n_repeats, dim=1)

        is_causal = past_kv is None
        y = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.dropout(self.o_proj(y))

        return y, present_kv

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

# --- NEW: Mixture-of-Experts Layer ---
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
            final_output[token_mask] = expert_outputs

        # 5. Scale the output by the gate probability and reshape back.
        # This is a key step to make the routing decision differentiable. The gradient
        # can flow through `top_expert_probs` back to the gating network's weights.
        final_output = final_output * top_expert_probs.unsqueeze(-1)
        final_output = final_output.view(batch_size, seq_len, d_model)

        # Reshape expert_indices back to (batch_size, seq_len) for tracking
        expert_indices_reshaped = expert_indices.view(batch_size, seq_len)

        return final_output, aux_loss, expert_indices_reshaped


# --- MODIFIED: Block to support MoE and Gradient Checkpointing ---
class Block(nn.Module):
    """
    A single Transformer block, now with a choice between a standard FFN and an MoE layer.

    This block conditionally instantiates either a `FeedForward` module or a
    `MixtureOfExperts` module based on the model configuration. The forward pass
    is adjusted to handle the auxiliary loss that is returned by the MoE layer.
    """
    def __init__(self, config: LunarisCodexConfig):
        super().__init__()
        self.attention = Attention(config)
        self.ffn_norm = nn.RMSNorm(config.d_model, eps=1e-5)

        # Conditionally create either a standard FFN or a Mixture-of-Experts layer.
        if config.n_experts is not None and config.n_experts > 0:
            self.feed_forward = MixtureOfExperts(config)
            self.is_moe = True
            print(f"Block initialized with Mixture-of-Experts ({config.n_experts} experts).")
        else:
            self.feed_forward = FeedForward(config)
            self.is_moe = False
            print("Block initialized with standard FeedForward network.")

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass of the transformer block. Now uses activation checkpointing to save memory during training.
        """
        def _inner_forward(
            x_inner: torch.Tensor,
            freqs_cis_inner: torch.Tensor,
            past_kv_inner: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        ):
            # First residual connection: Attention
            # QK-Reorder-LN: Input to attention is x_inner directly, norm is inside attention
            attn_output, new_kv = self.attention(x_inner, freqs_cis_inner, past_kv_inner)
            h = x_inner + attn_output

            # Prepare for the second residual connection (FFN or MoE)
            aux_loss = None
            expert_indices = None
            ffn_input = self.ffn_norm(h)

            # Apply either the FFN or the MoE layer.
            if self.is_moe:
                # If it's an MoE layer, it returns the output, auxiliary loss, and expert indices.
                ffn_output, aux_loss, expert_indices = self.feed_forward(ffn_input)
            else:
                # A standard FFN just returns the output.
                ffn_output = self.feed_forward(ffn_input)

            # Second residual connection
            out = h + ffn_output

            # The block now propagates the auxiliary loss and expert indices upwards.
            return out, new_kv, aux_loss, expert_indices

        if self.training:
            # During training, use checkpointing to save memory.
            # `use_reentrant=False` is recommended for better performance and compatibility.
            return checkpoint(
                _inner_forward, x, freqs_cis, past_kv, use_reentrant=False
            )
        else:
            # During evaluation, run the forward pass directly.
            return _inner_forward(x, freqs_cis, past_kv)


# --- MODIFIED: Main LunarisCodex class to handle auxiliary loss ---
class LunarisCodex(nn.Module):
    """
    Complete LunarisCodex Language Model, now with optional MoE support.

    This class integrates the MoE blocks and handles the aggregation of the
    auxiliary load balancing loss from all MoE layers during the forward pass.
    The final loss returned during training is a combination of the main
    cross-entropy loss and this total auxiliary loss.
    """
    def __init__(self, config: LunarisCodexConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.d_model),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
            ln_f = nn.RMSNorm(config.d_model, eps=1e-5),
            drop = nn.Dropout(config.dropout),
        ))

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        freqs_cis = precompute_freqs_cis(
            self.config.d_model // self.config.n_heads,
            self.config.max_seq_len,
            self.config.rope_theta,
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

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

        if isinstance(module, (Attention, FeedForward, MixtureOfExperts)):
            for name, p in module.named_parameters():
                if name.endswith("o_proj.weight") or name.endswith("w2.weight"):
                    torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layers))

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, Optional[tuple], List[Tuple[torch.Tensor, torch.Tensor]], Optional[List[torch.Tensor]]]:
        """
        Forward pass of the model.

        When training with MoE layers, this method aggregates the auxiliary balancing
        loss from each `Block`. The final loss returned for backpropagation is the sum
        of the standard cross-entropy loss and this aggregated auxiliary loss.

        To enable detailed monitoring, the returned `loss` during training is a tuple:
        (total_loss, main_cross_entropy_loss, final_auxiliary_loss). This allows the
        training script to log each component separately.

        Args:
            idx: Input token indices.
            targets: Target token indices for loss calculation.
            past_key_values: KV cache for efficient generation.

        Returns:
            A tuple containing:
            - logits: The model's output logits.
            - loss: A tuple `(total, main, aux)` during training, or `None` during inference.
            - new_past_key_values: The updated KV cache.
            - expert_indices_list: List of expert indices from each MoE layer during training, or `None` during inference.
        """
        B, T = idx.shape
        start_pos = past_key_values[0][0].shape[-2] if past_key_values is not None else 0
        assert start_pos + T <= self.config.max_seq_len, \
            f"Sequence length {start_pos + T} exceeds model's max_seq_len {self.config.max_seq_len}"

        x = self.transformer.wte(idx)
        x = self.transformer.drop(x)
        freqs_cis = self.freqs_cis[start_pos : start_pos + T]

        new_past_key_values = []
        # Accumulate the auxiliary loss from all MoE layers.
        total_aux_loss = 0.0
        # Create list to collect expert indices from MoE layers
        expert_indices_list = []

        for i, block in enumerate(self.transformer.h):
            past_kv_for_block = past_key_values[i] if past_key_values is not None else None
            # The block returns an optional auxiliary loss and expert indices.
            x, new_kv, aux_loss, expert_indices = block(x, freqs_cis, past_kv_for_block)
            if aux_loss is not None:
                total_aux_loss += aux_loss
            if expert_indices is not None:
                expert_indices_list.append(expert_indices)
            new_past_key_values.append(new_kv)

        x = self.transformer.ln_f(x)

        loss = None # This will now hold the tuple of losses or None

        if targets is not None:
            logits = self.lm_head(x)

            # Calculate the main cross-entropy loss for language modeling.
            main_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

            # Average the auxiliary loss over the number of MoE layers to keep its
            # magnitude consistent, regardless of how many MoE layers are in the model.
            num_moe_layers = sum(1 for block in self.transformer.h if getattr(block, 'is_moe', False))

            final_aux_loss = total_aux_loss
            if num_moe_layers > 0:
                final_aux_loss /= num_moe_layers

            # The total loss used for backpropagation is the sum of the main loss
            # and the final, scaled auxiliary loss.
            total_loss = main_loss + final_aux_loss

            # Group all loss components into a tuple. This is highly useful for
            # the training script, as it allows for detailed logging and monitoring
            # of both the model's prediction accuracy (main_loss) and its expert
            # load balancing behavior (final_aux_loss).
            loss = (total_loss, main_loss, final_aux_loss)

            # Return expert indices during training
            return logits, loss, new_past_key_values, expert_indices_list

        else: # Inference mode
            logits = self.lm_head(x[:, [-1], :])
            # During inference, we don't calculate the loss or return expert indices.
            loss = None
            return logits, loss, new_past_key_values, None

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
        past_key_values = None
        for _ in range(max_new_tokens):
            current_len = past_key_values[0][0].shape[-2] if past_key_values else idx.shape[1]
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
