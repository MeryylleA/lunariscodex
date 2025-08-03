"""
Full definition of a LunarisCodex Language Model, optimized for single-GPU (H100/GH200) training with Switch-style MoE.

Key engineering features:
- Switch MoE (k=1) with capacity factor and token dropping for stability and OOM avoidance.
- Efficient dispatch/combination via contiguous permutation, minimizing scatter overhead.
- Router losses: load-balancing + z-loss; no probability scaling of expert outputs.
- QK-Reorder-LN in attention (RMSNorm on Q and K only), Flash Attention via SDP.
- bfloat16-first design: compute in bf16 where safe, upcast router logits to fp32.
- Gradient checkpointing; torch.compile-friendly code paths.
- Optimizer with fused AdamW and router-specific learning rate.

Note: This file is self-contained and ready for training. Use bf16 on H100/GH200.
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
    # MoE configuration
    n_experts: Optional[int] = 8
    n_experts_per_token: int = 1  # Switch Transformer: 1
    aux_loss_weight: float = 1e-2
    capacity_factor: float = 1.25
    router_z_loss_weight: float = 1e-3


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end, dtype=torch.float32)  # device set when registered as buffer
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64 in two real channels
    return freqs_cis


def apply_rotary_emb(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    # xq, xk: [B, H, T, D]
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(0)  # [1,1,T,D/2]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.to(dtype=xq.dtype), xk_out.to(dtype=xk.dtype)


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

        # QK-Reorder-LN: Norm only Q and K
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
        q, k, v = torch.split(
            qkv,
            [self.n_heads * self.head_dim, self.n_kv_heads * self.head_dim, self.n_kv_heads * self.head_dim],
            dim=-1,
        )

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, T, D]
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

        # Ensure contiguous bf16 for Flash SDP
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        is_causal = past_kv is None
        y = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.dropout(self.o_proj(y))
        return y, present_kv


class FeedForward(nn.Module):
    def __init__(self, config: LunarisCodexConfig):
        super().__init__()
        hidden_dim = int(config.ffn_hidden_multiplier * config.d_model)
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)

        self.w13 = nn.Linear(config.d_model, 2 * hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.d_model, bias=False)
        # Keep dropout configurable; default 0.0 for small-batch single GPU
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor):
        gate, up = self.w13(x).chunk(2, dim=-1)
        swiglu = F.silu(gate) * up
        return self.dropout(self.w2(swiglu))


class MixtureOfExperts(nn.Module):
    """
    Switch-style MoE (k=1) with capacity-aware routing and drops.
    - Routing by argmax of router logits (fp32).
    - No probability scaling of expert outputs.
    - Load-balancing loss + router z-loss for stability.
    - Efficient contiguous dispatch/combination using sort + index_copy.
    """
    def __init__(self, config: LunarisCodexConfig):
        super().__init__()
        assert config.n_experts is not None and config.n_experts > 0
        assert config.n_experts_per_token == 1, "Switch MoE in this implementation requires k=1"
        self.n_experts = config.n_experts
        self.aux_loss_weight = config.aux_loss_weight
        self.capacity_factor = config.capacity_factor
        self.z_loss_weight = config.router_z_loss_weight

        self.gate = nn.Linear(config.d_model, self.n_experts, bias=False)
        self.experts = nn.ModuleList([FeedForward(config) for _ in range(self.n_experts)])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        x_flat = x.view(-1, D)  # [N, D]
        N = x_flat.size(0)

        # Router logits in fp32 for stability
        router_logits = self.gate(x_flat.to(torch.float32))  # [N, E]
        router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)

        # Assign by argmax logits (top-1)
        top_expert_indices = torch.argmax(router_logits, dim=-1)  # [N]

        # Load-balancing loss (Switch)
        prob_mass = router_probs.mean(dim=0)  # [E]
        tokens_one_hot = F.one_hot(top_expert_indices, num_classes=self.n_experts).to(torch.float32)  # [N, E]
        fraction_tokens = tokens_one_hot.mean(dim=0)  # [E]
        balance_loss = (prob_mass * fraction_tokens).sum() * self.aux_loss_weight * self.n_experts

        # Router z-loss
        z = torch.logsumexp(router_logits, dim=-1)  # [N]
        z_loss = (z.pow(2).mean()) * self.z_loss_weight

        aux_loss = balance_loss + z_loss

        # Sort tokens by expert to form contiguous segments
        sorted_assign, sort_idx = torch.sort(top_expert_indices)
        x_perm = x_flat.index_select(0, sort_idx)
        counts = torch.bincount(sorted_assign, minlength=self.n_experts)  # [E]
        # Capacity per expert
        C = int(math.ceil((N / max(1, self.n_experts)) * self.capacity_factor))
        keep_counts = torch.clamp(counts, max=C)

        # Offsets for full and kept segments
        offsets = torch.cumsum(F.pad(counts, (1, 0)), dim=0)  # [E+1]
        keep_offsets = torch.cumsum(F.pad(keep_counts, (1, 0)), dim=0)  # [E+1]

        output_flat = torch.zeros_like(x_flat)
        keep_mask = torch.zeros(N, dtype=torch.bool, device=x.device)

        # Dispatch/compute/combine for each expert
        # Loops over experts are acceptable; E is small (e.g., 8-64)
        for i in range(self.n_experts):
            start = int(offsets[i].item())
            kept = int(keep_counts[i].item())
            if kept == 0:
                continue
            seg = x_perm[start:start + kept]
            y = self.experts[i](seg)
            idx_slice = sort_idx[start:start + kept]
            output_flat.index_copy_(0, idx_slice, y)
            keep_mask.index_copy_(
                0, idx_slice, torch.ones(kept, dtype=torch.bool, device=keep_mask.device)
            )

        # No probability scaling of outputs (Switch-style); dropped tokens contribute zero here.
        output = output_flat.view(B, T, D)
        expert_indices_reshaped = top_expert_indices.view(B, T)
        keep_mask = keep_mask.view(B, T)
        return output, aux_loss, expert_indices_reshaped, keep_mask


class Block(nn.Module):
    def __init__(self, config: LunarisCodexConfig):
        super().__init__()
        self.attention = Attention(config)
        self.ffn_norm = nn.RMSNorm(config.d_model, eps=1e-5)

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
    ) -> Tuple[
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        def _inner_forward(
            x_inner: torch.Tensor,
            freqs_cis_inner: torch.Tensor,
            past_kv_inner: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        ):
            attn_output, new_kv = self.attention(x_inner, freqs_cis_inner, past_kv_inner)
            h = x_inner + attn_output

            aux_loss = None
            expert_indices = None
            keep_mask = None
            ffn_input = self.ffn_norm(h)

            if self.is_moe:
                ffn_output, aux_loss, expert_indices, keep_mask = self.feed_forward(ffn_input)
            else:
                ffn_output = self.feed_forward(ffn_input)

            out = h + ffn_output  # dropped tokens only get residual h
            return out, new_kv, aux_loss, expert_indices, keep_mask

        if self.training:
            return checkpoint(_inner_forward, x, freqs_cis, past_kv, use_reentrant=False)
        else:
            return _inner_forward(x, freqs_cis, past_kv)


class LunarisCodex(nn.Module):
    def __init__(self, config: LunarisCodexConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.d_model),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
            ln_f=nn.RMSNorm(config.d_model, eps=1e-5),
            drop=nn.Dropout(config.dropout),
        ))

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # weight tying
        self.transformer.wte.weight = self.lm_head.weight

        freqs_cis = precompute_freqs_cis(
            self.config.d_model // self.config.n_heads,
            self.config.max_seq_len,
            self.config.rope_theta,
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        self.apply(self._init_weights)

        print(f"Number of parameters: {self.get_num_params()/1e6:.2f}M")
        if config.n_experts is not None and config.n_experts > 0:
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

        # Scale selected output weights for deep nets
        if isinstance(module, (Attention, FeedForward, MixtureOfExperts)):
            for name, p in module.named_parameters():
                if name.endswith("o_proj.weight") or name.endswith("w2.weight"):
                    torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layers))

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[
        torch.Tensor,
        Optional[tuple],
        List[Tuple[torch.Tensor, torch.Tensor]],
        Optional[List[torch.Tensor]],
    ]:
        B, T = idx.shape
        start_pos = past_key_values[0][0].shape[-2] if past_key_values is not None else 0
        assert start_pos + T <= self.config.max_seq_len, \
            f"Sequence length {start_pos + T} exceeds model's max_seq_len {self.config.max_seq_len}"

        x = self.transformer.wte(idx)
        x = self.transformer.drop(x)
        freqs_cis = self.freqs_cis[start_pos: start_pos + T].to(dtype=torch.float32, device=x.device)

        new_past_key_values: List[Tuple[torch.Tensor, torch.Tensor]] = []
        total_aux_loss = x.new_zeros(())
        expert_indices_list = []
        keep_masks_list = []

        for i, block in enumerate(self.transformer.h):
            past_kv_for_block = past_key_values[i] if past_key_values is not None else None
            x, new_kv, aux_loss, expert_indices, keep_mask = block(x, freqs_cis, past_kv_for_block)

            if aux_loss is not None:
                total_aux_loss = total_aux_loss + aux_loss.to(x.dtype)

            if expert_indices is not None:
                expert_indices_list.append(expert_indices)
            if keep_mask is not None:
                keep_masks_list.append(keep_mask)

            new_past_key_values.append(new_kv)

        x = self.transformer.ln_f(x)

        loss = None
        logits = self.lm_head(x) if targets is not None else self.lm_head(x[:, [-1], :])

        if targets is not None:
            main_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

            num_moe_layers = sum(1 for block in self.transformer.h if getattr(block, 'is_moe', False))
            final_aux_loss = total_aux_loss
            if num_moe_layers > 0:
                final_aux_loss = final_aux_loss / num_moe_layers

            total_loss = main_loss + final_aux_loss.to(main_loss.dtype)
            loss = (total_loss, main_loss, final_aux_loss)

            # Return per-layer expert indices and keep masks for logging
            return logits, loss, new_past_key_values, [expert_indices_list, keep_masks_list]
        else:
            return logits, None, new_past_key_values, None

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # Split parameters for weight decay, and separate router with smaller LR
        router_params, decay_params, nodecay_params = [], [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if 'feed_forward.gate' in n:
                router_params.append(p)
            elif p.dim() >= 2:
                decay_params.append(p)
            else:
                nodecay_params.append(p)

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay, 'lr': learning_rate},
            {'params': nodecay_params, 'weight_decay': 0.0, 'lr': learning_rate},
            {'params': router_params, 'weight_decay': 0.0, 'lr': learning_rate * 0.5},
        ]

        print(f"num decayed parameter tensors: {len(decay_params)}, with {sum(p.numel() for p in decay_params):,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {sum(p.numel() for p in nodecay_params):,} parameters")
        print(f"num router parameter tensors: {len(router_params)}, with {sum(p.numel() for p in router_params):,} parameters")

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        optimizer = torch.optim.AdamW(optim_groups, betas=betas, fused=use_fused)
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
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        self.train()
        return idx


# Optional helper to wrap with torch.compile in training script:
def compile_model_if_available(model: nn.Module):
    try:
        model = torch.compile(model, mode="max-autotune")
        print("Model compiled with torch.compile (max-autotune).")
    except Exception as e:
        print(f"torch.compile not enabled or failed: {e}")
    return model
