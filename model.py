import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from flash_attn import flash_attn_func

torch.backends.cuda.matmul.allow_tf32 = True

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
    ):
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

class LoRALinear(nn.Linear):
    def __init__(self, in_features, out_features, rank=32, alpha=1.0, bias=False):
        super().__init__(in_features, out_features, bias=bias)
        self.rank = rank
        self.alpha = alpha
        self.lora_A = nn.Parameter(torch.randn(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

    def forward(self, x):
        original = super().forward(x)
        lora_adaptation = (x @ self.lora_A) @ self.lora_B
        return original + self.alpha * lora_adaptation

class LunarisMind(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if self.config.vocab_size is None:
            raise ValueError("vocab_size must be provided!")
        self.vocab_size = self.config.vocab_size
        self.d_model = self.config.d_model
        self.n_layers = self.config.n_layers
        self.n_heads = self.config.n_heads
        self.head_dim = self.d_model // self.n_heads
        self.max_seq_len = self.config.max_seq_len
        self.dropout = self.config.dropout

        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.layers = nn.ModuleList([
            TransformerDecoderBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                dropout=self.dropout,
                layer_norm_epsilon=1e-5,
                ff_mult=3,
                activation=self.config.activation,
                lora_rank=self.config.lora_rank
            ) for _ in range(self.n_layers)
        ])
        self.final_layer_norm = nn.LayerNorm(self.d_model, eps=1e-5)
        self.lm_head = lambda x: F.linear(x, self.token_embedding.weight)

        self._init_alibi_slopes()
        self._init_weights_properly()

    def _init_alibi_slopes(self):
        slopes = torch.tensor([2 ** (-(8 / self.n_heads) * i) for i in range(self.n_heads)])
        self.register_buffer("alibi_slopes", slopes)

    def _init_weights_properly(self):
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.ones_(self.final_layer_norm.weight)
        nn.init.zeros_(self.final_layer_norm.bias)
        for layer in self.layers:
            nn.init.ones_(layer.ln_1.weight)
            nn.init.zeros_(layer.ln_1.bias)
            nn.init.ones_(layer.ln_2.weight)
            nn.init.zeros_(layer.ln_2.bias)
            nn.init.normal_(layer.attn.qkv_proj.weight, mean=0.0, std=0.02)
            scale = 1.0 / math.sqrt(2.0 * self.n_layers)
            nn.init.normal_(layer.attn.output_proj.weight, mean=0.0, std=0.02 * scale)
            nn.init.normal_(layer.ff.fc1.weight, mean=0.0, std=0.02)
            nn.init.normal_(layer.ff.fc2.weight, mean=0.0, std=0.02 * scale)

    def get_alibi_mask(self, seq_len, device=None):
        if device is None:
            device = self.alibi_slopes.device
        causal_mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device), diagonal=1)
        pos_indices = torch.arange(seq_len, device=device).unsqueeze(0) - torch.arange(seq_len, device=device).unsqueeze(1)
        pos_indices = pos_indices.to(torch.float32)
        alibi_slopes = self.alibi_slopes.to(device)
        alibi_bias = alibi_slopes.view(-1, 1, 1) * pos_indices.unsqueeze(0)
        return causal_mask, alibi_bias

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.size()
        device = input_ids.device
        x = self.token_embedding(input_ids)
        causal_mask, alibi_bias = self.get_alibi_mask(seq_len, device=device)
        if attention_mask is not None:
            # Combina máscara causal com máscara de padding
            causal_mask = causal_mask | (~attention_mask.bool().unsqueeze(1).expand(-1, seq_len, -1))
        for layer in self.layers:
            x = layer(x, causal_mask, alibi_bias)
        x = self.final_layer_norm(x)
        logits = self.lm_head(x)
        return logits

    def generate(self, input_ids, max_length=200, temperature=0.8, top_k=50, top_p=0.95, repetition_penalty=1.1, eos_token_id=None):
        device = input_ids.device
        batch_size = input_ids.size(0)
        generated = input_ids.clone()
        attention_mask = (input_ids != eos_token_id).long() if eos_token_id is not None else torch.ones_like(input_ids)
        
        for _ in range(max_length - input_ids.size(1)):
            logits = self.forward(generated, attention_mask)[:, -1, :] / temperature
            if repetition_penalty != 1.0:
                for b in range(batch_size):
                    for token_id in set(generated[b].tolist()):
                        if logits[b, token_id] > 0:
                            logits[b, token_id] /= repetition_penalty
                        else:
                            logits[b, token_id] *= repetition_penalty
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = -float('inf')
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = torch.zeros_like(logits, dtype=torch.bool).scatter_(
                    dim=-1, index=sorted_indices, src=sorted_indices_to_remove
                )
                logits[indices_to_remove] = -float('inf')
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)
            if eos_token_id is not None and (next_token == eos_token_id).any():
                break
        return generated

class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, layer_norm_epsilon=1e-5, ff_mult=3, activation="swiglu", lora_rank=32):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.ln_1 = nn.LayerNorm(d_model, eps=layer_norm_epsilon)
        self.ln_2 = nn.LayerNorm(d_model, eps=layer_norm_epsilon)
        self.attn = SelfAttention(d_model, n_heads, dropout, lora_rank=lora_rank)
        self.ff = FeedForward(d_model, ff_mult, dropout, activation, lora_rank=lora_rank)
        self.dropout = nn.Dropout(dropout)
        self.layer_scale1 = nn.Parameter(torch.ones(d_model) * 0.1)
        self.layer_scale2 = nn.Parameter(torch.ones(d_model) * 0.1)

    def forward(self, x, causal_mask, alibi_bias):
        residual = x
        x = self.ln_1(x)
        x = self.attn(x, causal_mask, alibi_bias)
        x = self.dropout(x)
        x = residual + self.layer_scale1.unsqueeze(0).unsqueeze(0) * x
        residual = x
        x = self.ln_2(x)
        x = self.ff(x)
        x = self.dropout(x)
        x = residual + self.layer_scale2.unsqueeze(0).unsqueeze(0) * x
        return x

class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, lora_rank=32):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert self.head_dim * n_heads == d_model, "d_model deve ser divisível por n_heads"
        self.qkv_proj = LoRALinear(d_model, 3 * d_model, rank=lora_rank, bias=False)
        self.output_proj = LoRALinear(d_model, d_model, rank=lora_rank, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.dropout_p = dropout

    def forward(self, x, causal_mask, alibi_bias):
        batch_size, seq_len, d_model = x.size()
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            attn_output = flash_attn_func(
                q, k, v,
                dropout_p=self.dropout_p if self.training else 0.0,
                causal=True,
                softmax_scale=1.0 / math.sqrt(self.head_dim)
            )

        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, d_model)
        output = self.output_proj(attn_output)
        output = self.resid_dropout(output)
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, ff_mult=3, dropout=0.1, activation="swiglu", lora_rank=32):
        super().__init__()
        self.d_model = d_model
        self.activation = activation
        if activation == "swiglu":
            self.d_ff = d_model * ff_mult * 2
            self.intermediate_dim = d_model * ff_mult
        else:
            self.d_ff = d_model * ff_mult
            self.intermediate_dim = self.d_ff
        self.fc1 = LoRALinear(d_model, self.d_ff, rank=lora_rank)
        self.fc2 = LoRALinear(self.intermediate_dim, d_model, rank=lora_rank)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if self.activation == "swiglu":
            x1, x2 = self.fc1(x).chunk(2, dim=-1)
            x = F.silu(x1) * x2
        elif self.activation == "gelu":
            x = F.gelu(self.fc1(x))
        else:
            raise ValueError(f"Ativação não suportada: {self.activation}")
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Carrega o tokenizador StarCoder
    tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoder")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    config = LunarisCodexConfig(vocab_size=tokenizer.vocab_size)  # ~49K
    model = LunarisMind(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total de parâmetros: {total_params:,}")
    
    for name, param in model.named_parameters():
        if 'lora_' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    trainable_params = count_parameters(model)
    print(f"Parâmetros treináveis (LoRA): {trainable_params:,}")
    
    if hasattr(torch, 'compile'):
        model = torch.compile(model)
        print("Modelo compilado com torch.compile")
