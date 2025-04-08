import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import numpy as np
from torch.optim import AdamW
from torch import amp
from safetensors.torch import save_file
import os

from model import LunarisMind, LunarisCodexConfig, count_parameters

# Configuração inicial
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True  # Otimiza pra H100

class CodeDataset:
    def __init__(self, memmap_file, num_sequences, max_length=1024):
        self.data = np.memmap(memmap_file, dtype=np.int32, mode='r', shape=(num_sequences, max_length))
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.data[idx], dtype=torch.long)
        attention_mask = (input_ids != tokenizer.pad_token_id).long()  # Máscara de padding
        return {"input_ids": input_ids, "attention_mask": attention_mask}

def compute_metrics(logits, targets, attention_mask):
    logits = logits[..., :-1, :].contiguous()
    targets = targets[..., 1:].contiguous()
    mask = attention_mask[..., 1:].contiguous()
    
    loss = nn.CrossEntropyLoss(reduction='none')(logits.view(-1, logits.size(-1)), targets.view(-1))
    loss = (loss * mask.view(-1)).sum() / mask.sum()
    perplexity = torch.exp(loss)
    
    preds = torch.argmax(logits, dim=-1)
    correct = ((preds == targets) & mask.bool()).float().sum()
    total = mask.sum()
    top1_acc = correct / total
    return loss, perplexity, top1_acc

def save_checkpoint(model, optimizer, epoch, path="checkpoints/lunaris_mind_epoch_{}.safetensors"):
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = path.format(epoch)
    
    state_dict = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": torch.tensor([epoch])
    }
    
    save_file(state_dict, checkpoint_path)
    print(f"Checkpoint salvo em {checkpoint_path}")

def train_model(model, dataloader, num_epochs=3, lr=5e-4, device="cuda", save_interval=1):
    model = model.to(device)
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01, fused=True)
    scaler = amp.GradScaler('cuda')

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_perplexity = 0
        total_top1_acc = 0
        num_batches = len(dataloader)

        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            if batch_idx == 0:
                print(f"Epoch {epoch+1}, Batch 0 shape: {input_ids.shape}, "
                      f"Min: {input_ids.min().item()}, Max: {input_ids.max().item()}")

            optimizer.zero_grad(set_to_none=True)
            with amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(input_ids, attention_mask=attention_mask)
                loss, perplexity, top1_acc = compute_metrics(logits, input_ids, attention_mask)

            if torch.isnan(loss):
                print(f"Loss virou NaN no batch {batch_idx}! Parando pra debug.")
                print(f"Logits min: {logits.min().item()}, max: {logits.max().item()}")
                break

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_perplexity += perplexity.item()
            total_top1_acc += top1_acc.item()

            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{num_batches}, "
                      f"Loss: {loss.item():.4f}, Perplexity: {perplexity.item():.2f}, Top-1 Acc: {top1_acc.item():.4f}")

        avg_loss = total_loss / num_batches
        avg_perplexity = total_perplexity / num_batches
        avg_top1_acc = total_top1_acc / num_batches
        print(f"Epoch {epoch+1}/{num_epochs} concluído, "
              f"Perda média: {avg_loss:.4f}, Perplexity média: {avg_perplexity:.2f}, "
              f"Acurácia Top-1 média: {avg_top1_acc:.4f}")

        if (epoch + 1) % save_interval == 0:
            save_checkpoint(model, optimizer, epoch + 1)

    return model

if __name__ == "__main__":
    # Carrega o tokenizador StarCoder
    tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoder")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Configura o dataset
    memmap_file = "the_stack_lunaris.memmap"
    max_length = 1024
    num_sequences = 1000000  # 1M exemplos do processamento

    dataset = CodeDataset(memmap_file, num_sequences, max_length=max_length)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)

    # Configura o modelo
    config = LunarisCodexConfig(
        vocab_size=tokenizer.vocab_size,  # ~49K do StarCoder
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
        lora_rank=32
    )

    model = LunarisMind(config)
    model.final_layer_norm = nn.LayerNorm(model.d_model, eps=1e-5)
    print(f"Total de parâmetros: {sum(p.numel() for p in model.parameters()):,}")

    for name, param in model.named_parameters():
        if 'lora_' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    trainable_params = count_parameters(model)
    print(f"Parâmetros treináveis (LoRA): {trainable_params:,}")

    if hasattr(torch, 'compile'):
        model = torch.compile(model, mode="reduce-overhead")
        print("Modelo compilado com torch.compile")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    trained_model = train_model(model, dataloader, num_epochs=3, lr=5e-4, device=device, save_interval=1)
    print("Treinamento concluído!")
