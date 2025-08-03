"""
Main Training Script for the LunarisCodex Language Model (Optimized MoE Version).

Key engineering features:
- Works with Switch-style MoE (k=1) with capacity-aware routing and drops.
- bfloat16 autocast on H100/GH200; TF32 enabled; fused AdamW with router-specific LR.
- Gradient accumulation, DDP-friendly (optional), and gradient clipping.
- torch.compile integration with safe fallback.
- Robust checkpointing/resume; memory-efficient dataloading.
- W&B logging: total/main/aux losses, perplexity, LR, grad-norm, expert utilization and drop rates.

Usage:
    python train_moe.py path/to/config.yaml
"""

import os
import time
import math
import glob
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, List
from contextlib import nullcontext

import yaml
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from tqdm import tqdm

from model_moe import LunarisCodex, LunarisCodexConfig, compile_model_if_available

# ---------------------------
# Config
# ---------------------------

@dataclass
class TrainConfig:
    model: LunarisCodexConfig = field(default_factory=LunarisCodexConfig)
    data_dir: str = "data/"
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    warmup_steps: int = 2000
    max_steps: int = 600000
    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    num_epochs: int = 1
    grad_clip: float = 1.0
    device: str = "cuda"
    compile_model: bool = True
    out_dir: str = "checkpoints"
    log_interval: int = 20
    save_interval: int = 1000
    save_latest_always: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    wandb_project: Optional[str] = "lunaris-codex-moe"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None

    @property
    def sequence_length(self):
        return self.model.max_seq_len

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        model_config_dict = config_dict.pop("model", {})
        model_config = LunarisCodexConfig(**model_config_dict)

        # Backward-compat defaults for MoE
        if model_config.n_experts and model_config.aux_loss_weight is None:
            model_config.aux_loss_weight = 1e-2
        if getattr(model_config, "capacity_factor", None) is None:
            model_config.capacity_factor = 1.25
        if getattr(model_config, "router_z_loss_weight", None) is None:
            model_config.router_z_loss_weight = 1e-3

        config_dict['model'] = model_config
        # normalize numeric fields
        float_fields = ['learning_rate', 'weight_decay', 'beta1', 'beta2', 'grad_clip']
        int_fields = ['warmup_steps', 'max_steps', 'batch_size', 'gradient_accumulation_steps',
                      'num_epochs', 'save_interval', 'log_interval', 'num_workers', 'prefetch_factor']
        for key in float_fields:
            if key in config_dict:
                config_dict[key] = float(config_dict[key])
        for key in int_fields:
            if key in config_dict:
                config_dict[key] = int(config_dict[key])
        return cls(**config_dict)

# ---------------------------
# Dataset
# ---------------------------

class ShardDataset(Dataset):
    """
    Memory-efficient dataset over .npy token shards. Produces (x, y) with length seq_len.
    Pads last sample with -1 if needed (ignored in CE loss).
    """
    def __init__(self, data_dir: str, sequence_length: int):
        super().__init__()
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.shards = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
        if not self.shards:
            raise ValueError(f"No .npy files found in directory: {data_dir}")
        # Lazy memory map shards
        self.mmap_shards = [np.load(shard, mmap_mode='r') for shard in self.shards]
        self.shard_lengths = [len(shard) for shard in self.mmap_shards]
        total_tokens = sum(self.shard_lengths)
        self.total_samples = total_tokens // self.sequence_length
        self.cumulative_lengths = np.cumsum(self.shard_lengths)
        print(f"[DATA] Loaded {len(self.shards)} shards. Total tokens: {total_tokens/1e9:.2f}B.")
        print(f"[DATA] Creating {self.total_samples:,} non-overlapping samples of length {self.sequence_length}.")

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        L = self.sequence_length
        token_start_pos = idx * L
        shard_idx = np.searchsorted(self.cumulative_lengths, token_start_pos, side='right')
        local_start_idx = token_start_pos if shard_idx == 0 else token_start_pos - self.cumulative_lengths[shard_idx - 1]
        seq_len_with_target = L + 1

        if local_start_idx + seq_len_with_target <= self.shard_lengths[shard_idx]:
            seq = self.mmap_shards[shard_idx][local_start_idx: local_start_idx + seq_len_with_target]
        else:
            remaining_len = self.shard_lengths[shard_idx] - local_start_idx
            seq_part1 = self.mmap_shards[shard_idx][local_start_idx: local_start_idx + remaining_len]
            need = seq_len_with_target - remaining_len
            if shard_idx + 1 < len(self.mmap_shards):
                seq_part2 = self.mmap_shards[shard_idx + 1][:need]
                seq = np.concatenate((seq_part1, seq_part2))
            else:
                seq = seq_part1

        if len(seq) < seq_len_with_target:
            pad_len = seq_len_with_target - len(seq)
            seq = np.pad(seq, (0, pad_len), 'constant', constant_values=-1)

        seq_tensor = torch.from_numpy(seq.astype(np.int64))
        x, y = seq_tensor[:-1], seq_tensor[1:]
        return x, y

# ---------------------------
# DDP / utils
# ---------------------------

def setup_ddp():
    is_ddp = 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1
    if is_ddp:
        init_process_group("nccl")
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        torch.cuda.set_device(local_rank)
        print(f"[DDP] Setup complete: rank {rank}, world_size {world_size}, local_rank {local_rank}")
        return True, rank, world_size, local_rank
    return False, 0, 1, 0

def get_lr(step, config: TrainConfig):
    if step < config.warmup_steps:
        return config.learning_rate * step / max(1, config.warmup_steps)
    if step >= config.max_steps:
        return config.learning_rate * 0.01
    decay_ratio = (step - config.warmup_steps) / max(1, (config.max_steps - config.warmup_steps))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return (config.learning_rate * 0.01) + coeff * (config.learning_rate * 0.99)

def unwrap_model_keys(state_dict):
    unwrapped = {}
    prefixes_to_remove = ['_orig_mod.module.', 'module.', '_orig_mod.']
    for k, v in state_dict.items():
        new_k = k
        for prefix in prefixes_to_remove:
            if new_k.startswith(prefix):
                new_k = new_k[len(prefix):]
                break
        unwrapped[new_k] = v
    return unwrapped

# ---------------------------
# Training
# ---------------------------

def train(config_path: str):
    config = TrainConfig.from_yaml(config_path)
    is_ddp, rank, world_size, local_rank = setup_ddp()
    is_master = (rank == 0)

    # Seeds and perf knobs
    torch.manual_seed(1337 + rank)
    np.random.seed(1337 + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device_type = 'cuda' if 'cuda' in config.device else 'cpu'
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=amp_dtype) if device_type == 'cuda' else nullcontext()

    # Logging header
    if is_master:
        os.makedirs(config.out_dir, exist_ok=True)
        if config.wandb_run_name is None:
            config.wandb_run_name = f"run-moe-{time.strftime('%Y-%m-%d-%H-%M')}"
        print("-" * 60)
        print(" LUNARIS CODEX - Optimized Switch-MoE Training")
        print("-" * 60)
        print(f"Model: {config.model}")
        if config.model.n_experts:
            print(f"MoE: experts={config.model.n_experts}, cap_factor={config.model.capacity_factor}, "
                  f"aux={config.model.aux_loss_weight}, z={config.model.router_z_loss_weight}")
        print(f"Data: {config.data_dir}, SeqLen={config.sequence_length}, Device={config.device}, bf16={use_bf16}")
        print(f"Batch={config.batch_size}, Accum={config.gradient_accumulation_steps}, LR={config.learning_rate}")
        print("-" * 60)

    # W&B
    if is_master and config.wandb_project:
        import wandb
        wandb.init(project=config.wandb_project, entity=config.wandb_entity,
                   name=config.wandb_run_name, config=asdict(config))

    # Data
    train_dataset = ShardDataset(data_dir=config.data_dir, sequence_length=config.sequence_length)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True) if is_ddp else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=(config.num_workers > 0 and config.persistent_workers),
        prefetch_factor=(config.prefetch_factor if config.num_workers > 0 else None),
        drop_last=True,
    )

    # Model
    model = LunarisCodex(config.model).to(config.device, dtype=torch.bfloat16 if use_bf16 else torch.float32)

    # Optional compile
    if config.compile_model and device_type == 'cuda':
        if is_master: print("[MODEL] Compiling with torch.compile ...")
        try:
            model = compile_model_if_available(model)
        except Exception as e:
            if is_master: print(f"[WARN] torch.compile failed, continuing without compile: {e}")

    # DDP
    if is_ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    raw_model = model.module if is_ddp else model

    # Optimizer
    optimizer = raw_model.configure_optimizers(
        weight_decay=config.weight_decay,
        learning_rate=config.learning_rate,
        betas=(config.beta1, config.beta2),
        device_type=device_type,
    )

    # Resume
    current_step, current_epoch = 0, 0
    latest_path = os.path.join(config.out_dir, "latest_checkpoint.pt")
    if os.path.exists(latest_path):
        if is_master: print(f"[RESUME] Loading checkpoint: {latest_path}")
        state = torch.load(latest_path, map_location=config.device)
        raw_model.load_state_dict(unwrap_model_keys(state['model']), strict=True)
        optimizer.load_state_dict(state['optimizer'])
        current_step = int(state.get('step', 0))
        current_epoch = int(state.get('epoch', 0))

    optimizer.zero_grad(set_to_none=True)
    if is_ddp:
        assert train_sampler is not None
        train_sampler.set_epoch(current_epoch)

    if is_master:
        print(f"\n[TRAIN] Starting at step {current_step} up to {config.max_steps} ...")
        pbar = tqdm(total=config.max_steps, desc="Steps", initial=current_step, ncols=120)

    # Training loop
    data_iter = iter(train_loader)
    while current_step < config.max_steps:
        current_step += 1
        # LR schedule
        lr = get_lr(current_step, config)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # Accumulators for logging
        accum_total = 0.0
        accum_main = 0.0
        accum_aux = 0.0
        # For expert routing logs (first MoE layer) and drop rates
        first_layer_expert_indices = None
        first_layer_keep_mask = None

        for micro in range(config.gradient_accumulation_steps):
            is_last_micro = (micro == config.gradient_accumulation_steps - 1)
            ddp_context = model.no_sync() if is_ddp and not is_last_micro else nullcontext()

            with ddp_context:
                try:
                    x, y = next(data_iter)
                except StopIteration:
                    current_epoch += 1
                    if is_ddp:
                        train_sampler.set_epoch(current_epoch)
                    data_iter = iter(train_loader)
                    x, y = next(data_iter)

                x = x.to(config.device, non_blocking=True)
                y = y.to(config.device, non_blocking=True)

                with autocast_ctx:
                    logits, loss_tuple, _, aux_list = model(x, targets=y, past_key_values=None)
                    # loss_tuple = (total, main, aux)
                    total_loss, main_loss, aux_loss = loss_tuple
                    # Scale for gradient accumulation
                    total_loss = total_loss / config.gradient_accumulation_steps

                # Accumulate for logs (use .item() after scaling)
                accum_total += float(total_loss.item())
                if main_loss is not None:
                    accum_main += float(main_loss.item()) / config.gradient_accumulation_steps
                if aux_loss is not None:
                    accum_aux += float(aux_loss.item()) / config.gradient_accumulation_steps

                # Capture first MoE layer routing and keep mask when available
                if aux_list is not None and isinstance(aux_list, list) and len(aux_list) == 2:
                    indices_list, keep_masks_list = aux_list
                    if indices_list and first_layer_expert_indices is None:
                        first_layer_expert_indices = indices_list[0].detach()
                    if keep_masks_list and first_layer_keep_mask is None:
                        first_layer_keep_mask = keep_masks_list[0].detach()

                total_loss.backward()

        # Clip and step
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # Logging
        if is_master:
            pbar.update(1)
            if current_step % config.log_interval == 0:
                ppl = math.exp(accum_main) if accum_main < 20 else float('inf')
                pbar.set_postfix({
                    "loss": f"{accum_total:.3f}",
                    "main": f"{accum_main:.3f}",
                    "aux": f"{accum_aux:.4f}",
                    "ppl": f"{ppl:.2f}",
                    "lr": f"{lr:.2e}",
                    "gnorm": f"{float(grad_norm):.2f}",
                })

                if config.wandb_project:
                    import wandb
                    log_data = {
                        "step": current_step,
                        "epoch": current_epoch,
                        "loss/total": accum_total,
                        "loss/main": accum_main,
                        "loss/aux": accum_aux,
                        "perplexity": ppl,
                        "lr": lr,
                        "grad_norm": float(grad_norm),
                    }

                    # Expert utilization & drop rate from the first MoE layer
                    if first_layer_expert_indices is not None:
                        num_experts = raw_model.config.n_experts or 0
                        if num_experts > 0:
                            counts = torch.bincount(first_layer_expert_indices.view(-1), minlength=num_experts)
                            util = counts.float() / counts.sum().clamp_min(1)
                            log_data.update({f"experts/util_layer0/e{i}": util[i].item() for i in range(num_experts)})

                    if first_layer_keep_mask is not None:
                        keep_fraction = first_layer_keep_mask.float().mean().item()
                        drop_fraction = 1.0 - keep_fraction
                        log_data["experts/drop_rate_layer0"] = drop_fraction

                    wandb.log(log_data)

        # Checkpointing
        if is_master and current_step % config.save_interval == 0:
            ckpt = {
                'model': (raw_model.state_dict()),
                'optimizer': optimizer.state_dict(),
                'config': asdict(config),
                'step': current_step,
                'epoch': current_epoch,
            }
            save_path = os.path.join(config.out_dir, f"ckpt_{current_step}.pt")
            torch.save(ckpt, save_path)
            if config.save_latest_always:
                latest_path = os.path.join(config.out_dir, "latest_checkpoint.pt")
                torch.save(ckpt, latest_path)
            print(f"\n[CKPT] Saved checkpoint: {save_path}")

    # Finalize
    if is_master:
        print("\n[TRAIN] Max steps reached. Finishing.")
        pbar.close()
        if config.wandb_project:
            import wandb
            wandb.finish()
    if is_ddp:
        destroy_process_group()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Train a LunarisCodex-MoE model (optimized).")
    parser.add_argument("config", type=str, help="Path to the MoE config.yaml file.")
    args = parser.parse_args()
    train(args.config)
