"""
Main Training Script for the LunarisCodex Language Model - Fused PyTorch/XLA Version

This script represents a fusion of best practices for training on Google Cloud TPUs.
It combines the robust, functionally correct components (like safe checkpointing and
correct metric aggregation) with the most modern PyTorch/XLA APIs (like the recommended
torch_xla.launch).

Key TPU Optimizations:
- **Modern Multi-core Launcher:** Uses `torch_xla.launch()` for efficient multi-process training.
- **Optimized Data Loading:** Implements `MpDeviceLoader` for asynchronous, parallel data transfer.
- **Explicit Mixed Precision:** Uses `torch.autocast` for clear control over TPU's native bfloat16 usage.
- **XLA Graph Compilation:** Leverages `torch.compile` with the XLA backend for optimal performance.
- **Robust Synchronization & I/O:** Utilizes `xm.optimizer_step`, `xm.mesh_reduce`, and `xm.save`
  for correct and safe distributed training, logging, and checkpointing.
"""

import os
import time
import math
import glob
from dataclasses import dataclass, field
from typing import Optional

import yaml
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# --- PyTorch/XLA Imports ---
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl

from model import LunarisCodex, LunarisCodexConfig

# --- Configuration Dataclass (Unchanged) ---
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
    compile_model: bool = True
    out_dir: str = "checkpoints_tpu"
    log_interval: int = 20
    save_interval: int = 1000
    wandb_project: Optional[str] = "lunaris-codex-tpu"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = f"tpu-run-{time.strftime('%Y-%m-%d-%H-%M')}"

    @property
    def sequence_length(self):
        return self.model.max_seq_len

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        model_config_dict = config_dict.pop("model", {})
        model_config = LunarisCodexConfig(**model_config_dict)
        config_dict['model'] = model_config
        float_fields = ['learning_rate', 'weight_decay', 'beta1', 'beta2', 'grad_clip']
        int_fields = ['warmup_steps', 'max_steps', 'batch_size', 'gradient_accumulation_steps', 'num_epochs', 'save_interval', 'log_interval']
        for key in float_fields:
            if key in config_dict:
                config_dict[key] = float(config_dict[key])
        for key in int_fields:
            if key in config_dict:
                config_dict[key] = int(config_dict[key])
        return cls(**config_dict)

# --- Sharded Memory-Mapped Dataset (Unchanged) ---
class ShardDataset(Dataset):
    def __init__(self, data_dir: str, sequence_length: int):
        super().__init__()
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.shards = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
        if not self.shards:
            raise ValueError(f"No .npy files found in directory: {data_dir}")
        total_tokens = sum(np.load(shard, mmap_mode='r').shape[0] for shard in self.shards)
        self.total_samples = total_tokens // self.sequence_length
        # Use master process for printing to avoid log spam
        if xm.is_master_ordinal():
            print(f"[DATA] Loaded {len(self.shards)} shards. Total tokens: {total_tokens/1e9:.2f}B.")
            print(f"[DATA] Creating {self.total_samples:,} non-overlapping samples of length {self.sequence_length}.")
        self.mmap_shards = [np.load(shard, mmap_mode='r') for shard in self.shards]
        self.shard_lengths = [len(shard) for shard in self.mmap_shards]
        self.cumulative_lengths = np.cumsum(self.shard_lengths)

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        token_start_pos = idx * self.sequence_length
        shard_idx = np.searchsorted(self.cumulative_lengths, token_start_pos, side='right')
        local_start_idx = token_start_pos if shard_idx == 0 else token_start_pos - self.cumulative_lengths[shard_idx - 1]
        seq_len_with_target = self.sequence_length + 1
        if local_start_idx + seq_len_with_target > self.shard_lengths[shard_idx]:
            remaining_len = self.shard_lengths[shard_idx] - local_start_idx
            seq_part1 = self.mmap_shards[shard_idx][local_start_idx : local_start_idx + remaining_len]
            if shard_idx + 1 < len(self.mmap_shards):
                needed_from_next = seq_len_with_target - remaining_len
                seq_part2 = self.mmap_shards[shard_idx + 1][:needed_from_next]
                seq = np.concatenate((seq_part1, seq_part2))
            else:
                seq = seq_part1
        else:
            seq = self.mmap_shards[shard_idx][local_start_idx : local_start_idx + seq_len_with_target]
        if len(seq) < seq_len_with_target:
            pad_len = seq_len_with_target - len(seq)
            seq = np.pad(seq, (0, pad_len), 'constant', constant_values=-1)
        seq_tensor = torch.from_numpy(seq.astype(np.int64))
        x, y = seq_tensor[:-1], seq_tensor[1:]
        return x, y

# --- Learning Rate Scheduler (Unchanged) ---
def get_lr(step, config: TrainConfig):
    if step < config.warmup_steps:
        return config.learning_rate * step / config.warmup_steps
    if step >= config.max_steps:
        return config.learning_rate * 0.01
    decay_ratio = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return (config.learning_rate * 0.01) + coeff * (config.learning_rate * 0.99)

# --- Checkpoint Key Unwrapping (Unchanged) ---
def unwrap_model_keys(state_dict):
    unwrapped = {}
    for k, v in state_dict.items():
        new_k = k
        if new_k.startswith('_orig_mod.'):
            new_k = new_k[len('_orig_mod.'):]
        unwrapped[new_k] = v
    return unwrapped

# --- Main Training Function (Called by torch_xla.launch) ---
def main_loop(config: TrainConfig):
    # --- XLA Setup ---
    is_master_process = xm.is_master_ordinal()
    device = xm.xla_device()
    world_size = xm.xrt_world_size()

    torch.manual_seed(1337 + xm.get_ordinal())
    dtype = torch.bfloat16
    ctx = torch.autocast(device_type='xla', dtype=dtype)

    if is_master_process:
        os.makedirs(config.out_dir, exist_ok=True)
        print("-" * 50)
        print(" " * 10 + "LUNARIS CODEX TRAINING ON CLOUD TPU")
        print("-" * 50)
        print(f"Model: {config.model}")
        print(f"Data: {config.data_dir}")
        print(f"Total Batch size (all cores): {config.batch_size * world_size}")
        print("-" * 50)

    if is_master_process and config.wandb_project:
        import wandb
        wandb.init(project=config.wandb_project, entity=config.wandb_entity, name=config.wandb_run_name, config=config.__dict__)

    # --- Data Loading for XLA ---
    train_dataset = ShardDataset(data_dir=config.data_dir, sequence_length=config.sequence_length)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=xm.get_ordinal(),
        shuffle=True
    )
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, sampler=train_sampler, num_workers=4)
    train_loader = pl.MpDeviceLoader(train_loader, device)

    # --- Model & Optimizer Initialization ---
    model = LunarisCodex(config.model).to(device)
    if config.compile_model:
        if is_master_process: print("[MODEL] Compiling model for XLA...")
        model = torch.compile(model)

    optimizer = model.configure_optimizers(
        weight_decay=config.weight_decay,
        learning_rate=config.learning_rate,
        betas=(config.beta1, config.beta2),
        device_type='xla'
    )

    # --- Checkpoint Resuming ---
    current_step = 0
    checkpoint_path = os.path.join(config.out_dir, "latest_checkpoint.pt")
    if os.path.exists(checkpoint_path):
        if is_master_process: print(f"[SETUP] Resuming from checkpoint: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location='cpu') # Load to CPU first
        unwrapped_state_dict = unwrap_model_keys(state['model'])
        model.load_state_dict(unwrapped_state_dict)
        optimizer.load_state_dict(state['optimizer'])
        current_step = state['step']
        if is_master_process: print(f"[SETUP] Resumed successfully. Starting from step {current_step}")
        # Barrier to ensure all processes have loaded before continuing
        xm.rendezvous('checkpoint_loaded')

    # --- Training Loop ---
    if is_master_process:
        pbar = tqdm(total=config.max_steps, desc="Training Steps", initial=current_step, ncols=120)

    data_iter = iter(train_loader)
    optimizer.zero_grad(set_to_none=True)

    while current_step < config.max_steps:
        lr = get_lr(current_step, config)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        accumulated_loss = 0.0
        for micro_step in range(config.gradient_accumulation_steps):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                x, y = next(data_iter)

            with ctx:
                _, loss, _ = model(x, targets=y, past_key_values=None)
                loss = loss / config.gradient_accumulation_steps
            
            accumulated_loss += loss.item()
            loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        xm.optimizer_step(optimizer, barrier=True)
        optimizer.zero_grad(set_to_none=True)
        current_step += 1

        # --- Logging and Checkpointing (Master Process Only) ---
        if is_master_process:
            pbar.update(1)
            if current_step % config.log_interval == 0:
                # Use mesh_reduce for accurate global loss
                log_loss = xm.mesh_reduce('loss_reduce', accumulated_loss, sum) / world_size
                try:
                    perplexity = math.exp(log_loss)
                except (OverflowError, ValueError):
                    perplexity = float('inf')

                postfix_data = {"loss": f"{log_loss:.3f}", "ppl": f"{perplexity:.2f}" if perplexity != float('inf') else "inf"}
                pbar.set_postfix(postfix_data)

                if config.wandb_project:
                    wandb.log({"step": current_step, "loss": log_loss, "perplexity": perplexity, "lr": lr, "grad_norm": grad_norm.item()})

            if current_step > 0 and current_step % config.save_interval == 0:
                # Use xm.save for safe checkpointing
                checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'config': config.__dict__, 'step': current_step}
                save_path = os.path.join(config.out_dir, f"ckpt_{current_step}.pt")
                xm.save(checkpoint, save_path)
                latest_path = os.path.join(config.out_dir, "latest_checkpoint.pt")
                xm.save(checkpoint, latest_path)
                print(f"\n[CHECKPOINT] Saved checkpoint to {save_path}")

    if is_master_process:
        print("\nMax steps reached. Finishing training.")
        pbar.close()
        if config.wandb_project:
            wandb.finish()

# --- Script Entry Point ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a LunarisCodex model on Cloud TPUs.")
    parser.add_argument("config", type=str, help="Path to the config.yaml file.")
    args = parser.parse_args()
    config = TrainConfig.from_yaml(args.config)

    # Use the modern torch_xla.launch to spawn the main_loop on each TPU core
    torch_xla.launch(main_loop, args=(config,))
