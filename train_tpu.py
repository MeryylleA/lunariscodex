# train_tpu.py
# A dedicated PyTorch/XLA version of the training script, optimized for Google Cloud TPUs.
# This script is a port of the original CUDA-based train.py.

import os
import time
import math
import glob
from dataclasses import dataclass, field
from typing import Optional

import yaml
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from tqdm import tqdm

# --- XLA Imports ---
# XLA: Import core XLA utilities for device management, multiprocessing, and data loading.
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl

# Assuming model.py contains the LunarisCodex and LunarisCodexConfig classes
from model import LunarisCodex, LunarisCodexConfig

# --- Configuration Dataclass (Adapted for TPU) ---
@dataclass
class TrainConfig:
    # Model configuration
    model: LunarisCodexConfig = field(default_factory=LunarisCodexConfig)

    # Data configuration
    data_dir: str = "data/"
    sequence_length: int = 1024

    # Optimizer configuration
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95

    # Scheduler configuration
    warmup_steps: int = 2000
    max_steps: int = 600000

    # Training configuration
    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    num_epochs: int = 1 # Set to a large number for step-based training
    grad_clip: float = 1.0
    # XLA: 'device' is now managed by XLA, so we remove the config option.
    # XLA: torch.compile is not used with XLA; XLA has its own graph compilation.

    # I/O and Logging
    out_dir: str = "checkpoints"
    log_interval: int = 20
    save_interval: int = 1000

    # W&B configuration
    wandb_project: Optional[str] = "lunaris-codex-tpu" # Changed project name for clarity
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = f"run-tpu-{time.strftime('%Y-%m-%d-%H-%M')}"

    @classmethod
    def from_yaml(cls, path: str):
        """Loads configuration from a YAML file, ensuring correct types."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Remove keys not present in the dataclass for TPU version
        config_dict.pop('device', None)
        config_dict.pop('compile_model', None)

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


# --- Sharded Memory-Mapped Dataset (Unchanged from original) ---
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

        # XLA: Guard print statements to only run on the master process.
        if xm.is_master_process():
            print(f"[DATA] Loaded {len(self.shards)} shards. Total tokens: {total_tokens/1e9:.2f}B.")
            print(f"[DATA] Creating {self.total_samples:,} non-overlapping samples of length {self.sequence_length}.")

        self.mmap_shards = [np.load(shard, mmap_mode='r') for shard in self.mmap_shards]
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

# --- DDP Setup Function (Removed) ---
# XLA: The manual DDP setup is replaced by the xmp.spawn entry point and xm utilities.

# --- Learning Rate Scheduler (Unchanged from original) ---
def get_lr(step, config: TrainConfig):
    if step < config.warmup_steps:
        return config.learning_rate * step / config.warmup_steps
    if step >= config.max_steps:
        return config.learning_rate * 0.01

    decay_ratio = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return (config.learning_rate * 0.01) + coeff * (config.learning_rate * 0.99)

# --- Robust checkpoint key unwrapping (Unchanged but less critical for XLA) ---
def unwrap_model_keys(state_dict):
    """Remove potential prefixes from model state dict keys."""
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

# --- Main Training Worker Function for XLA ---
def _mp_fn(index, config: TrainConfig):
    """
    This is the main worker function that will be executed on each TPU core.
    The 'index' argument is provided by xmp.spawn and is required, but often unused.
    """
    # XLA: Obtain rank, world size, and the XLA device for this process.
    rank = xm.get_ordinal()
    world_size = xm.xla_world_size()
    device = xm.xla_device()
    is_master_process = xm.is_master_process()

    torch.manual_seed(1337 + rank)
    # XLA: CUDA-specific backend settings are not applicable.

    # XLA: Master process handles all printing and directory creation.
    if is_master_process:
        os.makedirs(config.out_dir, exist_ok=True)
        print("-" * 50)
        print(" " * 10 + "LUNARIS CODEX TRAINING (PyTorch/XLA)")
        print("-" * 50)
        print(f"World Size: {world_size} TPU cores")
        print(f"Model: {config.model}")
        print(f"Data: {config.data_dir}")
        print(f"Batch size (per core): {config.batch_size}")
        print(f"Gradient accumulation: {config.gradient_accumulation_steps}")
        print(f"Learning rate: {config.learning_rate}")
        print(f"Max steps: {config.max_steps}")
        print("-" * 50)

    # XLA: Initialize wandb only on the master process to prevent multiple logs.
    if is_master_process and config.wandb_project:
        import wandb
        wandb.init(project=config.wandb_project, entity=config.wandb_entity, name=config.wandb_run_name, config=config.__dict__)

    # XLA: Data loading setup for distributed training on TPUs.
    train_dataset = ShardDataset(data_dir=config.data_dir, sequence_length=config.sequence_length)
    # XLA: Use DistributedSampler with XLA's rank and world_size for proper data sharding.
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    # XLA: Wrap the DataLoader with MpDeviceLoader. This is the key to efficiently sending
    # data to the TPU device on each core without manual .to(device) calls in the loop.
    train_loader = pl.MpDeviceLoader(train_loader, device)

    # XLA: Move model to the XLA device. No DDP or torch.compile wrapper is needed.
    model = LunarisCodex(config.model).to(device)

    if is_master_process:
        num_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"[MODEL] Number of parameters: {num_params:.2f}M")

    # XLA: The fused AdamW is a CUDA-specific optimization and is not applicable here.
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, betas=(config.beta1, config.beta2), weight_decay=config.weight_decay)

    current_step = 0
    current_epoch = 0
    # XLA: raw_model is just the model itself, as there's no DDP wrapper.
    raw_model = model
    checkpoint_path = os.path.join(config.out_dir, "latest_checkpoint.pt")

    if os.path.exists(checkpoint_path):
        if is_master_process: print(f"[SETUP] Resuming from checkpoint: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location='cpu')
        unwrapped_state_dict = unwrap_model_keys(state['model'])
        raw_model.load_state_dict(unwrapped_state_dict)
        optimizer.load_state_dict(state['optimizer'])
        current_step = state['step']
        current_epoch = state.get('epoch', 0)
        # XLA: Use a barrier to ensure all processes have loaded the checkpoint before proceeding.
        xm.rendezvous("checkpoint_loaded")
        if is_master_process: print(f"[SETUP] Resumed successfully. Starting from step {current_step}")

    optimizer.zero_grad(set_to_none=True)
    train_sampler.set_epoch(current_epoch)

    if is_master_process:
        print(f"\n[TRAIN] Starting training from step {current_step} up to {config.max_steps} steps...")
        pbar = tqdm(total=config.max_steps, desc="Training Steps", initial=current_step, ncols=120)

    data_iter = iter(train_loader)

    while current_step < config.max_steps:
        current_step += 1

        lr = get_lr(current_step, config)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        accumulated_loss = 0.0

        for _ in range(config.gradient_accumulation_steps):
            # XLA: The DDP no_sync context is not needed. XLA handles gradient sync.
            # XLA: MpDeviceLoader automatically moves data to the correct TPU core.
            try:
                x, y = next(data_iter)
            except StopIteration:
                current_epoch += 1
                train_sampler.set_epoch(current_epoch)
                data_iter = iter(train_loader)
                x, y = next(data_iter)

            # XLA: torch.amp.autocast is not needed; XLA manages mixed-precision (bfloat16) on TPUs.
            logits, loss = model(x, y)
            loss = loss / config.gradient_accumulation_steps
            accumulated_loss += loss.item()
            loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

        # XLA: Use xm.optimizer_step to update weights. This function handles the gradient
        # reduction across all TPU cores and the optimizer step in a single call.
        # The barrier=True ensures all cores complete the step before the next iteration.
        xm.optimizer_step(optimizer, barrier=True)
        optimizer.zero_grad(set_to_none=True)

        # XLA: All I/O and logging operations are performed only on the master process.
        if is_master_process:
            pbar.update(1)

            if current_step % config.log_interval == 0:
                log_loss = accumulated_loss
                try:
                    perplexity = math.exp(log_loss)
                except (OverflowError, ValueError):
                    perplexity = float('inf')

                current_lr = lr
                pbar.set_postfix({
                    "loss": f"{log_loss:.3f}",
                    "ppl": f"{perplexity:.2f}" if perplexity != float('inf') else "inf",
                    "lr": f"{current_lr:.2e}",
                    "gnorm": f"{grad_norm.item():.2f}"
                })

                if config.wandb_project:
                    wandb.log({
                        "step": current_step, "loss": log_loss, "perplexity": perplexity,
                        "lr": current_lr, "grad_norm": grad_norm.item(), "epoch": current_epoch
                    })

            if current_step > 0 and current_step % config.save_interval == 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': config.__dict__,
                    'step': current_step,
                    'epoch': current_epoch,
                }
                save_path = os.path.join(config.out_dir, f"ckpt_{current_step}.pt")
                # XLA: Use xm.save to correctly save the model from a distributed TPU environment.
                # It ensures only the master process writes to disk and adds a barrier for synchronization.
                xm.save(checkpoint, save_path)
                latest_path = os.path.join(config.out_dir, "latest_checkpoint.pt")
                xm.save(checkpoint, latest_path)
                print(f"\n[CHECKPOINT] Saved checkpoint to {save_path}")

    if is_master_process:
        print("\nMax steps reached. Finishing training.")
        pbar.close()
        if config.wandb_project:
            wandb.finish()
    # XLA: No need for destroy_process_group; XLA manages the process group lifecycle.

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Train a LunarisCodex model on TPUs using PyTorch/XLA.")
    parser.add_argument("config", type=str, help="Path to the config.yaml file.")
    args = parser.parse_args()

    # XLA: This is the main entry point for a PyTorch/XLA multiprocessing program.
    # It spawns one process for each available TPU core and calls the target function (_mp_fn).
    # The config object is passed as an argument to the worker function.
    config = TrainConfig.from_yaml(args.config)
    xmp.spawn(_mp_fn, args=(config,), start_method='fork')
