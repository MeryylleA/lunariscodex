# train_tpu.py
# A robust, feature-rich training script for the LunarisCodex model.
# REFACTORED for PyTorch/XLA and distributed training on Google Cloud TPUs.

import os
import time
import math
import glob
import yaml
from dataclasses import dataclass, field
from typing import Optional

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from tqdm import tqdm

# XLA: Import PyTorch/XLA libraries
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

# Assuming model.py contains the LunarisCodex and LunarisCodexConfig classes
from model import LunarisCodex, LunarisCodexConfig

# --- Configuration Dataclass (Unchanged, device field is now ignored) ---
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
    device: str = "tpu" # Ignored, device is set by XLA
    compile_model: bool = True # torch.compile is not typically used with XLA

    # I/O and Logging
    out_dir: str = "checkpoints"
    log_interval: int = 20
    save_interval: int = 1000

    # W&B configuration
    wandb_project: Optional[str] = "lunaris-codex"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = f"run-tpu-{time.strftime('%Y-%m-%d-%H-%M')}"

    @classmethod
    def from_yaml(cls, path: str):
        """Loads configuration from a YAML file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        model_config_dict = config_dict.pop("model", {})
        model_config = LunarisCodexConfig(**model_config_dict)
        config_dict['model'] = model_config
        # XLA: torch.compile is not used with XLA, so we disable it.
        if 'compile_model' in config_dict:
            print("Note: 'compile_model' is set to False for XLA/TPU training.")
            config_dict['compile_model'] = False
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

        self.mmap_shards = [np.load(shard, mmap_mode='r') for shard in self.shards]
        self.shard_lengths = [len(shard) for shard in self.mmap_shards]
        self.cumulative_lengths = np.cumsum(self.shard_lengths)

        self.total_length = max(0, self.cumulative_lengths[-1] - sequence_length)

        # XLA: Logging will only appear from the master process later on
        # print(f"[DATA] Loaded {len(self.shards)} shards. Effective total tokens: {self.total_length / 1e9:.2f}B.")

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        shard_idx = np.searchsorted(self.cumulative_lengths, idx, side='right')
        local_idx = idx if shard_idx == 0 else idx - self.cumulative_lengths[shard_idx - 1]

        if local_idx + self.sequence_length + 1 > self.shard_lengths[shard_idx]:
            remaining_len = self.shard_lengths[shard_idx] - local_idx
            seq_part1 = self.mmap_shards[shard_idx][local_idx : local_idx + remaining_len]
            needed_from_next = self.sequence_length + 1 - remaining_len
            seq_part2 = self.mmap_shards[shard_idx + 1][:needed_from_next]
            seq = np.concatenate((seq_part1, seq_part2))
        else:
            seq = self.mmap_shards[shard_idx][local_idx : local_idx + self.sequence_length + 1]

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

# --- Robust checkpoint key unwrapping (Unchanged, useful for loading DDP checkpoints) ---
def unwrap_model_keys(state_dict):
    unwrapped = {}
    # Handles checkpoints from DDP, torch.compile, or both
    prefixes_to_remove = ['_orig_mod.module.', 'module.', '_orig_mod.']
    for k, v in state_dict.items():
        new_k = k
        for prefix in prefixes_to_remove:
            if new_k.startswith(prefix):
                new_k = new_k[len(prefix):]
                break
        unwrapped[new_k] = v
    return unwrapped

# --- XLA: Main Training Function for a single process ---
def _mp_fn(index, config: TrainConfig):
    """ Main training function to be spawned by xmp.spawn. """
    # XLA: Distributed setup
    torch.manual_seed(1337 + index)
    device = xm.xla_device()
    rank = xm.get_ordinal()
    world_size = xm.xla_world_size()
    is_master_process = xm.is_master_process()

    # Note: TF32 and CUDNN are CUDA-specific and removed.
    # Dtype is bfloat16 for TPUs.
    dtype = torch.bfloat16
    ctx = torch.amp.autocast(device_type='xla', dtype=dtype)

    if is_master_process:
        os.makedirs(config.out_dir, exist_ok=True)
        print("-" * 50)
        print(" " * 10 + "LUNARIS CODEX TPU TRAINING (XLA)")
        print("-" * 50)
        print(f"Model: {config.model}")
        print(f"Data: {config.data_dir}")
        print(f"Batch size per device: {config.batch_size}")
        print(f"Total batch size: {config.batch_size * world_size}")
        print(f"Gradient accumulation: {config.gradient_accumulation_steps}")
        print(f"Learning rate: {config.learning_rate}")
        print(f"Max steps: {config.max_steps}")
        print(f"World size: {world_size}")
        print("-" * 50)

    if is_master_process and config.wandb_project:
        import wandb
        wandb.init(project=config.wandb_project, entity=config.wandb_entity, name=config.wandb_run_name, config=config.__dict__)

    train_dataset = ShardDataset(data_dir=config.data_dir, sequence_length=config.sequence_length)
    # XLA: Use a standard DistributedSampler with XLA rank and world_size
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    # XLA: pin_memory is a CUDA feature and removed. num_workers can be low.
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, sampler=train_sampler, num_workers=2)
    # XLA: Wrap DataLoader with MpDeviceLoader for efficient data transfer to TPU cores.
    train_loader = pl.MpDeviceLoader(train_loader, device)

    model = LunarisCodex(config.model).to(device)

    if is_master_process:
        # Model parameters are already available on all devices, just print from master.
        num_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"[MODEL] Number of parameters: {num_params:.2f}M")

    # XLA: torch.compile is not used. DDP wrapper is removed.
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, betas=(config.beta1, config.beta2), weight_decay=config.weight_decay)

    current_step = 0
    current_epoch = 0
    # XLA: No DDP wrapper, so model is the raw model.
    raw_model = model
    checkpoint_path = os.path.join(config.out_dir, "latest_checkpoint.pt")

    if os.path.exists(checkpoint_path):
        # All processes load the checkpoint to have the same model and optimizer state.
        # map_location is set to 'cpu' to avoid OOM on device 0 before distributing.
        state = torch.load(checkpoint_path, map_location='cpu')

        unwrapped_state_dict = unwrap_model_keys(state['model'])
        raw_model.load_state_dict(unwrapped_state_dict)

        optimizer.load_state_dict(state['optimizer'])
        current_step = state['step']
        current_epoch = state.get('epoch', 0)
        if is_master_process:
            print(f"[SETUP] Resumed successfully from checkpoint. Starting from step {current_step}")

    optimizer.zero_grad(set_to_none=True)

    pbar = None
    if is_master_process:
        print(f"\n[TRAIN] Starting training from step {current_step} up to {config.max_steps} steps...")
        pbar = tqdm(total=config.max_steps, desc="Training Steps", initial=current_step, ncols=120)

    data_iter = iter(train_loader)

    while current_step < config.max_steps:
        lr = get_lr(current_step, config)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        accumulated_loss = 0.0

        for micro_step in range(config.gradient_accumulation_steps):
            try:
                x, y = next(data_iter)
            except StopIteration:
                current_epoch += 1
                # XLA: set_epoch on sampler is still needed for correct shuffling
                train_loader.sampler.set_epoch(current_epoch)
                data_iter = iter(train_loader)
                x, y = next(data_iter)

            # XLA: No .to(device) call needed; MpDeviceLoader handles it.
            with ctx:
                logits, loss = model(x, y)
                loss = loss / config.gradient_accumulation_steps

            accumulated_loss += loss.item()
            loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        # XLA: Use xm.optimizer_step to perform an all-reduce on gradients and update weights.
        xm.optimizer_step(optimizer)
        optimizer.zero_grad(set_to_none=True)

        current_step += 1

        if is_master_process:
            pbar.update(1)

        # Logging (master process handles aggregation and output)
        if current_step % config.log_interval == 0:
            # XLA: Aggregate metrics from all devices for accurate logging.
            # Use xm.mesh_reduce to average the loss across all TPU cores.
            log_loss = xm.mesh_reduce('loss_reduce', accumulated_loss, np.mean)
            grad_norm_val = xm.mesh_reduce('gnorm_reduce', grad_norm.item(), np.mean)

            if is_master_process:
                if log_loss < 100:
                    try:
                        perplexity = math.exp(log_loss)
                    except (OverflowError, ValueError):
                        perplexity = float('inf')
                else:
                    perplexity = float('inf')

                current_lr = lr # Same on all processes, no need to reduce

                postfix_data = {
                    "loss": f"{log_loss:.3f}",
                    "ppl": f"{perplexity:.2f}" if perplexity != float('inf') else "inf",
                    "lr": f"{current_lr:.2e}",
                    "gnorm": f"{grad_norm_val:.2f}"
                }
                pbar.set_postfix(postfix_data)

                if config.wandb_project:
                    wandb.log({
                        "step": current_step,
                        "loss": log_loss,
                        "perplexity": perplexity,
                        "lr": current_lr,
                        "grad_norm": grad_norm_val,
                        "epoch": current_epoch
                    })

        # Checkpointing (master process handles saving)
        if current_step > 0 and current_step % config.save_interval == 0:
            checkpoint = {
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': config.__dict__,
                'step': current_step,
                'epoch': current_epoch,
            }
            # XLA: Use xm.save to ensure all processes are finished and only master saves.
            save_path = os.path.join(config.out_dir, f"ckpt_{current_step}.pt")
            xm.save(checkpoint, save_path)

            latest_path = os.path.join(config.out_dir, "latest_checkpoint.pt")
            xm.save(checkpoint, latest_path) # xm.save is guarded, no need for extra if

            if is_master_process:
                print(f"\n[CHECKPOINT] Saved checkpoint to {save_path}")

    # Cleanup
    if is_master_process:
        print("\nMax steps reached. Finishing training.")
        pbar.close()
        if config.wandb_project:
            wandb.finish()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Train a LunarisCodex model on TPUs using PyTorch/XLA.")
    parser.add_argument("config", type=str, help="Path to the config.yaml file.")
    args = parser.parse_args()
    
    # Load configuration from YAML
    config = TrainConfig.from_yaml(args.config)
    
    # XLA: Use xmp.spawn to launch the training function on all available TPU cores
    print("Starting XLA multiprocessing spawn...")
    xmp.spawn(_mp_fn, args=(config,), nprocs=None, start_method='fork')
