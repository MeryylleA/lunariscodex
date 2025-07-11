# train_tpu_fixed.py
# TPU-optimized training script for the LunarisCodex model using PyTorch XLA 2.7+
# Fixed for PJRT runtime compatibility and GradScaler issues

import os
import time
import math
import glob
from dataclasses import dataclass, field
from typing import Optional
from contextlib import nullcontext
import functools

import yaml
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# TPU/XLA imports - Updated for PyTorch XLA 2.7+
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.utils.utils as xu
from torch_xla.amp import autocast, GradScaler
import torch_xla.runtime as xr

# Assuming model.py contains the LunarisCodex and LunarisCodexConfig classes
from model import LunarisCodex, LunarisCodexConfig

# --- Configuration Dataclass ---
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
    num_epochs: int = 1
    grad_clip: float = 1.0
    compile_model: bool = True
    
    # TPU-specific configuration
    tpu_cores: int = 4
    mixed_precision: bool = True
    sync_every_n_steps: int = 1

    # I/O and Logging
    out_dir: str = "checkpoints"
    log_interval: int = 20
    save_interval: int = 1000

    # W&B configuration
    wandb_project: Optional[str] = "lunaris-codex-tpu"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = f"tpu-run-{time.strftime('%Y-%m-%d-%H-%M')}"

    @classmethod
    def from_yaml(cls, path: str):
        """Loads configuration from a YAML file, ensuring correct types."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)

        model_config_dict = config_dict.pop("model", {})
        model_config = LunarisCodexConfig(**model_config_dict)
        config_dict['model'] = model_config

        float_fields = ['learning_rate', 'weight_decay', 'beta1', 'beta2', 'grad_clip']
        int_fields = ['warmup_steps', 'max_steps', 'batch_size', 'gradient_accumulation_steps', 
                      'num_epochs', 'save_interval', 'log_interval', 'tpu_cores', 'sync_every_n_steps']

        for key in float_fields:
            if key in config_dict:
                config_dict[key] = float(config_dict[key])
        for key in int_fields:
            if key in config_dict:
                config_dict[key] = int(config_dict[key])

        return cls(**config_dict)


# --- TPU-Optimized Sharded Dataset ---
class TPUShardDataset(Dataset):
    """TPU-optimized dataset with efficient data loading and caching."""
    
    def __init__(self, data_dir: str, sequence_length: int, device=None):
        super().__init__()
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.device = device or xm.xla_device()
        
        self.shards = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
        if not self.shards:
            raise ValueError(f"No .npy files found in directory: {data_dir}")

        # Count total tokens across all shards
        total_tokens = sum(np.load(shard, mmap_mode='r').shape[0] for shard in self.shards)
        self.total_samples = total_tokens // self.sequence_length

        if xm.is_master_ordinal():
            print(f"[TPU-DATA] Loaded {len(self.shards)} shards. Total tokens: {total_tokens/1e9:.2f}B.")
            print(f"[TPU-DATA] Creating {self.total_samples:,} non-overlapping samples of length {self.sequence_length}.")
        
        # Use memory mapping for efficient data loading
        self.mmap_shards = [np.load(shard, mmap_mode='r') for shard in self.shards]
        self.shard_lengths = [len(shard) for shard in self.mmap_shards]
        self.cumulative_lengths = np.cumsum(self.shard_lengths)

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # Calculate token start position for this block
        token_start_pos = idx * self.sequence_length
        
        # Find which shard this block starts in
        shard_idx = np.searchsorted(self.cumulative_lengths, token_start_pos, side='right')
        
        # Calculate local index within the shard
        local_start_idx = token_start_pos if shard_idx == 0 else token_start_pos - self.cumulative_lengths[shard_idx - 1]
        
        # We need sequence_length + 1 tokens for x and y
        seq_len_with_target = self.sequence_length + 1
        
        # Handle reading across shards if necessary
        if local_start_idx + seq_len_with_target > self.shard_lengths[shard_idx]:
            # Part 1 from first shard
            remaining_len = self.shard_lengths[shard_idx] - local_start_idx
            seq_part1 = self.mmap_shards[shard_idx][local_start_idx : local_start_idx + remaining_len]

            # Part 2 from next shard
            if shard_idx + 1 < len(self.mmap_shards):
                needed_from_next = seq_len_with_target - remaining_len
                seq_part2 = self.mmap_shards[shard_idx + 1][:needed_from_next]
                seq = np.concatenate((seq_part1, seq_part2))
            else:
                seq = seq_part1
        else:
            # Entire block is within a single shard
            seq = self.mmap_shards[shard_idx][local_start_idx : local_start_idx + seq_len_with_target]
        
        # Pad if sequence is too short
        if len(seq) < seq_len_with_target:
            pad_len = seq_len_with_target - len(seq)
            seq = np.pad(seq, (0, pad_len), 'constant', constant_values=-1)

        seq_tensor = torch.from_numpy(seq.astype(np.int64))
        x, y = seq_tensor[:-1], seq_tensor[1:]
        return x, y


# --- TPU Learning Rate Scheduler ---
def get_lr(step, config: TrainConfig):
    """Learning rate scheduler with TPU-optimized warmup and decay."""
    if step < config.warmup_steps:
        return config.learning_rate * step / config.warmup_steps
    if step >= config.max_steps:
        return config.learning_rate * 0.01

    decay_ratio = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return (config.learning_rate * 0.01) + coeff * (config.learning_rate * 0.99)


# --- TPU Checkpoint Management ---
def save_checkpoint_tpu(model, optimizer, config, step, epoch, save_path):
    """TPU-optimized checkpoint saving."""
    if xm.is_master_ordinal():
        # Get the raw model state (unwrapped from any DDP/compile wrappers)
        raw_model = model
        if hasattr(model, 'module'):
            raw_model = model.module
        if hasattr(raw_model, '_orig_mod'):
            raw_model = raw_model._orig_mod
            
        checkpoint = {
            'model': raw_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'config': config.__dict__,
            'step': step,
            'epoch': epoch,
        }
        
        # Use xm.save for TPU-optimized saving
        xm.save(checkpoint, save_path)
        print(f"\n[TPU-CHECKPOINT] Saved checkpoint to {save_path}")


def load_checkpoint_tpu(model, optimizer, checkpoint_path, device):
    """TPU-optimized checkpoint loading."""
    if os.path.exists(checkpoint_path):
        print(f"[TPU-SETUP] Loading checkpoint: {checkpoint_path}")
        
        # Load checkpoint to TPU device
        state = torch.load(checkpoint_path, map_location=device)
        
        # Get the raw model for state dict loading
        raw_model = model
        if hasattr(model, 'module'):
            raw_model = model.module
        if hasattr(raw_model, '_orig_mod'):
            raw_model = raw_model._orig_mod
            
        raw_model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        
        return state['step'], state.get('epoch', 0)
    return 0, 0


# --- Main TPU Training Function ---
def train_tpu(rank, config: TrainConfig):
    """Main TPU training function for a single TPU core."""
    
    # Setup TPU device
    device = xm.xla_device()
    
    # Fixed for PyTorch XLA 2.7+: Use runtime API
    world_size = xr.global_runtime_device_count()
    is_master = xm.is_master_ordinal()
    
    if is_master:
        print(f"[TPU-SETUP] Starting TPU training on {world_size} cores")
        print(f"[TPU-SETUP] Device: {device}")
        print("-" * 50)
        print(" " * 10 + "LUNARIS CODEX TPU TRAINING")
        print("-" * 50)
        print(f"Model: {config.model}")
        print(f"Data: {config.data_dir}")
        print(f"Batch size per core: {config.batch_size}")
        print(f"Global batch size: {config.batch_size * world_size}")
        print(f"Gradient accumulation: {config.gradient_accumulation_steps}")
        print(f"Learning rate: {config.learning_rate}")
        print(f"Max steps: {config.max_steps}")
        print("-" * 50)

    # Set random seed
    torch.manual_seed(1337 + rank)
    
    # Initialize W&B on master process
    if is_master and config.wandb_project:
        import wandb
        wandb.init(
            project=config.wandb_project, 
            entity=config.wandb_entity, 
            name=config.wandb_run_name, 
            config=config.__dict__
        )

    # Setup dataset and dataloader
    train_dataset = TPUShardDataset(
        data_dir=config.data_dir, 
        sequence_length=config.sequence_length,
        device=device
    )
    
    # TPU-optimized sampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, 
        num_replicas=world_size, 
        rank=rank,
        shuffle=True
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        sampler=train_sampler,
        num_workers=0,  # TPU works best with num_workers=0
        pin_memory=False,  # Not needed for TPU
        drop_last=True
    )
    
    # Wrap with TPU parallel loader
    train_loader = pl.MpDeviceLoader(train_loader, device)

    # Initialize model
    model = LunarisCodex(config.model).to(device)
    
    if is_master:
        num_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"[TPU-MODEL] Number of parameters: {num_params:.2f}M")

    # TPU model compilation - Use XLA optimization
    if config.compile_model:
        if is_master:
            print("[TPU-MODEL] Compiling model for TPU...")
        # For TPU, avoid torch.compile and rely on XLA's built-in optimization
        # model = torch.compile(model, backend='openxla')
        pass  # XLA automatically optimizes the model

    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.learning_rate, 
        betas=(config.beta1, config.beta2), 
        weight_decay=config.weight_decay
    )
    
    # Initialize gradient scaler for mixed precision
    # Fixed: Use proper GradScaler initialization for PyTorch XLA 2.7+
    scaler = None
    if config.mixed_precision:
        # For PyTorch XLA 2.7+, initialize GradScaler without device parameter
        # The device will be handled internally by XLA
        scaler = GradScaler()

    # Setup checkpoint directory
    if is_master:
        os.makedirs(config.out_dir, exist_ok=True)

    # Load checkpoint if exists
    checkpoint_path = os.path.join(config.out_dir, "latest_checkpoint.pt")
    current_step, current_epoch = load_checkpoint_tpu(model, optimizer, checkpoint_path, device)

    if is_master:
        print(f"\n[TPU-TRAIN] Starting training from step {current_step} up to {config.max_steps} steps...")
        pbar = tqdm(total=config.max_steps, desc="TPU Training Steps", initial=current_step, ncols=120)

    # Training loop
    optimizer.zero_grad()
    
    while current_step < config.max_steps:
        # Set epoch for proper shuffling
        train_sampler.set_epoch(current_epoch)
        
        for batch_idx, (x, y) in enumerate(train_loader):
            current_step += 1
            
            # Learning rate scheduling
            lr = get_lr(current_step, config)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            accumulated_loss = 0.0

            # Gradient accumulation loop
            for micro_step in range(config.gradient_accumulation_steps):
                # Forward pass with mixed precision
                if config.mixed_precision and scaler is not None:
                    with autocast(device_type='cuda'):
                        logits, loss, _ = model(x, targets=y)
                        loss = loss / config.gradient_accumulation_steps
                else:
                    logits, loss, _ = model(x, targets=y)
                    loss = loss / config.gradient_accumulation_steps

                accumulated_loss += loss.item()

                # Backward pass
                if config.mixed_precision and scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

            # Gradient clipping and optimizer step
            if config.mixed_precision and scaler is not None:
                scaler.unscale_(optimizer)
                
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            
            # TPU-optimized optimizer step
            if config.mixed_precision and scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                xm.optimizer_step(optimizer)  # Use XLA optimizer step for better performance
                
            optimizer.zero_grad()

            # Synchronize TPU cores periodically
            if current_step % config.sync_every_n_steps == 0:
                xm.mark_step()

            # Progress tracking and logging
            if is_master:
                pbar.update(1)

                if current_step % config.log_interval == 0:
                    log_loss = accumulated_loss

                    if log_loss < 100:
                        try:
                            perplexity = math.exp(log_loss)
                        except (OverflowError, ValueError):
                            perplexity = float('inf')
                    else:
                        perplexity = float('inf')

                    current_lr = lr

                    postfix_data = {
                        "loss": f"{log_loss:.3f}",
                        "ppl": f"{perplexity:.2f}" if perplexity != float('inf') else "inf",
                        "lr": f"{current_lr:.2e}",
                        "gnorm": f"{grad_norm.item():.2f}"
                    }
                    pbar.set_postfix(postfix_data)

                    if config.wandb_project:
                        wandb.log({
                            "step": current_step,
                            "loss": log_loss,
                            "perplexity": perplexity,
                            "lr": current_lr,
                            "grad_norm": grad_norm.item(),
                            "epoch": current_epoch
                        })

                # Checkpointing
                if current_step > 0 and current_step % config.save_interval == 0:
                    save_path = os.path.join(config.out_dir, f"ckpt_{current_step}.pt")
                    save_checkpoint_tpu(model, optimizer, config, current_step, current_epoch, save_path)
                    
                    latest_path = os.path.join(config.out_dir, "latest_checkpoint.pt")
                    save_checkpoint_tpu(model, optimizer, config, current_step, current_epoch, latest_path)

            # Check if we've reached max steps
            if current_step >= config.max_steps:
                break

        current_epoch += 1
        
        if current_step >= config.max_steps:
            break

    # Final cleanup
    if is_master:
        print("\n[TPU-TRAIN] Max steps reached. Finishing training.")
        pbar.close()
        if config.wandb_project:
            wandb.finish()

    # Wait for all TPU cores to finish
    xm.rendezvous("training_finished")


# --- TPU Multi-Processing Entry Point ---
def main():
    """Main entry point for TPU training."""
    import argparse
    parser = argparse.ArgumentParser(description="Train LunarisCodex on TPU")
    parser.add_argument("config", type=str, help="Path to the config.yaml file")
    args = parser.parse_args()
    
    # Load configuration
    config = TrainConfig.from_yaml(args.config)
    
    # Print TPU information
    print(f"[TPU-SETUP] Initializing TPU training")
    print(f"[TPU-SETUP] Using PyTorch XLA version: {torch.__version__}")
    
    # For PyTorch XLA 2.7+, use None to auto-detect all available devices
    xmp.spawn(
        train_tpu,
        args=(config,),
        nprocs=None,  # Auto-detect all available TPU cores
        start_method='spawn'
    )


if __name__ == '__main__':
    main()
