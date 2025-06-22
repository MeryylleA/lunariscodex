# train.py
# A robust, feature-rich training script for the LunarisCodex model.
# CORRECTED VERSION - Fixed gradient accumulation logging, LR scheduler, and other subtle bugs

import os
import time
import math
import glob
from dataclasses import dataclass, field
from typing import Optional
from contextlib import nullcontext

import yaml
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, get_rank, get_world_size
from tqdm import tqdm

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
    num_epochs: int = 1 # Set to a large number for step-based training
    grad_clip: float = 1.0
    device: str = "cuda"
    compile_model: bool = True

    # I/O and Logging
    out_dir: str = "checkpoints"
    log_interval: int = 20
    save_interval: int = 1000

    # W&B configuration
    wandb_project: Optional[str] = "lunaris-codex"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = f"run-{time.strftime('%Y-%m-%d-%H-%M')}"

    @classmethod
    def from_yaml(cls, path: str):
        """Loads configuration from a YAML file, ensuring correct types."""
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


# --- Sharded Memory-Mapped Dataset ---
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

        # BUG FIX: The total length must guarantee that any valid index `i` has `sequence_length + 1`
        # subsequent tokens available for `x` and `y`. The previous calculation was off by one,
        # allowing an index to be requested that was too close to the end of the total token stream,
        # which would cause an IndexError when trying to read across the *final* shard boundary.
        # This new calculation is conservative and inherently safe.
        self.total_length = max(0, self.cumulative_lengths[-1] - self.sequence_length - 1)

        print(f"[DATA] Loaded {len(self.shards)} shards. Effective total samples: {self.total_length}. Total tokens: {self.cumulative_lengths[-1] / 1e9:.2f}B.")

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        # Find which shard contains this index
        shard_idx = np.searchsorted(self.cumulative_lengths, idx, side='right')
        local_idx = idx if shard_idx == 0 else idx - self.cumulative_lengths[shard_idx - 1]

        # Handle cross-shard sequences. Because __len__ is now correct, this logic will
        # never be triggered for the final shard in a way that causes an IndexError.
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

# --- DDP Setup ---
def setup_ddp():
    is_ddp = 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1
    if is_ddp:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        print(f"[DDP] Setup complete: rank {rank}, world_size {world_size}")
        return True, rank, world_size
    return False, 0, 1

# --- Learning Rate Scheduler ---
def get_lr(step, config: TrainConfig):
    # CORRECTED: Lower minimum LR floor for better fine-tuning
    if step < config.warmup_steps:
        return config.learning_rate * step / config.warmup_steps
    if step >= config.max_steps:
        return config.learning_rate * 0.01  # Lower floor

    decay_ratio = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return (config.learning_rate * 0.01) + coeff * (config.learning_rate * 0.99)

# --- Robust checkpoint key unwrapping ---
def unwrap_model_keys(state_dict):
    """Remove DDP and torch.compile prefixes from model state dict keys."""
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

# --- Main Training Function ---
def train(config_path: str):
    config = TrainConfig.from_yaml(config_path)
    is_ddp, rank, world_size = setup_ddp()
    is_master_process = rank == 0

    torch.manual_seed(1337 + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    device_type = 'cuda' if 'cuda' in config.device else 'cpu'
    ctx = torch.amp.autocast(device_type=device_type, dtype=dtype)

    if is_master_process:
        os.makedirs(config.out_dir, exist_ok=True)
        print("-" * 50)
        print(" " * 15 + "LUNARIS CODEX TRAINING")
        print("-" * 50)
        print(f"Model: {config.model}")
        print(f"Data: {config.data_dir}")
        print(f"Batch size: {config.batch_size}")
        print(f"Gradient accumulation: {config.gradient_accumulation_steps}")
        print(f"Learning rate: {config.learning_rate}")
        print(f"Max steps: {config.max_steps}")
        print("-" * 50)

    if is_master_process and config.wandb_project:
        import wandb
        wandb.init(project=config.wandb_project, entity=config.wandb_entity, name=config.wandb_run_name, config=config.__dict__)

    train_dataset = ShardDataset(data_dir=config.data_dir, sequence_length=config.sequence_length)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True) if is_ddp else None
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)

    model = LunarisCodex(config.model).to(config.device)

    if is_master_process:
        num_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"[MODEL] Number of parameters: {num_params:.2f}M")

    if config.compile_model:
        if is_master_process: print("[MODEL] Compiling model...")
        model = torch.compile(model)
    if is_ddp:
        model = DDP(model, device_ids=[int(os.environ['LOCAL_RANK'])])

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, betas=(config.beta1, config.beta2), weight_decay=config.weight_decay)

    current_step = 0
    current_epoch = 0
    raw_model = model.module if is_ddp else model
    checkpoint_path = os.path.join(config.out_dir, "latest_checkpoint.pt")

    if os.path.exists(checkpoint_path):
        if is_master_process: print(f"[SETUP] Resuming from checkpoint: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=config.device)
        unwrapped_state_dict = unwrap_model_keys(state['model'])
        raw_model.load_state_dict(unwrapped_state_dict)
        optimizer.load_state_dict(state['optimizer'])
        current_step = state['step']
        current_epoch = state.get('epoch', 0)
        if is_master_process: print(f"[SETUP] Resumed successfully. Starting from step {current_step}")

    optimizer.zero_grad(set_to_none=True)

    # BUG FIX: The DDP sampler must be seeded differently each epoch to ensure proper shuffling.
    # This was missing for the first epoch (epoch 0), causing all ranks to get identical data.
    if is_ddp:
        train_sampler.set_epoch(current_epoch)

    if is_master_process:
        print(f"\n[TRAIN] Starting training from step {current_step} up to {config.max_steps} steps...")
        pbar = tqdm(total=config.max_steps, desc="Training Steps", initial=current_step, ncols=120)

    data_iter = iter(train_loader)

    while current_step < config.max_steps:
        # BUG FIX: The step counter must be incremented *before* calculating the learning rate.
        # Previously, the LR for step N was used on step N+1, causing an off-by-one schedule
        # and an LR of 0.0 for the very first step.
        current_step += 1

        # Control learning rate manually
        lr = get_lr(current_step, config)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        accumulated_loss = 0.0

        # Loop for gradient accumulation
        for micro_step in range(config.gradient_accumulation_steps):
            # BUG FIX: Use DDP's no_sync context manager to avoid redundant gradient all-reduces.
            # Gradients are now synchronized only on the final micro-step, not on every one,
            # which is a major performance improvement for distributed training.
            is_last_micro_step = (micro_step == config.gradient_accumulation_steps - 1)
            ddp_context = model.no_sync() if is_ddp and not is_last_micro_step else nullcontext()

            with ddp_context:
                try:
                    x, y = next(data_iter)
                except StopIteration:
                    current_epoch += 1
                    if is_ddp:
                        train_loader.sampler.set_epoch(current_epoch)
                    data_iter = iter(train_loader)
                    x, y = next(data_iter)

                x, y = x.to(config.device, non_blocking=True), y.to(config.device, non_blocking=True)

                # Forward pass
                with ctx:
                    logits, loss = model(x, y)
                    loss = loss / config.gradient_accumulation_steps

                accumulated_loss += loss.item()

                # Backward pass (will sync grads only if not in ddp_context)
                loss.backward()

        # Update weights
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if is_master_process:
            pbar.update(1)

            # Logging
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
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': config.__dict__,
                    'step': current_step,
                    'epoch': current_epoch,
                }
                save_path = os.path.join(config.out_dir, f"ckpt_{current_step}.pt")
                torch.save(checkpoint, save_path)

                latest_path = os.path.join(config.out_dir, "latest_checkpoint.pt")
                torch.save(checkpoint, latest_path)
                print(f"\n[CHECKPOINT] Saved checkpoint to {save_path}")

    # Cleanup
    if is_master_process:
        print("\nMax steps reached. Finishing training.")
        pbar.close()
        if config.wandb_project:
            wandb.finish()
    if is_ddp:
        destroy_process_group()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Train a LunarisCodex model.")
    parser.add_argument("config", type=str, help="Path to the config.yaml file.")
    args = parser.parse_args()
    train(args.config)
