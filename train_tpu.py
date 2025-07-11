# train_tpu.py
# A robust, feature-rich training script for the LunarisCodex model, ported for PyTorch/XLA on Google Cloud TPU.
# CORRECTED VERSION - Fixed gradient accumulation logging, LR scheduler, and other subtle bugs
# PORTED FOR TPU - Adapted from the original DDP script for single-host, multi-device TPU training.

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
# XLA: DDP is not used. XLA handles distribution implicitly.
# We keep torch.distributed for host-side process group initialization and cleanup.
from torch.distributed import init_process_group, destroy_process_group
from tqdm import tqdm

# XLA: Import required PyTorch/XLA libraries for TPU device handling, runtime info, and distributed operations.
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

# Assuming model.py contains the LunarisCodex and LunarisCodexConfig classes
from model import LunarisCodex, LunarisCodexConfig

# --- Configuration Dataclass (Unchanged structure) ---
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
    # XLA: The 'device' field is now illustrative; the script will dynamically use the available TPU device.
    device: str = "tpu"
    compile_model: bool = True

    # I/O and Logging
    out_dir: str = "checkpoints"
    log_interval: int = 20
    save_interval: int = 1000

    # W&B configuration
    wandb_project: Optional[str] = "lunaris-codex-tpu"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = f"run-tpu-{time.strftime('%Y-%m-%d-%H-%M')}"

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

        # XLA: This print will appear on each process/chip.
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

# --- TPU Setup ---
def setup_tpu():
    """Initializes the distributed environment for PyTorch/XLA."""
    # XLA: When using PJRT_DEVICE=TPU and a launcher like torchrun or gcloud,
    # these environment variables are set automatically for each process.
    is_distributed = 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1
    if is_distributed:
        # XLA: Use torch.distributed.init_process_group with the 'gloo' backend for host-side
        # coordination. This is not for model gradient communication, which XLA handles transparently.
        init_process_group("gloo")
        # XLA: Use the torch_xla.runtime to get the true ordinal and world size for the TPU mesh.
        rank = xr.global_ordinal()
        world_size = xr.world_size()
        print(f"[TPU] Setup complete: rank {rank}, world_size {world_size}")
        return True, rank, world_size
    return False, 0, 1

# --- Learning Rate Scheduler (Unchanged) ---
def get_lr(step, config: TrainConfig):
    if step < config.warmup_steps:
        return config.learning_rate * step / config.warmup_steps
    if step >= config.max_steps:
        return config.learning_rate * 0.01

    decay_ratio = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return (config.learning_rate * 0.01) + coeff * (config.learning_rate * 0.99)

# --- Robust checkpoint key unwrapping ---
def unwrap_model_keys(state_dict):
    """Remove torch.compile prefixes from model state dict keys."""
    unwrapped = {}
    # XLA: DDP-related prefixes ('module.', '_orig_mod.module.') are removed as DDP is not used.
    # We only need to handle the prefix added by torch.compile.
    prefixes_to_remove = ['_orig_mod.']

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
    # XLA: Set up the TPU distributed environment.
    is_tpu, rank, world_size = setup_tpu()
    is_master_process = rank == 0

    torch.manual_seed(1337 + rank)

    # XLA: Acquire the TPU device for the current process. This is the key step for device placement.
    device = xm.xla_device()
    # XLA: TPUs excel with bfloat16. The device type for AMP must be 'xla'.
    dtype = torch.bfloat16
    device_type = 'xla'
    ctx = torch.amp.autocast(device_type=device_type, dtype=dtype)

    if is_master_process:
        os.makedirs(config.out_dir, exist_ok=True)
        print("-" * 50)
        print(" " * 10 + "LUNARIS CODEX TRAINING (TPU/XLA)")
        print("-" * 50)
        print(f"Model: {config.model}")
        print(f"Data: {config.data_dir}")
        print(f"Batch size: {config.batch_size}")
        print(f"Gradient accumulation: {config.gradient_accumulation_steps}")
        print(f"Learning rate: {config.learning_rate}")
        print(f"Max steps: {config.max_steps}")
        print(f"Running on {world_size} TPU devices.")
        print("-" * 50)

    # XLA: Only the master process (rank 0) should initialize WandB to avoid duplicate runs.
    if is_master_process and config.wandb_project:
        import wandb
        wandb.init(project=config.wandb_project, entity=config.wandb_entity, name=config.wandb_run_name, config=config.__dict__)

    train_dataset = ShardDataset(data_dir=config.data_dir, sequence_length=config.sequence_length)
    # XLA: DistributedSampler is still the correct way to shard data across distributed processes.
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    ) if is_tpu else None
    # XLA: pin_memory=True is for GPUs and should be False for TPUs.
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, sampler=train_sampler, num_workers=4, pin_memory=False)

    # XLA: Move model to the specific XLA device assigned to this process.
    model = LunarisCodex(config.model).to(device)

    if is_master_process:
        num_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"[MODEL] Number of parameters: {num_params:.2f}M")

    if config.compile_model:
        if is_master_process: print("[MODEL] Compiling model with torch.compile...")
        # XLA: torch.compile works with the XLA backend, providing significant speedups.
        model = torch.compile(model)
    
    # XLA: The DDP wrapper is removed. XLA handles data parallelism automatically.
    # if is_ddp: model = DDP(...)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, betas=(config.beta1, config.beta2), weight_decay=config.weight_decay)

    current_step = 0
    current_epoch = 0
    # XLA: raw_model is simply the model itself, as there is no DDP wrapper to unwrap.
    raw_model = model
    checkpoint_path = os.path.join(config.out_dir, "latest_checkpoint.pt")

    if os.path.exists(checkpoint_path):
        if is_master_process: print(f"[SETUP] Resuming from checkpoint: {checkpoint_path}")
        # XLA: Load checkpoint on CPU first to avoid device-to-device mapping issues.
        state = torch.load(checkpoint_path, map_location='cpu')
        unwrapped_state_dict = unwrap_model_keys(state['model'])
        raw_model.load_state_dict(unwrapped_state_dict)
        optimizer.load_state_dict(state['optimizer'])
        current_step = state['step']
        current_epoch = state.get('epoch', 0)
        if is_master_process: print(f"[SETUP] Resumed successfully. Starting from step {current_step}")

    optimizer.zero_grad(set_to_none=True)

    if is_tpu:
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

        for micro_step in range(config.gradient_accumulation_steps):
            # XLA: The DDP `no_sync` context is not needed. Gradients are accumulated locally on each
            # TPU core's HBM. The cross-replica synchronization happens only during the optimizer step.
            try:
                x, y = next(data_iter)
            except StopIteration:
                current_epoch += 1
                if is_tpu:
                    train_loader.sampler.set_epoch(current_epoch)
                data_iter = iter(train_loader)
                x, y = next(data_iter)

            # XLA: Move data batch to the current process's assigned TPU device.
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            with ctx:
                logits, loss, _ = model(x, targets=y)
                loss = loss / config.gradient_accumulation_steps

            accumulated_loss += loss.item()
            loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        
        # XLA: Replace `optimizer.step()` with `xm.optimizer_step()`.
        # This function performs the gradient reduction (all-reduce) across all TPU replicas
        # and then executes the optimizer's weight update step. `barrier=True` ensures
        # the step completes on all devices before the program proceeds, which is crucial for consistency.
        xm.optimizer_step(optimizer, barrier=True)
        optimizer.zero_grad(set_to_none=True)

        # XLA: All I/O (logging, progress bar, checkpointing) should be guarded to run only on the master process.
        if is_master_process:
            pbar.update(1)

            if current_step % config.log_interval == 0:
                log_loss = accumulated_loss
                perplexity = math.exp(log_loss) if log_loss < 100 else float('inf')
                current_lr = lr

                postfix_data = { "loss": f"{log_loss:.3f}", "ppl": f"{perplexity:.2f}" if perplexity != float('inf') else "inf", "lr": f"{current_lr:.2e}", "gnorm": f"{grad_norm.item():.2f}" }
                pbar.set_postfix(postfix_data)

                if config.wandb_project:
                    wandb.log({ "step": current_step, "loss": log_loss, "perplexity": perplexity, "lr": current_lr, "grad_norm": grad_norm.item(), "epoch": current_epoch })

            if current_step > 0 and current_step % config.save_interval == 0:
                # XLA: The xm.optimizer_step barrier ensures all model replicas are synchronized before saving.
                checkpoint = { 'model': raw_model.state_dict(), 'optimizer': optimizer.state_dict(), 'config': config.__dict__, 'step': current_step, 'epoch': current_epoch, }
                save_path = os.path.join(config.out_dir, f"ckpt_{current_step}.pt")
                # XLA: Use `xm.save` for safe and potentially optimized checkpointing in a distributed setting.
                # It typically saves from the master process while other processes wait.
                xm.save(checkpoint, save_path)

                latest_path = os.path.join(config.out_dir, "latest_checkpoint.pt")
                xm.save(checkpoint, latest_path)
                print(f"\n[CHECKPOINT] Saved checkpoint to {save_path}")

    # Cleanup
    if is_master_process:
        print("\nMax steps reached. Finishing training.")
        pbar.close()
        if config.wandb_project:
            wandb.finish()
    if is_tpu:
        # XLA: Use a rendezvous to ensure all processes finish training and cleanup properly.
        xm.rendezvous("training_complete")
        destroy_process_group()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Train a LunarisCodex model on Cloud TPU with PyTorch/XLA.")
    parser.add_argument("config", type=str, help="Path to the config.yaml file.")
    args = parser.parse_args()
    train(args.config)
