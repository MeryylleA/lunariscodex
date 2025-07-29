"""
Main Training Script for the LunarisCodex Language Model

--- VERSION FOR FSDP + MOE EXPERIMENT ---
This script is adapted to train the Mixture-of-Experts (MoE) version of the model
using PyTorch's Fully Sharded Data Parallel (FSDP) for large-scale training.

Key Changes:
- **FSDP Backend**: Replaces DDP with FSDP for memory and compute efficiency.
- **Sharding Strategy**: Implements FULL_SHARD (ZeRO-3) to shard model params, gradients, and optimizer state.
- **Auto-Wrap Policy**: A custom policy wraps each transformer 'Block' into its own FSDP unit,
  which is critical for handling MoE layers efficiently.
- **Mixed Precision**: Leverages H100 Tensor Cores with BF16 mixed precision.
- **Activation Checkpointing**: Applied to transformer blocks to trade compute for memory.
- **Advanced Checkpointing**: Supports both sharded (efficient) and full (debug) checkpoints,
  with logic to resume transparently from either type.
- **Enhanced Logging**: Reports FSDP parameter counts and memory usage to W&B.
"""

import os
import time
import math
import glob
import yaml
import argparse
from dataclasses import dataclass, field
from typing import Optional, Type
from contextlib import nullcontext
from functools import partial

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.distributed import init_process_group, destroy_process_group, get_rank, get_world_size
from tqdm import tqdm

# --- FSDP Imports ---
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    StateDictType,
    FullStateDictConfig,
    ShardedStateDictConfig,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    enable_wrap,
    wrap,
)
from torch.distributed.fsdp.api import ShardedOptimizer
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter, load_state_dict, save_state_dict


# --- MODIFICAÇÃO MoE ---
from model_moe import LunarisCodex, LunarisCodexConfig, Block # Import Block for wrapping policy

# A classe TrainConfig foi estendida com um parâmetro `backend`.
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
    wandb_project: Optional[str] = "lunaris-codex-fsdp-moe"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = f"run-fsdp-moe-{time.strftime('%Y-%m-%d-%H-%M')}"
    backend: str = "fsdp"  # 'fsdp' ou 'ddp'

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
        if 'aux_loss_weight' not in model_config_dict:
             model_config.aux_loss_weight = 1e-2
        int_fields = ['warmup_steps', 'max_steps', 'batch_size', 'gradient_accumulation_steps', 'num_epochs', 'save_interval', 'log_interval']
        for key in float_fields:
            if key in config_dict:
                config_dict[key] = float(config_dict[key])
        for key in int_fields:
            if key in config_dict:
                config_dict[key] = int(config_dict[key])
        return cls(**config_dict)

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


def setup_distributed(backend: str):
    is_dist = 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1
    if not is_dist:
        return False, 0, 1

    # FSDP requires a device-specific process group
    # Note: this is a breaking change for DDP if not handled carefully, but we'll use a CLI flag.
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    init_process_group("nccl") # FSDP manages device selection internally
    rank = get_rank()
    world_size = get_world_size()
    print(f"[DIST-SETUP] Backend: {backend.upper()}. Rank {rank}/{world_size} on device {torch.cuda.current_device()}.")
    return True, rank, world_size


def get_lr(step, config: TrainConfig):
    # Identical to original script
    if step < config.warmup_steps:
        return config.learning_rate * step / config.warmup_steps
    if step >= config.max_steps:
        return config.learning_rate * 0.01
    decay_ratio = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return (config.learning_rate * 0.01) + coeff * (config.learning_rate * 0.99)


def apply_activation_checkpointing(model, block_class: Type[torch.nn.Module]):
    """Applies activation checkpointing to specified modules."""
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper,
        CheckpointImpl,
        apply_activation_checkpointing,
    )
    non_reentrant_wrapper = partial(
        checkpoint_wrapper,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )
    check_fn = lambda m: isinstance(m, block_class)
    apply_activation_checkpointing(
        model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn
    )


def train(config_path: str, backend_override: str):
    config = TrainConfig.from_yaml(config_path)
    if backend_override:
        config.backend = backend_override

    is_dist, rank, world_size = setup_distributed(config.backend)
    is_master_process = rank == 0

    torch.manual_seed(1337 + rank)
    # torch.use_deterministic_algorithms(True) # Uncomment for debugging reproducibility
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dtype = torch.bfloat16 # H100s love bfloat16
    device = torch.device("cuda")

    if is_master_process:
        os.makedirs(config.out_dir, exist_ok=True)
        print("-" * 60)
        print(f"       LUNARIS CODEX TRAINING (BACKEND: {config.backend.upper()})")
        print("-" * 60)
        print(f"Model Config: {config.model}")
        if config.model.n_experts:
             print(f"--> MoE Enabled: {config.model.n_experts} experts, aux_loss_weight={config.model.aux_loss_weight:.4f}")
        print(f"Data: {config.data_dir}, SeqLen: {config.sequence_length}")
        print(f"Device: {device}, Precision: {dtype}")
        print("-" * 60)

    if is_master_process and config.wandb_project:
        import wandb
        wandb.init(project=config.wandb_project, entity=config.wandb_entity, name=config.wandb_run_name, config=vars(config))


    train_dataset = ShardDataset(data_dir=config.data_dir, sequence_length=config.sequence_length)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, rank=rank, num_replicas=world_size, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)

    # --- FSDP/DDP Model Initialization ---
    model_policy = None
    if config.backend == 'fsdp':
        # 1. Define FSDP policies
        model_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={Block})
        bf16_precision = MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype)

        # 2. Instantiate base model on meta device to prevent allocating full model on rank 0
        with torch.device('meta'):
            model = LunarisCodex(config.model)

        # 3. Apply activation checkpointing BEFORE wrapping with FSDP
        apply_activation_checkpointing(model, block_class=Block)

        # 4. Wrap model with FSDP
        model = FSDP(
            model,
            auto_wrap_policy=model_policy,
            mixed_precision=bf16_precision,
            sharding_strategy=ShardingStrategy.FULL_SHARD, # ZeRO-3
            device_id=torch.cuda.current_device(),
            limit_all_gathers=True, # Important for performance
            use_orig_params=True # Required for torch.compile and optimizers like FusedAdam
        )
    else: # Fallback to DDP
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = LunarisCodex(config.model).to(device)

    if config.compile_model:
        if is_master_process: print("[MODEL] Compiling model with torch.compile()...")
        model = torch.compile(model)

    if config.backend == 'ddp' and is_dist:
         model = DDP(model, device_ids=[torch.cuda.current_device()])

    # --- Logging FSDP stats ---
    if config.backend == 'fsdp' and is_master_process:
        total_params = sum(p.numel() for p in model.parameters())
        # To get the sharded count, we need to check on one rank
        sharded_params = sum(p.numel() for p in model.parameters() if p.is_cuda)
        print(f"[FSDP] Total Parameters: {total_params/1e9:.2f}B")
        print(f"[FSDP] Sharded Parameters on Rank 0: {sharded_params/1e6:.2f}M")
        if config.wandb_project:
            wandb.log({"params/total": total_params, "params/sharded_rank0": sharded_params})


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(config.beta1, config.beta2),
        eps=1e-8,
        fused=True, # Recommended for performance on CUDA
    )

    current_step, current_epoch = 0, 0
    # --- FSDP Checkpoint Resumption Logic ---
    sharded_ckpt_dir = os.path.join(config.out_dir, "latest_sharded_checkpoint")
    full_ckpt_path = os.path.join(config.out_dir, "latest_full_checkpoint.pt")

    if os.path.exists(sharded_ckpt_dir):
        if is_master_process: print(f"[SETUP] Resuming from SHARDED checkpoint: {sharded_ckpt_dir}")
        # Sharded load needs the model and optimizer definitions first
        state_dict = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
        load_state_dict(state_dict, FileSystemReader(sharded_ckpt_dir))
        model.load_state_dict(state_dict["model"])
        optimizer.load_state_dict(state_dict["optimizer"])
        # Load metadata from a separate file
        metadata = torch.load(os.path.join(sharded_ckpt_dir, "metadata.pt"))
        current_step, current_epoch = metadata['step'], metadata['epoch']
    elif os.path.exists(full_ckpt_path):
        if is_master_process: print(f"[SETUP] Resuming from FULL checkpoint: {full_ckpt_path}")
        state = torch.load(full_ckpt_path, map_location='cpu') # Load to CPU to avoid OOM
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        current_step, current_epoch = state['step'], state['epoch']
    
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
        accumulated_main_loss = 0.0
        accumulated_aux_loss = 0.0

        for micro_step in range(config.gradient_accumulation_steps):
            is_last_micro_step = (micro_step == config.gradient_accumulation_steps - 1)
            # FSDP handles gradient sync automatically, no_sync context is not needed
            # DDP still needs it
            sync_context = model.no_sync if config.backend == 'ddp' and is_dist and not is_last_micro_step else nullcontext

            with sync_context():
                try:
                    x, y = next(data_iter)
                except StopIteration:
                    current_epoch += 1
                    train_sampler.set_epoch(current_epoch)
                    data_iter = iter(train_loader)
                    x, y = next(data_iter)

                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

                with torch.amp.autocast(device_type='cuda', dtype=dtype):
                    logits, (loss, main_loss, aux_loss), _ = model(x, targets=y)
                    loss = loss / config.gradient_accumulation_steps

                accumulated_loss += loss.detach().float()
                if main_loss is not None:
                    accumulated_main_loss += main_loss.detach().float() / config.gradient_accumulation_steps
                if aux_loss is not None:
                    accumulated_aux_loss += aux_loss.detach().float() / config.gradient_accumulation_steps

                loss.backward()
        
        # All-reduce and clipping for gradients
        if config.backend == 'fsdp':
             grad_norm = model.clip_grad_norm_(config.grad_clip)
        else:
             grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

        optimizer.step()
        optimizer.zero_grad()

        if is_master_process:
            pbar.update(1)
            if current_step % config.log_interval == 0:
                log_loss_main = accumulated_main_loss.item()
                try:
                    perplexity = math.exp(log_loss_main)
                except (OverflowError, ValueError):
                    perplexity = float('inf')

                peak_mem_gb = torch.cuda.max_memory_allocated(device) / 1e9
                postfix_data = {
                    "loss": f"{accumulated_loss.item():.3f}",
                    "loss_main": f"{log_loss_main:.3f}",
                    "loss_aux": f"{accumulated_aux_loss.item():.4f}",
                    "ppl": f"{perplexity:.2f}",
                    "lr": f"{lr:.2e}",
                    "gnorm": f"{grad_norm.item():.2f}",
                    "mem_gb": f"{peak_mem_gb:.2f}"
                }
                pbar.set_postfix(postfix_data)

                if config.wandb_project:
                    wandb.log({
                        "step": current_step,
                        "epoch": current_epoch,
                        "loss/total": accumulated_loss.item(),
                        "loss/main": log_loss_main,
                        "loss/aux": accumulated_aux_loss.item(),
                        "perplexity": perplexity,
                        "lr": lr,
                        "grad_norm": grad_norm.item(),
                        "memory/peak_gpu_gb": peak_mem_gb,
                    })

            if current_step > 0 and current_step % config.save_interval == 0:
                # --- FSDP Checkpoint Saving Logic ---
                # 1. Save efficient sharded checkpoint (primary)
                sharded_save_dir = os.path.join(config.out_dir, f"ckpt_{current_step}_sharded")
                latest_sharded_save_dir = os.path.join(config.out_dir, "latest_sharded_checkpoint")

                writer = FileSystemWriter(sharded_save_dir)
                state_dict = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
                save_state_dict(state_dict, writer)
                # Save metadata on master process
                if is_master_process:
                    torch.save({'step': current_step, 'epoch': current_epoch}, os.path.join(sharded_save_dir, "metadata.pt"))
                    # Symlink for `latest`
                    if os.path.lexists(latest_sharded_save_dir):
                        os.remove(latest_sharded_save_dir)
                    os.symlink(os.path.basename(sharded_save_dir), latest_sharded_save_dir)
                    print(f"\n[CHECKPOINT] Saved SHARDED checkpoint to {sharded_save_dir}")

                # 2. Save a full, consolidated checkpoint for debugging/transfer
                full_save_path = os.path.join(config.out_dir, f"ckpt_{current_step}_full.pt")
                latest_full_save_path = os.path.join(config.out_dir, "latest_full_checkpoint.pt")
                
                with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
                     cpu_state = model.state_dict()
                     if is_master_process:
                         checkpoint = {
                             'model': cpu_state,
                             'optimizer': optimizer.state_dict(), # Note: This is rank-specific, full optim state requires more work not requested.
                             'config': vars(config),
                             'step': current_step,
                             'epoch': current_epoch,
                         }
                         torch.save(checkpoint, full_save_path)
                         # Symlink for `latest`
                         if os.path.lexists(latest_full_save_path):
                            os.remove(latest_full_save_path)
                         os.symlink(os.path.basename(full_save_path), latest_full_save_path)
                         print(f"[CHECKPOINT] Saved FULL state checkpoint to {full_save_path}")


    if is_master_process:
        print("\nMax steps reached. Finishing training.")
        pbar.close()
        if config.wandb_project:
            wandb.finish()
    if is_dist:
        destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a LunarisCodex-MoE model with FSDP.")
    parser.add_argument("config", type=str, help="Path to the MoE config.yaml file.")
    parser.add_argument("--backend", type=str, default="fsdp", choices=["fsdp", "ddp"], help="Distributed backend to use.")
    args = parser.parse_args()
    train(args.config, args.backend)
