"""
Main Training Script for the LunarisCodex Language Model (FSDP + Optimized Switch-MoE)

- FSDP with FULL_SHARD (ZeRO-3), per-Block auto-wrap.
- BF16 mixed precision (H100/GH200), AMP-enabled forward.
- Activation checkpointing per Block (non-reentrant).
- torch.compile integration with fallback.
- Sharded and full checkpoints; robust resume from either.
- Optimizer with router-specific LR group (matches model_moe).
- W&B logging: losses, ppl, grad-norm, GPU mem, expert utilization and drop rate.

Usage:
    torchrun --nproc_per_node=NUM_GPUS train_moe_fsdp.py config.yaml --backend fsdp
"""

import os
import time
import math
import glob
import yaml
import argparse
from dataclasses import dataclass, field, asdict
from typing import Optional, Type
from contextlib import nullcontext
from functools import partial

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.distributed import init_process_group, destroy_process_group, get_rank, get_world_size

from tqdm import tqdm

# FSDP imports
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    StateDictType,
    FullStateDictConfig,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter, load_state_dict, save_state_dict

# Model (new architecture)
from model_moe import LunarisCodex, LunarisCodexConfig, Block, compile_model_if_available


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
    wandb_project: Optional[str] = "lunaris-codex-fsdp-moe"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    backend: str = "fsdp"  # "fsdp" or "ddp"

    @property
    def sequence_length(self):
        return self.model.max_seq_len

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        model_config_dict = config_dict.pop("model", {})
        model_config = LunarisCodexConfig(**model_config_dict)

        # Backward compat defaults
        if model_config.n_experts and getattr(model_config, "aux_loss_weight", None) is None:
            model_config.aux_loss_weight = 1e-2
        if getattr(model_config, "capacity_factor", None) is None:
            model_config.capacity_factor = 1.25
        if getattr(model_config, "router_z_loss_weight", None) is None:
            model_config.router_z_loss_weight = 1e-3

        config_dict['model'] = model_config
        # Normalize numerics
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


class ShardDataset(Dataset):
    """
    Memory-efficient dataset over .npy token shards. Produces (x, y) of length seq_len.
    Pads last sample with -1 (ignored in CE).
    """
    def __init__(self, data_dir: str, sequence_length: int):
        super().__init__()
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.shards = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
        if not self.shards:
            raise ValueError(f"No .npy files found in directory: {data_dir}")
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
            remaining = self.shard_lengths[shard_idx] - local_start_idx
            seq_part1 = self.mmap_shards[shard_idx][local_start_idx: local_start_idx + remaining]
            need = seq_len_with_target - remaining
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


def setup_distributed(backend: str):
    is_dist = 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1
    if not is_dist:
        return False, 0, 1, 0
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    init_process_group("nccl")
    rank = get_rank()
    world_size = get_world_size()
    print(f"[DIST] Backend: {backend.upper()} | Rank {rank}/{world_size} | Local rank {local_rank}")
    return True, rank, world_size, local_rank


def get_lr(step, config: TrainConfig):
    if step < config.warmup_steps:
        return config.learning_rate * step / max(1, config.warmup_steps)
    if step >= config.max_steps:
        return config.learning_rate * 0.01
    decay_ratio = (step - config.warmup_steps) / max(1, (config.max_steps - config.warmup_steps))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return (config.learning_rate * 0.01) + coeff * (config.learning_rate * 0.99)


def apply_activation_checkpointing(model, block_class: Type[torch.nn.Module]):
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper,
        CheckpointImpl,
        apply_activation_checkpointing as apply_ac,
    )
    non_reentrant = partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT)
    check_fn = lambda m: isinstance(m, block_class)
    apply_ac(model, checkpoint_wrapper_fn=non_reentrant, check_fn=check_fn)


def configure_fsdp_model(config: TrainConfig) -> FSDP:
    # Mixed precision config
    bf16 = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16)

    # Auto-wrap Blocks
    policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={Block})

    # Construct on meta device to reduce peak memory on rank0
    with torch.device('meta'):
        base = LunarisCodex(config.model)

    # Activation checkpointing before wrap
    apply_activation_checkpointing(base, block_class=Block)

    # Wrap with FSDP
    fsdp_model = FSDP(
        base,
        auto_wrap_policy=policy,
        mixed_precision=bf16,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        use_orig_params=True,
    )
    return fsdp_model


def build_optimizer(model: torch.nn.Module, config: TrainConfig, device_type: str):
    # Use the model's optimizer configuration for router param group handling
    # We must call it on the non-FSDP-wrapped module; however FSDP exposes .named_parameters similarly.
    # To preserve param-group strategy, delegate to the model method if possible:
    try:
        raw = model
        while hasattr(raw, "module"):
            raw = raw.module
        opt = raw.configure_optimizers(
            weight_decay=config.weight_decay,
            learning_rate=config.learning_rate,
            betas=(config.beta1, config.beta2),
            device_type=device_type,
        )
        return opt
    except Exception:
        # Fallback: fused AdamW over all params
        fused_available = 'fused' in torch.optim.AdamW.__init__.__code__.co_varnames
        use_fused = fused_available and device_type == 'cuda'
        return torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                 betas=(config.beta1, config.beta2),
                                 weight_decay=config.weight_decay, fused=use_fused)


def save_checkpoints(config: TrainConfig, model, optimizer, step: int, epoch: int, is_master: bool):
    # 1) Primary: SHARDED checkpoint
    save_dir = os.path.join(config.out_dir, f"ckpt_{step}_sharded")
    latest_dir = os.path.join(config.out_dir, "latest_sharded_checkpoint")
    writer = FileSystemWriter(save_dir)
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
    save_state_dict(state, writer)
    if is_master:
        torch.save({'step': step, 'epoch': epoch}, os.path.join(save_dir, "metadata.pt"))
        if os.path.lexists(latest_dir):
            os.remove(latest_dir)
        os.symlink(os.path.basename(save_dir), latest_dir)
        print(f"[CKPT] Saved SHARDED checkpoint: {save_dir}")

    # 2) FULL CPU checkpoint (rank0 only)
    full_path = os.path.join(config.out_dir, f"ckpt_{step}_full.pt")
    latest_full = os.path.join(config.out_dir, "latest_full_checkpoint.pt")
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT,
                              FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
        cpu_state = model.state_dict()
    if is_master:
        checkpoint = {
            'model': cpu_state,
            'optimizer': optimizer.state_dict(),  # rank0 view; full optim state not consolidated
            'config': asdict(config),
            'step': step,
            'epoch': epoch,
        }
        torch.save(checkpoint, full_path)
        if os.path.lexists(latest_full):
            os.remove(latest_full)
        os.symlink(os.path.basename(full_path), latest_full)
        print(f"[CKPT] Saved FULL checkpoint: {full_path}")


def try_resume(config: TrainConfig, model, optimizer, rank: int):
    # Prefer SHARDED resume
    latest_sharded = os.path.join(config.out_dir, "latest_sharded_checkpoint")
    latest_full = os.path.join(config.out_dir, "latest_full_checkpoint.pt")
    step, epoch = 0, 0

    if os.path.exists(latest_sharded):
        # Resolve symlink to directory
        if os.path.islink(latest_sharded):
            target = os.readlink(latest_sharded)
            sharded_dir = os.path.join(config.out_dir, target)
        else:
            sharded_dir = latest_sharded
        print(f"[RESUME] Rank {rank} loading SHARDED: {sharded_dir}")
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
        load_state_dict(state, FileSystemReader(sharded_dir))
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        meta_path = os.path.join(sharded_dir, "metadata.pt")
        if os.path.exists(meta_path):
            meta = torch.load(meta_path, map_location='cpu')
            step = int(meta.get('step', 0))
            epoch = int(meta.get('epoch', 0))
        return step, epoch

    if os.path.exists(latest_full):
        print(f"[RESUME] Rank {rank} loading FULL: {latest_full}")
        state = torch.load(latest_full, map_location='cpu')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        step = int(state.get('step', 0))
        epoch = int(state.get('epoch', 0))
        return step, epoch

    return step, epoch


def train(config_path: str, backend_override: str):
    config = TrainConfig.from_yaml(config_path)
    if backend_override:
        config.backend = backend_override

    is_dist, rank, world_size, local_rank = setup_distributed(config.backend)
    is_master = (rank == 0)

    torch.manual_seed(1337 + rank)
    np.random.seed(1337 + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device_type = 'cuda' if 'cuda' in config.device else 'cpu'
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=amp_dtype) if device_type == 'cuda' else nullcontext()

    if is_master:
        os.makedirs(config.out_dir, exist_ok=True)
        if config.wandb_run_name is None:
            config.wandb_run_name = f"run-fsdp-moe-{time.strftime('%Y-%m-%d-%H-%M')}"
        print("-" * 60)
        print(f"LUNARIS CODEX FSDP+MoE Training | Backend={config.backend.upper()}")
        print("-" * 60)
        print(f"Model: {config.model}")
        if config.model.n_experts:
            print(f"MoE: experts={config.model.n_experts}, cap={config.model.capacity_factor}, "
                  f"aux={config.model.aux_loss_weight}, z={config.model.router_z_loss_weight}")
        print(f"Data: {config.data_dir}, SeqLen={config.sequence_length}, Device={config.device}, bf16={use_bf16}")
        print(f"Batch={config.batch_size}, Accum={config.gradient_accumulation_steps}, LR={config.learning_rate}")
        print("-" * 60)

    if is_master and config.wandb_project:
        import wandb
        wandb.init(project=config.wandb_project, entity=config.wandb_entity,
                   name=config.wandb_run_name, config=asdict(config))

    # Data
    dataset = ShardDataset(data_dir=config.data_dir, sequence_length=config.sequence_length)
    if is_dist:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, rank=rank, num_replicas=world_size, shuffle=True)
    else:
        sampler = None
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=(config.num_workers > 0 and config.persistent_workers),
        prefetch_factor=(config.prefetch_factor if config.num_workers > 0 else None),
        drop_last=True,
    )

    # Model
    if config.backend == 'fsdp':
        model = configure_fsdp_model(config)
    else:
        # DDP fallback
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = LunarisCodex(config.model).to(config.device, dtype=torch.bfloat16 if use_bf16 else torch.float32)
        if is_dist:
            model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # torch.compile (safe try)
    if config.compile_model and device_type == 'cuda':
        if is_master:
            print("[MODEL] Attempting torch.compile ...")
        try:
            model = compile_model_if_available(model)
        except Exception as e:
            if is_master:
                print(f"[WARN] torch.compile failed, continuing without: {e}")

    # Optimizer with router group handling (delegated to model method when possible)
    optimizer = build_optimizer(model, config, device_type=device_type)

    # Resume
    step, epoch = try_resume(config, model, optimizer, rank)
    if is_dist and sampler is not None:
        sampler.set_epoch(epoch)

    if is_master:
        print(f"[TRAIN] Starting at step {step} -> {config.max_steps}")
        pbar = tqdm(total=config.max_steps, desc="Steps", initial=step, ncols=120)

    # Training loop
    model.train()
    data_iter = iter(loader)

    while step < config.max_steps:
        step += 1
        lr = get_lr(step, config)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        accum_total = 0.0
        accum_main = 0.0
        accum_aux = 0.0
        first_layer_indices = None
        first_layer_keep = None

        for micro in range(config.gradient_accumulation_steps):
            try:
                x, y = next(data_iter)
            except StopIteration:
                epoch += 1
                if is_dist and sampler is not None:
                    sampler.set_epoch(epoch)
                data_iter = iter(loader)
                x, y = next(data_iter)

            x = x.to(config.device, non_blocking=True)
            y = y.to(config.device, non_blocking=True)

            with autocast_ctx:
                outputs = model(x, targets=y, past_key_values=None)
            logits, loss_tuple, _, aux_list = outputs
            total_loss, main_loss, aux_loss = loss_tuple
            total_loss = total_loss / config.gradient_accumulation_steps

            accum_total += float(total_loss.item())
            accum_main += float(main_loss.item()) / config.gradient_accumulation_steps
            accum_aux += float(aux_loss.item()) / config.gradient_accumulation_steps

            if aux_list is not None and isinstance(aux_list, list) and len(aux_list) == 2:
                indices_list, keep_masks_list = aux_list
                if indices_list and first_layer_indices is None:
                    first_layer_indices = indices_list[0].detach()
                if keep_masks_list and first_layer_keep is None:
                    first_layer_keep = keep_masks_list[0].detach()

            total_loss.backward()

        # Clip gradients
        if config.backend == 'fsdp':
            grad_norm = model.clip_grad_norm_(config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if is_master:
            pbar.update(1)
            if step % config.log_interval == 0:
                ppl = math.exp(accum_main) if accum_main < 20 else float('inf')
                peak_gb = torch.cuda.max_memory_allocated(device=config.device) / 1e9
                pbar.set_postfix({
                    "loss": f"{accum_total:.3f}",
                    "main": f"{accum_main:.3f}",
                    "aux": f"{accum_aux:.4f}",
                    "ppl": f"{ppl:.2f}",
                    "lr": f"{lr:.2e}",
                    "gnorm": f"{float(grad_norm):.2f}",
                    "mem_gb": f"{peak_gb:.2f}",
                })

                if config.wandb_project:
                    import wandb
                    log = {
                        "step": step,
                        "epoch": epoch,
                        "loss/total": accum_total,
                        "loss/main": accum_main,
                        "loss/aux": accum_aux,
                        "perplexity": ppl,
                        "lr": lr,
                        "grad_norm": float(grad_norm),
                        "memory/peak_gpu_gb": peak_gb,
                    }
                    if first_layer_indices is not None and (config.model.n_experts or 0) > 0:
                        num_exp = config.model.n_experts
                        counts = torch.bincount(first_layer_indices.view(-1), minlength=num_exp)
                        util = counts.float() / counts.sum().clamp_min(1)
                        log.update({f"experts/util_layer0/e{i}": util[i].item() for i in range(num_exp)})
                    if first_layer_keep is not None:
                        keep_frac = first_layer_keep.float().mean().item()
                        log["experts/drop_rate_layer0"] = 1.0 - keep_frac
                    wandb.log(log)

        # Checkpointing
        if is_master and step % config.save_interval == 0:
            save_checkpoints(config, model, optimizer, step, epoch, is_master)
            if config.save_latest_always:
                # update latest symlinks are handled in save_* functions
                pass

    # Finalize
    if is_master:
        print("\n[TRAIN] Max steps reached. Finishing.")
        pbar.close()
        if config.wandb_project:
            import wandb
            wandb.finish()
    if is_dist:
        destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a LunarisCodex-MoE model with FSDP (optimized).")
    parser.add_argument("config", type=str, help="Path to the MoE config.yaml file.")
    parser.add_argument("--backend", type=str, default="fsdp", choices=["fsdp", "ddp"], help="Distributed backend.")
    args = parser.parse_args()
    train(args.config, args.backend)
