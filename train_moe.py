"""
Main Training Script for the LunarisCodex Language Model (MoE Version).

This script is adapted to train the Mixture-of-Experts (MoE) version of the model.
It handles the specific requirements of MoE training, such as the auxiliary load
balancing loss and logging expert utilization.

The training loop is designed to unpack the model's output, which includes the
total loss, the main cross-entropy loss, and the auxiliary loss. These components
are logged separately to monitor both model performance and expert load balancing,
which is a critical aspect of stable MoE training.

Perplexity is calculated using only the main cross-entropy loss to ensure a
meaningful measure of the model's predictive performance. Additionally, the script
captures expert routing decisions and logs a utilization bar chart to Weights & Biases
to provide a clear visualization of how tokens are distributed across experts.
"""

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

# Import the MoE-enabled model and configuration from the corresponding model file.
from model_moe import LunarisCodex, LunarisCodexConfig

# The TrainConfig class itself requires no major changes, as the `from_yaml`
# classmethod will automatically load MoE parameters if they are present in the YAML file.
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
    wandb_project: Optional[str] = "lunaris-codex-moe" # Suggested W&B project for the MoE experiment
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = f"run-moe-{time.strftime('%Y-%m-%d-%H-%M')}"

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
        if 'aux_loss_weight' not in model_config_dict: # Add the default value if it's missing from the yaml
             model_config.aux_loss_weight = 1e-2
        int_fields = ['warmup_steps', 'max_steps', 'batch_size', 'gradient_accumulation_steps', 'num_epochs', 'save_interval', 'log_interval']
        for key in float_fields:
            if key in config_dict:
                config_dict[key] = float(config_dict[key])
        for key in int_fields:
            if key in config_dict:
                config_dict[key] = int(config_dict[key])
        return cls(**config_dict)

# The Dataset and other utility functions remain the same.
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

def setup_ddp():
    is_ddp = 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1
    if is_ddp:
        init_process_group("nccl")
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        torch.cuda.set_device(local_rank)
        print(f"[DDP] Setup complete: rank {rank}, world_size {world_size}")
        return True, rank, world_size
    return False, 0, 1

def get_lr(step, config: TrainConfig):
    if step < config.warmup_steps:
        return config.learning_rate * step / config.warmup_steps
    if step >= config.max_steps:
        return config.learning_rate * 0.01
    decay_ratio = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
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
        print(" " * 10 + "LUNARIS CODEX MoE TRAINING")
        print("-" * 50)
        print(f"Model Config: {config.model}")
        if config.model.n_experts:
             print(f"--> MoE Enabled: {config.model.n_experts} experts, aux_loss_weight={config.model.aux_loss_weight:.4f}")
        print(f"Data: {config.data_dir}, SeqLen: {config.sequence_length}")
        print("-" * 50)

    if is_master_process and config.wandb_project:
        import wandb
        wandb.init(project=config.wandb_project, entity=config.wandb_entity, name=config.wandb_run_name, config=config.__dict__)

    train_dataset = ShardDataset(data_dir=config.data_dir, sequence_length=config.sequence_length)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True) if is_ddp else None
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)

    model = LunarisCodex(config.model).to(config.device)

    if config.compile_model:
        if is_master_process: print("[MODEL] Compiling model...")
        model = torch.compile(model)
    if is_ddp:
        model = DDP(model, device_ids=[int(os.environ['LOCAL_RANK'])])

    raw_model = model.module if is_ddp else model
    optimizer = raw_model.configure_optimizers(
        weight_decay=config.weight_decay,
        learning_rate=config.learning_rate,
        betas=(config.beta1, config.beta2),
        device_type=device_type
    )

    current_step, current_epoch = 0, 0
    checkpoint_path = os.path.join(config.out_dir, "latest_checkpoint.pt")
    if os.path.exists(checkpoint_path):
        if is_master_process: print(f"[SETUP] Resuming from checkpoint: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=config.device)
        raw_model.load_state_dict(unwrap_model_keys(state['model']))
        optimizer.load_state_dict(state['optimizer'])
        current_step = state['step']
        current_epoch = state.get('epoch', 0)

    optimizer.zero_grad(set_to_none=True)
    if is_ddp:
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

        # Initialize accumulators for the different loss components.
        # This is necessary for accurate logging across gradient accumulation steps.
        accumulated_loss = 0.0
        accumulated_main_loss = 0.0
        accumulated_aux_loss = 0.0
        expert_indices_list = None # Ensure the variable exists outside the loop.

        for micro_step in range(config.gradient_accumulation_steps):
            is_last_micro_step = (micro_step == config.gradient_accumulation_steps - 1)
            # When using DDP with gradient accumulation, we only want to sync gradients
            # on the final micro-step. `model.no_sync()` prevents DDP from reducing
            # gradients across processes on all but the last micro-step.
            ddp_context = model.no_sync() if is_ddp and not is_last_micro_step else nullcontext()

            with ddp_context:
                try:
                    x, y = next(data_iter)
                except StopIteration:
                    current_epoch += 1
                    if is_ddp: train_loader.sampler.set_epoch(current_epoch)
                    data_iter = iter(train_loader)
                    x, y = next(data_iter)

                x, y = x.to(config.device, non_blocking=True), y.to(config.device, non_blocking=True)

                with ctx:
                    # The MoE model returns a tuple for loss: (total_loss, main_loss, aux_loss).
                    # We unpack it here. The `expert_indices_list` is also captured for logging.
                    logits, (loss, main_loss, aux_loss), _, expert_indices_list = model(x, targets=y, past_key_values=None)
                    # Scale the loss to account for gradient accumulation.
                    loss = loss / config.gradient_accumulation_steps

                # Accumulate each loss component separately for accurate logging.
                accumulated_loss += loss.item()
                if main_loss is not None:
                    # The main_loss is the raw cross-entropy loss for this micro-batch.
                    # We scale it by the number of accumulation steps before adding it to the accumulator.
                    accumulated_main_loss += main_loss.item() / config.gradient_accumulation_steps
                if aux_loss is not None:
                    # The same applies to the auxiliary loss.
                    accumulated_aux_loss += aux_loss.item() / config.gradient_accumulation_steps

                loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if is_master_process:
            pbar.update(1)
            if current_step % config.log_interval == 0:
                # Perplexity is calculated only from the main cross-entropy loss, as the
                # auxiliary loss is for regularization and not a measure of predictive performance.
                log_loss_main = accumulated_main_loss
                try:
                    perplexity = math.exp(log_loss_main)
                except (OverflowError, ValueError):
                    perplexity = float('inf')

                # Log the separate loss components to the progress bar for real-time monitoring.
                postfix_data = {
                    "loss": f"{accumulated_loss:.3f}",
                    "loss_main": f"{log_loss_main:.3f}",
                    "loss_aux": f"{accumulated_aux_loss:.4f}",
                    "ppl": f"{perplexity:.2f}",
                    "lr": f"{lr:.2e}",
                    "gnorm": f"{grad_norm.item():.2f}"
                }
                pbar.set_postfix(postfix_data)

                if config.wandb_project:
                    # Create a single dictionary for all W&B logs for this step.
                    log_data = {
                        "step": current_step,
                        "epoch": current_epoch,
                        "loss/total": accumulated_loss,
                        "loss/main": log_loss_main,
                        "loss/aux": accumulated_aux_loss,
                        "perplexity": perplexity,
                        "lr": lr,
                        "grad_norm": grad_norm.item(),
                    }

                    # Check if expert routing data is available for logging.
                    if expert_indices_list:
                        # For visualization, we'll focus on the expert choices in the first MoE layer.
                        first_moe_layer_indices = expert_indices_list[0].detach().cpu()

                        # Count the number of tokens routed to each expert.
                        num_experts = raw_model.config.n_experts
                        expert_counts = torch.bincount(first_moe_layer_indices.view(-1), minlength=num_experts)

                        # Create a wandb.Table to be plotted as a bar chart.
                        table = wandb.Table(columns=["Expert ID", "Token Count"])
                        for i in range(num_experts):
                            table.add_data(f"Expert {i}", expert_counts[i].item())

                        # Add the bar chart plot to our logging dictionary.
                        log_data["expert_utilization/layer_0"] = wandb.plot.bar(
                            table, "Expert ID", "Token Count", title="Expert Utilization (Layer 0)"
                        )

                    # Log all metrics for this step to W&B at once.
                    wandb.log(log_data)

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

    if is_master_process:
        print("\nMax steps reached. Finishing training.")
        pbar.close()
        if config.wandb_project:
            wandb.finish()
    if is_ddp:
        destroy_process_group()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Train a LunarisCodex-MoE model.")
    parser.add_argument("config", type=str, help="Path to the MoE config.yaml file.")
    args = parser.parse_args()
    train(args.config)
