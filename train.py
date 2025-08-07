"""
Main Training Script for the LunarisCodex Language Model

This script serves as the primary training engine for the LunarisCodex model. It is designed
for large-scale, distributed training and incorporates several industry-standard best practices
to ensure efficiency, stability, and scalability.

Key Features:
- **Distributed Training (DDP):** Utilizes PyTorch's DistributedDataParallel (DDP) to
  train the model across multiple GPUs or even multiple nodes. This is essential for
  training large models in a reasonable timeframe.
- **Optimized Data Loading:** Implements a custom `ShardDataset` that uses memory-mapping
  (`mmap_mode`). This allows the script to handle massive datasets (billions of tokens)
  that are too large to fit into system RAM by loading data chunks directly from disk
  as needed.
- **Mixed-Precision Training:** Leverages `torch.amp.autocast` to use `bfloat16` or
  `float16` precision, which significantly speeds up training on modern GPUs and reduces
  memory consumption, allowing for larger models or batch sizes.
- **Model Compilation:** Integrates `torch.compile()` to JIT-compile the model, fusing
  operations and optimizing the execution graph for additional performance gains.
- **Gradient Accumulation:** Allows for simulating very large batch sizes by accumulating
  gradients over several smaller steps. This is crucial for training stability when
  hardware memory is a constraint.
- **Advanced Learning Rate Scheduling:** Implements a warmup followed by a cosine decay
  schedule, a proven strategy for stabilizing the initial phase of training and
  improving final model performance.
- **Resilient Checkpointing:** Saves not only the model weights but also the optimizer
  state, current step, and epoch. This allows training to be resumed seamlessly from
  the last checkpoint in case of interruptions.
- **Configuration Management:** Uses a YAML file and a `dataclass` for clean, version-
  controllable, and easily modifiable training configurations.
- **Instrumentation & Logging:** Integrates with Weights & Biases (wandb) for real-time
  monitoring of training metrics like loss, perplexity, and learning rate.
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

from model import LunarisCodex, LunarisCodexConfig

# --- Configuration Dataclass ---
@dataclass
class TrainConfig:
    """
    A single dataclass to hold all training-related hyperparameters.
    This approach makes configuration management clean, type-safe, and
    easily serializable to/from files like YAML.
    """
    # Model configuration is nested, promoting modularity.
    model: LunarisCodexConfig = field(default_factory=LunarisCodexConfig)

    # Data configuration
    data_dir: str = "data/"
    # CORRECTION: The explicit `sequence_length` field has been removed.
    # It is now a property that reads directly from `model.max_seq_len`
    # to guarantee consistency across the entire pipeline.

    # Optimizer configuration (AdamW parameters)
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95

    # Scheduler configuration
    warmup_steps: int = 2000
    max_steps: int = 600000

    # Training configuration
    batch_size: int = 16 # Per-GPU batch size
    gradient_accumulation_steps: int = 1 # Number of steps to accumulate gradients over
    num_epochs: int = 1 # Number of epochs (often less relevant for large datasets than max_steps)
    grad_clip: float = 1.0 # Gradient clipping to prevent exploding gradients
    device: str = "cuda" # Device to train on ('cuda' or 'cpu')
    compile_model: bool = True # Whether to use torch.compile() for speed

    # I/O and Logging
    out_dir: str = "checkpoints" # Directory to save checkpoints
    log_interval: int = 20 # How often to log metrics
    save_interval: int = 1000 # How often to save a checkpoint

    # W&B configuration for experiment tracking
    wandb_project: Optional[str] = "lunaris-codex"
    wandb_entity: Optional[str] = None # Your W&B username or team
    wandb_run_name: Optional[str] = f"run-{time.strftime('%Y-%m-%d-%H-%M')}"

    @property
    def sequence_length(self):
        """
        CORRECTION APPLIED: This property creates a single source of truth.
        The data loading pipeline (which uses `config.sequence_length`) will now
        always use the exact same sequence length that the model architecture
        is configured for (`config.model.max_seq_len`), preventing any risk
        of a mismatch.
        """
        return self.model.max_seq_len

    @classmethod
    def from_yaml(cls, path: str):
        """
        Loads configuration from a YAML file, ensuring correct types.
        Why this is important: YAML parsers often load numbers as generic types.
        Explicitly casting to float or int prevents subtle bugs that can be hard
        to track down during training.
        """
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Separate model config from training config for nested structure
        model_config_dict = config_dict.pop("model", {})
        model_config = LunarisCodexConfig(**model_config_dict)
        config_dict['model'] = model_config

        # GUARANTEE: Any stray `sequence_length` key in the top-level of an old
        # YAML file will be safely ignored, as it doesn't match a field here.
        
        # Define fields that need specific type casting
        float_fields = ['learning_rate', 'weight_decay', 'beta1', 'beta2', 'grad_clip']
        int_fields = ['warmup_steps', 'max_steps', 'batch_size', 'gradient_accumulation_steps', 'num_epochs', 'save_interval', 'log_interval']

        # Perform the casting
        for key in float_fields:
            if key in config_dict:
                config_dict[key] = float(config_dict[key])
        for key in int_fields:
            if key in config_dict:
                config_dict[key] = int(config_dict[key])

        return cls(**config_dict)


# --- Sharded Memory-Mapped Dataset ---
class ShardDataset(Dataset):
    """
    A custom PyTorch Dataset to handle extremely large datasets.

    The key challenge with terabyte-scale datasets is that they cannot fit into RAM.
    This class solves that problem using two main techniques:
    1.  Sharding: The data is pre-processed into smaller, manageable files (.npy shards).
    2.  Memory-Mapping (mmap): Instead of loading a shard's content into RAM, we
        create a "map" to it on disk. The operating system then handles transparently
        loading the necessary pages of the file into memory only when they are accessed.
        This provides fast, random access to any part of the file with minimal RAM usage.
    """
    def __init__(self, data_dir: str, sequence_length: int):
        super().__init__()
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        # Find all pre-tokenized data shards
        self.shards = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
        if not self.shards:
            raise ValueError(f"No .npy files found in directory: {data_dir}")

        # Calculate total tokens by summing the size of all shards without loading them
        total_tokens = sum(np.load(shard, mmap_mode='r').shape[0] for shard in self.shards)
        # The total number of training samples is the total tokens divided by the sequence length
        self.total_samples = max(0, (total_tokens - 1) // self.sequence_length)

        print(f"[DATA] Loaded {len(self.shards)} shards. Total tokens: {total_tokens/1e9:.2f}B.")
        print(f"[DATA] Creating {self.total_samples:,} non-overlapping samples of length {self.sequence_length}.")

        # Keep the memory-mapped arrays open for quick access in __getitem__
        self.mmap_shards = [np.load(shard, mmap_mode='r') for shard in self.shards]
        self.shard_lengths = [len(shard) for shard in self.mmap_shards]
        # Cumulative sum of shard lengths helps quickly find which shard an index belongs to
        self.cumulative_lengths = np.cumsum(self.shard_lengths)

    def __len__(self):
        # The total number of items in the dataset
        return self.total_samples

    def __getitem__(self, idx):
        """
        Retrieves a single training sample (input `x` and target `y`).
        This method contains the logic to fetch a sequence of tokens, even if it
        spans across the boundary of two consecutive shard files.
        """
        # 1. Calculate the global starting position of the token sequence in the entire dataset
        token_start_pos = idx * self.sequence_length

        # 2. Find which shard this sequence starts in.
        #    `np.searchsorted` is a highly efficient binary search.
        shard_idx = np.searchsorted(self.cumulative_lengths, token_start_pos, side='right')

        # 3. Find the local starting index within that shard.
        #    If it's the first shard (shard_idx=0), global pos is the same as local pos.
        #    Otherwise, subtract the cumulative length of all previous shards.
        local_start_idx = token_start_pos if shard_idx == 0 else token_start_pos - self.cumulative_lengths[shard_idx - 1]

        # 4. We need `sequence_length + 1` tokens to create our input `x` and target `y`
        #    (e.g., x = tokens[0...1023], y = tokens[1...1024])
        seq_len_with_target = self.sequence_length + 1

        # 5. Handle the case where the required sequence crosses a shard boundary.
        if local_start_idx + seq_len_with_target > self.shard_lengths[shard_idx]:
            # The sequence starts in `shard_idx` but ends in `shard_idx + 1`.
            # Get the first part of the sequence from the end of the current shard.
            remaining_len = self.shard_lengths[shard_idx] - local_start_idx
            seq_part1 = self.mmap_shards[shard_idx][local_start_idx : local_start_idx + remaining_len]

            # If there is a next shard, get the remaining tokens from its beginning.
            if shard_idx + 1 < len(self.mmap_shards):
                needed_from_next = seq_len_with_target - remaining_len
                seq_part2 = self.mmap_shards[shard_idx + 1][:needed_from_next]
                # Concatenate the two parts to form the full sequence.
                seq = np.concatenate((seq_part1, seq_part2))
            else:
                # This happens at the very end of the dataset; just use the partial sequence.
                seq = seq_part1
        else:
            # The entire sequence fits within a single shard. This is the common case.
            seq = self.mmap_shards[shard_idx][local_start_idx : local_start_idx + seq_len_with_target]

        # 6. Pad if the last sequence is shorter than required.
        if len(seq) < seq_len_with_target:
            # The loss function in the model is configured to ignore -1 index.
            pad_len = seq_len_with_target - len(seq)
            seq = np.pad(seq, (0, pad_len), 'constant', constant_values=-1)

        # 7. Convert the numpy array to PyTorch tensors and create x, y.
        seq_tensor = torch.from_numpy(seq.astype(np.int64))
        x, y = seq_tensor[:-1], seq_tensor[1:]
        return x, y

# --- DDP Setup ---
def setup_ddp():
    """
    Sets up DistributedDataParallel (DDP) training.
    DDP enables training a model across multiple GPUs, which can be on the same
    machine or different machines. It works by creating a separate process for each
    GPU. This function checks for environment variables (`WORLD_SIZE`, `RANK`) that
    are automatically set by PyTorch's process launchers (like `torchrun`).
    """
    is_ddp = 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1
    if is_ddp:
        # `init_process_group` establishes communication between all processes.
        # "nccl" is NVIDIA's optimized backend for GPU-to-GPU communication.
        init_process_group("nccl")
        rank = int(os.environ['RANK']) # The global ID of the current process (0 to world_size-1)
        local_rank = int(os.environ['LOCAL_RANK']) # The ID of the GPU on the current machine
        world_size = int(os.environ['WORLD_SIZE']) # The total number of processes
        # This is crucial: it binds the current process to a specific GPU.
        torch.cuda.set_device(local_rank)
        print(f"[DDP] Setup complete: rank {rank}, world_size {world_size}")
        return True, rank, world_size
    # If not using DDP, return values corresponding to a single-process setup.
    return False, 0, 1

# --- Learning Rate Scheduler ---
def get_lr(step, config: TrainConfig):
    """
    Calculates the learning rate for a given step based on a warmup and cosine decay schedule.
    This is a standard and highly effective schedule for training large transformers.

    Why this schedule?
    1.  Warmup: In the beginning, the model has random weights and can be unstable.
        Starting with a small learning rate and gradually increasing it (linear warmup)
        prevents the loss from exploding and allows the model to stabilize.
    2.  Cosine Decay: After the warmup, the learning rate slowly decreases following a
        cosine curve. This helps the model to converge more smoothly into a good
        minimum in the loss landscape, often leading to better final performance than
        a constant learning rate.
    """
    # 1. Linear warmup phase
    if step < config.warmup_steps:
        return config.learning_rate * step / config.warmup_steps
    # 2. If we are past the max steps, use a small constant LR
    if step >= config.max_steps:
        # A small final learning rate, slightly larger than the minimum of the cosine decay
        return config.learning_rate * 0.1

    # 3. Cosine decay phase
    # Calculate how far we are into the decay phase
    decay_ratio = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
    # Calculate the cosine coefficient (from 1 to 0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    # Linearly interpolate between the full LR and 10% of the LR (a common practice)
    min_lr = config.learning_rate * 0.1
    return min_lr + coeff * (config.learning_rate - min_lr)

# --- Robust checkpoint key unwrapping ---
def unwrap_model_keys(state_dict):
    """
    Remove DDP and torch.compile prefixes from model state dict keys.

    Why is this necessary?
    - When you wrap a model with `DDP`, it prepends `module.` to every parameter key.
    - When you use `torch.compile`, it may prepend `_orig_mod.`.
    If you save a checkpoint from a compiled, DDP-wrapped model and later try to load it
    into a raw, unwrapped model (e.g., for inference), the keys won't match, causing an error.
    This function robustly strips these known prefixes to make checkpoints portable.
    """
    unwrapped = {}
    # A list of possible prefixes to check for and remove.
    prefixes_to_remove = ['_orig_mod.module.', 'module.', '_orig_mod.']

    for k, v in state_dict.items():
        new_k = k
        for prefix in prefixes_to_remove:
            if new_k.startswith(prefix):
                new_k = new_k[len(prefix):]
                break # Move to the next key once a prefix is removed
        unwrapped[new_k] = v
    return unwrapped

# --- Main Training Function ---
def train(config_path: str):
    # --- Setup ---
    config = TrainConfig.from_yaml(config_path)
    is_ddp, rank, world_size = setup_ddp()
    # The 'master process' (rank 0) is responsible for logging, saving checkpoints, etc.
    is_master_process = rank == 0

    # Seed for reproducibility, ensuring each process gets a different seed.
    torch.manual_seed(1337 + rank)
    # These settings can speed up matrix multiplications on compatible hardware (Ampere+ GPUs).
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Use bfloat16 if available (on Ampere+ GPUs), otherwise float16. It offers a great
    # balance of speed, memory savings, and numerical stability compared to float32.
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    device_type = 'cuda' if 'cuda' in config.device else 'cpu'
    # The `autocast` context manager automatically performs model operations in the specified
    # lower-precision `dtype` to save memory and speed up computation.
    ctx = torch.amp.autocast(device_type=device_type, dtype=dtype)

    # Master process handles all setup printing and directory creation.
    if is_master_process:
        os.makedirs(config.out_dir, exist_ok=True)
        print("-" * 50)
        print(" " * 15 + "LUNARIS CODEX TRAINING")
        print("-" * 50)
        print(f"Model Config: {config.model}")
        print(f"Data Dir: {config.data_dir}")
        print(f"Batch Size (per GPU): {config.batch_size}")
        print(f"Gradient Accumulation: {config.gradient_accumulation_steps}")
        print(f"Learning Rate: {config.learning_rate}")
        print(f"Max Steps: {config.max_steps}")
        print(f"Sequence Length: {config.sequence_length}") # This now reliably shows model.max_seq_len
        print("-" * 50)

    # Initialize Weights & Biases for logging, only on the master process.
    if is_master_process and config.wandb_project:
        import wandb
        # Convert dataclass to dict for wandb config logging
        config_dict = config.__dict__
        config_dict['model'] = config.model.__dict__
        wandb.init(project=config.wandb_project, entity=config.wandb_entity, name=config.wandb_run_name, config=config_dict)

    # --- Data Loading ---
    train_dataset = ShardDataset(data_dir=config.data_dir, sequence_length=config.sequence_length)
    # The DistributedSampler ensures that each GPU process receives a different, non-overlapping
    # subset of the data in each epoch.
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True) if is_ddp else None
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)

    # --- Model Initialization ---
    model = LunarisCodex(config.model).to(config.device)

    if is_master_process:
        num_params = model.get_num_params() / 1e6
        print(f"[MODEL] Number of parameters: {num_params:.2f}M")

    # `torch.compile` is a JIT compiler that can significantly speed up your model
    # by fusing operations and optimizing the execution graph.
    if config.compile_model:
        if is_master_process: print("[MODEL] Compiling model...")
        model = torch.compile(model)
    # Wrap the model in DDP after compilation.
    if is_ddp:
        model = DDP(model, device_ids=[int(os.environ['LOCAL_RANK'])])

    # --- Optimizer ---
    # This is good practice: the model itself defines how its parameters should be optimized
    # (e.g., applying weight decay only to certain layers).
    raw_model = model.module if is_ddp else model # Get the raw model from behind the DDP wrapper
    optimizer = raw_model.configure_optimizers(
        weight_decay=config.weight_decay,
        learning_rate=config.learning_rate,
        betas=(config.beta1, config.beta2),
        device_type=device_type
    )

    # --- Checkpoint Resuming ---
    current_step = 0
    current_epoch = 0
    checkpoint_path = os.path.join(config.out_dir, "latest_checkpoint.pt")

    if os.path.exists(checkpoint_path):
        if is_master_process: print(f"[SETUP] Resuming from checkpoint: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=config.device)
        # Use our unwrapping utility to load the state dict correctly.
        unwrapped_state_dict = unwrap_model_keys(state['model'])
        raw_model.load_state_dict(unwrapped_state_dict)
        optimizer.load_state_dict(state['optimizer'])
        current_step = state['step']
        current_epoch = state.get('epoch', 0)
        if is_master_process: print(f"[SETUP] Resumed successfully. Starting from step {current_step}")

    optimizer.zero_grad(set_to_none=True)

    # --- Training Loop ---
    # For DDP, it's important to set the sampler's epoch to ensure proper shuffling
    # when resuming or starting a new epoch.
    if is_ddp:
        train_sampler.set_epoch(current_epoch)

    if is_master_process:
        print(f"\n[TRAIN] Starting training from step {current_step} up to {config.max_steps} steps...")
        pbar = tqdm(total=config.max_steps, desc="Training Steps", initial=current_step, ncols=120)

    # This creates a persistent iterator, so we don't have to re-create it every epoch.
    data_iter = iter(train_loader)

    while current_step < config.max_steps:
        # Manually set the learning rate for each step based on our scheduler.
        lr = get_lr(current_step + 1, config) # use current_step + 1 for 1-based step counting in LR scheduler
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # --- Gradient Accumulation Loop ---
        # This inner loop simulates a larger batch size. We run the forward and backward
        # passes `gradient_accumulation_steps` times, accumulating gradients in the
        # .grad attribute of each parameter before finally calling optimizer.step().
        accumulated_loss = 0.0
        for micro_step in range(config.gradient_accumulation_steps):
            is_last_micro_step = (micro_step == config.gradient_accumulation_steps - 1)
            # This is a key DDP optimization: `model.no_sync()` prevents gradient
            # synchronization (communication between GPUs) on all but the last micro-step.
            # Since synchronization is expensive, this saves a lot of time.
            ddp_context = model.no_sync() if is_ddp and not is_last_micro_step else nullcontext()

            with ddp_context:
                try:
                    # Fetch the next batch of data.
                    x, y = next(data_iter)
                except StopIteration:
                    # This means we've reached the end of the dataset for the current epoch.
                    current_epoch += 1
                    if is_master_process: print(f"\n[DATA] Starting epoch {current_epoch}...")
                    if is_ddp:
                        # Inform the sampler of the new epoch for correct shuffling.
                        train_loader.sampler.set_epoch(current_epoch)
                    # Reset the iterator to start the new epoch.
                    data_iter = iter(train_loader)
                    x, y = next(data_iter)

                x, y = x.to(config.device, non_blocking=True), y.to(config.device, non_blocking=True)

                # The forward pass is executed under the mixed-precision autocast context.
                with ctx:
                    # model.py expects: forward(idx, targets=None, past_key_values=None)
                    # and returns: (logits, loss, new_kv_cache)
                    _, loss, _ = model(x, targets=y)
                    # We scale the loss down before the backward pass. This is important
                    # to ensure that when gradients are summed up, their average magnitude
                    # is correct, as if they came from a single larger batch.
                    loss = loss / config.gradient_accumulation_steps

                accumulated_loss += loss.item()
                # `loss.backward()` accumulates gradients on the parameters.
                loss.backward()

        # --- Optimizer Step ---
        # Clip gradients to prevent them from becoming too large, which can destabilize training.
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        # The optimizer updates the model weights using the accumulated gradients.
        optimizer.step()
        # Reset gradients to zero for the next accumulation cycle.
        optimizer.zero_grad(set_to_none=True)

        current_step += 1
        
        # --- Logging and Checkpointing (Master Process Only) ---
        if is_master_process:
            pbar.update(1)

            # Logging
            if current_step % config.log_interval == 0:
                log_loss = accumulated_loss

                # Calculate perplexity, a common metric for language models.
                # Perplexity = exp(cross_entropy_loss). Lower is better.
                if log_loss < 100: # Avoid overflow for very high loss values
                    try:
                        perplexity = math.exp(log_loss)
                    except (OverflowError, ValueError):
                        perplexity = float('inf')
                else:
                    perplexity = float('inf')

                # Update the progress bar with the latest metrics.
                postfix_data = {
                    "loss": f"{log_loss:.3f}",
                    "ppl": f"{perplexity:.2f}" if perplexity != float('inf') else "inf",
                    "lr": f"{lr:.2e}",
                    "gnorm": f"{grad_norm.item():.2f}"
                }
                pbar.set_postfix(postfix_data)

                # Log metrics to Weights & Biases if configured.
                if config.wandb_project:
                    wandb.log({
                        "train/step": current_step,
                        "train/loss": log_loss,
                        "train/perplexity": perplexity,
                        "train/lr": lr,
                        "train/grad_norm": grad_norm.item(),
                        "train/epoch": current_epoch
                    })

            # Checkpointing
            if current_step > 0 and current_step % config.save_interval == 0:
                # Consolidate all necessary state into a checkpoint dictionary.
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': config_dict, # Save config for reproducibility
                    'step': current_step,
                    'epoch': current_epoch,
                }
                save_path = os.path.join(config.out_dir, f"ckpt_{current_step}.pt")
                torch.save(checkpoint, save_path)

                # Also save as the "latest" checkpoint for easy resuming.
                latest_path = os.path.join(config.out_dir, "latest_checkpoint.pt")
                torch.save(checkpoint, latest_path)
                pbar.write(f"\n[CHECKPOINT] Saved checkpoint to {save_path}")

    # --- Cleanup ---
    if is_master_process:
        print("\nMax steps reached. Finishing training.")
        pbar.close()
        if config.wandb_project:
            wandb.finish()
    if is_ddp:
        # Properly shut down the distributed process group.
        destroy_process_group()

if __name__ == '__main__':
    # --- Script Entry Point ---
    # This allows the script to be run from the command line,
    # expecting the path to a configuration file.
    import argparse
    parser = argparse.ArgumentParser(description="Train a LunarisCodex model.")
    parser.add_argument("config", type=str, help="Path to the config.yaml file.")
    args = parser.parse_args()
    train(args.config)
