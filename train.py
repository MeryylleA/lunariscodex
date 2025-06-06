import argparse
import hashlib
import logging
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch import amp # PyTorch's Automatic Mixed Precision
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from transformers import AutoTokenizer

try:
    from model import LunarisCodexConfig, LunarisMind
except ImportError:
    logging.warning("model.py not found, using placeholder classes for LunarisCodexConfig and LunarisMind.")
    # Placeholder classes
    class LunarisCodexConfig:
        def __init__(self, vocab_size=50257, d_model=768, n_layers=12, n_heads=12,
                     max_seq_len=1024, dropout=0.1, activation="gelu", lora_rank=0,
                     pad_token_id=0):
            self.vocab_size = vocab_size
            self.d_model = d_model
            self.n_layers = n_layers
            self.n_heads = n_heads
            self.max_seq_len = max_seq_len
            self.dropout = dropout
            self.activation = activation
            self.lora_rank = lora_rank
            self.pad_token_id = pad_token_id
            self.__dict__.update(locals()) # To mimic attribute access for checkpointing

    class LunarisMind(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            # Example: a simple embedding and a linear layer
            self.embedding = nn.Embedding(config.vocab_size, config.d_model)
            self.lm_head = nn.Linear(config.d_model, config.vocab_size)
            logging.info("Using placeholder LunarisMind model.")

        def forward(self, input_ids, attention_mask=None):
            # attention_mask is not used in this placeholder, but often is in transformers
            x = self.embedding(input_ids)
            logits = self.lm_head(x)
            return logits

"""
Multi-GPU Training Script for Lunaris Codex using DistributedDataParallel (DDP)

Launch with torchrun:
    torchrun --standalone --nproc_per_node=NUM_GPUS train.py [training arguments]

Example:
    torchrun --standalone --nproc_per_node=4 train.py \
        --memmap_file_train /path/to/train.memmap \
        --batch_size 16 \
        --num_epochs 3
"""

# Setup logging - only detailed logging from rank 0
def setup_logging(rank):
    if rank == 0:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")
    return logging.getLogger(__name__)

def setup_distributed(rank, world_size):
    """Initialize the distributed process group only."""
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '12355') # Ensure this port is free

    # Initialize the process group
    # NCCL is for GPU. For CPU-only distributed training, 'gloo' would be used.
    # This script assumes DDP implies multi-GPU with NCCL.
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        init_method='env://'
    )

def cleanup_distributed():
    """Clean up the distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()

def set_seed(seed_value=42):
    """Sets the seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

def compute_sha256(filepath):
    """Compute SHA-256 hash of a file for integrity verification."""
    hash_sha256 = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    except Exception as e:
        logging.warning(f"Failed to compute SHA-256 for {filepath}: {e}")
        return None

class MemmapCodeDataset(Dataset):
    def __init__(self, memmap_file, tokenizer_pad_id, dtype_str="int32"): # Assinatura correta
        # Only log from rank 0 to avoid duplicate messages
        # Check if dist is initialized before calling get_rank
        current_rank = 0
        if dist.is_initialized():
            current_rank = dist.get_rank()

        if current_rank == 0:
            logging.info(f"Attempting to load memory-mapped dataset from {memmap_file}")

        if not os.path.exists(memmap_file):
            raise FileNotFoundError(f"Memmap/Npy file not found: {memmap_file}")

        try:
            # Usando np.load com mmap_mode='r'. Isso lê o cabeçalho do .npy
            # para obter shape e dtype, e depois mapeia a memória.
            self.data = np.load(memmap_file, mmap_mode='r')
            num_sequences, max_length = self.data.shape # Shape é inferido do arquivo!

            if current_rank == 0:
                logging.info(f"Dataset loaded successfully with shape: ({num_sequences}, {max_length})")

        except ValueError as e:
            logging.error(f"Error loading memmap/npy file (check format/corruption): {memmap_file} - {e}")
            raise
        except Exception as e:
            logging.error(f"An unexpected error occurred while loading {memmap_file}: {e}")
            raise


        self.max_length = max_length # Define self.max_length a partir do shape lido
        self.tokenizer_pad_id = tokenizer_pad_id

        if self.tokenizer_pad_id is None:
            raise ValueError("tokenizer_pad_id cannot be None for MemmapCodeDataset.")

        if current_rank == 0:
            logging.info(f"Dataset ready. Using Pad ID: {self.tokenizer_pad_id}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Ensure data is copied from memmap to avoid issues with multiprocessing or DDP
        input_ids_np = np.array(self.data[idx], dtype=np.int64) # Convert to np.int64 for PyTorch LongTensor
        input_ids = torch.from_numpy(input_ids_np)
        attention_mask = (input_ids != self.tokenizer_pad_id).long()
        return {"input_ids": input_ids, "attention_mask": attention_mask}

def compute_metrics(logits, targets, attention_mask):
    """Computes loss, perplexity, and accuracy, ignoring padded tokens."""
    # Shift logits and labels for next token prediction
    logits_shifted = logits[..., :-1, :].contiguous()
    targets_shifted = targets[..., 1:].contiguous()
    attention_mask_shifted = attention_mask[..., 1:].contiguous() # Also shift attention mask

    # Flatten the tokens
    logits_flat = logits_shifted.view(-1, logits_shifted.size(-1))
    targets_flat = targets_shifted.view(-1)
    active_mask = attention_mask_shifted.view(-1).bool() # Mask for active (non-padded) tokens

    # Filter out padded tokens
    if not active_mask.any(): # Handle cases with no active tokens (e.g., all padding)
        device = logits.device
        return (torch.tensor(0.0, device=device, requires_grad=True), # loss
                torch.tensor(float("inf"), device=device),             # perplexity
                torch.tensor(0.0, device=device))                      # accuracy

    logits_active = logits_flat[active_mask]
    targets_active = targets_flat[active_mask]

    if logits_active.numel() == 0: # Should be caught by active_mask.any() but as a safeguard
        device = logits.device
        return (torch.tensor(0.0, device=device, requires_grad=True),
                torch.tensor(float("inf"), device=device),
                torch.tensor(0.0, device=device))

    loss_fn = nn.CrossEntropyLoss(reduction="sum") # Sum loss over active tokens
    loss = loss_fn(logits_active, targets_active)
    num_active_tokens = active_mask.sum()

    # Average loss over active tokens for this micro-batch
    avg_loss = loss / num_active_tokens if num_active_tokens > 0 else torch.tensor(0.0, device=logits.device, requires_grad=True)
    perplexity = torch.exp(torch.clamp(avg_loss, max=20)) # Clamp avg_loss to avoid overflow in exp
    preds = torch.argmax(logits_active, dim=-1)
    accuracy = (preds == targets_active).float().mean() if num_active_tokens > 0 else torch.tensor(0.0, device=logits.device)

    return avg_loss, perplexity, accuracy # Return the average loss for the micro-batch

def save_checkpoint(model, optimizer, epoch, step, current_loss, args, is_best=False, scheduler=None, rank=0):
    """Save checkpoint only from rank 0 to avoid conflicts."""
    if rank != 0:
        return

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    base_filename = f"lunaris_codex_epoch-{epoch+1}_step-{step}"
    checkpoint_path = os.path.join(args.checkpoint_dir, f"{base_filename}.pt")

    # Get the original model (unwrap DDP if necessary)
    model_to_save = model.module if hasattr(model, 'module') else model

    checkpoint_data = {
        "model_state_dict": model_to_save.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "step": step,
        "loss": current_loss, # This should be validation loss or relevant tracked loss
        "config_args": model_to_save.config.__dict__ if hasattr(model_to_save, 'config') else {}, # Save model configuration
        "train_args": vars(args), # Save training arguments
        "torch_version": torch.__version__,
        "model_class": model_to_save.__class__.__name__,
        "random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        checkpoint_data["torch_cuda_random_state_all"] = [
            torch.cuda.get_rng_state(i) for i in range(torch.cuda.device_count())
        ]

    try:
        torch.save(checkpoint_data, checkpoint_path)
        file_hash = compute_sha256(checkpoint_path)
        if file_hash:
            hash_file = checkpoint_path + ".sha256"
            with open(hash_file, "w") as f:
                f.write(f"{file_hash} {os.path.basename(checkpoint_path)}\n")

        logging.info(f"Checkpoint saved: {checkpoint_path}")

        if is_best:
            best_path = os.path.join(args.checkpoint_dir, "best_model.pt")
            torch.save(checkpoint_data, best_path) # Save a copy as best_model.pt
            best_hash = compute_sha256(best_path)
            if best_hash:
                with open(best_path + ".sha256", "w") as f:
                    f.write(f"{best_hash} best_model.pt\n")
            logging.info(f"Best checkpoint saved: {best_path}")

    except Exception as e:
        logging.error(f"Failed to save checkpoint: {e}")

def verify_checkpoint_integrity(checkpoint_path):
    """Verify checkpoint integrity using SHA-256 hash."""
    hash_file = checkpoint_path + ".sha256"
    if not os.path.exists(hash_file):
        logging.warning(f"No hash file found for {checkpoint_path}, skipping integrity check.")
        return True # Assume valid if no hash file

    try:
        with open(hash_file, "r") as f:
            expected_hash = f.read().split()[0]
        actual_hash = compute_sha256(checkpoint_path)
        if actual_hash and actual_hash == expected_hash:
            logging.info(f"Checkpoint integrity verified: {checkpoint_path}")
            return True
        else:
            logging.error(f"Checkpoint integrity check FAILED: {checkpoint_path}. Expected {expected_hash}, got {actual_hash}")
            return False
    except Exception as e:
        logging.warning(f"Could not verify checkpoint integrity for {checkpoint_path}: {e}")
        return True # Be lenient if verification process fails

def load_checkpoint(model, optimizer, args, device, scheduler=None, rank=0):
    """Load checkpoint with DDP compatibility."""
    start_epoch, start_step, min_val_loss = 0, 0, float("inf")
    checkpoint_to_load = args.resume_from_checkpoint

    # Auto-resume logic if no specific checkpoint is given
    if not checkpoint_to_load and args.checkpoint_dir and os.path.exists(args.checkpoint_dir): # Check if dir exists
        potential_best_checkpoint = os.path.join(args.checkpoint_dir, "best_model.pt")
        if os.path.isfile(potential_best_checkpoint):
            if rank == 0: logging.info(f"Found 'best_model.pt'. Attempting to load.")
            checkpoint_to_load = potential_best_checkpoint
        else:
            checkpoints = [f for f in os.listdir(args.checkpoint_dir) if f.endswith(".pt") and "best_model" not in f]
            if checkpoints:
                # Sort by modification time to get the latest
                checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(args.checkpoint_dir, x)))
                latest_checkpoint = os.path.join(args.checkpoint_dir, checkpoints[-1])
                if rank == 0: logging.info(f"Found latest checkpoint '{latest_checkpoint}'. Attempting to load.")
                checkpoint_to_load = latest_checkpoint

    if checkpoint_to_load and os.path.isfile(checkpoint_to_load):
        if not verify_checkpoint_integrity(checkpoint_to_load):
            if rank == 0: logging.error(f"Integrity check failed for {checkpoint_to_load}. Will start from scratch.")
            return start_epoch, start_step, min_val_loss

        if rank == 0: logging.info(f"Loading checkpoint: {checkpoint_to_load}")
        try:
            # Load checkpoint to the current device
            checkpoint = torch.load(checkpoint_to_load, map_location=device, weights_only=False) # weights_only=False for full state

            # Handle DDP model loading (model might be DDP, checkpoint might be from DDP or single GPU)
            target_model_for_load = model.module if hasattr(model, 'module') else model
            model_state_dict = checkpoint["model_state_dict"]

            is_checkpoint_ddp = any(k.startswith("module.") for k in model_state_dict.keys())

            if not is_checkpoint_ddp: # Checkpoint is from a raw model
                 target_model_for_load.load_state_dict(model_state_dict, strict=False)
            else: # Checkpoint is from a DDP model (has 'module.' prefix)
                model_state_dict_stripped = {k.replace("module.", ""): v for k, v in model_state_dict.items()}
                target_model_for_load.load_state_dict(model_state_dict_stripped, strict=False)


            # Load optimizer and scheduler states, carefully handling potential mismatches
            checkpoint_train_args = checkpoint.get("train_args", {})
            checkpoint_lora_rank = checkpoint_train_args.get("lora_rank", 0) # LoRA rank from checkpoint
            current_lora_rank = getattr(args, 'lora_rank', 0) # Current LoRA rank

            # Only load optimizer/scheduler if LoRA settings are compatible or not using LoRA
            if checkpoint_lora_rank == current_lora_rank:
                if optimizer and "optimizer_state_dict" in checkpoint:
                    try:
                        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                        if rank == 0: logging.info("Optimizer state loaded.")
                    except Exception as e:
                        if rank == 0: logging.warning(f"Could not load optimizer state: {e}. Reinitializing optimizer.")

                if scheduler and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None:
                    ckpt_scheduler_type = checkpoint_train_args.get("lr_scheduler_type", "plateau")
                    if ckpt_scheduler_type == args.lr_scheduler_type: # Check if scheduler types match
                        try:
                            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                            if rank == 0: logging.info(f"Scheduler ({args.lr_scheduler_type}) state loaded.")
                        except Exception as e:
                            if rank == 0: logging.warning(f"Could not load scheduler state: {e}. Reinitializing scheduler.")
                    else:
                        if rank == 0: logging.warning(f"Scheduler type mismatch (ckpt: {ckpt_scheduler_type}, current: {args.lr_scheduler_type}). Skipping scheduler state load.")
            else:
                if rank == 0: logging.warning(f"LoRA rank mismatch (ckpt: {checkpoint_lora_rank}, current: {current_lora_rank}). Skipping optimizer/scheduler load.")


            start_epoch = checkpoint.get("epoch", 0) + 1 # Resume from next epoch
            start_step = checkpoint.get("step", 0)      # Resume from this global step
            min_val_loss = checkpoint.get("loss", float("inf")) # Tracked loss (usually validation)

            # Load RNG states for reproducibility
            if "random_state" in checkpoint: random.setstate(checkpoint["random_state"])
            if "numpy_random_state" in checkpoint: np.random.set_state(checkpoint["numpy_random_state"])
            if "torch_random_state" in checkpoint: torch.set_rng_state(checkpoint["torch_random_state"])
            if "torch_cuda_random_state_all" in checkpoint and torch.cuda.is_available():
                cuda_states = checkpoint["torch_cuda_random_state_all"]
                for i, state in enumerate(cuda_states):
                    if i < torch.cuda.device_count(): # Ensure not to go out of bounds
                        torch.cuda.set_rng_state(state, device=i)

            if rank == 0: logging.info(f"Resuming training from epoch {start_epoch}, global step {start_step}")

        except Exception as e:
            if rank == 0: logging.error(f"Failed to load checkpoint: {e}. Starting from scratch.")
            start_epoch, start_step, min_val_loss = 0, 0, float("inf") # Reset if loading fails
    else:
        if checkpoint_to_load and rank == 0: # Checkpoint specified but not found
            logging.warning(f"Checkpoint file not found: {checkpoint_to_load}. Starting from scratch.")
        elif rank == 0: # No checkpoint specified or found in dir
            logging.info("No checkpoint specified or found. Starting training from scratch.")

    return start_epoch, start_step, min_val_loss

def train_model_loop(model, train_dataloader, val_dataloader, tokenizer, args, device, local_rank, rank, world_size):
    """Main training loop with DDP support and corrected gradient accumulation."""
    model.to(device) # Move model to the target device first

    if world_size > 1:
        if device.type == 'cuda':
            model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
            if rank == 0: logging.info(f"Model wrapped with DDP for CUDA device {local_rank}.")
        else:
            if rank == 0:
                logging.error(f"DDP (world_size > 1) with NCCL backend requires CUDA, but device is CPU. Training may fail or be incorrect.")
            model = DDP(model, find_unused_parameters=False) # For CPU DDP (with gloo)
            if rank == 0: logging.info("Model wrapped with DDP for CPU (ensure 'gloo' backend was used if this is multi-CPU).")

    args.use_lora = args.lora_rank > 0
    optimizer_params = []
    if args.use_lora:
        if rank == 0: logging.info(f"Configuring optimizer for LoRA training (rank={args.lora_rank}).")
        for name, param in model.named_parameters():
            if "lora_" in name or "ls_gamma" in name:
                param.requires_grad = True
                optimizer_params.append(param)
            else:
                param.requires_grad = False
    else:
        if rank == 0: logging.info("Configuring optimizer for full model training.")
        for param in model.parameters():
            param.requires_grad = True
            optimizer_params.append(param)

    num_trainable_params = sum(p.numel() for p in optimizer_params if p.requires_grad)
    if num_trainable_params == 0:
        raise ValueError("No trainable parameters found for the optimizer. Check LoRA setup or model parameters.")
    if rank == 0: logging.info(f"Number of parameters to be optimized: {num_trainable_params:,}")

    optimizer = AdamW(
        optimizer_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        fused=(args.adam_fused and device.type == "cuda" and torch.cuda.is_available())
    )

    scheduler = None
    if args.lr_scheduler_type == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=args.lr_scheduler_patience)
        if rank == 0: logging.info(f"Using ReduceLROnPlateau scheduler with patience {args.lr_scheduler_patience}.")
    elif args.lr_scheduler_type == "cosine_warm_restarts":
        t_0 = args.cosine_t_0
        if t_0 is None:
            if len(train_dataloader) > 0 and args.accumulation_steps > 0:
                 t_0 = len(train_dataloader) // args.accumulation_steps
            else:
                t_0 = 1
            if t_0 == 0: t_0 = 1
            if rank == 0: logging.info(f"CosineAnnealingWarmRestarts T_0 defaulting to {t_0} steps.")
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=t_0, T_mult=args.cosine_t_mult, eta_min=args.cosine_eta_min)
        if rank == 0: logging.info(f"Using CosineAnnealingWarmRestarts scheduler (T_0={t_0}, T_mult={args.cosine_t_mult}, eta_min={args.cosine_eta_min}).")

    scaler = amp.GradScaler(enabled=(args.mixed_precision_dtype is not None and device.type == "cuda"))
    start_epoch, global_step, best_val_loss = load_checkpoint(model, optimizer, args, device, scheduler, rank)

    if args.use_torch_compile and hasattr(torch, "compile"):
        if rank == 0: logging.info(f"Attempting to compile model with torch.compile (mode: {args.torch_compile_mode})...")
        try:
            model_to_compile = model.module if hasattr(model, 'module') else model
            compiled_model = torch.compile(model_to_compile, mode=args.torch_compile_mode)
            if hasattr(model, 'module'):
                model.module = compiled_model
            else:
                model = compiled_model
            if rank == 0: logging.info("Model compiled successfully.")
        except Exception as e:
            if rank == 0: logging.warning(f"Model compilation failed: {e}. Proceeding without compilation.")

    for epoch in range(start_epoch, args.num_epochs):
        model.train()

        if world_size > 1 and hasattr(train_dataloader.sampler, 'set_epoch'):
            train_dataloader.sampler.set_epoch(epoch)

        optimizer.zero_grad()
        epoch_metrics = {"loss": 0.0, "perplexity": 0.0, "accuracy": 0.0, "count": 0}
        train_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs} (Opt Steps: {global_step})", disable=(rank != 0))

        for batch_idx, batch in enumerate(train_iterator):
            input_ids = batch["input_ids"].to(device, non_blocking=args.pin_memory)
            attention_mask = batch["attention_mask"].to(device, non_blocking=args.pin_memory)

            cast_dtype = torch.float32
            use_autocast = args.mixed_precision_dtype is not None and device.type == "cuda"
            if use_autocast:
                cast_dtype = torch.bfloat16 if args.mixed_precision_dtype == "bf16" else torch.float16

            with amp.autocast(device_type=device.type, enabled=use_autocast, dtype=cast_dtype):
                logits = model(input_ids, attention_mask=attention_mask)
                loss, perplexity, accuracy = compute_metrics(logits, input_ids, attention_mask)

            if torch.isnan(loss) or torch.isinf(loss):
                if rank == 0:
                    logging.error(f"NaN or Inf loss encountered at Epoch {epoch+1}, Mini-batch {batch_idx}. Skipping batch.")
                continue

            scaled_loss = scaler.scale(loss)
            scaled_loss.backward()

            if (batch_idx + 1) % args.accumulation_steps == 0 or (batch_idx + 1) == len(train_dataloader):
                if args.grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(optimizer_params, args.grad_clip_norm)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if args.lr_scheduler_type == "cosine_warm_restarts" and scheduler is not None:
                    if len(train_dataloader) > 0:
                        scheduler.step(epoch + batch_idx / len(train_dataloader))
                    else:
                        scheduler.step(epoch)

                if rank == 0:
                    train_iterator.set_description(f"Epoch {epoch+1}/{args.num_epochs} (Opt Steps: {global_step})")

            if torch.isfinite(loss):
                epoch_metrics["loss"] += loss.item()
                epoch_metrics["perplexity"] += perplexity.item() if torch.isfinite(perplexity) else 20.0
                epoch_metrics["accuracy"] += accuracy.item()
                epoch_metrics["count"] += 1

            if rank == 0:
                train_iterator.set_postfix({
                    "loss": f"{loss.item():.4f}" if torch.isfinite(loss) else "NaN",
                    "ppl": f"{perplexity.item():.2f}" if torch.isfinite(perplexity) else "Inf",
                    "acc": f"{accuracy.item():.3f}" if torch.isfinite(accuracy) else "NaN",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}"
                })

            if global_step > 0 and global_step % args.log_interval == 0 and rank == 0 and ( (batch_idx + 1) % args.accumulation_steps == 0 or (batch_idx + 1) == len(train_dataloader) ):
                logging.info(f"E{epoch+1} B{batch_idx+1} OptS{global_step} | Loss: {loss.item():.4f}, PPL: {perplexity.item():.2f}, Acc: {accuracy.item():.3f}, LR: {optimizer.param_groups[0]['lr']:.2e}")

            if args.save_strategy == "steps" and global_step > 0 and global_step % args.save_steps == 0:
                loss_for_step_checkpoint = loss.item()
                is_best_step = False

                if args.validation_interval_steps > 0 and global_step % args.validation_interval_steps == 0 and val_dataloader:
                    val_loss_eval, val_ppl_eval, val_acc_eval = evaluate_model(model, val_dataloader, args, device, rank, world_size)
                    if rank == 0:
                        logging.info(f"Opt Step {global_step} Validation - Loss: {val_loss_eval:.4f}, PPL: {val_ppl_eval:.2f}, Acc: {val_acc_eval:.3f}")

                    if args.lr_scheduler_type == "plateau" and scheduler is not None:
                        scheduler.step(val_loss_eval)

                    loss_for_step_checkpoint = val_loss_eval
                    if val_loss_eval < best_val_loss:
                        best_val_loss = val_loss_eval
                        is_best_step = True

                if world_size > 1: dist.barrier()
                save_checkpoint(model, optimizer, epoch, global_step, loss_for_step_checkpoint, args, is_best_step, scheduler, rank)

        if epoch_metrics["count"] > 0:
            avg_epoch_loss = epoch_metrics["loss"] / epoch_metrics["count"]
            avg_epoch_ppl = epoch_metrics["perplexity"] / epoch_metrics["count"]
            avg_epoch_acc = epoch_metrics["accuracy"] / epoch_metrics["count"]

            if rank == 0:
                logging.info(f"Epoch {epoch+1} Training Avg - Loss: {avg_epoch_loss:.4f}, PPL: {avg_epoch_ppl:.2f}, Acc: {avg_epoch_acc:.3f}")
        else:
             avg_epoch_loss = float('inf')
             if rank == 0: logging.warning(f"Epoch {epoch+1} had no training batches with finite loss.")

        if val_dataloader:
            val_loss, val_ppl, val_acc = evaluate_model(model, val_dataloader, args, device, rank, world_size)
            if rank == 0:
                logging.info(f"Epoch {epoch+1} Validation - Loss: {val_loss:.4f}, PPL: {val_ppl:.2f}, Acc: {val_acc:.3f}")

            if args.lr_scheduler_type == "plateau" and scheduler is not None:
                scheduler.step(val_loss)

            is_best_epoch = val_loss < best_val_loss
            if is_best_epoch:
                best_val_loss = val_loss
                if rank == 0: logging.info(f"New best validation loss: {best_val_loss:.4f}")

            if args.save_strategy == "epoch":
                if world_size > 1: dist.barrier()
                save_checkpoint(model, optimizer, epoch, global_step, val_loss, args, is_best_epoch, scheduler, rank)
        elif args.save_strategy == "epoch":
            if world_size > 1: dist.barrier()
            save_checkpoint(model, optimizer, epoch, global_step, avg_epoch_loss, args, False, scheduler, rank)

    if rank == 0:
        logging.info("Training completed.")
    if world_size > 1:
        dist.barrier()
    return model

def evaluate_model(model, dataloader, args, device, rank, world_size):
    """Evaluate model with metric aggregation across processes."""
    model.eval()
    total_loss, total_perplexity, total_accuracy, count = 0.0, 0.0, 0.0, 0

    with torch.no_grad():
        iterator = tqdm(dataloader, desc="Evaluating", disable=(rank != 0))

        for batch in iterator:
            input_ids = batch["input_ids"].to(device, non_blocking=args.pin_memory)
            attention_mask = batch["attention_mask"].to(device, non_blocking=args.pin_memory)

            cast_dtype = torch.float32
            use_autocast = args.mixed_precision_dtype is not None and device.type == "cuda"
            if use_autocast:
                cast_dtype = torch.bfloat16 if args.mixed_precision_dtype == "bf16" else torch.float16

            with amp.autocast(device_type=device.type, enabled=use_autocast, dtype=cast_dtype):
                logits = model(input_ids, attention_mask=attention_mask)
                loss, perplexity, accuracy = compute_metrics(logits, input_ids, attention_mask)

            if torch.isfinite(loss):
                total_loss += loss.item()
                total_perplexity += perplexity.item() if torch.isfinite(perplexity) else 20.0
                total_accuracy += accuracy.item()
                count += 1
            elif rank == 0:
                logging.debug(f"Non-finite loss ({loss.item()}) in evaluation batch.")

    if world_size > 1:
        local_metrics = torch.tensor([total_loss, total_perplexity, total_accuracy, count], dtype=torch.float64, device=device)
        dist.all_reduce(local_metrics, op=dist.ReduceOp.SUM)
        global_total_loss, global_total_perplexity, global_total_accuracy, global_count = local_metrics.tolist()
    else:
        global_total_loss, global_total_perplexity, global_total_accuracy, global_count = total_loss, total_perplexity, total_accuracy, count

    model.train()

    if global_count == 0:
        if rank == 0: logging.warning("Evaluation resulted in no valid batches across all processes.")
        return float('inf'), float('inf'), 0.0

    avg_loss = global_total_loss / global_count
    avg_perplexity = global_total_perplexity / global_count
    avg_accuracy = global_total_accuracy / global_count

    return avg_loss, avg_perplexity, avg_accuracy

def main():
    parser = argparse.ArgumentParser(description="Train the Lunaris Codex model with DDP")

    dist_group = parser.add_argument_group("Distributed Training")
    dist_group.add_argument("--local_rank", type=int, default=os.getenv('LOCAL_RANK', -1), help="Local rank for distributed training (set by torchrun/launcher)")

    data_group = parser.add_argument_group("Dataset")
    data_group.add_argument("--memmap_file_train", type=str, required=True, help="Path to training data memmap file.")
    data_group.add_argument("--memmap_file_val", type=str, default=None, help="Path to validation data memmap file.")
    # --- CORREÇÃO: Removidos argumentos num_sequences e dataset_max_length que não são mais necessários ---
    data_group.add_argument("--tokenizer_name_or_path", type=str, default="bigcode/starcoder", help="Tokenizer name or path.")
    data_group.add_argument("--dataset_dtype", type=str, default="int32", choices=["int16", "int32"], help="Numpy dtype for memmap data.")

    model_group = parser.add_argument_group("Model")
    model_group.add_argument("--vocab_size", type=int, default=None, help="Vocabulary size (defaults to tokenizer vocab size).")
    model_group.add_argument("--d_model", type=int, default=768, help="Model dimension.")
    model_group.add_argument("--n_layers", type=int, default=10, help="Number of transformer layers.")
    model_group.add_argument("--n_heads", type=int, default=12, help="Number of attention heads.")
    model_group.add_argument("--model_max_seq_len", type=int, default=1024, help="Max sequence length model can handle.")
    model_group.add_argument("--dropout", type=float, default=0.1, help="Dropout rate.")
    model_group.add_argument("--activation", type=str, default="swiglu", choices=["swiglu", "gelu"], help="Activation function.")
    model_group.add_argument("--lora_rank", type=int, default=0, help="LoRA rank (0 for no LoRA).")

    train_group = parser.add_argument_group("Training")
    train_group.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs.")
    train_group.add_argument("--batch_size", type=int, default=16, help="Per-GPU batch size.")
    train_group.add_argument("--accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    train_group.add_argument("--learning_rate", type=float, default=5e-5, help="Initial learning rate.")
    train_group.add_argument("--lr_scheduler_type", type=str, default="plateau", choices=["plateau", "cosine_warm_restarts", "none"], help="LR scheduler type.")
    train_group.add_argument("--lr_scheduler_patience", type=int, default=2, help="Patience for ReduceLROnPlateau scheduler.")
    train_group.add_argument("--cosine_t_0", type=int, default=None, help="T_0 for CosineAnnealingWarmRestarts (steps). Defaults to steps per epoch.")
    train_group.add_argument("--cosine_t_mult", type=int, default=1, help="T_mult for CosineAnnealingWarmRestarts.")
    train_group.add_argument("--cosine_eta_min", type=float, default=0.0, help="eta_min for CosineAnnealingWarmRestarts.")
    train_group.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for AdamW.")
    train_group.add_argument("--adam_fused", action="store_true", help="Use fused AdamW if available.")
    train_group.add_argument("--grad_clip_norm", type=float, default=1.0, help="Gradient clipping norm (0 for no clipping).")
    train_group.add_argument("--mixed_precision_dtype", type=str, default=None, choices=["fp16", "bf16"], help="Mixed precision type (None for fp32).")
    train_group.add_argument("--num_workers", type=int, default=0, help="Number of dataloader workers.")
    train_group.add_argument("--pin_memory", action="store_true", help="Use pin_memory for DataLoader.")
    train_group.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    ckpt_group = parser.add_argument_group("Checkpoints")
    ckpt_group.add_argument("--checkpoint_dir", type=str, default="checkpoints_lunaris", help="Directory to save checkpoints.")
    ckpt_group.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to specific checkpoint to resume from.")
    ckpt_group.add_argument("--save_strategy", type=str, default="epoch", choices=["epoch", "steps"], help="When to save checkpoints.")
    ckpt_group.add_argument("--save_steps", type=int, default=1000, help="Save checkpoint every N steps if save_strategy is 'steps'.")
    ckpt_group.add_argument("--log_interval", type=int, default=100, help="Log training metrics every N optimizer steps.")
    ckpt_group.add_argument("--validation_interval_steps", type=int, default=0, help="Validate every N steps if save_strategy is 'steps' (0 for no step validation).")

    opt_group = parser.add_argument_group("Optimization")
    opt_group.add_argument("--use_torch_compile", action="store_true", help="Use torch.compile for model optimization.")
    opt_group.add_argument("--torch_compile_mode", type=str, default="reduce-overhead", help="Mode for torch.compile.")
    opt_group.add_argument("--allow_tf32", action="store_true", help="Allow TF32 on Ampere GPUs.")
    opt_group.add_argument("--cudnn_benchmark", action="store_true", help="Enable cuDNN benchmark mode.")

    args = parser.parse_args()

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank if args.local_rank != -1 else 0 ))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    is_ddp = world_size > 1

    logger = setup_logging(rank)

    if is_ddp:
        if not torch.cuda.is_available():
            logger.error("Distributed training (DDP) with NCCL backend requires CUDA, but CUDA is not available. Exiting.")
            return # Exit if DDP is requested but no CUDA for NCCL
        torch.cuda.set_device(local_rank)
        setup_distributed(rank, world_size) # Uses nccl backend
        logger.info(f"DDP Initialized: Rank {rank}/{world_size}, Local Rank: {local_rank}, Device: cuda:{local_rank}")
    else:
        logger.info("DDP not initialized. Running in single-process mode.")
        if torch.cuda.is_available():
            logger.info(f"Using device: cuda:{local_rank}")
        else:
            logger.info(f"Using device: cpu")


    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    if args.allow_tf32 and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        if rank == 0: logger.info("TF32 matmul support enabled for CUDA.")
    if args.cudnn_benchmark and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        if rank == 0: logger.info("cuDNN benchmark mode enabled.")

    set_seed(args.seed + rank)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
    if tokenizer.pad_token_id is None:
        if rank == 0: logger.warning("Tokenizer does not have a pad_token_id.")
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            if rank == 0: logger.info(f"Using eos_token_id ({tokenizer.eos_token_id}) as pad_token_id.")
        else:
            added_tokens = tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            if added_tokens > 0 and rank == 0:
                logger.info(f"Added new pad_token: '[PAD]' (ID: {tokenizer.pad_token_id}). Model vocab size may need adjustment.")
    if tokenizer.pad_token_id is None:
        raise ValueError("Could not set a pad_token_id for the tokenizer.")

    # --- CORREÇÃO: Chamada ao construtor de MemmapCodeDataset corrigida ---
    train_dataset = MemmapCodeDataset(
        memmap_file=args.memmap_file_train,
        tokenizer_pad_id=tokenizer.pad_token_id,
        dtype_str=args.dataset_dtype
    )

    train_sampler = None
    shuffle_dataloader = True
    if is_ddp: # world_size > 1
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
        )
        shuffle_dataloader = False

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=shuffle_dataloader, sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=(args.pin_memory and torch.cuda.is_available()), drop_last=True
    )

    val_dataloader = None
    if args.memmap_file_val:
        # --- CORREÇÃO: Chamada ao construtor de MemmapCodeDataset corrigida ---
        val_dataset = MemmapCodeDataset(
            memmap_file=args.memmap_file_val,
            tokenizer_pad_id=tokenizer.pad_token_id,
            dtype_str=args.dataset_dtype
        )
        val_sampler = None
        if is_ddp: # world_size > 1
            val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

        val_dataloader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, sampler=val_sampler,
            num_workers=args.num_workers, pin_memory=(args.pin_memory and torch.cuda.is_available())
        )

    vocab_size_to_use = args.vocab_size if args.vocab_size is not None else len(tokenizer)
    if args.vocab_size is not None and args.vocab_size != len(tokenizer) and rank == 0:
        logger.warning(f"Provided --vocab_size ({args.vocab_size}) differs from tokenizer vocab size ({len(tokenizer)}). Using {vocab_size_to_use}.")

    config = LunarisCodexConfig(
        vocab_size=vocab_size_to_use, d_model=args.d_model, n_layers=args.n_layers,
        n_heads=args.n_heads, max_seq_len=args.model_max_seq_len, dropout=args.dropout,
        activation=args.activation, lora_rank=args.lora_rank, pad_token_id=tokenizer.pad_token_id
    )
    model = LunarisMind(config)


    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model created - Total params: {total_params:,}, Initially Trainable: {trainable_params:,}")
        effective_batch_size = args.batch_size * world_size * args.accumulation_steps
        logger.info(f"Effective batch size: {args.batch_size} (per GPU) * {world_size} (GPUs) * {args.accumulation_steps} (accum) = {effective_batch_size}")


    try:
        trained_model = train_model_loop(
            model, train_dataloader, val_dataloader, tokenizer,
            args, device, local_rank, rank, world_size
        )
        if rank == 0: logger.info("Training completed successfully!")
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
    finally:
        if is_ddp: # world_size > 1
            cleanup_distributed()
            if rank == 0: logger.info("DDP environment cleaned up.")

if __name__ == "__main__":
    main()
