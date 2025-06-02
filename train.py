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

# Project-specific imports
from model import LunarisCodexConfig, LunarisMind

"""
Multi-GPU Training Script for Lunaris Codex using DistributedDataParallel (DDP)

Launch with torchrun:
    torchrun --standalone --nproc_per_node=NUM_GPUS train.py [training arguments]

Example:
    torchrun --standalone --nproc_per_node=4 train.py \
        --memmap_file_train /path/to/train.memmap \
        --num_sequences_train 100000 \
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
    os.environ.setdefault('MASTER_PORT', '12355')

    # Initialize the process group
    dist.init_process_group(
        backend="nccl",  # Use NCCL for NVIDIA GPUs
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
    def __init__(self, memmap_file, num_sequences, max_length=1024, tokenizer_pad_id=0, dtype_str="int32"):
        # Only log from rank 0 to avoid duplicate messages
        if not dist.is_initialized() or dist.get_rank() == 0:
            logging.info(f"Loading dataset from {memmap_file} with {num_sequences} sequences and max_length {max_length}")

        dtype = np.int16 if dtype_str == "int16" else np.int32

        if not os.path.exists(memmap_file):
            raise FileNotFoundError(f"Memmap file not found: {memmap_file}")

        try:
            self.data = np.memmap(memmap_file, dtype=dtype, mode="r", shape=(num_sequences, max_length))
        except ValueError as e:
            logging.error(f"Error loading memmap (check shape/dtype): {memmap_file} - {e}")
            raise

        self.max_length = max_length
        self.tokenizer_pad_id = tokenizer_pad_id

        if self.tokenizer_pad_id is None:
            raise ValueError("tokenizer_pad_id cannot be None for MemmapCodeDataset.")

        if not dist.is_initialized() or dist.get_rank() == 0:
            logging.info(f"Dataset loaded successfully. Using Pad ID: {self.tokenizer_pad_id}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_ids_np = np.array(self.data[idx], dtype=np.int64)
        input_ids = torch.from_numpy(input_ids_np)
        attention_mask = (input_ids != self.tokenizer_pad_id).long()
        return {"input_ids": input_ids, "attention_mask": attention_mask}

def compute_metrics(logits, targets, attention_mask):
    """Computes loss, perplexity, and accuracy, ignoring padded tokens."""
    logits_shifted = logits[..., :-1, :].contiguous()
    targets_shifted = targets[..., 1:].contiguous()
    attention_mask_shifted = attention_mask[..., 1:].contiguous()

    logits_flat = logits_shifted.view(-1, logits_shifted.size(-1))
    targets_flat = targets_shifted.view(-1)
    active_mask = attention_mask_shifted.view(-1).bool()

    if not active_mask.any():
        device = logits.device
        return (torch.tensor(0.0, device=device, requires_grad=True),
                torch.tensor(float("inf"), device=device),
                torch.tensor(0.0, device=device))

    logits_active = logits_flat[active_mask]
    targets_active = targets_flat[active_mask]

    if logits_active.numel() == 0:
        device = logits.device
        return (torch.tensor(0.0, device=device, requires_grad=True),
                torch.tensor(float("inf"), device=device),
                torch.tensor(0.0, device=device))

    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    loss = loss_fn(logits_active, targets_active)
    num_active_tokens = active_mask.sum()
    avg_loss = loss / num_active_tokens if num_active_tokens > 0 else torch.tensor(0.0, device=logits.device, requires_grad=True)
    perplexity = torch.exp(torch.clamp(avg_loss, max=20))
    preds = torch.argmax(logits_active, dim=-1)
    accuracy = (preds == targets_active).float().mean() if num_active_tokens > 0 else torch.tensor(0.0, device=logits.device)

    return avg_loss, perplexity, accuracy

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
        "loss": current_loss,
        "config_args": model_to_save.config.__dict__,
        "train_args": vars(args),
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
            torch.save(checkpoint_data, best_path)
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
        return True

    try:
        with open(hash_file, "r") as f:
            expected_hash = f.read().split()[0]
        actual_hash = compute_sha256(checkpoint_path)
        if actual_hash and actual_hash == expected_hash:
            logging.info(f"Checkpoint integrity verified: {checkpoint_path}")
            return True
        else:
            logging.error(f"Checkpoint integrity check FAILED: {checkpoint_path}")
            return False
    except Exception as e:
        logging.warning(f"Could not verify checkpoint integrity for {checkpoint_path}: {e}")
        return True

def load_checkpoint(model, optimizer, args, device, scheduler=None, rank=0):
    """Load checkpoint with DDP compatibility."""
    start_epoch, start_step, min_val_loss = 0, 0, float("inf")
    checkpoint_to_load = args.resume_from_checkpoint

    if not checkpoint_to_load and args.checkpoint_dir:
        potential_best_checkpoint = os.path.join(args.checkpoint_dir, "best_model.pt")
        if os.path.isfile(potential_best_checkpoint):
            if rank == 0:
                logging.info(f"Found 'best_model.pt' in checkpoint directory. Attempting to load it.")
            checkpoint_to_load = potential_best_checkpoint
        else:
            checkpoints = [f for f in os.listdir(args.checkpoint_dir) if f.endswith(".pt") and "best_model" not in f]
            if checkpoints:
                checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(args.checkpoint_dir, x)))
                latest_checkpoint = os.path.join(args.checkpoint_dir, checkpoints[-1])
                if rank == 0:
                    logging.info(f"Found latest checkpoint '{latest_checkpoint}'. Attempting to load it.")
                checkpoint_to_load = latest_checkpoint

    if checkpoint_to_load and os.path.isfile(checkpoint_to_load):
        if not verify_checkpoint_integrity(checkpoint_to_load):
            if rank == 0:
                logging.error(f"Integrity check failed for {checkpoint_to_load}. Will start from scratch.")
            return start_epoch, start_step, min_val_loss

        if rank == 0:
            logging.info(f"Loading checkpoint: {checkpoint_to_load}")

        try:
            checkpoint = torch.load(checkpoint_to_load, map_location=device, weights_only=False)

            # Handle DDP model loading
            target_model = model.module if hasattr(model, 'module') else model
            model_state_dict = checkpoint["model_state_dict"]

            # Handle module prefix mismatch
            is_model_ddp = hasattr(model, 'module')
            is_checkpoint_ddp = any(k.startswith("module.") for k in model_state_dict.keys())

            if is_model_ddp and not is_checkpoint_ddp:
                # Current model is DDP but checkpoint is not - add module prefix
                model_state_dict = {f"module.{k}": v for k, v in model_state_dict.items()}
                model.load_state_dict(model_state_dict, strict=False)
            elif not is_model_ddp and is_checkpoint_ddp:
                # Current model is not DDP but checkpoint is - remove module prefix
                model_state_dict = {k.replace("module.", ""): v for k, v in model_state_dict.items()}
                model.load_state_dict(model_state_dict, strict=False)
            else:
                # Load normally
                target_model.load_state_dict(model_state_dict, strict=False)

            # Load optimizer and scheduler states
            checkpoint_train_args = checkpoint.get("train_args", {})
            checkpoint_lora_rank = checkpoint_train_args.get("lora_rank", 0)
            current_lora_rank = getattr(args, 'lora_rank', 0)

            if checkpoint_lora_rank == current_lora_rank:
                if optimizer and "optimizer_state_dict" in checkpoint:
                    try:
                        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                        if rank == 0:
                            logging.info("Optimizer state loaded.")
                    except Exception as e:
                        if rank == 0:
                            logging.warning(f"Could not load optimizer state: {e}. Reinitializing optimizer.")

                if scheduler and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None:
                    ckpt_scheduler_type = checkpoint_train_args.get("lr_scheduler_type", "plateau")
                    if ckpt_scheduler_type == args.lr_scheduler_type:
                        try:
                            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                            if rank == 0:
                                logging.info(f"Scheduler ({args.lr_scheduler_type}) state loaded.")
                        except Exception as e:
                            if rank == 0:
                                logging.warning(f"Could not load scheduler state: {e}. Reinitializing scheduler.")
                    else:
                        if rank == 0:
                            logging.warning(f"Scheduler type mismatch. Skipping scheduler state load.")
            else:
                if rank == 0:
                    logging.warning(f"LoRA rank mismatch. Skipping optimizer/scheduler load.")

            start_epoch = checkpoint.get("epoch", 0) + 1
            start_step = checkpoint.get("step", 0)
            min_val_loss = checkpoint.get("loss", float("inf"))

            # Load RNG states
            if "random_state" in checkpoint:
                random.setstate(checkpoint["random_state"])
            if "numpy_random_state" in checkpoint:
                np.random.set_state(checkpoint["numpy_random_state"])
            if "torch_random_state" in checkpoint:
                torch.set_rng_state(checkpoint["torch_random_state"])
            if "torch_cuda_random_state_all" in checkpoint and torch.cuda.is_available():
                cuda_states = checkpoint["torch_cuda_random_state_all"]
                for i, state in enumerate(cuda_states):
                    if i < torch.cuda.device_count():
                        torch.cuda.set_rng_state(state, device=i)

            if rank == 0:
                logging.info(f"Resuming training from epoch {start_epoch}, global step {start_step}")

        except Exception as e:
            if rank == 0:
                logging.error(f"Failed to load checkpoint: {e}. Starting from scratch.")
            start_epoch, start_step, min_val_loss = 0, 0, float("inf")
    else:
        if checkpoint_to_load and rank == 0:
            logging.warning(f"Checkpoint file not found: {checkpoint_to_load}. Starting from scratch.")
        elif rank == 0:
            logging.info("No checkpoint specified or found. Starting training from scratch.")

    return start_epoch, start_step, min_val_loss

def train_model_loop(model, train_dataloader, val_dataloader, tokenizer, args, device, local_rank, rank, world_size):
    """Main training loop with DDP support."""
    model.to(device)

    # Wrap model with DDP using local_rank for device-specific arguments
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    args.use_lora = args.lora_rank > 0
    optimizer_params = []

    if args.use_lora:
        if rank == 0:
            logging.info(f"Configuring optimizer for LoRA training (rank={args.lora_rank}).")
        for name, param in model.named_parameters():
            if "lora_" in name or "ls_gamma" in name:
                param.requires_grad = True
                optimizer_params.append(param)
            else:
                param.requires_grad = False
    else:
        if rank == 0:
            logging.info("Configuring optimizer for full model training.")
        for param in model.parameters():
            param.requires_grad = True
            optimizer_params.append(param)

    num_trainable_params = sum(p.numel() for p in optimizer_params)
    if num_trainable_params == 0:
        raise ValueError("No trainable parameters found for the optimizer.")

    if rank == 0:
        logging.info(f"Number of parameters to be optimized: {num_trainable_params:,}")

    optimizer = AdamW(
        optimizer_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        fused=(args.adam_fused and device.type == "cuda" and torch.cuda.is_available())
    )

    # Scheduler setup
    scheduler = None
    if args.lr_scheduler_type == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=args.lr_scheduler_patience)
        if rank == 0:
            logging.info(f"Using ReduceLROnPlateau scheduler with patience {args.lr_scheduler_patience}.")
    elif args.lr_scheduler_type == "cosine_warm_restarts":
        t_0 = args.cosine_t_0
        if t_0 is None:
            t_0 = len(train_dataloader) // args.accumulation_steps
            if t_0 == 0: t_0 = 1
            if rank == 0:
                logging.info(f"CosineAnnealingWarmRestarts T_0 defaulting to {t_0}")
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=t_0, T_mult=args.cosine_t_mult, eta_min=args.cosine_eta_min)
        if rank == 0:
            logging.info(f"Using CosineAnnealingWarmRestarts scheduler.")

    scaler = amp.GradScaler(enabled=(args.mixed_precision_dtype is not None and device.type == "cuda"))
    start_epoch, global_step, best_val_loss = load_checkpoint(model, optimizer, args, device, scheduler, rank)

    # Compile model if requested
    if args.use_torch_compile and hasattr(torch, "compile"):
        if rank == 0:
            logging.info(f"Attempting to compile model with torch.compile...")
        try:
            model = torch.compile(model, mode=args.torch_compile_mode)
            if rank == 0:
                logging.info("Model compiled successfully.")
        except Exception as e:
            if rank == 0:
                logging.warning(f"Model compilation failed: {e}. Proceeding without compilation.")

    for epoch in range(start_epoch, args.num_epochs):
        model.train()

        # Set epoch for DistributedSampler to ensure proper shuffling
        train_dataloader.sampler.set_epoch(epoch)

        optimizer.zero_grad()
        epoch_metrics = {"loss": 0.0, "perplexity": 0.0, "accuracy": 0.0, "count": 0}

        # Show progress bar only on rank 0
        if rank == 0:
            train_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs} (Opt Steps: {global_step})")
        else:
            train_iterator = train_dataloader

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
                    logging.error(f"NaN or Inf loss encountered at Epoch {epoch+1}, Mini-batch {batch_idx}. Skipping.")
                continue

            # Normalize loss for accumulation
            if args.accumulation_steps > 1:
                loss = loss / args.accumulation_steps

            scaler.scale(loss).backward()

            # Optimizer step after accumulation
            if (batch_idx + 1) % args.accumulation_steps == 0 or (batch_idx + 1) == len(train_dataloader):
                if args.grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(optimizer_params, args.grad_clip_norm)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                # Step cosine scheduler
                if args.lr_scheduler_type == "cosine_warm_restarts" and scheduler is not None:
                    scheduler.step()

                # Update progress bar only on rank 0
                if rank == 0:
                    train_iterator.set_description(f"Epoch {epoch+1}/{args.num_epochs} (Opt Steps: {global_step})")

            if torch.isfinite(loss):
                unnormalized_loss = loss.item() * args.accumulation_steps if args.accumulation_steps > 1 else loss.item()
                epoch_metrics["loss"] += unnormalized_loss
                epoch_metrics["perplexity"] += perplexity.item() if torch.isfinite(perplexity) else 20
                epoch_metrics["accuracy"] += accuracy.item()
                epoch_metrics["count"] += 1

            # Update progress bar postfix only on rank 0
            if rank == 0 and hasattr(train_iterator, 'set_postfix'):
                train_iterator.set_postfix({
                    "loss": f"{loss.item():.4f}" if torch.isfinite(loss) else "NaN",
                    "ppl": f"{perplexity.item():.2f}" if torch.isfinite(perplexity) else "Inf",
                    "acc": f"{accuracy.item():.3f}" if torch.isfinite(accuracy) else "NaN",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}"
                })

            # Logging and step-based checkpointing
            if global_step > 0 and global_step % args.log_interval == 0 and rank == 0:
                current_loss = loss.item() * args.accumulation_steps if args.accumulation_steps > 1 else loss.item()
                current_ppl = perplexity.item() if torch.isfinite(perplexity) else float('inf')
                current_acc = accuracy.item() if torch.isfinite(accuracy) else float('nan')
                logging.info(f"E{epoch+1} B{batch_idx+1} OptS{global_step} | Loss: {current_loss:.4f}, PPL: {current_ppl:.2f}, Acc: {current_acc:.3f}")

            if args.save_strategy == "steps" and global_step > 0 and global_step % args.save_steps == 0:
                current_eval_loss = loss.item() * args.accumulation_steps if args.accumulation_steps > 1 else loss.item()
                is_best_step = False

                if args.validation_interval_steps > 0 and global_step % args.validation_interval_steps == 0 and val_dataloader:
                    val_loss_eval, val_ppl_eval, val_acc_eval = evaluate_model(model, val_dataloader, args, device, rank, world_size)
                    if rank == 0:
                        logging.info(f"Opt Step {global_step} Validation - Loss: {val_loss_eval:.4f}, PPL: {val_ppl_eval:.2f}, Acc: {val_acc_eval:.3f}")

                    if args.lr_scheduler_type == "plateau" and scheduler is not None:
                        scheduler.step(val_loss_eval)

                    current_eval_loss = val_loss_eval
                    if val_loss_eval < best_val_loss:
                        best_val_loss = val_loss_eval
                        is_best_step = True

                # Synchronize before saving
                dist.barrier()
                save_checkpoint(model, optimizer, epoch, global_step, current_eval_loss, args, is_best_step, scheduler, rank)

        # End of epoch processing
        if epoch_metrics["count"] > 0:
            avg_epoch_loss = epoch_metrics["loss"] / epoch_metrics["count"]
            avg_epoch_ppl = epoch_metrics["perplexity"] / epoch_metrics["count"]
            avg_epoch_acc = epoch_metrics["accuracy"] / epoch_metrics["count"]

            if rank == 0:
                logging.info(f"Epoch {epoch+1} Training Avg - Loss: {avg_epoch_loss:.4f}, PPL: {avg_epoch_ppl:.2f}, Acc: {avg_epoch_acc:.3f}")

        # Validation at end of epoch
        if val_dataloader:
            val_loss, val_ppl, val_acc = evaluate_model(model, val_dataloader, args, device, rank, world_size)
            if rank == 0:
                logging.info(f"Epoch {epoch+1} Validation - Loss: {val_loss:.4f}, PPL: {val_ppl:.2f}, Acc: {val_acc:.3f}")

            if args.lr_scheduler_type == "plateau" and scheduler is not None:
                scheduler.step(val_loss)

            is_best_epoch = val_loss < best_val_loss
            if is_best_epoch:
                best_val_loss = val_loss
                if rank == 0:
                    logging.info(f"New best validation loss: {best_val_loss:.4f}")

            if args.save_strategy == "epoch":
                dist.barrier()
                save_checkpoint(model, optimizer, epoch, global_step, val_loss, args, is_best_epoch, scheduler, rank)
        elif args.save_strategy == "epoch":
            dist.barrier()
            avg_loss = avg_epoch_loss if epoch_metrics["count"] > 0 else float('inf')
            save_checkpoint(model, optimizer, epoch, global_step, avg_loss, args, False, scheduler, rank)

    if rank == 0:
        logging.info("Training completed.")

    return model

def evaluate_model(model, dataloader, args, device, rank, world_size):
    """Evaluate model with metric aggregation across processes."""
    model.eval()
    total_loss, total_perplexity, total_accuracy, count = 0.0, 0.0, 0.0, 0

    with torch.no_grad():
        # Show progress only on rank 0
        iterator = tqdm(dataloader, desc="Evaluating") if rank == 0 else dataloader

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
                total_perplexity += perplexity.item() if torch.isfinite(perplexity) else 20
                total_accuracy += accuracy.item()
                count += 1

    model.train()

    if count == 0:
        if rank == 0:
            logging.warning("Evaluation resulted in no valid batches.")
        return float('inf'), float('inf'), 0.0

    # Aggregate metrics across all processes
    local_loss = torch.tensor(total_loss, device=device)
    local_perplexity = torch.tensor(total_perplexity, device=device)
    local_accuracy = torch.tensor(total_accuracy, device=device)
    local_count = torch.tensor(count, device=device)

    # Sum across all processes
    dist.all_reduce(local_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(local_perplexity, op=dist.ReduceOp.SUM)
    dist.all_reduce(local_accuracy, op=dist.ReduceOp.SUM)
    dist.all_reduce(local_count, op=dist.ReduceOp.SUM)

    # Calculate global averages
    global_count = local_count.item()
    if global_count > 0:
        avg_loss = local_loss.item() / global_count
        avg_perplexity = local_perplexity.item() / global_count
        avg_accuracy = local_accuracy.item() / global_count
    else:
        avg_loss, avg_perplexity, avg_accuracy = float('inf'), float('inf'), 0.0

    return avg_loss, avg_perplexity, avg_accuracy

def main():
    parser = argparse.ArgumentParser(description="Train the Lunaris Codex model with DDP")

    # Distributed training arguments
    dist_group = parser.add_argument_group("Distributed Training")
    dist_group.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training (set by torchrun)")

    # Dataset arguments
    data_group = parser.add_argument_group("Dataset")
    data_group.add_argument("--memmap_file_train", type=str, required=True)
    data_group.add_argument("--num_sequences_train", type=int, required=True)
    data_group.add_argument("--memmap_file_val", type=str, default=None)
    data_group.add_argument("--num_sequences_val", type=int, default=0)
    data_group.add_argument("--tokenizer_name_or_path", type=str, default="bigcode/starcoder")
    data_group.add_argument("--dataset_max_length", type=int, default=1024)
    data_group.add_argument("--dataset_dtype", type=str, default="int32", choices=["int16", "int32"])

    # Model arguments
    model_group = parser.add_argument_group("Model")
    model_group.add_argument("--vocab_size", type=int, default=None)
    model_group.add_argument("--d_model", type=int, default=768)
    model_group.add_argument("--n_layers", type=int, default=10)
    model_group.add_argument("--n_heads", type=int, default=12)
    model_group.add_argument("--model_max_seq_len", type=int, default=1024)
    model_group.add_argument("--dropout", type=float, default=0.1)
    model_group.add_argument("--activation", type=str, default="swiglu", choices=["swiglu", "gelu"])
    model_group.add_argument("--lora_rank", type=int, default=0)

    # Training arguments
    train_group = parser.add_argument_group("Training")
    train_group.add_argument("--num_epochs", type=int, default=3)
    train_group.add_argument("--batch_size", type=int, default=16, help="Per-GPU batch size")
    train_group.add_argument("--accumulation_steps", type=int, default=1)
    train_group.add_argument("--learning_rate", type=float, default=5e-5)
    train_group.add_argument("--lr_scheduler_type", type=str, default="plateau", choices=["plateau", "cosine_warm_restarts"])
    train_group.add_argument("--lr_scheduler_patience", type=int, default=2)
    train_group.add_argument("--cosine_t_0", type=int, default=None)
    train_group.add_argument("--cosine_t_mult", type=int, default=1)
    train_group.add_argument("--cosine_eta_min", type=float, default=0.0)
    train_group.add_argument("--weight_decay", type=float, default=0.01)
    train_group.add_argument("--adam_fused", action="store_true")
    train_group.add_argument("--grad_clip_norm", type=float, default=1.0)
    train_group.add_argument("--mixed_precision_dtype", type=str, default=None, choices=["fp16", "bf16"])
    train_group.add_argument("--num_workers", type=int, default=0)
    train_group.add_argument("--pin_memory", action="store_true")
    train_group.add_argument("--seed", type=int, default=42)

    # Checkpoint arguments
    ckpt_group = parser.add_argument_group("Checkpoints")
    ckpt_group.add_argument("--checkpoint_dir", type=str, default="checkpoints_lunaris")
    ckpt_group.add_argument("--resume_from_checkpoint", type=str, default=None)
    ckpt_group.add_argument("--save_strategy", type=str, default="epoch", choices=["epoch", "steps"])
    ckpt_group.add_argument("--save_steps", type=int, default=1000)
    ckpt_group.add_argument("--log_interval", type=int, default=100)
    ckpt_group.add_argument("--validation_interval_steps", type=int, default=0)

    # Optimization arguments
    opt_group = parser.add_argument_group("Optimization")
    opt_group.add_argument("--use_torch_compile", action="store_true")
    opt_group.add_argument("--torch_compile_mode", type=str, default="reduce-overhead")
    opt_group.add_argument("--allow_tf32", action="store_true")
    opt_group.add_argument("--cudnn_benchmark", action="store_true")

    args = parser.parse_args()

    # Get distributed training info from environment variables (set by torchrun)
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Override local_rank if provided via command line
    if args.local_rank != -1:
        local_rank = args.local_rank

    # Set CUDA device early for this process
    if torch.cuda.is_available() and world_size > 1:
        torch.cuda.set_device(local_rank)

    # Setup distributed training
    if world_size > 1:
        setup_distributed(rank, world_size)

    # Setup logging
    logger = setup_logging(rank)

    # Set device
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # Performance optimizations
    if args.allow_tf32 and torch.cuda.is_available():
        if torch.cuda.get_device_capability()[0] >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            if rank == 0:
                logger.info("TF32 matmul support enabled for CUDA.")

    if args.cudnn_benchmark and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        if rank == 0:
            logger.info("cuDNN benchmark mode enabled.")

    # Set seed
    set_seed(args.seed)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
    if tokenizer.pad_token_id is None:
        if rank == 0:
            logger.warning("Tokenizer does not have a pad_token_id.")
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            if rank == 0:
                logger.info(f"Using eos_token_id ({tokenizer.eos_token_id}) as pad_token_id.")
        else:
            added_tokens = tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            if added_tokens > 0 and rank == 0:
                logger.info(f"Added new pad_token: '[PAD]' (ID: {tokenizer.pad_token_id}).")

    if tokenizer.pad_token_id is None:
        raise ValueError("Could not set a pad_token_id for the tokenizer.")

    # Create datasets
    train_dataset = MemmapCodeDataset(
        args.memmap_file_train, args.num_sequences_train,
        args.dataset_max_length, tokenizer.pad_token_id, args.dataset_dtype
    )

    # Use DistributedSampler for training data
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True
        )
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=(args.pin_memory and torch.cuda.is_available()),
        drop_last=True
    )

    # Validation dataloader
    val_dataloader = None
    if args.memmap_file_val and args.num_sequences_val > 0:
        val_dataset = MemmapCodeDataset(
            args.memmap_file_val, args.num_sequences_val,
            args.dataset_max_length, tokenizer.pad_token_id, args.dataset_dtype
        )

        if world_size > 1:
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False
            )
        else:
            val_sampler = None

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=args.num_workers,
            pin_memory=(args.pin_memory and torch.cuda.is_available())
        )

    # Model setup
    vocab_size_to_use = args.vocab_size if args.vocab_size is not None else len(tokenizer)
    if args.vocab_size is not None and args.vocab_size != len(tokenizer) and rank == 0:
        logger.warning(f"Provided --vocab_size ({args.vocab_size}) differs from tokenizer vocab size ({len(tokenizer)}).")

    config = LunarisCodexConfig(
        vocab_size=vocab_size_to_use,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        max_seq_len=args.model_max_seq_len,
        dropout=args.dropout,
        activation=args.activation,
        lora_rank=args.lora_rank,
        pad_token_id=tokenizer.pad_token_id
    )

    model = LunarisMind(config)

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model created - Total params: {total_params:,}, Trainable: {trainable_params:,}")
        logger.info(f"Effective batch size: {args.batch_size} * {world_size} = {args.batch_size * world_size}")

    try:
        # Start training
        trained_model = train_model_loop(model, train_dataloader, val_dataloader, tokenizer, args, device, local_rank, rank, world_size)

        if rank == 0:
            logger.info("Training completed successfully!")

    finally:
        # Cleanup distributed training
        if world_size > 1:
            cleanup_distributed()

if __name__ == "__main__":
    main()
