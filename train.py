import argparse
import hashlib
import logging
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch import amp # PyTorch's Automatic Mixed Precision
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

# Project-specific imports
from model import LunarisCodexConfig, LunarisMind # count_parameters is not used directly here

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

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
        logger.warning(f"Failed to compute SHA-256 for {filepath}: {e}")
        return None

class MemmapCodeDataset(Dataset):
    def __init__(self, memmap_file, num_sequences, max_length=1024, tokenizer_pad_id=0, dtype_str="int32"):
        logger.info(f"Loading dataset from {memmap_file} with {num_sequences} sequences and max_length {max_length}")
        dtype = np.int16 if dtype_str == "int16" else np.int32

        if not os.path.exists(memmap_file):
            raise FileNotFoundError(f"Memmap file not found: {memmap_file}")

        try:
            self.data = np.memmap(memmap_file, dtype=dtype, mode="r", shape=(num_sequences, max_length))
        except ValueError as e:
            logger.error(f"Error loading memmap (check shape/dtype): {memmap_file} - {e}")
            raise

        self.max_length = max_length
        self.tokenizer_pad_id = tokenizer_pad_id
        if self.tokenizer_pad_id is None:
            raise ValueError("tokenizer_pad_id cannot be None for MemmapCodeDataset.")
        logger.info(f"Dataset loaded successfully. Using Pad ID: {self.tokenizer_pad_id}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_ids_np = np.array(self.data[idx], dtype=np.int64)
        input_ids = torch.from_numpy(input_ids_np)
        attention_mask = (input_ids != self.tokenizer_pad_id).long()
        return {"input_ids": input_ids, "attention_mask": attention_mask}

def compute_metrics(logits, targets, attention_mask):
    """Computes loss, perplexity, and accuracy, ignoring padded tokens."""
    # Shift logits and labels for next token prediction
    logits_shifted = logits[..., :-1, :].contiguous()
    targets_shifted = targets[..., 1:].contiguous()
    attention_mask_shifted = attention_mask[..., 1:].contiguous() # Mask for shifted targets

    # Flatten the tokens
    logits_flat = logits_shifted.view(-1, logits_shifted.size(-1))
    targets_flat = targets_shifted.view(-1)
    active_mask = attention_mask_shifted.view(-1).bool()

    if not active_mask.any():
        logger.warning("No active tokens for loss calculation in this batch.")
        device = logits.device
        return (torch.tensor(0.0, device=device, requires_grad=True), # ensure loss has grad_fn for backward
                torch.tensor(float("inf"), device=device),
                torch.tensor(0.0, device=device))

    # Select only active (non-padded) tokens
    logits_active = logits_flat[active_mask]
    targets_active = targets_flat[active_mask]

    if logits_active.numel() == 0: # Should be caught by active_mask.any() but as a safeguard
        logger.warning("Zero active tokens after filtering for loss calculation.")
        device = logits.device
        return (torch.tensor(0.0, device=device, requires_grad=True),
                torch.tensor(float("inf"), device=device),
                torch.tensor(0.0, device=device))


    loss_fn = nn.CrossEntropyLoss(reduction="sum") # Sum loss over active tokens
    loss = loss_fn(logits_active, targets_active)
    num_active_tokens = active_mask.sum()

    avg_loss = loss / num_active_tokens if num_active_tokens > 0 else torch.tensor(0.0, device=logits.device, requires_grad=True)

    perplexity = torch.exp(torch.clamp(avg_loss, max=20))  # Clamp to prevent overflow
    preds = torch.argmax(logits_active, dim=-1)
    accuracy = (preds == targets_active).float().mean() if num_active_tokens > 0 else torch.tensor(0.0, device=logits.device)

    return avg_loss, perplexity, accuracy

def save_checkpoint(model, optimizer, epoch, step, current_loss, args, is_best=False, scheduler=None):
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    base_filename = f"lunaris_codex_epoch-{epoch+1}_step-{step}"
    checkpoint_path = os.path.join(args.checkpoint_dir, f"{base_filename}.pt")

    model_to_save = model._orig_mod if hasattr(model, "_orig_mod") else model

    checkpoint_data = {
        "model_state_dict": model_to_save.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "step": step,
        "loss": current_loss,
        "config_args": model_to_save.config.__dict__, # Save model config (from LunarisCodexConfig)
        "train_args": vars(args), # Save training script args
        "torch_version": torch.__version__,
        "model_class": model_to_save.__class__.__name__
    }

    try:
        torch.save(checkpoint_data, checkpoint_path)
        file_hash = compute_sha256(checkpoint_path)
        if file_hash:
            hash_file = checkpoint_path + ".sha256"
            with open(hash_file, "w") as f:
                f.write(f"{file_hash}  {os.path.basename(checkpoint_path)}\n")
        logger.info(f"Checkpoint saved: {checkpoint_path}")

        if is_best:
            best_path = os.path.join(args.checkpoint_dir, "best_model.pt")
            torch.save(checkpoint_data, best_path)
            best_hash = compute_sha256(best_path)
            if best_hash:
                with open(best_path + ".sha256", "w") as f:
                    f.write(f"{best_hash}  best_model.pt\n")
            logger.info(f"Best checkpoint saved: {best_path}")
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")
        # Not raising here to allow training to continue if saving fails once

def verify_checkpoint_integrity(checkpoint_path):
    hash_file = checkpoint_path + ".sha256"
    if not os.path.exists(hash_file):
        logger.warning(f"No hash file found for {checkpoint_path}, skipping integrity check.")
        return True
    try:
        with open(hash_file, "r") as f:
            expected_hash = f.read().split()[0]
        actual_hash = compute_sha256(checkpoint_path)
        if actual_hash and actual_hash == expected_hash:
            logger.info(f"Checkpoint integrity verified: {checkpoint_path}")
            return True
        else:
            logger.error(f"Checkpoint integrity check FAILED: {checkpoint_path}. Expected {expected_hash}, got {actual_hash}")
            return False
    except Exception as e:
        logger.warning(f"Could not verify checkpoint integrity for {checkpoint_path}: {e}")
        return True # Be lenient if verification process itself fails

def load_checkpoint(model, optimizer, args, device, scheduler=None):
    start_epoch, start_step, min_val_loss = 0, 0, float("inf")
    checkpoint_to_load = args.resume_from_checkpoint

    if not checkpoint_to_load and args.checkpoint_dir:
        # Try to load 'best_model.pt' if no specific checkpoint is given
        potential_best_checkpoint = os.path.join(args.checkpoint_dir, "best_model.pt")
        if os.path.isfile(potential_best_checkpoint):
            logger.info(f"Found 'best_model.pt' in checkpoint directory. Attempting to load it.")
            checkpoint_to_load = potential_best_checkpoint
        else: # Try to load the latest checkpoint if best_model.pt doesn't exist
            checkpoints = [f for f in os.listdir(args.checkpoint_dir) if f.endswith(".pt") and "best_model" not in f]
            if checkpoints:
                checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(args.checkpoint_dir, x)))
                latest_checkpoint = os.path.join(args.checkpoint_dir, checkpoints[-1])
                logger.info(f"Found latest checkpoint '{latest_checkpoint}'. Attempting to load it.")
                checkpoint_to_load = latest_checkpoint


    if checkpoint_to_load and os.path.isfile(checkpoint_to_load):
        if not verify_checkpoint_integrity(checkpoint_to_load):
            logger.error(f"Integrity check failed for {checkpoint_to_load}. Will start from scratch.")
            return start_epoch, start_step, min_val_loss

        logger.info(f"Loading checkpoint: {checkpoint_to_load}")
        try:
            checkpoint = torch.load(checkpoint_to_load, map_location=device, weights_only=False)

            target_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            model_state_dict = checkpoint["model_state_dict"]

            # Adjust keys if model was compiled and checkpoint is not, or vice-versa
            is_model_compiled_now = hasattr(model, "_orig_mod")
            is_checkpoint_compiled = any(k.startswith("_orig_mod.") for k in model_state_dict.keys())

            if is_model_compiled_now and not is_checkpoint_compiled:
                # Current model is compiled, checkpoint is not. No key adjustment needed for target_model.load_state_dict
                pass
            elif not is_model_compiled_now and is_checkpoint_compiled:
                 # Current model is not compiled, checkpoint is. Strip _orig_mod.
                model_state_dict = {k.replace("_orig_mod.", ""): v for k, v in model_state_dict.items()}

            missing_keys, unexpected_keys = target_model.load_state_dict(model_state_dict, strict=False)
            if missing_keys: logger.warning(f"Missing keys when loading model state: {missing_keys}")
            if unexpected_keys: logger.warning(f"Unexpected keys when loading model state: {unexpected_keys}")


            # Load optimizer and scheduler states
            # Check LoRA config compatibility before loading optimizer/scheduler
            checkpoint_train_args = checkpoint.get("train_args", {})
            checkpoint_lora_rank = checkpoint_train_args.get("lora_rank", 0)
            current_lora_rank = getattr(args, 'lora_rank', 0)

            if checkpoint_lora_rank == current_lora_rank:
                if optimizer and "optimizer_state_dict" in checkpoint:
                    try:
                        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                        logger.info("Optimizer state loaded.")
                    except Exception as e: # Catch specific errors if possible
                        logger.warning(f"Could not load optimizer state: {e}. Reinitializing optimizer.")
                if scheduler and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None:
                    try:
                        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                        logger.info("Scheduler state loaded.")
                    except Exception as e:
                        logger.warning(f"Could not load scheduler state: {e}. Reinitializing scheduler.")
            else:
                logger.warning(f"LoRA rank mismatch (checkpoint: {checkpoint_lora_rank}, current: {current_lora_rank}). Skipping optimizer/scheduler load.")


            start_epoch = checkpoint.get("epoch", 0) + 1 # Resume from NEXT epoch
            start_step = checkpoint.get("step", 0) # global_step will continue from here
            min_val_loss = checkpoint.get("loss", float("inf")) # This is likely val_loss or train_loss

            logger.info(f"Resuming training from epoch {start_epoch}, global step {start_step}")

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}. Starting from scratch.")
            start_epoch, start_step, min_val_loss = 0, 0, float("inf") # Reset
    else:
        if checkpoint_to_load: # if a path was given but not found
            logger.warning(f"Checkpoint file not found: {checkpoint_to_load}. Starting from scratch.")
        else:
            logger.info("No checkpoint specified or found. Starting training from scratch.")

    return start_epoch, start_step, min_val_loss


def train_model_loop(model, train_dataloader, val_dataloader, tokenizer, args):
    device = torch.device(args.device)
    model.to(device)

    args.use_lora = args.lora_rank > 0
    optimizer_params = []
    if args.use_lora:
        logger.info(f"Configuring optimizer for LoRA training (rank={args.lora_rank}).")
        for name, param in model.named_parameters():
            if "lora_" in name or "ls_gamma" in name: # Train LoRA and LayerScale params
                if param.requires_grad: # Check if it's already set (it should be by model init)
                    optimizer_params.append(param)
                else: # If not, set it and add
                    param.requires_grad = True
                    optimizer_params.append(param)
            else:
                param.requires_grad = False # Freeze other params
    else:
        logger.info("Configuring optimizer for full model training.")
        for param in model.parameters():
            param.requires_grad = True # Ensure all params are trainable for full finetune
            optimizer_params.append(param)

    num_trainable_params = sum(p.numel() for p in optimizer_params)
    if num_trainable_params == 0:
        raise ValueError("No trainable parameters found for the optimizer. Check LoRA setup or model parameters.")
    logger.info(f"Number of parameters to be optimized: {num_trainable_params:,}")


    optimizer = AdamW(
        optimizer_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        fused=(args.adam_fused and device.type == "cuda" and torch.cuda.is_available())
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=args.lr_scheduler_patience)
    scaler = amp.GradScaler(enabled=(args.mixed_precision_dtype is not None and device.type == "cuda" and torch.cuda.is_available()))

    start_epoch, global_step, best_val_loss = load_checkpoint(model, optimizer, args, device, scheduler)

    if args.use_torch_compile and hasattr(torch, "compile"):
        logger.info(f"Attempting to compile model with torch.compile (mode: {args.torch_compile_mode})...")
        try:
            # Ensure we compile the unwrapped model if it's already wrapped (e.g. by DDP or previous compile)
            model_to_compile = model
            while hasattr(model_to_compile, "_orig_mod"): # Unwrap if multiply wrapped
                 model_to_compile = model_to_compile._orig_mod
            while hasattr(model_to_compile, "module"): # Unwrap DDP
                 model_to_compile = model_to_compile.module

            compiled_model = torch.compile(model_to_compile, mode=args.torch_compile_mode)

            # If original model was a wrapper (like DDP), re-wrap the compiled part
            if model is not model_to_compile:
                if hasattr(model, "_orig_mod"): model._orig_mod = compiled_model
                elif hasattr(model, "module"): model.module = compiled_model
                else: model = compiled_model # Fallback
            else:
                model = compiled_model
            logger.info("Model compiled successfully.")
        except Exception as e:
            logger.warning(f"Model compilation failed: {e}. Proceeding without compilation.")


    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        epoch_metrics = {"loss": 0.0, "perplexity": 0.0, "accuracy": 0.0, "count": 0}
        train_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs} (Step {global_step})")

        for batch_idx, batch in enumerate(train_iterator):
            input_ids = batch["input_ids"].to(device, non_blocking=args.pin_memory if device.type == "cuda" else False)
            attention_mask = batch["attention_mask"].to(device, non_blocking=args.pin_memory if device.type == "cuda" else False)

            optimizer.zero_grad(set_to_none=True)

            cast_dtype = torch.float32
            use_autocast = args.mixed_precision_dtype is not None and device.type == "cuda" and torch.cuda.is_available()
            if use_autocast:
                cast_dtype = torch.bfloat16 if args.mixed_precision_dtype == "bf16" else torch.float16

            with amp.autocast(device_type=device.type, enabled=use_autocast, dtype=cast_dtype):
                logits = model(input_ids, attention_mask=attention_mask)
                loss, perplexity, accuracy = compute_metrics(logits, input_ids, attention_mask)

            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(f"NaN or Inf loss encountered at Epoch {epoch+1}, Batch {batch_idx}. Skipping batch.")
                # Potentially skip optimizer step or stop training
                # For now, just log and continue, but this often indicates deeper issues.
                continue # Skip this batch

            scaler.scale(loss).backward()
            if args.grad_clip_norm > 0:
                scaler.unscale_(optimizer) # Unscale before clipping
                torch.nn.utils.clip_grad_norm_(optimizer_params, args.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            global_step += 1

            if torch.isfinite(loss): # Only accumulate finite metrics
                epoch_metrics["loss"] += loss.item()
                epoch_metrics["perplexity"] += perplexity.item() if torch.isfinite(perplexity) else 20 # Use a high finite value for Inf PPL
                epoch_metrics["accuracy"] += accuracy.item()
                epoch_metrics["count"] += 1


            train_iterator.set_postfix({
                "loss": f"{loss.item():.4f}" if torch.isfinite(loss) else "NaN",
                "ppl": f"{perplexity.item():.2f}" if torch.isfinite(perplexity) else "Inf",
                "acc": f"{accuracy.item():.3f}" if torch.isfinite(accuracy) else "NaN",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}"
            })

            if global_step % args.log_interval == 0:
                current_loss = loss.item() if torch.isfinite(loss) else float('nan')
                current_ppl = perplexity.item() if torch.isfinite(perplexity) else float('inf')
                current_acc = accuracy.item() if torch.isfinite(accuracy) else float('nan')
                logger.info(f"E{epoch+1} S{global_step} | Loss: {current_loss:.4f}, PPL: {current_ppl:.2f}, Acc: {current_acc:.3f}")

            if args.save_strategy == "steps" and global_step % args.save_steps == 0:
                current_eval_loss = loss.item() # Use current train loss as proxy if no validation
                is_best_step = False
                if args.validation_interval_steps > 0 and global_step % args.validation_interval_steps == 0 and val_dataloader:
                    val_loss, val_ppl, val_acc = evaluate_model(model, val_dataloader, args, device)
                    logger.info(f"Step {global_step} Validation - Loss: {val_loss:.4f}, PPL: {val_ppl:.2f}, Acc: {val_acc:.3f}")
                    scheduler.step(val_loss)
                    current_eval_loss = val_loss
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        is_best_step = True
                save_checkpoint(model, optimizer, epoch, global_step, current_eval_loss, args, is_best_step, scheduler)

        # End of epoch
        if epoch_metrics["count"] > 0:
            avg_epoch_loss = epoch_metrics["loss"] / epoch_metrics["count"]
            avg_epoch_ppl = epoch_metrics["perplexity"] / epoch_metrics["count"]
            avg_epoch_acc = epoch_metrics["accuracy"] / epoch_metrics["count"]
            logger.info(f"Epoch {epoch+1} Training Avg - Loss: {avg_epoch_loss:.4f}, PPL: {avg_epoch_ppl:.2f}, Acc: {avg_epoch_acc:.3f}")
        else:
            logger.warning(f"Epoch {epoch+1} completed with no valid batches processed for metrics.")


        if val_dataloader:
            val_loss, val_ppl, val_acc = evaluate_model(model, val_dataloader, args, device)
            logger.info(f"Epoch {epoch+1} Validation - Loss: {val_loss:.4f}, PPL: {val_ppl:.2f}, Acc: {val_acc:.3f}")
            scheduler.step(val_loss)
            is_best_epoch = val_loss < best_val_loss
            if is_best_epoch:
                best_val_loss = val_loss
                logger.info(f"New best validation loss: {best_val_loss:.4f}")

            if args.save_strategy == "epoch":
                save_checkpoint(model, optimizer, epoch, global_step, val_loss, args, is_best_epoch, scheduler)
        elif args.save_strategy == "epoch": # Save even if no validation
             save_checkpoint(model, optimizer, epoch, global_step, avg_epoch_loss if epoch_metrics["count"] > 0 else float('inf'), args, False, scheduler)


    logger.info("Training completed.")
    return model

def evaluate_model(model, dataloader, args, device):
    model.eval()
    total_loss, total_perplexity, total_accuracy, count = 0.0, 0.0, 0.0, 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device, non_blocking=args.pin_memory if device.type == "cuda" else False)
            attention_mask = batch["attention_mask"].to(device, non_blocking=args.pin_memory if device.type == "cuda" else False)

            cast_dtype = torch.float32
            use_autocast = args.mixed_precision_dtype is not None and device.type == "cuda" and torch.cuda.is_available()
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

    model.train() # Set back to train mode
    if count == 0:
        logger.warning("Evaluation resulted in no valid batches.")
        return float('inf'), float('inf'), 0.0
    return total_loss / count, total_perplexity / count, total_accuracy / count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Lunaris Codex model")
    # ... (Dataset arguments remain the same) ...
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
    model_group.add_argument("--vocab_size", type=int, default=None, help="Vocabulary size, inferred from tokenizer if None.")
    model_group.add_argument("--d_model", type=int, default=768)
    model_group.add_argument("--n_layers", type=int, default=10)
    model_group.add_argument("--n_heads", type=int, default=12)
    model_group.add_argument("--model_max_seq_len", type=int, default=1024)
    model_group.add_argument("--dropout", type=float, default=0.1)
    model_group.add_argument("--activation", type=str, default="swiglu", choices=["swiglu", "gelu"])
    model_group.add_argument("--lora_rank", type=int, default=0, help="Rank for LoRA. 0 means full finetuning.")
    # No need for ff_multiplier here, it's internal to LunarisCodexConfig or can be added if you want to override default

    # ... (Training, Checkpoint, Optimization arguments remain largely the same structure) ...
    train_group = parser.add_argument_group("Training")
    train_group.add_argument("--num_epochs", type=int, default=3)
    train_group.add_argument("--batch_size", type=int, default=16)
    train_group.add_argument("--learning_rate", type=float, default=5e-5)
    train_group.add_argument("--lr_scheduler_patience", type=int, default=2)
    train_group.add_argument("--weight_decay", type=float, default=0.01)
    train_group.add_argument("--adam_fused", action="store_true", help="Use fused AdamW if available (CUDA only).")
    train_group.add_argument("--grad_clip_norm", type=float, default=1.0, help="Max norm for gradient clipping. 0 to disable.")
    train_group.add_argument("--mixed_precision_dtype", type=str, default=None, choices=["fp16", "bf16"], help="Enable mixed precision training with 'fp16' or 'bf16'.")
    train_group.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    train_group.add_argument("--num_workers", type=int, default=0, help="Number of DataLoader workers.")
    train_group.add_argument("--pin_memory", action="store_true", help="Use pinned memory for DataLoader (CUDA only).")
    train_group.add_argument("--seed", type=int, default=42)

    ckpt_group = parser.add_argument_group("Checkpoints")
    ckpt_group.add_argument("--checkpoint_dir", type=str, default="checkpoints_lunaris")
    ckpt_group.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume from.")
    ckpt_group.add_argument("--save_strategy", type=str, default="epoch", choices=["epoch", "steps"])
    ckpt_group.add_argument("--save_steps", type=int, default=1000, help="Save checkpoint every N steps if save_strategy is 'steps'.")
    ckpt_group.add_argument("--log_interval", type=int, default=100, help="Log training metrics every N steps.")
    ckpt_group.add_argument("--validation_interval_steps", type=int, default=0, help="Run validation every N steps if save_strategy is 'steps'. 0 to disable step-based validation.")

    opt_group = parser.add_argument_group("Optimization")
    opt_group.add_argument("--use_torch_compile", action="store_true", help="Enable torch.compile for model optimization.")
    opt_group.add_argument("--torch_compile_mode", type=str, default="reduce-overhead", help="Mode for torch.compile.")
    opt_group.add_argument("--allow_tf32", action="store_true", help="Allow TF32 on Ampere GPUs and newer.")
    opt_group.add_argument("--cudnn_benchmark", action="store_true", help="Enable cuDNN benchmark mode.")

    args = parser.parse_args()

    if args.allow_tf32 and args.device == "cuda" and torch.cuda.is_available():
        if torch.cuda.get_device_capability()[0] >= 8: # Check for Ampere or newer
            torch.backends.cuda.matmul.allow_tf32 = True
            logger.info("TF32 matmul support enabled for CUDA.")
        else:
            logger.info("TF32 matmul support not enabled (requires Ampere GPU or newer).")

    if args.cudnn_benchmark and args.device == "cuda" and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        logger.info("cuDNN benchmark mode enabled.")

    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
    if tokenizer.pad_token_id is None:
        logger.warning("Tokenizer does not have a pad_token_id.")
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            logger.info(f"Using eos_token_id ({tokenizer.eos_token_id}) as pad_token_id.")
        else:
            # Add a new pad token if no EOS token exists either
            added_tokens = tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            if added_tokens > 0:
                 logger.info(f"Added new pad_token: '[PAD]' (ID: {tokenizer.pad_token_id}). Vocab size potentially increased.")
            else: # Should not happen if pad_token is not already there
                 logger.warning(f"Attempted to add [PAD] token but it might already exist or failed. Pad ID: {tokenizer.pad_token_id}")

    if tokenizer.pad_token_id is None: # Still None after attempts
        raise ValueError("Could not set a pad_token_id for the tokenizer. Please ensure your tokenizer has one, or can have one added.")


    train_dataset = MemmapCodeDataset(
        args.memmap_file_train, args.num_sequences_train,
        args.dataset_max_length, tokenizer.pad_token_id, args.dataset_dtype
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(args.pin_memory and args.device == "cuda" and torch.cuda.is_available()),
        drop_last=True # Good practice for consistent batch sizes, especially with AMP/DDP
    )

    val_dataloader = None
    if args.memmap_file_val and args.num_sequences_val > 0:
        val_dataset = MemmapCodeDataset(
            args.memmap_file_val, args.num_sequences_val,
            args.dataset_max_length, tokenizer.pad_token_id, args.dataset_dtype
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=(args.pin_memory and args.device == "cuda" and torch.cuda.is_available())
        )

    vocab_size_to_use = args.vocab_size if args.vocab_size is not None else len(tokenizer)

    # CRITICAL CHANGE: Pass pad_token_id to LunarisCodexConfig
    config = LunarisCodexConfig(
        vocab_size=vocab_size_to_use,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        max_seq_len=args.model_max_seq_len,
        dropout=args.dropout,
        activation=args.activation,
        lora_rank=args.lora_rank,
        pad_token_id=tokenizer.pad_token_id # <-- Ensure this is passed
    )
    model = LunarisMind(config)

    # Logging model parameter information based on LoRA status occurs inside train_model_loop
    # when setting up optimizer_params, which is more accurate after model initialization.

    trained_model = train_model_loop(model, train_dataloader, val_dataloader, tokenizer, args)
