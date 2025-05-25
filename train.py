import argparse
import hashlib
import logging
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch import amp
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

# Project-specific imports
from model import LunarisCodexConfig, LunarisMind, count_parameters

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def set_seed(seed_value=42):
    """
    Sets random seeds for Python, NumPy, and PyTorch to ensure reproducible results.
    
    If CUDA is available, also sets the seed for all CUDA devices.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

def compute_sha256(filepath):
    """
    Computes the SHA-256 hash of a file.
    
    Args:
        filepath: Path to the file to hash.
    
    Returns:
        The hexadecimal SHA-256 hash string if successful, or None if an error occurs.
    """
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
        """
        Initializes a memory-mapped dataset for tokenized sequences.
        
        Loads a memmap file containing pre-tokenized sequences for efficient access during training or evaluation. Raises FileNotFoundError if the file does not exist, and logs errors on loading issues.
        
        Args:
            memmap_file: Path to the memory-mapped file containing tokenized data.
            num_sequences: Number of sequences stored in the memmap file.
            max_length: Maximum sequence length for each entry.
            tokenizer_pad_id: Token ID used for padding in the dataset.
            dtype_str: Data type of the stored tokens, either "int16" or "int32".
        """
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
        logger.info(f"Dataset loaded successfully. Pad ID: {self.tokenizer_pad_id}")

    def __len__(self):
        """
        Returns the number of sequences in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves a tokenized sequence and its attention mask at the specified index.
        
        Args:
            idx: Index of the sequence to retrieve.
        
        Returns:
            A dictionary containing:
                - 'input_ids': Tensor of token IDs for the sequence.
                - 'attention_mask': Tensor indicating non-padding tokens (1 for tokens, 0 for padding).
        """
        input_ids = torch.from_numpy(np.array(self.data[idx], dtype=np.int64))
        attention_mask = (input_ids != self.tokenizer_pad_id).long()
        return {"input_ids": input_ids, "attention_mask": attention_mask}

def compute_metrics(logits, targets, attention_mask):
    """
    Calculates average cross-entropy loss, perplexity, and accuracy over non-padded tokens.
    
    The function shifts logits, targets, and attention masks to align predictions with targets, then computes metrics only on positions where the attention mask is active (i.e., not padding). If no active tokens are present, returns zero loss, infinite perplexity, and zero accuracy.
    
    Args:
        logits: Model output logits of shape (batch_size, seq_len, vocab_size).
        targets: Target token indices of shape (batch_size, seq_len).
        attention_mask: Attention mask of shape (batch_size, seq_len), where 1 indicates valid tokens.
    
    Returns:
        A tuple of (average loss, perplexity, accuracy) computed over active (non-padded) tokens.
    """
    logits_shifted = logits[..., :-1, :].contiguous()
    targets_shifted = targets[..., 1:].contiguous()
    attention_mask_shifted = attention_mask[..., 1:].contiguous()

    logits_flat = logits_shifted.view(-1, logits_shifted.size(-1))
    targets_flat = targets_shifted.view(-1)
    active_mask = attention_mask_shifted.view(-1).bool()

    if not active_mask.any():
        logger.warning("No active tokens for loss calculation")
        device = logits.device
        return (torch.tensor(0.0, device=device),
                torch.tensor(float("inf"), device=device),
                torch.tensor(0.0, device=device))

    logits_active = logits_flat[active_mask]
    targets_active = targets_flat[active_mask]

    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    loss = loss_fn(logits_active, targets_active)
    num_active_tokens = active_mask.sum()
    avg_loss = loss / num_active_tokens

    perplexity = torch.exp(torch.clamp(avg_loss, max=20))  # Clamp to prevent overflow
    preds = torch.argmax(logits_active, dim=-1)
    accuracy = (preds == targets_active).float().mean()

    return avg_loss, perplexity, accuracy

def save_checkpoint(model, optimizer, epoch, step, current_loss, args, is_best=False, scheduler=None):
    """
    Saves the current model, optimizer, and scheduler states to a checkpoint file with SHA-256 integrity verification.
    
    The checkpoint includes model weights, optimizer and scheduler states, training progress, configuration, and metadata. A SHA-256 hash file is written alongside the checkpoint for later integrity checks. If `is_best` is True, also saves a copy as `best_model.pt` with its own hash.
    
    Args:
        epoch: The current training epoch (zero-based).
        step: The current training step.
        current_loss: The loss value at the time of checkpointing.
        is_best: If True, saves an additional copy as the best model checkpoint.
        scheduler: Optional learning rate scheduler to save state for.
    
    Raises:
        Exception: If saving the checkpoint or hash file fails.
    """
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    base_filename = f"lunaris_codex_epoch-{epoch+1}_step-{step}"
    checkpoint_path = os.path.join(args.checkpoint_dir, f"{base_filename}.pt")

    # Get the actual model (handle torch.compile wrapper)
    model_to_save = model._orig_mod if hasattr(model, "_orig_mod") else model

    checkpoint_data = {
        "model_state_dict": model_to_save.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "step": step,
        "loss": current_loss,
        "config": model_to_save.config.__dict__,
        "args": vars(args),
        "torch_version": torch.__version__,
        "model_class": model_to_save.__class__.__name__
    }

    try:
        torch.save(checkpoint_data, checkpoint_path)

        # Compute and store SHA-256 hash
        file_hash = compute_sha256(checkpoint_path)
        if file_hash:
            hash_file = checkpoint_path + ".sha256"
            with open(hash_file, "w") as f:
                f.write(f"{file_hash}  {os.path.basename(checkpoint_path)}\n")

        logger.info(f"Checkpoint saved: {checkpoint_path}")

        if is_best:
            best_path = os.path.join(args.checkpoint_dir, "best_model.pt")
            torch.save(checkpoint_data, best_path)
            if file_hash:
                with open(best_path + ".sha256", "w") as f:
                    f.write(f"{compute_sha256(best_path)}  best_model.pt\n")
            logger.info(f"Best checkpoint saved: {best_path}")

    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")
        raise

def verify_checkpoint_integrity(checkpoint_path):
    """
    Verifies the integrity of a checkpoint file by comparing its SHA-256 hash to a stored hash.
    
    If the corresponding `.sha256` file is missing or verification fails due to an error, the function assumes the checkpoint is valid and returns True. Returns False only if the hash file exists and the hashes do not match.
    """
    hash_file = checkpoint_path + ".sha256"
    if not os.path.exists(hash_file):
        logger.warning(f"No hash file found for {checkpoint_path}")
        return True  # Assume valid if no hash file

    try:
        with open(hash_file, "r") as f:
            expected_hash = f.read().split()[0]

        actual_hash = compute_sha256(checkpoint_path)
        if actual_hash and actual_hash == expected_hash:
            logger.info(f"Checkpoint integrity verified: {checkpoint_path}")
            return True
        else:
            logger.error(f"Checkpoint integrity check failed: {checkpoint_path}")
            return False
    except Exception as e:
        logger.warning(f"Could not verify checkpoint integrity: {e}")
        return True  # Assume valid if verification fails

def load_checkpoint(model, optimizer, args, device, scheduler=None):
    """
    Loads model, optimizer, and scheduler states from a checkpoint file after verifying its integrity.
    
    If a checkpoint is specified or found in the checkpoint directory, verifies its SHA-256 hash before loading. Handles compatibility between compiled and uncompiled model state dictionaries. Loads optimizer and scheduler states only if the LoRA configuration matches between the checkpoint and current run. Returns the starting epoch, step, and minimum validation loss for resuming training. If loading fails or no checkpoint is found, returns defaults to start training from scratch.
    
    Returns:
        Tuple of (start_epoch, start_step, min_val_loss) for resuming training.
    """
    start_epoch, start_step, min_val_loss = 0, 0, float("inf")

    checkpoint_to_load = args.resume_from_checkpoint
    if not checkpoint_to_load and args.checkpoint_dir:
        best_checkpoint = os.path.join(args.checkpoint_dir, "best_model.pt")
        if os.path.isfile(best_checkpoint):
            logger.info("Loading best_model.pt")
            checkpoint_to_load = best_checkpoint

    if checkpoint_to_load and os.path.isfile(checkpoint_to_load):
        # Verify integrity before loading
        if not verify_checkpoint_integrity(checkpoint_to_load):
            logger.error(f"Checkpoint integrity verification failed: {checkpoint_to_load}")
            return start_epoch, start_step, min_val_loss

        logger.info(f"Loading checkpoint: {checkpoint_to_load}")
        try:
            checkpoint = torch.load(checkpoint_to_load, map_location=device, weights_only=False)

            # Load model state
            model_state_dict = checkpoint["model_state_dict"]
            target_model = model._orig_mod if hasattr(model, "_orig_mod") else model

            # Handle potential state dict key mismatches
            if hasattr(model, "_orig_mod") and not any(k.startswith("_orig_mod.") for k in model_state_dict.keys()):
                # Checkpoint from uncompiled model, current model is compiled
                pass
            elif not hasattr(model, "_orig_mod") and any(k.startswith("_orig_mod.") for k in model_state_dict.keys()):
                # Checkpoint from compiled model, current model is uncompiled
                model_state_dict = {k.replace("_orig_mod.", ""): v for k, v in model_state_dict.items()}

            missing_keys, unexpected_keys = target_model.load_state_dict(model_state_dict, strict=False)
            if missing_keys:
                logger.warning(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys: {unexpected_keys}")

            # Load optimizer and scheduler states (with compatibility check)
            current_lora = getattr(args, 'lora_rank', 0) > 0
            checkpoint_lora = checkpoint.get("args", {}).get("lora_rank", 0) > 0

            if current_lora == checkpoint_lora:
                if optimizer and "optimizer_state_dict" in checkpoint:
                    try:
                        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                        logger.info("Optimizer state loaded")
                    except Exception as e:
                        logger.warning(f"Failed to load optimizer state: {e}")

                if scheduler and checkpoint.get("scheduler_state_dict"):
                    try:
                        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                        logger.info("Scheduler state loaded")
                    except Exception as e:
                        logger.warning(f"Failed to load scheduler state: {e}")
            else:
                logger.info("LoRA configuration mismatch - skipping optimizer/scheduler states")

            start_epoch = checkpoint.get("epoch", 0)
            start_step = checkpoint.get("step", 0)
            min_val_loss = checkpoint.get("loss", float("inf"))

            logger.info(f"Resuming from epoch {start_epoch + 1}, step {start_step}")

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            start_epoch, start_step, min_val_loss = 0, 0, float("inf")
    else:
        logger.info("Starting training from scratch")

    return start_epoch, start_step, min_val_loss

def train_model_loop(model, train_dataloader, val_dataloader, tokenizer, args):
    """
    Trains the model using the provided data loaders, optimizer, and configuration.
    
    Runs the main training loop with support for mixed precision, gradient clipping, checkpointing, validation, and optional model compilation. Tracks and logs training and validation metrics, saves checkpoints according to the configured strategy, and handles best model tracking. Raises an exception if invalid loss values are encountered.
    
    Args:
        train_dataloader: DataLoader yielding training batches.
        val_dataloader: DataLoader yielding validation batches, or None to skip validation.
        tokenizer: Tokenizer used for model input preparation.
        args: Namespace containing training and optimization configuration.
    
    Returns:
        The trained model, potentially compiled if requested.
    """
    device = torch.device(args.device)
    model.to(device)

    # Setup optimizer parameters
    args.use_lora = getattr(args, 'lora_rank', 0) > 0
    if args.use_lora:
        optimizer_params = [p for n, p in model.named_parameters() if "lora_" in n and p.requires_grad]
    else:
        optimizer_params = [p for p in model.parameters() if p.requires_grad]

    if not optimizer_params:
        raise ValueError("No trainable parameters found!")

    logger.info(f"Optimizing {len(optimizer_params)} parameter groups")

    # Setup training components
    optimizer = AdamW(
        optimizer_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        fused=(args.adam_fused and device.type == "cuda")
    )

    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.lr_scheduler_patience
    )

    scaler = amp.GradScaler(enabled=(args.mixed_precision_dtype is not None and device.type == "cuda"))

    # Load checkpoint if available
    start_epoch, global_step, best_val_loss = load_checkpoint(model, optimizer, args, device, scheduler)

    # Compile model if requested
    if args.use_torch_compile and hasattr(torch, "compile"):
        logger.info("Compiling model with torch.compile...")
        try:
            model_to_compile = model._orig_mod if hasattr(model, "_orig_mod") else model
            compiled_model = torch.compile(model_to_compile, mode=args.torch_compile_mode)

            if hasattr(model, "_orig_mod"):
                model._orig_mod = compiled_model
            else:
                model = compiled_model
            logger.info("Model compiled successfully")
        except Exception as e:
            logger.warning(f"Failed to compile model: {e}")

    # Training loop
    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        epoch_metrics = {"loss": 0.0, "perplexity": 0.0, "accuracy": 0.0}

        train_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}")

        for batch_idx, batch in enumerate(train_iterator):
            input_ids = batch["input_ids"].to(device, non_blocking=args.pin_memory)
            attention_mask = batch["attention_mask"].to(device, non_blocking=args.pin_memory)

            optimizer.zero_grad(set_to_none=True)

            # Mixed precision settings
            cast_dtype = torch.float32
            use_autocast = args.mixed_precision_dtype is not None and device.type == "cuda"
            if use_autocast:
                cast_dtype = torch.bfloat16 if args.mixed_precision_dtype == "bf16" else torch.float16

            with amp.autocast(device_type=device.type, enabled=use_autocast, dtype=cast_dtype):
                logits = model(input_ids, attention_mask=attention_mask)
                loss, perplexity, accuracy = compute_metrics(logits, input_ids, attention_mask)

            # Check for invalid loss
            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(f"Invalid loss at epoch {epoch+1}, batch {batch_idx}")
                raise RuntimeError("Training stopped due to invalid loss")

            # Backward pass
            scaler.scale(loss).backward()

            if args.grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(optimizer_params, max_norm=args.grad_clip_norm)

            scaler.step(optimizer)
            scaler.update()
            global_step += 1

            # Update metrics
            epoch_metrics["loss"] += loss.item()
            epoch_metrics["perplexity"] += perplexity.item() if torch.isfinite(perplexity) else 0
            epoch_metrics["accuracy"] += accuracy.item()

            # Update progress bar
            train_iterator.set_postfix({
                "loss": f"{loss.item():.4f}",
                "ppl": f"{perplexity.item():.2f}" if torch.isfinite(perplexity) else "inf",
                "acc": f"{accuracy.item():.3f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}"
            })

            # Logging and validation
            if global_step % args.log_interval == 0:
                logger.info(f"E{epoch+1} S{global_step} | Loss: {loss.item():.4f}, PPL: {perplexity.item():.2f}, Acc: {accuracy.item():.3f}")

            # Step-based saving
            if args.save_strategy == "steps" and global_step % args.save_steps == 0:
                val_loss = loss.item()
                is_best = False

                if args.validation_interval_steps > 0 and global_step % args.validation_interval_steps == 0 and val_dataloader:
                    val_loss, _, _ = evaluate_model(model, val_dataloader, args, device)
                    scheduler.step(val_loss)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        is_best = True

                save_checkpoint(model, optimizer, epoch, global_step, val_loss, args, is_best, scheduler)

        # End of epoch processing
        num_batches = len(train_dataloader)
        avg_metrics = {k: v / num_batches for k, v in epoch_metrics.items()}
        logger.info(f"Epoch {epoch+1} Training - Loss: {avg_metrics['loss']:.4f}, PPL: {avg_metrics['perplexity']:.2f}, Acc: {avg_metrics['accuracy']:.3f}")

        # Validation
        if val_dataloader:
            val_loss, val_ppl, val_acc = evaluate_model(model, val_dataloader, args, device)
            logger.info(f"Epoch {epoch+1} Validation - Loss: {val_loss:.4f}, PPL: {val_ppl:.2f}, Acc: {val_acc:.3f}")

            scheduler.step(val_loss)
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                logger.info(f"New best validation loss: {best_val_loss:.4f}")

            if args.save_strategy == "epoch":
                save_checkpoint(model, optimizer, epoch, global_step, val_loss, args, is_best, scheduler)
        elif args.save_strategy == "epoch":
            save_checkpoint(model, optimizer, epoch, global_step, avg_metrics['loss'], args, False, scheduler)

    logger.info("Training completed successfully!")
    return model

def evaluate_model(model, dataloader, args, device):
    """
    Evaluates the model on a validation or test dataset and computes average loss, perplexity, and accuracy.
    
    Runs the model in evaluation mode without gradient computation, processes each batch with optional mixed precision, and aggregates metrics across all batches.
    
    Args:
        model: The model to evaluate.
        dataloader: DataLoader providing evaluation data batches.
        args: Configuration object with evaluation and precision settings.
        device: Device on which to perform evaluation.
    
    Returns:
        A tuple of (average loss, average perplexity, average accuracy) over the dataset.
    """
    model.eval()
    total_loss, total_perplexity, total_accuracy = 0.0, 0.0, 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
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
                total_perplexity += perplexity.item() if torch.isfinite(perplexity) else 0
                total_accuracy += accuracy.item()

    num_batches = len(dataloader)
    model.train()
    return (total_loss / num_batches, total_perplexity / num_batches, total_accuracy / num_batches)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Lunaris Codex model")

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
    train_group.add_argument("--batch_size", type=int, default=16)
    train_group.add_argument("--learning_rate", type=float, default=5e-5)
    train_group.add_argument("--lr_scheduler_patience", type=int, default=2)
    train_group.add_argument("--weight_decay", type=float, default=0.01)
    train_group.add_argument("--adam_fused", action="store_true")
    train_group.add_argument("--grad_clip_norm", type=float, default=1.0)
    train_group.add_argument("--mixed_precision_dtype", type=str, choices=["fp16", "bf16"])
    train_group.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
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

    # PyTorch optimization arguments
    opt_group = parser.add_argument_group("Optimization")
    opt_group.add_argument("--use_torch_compile", action="store_true")
    opt_group.add_argument("--torch_compile_mode", type=str, default="reduce-overhead")
    opt_group.add_argument("--allow_tf32", action="store_true")
    opt_group.add_argument("--cudnn_benchmark", action="store_true")

    args = parser.parse_args()

    # Setup PyTorch optimizations
    if args.allow_tf32 and args.device == "cuda" and torch.cuda.is_available():
        if torch.cuda.get_device_capability()[0] >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            logger.info("TF32 enabled for CUDA matmuls")

    if args.cudnn_benchmark and args.device == "cuda":
        torch.backends.cudnn.benchmark = True
        logger.info("cuDNN benchmark enabled")

    set_seed(args.seed)

    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.add_special_tokens({"pad_token": "<|PAD|>"})
            logger.warning(f"Added pad_token, new vocab size: {len(tokenizer)}")

    # Setup datasets
    train_dataset = MemmapCodeDataset(
        args.memmap_file_train, args.num_sequences_train,
        args.dataset_max_length, tokenizer.pad_token_id, args.dataset_dtype
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(args.pin_memory and args.device == "cuda")
    )

    val_dataloader = None
    if args.memmap_file_val and args.num_sequences_val > 0:
        val_dataset = MemmapCodeDataset(
            args.memmap_file_val, args.num_sequences_val,
            args.dataset_max_length, tokenizer.pad_token_id, args.dataset_dtype
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=(args.pin_memory and args.device == "cuda")
        )

    # Setup model
    vocab_size = args.vocab_size if args.vocab_size else len(tokenizer)
    config = LunarisCodexConfig(
        vocab_size=vocab_size, d_model=args.d_model, n_layers=args.n_layers,
        n_heads=args.n_heads, max_seq_len=args.model_max_seq_len,
        dropout=args.dropout, activation=args.activation, lora_rank=args.lora_rank
    )
    model = LunarisMind(config)

    # Configure LoRA if enabled
    if config.lora_rank > 0:
        logger.info(f"LoRA training enabled (rank={config.lora_rank})")
        for name, param in model.named_parameters():
            param.requires_grad = "lora_" in name

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Trainable: {trainable_params:,}/{total_params:,} ({trainable_params/total_params*100:.2f}%)")
    else:
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Full training - Total parameters: {total_params:,}")

    # Start training
    trained_model = train_model_loop(model, train_dataloader, val_dataloader, tokenizer, args)
