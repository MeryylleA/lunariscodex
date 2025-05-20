# train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import amp
import os
import argparse
import logging
import random
import sys 
from tqdm import tqdm

# Project-specific imports
from model import LunarisMind, LunarisCodexConfig, count_parameters

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def set_seed(seed_value=42):
    """Sets the seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

class MemmapCodeDataset(Dataset):
    def __init__(self, memmap_file, num_sequences, max_length=1024, tokenizer_pad_id=0, dtype_str="int32"):
        logger.info(f"Loading dataset from {memmap_file} with {num_sequences} sequences and max_length {max_length}")
        dtype = np.int16 if dtype_str == "int16" else np.int32
        try:
            self.data = np.memmap(memmap_file, dtype=dtype, mode='r', shape=(num_sequences, max_length))
        except FileNotFoundError:
            logger.error(f"Memmap file not found: {memmap_file}"); 
            raise
        except ValueError as e:
            logger.error(f"Error loading memmap (check shape/dtype?): {memmap_file} - {e}"); raise
        
        self.max_length = max_length
        self.tokenizer_pad_id = tokenizer_pad_id 
        logger.info(f"Dataset loaded. Pad ID for attention mask: {self.tokenizer_pad_id}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_ids = torch.from_numpy(np.array(self.data[idx], dtype=np.int64))
        attention_mask = (input_ids != self.tokenizer_pad_id).long() 
        return {"input_ids": input_ids, "attention_mask": attention_mask}

def compute_metrics(logits, targets, attention_mask, pad_token_id_for_loss_ignore=-100):
    """Computes loss, perplexity, and top-1 accuracy, ignoring padded tokens."""
    logits_shifted = logits[..., :-1, :].contiguous()
    targets_shifted = targets[..., 1:].contiguous()
    attention_mask_shifted = attention_mask[..., 1:].contiguous() 

    targets_masked = targets_shifted.masked_fill(~attention_mask_shifted.bool(), pad_token_id_for_loss_ignore)

    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id_for_loss_ignore, reduction='sum')
    loss = loss_fn(logits_shifted.view(-1, logits_shifted.size(-1)), targets_masked.view(-1))
    
    num_active_tokens = attention_mask_shifted.sum()
    if num_active_tokens.item() == 0: 
        return torch.tensor(0.0, device=logits.device), torch.tensor(float('inf'), device=logits.device), torch.tensor(0.0, device=logits.device)

    avg_loss = loss / num_active_tokens
    perplexity = torch.exp(avg_loss)
    
    preds = torch.argmax(logits_shifted, dim=-1)
    correct_preds = (preds == targets_shifted) & attention_mask_shifted.bool() 
    
    accuracy = correct_preds.sum().float() / num_active_tokens
    return avg_loss, perplexity, accuracy

def save_checkpoint(model, optimizer, epoch, step, current_loss, args, is_best=False, scheduler=None):
    """Saves training checkpoint."""
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    base_filename_no_ext = f"lunaris_codex_epoch-{epoch+1}_step-{step}" 
    checkpoint_path = os.path.join(args.checkpoint_dir, f"{base_filename_no_ext}.pt")

    model_to_save = model._orig_mod if hasattr(model, '_orig_mod') and isinstance(model._orig_mod, nn.Module) else model

    checkpoint_content = {
        "model_state_dict": model_to_save.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch, 
        "step": step,
        "loss": current_loss, 
        "config": model_to_save.config.__dict__, 
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "args": vars(args) 
    }

    torch.save(checkpoint_content, checkpoint_path)
    logger.info(f"Checkpoint saved to {checkpoint_path}")

    if is_best:
        best_path = os.path.join(args.checkpoint_dir, "best_model.pt")
        torch.save(checkpoint_content, best_path)
        logger.info(f"Best checkpoint (val_loss: {current_loss:.4f}) saved to {best_path}")

def load_checkpoint(model, optimizer, args, device, scheduler=None):
    """Loads training checkpoint with enhanced logging."""
    start_epoch, start_step, min_val_loss = 0, 0, float('inf')
    
    checkpoint_to_load = args.resume_from_checkpoint
    if not checkpoint_to_load and args.checkpoint_dir: 
        potential_best_checkpoint = os.path.join(args.checkpoint_dir, "best_model.pt")
        if os.path.isfile(potential_best_checkpoint):
            logger.info(f"No specific checkpoint provided via --resume_from_checkpoint. Found 'best_model.pt' in {args.checkpoint_dir}. Attempting to load it.")
            checkpoint_to_load = potential_best_checkpoint
        else:
            logger.info(f"No specific checkpoint provided and 'best_model.pt' not found in {args.checkpoint_dir}. Looking for other .pt files.")
            if os.path.isdir(args.checkpoint_dir): # Check if checkpoint_dir actually exists
                pt_files = [f for f in os.listdir(args.checkpoint_dir) if f.endswith('.pt') and f != "best_model.pt"]
                if pt_files:
                    latest_checkpoint = sorted(pt_files, reverse=True)[0] 
                    checkpoint_to_load = os.path.join(args.checkpoint_dir, latest_checkpoint)
                    logger.info(f"Attempting to load latest available checkpoint: {checkpoint_to_load}")
            else:
                logger.info(f"Checkpoint directory {args.checkpoint_dir} does not exist. Cannot search for latest checkpoint.")


    if checkpoint_to_load and os.path.isfile(checkpoint_to_load):
        logger.info(f"Loading checkpoint from: {checkpoint_to_load}")
        try:
            checkpoint = torch.load(checkpoint_to_load, map_location=device)
            
            model_state_dict = checkpoint.get("model_state_dict")
            if model_state_dict:
                current_model_is_compiled = hasattr(model, '_orig_mod') and isinstance(model._orig_mod, nn.Module)
                checkpoint_was_from_compiled = any(k.startswith("_orig_mod.") for k in model_state_dict.keys())
                target_model_for_load_state_dict = model._orig_mod if current_model_is_compiled else model
                
                if not current_model_is_compiled and checkpoint_was_from_compiled:
                    logger.info("Checkpoint is from a compiled model, but current model is not. Stripping '_orig_mod.' prefix.")
                    model_state_dict = {k.replace('_orig_mod.', ''): v for k, v in model_state_dict.items()}
                
                missing_keys, unexpected_keys = target_model_for_load_state_dict.load_state_dict(model_state_dict, strict=False)
                if missing_keys: logger.warning(f"Missing keys when loading model_state_dict: {missing_keys}")
                if unexpected_keys: logger.warning(f"Unexpected keys when loading model_state_dict: {unexpected_keys}")
                logger.info("Model state loaded successfully.")
            else:
                logger.warning("Model state_dict not found in checkpoint.")

            if optimizer and "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                logger.info("Optimizer state loaded successfully.")
            elif optimizer:
                logger.warning("Optimizer state_dict not found in checkpoint. Optimizer will start fresh.")

            if scheduler and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                logger.info("Scheduler state loaded successfully.")
            elif scheduler:
                logger.warning("Scheduler state_dict not found in checkpoint. Scheduler will start fresh (if applicable).")
            
            start_epoch = checkpoint.get("epoch", -1) + 1 
            start_step = checkpoint.get("step", 0)
            min_val_loss_from_ckpt = checkpoint.get("loss", float('inf')) 
            
            logger.info(f"Checkpoint states loaded. Training will resume/start from epoch {start_epoch} (0-indexed), global step {start_step}.")
            logger.info(f"  Previous loss (often val_loss or last train loss) from checkpoint: {min_val_loss_from_ckpt:.4f}")
            # Set best_val_loss to the one from checkpoint only if we are actually resuming meaningfully
            # If starting epoch 0, we are essentially starting fresh or this is the initial load.
            if start_epoch > 0 : # Check if it's a meaningful resume. start_epoch is 1-indexed here for loop.
                 min_val_loss = min_val_loss_from_ckpt
                 logger.info(f"  Setting current best_val_loss to checkpoint's loss: {min_val_loss:.4f}")


            if "config" in checkpoint:
                logger.info(f"  Model config from checkpoint (for reference): {checkpoint['config']}")
            if "args" in checkpoint:
                logger.info(f"  Training args from checkpoint (for reference): {checkpoint['args']}")

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}. Starting training from scratch.", exc_info=True)
            start_epoch, start_step, min_val_loss = 0, 0, float('inf')
    else:
        if args.resume_from_checkpoint: 
             logger.warning(f"Specified checkpoint {args.resume_from_checkpoint} not found. Starting training from scratch.")
        else:
             logger.info("No checkpoint found or specified. Starting training from scratch.")
        start_epoch, start_step, min_val_loss = 0, 0, float('inf') 
        
    return start_epoch, start_step, min_val_loss

def train_model_loop(model, train_dataloader, val_dataloader, tokenizer, args):
    device = torch.device(args.device)
    model.to(device)

    optimizer_params = []
    if args.use_lora:
        for name, param in model.named_parameters():
            if 'lora_' in name and param.requires_grad: 
                optimizer_params.append(param)
    else: 
        for param in model.parameters():
            if param.requires_grad:
                optimizer_params.append(param)
    
    if not optimizer_params:
        logger.error("No trainable parameters found for the optimizer! Check LoRA setup or model's requires_grad flags."); 
        return model 
    logger.info(f"Optimizing {len(optimizer_params)} parameter tensors.")

    optimizer = AdamW(optimizer_params, lr=args.learning_rate, weight_decay=args.weight_decay, 
                      fused=(args.adam_fused and device.type == 'cuda')) 
    # REMOVED verbose=True from ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=args.lr_scheduler_patience) 
    scaler = amp.GradScaler(enabled=(args.mixed_precision_dtype is not None and device.type == 'cuda'))
    
    start_epoch_for_loop, global_step, best_val_loss = load_checkpoint(model, optimizer, args, device, scheduler)
    # start_epoch_for_loop is 0-indexed if starting fresh, or (saved_epoch + 1) if resuming.
    # The loop range should be `range(start_epoch_for_loop, args.num_epochs)`

    if args.use_torch_compile and hasattr(torch, 'compile'):
        logger.info(f"Compiling model with torch.compile (mode: {args.torch_compile_mode})...")
        try:
            model_to_compile = model._orig_mod if hasattr(model, '_orig_mod') and isinstance(model._orig_mod, nn.Module) else model
            compiled_part = torch.compile(model_to_compile, mode=args.torch_compile_mode)
            if hasattr(model, '_orig_mod') and isinstance(model._orig_mod, nn.Module):
                model._orig_mod = compiled_part
            else:
                model = compiled_part
            logger.info("Model compiled successfully.")
        except Exception as e:
            logger.error(f"Failed to compile model: {e}. Continuing without compilation.", exc_info=True)

    for epoch in range(start_epoch_for_loop, args.num_epochs): 
        model.train()
        epoch_loss, epoch_perplexity, epoch_top1_acc = 0.0, 0.0, 0.0
        
        # Display epoch as 1-indexed in tqdm and logs for user-friendliness
        train_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Training]") 
        
        for batch_idx, batch in enumerate(train_iterator):
            input_ids = batch["input_ids"].to(device, non_blocking=args.pin_memory)
            attention_mask = batch["attention_mask"].to(device, non_blocking=args.pin_memory)

            # Log first batch info only if it's truly the first batch of the entire training run (not resuming mid-first-epoch)
            if epoch == 0 and start_epoch_for_loop == 0 and batch_idx == 0: 
                logger.info(f"First batch processing - Epoch {epoch+1}, Shape: {input_ids.shape}, Pad ID (for attention mask): {tokenizer.pad_token_id}")

            optimizer.zero_grad(set_to_none=True)
            
            cast_dtype = torch.float32
            if args.mixed_precision_dtype and device.type == 'cuda':
                cast_dtype = torch.bfloat16 if args.mixed_precision_dtype == 'bf16' else torch.float16
            
            with amp.autocast(device_type=device.type, enabled=(args.mixed_precision_dtype is not None and device.type == 'cuda'), dtype=cast_dtype):
                logits = model(input_ids, attention_mask=attention_mask)
                loss, perplexity, top1_acc = compute_metrics(logits, input_ids, attention_mask)

            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(f"Invalid loss (NaN or Inf) at epoch {epoch+1}, batch {batch_idx+1}, global_step {global_step}! Stopping training."); 
                return model 
            
            scaler.scale(loss).backward()
            if args.grad_clip_norm > 0:
                scaler.unscale_(optimizer) 
                torch.nn.utils.clip_grad_norm_(optimizer_params, max_norm=args.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            
            global_step += 1
            epoch_loss += loss.item()
            current_ppl_item = perplexity.item()
            epoch_perplexity += current_ppl_item if not (torch.isinf(perplexity) or torch.isnan(perplexity)) else (epoch_perplexity / (batch_idx + 1) if batch_idx > 0 else 0.0)
            epoch_top1_acc += top1_acc.item()

            train_iterator.set_postfix({"loss": f"{loss.item():.4f}", 
                                        "ppl": f"{current_ppl_item:.2f}" if not (torch.isinf(perplexity) or torch.isnan(perplexity)) else "inf/nan",
                                        "acc": f"{top1_acc.item():.3f}",
                                        "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                                        "step": global_step})

            if global_step % args.log_interval == 0:
                 logger.info(f"E{epoch+1} S{global_step} B{batch_idx+1}/{len(train_dataloader)} | Loss: {loss.item():.4f}, PPL: {current_ppl_item:.2f}, Acc: {top1_acc.item():.3f}, LR: {optimizer.param_groups[0]['lr']:.2e}")

            if args.save_strategy == "steps" and global_step % args.save_steps == 0:
                val_loss_for_save = loss.item() 
                is_best_for_save = False
                if args.validation_interval_steps > 0 and global_step % args.validation_interval_steps == 0 and val_dataloader:
                    logger.info(f"Running mid-epoch validation at step {global_step}...")
                    val_loss_for_save, _, _ = evaluate_model(model, val_dataloader, tokenizer, args, device)
                    if val_loss_for_save < best_val_loss:
                        best_val_loss = val_loss_for_save; is_best_for_save = True
                        logger.info(f"New best validation loss: {best_val_loss:.4f} (at step {global_step})")
                    scheduler.step(val_loss_for_save) 
                save_checkpoint(model, optimizer, epoch, global_step, val_loss_for_save, args, is_best=is_best_for_save, scheduler=scheduler)
        
        avg_epoch_loss = epoch_loss / len(train_dataloader) if len(train_dataloader) > 0 else 0
        avg_epoch_perplexity = epoch_perplexity / len(train_dataloader) if len(train_dataloader) > 0 else float('inf')
        avg_epoch_top1_acc = epoch_top1_acc / len(train_dataloader) if len(train_dataloader) > 0 else 0
        logger.info(f"End of Epoch {epoch+1}/{args.num_epochs} - Training | Avg Loss: {avg_epoch_loss:.4f}, Avg PPL: {avg_epoch_perplexity:.2f}, Avg Acc: {avg_epoch_top1_acc:.3f}")

        if val_dataloader:
            logger.info(f"Running end-of-epoch validation for Epoch {epoch+1}...")
            val_loss, val_perplexity, val_accuracy = evaluate_model(model, val_dataloader, tokenizer, args, device)
            logger.info(f"End of Epoch {epoch+1} - Validation | Loss: {val_loss:.4f}, PPL: {val_perplexity:.2f}, Acc: {val_accuracy:.3f}")
            
            is_current_epoch_best = val_loss < best_val_loss
            if is_current_epoch_best:
                best_val_loss = val_loss
                logger.info(f"New best validation loss: {best_val_loss:.4f} (at epoch {epoch+1})")
            scheduler.step(val_loss) 
            
            if args.save_strategy == "epoch":
                 save_checkpoint(model, optimizer, epoch, global_step, val_loss, args, is_best=is_current_epoch_best, scheduler=scheduler)
        elif args.save_strategy == "epoch": 
             save_checkpoint(model, optimizer, epoch, global_step, avg_epoch_loss, args, is_best=False, scheduler=scheduler) 
    
    logger.info("Training completed!")
    return model

def evaluate_model(model, dataloader, tokenizer, args, device):
    model.eval()
    total_loss, total_perplexity, total_accuracy = 0.0, 0.0, 0.0
    num_batches_processed = 0 
    
    eval_iterator = tqdm(dataloader, desc="[Validation/Test]")
    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_iterator): # Added batch_idx for robust perplexity
            num_batches_processed += 1
            input_ids = batch["input_ids"].to(device, non_blocking=args.pin_memory)
            attention_mask = batch["attention_mask"].to(device, non_blocking=args.pin_memory)
            
            cast_dtype = torch.bfloat16 if args.mixed_precision_dtype == 'bf16' else torch.float16 if args.mixed_precision_dtype == 'fp16' else torch.float32
            with amp.autocast(device_type=device.type, enabled=(args.mixed_precision_dtype is not None and device.type == 'cuda'), dtype=cast_dtype):
                logits = model(input_ids, attention_mask=attention_mask)
                loss, perplexity, accuracy = compute_metrics(logits, input_ids, attention_mask)
            
            if torch.isnan(loss) or torch.isinf(loss): 
                logger.warning("Invalid loss (NaN/Inf) during evaluation. Skipping batch."); continue
            
            total_loss += loss.item()
            current_ppl_item = perplexity.item()
            # More robust perplexity accumulation (average of logs, then exp)
            total_perplexity += current_ppl_item if not (torch.isinf(perplexity) or torch.isnan(perplexity)) else (total_perplexity / batch_idx if batch_idx > 0 else 0.0) # Fallback for early inf/nan
            total_accuracy += accuracy.item()

            eval_iterator.set_postfix({"val_loss": f"{loss.item():.4f}", 
                                       "val_ppl": f"{current_ppl_item:.2f}" if not (torch.isinf(perplexity) or torch.isnan(perplexity)) else "inf/nan",
                                       "val_acc": f"{accuracy.item():.3f}"})

    avg_loss = total_loss / num_batches_processed if num_batches_processed > 0 else 0
    # For perplexity, it's better to average the log perplexities (losses) and then exponentiate
    # However, for simplicity here, we keep the direct average of PPLs, acknowledging it's less robust with NaNs/Infs
    avg_perplexity = total_perplexity / num_batches_processed if num_batches_processed > 0 else float('inf')
    avg_accuracy = total_accuracy / num_batches_processed if num_batches_processed > 0 else 0
    
    model.train() 
    return avg_loss, avg_perplexity, avg_accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Lunaris Codex model.")
    # Dataset and Tokenizer Args
    parser.add_argument("--memmap_file_train", type=str, required=True, help="Path to the training .memmap dataset.")
    parser.add_argument("--num_sequences_train", type=int, required=True, help="Number of sequences in the training dataset.")
    parser.add_argument("--memmap_file_val", type=str, default=None, help="Path to the validation .memmap dataset (optional).")
    parser.add_argument("--num_sequences_val", type=int, default=0, help="Number of sequences in the validation dataset.")
    parser.add_argument("--tokenizer_name_or_path", type=str, default="bigcode/starcoder", help="Tokenizer name or path.")
    parser.add_argument("--dataset_max_length", type=int, default=1024, help="Max length used for dataset creation (for memmap shape).")
    parser.add_argument("--dataset_dtype", type=str, default="int32", choices=["int16", "int32"], help="Memmap dtype.")

    # Model Config Args (from LunarisCodexConfig)
    parser.add_argument("--vocab_size", type=int, default=None, help="Vocabulary size (if None, inferred from tokenizer).")
    parser.add_argument("--d_model", type=int, default=768, help="Model hidden dimension.")
    parser.add_argument("--n_layers", type=int, default=10, help="Number of Transformer layers.")
    parser.add_argument("--n_heads", type=int, default=12, help="Number of attention heads.")
    parser.add_argument("--model_max_seq_len", type=int, default=1024, help="Model's internal max sequence length (for ALiBi).")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate.")
    parser.add_argument("--activation", type=str, default="swiglu", choices=["swiglu", "gelu"], help="FFN activation.")
    parser.add_argument("--lora_rank", type=int, default=0, help="LoRA rank (0 or negative to disable LoRA and train full model).")
    
    # Training Args
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per device.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Initial learning rate.")
    parser.add_argument("--lr_scheduler_patience", type=int, default=2, help="Patience for ReduceLROnPlateau scheduler.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay (L2 regularization).")
    parser.add_argument("--adam_fused", action="store_true", help="Use NVIDIA's fused AdamW (CUDA only).")
    parser.add_argument("--grad_clip_norm", type=float, default=1.0, help="Gradient clipping norm (0 to disable).")
    parser.add_argument("--mixed_precision_dtype", type=str, default=None, choices=["fp16", "bf16"], help="Enable mixed precision (fp16 or bf16). CUDA only. Default: fp32.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu).")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers (0 for main process).")
    parser.add_argument("--pin_memory", action="store_true", help="Use pin_memory in DataLoader (CUDA only).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    
    # Checkpoint and Logging Args
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_lunaris", help="Directory to save checkpoints.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume training.")
    parser.add_argument("--save_strategy", type=str, default="epoch", choices=["epoch", "steps"], help="Checkpoint saving strategy.")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save checkpoint every X steps (if save_strategy='steps').")
    parser.add_argument("--log_interval", type=int, default=100, help="Log metrics every X steps.")
    parser.add_argument("--validation_interval_steps", type=int, default=0, help="Run validation every X steps (0 for end-of-epoch only).")

    # PyTorch Optimization Args
    parser.add_argument("--use_torch_compile", action="store_true", help="Enable torch.compile for model optimization.")
    parser.add_argument("--torch_compile_mode", type=str, default="reduce-overhead", 
                        choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"], 
                        help="Mode for torch.compile.")
    parser.add_argument("--allow_tf32", action="store_true", help="Allow TF32 on Ampere+ GPUs for matmuls.")
    parser.add_argument("--cudnn_benchmark", action="store_true", help="Enable cudnn.benchmark (if input sizes are fixed).")
    
    args = parser.parse_args()

    if args.allow_tf32 and args.device == "cuda" and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True; logger.info("TF32 for CUDA matmuls enabled.")
    if args.cudnn_benchmark and args.device == "cuda" and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True; logger.info("torch.backends.cudnn.benchmark enabled.")
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path, trust_remote_code=True) 
    if tokenizer.pad_token_id is None:
        pad_token_to_add = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else \
                           tokenizer.bos_token_id if tokenizer.bos_token_id is not None else None
        if pad_token_to_add is not None:
            tokenizer.pad_token_id = pad_token_to_add
            logger.info(f"Tokenizer pad_token_id was None. Set to: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
        else: 
            original_vocab_size = len(tokenizer)
            tokenizer.add_special_tokens({'pad_token': '<|PAD|>'})
            logger.warning(f"Tokenizer had no pad/eos/bos. Added new pad_token='<|PAD|>' (ID: {tokenizer.pad_token_id}). Vocab size changed from {original_vocab_size} to {len(tokenizer)}. Ensure model's vocab_size is updated if not inferred.")

    try:
        train_dataset = MemmapCodeDataset(args.memmap_file_train, args.num_sequences_train, args.dataset_max_length, tokenizer.pad_token_id, args.dataset_dtype)
    except FileNotFoundError:
        logger.error(f"CRITICAL: Training data file {args.memmap_file_train} not found. Please check the path and ensure data preparation was successful.")
        sys.exit(1) # Exit if training data is missing
    except ValueError as e:
        logger.error(f"CRITICAL: Error loading training data from {args.memmap_file_train}: {e}. Check shape/dtype or file integrity.")
        sys.exit(1)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                                  num_workers=args.num_workers, pin_memory=(args.pin_memory and args.device == "cuda"))
    
    val_dataloader = None
    if args.memmap_file_val and args.num_sequences_val > 0:
        try:
            val_dataset = MemmapCodeDataset(args.memmap_file_val, args.num_sequences_val, args.dataset_max_length, tokenizer.pad_token_id, args.dataset_dtype)
            val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                                        num_workers=args.num_workers, pin_memory=(args.pin_memory and args.device == "cuda"))
            logger.info(f"Validation dataset loaded from {args.memmap_file_val} ({args.num_sequences_val} sequences).")
        except FileNotFoundError:
            logger.warning(f"Validation data file {args.memmap_file_val} not found. Proceeding without validation data.")
        except ValueError as e: # Catch other potential errors like shape mismatch
            logger.error(f"Error loading validation data from {args.memmap_file_val}: {e}. Proceeding without validation.", exc_info=True)
    else:
        logger.info("No validation dataset provided or num_sequences_val is 0.")

    effective_vocab_size = args.vocab_size if args.vocab_size is not None else len(tokenizer)
    if args.vocab_size is not None and args.vocab_size != len(tokenizer):
        logger.warning(f"Provided --vocab_size ({args.vocab_size}) differs from tokenizer's actual vocab size ({len(tokenizer)}). Using the provided --vocab_size.")
    
    model_config = LunarisCodexConfig(
        vocab_size=effective_vocab_size, d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads,
        max_seq_len=args.model_max_seq_len, dropout=args.dropout, activation=args.activation,
        lora_rank=args.lora_rank, 
        use_flash_attention_if_available=(args.device == 'cuda') 
    )
    model = LunarisMind(model_config)
    
    args.use_lora = model_config.lora_rank > 0 and model_config.lora_rank is not None
    if args.use_lora:
        logger.info(f"Setting up for LoRA training (rank={model_config.lora_rank}).")
        trainable_params_val = 0
        all_params_val = 0
        for name, param in model.named_parameters():
            all_params_val += param.numel()
            if 'lora_' in name: 
                param.requires_grad = True
                trainable_params_val += param.numel()
            else: 
                param.requires_grad = False
        logger.info(f"Total model parameters: {all_params_val:,}. Trainable (LoRA) parameters: {trainable_params_val:,} ({trainable_params_val/all_params_val*100:.2f}%)")
    else:
        logger.info("Setting up for full model training (LoRA rank is 0 or not specified).")
        all_params_val = 0
        trainable_params_val = 0
        for param in model.parameters(): 
            param.requires_grad = True 
            all_params_val += param.numel()
            if param.requires_grad:
                trainable_params_val += param.numel()
        logger.info(f"Total model parameters: {all_params_val:,}. Trainable parameters: {trainable_params_val:,} ({trainable_params_val/all_params_val*100:.2f}%)")

    trained_model = train_model_loop(model, train_dataloader, val_dataloader, tokenizer, args)
