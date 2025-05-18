# inference.py
import torch
from transformers import AutoTokenizer
import argparse
import logging
import sys # For sys.exit
import os

# Project-specific imports
from model import LunarisMind, LunarisCodexConfig # Assuming model.py is in the same directory or accessible in PYTHONPATH

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- SCRIPT VERSION ---
SCRIPT_VERSION = "0.1.0"

def load_model_from_checkpoint(checkpoint_path: str, device: torch.device) -> tuple[LunarisMind, LunarisCodexConfig, dict]:
    """
    Loads the model, its configuration, and training arguments from a checkpoint.
    """
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except FileNotFoundError:
        logger.error(f"Checkpoint file not found: {checkpoint_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load checkpoint '{checkpoint_path}': {e}", exc_info=True)
        sys.exit(1)

    if "config" not in checkpoint:
        logger.error("Checkpoint does not contain 'config' key. Cannot instantiate model.")
        sys.exit(1)
    if "model_state_dict" not in checkpoint:
        logger.error("Checkpoint does not contain 'model_state_dict' key. Cannot load model weights.")
        sys.exit(1)

    # Instantiate config from checkpoint
    # The config in checkpoint is a dict, so ** unpacks it into LunarisCodexConfig constructor
    try:
        model_config_dict = checkpoint["config"]
        # Ensure all necessary fields for LunarisCodexConfig are present or provide defaults
        # This is a simplified example; you might need more robust handling if config structure changes
        required_config_fields = ["vocab_size", "d_model", "n_layers", "n_heads", "max_seq_len"] # Add more if LunarisCodexConfig requires them
        for field in required_config_fields:
            if field not in model_config_dict:
                logger.error(f"Checkpoint 'config' is missing required field: '{field}'.")
                # You could try to provide a default from a base config or args, or exit
                # For now, let's assume train.py saves a complete config.
                # If a field is genuinely optional in LunarisCodexConfig and has a default, this check might be too strict.
                # However, vocab_size, d_model etc. are usually essential.
                # A more robust way would be to have default values in LunarisCodexConfig itself.
                # If vocab_size is None here, that's a problem for model instantiation.
                if model_config_dict.get("vocab_size") is None and field == "vocab_size":
                     logger.error("vocab_size is None in the loaded config. This is required.")
                     sys.exit(1)


        model_config = LunarisCodexConfig(**model_config_dict)
        logger.info(f"Model configuration loaded from checkpoint: {model_config_dict}")
    except TypeError as e:
        logger.error(f"Error instantiating LunarisCodexConfig from checkpoint['config']. "
                     f"Ensure all required arguments for LunarisCodexConfig are present in the saved config. Error: {e}")
        logger.error(f"Config dictionary from checkpoint: {checkpoint.get('config')}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error loading model configuration: {e}", exc_info=True)
        sys.exit(1)

    # Instantiate model
    model = LunarisMind(model_config)
    
    # Load model state dict
    model_state_dict = checkpoint["model_state_dict"]
    
    # Handle potential '_orig_mod.' prefix if checkpoint was saved from a compiled model
    # and we are loading into a non-compiled model here (typical for inference).
    is_compiled_checkpoint = any(k.startswith("_orig_mod.") for k in model_state_dict.keys())
    if is_compiled_checkpoint:
        logger.info("Checkpoint appears to be from a torch.compile'd model. Stripping '_orig_mod.' prefix.")
        model_state_dict = {k.replace('_orig_mod.', ''): v for k, v in model_state_dict.items()}

    missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)
    if missing_keys:
        logger.warning(f"Missing keys when loading model state_dict: {missing_keys}")
    if unexpected_keys:
        logger.warning(f"Unexpected keys when loading model state_dict: {unexpected_keys}")
    
    model.to(device)
    model.eval() # Set model to evaluation mode
    logger.info("Model weights loaded successfully and model set to evaluation mode.")
    
    train_args_from_checkpoint = checkpoint.get("args", {}) # Get training args if available
    return model, model_config, train_args_from_checkpoint

def main():
    parser = argparse.ArgumentParser(description=f"Generate text using a trained Lunaris Codex model (v{SCRIPT_VERSION}).")
    
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to the model checkpoint (.pt file).")
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None,
                        help="Tokenizer name or path (e.g., 'gpt2', 'bigcode/starcoder'). "
                             "If None, tries to infer from training args in checkpoint.")
    parser.add_argument("--prompt", type=str, default="USER: Write a Python function to sort a list.\nASSISTANT:",
                        help="Input prompt for the model.")
    parser.add_argument("--max_new_tokens", type=int, default=100,
                        help="Maximum number of new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=None, # Default to None, use config's default
                        help="Sampling temperature. Overrides model config if set.")
    parser.add_argument("--top_k", type=int, default=None, # Default to None
                        help="Top-k filtering. K most O verbo é limitar. K. Overrides model config if set. 0 to disable.")
    parser.add_argument("--top_p", type=float, default=None, # Default to None
                        help="Nucleus (top-p) filtering. Overrides model config if set. 1.0 to disable.")
    parser.add_argument("--repetition_penalty", type=float, default=None, # Default to None
                        help="Repetition penalty. 1.0 means no penalty. Overrides model config if set.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for inference (e.g., 'cuda', 'cpu').")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for generation (for reproducibility).")

    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if args.device == "cuda" and torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        logger.info(f"Set generation seed to: {args.seed}")
        
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    model, model_config, train_args = load_model_from_checkpoint(args.checkpoint_path, device)

    # Determine tokenizer path
    tokenizer_path = args.tokenizer_name_or_path
    if not tokenizer_path:
        tokenizer_path = train_args.get("tokenizer_name_or_path")
        if not tokenizer_path:
            logger.error("Tokenizer not specified via --tokenizer_name_or_path and not found in checkpoint's training arguments.")
            sys.exit(1)
        logger.info(f"Using tokenizer from training arguments: {tokenizer_path}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True) # Assuming trust_remote_code for now
        # Ensure pad token is set for tokenizer, as generation might involve padding if batching or for internal model logic
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
                tokenizer.pad_token = tokenizer.eos_token
                logger.info(f"Set tokenizer pad_token_id to eos_token_id: {tokenizer.pad_token_id}")
            else: # Minimal fallback if no EOS
                tokenizer.add_special_tokens({'pad_token': '<|PAD|>'})
                logger.warning(f"Tokenizer lacked pad/eos. Added new pad_token='<|PAD|>' (ID: {tokenizer.pad_token_id}) for inference.")
    except Exception as e:
        logger.error(f"Failed to load tokenizer '{tokenizer_path}': {e}", exc_info=True)
        sys.exit(1)

    # Prepare input
    input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(device)

    # Generation parameters: use CLI args if provided, otherwise fallback to model_config defaults
    gen_temp = args.temperature if args.temperature is not None else model_config.temperature
    gen_top_k = args.top_k if args.top_k is not None else model_config.top_k
    gen_top_p = args.top_p if args.top_p is not None else model_config.top_p
    gen_rep_penalty = args.repetition_penalty if args.repetition_penalty is not None else model_config.repetition_penalty
    
    # EOS token ID for stopping generation
    # It's good to use the tokenizer's actual EOS token if available, or allow override
    eos_token_id_for_gen = tokenizer.eos_token_id
    # You might add an arg --eos_token_id to override this if needed.

    logger.info(f"Generating text with parameters: max_new_tokens={args.max_new_tokens}, temperature={gen_temp}, "
                f"top_k={gen_top_k}, top_p={gen_top_p}, repetition_penalty={gen_rep_penalty}, eos_token_id={eos_token_id_for_gen}")
    logger.info(f"Input prompt: \"{args.prompt}\"")
    
    # Generate
    with torch.no_grad(): # Ensure no gradients are calculated
        generated_ids_full = model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=gen_temp,
            top_k=gen_top_k,
            top_p=gen_top_p,
            repetition_penalty=gen_rep_penalty,
            eos_token_id=eos_token_id_for_gen
        )

    # Decode only the newly generated tokens
    # input_ids is (1, prompt_len), generated_ids_full is (1, prompt_len + new_tokens)
    num_prompt_tokens = input_ids.shape[1]
    generated_ids_new_only = generated_ids_full[0, num_prompt_tokens:] # Get the first (and only) batch item
    
    generated_text = tokenizer.decode(generated_ids_new_only, skip_special_tokens=True)

    logger.info("--- Generated Text ---")
    # For a "foda" visual, we'll use rich later. For now, simple print.
    print(args.prompt, end="") # Print the prompt
    print(generated_text)      # Print the generated part
    logger.info("--- End of Generation ---")

if __name__ == "__main__":
    main()
