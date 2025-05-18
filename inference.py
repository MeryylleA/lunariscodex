# inference.py
import torch
from transformers import AutoTokenizer
import argparse
import logging
import sys
import os
from rich.console import Console
from rich.panel import Panel
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.theme import Theme
from rich.text import Text
from rich.box import ROUNDED

# Project-specific imports
from model import LunarisMind, LunarisCodexConfig  # Assuming model.py is in the same directory

# --- SCRIPT VERSION ---
SCRIPT_VERSION = "0.2.0"

# Set up rich console for beautiful output
custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "success": "bold green",
    "model": "bold blue",
    "param": "magenta",
    "code": "green",
    "prompt": "italic yellow",
    "generation": "bold white"
})
console = Console(theme=custom_theme)

# Setup logging with rich handler
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)
logger = logging.getLogger("lunaris")

def show_header():
    """Display a fancy header for the script"""
    title = Text("Lunaris Codex Inference Engine", style="bold blue")
    subtitle = Text(f"v{SCRIPT_VERSION}", style="dim")
    header = Text.assemble(title, " ", subtitle)

    console.print("\n")
    console.print(Panel(header, border_style="blue", box=ROUNDED))
    console.print("\n")

def load_model_from_checkpoint(checkpoint_path: str, device: torch.device) -> tuple[LunarisMind, LunarisCodexConfig, dict]:
    """
    Loads the model, its configuration, and training arguments from a checkpoint.
    """
    with Progress(
        SpinnerColumn(),
        TextColumn("[model]Loading model from checkpoint...[/model]"),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task("", total=None)

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
        try:
            model_config_dict = checkpoint["config"]
            required_config_fields = ["vocab_size", "d_model", "n_layers", "n_heads", "max_seq_len"]
            for field in required_config_fields:
                if field not in model_config_dict:
                    logger.error(f"Checkpoint 'config' is missing required field: '{field}'.")
                    if model_config_dict.get("vocab_size") is None and field == "vocab_size":
                        logger.error("vocab_size is None in the loaded config. This is required.")
                        sys.exit(1)

            model_config = LunarisCodexConfig(**model_config_dict)
            logger.info(f"Model configuration loaded successfully")
        except TypeError as e:
            logger.error(f"Error instantiating LunarisCodexConfig: {e}")
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
        is_compiled_checkpoint = any(k.startswith("_orig_mod.") for k in model_state_dict.keys())
        if is_compiled_checkpoint:
            logger.info("Checkpoint appears to be from a compiled model. Stripping '_orig_mod.' prefix.")
            model_state_dict = {k.replace('_orig_mod.', ''): v for k, v in model_state_dict.items()}

        missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)
        if missing_keys:
            logger.warning(f"Missing keys when loading model state_dict: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys when loading model state_dict: {unexpected_keys}")

        model.to(device)
        model.eval()  # Set model to evaluation mode

    # Display model info after progress bar completes
    console.print(Panel(
        f"[success]✓ Model loaded successfully[/success]\n\n"
        f"[model]Architecture:[/model] Lunaris Mind\n"
        f"[model]Layers:[/model] {model_config.n_layers}\n"
        f"[model]Heads:[/model] {model_config.n_heads}\n"
        f"[model]Dimension:[/model] {model_config.d_model}\n"
        f"[model]Vocabulary Size:[/model] {model_config.vocab_size}\n"
        f"[model]Max Sequence Length:[/model] {model_config.max_seq_len}",
        title="Model Information",
        border_style="green",
        box=ROUNDED
    ))

    train_args_from_checkpoint = checkpoint.get("args", {})
    return model, model_config, train_args_from_checkpoint

def display_generation_params(params):
    """Display generation parameters in a formatted panel"""
    console.print(Panel(
        "\n".join([f"[param]{k}:[/param] {v}" for k, v in params.items()]),
        title="Generation Parameters",
        border_style="magenta",
        box=ROUNDED
    ))

def main():
    parser = argparse.ArgumentParser(description=f"Generate text using a trained Lunaris Codex model.")

    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to the model checkpoint (.pt file).")
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None,
                        help="Tokenizer name or path (e.g., 'gpt2', 'bigcode/starcoder').")
    parser.add_argument("--prompt", type=str, default="USER: Write a Python function to sort a list.\nASSISTANT:",
                        help="Input prompt for the model.")
    parser.add_argument("--prompt_file", type=str, default=None,
                        help="Path to a file containing the prompt (alternative to --prompt).")
    parser.add_argument("--max_new_tokens", type=int, default=100,
                        help="Maximum number of new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Sampling temperature. Overrides model config if set.")
    parser.add_argument("--top_k", type=int, default=None,
                        help="Top-k filtering. Overrides model config if set. 0 to disable.")
    parser.add_argument("--top_p", type=float, default=None,
                        help="Nucleus (top-p) filtering. Overrides model config if set. 1.0 to disable.")
    parser.add_argument("--repetition_penalty", type=float, default=None,
                        help="Repetition penalty. 1.0 means no penalty. Overrides model config if set.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for inference (e.g., 'cuda', 'cpu').")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for generation (for reproducibility).")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Save the generated text to a file.")
    parser.add_argument("--no_color", action="store_true",
                        help="Disable colored output.")

    args = parser.parse_args()

    # Apply no color if requested
    if args.no_color:
        console.no_color = True

    # Show fancy header
    show_header()

    # Set seed for reproducibility if provided
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if args.device == "cuda" and torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        logger.info(f"Random seed set to: {args.seed}")

    # Determine device
    device = torch.device(args.device)
    logger.info(f"Using device: [bold]{device}[/bold]")

    # Load model
    model, model_config, train_args = load_model_from_checkpoint(args.checkpoint_path, device)

    # Determine tokenizer path
    tokenizer_path = args.tokenizer_name_or_path
    if not tokenizer_path:
        tokenizer_path = train_args.get("tokenizer_name_or_path")
        if not tokenizer_path:
            logger.error("Tokenizer not specified via --tokenizer_name_or_path and not found in checkpoint's training arguments.")
            sys.exit(1)
        logger.info(f"Using tokenizer from training arguments: [bold]{tokenizer_path}[/bold]")

    # Load tokenizer
    with console.status("[model]Loading tokenizer...[/model]", spinner="dots"):
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
            # Ensure pad token is set for tokenizer
            if tokenizer.pad_token_id is None:
                if tokenizer.eos_token_id is not None:
                    tokenizer.pad_token_id = tokenizer.eos_token_id
                    tokenizer.pad_token = tokenizer.eos_token
                    logger.info(f"Set tokenizer pad_token_id to eos_token_id: {tokenizer.pad_token_id}")
                else:  # Minimal fallback if no EOS
                    tokenizer.add_special_tokens({'pad_token': '<|PAD|>'})
                    logger.warning(f"Tokenizer lacked pad/eos. Added new pad_token='<|PAD|>' (ID: {tokenizer.pad_token_id})")
            logger.info(f"Tokenizer loaded successfully: [bold]{tokenizer.__class__.__name__}[/bold]")
        except Exception as e:
            logger.error(f"Failed to load tokenizer '{tokenizer_path}': {e}", exc_info=True)
            sys.exit(1)

    # Get prompt from file if specified
    prompt = args.prompt
    if args.prompt_file:
        try:
            with open(args.prompt_file, 'r', encoding='utf-8') as f:
                prompt = f.read()
            logger.info(f"Loaded prompt from file: [bold]{args.prompt_file}[/bold]")
        except Exception as e:
            logger.error(f"Failed to load prompt from file '{args.prompt_file}': {e}")
            sys.exit(1)

    # Prepare input
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Generation parameters
    gen_temp = args.temperature if args.temperature is not None else getattr(model_config, 'temperature', 0.7)
    gen_top_k = args.top_k if args.top_k is not None else getattr(model_config, 'top_k', 0)
    gen_top_p = args.top_p if args.top_p is not None else getattr(model_config, 'top_p', 0.9)
    gen_rep_penalty = args.repetition_penalty if args.repetition_penalty is not None else getattr(model_config, 'repetition_penalty', 1.0)

    # EOS token ID for stopping generation
    eos_token_id_for_gen = tokenizer.eos_token_id

    # Display generation parameters
    display_generation_params({
        "Max new tokens": args.max_new_tokens,
        "Temperature": gen_temp,
        "Top-k": gen_top_k,
        "Top-p": gen_top_p,
        "Repetition penalty": gen_rep_penalty,
        "EOS token ID": eos_token_id_for_gen
    })

    # Display prompt
    console.print(Panel(
        Text(prompt, style="prompt"),
        title="Input Prompt",
        border_style="yellow",
        box=ROUNDED
    ))

    # Generate text
    logger.info("Starting text generation...")
    with console.status("[model]Generating text...[/model]", spinner="dots"):
        with torch.no_grad():  # Ensure no gradients are calculated
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
    num_prompt_tokens = input_ids.shape[1]
    generated_ids_new_only = generated_ids_full[0, num_prompt_tokens:]
    generated_text = tokenizer.decode(generated_ids_new_only, skip_special_tokens=True)

    # Display generated text
    console.print(Panel(
        Text(generated_text, style="generation"),
        title="Generated Text",
        border_style="green",
        box=ROUNDED
    ))

    # Save generated text to file if specified
    if args.output_file:
        try:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                f.write(prompt + generated_text)
            logger.info(f"[success]✓ Generated text saved to:[/success] [bold]{args.output_file}[/bold]")
        except Exception as e:
            logger.error(f"Failed to save generated text to file '{args.output_file}': {e}")

    # Show completion message
    console.print("\n[success]✓ Text generation completed[/success]\n")

if __name__ == "__main__":
    main()
