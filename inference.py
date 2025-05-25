import torch
from transformers import AutoTokenizer
import argparse
import logging
import sys
import os
import time
import psutil
import hashlib
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.syntax import Syntax
from rich.theme import Theme
from rich.text import Text
from rich.box import ROUNDED
from rich.table import Table
from rich.live import Live
from rich.prompt import Prompt
from pygments.util import ClassNotFound

# Project-specific imports
from model import LunarisMind, LunarisCodexConfig  # Assuming model.py is in the same directory

# --- SCRIPT VERSION ---
SCRIPT_VERSION = "0.3.5" # Log clareza tokenizer

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
    "generation": "bold white",
    "performance": "bright_cyan",
    "interactive": "bright_magenta",
    "dim_info": "dim cyan"
})
# Try to get terminal width, default to 120 if not available or too small
try:
    console_width = os.get_terminal_size().columns
    if console_width < 80: # Ensure a minimum reasonable width
        console_width = 120
except OSError: # Fallback if get_terminal_size() fails (e.g., not a TTY)
    console_width = 120

console = Console(theme=custom_theme, width=console_width)


# Setup logging with rich handler
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s", # Using RichHandler, format is less critical here
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True, markup=True, show_path=False)]
)
logger = logging.getLogger("lunaris")

def show_header():
    """
    Displays a styled header panel with the script name and version in the console.
    """
    title = Text("Lunaris Codex Inference Engine", style="bold blue")
    subtitle = Text(f"v{SCRIPT_VERSION} - Enhanced Edition", style="dim")
    header_text = Text.assemble(title, " ", subtitle)
    console.print("\n", Panel(header_text, border_style="blue", box=ROUNDED, expand=False), "\n")

# --- SHA-256 Utility Functions (adapted from train.py) ---
def compute_sha256(filepath: str) -> str | None:
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
        logger.warning(f"Failed to compute SHA-256 for {filepath}: {e}", exc_info=True)
        return None

def verify_checkpoint_integrity(checkpoint_path: str) -> bool:
    """
    Verifies the integrity of a checkpoint file by comparing its SHA-256 hash to a stored hash.
    
    If a corresponding `.sha256` file is present, the function reads the expected hash and compares it to the computed hash of the checkpoint file. Returns True if the hashes match, if the hash file is missing or empty, or if an error occurs during verification. Returns False only if a hash mismatch is detected.
    """
    hash_file = checkpoint_path + ".sha256"
    if not os.path.exists(hash_file):
        logger.warning(f"No hash file found for [dim_info]{checkpoint_path}[/dim_info]. Skipping SHA-256 verification.")
        return True

    try:
        with open(hash_file, "r") as f:
            content = f.read().split()
            if not content:
                logger.warning(f"Hash file [dim_info]{hash_file}[/dim_info] is empty.")
                return True
            expected_hash = content[0]

        actual_hash = compute_sha256(checkpoint_path)
        if actual_hash and actual_hash == expected_hash:
            logger.info(f"Checkpoint integrity verified: [model]{os.path.basename(checkpoint_path)}[/model] (SHA-256 match)")
            return True
        elif actual_hash:
            logger.error(f"SHA-256 Mismatch! Checkpoint integrity check failed for [dim_info]{checkpoint_path}[/dim_info].")
            logger.error(f"  Expected: [yellow]{expected_hash}[/yellow]")
            logger.error(f"  Actual:   [red]{actual_hash}[/red]")
            return False
        else:
            logger.warning(f"Could not compute SHA-256 for [dim_info]{checkpoint_path}[/dim_info] to verify integrity. Assuming valid.")
            return True
    except Exception as e:
        logger.warning(f"Could not verify checkpoint integrity for [dim_info]{checkpoint_path}[/dim_info] due to an error: {e}", exc_info=True)
        return True

def validate_checkpoint_exists(checkpoint_path: str) -> bool:
    """
    Checks if the checkpoint file exists and logs its presence and size.
    
    Logs an error if the file is missing, and a warning if the file size is suspiciously small. Returns True if the file exists and is accessible, otherwise False.
    """
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint file not found: [dim_info]{checkpoint_path}[/dim_info]")
        return False
    try:
        file_size = os.path.getsize(checkpoint_path) / (1024 * 1024)  # MB
        logger.info(f"Checkpoint file found: [model]{os.path.basename(checkpoint_path)}[/model] ({file_size:.2f} MB)")
        if file_size < 0.1:
            logger.warning("Checkpoint file seems very small. Ensure it's a valid model checkpoint.")
        return True
    except Exception as e:
        logger.error(f"Error accessing checkpoint file [dim_info]{checkpoint_path}[/dim_info]: {e}", exc_info=True)
        return False

def get_memory_usage():
    """
    Returns the current process memory usage in megabytes.
    """
    process = psutil.Process()
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024  # MB

def load_model_from_checkpoint(checkpoint_path: str, device: torch.device) -> tuple[LunarisMind, LunarisCodexConfig, dict] | None:
    """
    Loads a Lunaris Mind model, its configuration, and training arguments from a checkpoint file.
    
    Performs SHA-256 integrity verification on the checkpoint, validates required configuration fields, and handles both standard and torch compiled checkpoints. On success, returns the model instance, its configuration, and any training arguments found in the checkpoint. Returns None if integrity verification or loading fails.
    """
    if not verify_checkpoint_integrity(checkpoint_path):
        logger.error(f"Aborting due to checkpoint integrity verification failure for: [dim_info]{checkpoint_path}[/dim_info]")
        return None

    progress_description = "[model]Loading model from checkpoint...[/model]"
    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task(progress_description, total=100)

        progress.update(task, advance=10, description="[model]Loading checkpoint file...[/model]")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False) # weights_only=False for full state
            progress.update(task, advance=30, description="[model]Checkpoint data loaded.[/model]")
        except FileNotFoundError:
            logger.error(f"Checkpoint file not found: [dim_info]{checkpoint_path}[/dim_info]")
            return None
        except Exception as e:
            logger.error(f"Failed to load checkpoint file '[dim_info]{checkpoint_path}[/dim_info]': {e}", exc_info=True)
            return None

        required_keys = ["config", "model_state_dict"]
        missing_keys_list = [key for key in required_keys if key not in checkpoint]
        if missing_keys_list:
            logger.error(f"Checkpoint is missing required keys: {missing_keys_list}")
            return None
        progress.update(task, advance=10, description="[model]Validating checkpoint keys...[/model]")

        try:
            model_config_dict = checkpoint["config"]
            required_config_fields = ["vocab_size", "d_model", "n_layers", "n_heads", "max_seq_len"]
            for field in required_config_fields:
                if field not in model_config_dict:
                    logger.error(f"Checkpoint 'config' is missing required field: '{field}'.")
                    return None
                if model_config_dict[field] is None:
                    logger.error(f"Config field '{field}' is None in checkpoint, which is invalid.")
                    return None

            if 'lora_rank' not in model_config_dict:
                model_config_dict['lora_rank'] = 0
                logger.info("LoRA rank not found in checkpoint config, defaulting to 0 (no LoRA).")

            model_config = LunarisCodexConfig(**model_config_dict)
            logger.info("Model configuration loaded successfully from checkpoint.")
            progress.update(task, advance=20, description="[model]Model config processed.[/model]")

        except TypeError as e:
            logger.error(f"Error instantiating LunarisCodexConfig from checkpoint 'config': {e}", exc_info=True)
            logger.error(f"Config dictionary from checkpoint: {checkpoint.get('config')}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error loading model configuration: {e}", exc_info=True)
            return None

        model = LunarisMind(model_config) # Model instantiation messages (LayerScale, ALiBi) will print here
        model_state_dict = checkpoint["model_state_dict"]
        is_compiled_checkpoint = any(k.startswith("_orig_mod.") for k in model_state_dict.keys())
        if is_compiled_checkpoint:
            logger.info("Checkpoint is from a torch.compiled model. Stripping '_orig_mod.' prefix.")
            model_state_dict = {k.replace('_orig_mod.', ''): v for k, v in model_state_dict.items()}

        missing_keys_load, unexpected_keys_load = model.load_state_dict(model_state_dict, strict=False)
        if missing_keys_load:
            logger.warning(f"Missing keys when loading model state_dict: {missing_keys_load}")
        if unexpected_keys_load:
            logger.warning(f"Unexpected keys when loading model state_dict: {unexpected_keys_load}")
        progress.update(task, advance=20, description="[model]Model state loaded.[/model]")

        model.to(device)
        model.eval()
        progress.update(task, advance=10, description="[model]Model ready on device![/model]")

    total_params = sum(p.numel() for p in model.parameters())
    info_table = Table(show_header=False, box=ROUNDED, title_style="bold blue", border_style="blue")
    info_table.add_column("Property", style="param")
    info_table.add_column("Value", style="info")
    info_table.add_row("Architecture", "Lunaris Mind")
    info_table.add_row("Layers", str(model_config.n_layers))
    info_table.add_row("Attention Heads", str(model_config.n_heads))
    info_table.add_row("Model Dimension (d_model)", str(model_config.d_model))
    info_table.add_row("Vocabulary Size", f"{model_config.vocab_size:,}")
    info_table.add_row("Max Sequence Length", f"{model_config.max_seq_len:,}")
    if model_config.lora_rank > 0:
         info_table.add_row("LoRA Rank (from config)", str(model_config.lora_rank))
    info_table.add_row("Total Parameters", f"{total_params:,}")
    info_table.add_row("Loaded from", f"[dim_info]{os.path.basename(checkpoint_path)}[/dim_info]")
    console.print(Panel(info_table, title="[bold]Model Information[/bold]", border_style="green", expand=False))
    logger.info(f"Initial memory usage after model load: [performance]{get_memory_usage():.1f} MB[/performance]")

    train_args_from_checkpoint = checkpoint.get("args", {})
    return model, model_config, train_args_from_checkpoint

def display_generation_params(params):
    """
    Displays generation parameters in a styled table panel using the rich console.
    
    Args:
    	params: A dictionary of generation parameter names and their values to display.
    """
    param_table = Table(show_header=False, box=ROUNDED, title_style="bold magenta", border_style="magenta")
    param_table.add_column("Parameter", style="param")
    param_table.add_column("Value", style="info")
    for k, v in params.items():
        param_table.add_row(k, str(v))
    console.print(Panel(param_table, title="[bold]Generation Parameters[/bold]", border_style="magenta", expand=False))

def format_code_output(text: str, language: str = "python") -> Syntax | Text:
    """
    Formats code or text for console output with syntax highlighting if supported.
    
    If the specified language is unsupported or an error occurs, returns plain text formatting as a fallback.
    
    Args:
        text: The code or text to format.
        language: The programming language for syntax highlighting. If set to "auto" or empty, plain text formatting is used.
    
    Returns:
        A Rich Syntax object with highlighting if possible, otherwise a plain Text object.
    """
    try:
        if not language or language.lower() == 'auto':
            language = "text" # Keep it simple if auto, rely on pygments default lexer guessing
        return Syntax(text, language, theme="monokai", line_numbers=True, word_wrap=True)
    except ClassNotFound:
        logger.warning(f"Syntax highlighting for language '{language}' not found. Falling back to plain text.")
        return Text(text)
    except Exception as e:
        logger.warning(f"Error during syntax highlighting (lang: {language}): {e}. Falling back to plain text.", exc_info=True)
        return Text(text)

def stream_generation(model, tokenizer, input_ids, max_new_tokens, temperature, top_k, top_p, repetition_penalty, eos_token_id, device, syntax_highlight_lang):
    """
    Generates text from a model in a streaming, token-by-token fashion with live console updates.
    
    Args:
        input_ids: The initial input token IDs to prompt the model.
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature for controlling randomness.
        top_k: Limits sampling to the top-k most probable tokens.
        top_p: Nucleus sampling threshold; considers tokens with cumulative probability up to top_p.
        repetition_penalty: Penalty factor to discourage repetition in generated text.
        eos_token_id: Token ID that signals end of generation if encountered.
        syntax_highlight_lang: Language identifier for syntax highlighting of the generated output.
    
    Returns:
        A tuple containing the full sequence of generated token IDs, the generated text, and the average tokens per second achieved during generation.
    """
    model.eval()
    generated_ids = input_ids.clone()
    full_generated_text = ""
    start_time = time.time()
    # Variable to store the last next_token, initialized to a value that won't match eos_token_id
    last_next_token_item = -1
    tokens_generated_count = 0
    tokens_per_sec = 0.0


    # Use transient=True for the Live display so it cleans up after finishing
    with Live(console=console, refresh_per_second=12, transient=True) as live:
        for i in range(max_new_tokens):
            tokens_generated_count = i + 1
            current_seq_len = generated_ids.size(1)
            batch_size = generated_ids.size(0)
            current_attention_mask = torch.ones((batch_size, current_seq_len), dtype=torch.long, device=device)

            with torch.no_grad():
                logits = model.forward(generated_ids, attention_mask=current_attention_mask)[:, -1, :]
            logits = logits / max(temperature, 1e-5)
            if hasattr(model, '_apply_repetition_penalty_optimized') and repetition_penalty != 1.0:
                 logits = model._apply_repetition_penalty_optimized(logits, generated_ids, repetition_penalty)

            if top_k > 0:
                top_k_values, _ = torch.topk(logits, top_k)
                kth_value = top_k_values[:, -1].unsqueeze(-1)
                logits[logits < kth_value] = float('-inf')
            if 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = torch.zeros_like(logits, dtype=torch.bool).scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
                logits.masked_fill_(indices_to_remove, float('-inf'))

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            last_next_token_item = next_token.item() # Store the item for checking after loop
            new_token_text = tokenizer.decode(next_token[0], skip_special_tokens=True)
            full_generated_text += new_token_text
            elapsed_time = time.time() - start_time
            tokens_per_sec = tokens_generated_count / elapsed_time if elapsed_time > 0 else 0.0

            status_text = f"[performance]Streaming... Token {tokens_generated_count}/{max_new_tokens} | {tokens_per_sec:.1f} tok/s[/performance]"

            # Format for live update
            formatted_live_text = format_code_output(full_generated_text, syntax_highlight_lang)
            live.update(Panel(formatted_live_text, title=status_text, border_style="green", expand=False))

            generated_ids = torch.cat((generated_ids, next_token), dim=-1)
            if eos_token_id is not None and last_next_token_item == eos_token_id:
                break

    # After Live context finishes, print the final generated text and status
    final_status = ""
    # Check if loop completed due to EOS
    if eos_token_id is not None and last_next_token_item == eos_token_id:
        final_status = f"[performance]EOS token reached. Total tokens: {tokens_generated_count}. Avg: {tokens_per_sec:.1f} tok/s[/performance]"
    else: # Max tokens reached (or max_new_tokens was 0 if loop didn't run)
        final_status = f"[performance]Max tokens ({max_new_tokens if max_new_tokens > 0 else tokens_generated_count}) reached. Avg: {tokens_per_sec:.1f} tok/s[/performance]"

    logger.info(final_status)
    formatted_final_text = format_code_output(full_generated_text, syntax_highlight_lang)
    console.print(Panel(formatted_final_text, title="[bold]Streamed Output (Final)[/bold]", border_style="green", expand=False))

    return generated_ids, full_generated_text, tokens_per_sec


def interactive_mode(model, tokenizer, device, config_from_checkpoint, syntax_highlight_lang):
    """
    Runs an interactive command-line session for text generation with dynamic parameter adjustment.
    
    Allows users to enter prompts, receive streamed model responses with syntax highlighting, and modify generation parameters on the fly using commands. Supports conversation history, configurable settings, and special commands for help, clearing history, and exiting.
    """
    console.print(
        Panel(
            Text.assemble(
                ("[interactive]Interactive Mode Activated[/interactive]\n\n"
                 "Type your prompts and press Enter to generate.\n"
                 "To change settings, type e.g., ", Text("/set temp 0.7", style="bold dim_info"), " or ",
                 Text("/set tokens 50", style="bold dim_info"), "\n"
                 "Commands:\n"
                 "  /quit, /exit, /q - Exit interactive mode\n"
                 "  /clear - Clear conversation history\n"
                 "  /config - Show current generation settings\n"
                 "  /help - Show this help message")
            ), # End of Text.assemble
            title="[bold bright_magenta]Interactive Mode[/bold]",
            border_style="bright_magenta",
            expand=False
        ) # End of Panel
    ) # End of console.print
    conversation_history = ""
    generation_params = {
        "max_new_tokens": 100,
        "temperature": getattr(config_from_checkpoint, 'temperature', 0.7),
        "top_k": getattr(config_from_checkpoint, 'top_k', 50),
        "top_p": getattr(config_from_checkpoint, 'top_p', 0.95),
        "repetition_penalty": getattr(config_from_checkpoint, 'repetition_penalty', 1.1)
    }
    eos_token_id_for_gen = tokenizer.eos_token_id

    while True:
        try:
            user_input_raw = Prompt.ask(Text("\nYou", style="interactive"))
            if not user_input_raw.strip(): continue

            if user_input_raw.lower().startswith("/set"):
                try:
                    parts = user_input_raw.split()
                    if len(parts) < 3: raise ValueError("Not enough arguments for /set")
                    _, param, value_str = parts[0], parts[1].lower(), parts[2]

                    if param in ["temp", "temperature"]: generation_params["temperature"] = float(value_str)
                    elif param in ["tokens", "max_new_tokens"]: generation_params["max_new_tokens"] = int(value_str)
                    elif param == "top_k": generation_params["top_k"] = int(value_str)
                    elif param == "top_p": generation_params["top_p"] = float(value_str)
                    else: logger.warning(f"Unknown parameter for /set: {param}. Known: temp, tokens, top_k, top_p"); continue
                    logger.info(f"Parameter [param]{param}[/param] set to [info]{generation_params[param]}[/info]")
                except ValueError: logger.error("Invalid value for /set. Usage: /set <param> <value>")
                except Exception as e: logger.error(f"Error processing /set command: {e}", exc_info=True)
                continue
            elif user_input_raw.lower() in ['/quit', '/exit', '/q']:
                logger.info("[interactive]Exiting interactive mode. Goodbye![/interactive]")
                break
            elif user_input_raw.lower() == '/clear':
                conversation_history = ""
                logger.info("[success]Conversation history cleared.[/success]")
                continue
            elif user_input_raw.lower() == '/config': display_generation_params(generation_params); continue
            elif user_input_raw.lower() == '/help':
                console.print(Panel(
                    Text("Available commands:\n"
                         "  /quit, /exit, /q - Exit interactive mode\n"
                         "  /clear - Clear conversation history\n"
                         "  /config - Show current generation settings\n"
                         "  /set <param> <value> - Set generation param (e.g., /set temp 0.7)\n"
                         "     Params: temp, tokens, top_k, top_p\n"
                         "  /help - Show this help message", style="info"),
                    title="[bold yellow]Help[/bold]", border_style="yellow", expand=False
                )); continue

            full_prompt = f"{conversation_history}USER: {user_input_raw}\nASSISTANT:"
            input_ids = tokenizer.encode(full_prompt, return_tensors="pt", truncation=True, max_length=config_from_checkpoint.max_seq_len - generation_params["max_new_tokens"]).to(device)
            if input_ids.shape[1] == 0: logger.warning("Input prompt resulted in empty token sequence. Skipping."); continue

            console.print(Text("\nAssistant:", style="interactive"))
            _, response, _ = stream_generation(
                model, tokenizer, input_ids,
                generation_params["max_new_tokens"], generation_params["temperature"],
                generation_params["top_k"], generation_params["top_p"],
                generation_params["repetition_penalty"], eos_token_id_for_gen, device, syntax_highlight_lang
            )
            conversation_history += f"USER: {user_input_raw}\nASSISTANT: {response}\n"
            # Basic history trimming
            if len(tokenizer.encode(conversation_history)) > config_from_checkpoint.max_seq_len * 0.8 :
                logger.info("[dim_info]Conversation history trimmed due to length.[/dim_info]")
                parts = conversation_history.split("ASSISTANT:")
                conversation_history = "ASSISTANT:".join(parts[-(len(parts)//2):]) if len(parts) > 2 else conversation_history

        except KeyboardInterrupt: console.print(Text("\nGeneration interrupted. Type /quit to exit or continue.", style="warning"))
        except Exception as e: logger.error(f"Error in interactive mode: {e}", exc_info=True)

def main():
    """
    Runs the command-line interface for text generation using a Lunaris Codex model.
    
    Parses command-line arguments, loads the specified model checkpoint and tokenizer, and generates text based on the provided prompt or prompt file. Supports both interactive chat and single-prompt modes, with options for streaming output, syntax highlighting, and saving results to a file. Handles device selection, checkpoint integrity verification, tokenizer setup, and generation parameter configuration. Displays generation parameters, input prompt, and generated output with rich formatting. Logs progress, errors, and performance metrics throughout the process.
    """
    parser = argparse.ArgumentParser(
        description=f"Generate text using a trained Lunaris Codex model (Enhanced v{SCRIPT_VERSION}).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the model checkpoint (.pt file).")
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None, help="Tokenizer name or path. If None, tries checkpoint args.")
    parser.add_argument("--prompt", type=str, default="USER: Write a Python function to sort a list.\nASSISTANT:", help="Input prompt.")
    parser.add_argument("--prompt_file", type=str, default=None, help="Path to a file containing the prompt (overrides --prompt).")
    parser.add_argument("--max_new_tokens", type=int, default=150, help="Max new tokens for generation.")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature (e.g., 0.7). Overrides checkpoint/defaults.")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k filtering (e.g., 50). 0 to disable. Overrides.")
    parser.add_argument("--top_p", type=float, default=None, help="Nucleus (top-p) filtering (e.g., 0.95). 1.0 to disable. Overrides.")
    parser.add_argument("--repetition_penalty", type=float, default=None, help="Repetition penalty (e.g., 1.1). 1.0 is no penalty. Overrides.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device: 'cuda', 'cpu', 'mps'.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility (e.g., 42).")
    parser.add_argument("--output_file", type=str, default=None, help="Save full generated text (prompt + completion) to file.")
    parser.add_argument("--no_color", action="store_true", help="Disable colored output.")
    parser.add_argument("--interactive", "-i", action="store_true", help="Enable interactive chat mode.")
    parser.add_argument("--stream", action="store_true", help="Enable streaming generation for single prompt.")
    parser.add_argument("--syntax_highlight", type=str, default="auto", help="Language for syntax highlighting (python, js, cpp, java, text, auto).")

    args = parser.parse_args()

    if args.no_color: console.no_color = True
    show_header()

    if not validate_checkpoint_exists(args.checkpoint_path): sys.exit(1)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if args.device.startswith("cuda") and torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
        logger.info(f"Random seed set to: [info]{args.seed}[/info]")

    device = torch.device(args.device)
    if device.type == "cuda" and torch.cuda.is_available():
        logger.info(f"Using GPU: [model]{torch.cuda.get_device_name(0)}[/model] ([performance]{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB VRAM[/performance])")
    elif device.type == "mps" and torch.backends.mps.is_available(): logger.info("Using Apple Metal (MPS) device.")
    else: logger.info(f"Using device: [model]{device}[/model]")

    load_result = load_model_from_checkpoint(args.checkpoint_path, device)
    if load_result is None: sys.exit(1) # Critical error during model load
    model, model_config_from_ckpt, train_args_from_ckpt = load_result


    tokenizer_path_resolved = args.tokenizer_name_or_path or train_args_from_ckpt.get("tokenizer_name_or_path")
    if not tokenizer_path_resolved:
        tokenizer_path_resolved = "gpt2" # Sensible default if all else fails
        logger.warning(f"Tokenizer not specified or found in checkpoint. Defaulting to '{tokenizer_path_resolved}'. This might be incorrect.")
    else:
        source = "argument" if args.tokenizer_name_or_path else "checkpoint"
        logger.info(f"Using tokenizer specified by {source}: [model]{tokenizer_path_resolved}[/model]")

    with console.status("[model]Loading tokenizer...[/model]", spinner="dots"):
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path_resolved, trust_remote_code=True)
            if tokenizer.pad_token_id is None:
                if tokenizer.eos_token_id is not None:
                    tokenizer.pad_token_id = tokenizer.eos_token_id
                    tokenizer.pad_token = tokenizer.eos_token
                    logger.info(f"Tokenizer `pad_token_id` set to `eos_token_id`: {tokenizer.pad_token_id} ('{tokenizer.pad_token}')")
                else:
                    new_pad_token = "<|PAD|>"
                    tokenizer.add_special_tokens({'pad_token': new_pad_token})
                    logger.warning(f"Tokenizer lacked pad/eos. Added `pad_token`='{new_pad_token}' (ID: {tokenizer.pad_token_id}). Vocab size may change.")
            logger.info(f"Tokenizer ([model]{tokenizer_path_resolved}[/model]) loaded as class [model]{tokenizer.__class__.__name__}[/model] (Vocab size: {len(tokenizer)})")
        except Exception as e:
            logger.error(f"Failed to load tokenizer '{tokenizer_path_resolved}': {e}", exc_info=True)
            sys.exit(1)

    if args.interactive:
        interactive_mode(model, tokenizer, device, model_config_from_ckpt, args.syntax_highlight)
        return

    prompt = args.prompt
    if args.prompt_file:
        try:
            with open(args.prompt_file, 'r', encoding='utf-8') as f: prompt = f.read()
            logger.info(f"Loaded prompt from file: [dim_info]{args.prompt_file}[/dim_info]")
        except Exception as e:
            logger.error(f"Failed to load prompt from file '[dim_info]{args.prompt_file}[/dim_info]': {e}", exc_info=True)
            sys.exit(1)
    if not prompt.strip(): logger.error("Prompt is empty."); sys.exit(1)

    gen_temp = args.temperature if args.temperature is not None else getattr(model_config_from_ckpt, 'temperature', 0.7)
    gen_top_k = args.top_k if args.top_k is not None else getattr(model_config_from_ckpt, 'top_k', 50)
    gen_top_p = args.top_p if args.top_p is not None else getattr(model_config_from_ckpt, 'top_p', 0.95)
    gen_rep_penalty = args.repetition_penalty if args.repetition_penalty is not None else getattr(model_config_from_ckpt, 'repetition_penalty', 1.0)
    eos_token_id_for_gen = tokenizer.eos_token_id

    display_generation_params({
        "Max new tokens": args.max_new_tokens, "Temperature": f"{gen_temp:.2f}", "Top-k": gen_top_k,
        "Top-p": f"{gen_top_p:.2f}", "Repetition penalty": f"{gen_rep_penalty:.2f}",
        "EOS token ID": eos_token_id_for_gen if eos_token_id_for_gen is not None else "N/A"
    })
    console.print(Panel(Text(prompt, style="prompt"), title="[bold]Input Prompt[/bold]", border_style="yellow", box=ROUNDED, expand=False))

    try:
        logger.info("Encoding prompt and preparing for generation...")
        input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=model_config_from_ckpt.max_seq_len - args.max_new_tokens).to(device)
        if input_ids.shape[1] == 0: logger.error("Input prompt resulted in empty token sequence. Cannot generate."); sys.exit(1)

        logger.info(f"Starting text generation... (Input tokens: [info]{input_ids.shape[1]}[/info])")
        generated_text_completion = ""
        tokens_per_sec = 0.0

        if args.stream:
            _, generated_text_completion, tokens_per_sec = stream_generation(
                model, tokenizer, input_ids, args.max_new_tokens, gen_temp, gen_top_k,
                gen_top_p, gen_rep_penalty, eos_token_id_for_gen, device, args.syntax_highlight
            )
            # Final status and text are printed by stream_generation itself
        else: # Non-streaming
            start_time = time.time()
            with console.status("[model]Generating text...[/model]", spinner="dots"):
                with torch.no_grad():
                    generated_ids_full = model.generate(
                        input_ids, max_new_tokens=args.max_new_tokens, temperature=max(gen_temp, 1e-5),
                        top_k=gen_top_k if gen_top_k > 0 else None, top_p=gen_top_p if gen_top_p < 1.0 else None,
                        repetition_penalty=gen_rep_penalty, eos_token_id=eos_token_id_for_gen,
                        pad_token_id=tokenizer.pad_token_id
                    )
            generation_time = time.time() - start_time
            num_prompt_tokens = input_ids.shape[1]
            generated_ids_new_only = generated_ids_full[0, num_prompt_tokens:]
            tokens_generated_count = len(generated_ids_new_only)
            tokens_per_sec = tokens_generated_count / generation_time if generation_time > 0 else 0.0
            generated_text_completion = tokenizer.decode(generated_ids_new_only, skip_special_tokens=True)

            panel_title = f"Generated Text - [performance]{tokens_per_sec:.1f} tokens/sec[/performance]"
            formatted_output = format_code_output(generated_text_completion, args.syntax_highlight)
            console.print(Panel(formatted_output, title=f"[bold]{panel_title}[/bold]", border_style="green", box=ROUNDED, expand=False))

        if args.output_file:
            try:
                output_path = Path(args.output_file)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                full_text_to_save = prompt + generated_text_completion
                with open(output_path, 'w', encoding='utf-8') as f:
                    if args.output_file.endswith(('.md', '.markdown')):
                        f.write(f"# Generated Text\n\n**Input Prompt:**\n```\n{prompt}\n```\n\n**Generated Response:**\n")
                        f.write(f"```{args.syntax_highlight if args.syntax_highlight != 'auto' else 'text'}\n{generated_text_completion}\n```\n\n")
                        f.write(f"**Generation Parameters & Performance:**\n")
                        f.write(f"- Max new tokens: {args.max_new_tokens}\n- Temperature: {gen_temp:.2f}\n- Top-k: {gen_top_k}\n- Top-p: {gen_top_p:.2f}\n")
                        f.write(f"- Repetition penalty: {gen_rep_penalty:.2f}\n- Performance: {tokens_per_sec:.1f} tokens/sec\n")
                    else: f.write(full_text_to_save)
                logger.info(f"[success]✓ Generated text saved to:[/success] [dim_info]{output_path.resolve()}[/dim_info]")
            except Exception as e:
                logger.error(f"Failed to save generated text to file '[dim_info]{args.output_file}[/dim_info]': {e}", exc_info=True)

    except Exception as e:
        logger.error(f"An unexpected error occurred during text generation: {e}", exc_info=True)
        sys.exit(1)

    final_memory = get_memory_usage()
    logger.info(f"Final memory usage: [performance]{final_memory:.1f} MB[/performance]")
    logger.info(Text("✓ Text generation process completed.", style="success"))
    console.print() # Extra newline for cleaner exit

if __name__ == "__main__":
    main()
