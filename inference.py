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
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn # Corrected import
from rich.syntax import Syntax
from rich.theme import Theme
from rich.text import Text
from rich.box import ROUNDED
from rich.table import Table
from rich.live import Live
from rich.prompt import Prompt
from pygments.util import ClassNotFound

# Project-specific imports
from model import LunarisMind, LunarisCodexConfig

# --- SCRIPT VERSION ---
SCRIPT_VERSION = "0.3.6" # Applied compatibility fixes

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

try:
    console_width = os.get_terminal_size().columns
    if console_width < 80:
        console_width = 120
except (OSError, AttributeError): # Fallback if get_terminal_size() fails or not available
    console_width = 120

console = Console(theme=custom_theme, width=console_width)

# Setup logging with rich handler
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True, markup=True, show_path=False)]
)
logger = logging.getLogger("lunaris_inference") # Changed logger name for clarity

def show_header():
    title = Text("Lunaris Codex Inference Engine", style="bold blue")
    subtitle = Text(f"v{SCRIPT_VERSION} - Enhanced Edition", style="dim")
    header_text = Text.assemble(title, " ", subtitle)
    console.print("\n", Panel(header_text, border_style="blue", box=ROUNDED, expand=False), "\n")

def compute_sha256(filepath: str) -> str | None:
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
    hash_file = checkpoint_path + ".sha256"
    if not os.path.exists(hash_file):
        logger.warning(f"No hash file found for [dim_info]{checkpoint_path}[/dim_info]. Skipping SHA-256 verification.")
        return True
    try:
        with open(hash_file, "r") as f:
            content = f.read().split()
            if not content:
                logger.warning(f"Hash file [dim_info]{hash_file}[/dim_info] is empty. Assuming valid.")
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
        else: # actual_hash is None
            logger.warning(f"Could not compute SHA-256 for [dim_info]{checkpoint_path}[/dim_info] to verify. Assuming valid for now.")
            return True
    except Exception as e:
        logger.warning(f"Could not verify checkpoint integrity for [dim_info]{checkpoint_path}[/dim_info] due to an error: {e}", exc_info=True)
        return True # Be lenient if verification process fails

def validate_checkpoint_exists(checkpoint_path: str) -> bool:
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint file not found: [dim_info]{checkpoint_path}[/dim_info]")
        return False
    try:
        file_size = os.path.getsize(checkpoint_path) / (1024 * 1024)
        logger.info(f"Checkpoint file found: [model]{os.path.basename(checkpoint_path)}[/model] ({file_size:.2f} MB)")
        if file_size < 0.1: # Arbitrary small size threshold
            logger.warning("Checkpoint file seems very small. Ensure it's a valid model checkpoint.")
        return True
    except Exception as e:
        logger.error(f"Error accessing checkpoint file [dim_info]{checkpoint_path}[/dim_info]: {e}", exc_info=True)
        return False

def get_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024

def load_model_from_checkpoint(checkpoint_path: str, device: torch.device) -> tuple[LunarisMind, LunarisCodexConfig, dict] | None:
    if not verify_checkpoint_integrity(checkpoint_path):
        logger.error(f"Aborting due to checkpoint integrity verification failure for: [dim_info]{checkpoint_path}[/dim_info]")
        return None

    progress_description = "[model]Loading model from checkpoint...[/model]"
    with Progress(
        SpinnerColumn(), TextColumn("{task.description}"), BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console, transient=True
    ) as progress:
        task = progress.add_task(progress_description, total=100)
        progress.update(task, advance=5, description="[model]Loading checkpoint file...[/model]")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            progress.update(task, advance=25, description="[model]Checkpoint data loaded.[/model]")
        except FileNotFoundError:
            logger.error(f"Checkpoint file not found: [dim_info]{checkpoint_path}[/dim_info]")
            return None
        except Exception as e:
            logger.error(f"Failed to load checkpoint file '[dim_info]{checkpoint_path}[/dim_info]': {e}", exc_info=True)
            return None

        # Use "config_args" if from new train.py, else fallback to "config"
        model_config_dict_key = "config_args" if "config_args" in checkpoint else "config"
        if model_config_dict_key not in checkpoint:
            logger.error(f"Checkpoint is missing model configuration (expected key 'config_args' or 'config').")
            return None
        if "model_state_dict" not in checkpoint:
            logger.error("Checkpoint is missing 'model_state_dict'.")
            return None
        progress.update(task, advance=10, description="[model]Validating checkpoint keys...[/model]")

        try:
            model_config_dict = checkpoint[model_config_dict_key]
            required_config_fields = ["vocab_size", "d_model", "n_layers", "n_heads", "max_seq_len"]
            for field in required_config_fields:
                if field not in model_config_dict:
                    logger.error(f"Checkpoint config (key: '{model_config_dict_key}') is missing required field: '{field}'.")
                    return None
                if model_config_dict[field] is None and field != 'pad_token_id': # pad_token_id can be None initially from old ckpts
                    logger.error(f"Config field '{field}' is None in checkpoint, which is invalid.")
                    return None

            if 'lora_rank' not in model_config_dict:
                model_config_dict['lora_rank'] = 0 # Default for older checkpoints
                logger.info("LoRA rank not found in checkpoint config, defaulting to 0 (no LoRA).")
            if 'pad_token_id' not in model_config_dict:
                # LunarisCodexConfig now has a default for pad_token_id, so this is mostly for info
                logger.info("`pad_token_id` not found in checkpoint config; `LunarisCodexConfig` default will be used.")

            model_config = LunarisCodexConfig(**model_config_dict)
            logger.info("Model configuration loaded successfully from checkpoint.")
            progress.update(task, advance=20, description="[model]Model config processed.[/model]")
        except TypeError as e:
            logger.error(f"Error instantiating LunarisCodexConfig from checkpoint's config (key: '{model_config_dict_key}'): {e}", exc_info=True)
            logger.error("This might be due to a mismatch between saved config keys and LunarisCodexConfig constructor.")
            logger.error(f"Config dictionary from checkpoint: {checkpoint.get(model_config_dict_key)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error loading model configuration: {e}", exc_info=True)
            return None

        model = LunarisMind(model_config)
        model_state_dict = checkpoint["model_state_dict"]
        is_compiled_checkpoint = any(k.startswith("_orig_mod.") for k in model_state_dict.keys())
        if is_compiled_checkpoint:
            logger.info("Checkpoint appears to be from a torch.compiled model. Stripping '_orig_mod.' prefix from state_dict keys.")
            model_state_dict = {k.replace('_orig_mod.', ''): v for k, v in model_state_dict.items()}

        missing_keys_load, unexpected_keys_load = model.load_state_dict(model_state_dict, strict=False)
        if missing_keys_load: logger.warning(f"Missing keys when loading model state_dict: {missing_keys_load}")
        if unexpected_keys_load: logger.warning(f"Unexpected keys when loading model state_dict: {unexpected_keys_load}")
        progress.update(task, advance=20, description="[model]Model state loaded.[/model]")

        model.to(device)
        model.eval()
        progress.update(task, advance=20, description="[model]Model ready on device![/model]")

    total_params = sum(p.numel() for p in model.parameters())
    info_table = Table(show_header=False, box=ROUNDED, title_style="bold blue", border_style="blue")
    info_table.add_column("Property", style="param"); info_table.add_column("Value", style="info")
    info_table.add_row("Architecture", "Lunaris Mind"); info_table.add_row("Layers", str(model_config.n_layers))
    info_table.add_row("Attn Heads", str(model_config.n_heads)); info_table.add_row("d_model", str(model_config.d_model))
    info_table.add_row("Vocab Size", f"{model_config.vocab_size:,}"); info_table.add_row("Max Seq Len", f"{model_config.max_seq_len:,}")
    if model_config.lora_rank > 0: info_table.add_row("LoRA Rank", str(model_config.lora_rank))
    info_table.add_row("Pad Token ID (model cfg)", str(model_config.pad_token_id))
    info_table.add_row("Total Params", f"{total_params:,}")
    info_table.add_row("Loaded from", f"[dim_info]{os.path.basename(checkpoint_path)}[/dim_info]")
    console.print(Panel(info_table, title="[bold]Model Information[/bold]", border_style="green", expand=False))
    logger.info(f"Initial memory usage after model load: [performance]{get_memory_usage():.1f} MB[/performance]")

    train_args_from_checkpoint = checkpoint.get("train_args", checkpoint.get("args", {})) # Handle old "args" key
    return model, model_config, train_args_from_checkpoint

def display_generation_params(params):
    param_table = Table(show_header=False, box=ROUNDED, title_style="bold magenta", border_style="magenta")
    param_table.add_column("Parameter", style="param"); param_table.add_column("Value", style="info")
    for k, v in params.items(): param_table.add_row(k, str(v))
    console.print(Panel(param_table, title="[bold]Generation Parameters[/bold]", border_style="magenta", expand=False))

def format_code_output(text: str, language: str = "python") -> Syntax | Text:
    try:
        if not language or language.lower() == 'auto': language = "text"
        return Syntax(text, language, theme="monokai", line_numbers=True, word_wrap=True)
    except ClassNotFound:
        logger.warning(f"Syntax highlighter for language '{language}' not found. Falling back to plain text.")
        return Text(text)
    except Exception as e:
        logger.warning(f"Error during syntax highlighting (lang: {language}): {e}. Falling back to plain text.", exc_info=True)
        return Text(text)

def stream_generation(model, tokenizer, input_ids, max_new_tokens, temperature, top_k, top_p, repetition_penalty, eos_token_id, device, syntax_highlight_lang):
    model.eval()
    generated_ids = input_ids.clone()
    full_generated_text = ""
    start_time = time.time()
    last_next_token_item = -1
    tokens_generated_count = 0
    tokens_per_sec = 0.0

    with Live(console=console, refresh_per_second=12, transient=True) as live:
        for i in range(max_new_tokens):
            tokens_generated_count = i + 1
            current_seq_len = generated_ids.size(1)
            batch_size = generated_ids.size(0)
            # For streaming, the attention mask should cover the current sequence length
            current_attention_mask = torch.ones((batch_size, current_seq_len), dtype=torch.long, device=device)
            if model.config.pad_token_id is not None: # Use pad_token_id from model config if available
                 is_padding = (generated_ids == model.config.pad_token_id)
                 current_attention_mask[is_padding] = 0


            with torch.no_grad():
                # The model's forward pass will handle ALiBi.
                # The attention_mask here is primarily for any padding in generated_ids.
                logits = model.forward(generated_ids, attention_mask=current_attention_mask)[:, -1, :]

            logits = logits / max(temperature, 1e-5) # Avoid division by zero or too small temp
            if hasattr(model, '_apply_repetition_penalty_optimized') and repetition_penalty != 1.0:
                 # This method uses model.config.pad_token_id internally
                 logits = model._apply_repetition_penalty_optimized(logits, generated_ids, repetition_penalty)

            if top_k > 0:
                top_k_values, _ = torch.topk(logits, top_k)
                kth_value = top_k_values[:, -1].unsqueeze(-1)
                logits[logits < kth_value] = float('-inf')
            if 0.0 < top_p < 1.0: # Ensure top_p is within valid range
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = torch.zeros_like(logits, dtype=torch.bool).scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
                logits.masked_fill_(indices_to_remove, float('-inf'))

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            last_next_token_item = next_token.item()
            new_token_text = tokenizer.decode(next_token[0], skip_special_tokens=True) # Use next_token[0] for single batch item
            full_generated_text += new_token_text
            elapsed_time = time.time() - start_time
            tokens_per_sec = tokens_generated_count / elapsed_time if elapsed_time > 0 else 0.0

            status_text = f"[performance]Streaming... Token {tokens_generated_count}/{max_new_tokens} | {tokens_per_sec:.1f} tok/s[/performance]"
            formatted_live_text = format_code_output(full_generated_text, syntax_highlight_lang)
            live.update(Panel(formatted_live_text, title=status_text, border_style="green", expand=False))

            generated_ids = torch.cat((generated_ids, next_token), dim=-1)
            if eos_token_id is not None and last_next_token_item == eos_token_id:
                break

    final_status = (f"[performance]EOS token reached. Total tokens: {tokens_generated_count}. Avg: {tokens_per_sec:.1f} tok/s[/performance]"
                    if eos_token_id is not None and last_next_token_item == eos_token_id else
                    f"[performance]Max tokens ({max_new_tokens if max_new_tokens > 0 else tokens_generated_count}) reached. Avg: {tokens_per_sec:.1f} tok/s[/performance]")
    logger.info(final_status)
    formatted_final_text = format_code_output(full_generated_text, syntax_highlight_lang)
    console.print(Panel(formatted_final_text, title="[bold]Streamed Output (Final)[/bold]", border_style="green", expand=False))
    return generated_ids, full_generated_text, tokens_per_sec

def interactive_mode(model, tokenizer, device, model_config, syntax_highlight_lang): # model_config is LunarisCodexConfig instance
    console.print(
        Panel(Text.assemble(("[interactive]Interactive Mode Activated[/interactive]\n\n"
                 "Type your prompts and press Enter to generate.\n"
                 "To change settings, type e.g., ", Text("/set temp 0.7", style="bold dim_info"), " or ",
                 Text("/set tokens 50", style="bold dim_info"), "\n"
                 "Commands:\n"
                 "  /quit, /exit, /q - Exit interactive mode\n"
                 "  /clear - Clear conversation history\n"
                 "  /config - Show current generation settings\n"
                 "  /help - Show this help message")),
            title="[bold bright_magenta]Interactive Mode[/bold]", border_style="bright_magenta", expand=False))
    conversation_history = ""
    generation_params = {
        "max_new_tokens": 100,
        "temperature": model_config.temperature, # Use model_config defaults
        "top_k": model_config.top_k,
        "top_p": model_config.top_p,
        "repetition_penalty": model_config.repetition_penalty
    }
    eos_token_id_for_gen = tokenizer.eos_token_id

    while True:
        try:
            user_input_raw = Prompt.ask(Text("\nYou", style="interactive"))
            if not user_input_raw.strip(): continue
            if user_input_raw.lower().startswith("/set"):
                try:
                    parts = user_input_raw.split()
                    if len(parts) < 3: raise ValueError("Usage: /set <param> <value>")
                    param, value_str = parts[1].lower(), parts[2]
                    if param in ["temp", "temperature"]: generation_params["temperature"] = float(value_str)
                    elif param in ["tokens", "max_new_tokens"]: generation_params["max_new_tokens"] = int(value_str)
                    elif param == "top_k": generation_params["top_k"] = int(value_str)
                    elif param == "top_p": generation_params["top_p"] = float(value_str)
                    elif param == "rep_penalty": generation_params["repetition_penalty"] = float(value_str)
                    else: logger.warning(f"Unknown param for /set: {param}. Known: temp, tokens, top_k, top_p, rep_penalty"); continue
                    logger.info(f"Parameter [param]{param}[/param] set to [info]{generation_params.get(param, generation_params.get(parts[1]))}[/info]") # Robust get
                except ValueError as e: logger.error(f"Invalid value for /set: {e}")
                except Exception as e: logger.error(f"Error processing /set: {e}", exc_info=True)
                continue
            elif user_input_raw.lower() in ['/quit', '/exit', '/q']: logger.info("[interactive]Exiting interactive mode.[/interactive]"); break
            elif user_input_raw.lower() == '/clear': conversation_history = ""; logger.info("[success]Conversation history cleared.[/success]"); continue
            elif user_input_raw.lower() == '/config': display_generation_params(generation_params); continue
            elif user_input_raw.lower() == '/help': console.print(Panel(Text("Available commands:\n  /quit, /exit, /q\n  /clear\n  /config\n  /set <param> <value> (Params: temp, tokens, top_k, top_p, rep_penalty)\n  /help", style="info"),title="[bold yellow]Help[/bold]",border_style="yellow")); continue

            full_prompt = f"{conversation_history}USER: {user_input_raw}\nASSISTANT:"
            # Ensure max_length for encode respects available space for new tokens
            encode_max_len = model_config.max_seq_len - generation_params["max_new_tokens"]
            if encode_max_len <= 0: logger.error("max_new_tokens is too large for model_max_seq_len. Reduce max_new_tokens."); continue

            input_ids = tokenizer.encode(full_prompt, return_tensors="pt", truncation=True, max_length=encode_max_len).to(device)
            if input_ids.shape[1] == 0: logger.warning("Input prompt is empty after tokenization/truncation. Skipping."); continue

            console.print(Text("\nAssistant:", style="interactive"))
            _, response, _ = stream_generation(
                model, tokenizer, input_ids,
                generation_params["max_new_tokens"], generation_params["temperature"],
                generation_params["top_k"], generation_params["top_p"],
                generation_params["repetition_penalty"], eos_token_id_for_gen, device, syntax_highlight_lang)
            conversation_history += f"USER: {user_input_raw}\nASSISTANT: {response}\n"
            if len(tokenizer.encode(conversation_history)) > model_config.max_seq_len * 0.8:
                logger.info("[dim_info]Trimming conversation history due to length.[/dim_info]")
                # Simpler trim: keep last few exchanges
                history_parts = conversation_history.split("USER:")[1:] # Split by "USER:", ignore first empty if starts with USER
                if len(history_parts) > 3: # Keep roughly last 3 user-assistant pairs
                    conversation_history = "USER:" + "USER:".join(history_parts[-3:])
        except KeyboardInterrupt: console.print(Text("\nGeneration interrupted. Type /quit or continue.", style="warning"))
        except Exception as e: logger.error(f"Error in interactive mode: {e}", exc_info=True)

def main():
    parser = argparse.ArgumentParser(description=f"Generate text using Lunaris Codex model (v{SCRIPT_VERSION}).", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to model checkpoint (.pt).")
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None, help="Tokenizer. If None, tries checkpoint args or 'gpt2'.")
    parser.add_argument("--prompt", type=str, default="USER: Write a Python function to sort a list.\nASSISTANT:", help="Input prompt.")
    parser.add_argument("--prompt_file", type=str, default=None, help="File containing prompt (overrides --prompt).")
    parser.add_argument("--max_new_tokens", type=int, default=150, help="Max new tokens.")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature. Overrides model config.")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k. 0 to disable. Overrides model config.")
    parser.add_argument("--top_p", type=float, default=None, help="Top-p. 1.0 to disable. Overrides model config.")
    parser.add_argument("--repetition_penalty", type=float, default=None, help="Repetition penalty. Overrides model config.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device: cuda, cpu, mps.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--output_file", type=str, default=None, help="Save full generated text to file.")
    parser.add_argument("--no_color", action="store_true", help="Disable colored output.")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive chat mode.")
    parser.add_argument("--stream", action="store_true", help="Stream generation for single prompt.")
    parser.add_argument("--syntax_highlight", type=str, default="auto", help="Lang for syntax highlight (python, cpp, auto, etc.).")
    args = parser.parse_args()

    if args.no_color: console.no_color = True
    show_header()

    if not validate_checkpoint_exists(args.checkpoint_path): sys.exit(1)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if args.device.startswith("cuda") and torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
        logger.info(f"Random seed set to: [info]{args.seed}[/info]")

    device = torch.device(args.device)
    if device.type == "cuda" and torch.cuda.is_available(): logger.info(f"Using GPU: [model]{torch.cuda.get_device_name(0)}[/model]")
    elif device.type == "mps" and torch.backends.mps.is_available(): logger.info("Using Apple Metal (MPS) device.")
    else: logger.info(f"Using device: [model]{device}[/model]")

    load_result = load_model_from_checkpoint(args.checkpoint_path, device)
    if load_result is None: sys.exit(1)
    model, model_config_from_ckpt, train_args_from_ckpt = load_result

    tokenizer_path_resolved = args.tokenizer_name_or_path or train_args_from_ckpt.get("tokenizer_name_or_path")
    if not tokenizer_path_resolved:
        tokenizer_path_resolved = "gpt2" # A very common default
        logger.warning(f"Tokenizer not specified via --tokenizer_name_or_path or found in checkpoint 'train_args'. Defaulting to '{tokenizer_path_resolved}'. THIS MAY BE INCORRECT for your model.")
    else:
        source = "command line argument" if args.tokenizer_name_or_path else "checkpoint 'train_args'"
        logger.info(f"Using tokenizer specified by {source}: [model]{tokenizer_path_resolved}[/model]")

    with console.status("[model]Loading tokenizer...[/model]", spinner="dots"):
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path_resolved, trust_remote_code=True)
            if tokenizer.pad_token_id is None:
                if tokenizer.eos_token_id is not None:
                    tokenizer.pad_token_id = tokenizer.eos_token_id
                    tokenizer.pad_token = tokenizer.eos_token # Ensure pad_token attribute is also set
                    logger.info(f"Tokenizer `pad_token_id` was None, set to `eos_token_id`: {tokenizer.pad_token_id} ('{tokenizer.pad_token}')")
                else: # Add a new pad token if no EOS token exists either
                    new_pad_token = "<|PAD|>" # A common custom pad token
                    tokenizer.add_special_tokens({'pad_token': new_pad_token})
                    logger.warning(f"Tokenizer lacked `pad_token_id` and `eos_token_id`. Added new `pad_token`='{new_pad_token}' (ID: {tokenizer.pad_token_id}). Vocab size may change.")
            # Ensure model config's pad_token_id aligns with tokenizer if it was defaulted or from old checkpoint
            if model_config_from_ckpt.pad_token_id != tokenizer.pad_token_id:
                logger.info(f"Updating model's internal `pad_token_id` (was {model_config_from_ckpt.pad_token_id}) to match tokenizer's `pad_token_id` ({tokenizer.pad_token_id}).")
                model.config.pad_token_id = tokenizer.pad_token_id # Directly update the model's config instance

            logger.info(f"Tokenizer ([model]{tokenizer_path_resolved}[/model]) loaded. Class: [model]{tokenizer.__class__.__name__}[/model], Vocab: {len(tokenizer)}, Pad ID: {tokenizer.pad_token_id}")
        except Exception as e:
            logger.error(f"Failed to load tokenizer '{tokenizer_path_resolved}': {e}", exc_info=True)
            sys.exit(1)

    if args.interactive:
        interactive_mode(model, tokenizer, device, model_config_from_ckpt, args.syntax_highlight)
        return

    prompt_text = args.prompt
    if args.prompt_file:
        try:
            with open(args.prompt_file, 'r', encoding='utf-8') as f: prompt_text = f.read()
            logger.info(f"Loaded prompt from file: [dim_info]{args.prompt_file}[/dim_info]")
        except Exception as e:
            logger.error(f"Failed to load prompt from file '[dim_info]{args.prompt_file}[/dim_info]': {e}", exc_info=True); sys.exit(1)
    if not prompt_text.strip(): logger.error("Prompt is empty."); sys.exit(1)

    # Use model_config_from_ckpt for defaults, then override with CLI args if provided
    gen_temp = args.temperature if args.temperature is not None else model_config_from_ckpt.temperature
    gen_top_k = args.top_k if args.top_k is not None else model_config_from_ckpt.top_k
    gen_top_p = args.top_p if args.top_p is not None else model_config_from_ckpt.top_p
    gen_rep_penalty = args.repetition_penalty if args.repetition_penalty is not None else model_config_from_ckpt.repetition_penalty
    eos_token_id_for_gen = tokenizer.eos_token_id # Use tokenizer's EOS

    display_generation_params({
        "Max new tokens": args.max_new_tokens, "Temperature": f"{gen_temp:.2f}", "Top-k": gen_top_k,
        "Top-p": f"{gen_top_p:.2f}", "Repetition penalty": f"{gen_rep_penalty:.2f}",
        "EOS token ID": eos_token_id_for_gen if eos_token_id_for_gen is not None else "N/A",
        "Model Pad ID": model.config.pad_token_id, # Display what pad_token_id the model is using
        "Tokenizer Pad ID": tokenizer.pad_token_id
    })
    console.print(Panel(Text(prompt_text, style="prompt"), title="[bold]Input Prompt[/bold]", border_style="yellow", box=ROUNDED, expand=False))

    try:
        logger.info("Encoding prompt and preparing for generation...")
        # Ensure max_length for encode respects available space for new tokens
        encode_max_len = model_config_from_ckpt.max_seq_len - args.max_new_tokens
        if encode_max_len <=0:
            logger.error(f"max_new_tokens ({args.max_new_tokens}) is too large for model's max_seq_len ({model_config_from_ckpt.max_seq_len}). Reduce max_new_tokens."); sys.exit(1)
        input_ids = tokenizer.encode(prompt_text, return_tensors="pt", truncation=True, max_length=encode_max_len).to(device)
        if input_ids.shape[1] == 0: logger.error("Input prompt resulted in empty token sequence. Cannot generate."); sys.exit(1)

        logger.info(f"Starting text generation... (Input tokens: [info]{input_ids.shape[1]}[/info])")
        generated_text_completion = ""
        tokens_per_sec = 0.0

        if args.stream:
            _, generated_text_completion, tokens_per_sec = stream_generation(
                model, tokenizer, input_ids, args.max_new_tokens, gen_temp, gen_top_k,
                gen_top_p, gen_rep_penalty, eos_token_id_for_gen, device, args.syntax_highlight)
        else: # Non-streaming
            start_time = time.time()
            with console.status("[model]Generating text...[/model]", spinner="dots"):
                with torch.no_grad():
                    generated_ids_full = model.generate(
                        input_ids, max_new_tokens=args.max_new_tokens, temperature=max(gen_temp, 1e-5),
                        top_k=gen_top_k if gen_top_k > 0 else 0, # Pass 0 to model.generate if disabled
                        top_p=gen_top_p if 0.0 < gen_top_p < 1.0 else 1.0, # Pass 1.0 to model.generate if disabled
                        repetition_penalty=gen_rep_penalty, eos_token_id=eos_token_id_for_gen,
                        pad_token_id=tokenizer.pad_token_id # Crucial addition
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
                full_text_to_save = prompt_text + generated_text_completion

                # Get string representation from Rich objects if needed
                gen_text_str = generated_text_completion
                if isinstance(gen_text_str, (Text, Syntax)): # Should already be string from decode
                    gen_text_str = str(gen_text_str)

                with open(output_path, 'w', encoding='utf_8') as f:
                    if args.output_file.endswith(('.md', '.markdown')):
                        f.write(f"# Generated Text\n\n**Input Prompt:**\n```\n{prompt_text}\n```\n\n**Generated Response:**\n")
                        code_lang_hint = args.syntax_highlight if args.syntax_highlight.lower() != 'auto' else 'text'
                        f.write(f"```{code_lang_hint}\n{gen_text_str}\n```\n\n")
                        f.write(f"**Generation Parameters & Performance:**\n")
                        f.write(f"- Max new tokens: {args.max_new_tokens}\n- Temperature: {gen_temp:.2f}\n- Top-k: {gen_top_k}\n- Top-p: {gen_top_p:.2f}\n")
                        f.write(f"- Repetition penalty: {gen_rep_penalty:.2f}\n- Performance: {tokens_per_sec:.1f} tokens/sec\n")
                    else: f.write(full_text_to_save)
                logger.info(f"[success]✓ Generated text saved to:[/success] [dim_info]{output_path.resolve()}[/dim_info]")
            except Exception as e:
                logger.error(f"Failed to save generated text to file '[dim_info]{args.output_file}[/dim_info]': {e}", exc_info=True)
    except Exception as e:
        logger.error(f"An unexpected error occurred during text generation: {e}", exc_info=True); sys.exit(1)

    final_memory = get_memory_usage()
    logger.info(f"Final memory usage: [performance]{final_memory:.1f} MB[/performance]")
    logger.info(Text("✓ Text generation process completed.", style="success")); console.print()
if __name__ == "__main__":
    main()
