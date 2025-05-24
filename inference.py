# inference.py
import torch
from transformers import AutoTokenizer
import argparse
import logging
import sys
import os
import time
import psutil
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
from rich.prompt import Prompt, Confirm
from pygments.util import ClassNotFound # <-- ADICIONADO IMPORT

# Project-specific imports
from model import LunarisMind, LunarisCodexConfig  # Assuming model.py is in the same directory

# --- SCRIPT VERSION ---
SCRIPT_VERSION = "0.3.0"

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
    "interactive": "bright_magenta"
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
    subtitle = Text(f"v{SCRIPT_VERSION} - Enhanced Edition", style="dim")
    header = Text.assemble(title, " ", subtitle)

    console.print("\n")
    console.print(Panel(header, border_style="blue", box=ROUNDED))
    console.print("\n")

def validate_checkpoint(checkpoint_path: str) -> bool:
    """Validate checkpoint file exists and has required keys"""
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint file not found: {checkpoint_path}")
        return False

    try:
        file_size = os.path.getsize(checkpoint_path) / (1024 * 1024)  # MB
        logger.info(f"Checkpoint file size: {file_size:.2f} MB")

        if file_size < 1:
            logger.warning("Checkpoint file seems very small, validation may fail")
        return True
    except Exception as e:
        logger.error(f"Error validating checkpoint: {e}")
        return False

def get_memory_usage():
    """Get current memory usage"""
    process = psutil.Process()
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024  # MB

def load_model_from_checkpoint(checkpoint_path: str, device: torch.device) -> tuple[LunarisMind, LunarisCodexConfig, dict]:
    """
    Loads the model, its configuration, and training arguments from a checkpoint.
    Enhanced with better error handling and validation.
    """
    with Progress(
        SpinnerColumn(),
        TextColumn("[model]Loading model from checkpoint...[/model]"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task("Loading...", total=100)

        try:
            progress.update(task, advance=20)
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            progress.update(task, advance=30)
        except FileNotFoundError:
            logger.error(f"Checkpoint file not found: {checkpoint_path}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Failed to load checkpoint '{checkpoint_path}': {e}", exc_info=True)
            sys.exit(1)

        required_keys = ["config", "model_state_dict"]
        missing_keys_list = [key for key in required_keys if key not in checkpoint]
        if missing_keys_list: # Renomeado para evitar conflito com 'missing_keys' de load_state_dict
            logger.error(f"Checkpoint missing required keys: {missing_keys_list}")
            sys.exit(1)

        progress.update(task, advance=20)

        try:
            model_config_dict = checkpoint["config"]
            required_config_fields = ["vocab_size", "d_model", "n_layers", "n_heads", "max_seq_len"]

            for field in required_config_fields:
                if field not in model_config_dict:
                    logger.error(f"Checkpoint 'config' is missing required field: '{field}'.")
                    sys.exit(1)

                if model_config_dict[field] is None:
                    logger.error(f"Config field '{field}' is None. This is required.")
                    sys.exit(1)

            model_config = LunarisCodexConfig(**model_config_dict)
            logger.info(f"Model configuration loaded successfully")
            progress.update(task, advance=20)

        except TypeError as e:
            logger.error(f"Error instantiating LunarisCodexConfig: {e}")
            logger.error(f"Config dictionary from checkpoint: {checkpoint.get('config')}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Unexpected error loading model configuration: {e}", exc_info=True)
            sys.exit(1)

        model = LunarisMind(model_config)
        progress.update(task, advance=20)

        model_state_dict = checkpoint["model_state_dict"]
        is_compiled_checkpoint = any(k.startswith("_orig_mod.") for k in model_state_dict.keys())
        if is_compiled_checkpoint:
            logger.info("Checkpoint appears to be from a compiled model. Stripping '_orig_mod.' prefix.")
            model_state_dict = {k.replace('_orig_mod.', ''): v for k, v in model_state_dict.items()}

        missing_keys_load, unexpected_keys_load = model.load_state_dict(model_state_dict, strict=False) # Renomeado
        if missing_keys_load:
            logger.warning(f"Missing keys when loading model state_dict: {missing_keys_load}")
        if unexpected_keys_load:
            logger.warning(f"Unexpected keys when loading model state_dict: {unexpected_keys_load}")

        model.to(device)
        model.eval()
        progress.update(task, advance=10, description="[model]Model loaded![/model]")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    info_table = Table(show_header=False, box=ROUNDED)
    info_table.add_column("Property", style="model")
    info_table.add_column("Value", style="success")
    info_table.add_row("Architecture", "Lunaris Mind")
    info_table.add_row("Layers", str(model_config.n_layers))
    info_table.add_row("Attention Heads", str(model_config.n_heads))
    info_table.add_row("Model Dimension", str(model_config.d_model))
    info_table.add_row("Vocabulary Size", f"{model_config.vocab_size:,}")
    info_table.add_row("Max Sequence Length", f"{model_config.max_seq_len:,}")
    info_table.add_row("Total Parameters", f"{total_params:,}")
    info_table.add_row("Trainable Parameters", f"{trainable_params:,}")
    info_table.add_row("Parameter Ratio", f"{trainable_params/total_params*100:.2f}%")
    info_table.add_row("Memory Usage", f"{get_memory_usage():.1f} MB")
    console.print(Panel(info_table, title="Model Information", border_style="green"))

    train_args_from_checkpoint = checkpoint.get("args", {})
    return model, model_config, train_args_from_checkpoint

def display_generation_params(params):
    """Display generation parameters in a formatted table"""
    param_table = Table(show_header=False, box=ROUNDED)
    param_table.add_column("Parameter", style="param")
    param_table.add_column("Value", style="success")
    for k, v in params.items():
        param_table.add_row(k, str(v))
    console.print(Panel(param_table, title="Generation Parameters", border_style="magenta"))

def format_code_output(text: str, language: str = "python") -> Syntax:
    """Format code output with syntax highlighting"""
    try:
        return Syntax(text, language, theme="monokai", line_numbers=True)
    except ClassNotFound: # <-- MUDANÇA AQUI
        logger.warning(f"Syntax highlighting for language '{language}' not found. Falling back to 'text'.")
        return Syntax(text, "text", theme="monokai", line_numbers=False)
    except Exception as e: # <-- MUDANÇA AQUI
        logger.warning(f"Error during syntax highlighting (lang: {language}): {e}. Falling back to 'text'.")
        return Syntax(text, "text", theme="monokai", line_numbers=False)

def stream_generation(model, tokenizer, input_ids, max_new_tokens, temperature, top_k, top_p, repetition_penalty, eos_token_id, device):
    """Stream generation token by token for real-time output"""
    model.eval()
    generated_ids = input_ids.clone()
    generated_text = ""
    start_time = time.time()

    with Live(console=console, refresh_per_second=10) as live:
        for i in range(max_new_tokens):
            current_seq_len = generated_ids.size(1)
            batch_size = generated_ids.size(0)
            current_attention_mask = torch.ones((batch_size, current_seq_len), dtype=torch.long, device=device)

            with torch.no_grad():
                logits = model.forward(generated_ids, attention_mask=current_attention_mask)[:, -1, :]

            logits = logits / temperature
            logits = model._apply_repetition_penalty_optimized(logits, generated_ids, repetition_penalty)

            if top_k > 0:
                top_k_values, _ = torch.topk(logits, top_k)
                kth_value = top_k_values[:, -1].unsqueeze(-1)
                logits[logits < kth_value] = float('-inf')

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = torch.zeros_like(logits, dtype=torch.bool).scatter_(
                    dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
                logits.masked_fill_(indices_to_remove, float('-inf'))

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            new_token_text = tokenizer.decode(next_token[0], skip_special_tokens=True)
            generated_text += new_token_text
            elapsed_time = time.time() - start_time
            tokens_per_sec = (i + 1) / elapsed_time if elapsed_time > 0 else 0
            status_text = f"[performance]Generating... Token {i+1}/{max_new_tokens} | {tokens_per_sec:.1f} tok/s[/performance]"
            output_panel = Panel(
                Text(generated_text, style="generation"),
                title=status_text,
                border_style="green"
            )
            live.update(output_panel)
            generated_ids = torch.cat((generated_ids, next_token), dim=-1)
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break
    return generated_ids, generated_text, tokens_per_sec

def interactive_mode(model, tokenizer, device, config):
    """Interactive chat mode for continuous generation"""
    console.print(Panel(
        "[interactive]Interactive Mode Activated[/interactive]\n\n"
        "Type your prompts and press Enter to generate.\n"
        "Commands:\n"
        "  /quit - Exit interactive mode\n"
        "  /clear - Clear conversation history\n"
        "  /config - Show current generation settings\n"
        "  /help - Show this help message",
        title="Interactive Mode",
        border_style="bright_magenta"
    ))
    conversation_history = ""
    generation_params = {
        "max_new_tokens": 100,
        "temperature": getattr(config, 'temperature', 0.8),
        "top_k": getattr(config, 'top_k', 50),
        "top_p": getattr(config, 'top_p', 0.95),
        "repetition_penalty": getattr(config, 'repetition_penalty', 1.1)
    }
    while True:
        try:
            user_input = Prompt.ask("\n[interactive]You[/interactive]")
            if user_input.lower() in ['/quit', '/exit', '/q']:
                console.print("[interactive]Goodbye![/interactive]")
                break
            elif user_input.lower() == '/clear':
                conversation_history = ""
                console.print("[success]Conversation history cleared.[/success]")
                continue
            elif user_input.lower() == '/config':
                display_generation_params(generation_params)
                continue
            elif user_input.lower() == '/help':
                console.print(Panel(
                    "Available commands:\n"
                    "  /quit, /exit, /q - Exit interactive mode\n"
                    "  /clear - Clear conversation history\n"
                    "  /config - Show current generation settings\n"
                    "  /help - Show this help message",
                    title="Help",
                    border_style="yellow"
                ))
                continue
            full_prompt = f"{conversation_history}USER: {user_input}\nASSISTANT:"
            input_ids = tokenizer.encode(full_prompt, return_tensors="pt").to(device)
            console.print(f"\n[interactive]Assistant[/interactive]:")
            start_time = time.time()
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids,
                    max_new_tokens=generation_params["max_new_tokens"],
                    temperature=generation_params["temperature"],
                    top_k=generation_params["top_k"],
                    top_p=generation_params["top_p"],
                    repetition_penalty=generation_params["repetition_penalty"],
                    eos_token_id=tokenizer.eos_token_id
                )
            num_prompt_tokens = input_ids.shape[1]
            generated_ids_new = generated_ids[0, num_prompt_tokens:]
            response = tokenizer.decode(generated_ids_new, skip_special_tokens=True)
            generation_time = time.time() - start_time
            tokens_generated = len(generated_ids_new)
            tokens_per_sec = tokens_generated / generation_time if generation_time > 0 else 0
            console.print(Panel(
                Text(response, style="generation"),
                title=f"[performance]{tokens_per_sec:.1f} tokens/sec[/performance]",
                border_style="green"
            ))
            conversation_history += f"USER: {user_input}\nASSISTANT: {response}\n"
        except KeyboardInterrupt:
            console.print("\n[interactive]Interactive mode interrupted. Use /quit to exit properly.[/interactive]")
        except Exception as e:
            logger.error(f"Error in interactive mode: {e}", exc_info=True) # Adicionado exc_info=True para mais detalhes

def main():
    parser = argparse.ArgumentParser(description=f"Generate text using a trained Lunaris Codex model (Enhanced v{SCRIPT_VERSION}).")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the model checkpoint (.pt file).")
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None, help="Tokenizer name or path (e.g., 'gpt2', 'bigcode/starcoder').")
    parser.add_argument("--prompt", type=str, default="USER: Write a Python function to sort a list.\nASSISTANT:", help="Input prompt for the model.")
    parser.add_argument("--prompt_file", type=str, default=None, help="Path to a file containing the prompt (alternative to --prompt).")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Maximum number of new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature. Overrides model config if set.")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k filtering. Overrides model config if set. 0 to disable.")
    parser.add_argument("--top_p", type=float, default=None, help="Nucleus (top-p) filtering. Overrides model config if set. 1.0 to disable.")
    parser.add_argument("--repetition_penalty", type=float, default=None, help="Repetition penalty. 1.0 means no penalty. Overrides model config if set.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for inference (e.g., 'cuda', 'cpu').")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for generation (for reproducibility).")
    parser.add_argument("--output_file", type=str, default=None, help="Save the generated text to a file.")
    parser.add_argument("--no_color", action="store_true", help="Disable colored output.")
    parser.add_argument("--interactive", action="store_true", help="Enable interactive chat mode.")
    parser.add_argument("--stream", action="store_true", help="Enable streaming generation (real-time token output).")
    parser.add_argument("--syntax_highlight", type=str, default="python", help="Language for syntax highlighting (python, javascript, etc.).")
    args = parser.parse_args()

    if args.no_color:
        console.no_color = True
    show_header()
    if not validate_checkpoint(args.checkpoint_path):
        sys.exit(1)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if args.device == "cuda" and torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        logger.info(f"Random seed set to: {args.seed}")

    device = torch.device(args.device)
    if device.type == "cuda" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"Using GPU: [bold]{gpu_name}[/bold] ({gpu_memory:.1f}GB)")
    else:
        logger.info(f"Using device: [bold]{device}[/bold]")

    model, model_config, train_args = load_model_from_checkpoint(args.checkpoint_path, device)
    tokenizer_path = args.tokenizer_name_or_path
    if not tokenizer_path:
        tokenizer_path = train_args.get("tokenizer_name_or_path")
        if not tokenizer_path:
            logger.error("Tokenizer not specified via --tokenizer_name_or_path and not found in checkpoint's training arguments.")
            sys.exit(1)
        logger.info(f"Using tokenizer from training arguments: [bold]{tokenizer_path}[/bold]")

    with console.status("[model]Loading tokenizer...[/model]", spinner="dots"):
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
            if tokenizer.pad_token_id is None:
                if tokenizer.eos_token_id is not None:
                    tokenizer.pad_token_id = tokenizer.eos_token_id
                    tokenizer.pad_token = tokenizer.eos_token
                    logger.info(f"Set tokenizer pad_token_id to eos_token_id: {tokenizer.pad_token_id}")
                else:
                    tokenizer.add_special_tokens({'pad_token': '<|PAD|>'})
                    logger.warning(f"Tokenizer lacked pad/eos. Added new pad_token='<|PAD|>' (ID: {tokenizer.pad_token_id})")
            logger.info(f"Tokenizer loaded successfully: [bold]{tokenizer.__class__.__name__}[/bold]")
        except Exception as e:
            logger.error(f"Failed to load tokenizer '{tokenizer_path}': {e}", exc_info=True)
            sys.exit(1)

    if args.interactive:
        interactive_mode(model, tokenizer, device, model_config)
        return

    prompt = args.prompt
    if args.prompt_file:
        try:
            with open(args.prompt_file, 'r', encoding='utf-8') as f:
                prompt = f.read()
            logger.info(f"Loaded prompt from file: [bold]{args.prompt_file}[/bold]")
        except Exception as e:
            logger.error(f"Failed to load prompt from file '{args.prompt_file}': {e}", exc_info=True) # Adicionado exc_info=True
            sys.exit(1)

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    gen_temp = args.temperature if args.temperature is not None else getattr(model_config, 'temperature', 0.7)
    gen_top_k = args.top_k if args.top_k is not None else getattr(model_config, 'top_k', 0)
    gen_top_p = args.top_p if args.top_p is not None else getattr(model_config, 'top_p', 0.9)
    gen_rep_penalty = args.repetition_penalty if args.repetition_penalty is not None else getattr(model_config, 'repetition_penalty', 1.0)
    eos_token_id_for_gen = tokenizer.eos_token_id

    display_generation_params({
        "Max new tokens": args.max_new_tokens, "Temperature": gen_temp, "Top-k": gen_top_k,
        "Top-p": gen_top_p, "Repetition penalty": gen_rep_penalty, "EOS token ID": eos_token_id_for_gen
    })
    console.print(Panel(Text(prompt, style="prompt"), title="Input Prompt", border_style="yellow", box=ROUNDED))
    logger.info("Starting text generation...")

    if args.stream:
        generated_ids_full, generated_text, tokens_per_sec = stream_generation(
            model, tokenizer, input_ids, args.max_new_tokens, gen_temp, gen_top_k,
            gen_top_p, gen_rep_penalty, eos_token_id_for_gen, device
        )
        console.print(f"\n[performance]Generation completed at {tokens_per_sec:.1f} tokens/second[/performance]")
    else:
        start_time = time.time()
        with console.status("[model]Generating text...[/model]", spinner="dots"):
            with torch.no_grad():
                generated_ids_full = model.generate(
                    input_ids, max_new_tokens=args.max_new_tokens, temperature=gen_temp, top_k=gen_top_k,
                    top_p=gen_top_p, repetition_penalty=gen_rep_penalty, eos_token_id=eos_token_id_for_gen
                )
        generation_time = time.time() - start_time
        num_prompt_tokens = input_ids.shape[1]
        generated_ids_new_only = generated_ids_full[0, num_prompt_tokens:]
        tokens_generated = len(generated_ids_new_only)
        tokens_per_sec = tokens_generated / generation_time if generation_time > 0 else 0
        generated_text = tokenizer.decode(generated_ids_new_only, skip_special_tokens=True)

        panel_title = f"Generated Text - [performance]{tokens_per_sec:.1f} tokens/sec[/performance]" # Definido uma vez
        if args.syntax_highlight and generated_text.strip():
            try:
                highlighted_text_object = format_code_output(generated_text, args.syntax_highlight) # format_code_output agora lida com seus erros
                console.print(Panel(
                    highlighted_text_object,
                    title=panel_title,
                    border_style="green",
                    box=ROUNDED
                ))
            except Exception as e: # <-- MUDANÇA AQUI: Captura exceções ao tentar exibir/criar o Panel
                logger.error(f"Error displaying generated text with highlighting: {e}. Displaying as plain text.", exc_info=True)
                console.print(Panel(
                    Text(generated_text, style="generation"), # Fallback para texto simples
                    title=panel_title,
                    border_style="green",
                    box=ROUNDED
                ))
        else:
            console.print(Panel(
                Text(generated_text, style="generation"),
                title=panel_title,
                border_style="green",
                box=ROUNDED
            ))

    if args.output_file:
        try:
            output_path = Path(args.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                if args.output_file.endswith('.md'):
                    f.write(f"# Generated Text\n\n**Prompt:** {prompt}\n\n**Generated Response:**\n\n")
                    f.write(f"```{args.syntax_highlight}\n{generated_text}\n```\n\n**Generation Parameters:**\n")
                    f.write(f"- Temperature: {gen_temp}\n- Top-k: {gen_top_k}\n- Top-p: {gen_top_p}\n")
                    f.write(f"- Repetition penalty: {gen_rep_penalty}\n- Performance: {tokens_per_sec:.1f} tokens/sec\n")
                else:
                    f.write(prompt + generated_text)
            logger.info(f"[success]✓ Generated text saved to:[/success] [bold]{output_path}[/bold]")
        except Exception as e:
            logger.error(f"Failed to save generated text to file '{args.output_file}': {e}", exc_info=True) # Adicionado exc_info=True

    final_memory = get_memory_usage()
    console.print(f"\n[performance]Final memory usage: {final_memory:.1f} MB[/performance]")
    console.print("\n[success]✓ Text generation completed[/success]\n")

if __name__ == "__main__":
    main()
