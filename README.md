[![Lunaris Codex CI](https://github.com/MeryylleA/lunariscodex/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/MeryylleA/lunariscodex/actions/workflows/ci.yml)
[![Test prepare_data.py with Multiple Tokenizers](https://github.com/MeryylleA/lunariscodex/actions/workflows/test_prepare_data_tokenizers.yml/badge.svg?branch=main)](https://github.com/MeryylleA/lunariscodex/actions/workflows/test_prepare_data_tokenizers.yml)
[![Test Model Module](https://github.com/MeryylleA/lunariscodex/actions/workflows/test_model_module.yml/badge.svg?branch=main)](https://github.com/MeryylleA/lunariscodex/actions/workflows/test_model_module.yml)

# Lunaris Codex

**Lunaris Codex** is a highly flexible and customizable Transformer Decoder architecture designed for code generation and language modeling. Written entirely in PyTorch, it features state-of-the-art components and a complete pipeline for data preparation, training, and inference.

Our goal is to provide a clean, understandable, and powerful codebase that serves as an excellent starting point for researchers, students, and developers interested in building, training, and experimenting with modern language models.

> **Note:** This project focuses on delivering a robust, well-tested architecture and a complete toolkit for training, data processing, and inference. While we aspire to release large-scale pretrained models, the main value here is the code, documentation, and reproducible pipeline.

---

## Key Features

*   **Advanced Decoder-only Transformer Architecture (`model.py`):**
    *   Optimized for both code and natural language tasks.
    *   Highly configurable: number of layers, hidden dimensions, attention heads, activation functions (SwiGLU, GELU), and more.
    *   Implements modern components such as Pre-Layer Normalization and LayerScale for better training stability.
    *   Features tied input embedding and output language modeling head for improved parameter efficiency.
*   **Efficient Fine-tuning with LoRA (`LoRALinear`):**
    *   Built-in support for Low-Rank Adaptation via a custom `LoRALinear` layer.
    *   Easily toggled and configured for parameter-efficient adaptation.
*   **Optimized Attention Mechanisms:**
    *   **ALiBi (Attention with Linear Biases):** Integrated for superior long-context handling.
    *   **Optional FlashAttention:** Support for the `flash-attn` library for significant speedups on compatible NVIDIA GPUs.
    *   Robust PyTorch-native manual attention fallback ensuring correct ALiBi and padding mask handling.
*   **Versatile Data Preprocessing (`prepare_data.py` v0.2.1):**
    *   Comprehensive CLI for full control over data sourcing, tokenization, and processing.
    *   Processes Hugging Face Hub datasets (e.g., [Lunaris-Data](https://huggingface.co/datasets/meryyllebr543/lunaris-data)) with custom column mapping and formatting.
    *   Supports local text files (line-by-line, chunking, glob patterns).
    *   Flexible tokenizer loading with automatic `pad_token_id` management and detailed logging.
    *   Saves to efficient memory-mapped NumPy files (`.memmap`).
    *   Features like `--overwrite_output` and explicit tokenizer/output path requirements.
*   **Comprehensive Training Pipeline (`train.py`):**
    *   Extensive CLI configurability for all training aspects.
    *   Supports training from scratch and resuming from checkpoints.
    *   AdamW optimizer, gradient clipping, LR schedulers.
    *   Automatic Mixed Precision (AMP) support (`fp16` or `bf16`) for CUDA.
    *   Robust checkpointing (model, optimizer, scheduler, config, args) with "best model" saving.
    *   Detailed metrics logging and `torch.compile` support.
*   **Enhanced Text Generation/Inference (`inference.py` v0.2.0):**
    *   Rich, colorful CLI using the `rich` library for formatted outputs, progress indicators, and model/parameter information display.
    *   Load trained models from checkpoints.
    *   Autoregressive text generation with configurable parameters (temperature, top-k, top-p, repetition penalty).
    *   Features prompt loading from files, output saving, and a `--no_color` option.
*   **C++ Utility Toolkit:**
    *   **`lunaris_data_analyzer` (v0.2.0):** Inspects `.memmap` datasets, now with configurable `--pad_id`.
    *   **`lunaris_text_cleaner` (v0.3.5):** Cleans raw text, with improved multi-stage HTML cleaning (DOCTYPE, comments, scripts, styles, tags).
*   **Scalable and Tested:**
    *   Full E2E pipeline (data prep → train → inference) demonstrated with a ~3M parameter toy model on CPU.
*   **Continuous Integration:**
    *   GitHub Actions workflows test:
        *   Core Python pipeline (`prepare_data.py`, `train.py`, `inference.py` smoke test).
        *   C++ utilities.
        *   `prepare_data.py` with multiple tokenizers.
        *   `model.py` unit tests with `pytest` and Codecov.io integration.
*   **CodeQL Security Scanning:** Integrated for static analysis.

---

## Architecture Overview

Lunaris Codex implements a standard decoder-only Transformer architecture with several modern enhancements:
*   **Transformer Blocks:** A stack of `n_layers` identical decoder blocks using Pre-LayerNorm.
*   **Self-Attention:** Multi-Head Attention with integrated ALiBi. Features optional FlashAttention (CUDA) and a PyTorch-native fallback. LoRA can be applied.
*   **FeedForward Network (FFN):** Position-wise FFN with configurable SwiGLU/GELU activation and LoRA-compatible linear layers.
*   **Stability:** LayerScale is applied to sub-layer outputs.
*   **Embeddings:** Tied token embedding and language modeling head.

> The architecture is engineered for a balance of performance, modern features, code clarity, and extensive configurability, making it an excellent foundation for diverse NLP and code-related tasks.

---

## Getting Started

This section outlines the basic steps to get Lunaris Codex up and running. Note that training effective Large Language Models typically requires substantial datasets and computational resources. Please refer to the wiki for detailed guides.

For more detailed guides and tutorials on each step, please refer to the **[Project Wiki](https://github.com/MeryylleA/lunariscodex/wiki)**.

### 1. Installation & Setup

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/MeryylleA/lunariscodex.git
    cd lunariscodex
    ```
2.  **Create a Python Virtual Environment:**  
    (Recommended: Python 3.10 or 3.11)
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # For Linux/macOS (bash/zsh)
    # For Fish shell: source .venv/bin/activate.fish
    # For Windows (cmd): .venv\Scripts\activate
    ```
3.  **Install Dependencies:**
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```
    *See `requirements.txt` for core dependencies. For optional GPU acceleration with FlashAttention, install it separately as per its documentation.*

### 2. Data Preparation (`prepare_data.py`)

Prepare your dataset for training. While the example uses a sample of [Lunaris-Data](https://huggingface.co/datasets/meryyllebr543/lunaris-data), effective pre-training often requires datasets with millions of examples.

**Example: Preparing a sample of Lunaris-Data:**
```bash
python prepare_data.py \
    --data_source_type hf_dataset \
    --dataset_name_or_path meryyllebr543/lunaris-data \
    --tokenizer_name_or_path bigcode/starcoder \
    --output_path ./processed_data/lunaris_data_sample.memmap \
    --hf_dataset_data_dir data \
    --hf_input_column input \
    --hf_target_column output \
    --hf_formatting_template "USER: {input}\nASSISTANT: {target}" \
    --max_length 1024 \
    --add_special_tokens \
    --max_examples 1000 \
    --overwrite_output
```
*For detailed usage, see the [Data Preparation Pipeline](https://github.com/MeryylleA/lunariscodex/wiki/Data-Preparation-Pipeline) page on our Wiki or run `python prepare_data.py --help`.*

### 3. Training (`train.py`)

Train your Lunaris Codex model. The example below is for a small model; refer to our [Dataset and Training Guidelines](https://github.com/MeryylleA/lunariscodex/wiki/Dataset-and-Training-Guidelines) on the Wiki for information on training larger models.

**Example: Training a small test model on CPU with LoRA:**
```bash
python train.py \
    --memmap_file_train ./processed_data/lunaris_data_sample.memmap \
    --num_sequences_train 1000 \
    --tokenizer_name_or_path bigcode/starcoder \
    --dataset_max_length 1024 \
    --model_max_seq_len 1024 \
    --d_model 256 --n_layers 4 --n_heads 4 \
    --batch_size 4 --num_epochs 1 \
    --lora_rank 8 \
    --device cpu \
    --checkpoint_dir ./checkpoints_tutorial
```
*For full training options, see the [Command-Line Arguments for Training](https://github.com/MeryylleA/lunariscodex/wiki/Command-Line-Arguments-for-Training) page on our Wiki or run `python train.py --help`.*

### 4. Running Inference (`inference.py` v0.2.0)

Generate text with your trained model using the enhanced `inference.py` script, which features a rich command-line interface.

**Example:**
```bash
python inference.py \
    --checkpoint_path ./checkpoints_tutorial/best_model.pt \
    --tokenizer_name_or_path bigcode/starcoder \
    --prompt "USER: Write a Python function that calculates factorial.\nASSISTANT:" \
    --max_new_tokens 100 \
    --temperature 0.7
```
*Run `python inference.py --help` for all options, including loading prompts from files (`--prompt_file`), saving output (`--output_file`), and disabling rich formatting (`--no_color`).*

### 5. Using C++ Utilities (Optional)
Helper tools for data analysis and text cleaning are available in `data_analyzer/` and `text_cleaner/`. Each contains its own `README.md` with compilation and usage instructions.
*   **`lunaris_text_cleaner`**: Cleans raw text files before tokenization.
*   **`lunaris_data_analyzer`**: Inspects and validates `.memmap` dataset files.

---

## Documentation & Wiki

For in-depth information, tutorials, and advanced guides, please visit the **[Lunaris Codex Project Wiki](https://github.com/MeryylleA/lunariscodex/wiki)**.  
Key pages include:
*   **[Dataset and Training Guidelines](https://github.com/MeryylleA/lunariscodex/wiki/Dataset-and-Training-Guidelines)** (Guidance on data scale, hardware, and achieving good results)
*   [Data Preparation Pipeline](https://github.com/MeryylleA/lunariscodex/wiki/Data-Preparation-Pipeline) (`prepare_data.py`)
*   [Command-Line Arguments for Training](https://github.com/MeryylleA/lunariscodex/wiki/Command-Line-Arguments-for-Training) (`train.py`)
*   [Utility: Lunaris Text Cleaner](https://github.com/MeryylleA/lunariscodex/wiki/Lunaris-Text-Cleaner)
*   [Utility: Lunaris Data Analyzer](https://github.com/MeryylleA/lunariscodex/wiki/Lunaris-Data-Analyzer)
*   [Tutorial: Using the Lunaris-Data Dataset](https://github.com/MeryylleA/lunariscodex/wiki/Using-the-Lunaris-Data-Dataset)

---

## Roadmap

Our current focus and future plans include:
*   Further enhancing `inference.py` with features like interactive mode, batch generation, improved visual output, and advanced sampling techniques.
*   Expanding documentation: advanced tutorials, API reference, and more details for the **[Dataset and Training Guidelines](https://github.com/MeryylleA/lunariscodex/wiki/Dataset-and-Training-Guidelines)**.
*   **Providing pre-tokenized versions of common public datasets or robust scripts to process them efficiently.**
*   Example configurations for training on large-scale datasets like SlimPajama or The Stack.
*   Benchmarking performance and generation quality across different model sizes and hardware.
*   Exploring further optimizations such as gradient checkpointing and quantization techniques.
*   (Ambitious) Releasing small to medium pretrained base models if resources permit.

---

## License

This project is licensed under the **MIT License**.  
Copyright (c) 2024-2025 **Francisco Antonio**

See [`LICENSE`](LICENSE) for more details.

---

## Contributing & Community

Developed by **Francisco Antonio** ([@MeryylleA](https://github.com/MeryylleA) on GitHub, [@Meryylle](https://x.com/a93918) on X/Twitter).

Lunaris Codex is an open-source endeavor. Contributions, feedback, bug reports, and feature requests are highly encouraged! Please see our [CONTRIBUTING.md](CONTRIBUTING.md) guidelines and join our community. (https://discord.gg/JNsfzEwMtC)

Let's build something amazing together!

### Special Thanks
*   To the broader open-source AI community for their invaluable tools, research, and datasets.
*   To **Gemini (Google)** for extensive pair-programming sessions, architectural discussions, debugging support, and documentation assistance throughout the development of this project.

---

## Why "Lunaris"?

> *"Because great ideas are born in silence — and shine like the moon."*
