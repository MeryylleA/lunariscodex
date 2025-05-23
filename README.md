[![Lunaris Codex CI](https://github.com/MeryylleA/lunariscodex/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/MeryylleA/lunariscodex/actions/workflows/ci.yml)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/MeryylleA/lunariscodex)
[![codecov](https://codecov.io/gh/MeryylleA/lunariscodex/branch/main/graph/badge.svg?token=6FHOG5S0HQ)](https://codecov.io/gh/MeryylleA/lunariscodex)

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
*   **Versatile Data Preprocessing (`prepare_data.py` v0.2.2):** <!-- Updated version -->
    *   Comprehensive CLI for full control over data sourcing, tokenization, and processing.
    *   Processes Hugging Face Hub datasets (e.g., [Lunaris-Data](https://huggingface.co/datasets/meryyllebr543/lunaris-data)) with custom column mapping and formatting.
    *   Supports local text files (line-by-line, chunking, glob patterns).
    *   Flexible tokenizer loading with automatic `pad_token_id` management and **enhanced, detailed logging** of tokenizer properties. <!-- Added detail -->
    *   Saves to efficient memory-mapped NumPy files (`.memmap`).
    *   Features like `--overwrite_output` and explicit tokenizer/output path requirements.
*   **Comprehensive Training Pipeline (`train.py`):**
    *   Extensive CLI configurability for all training aspects.
    *   Supports training from scratch and resuming from checkpoints, with **improved logging for checkpoint state restoration**. <!-- Added detail -->
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
    *   **`BpeProcessor` (v0.2.0 - Evolved!):** Formerly `bpe_trainer`. Now trains BPE models from a corpus *and* tokenizes text using a trained model. Enables creation and usage of fully custom tokenizers. <!-- MODIFIED -->
    *   **`lunaris_data_analyzer` (v0.2.0):** Inspects `.memmap` datasets, now with configurable `--pad_id`.
    *   **`lunaris_text_cleaner` (v0.3.5):** Cleans raw text, with improved multi-stage HTML cleaning.
*   **Scalable and Tested:**
    *   Full E2E pipeline (data prep → train → inference) demonstrated with toy models on CPU, including overfitting/fine-tuning tests and successful training runs on GPU (CUDA with AMP). <!-- MODIFIED -->
*   **Continuous Integration (CI) & Automation:**
    *   A comprehensive GitHub Actions workflow (`ci.yml`) tests:
        *   Core Python pipeline (`prepare_data.py`, `train.py`, `inference.py` smoke test).
        *   Compilation and functionality of C++ utilities (including `BpeProcessor` train and tokenize actions). <!-- MODIFIED -->
        *   `model.py` unit tests using `pytest`, with coverage reports sent to Codecov.io.
    *   Automated Pull Request management for the primary developer, including auto-merge on CI success.
    *   Dependabot integration for automated dependency updates.
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

## Detailed Model Architecture

The Lunaris model is a sophisticated Transformer-based decoder architecture designed for advanced language understanding and generation tasks. Below is a breakdown of its core components and features:

### Core Components

*   **`LunarisCodexConfig`**: This class serves as the central configuration hub for the model. It defines crucial hyperparameters such as vocabulary size (`vocab_size`), model dimensionality (`d_model`), number of transformer layers (`n_layers`), number of attention heads (`n_heads`), maximum sequence length (`max_seq_len`), dropout rates, activation functions (e.g., SwiGLU), and LoRA rank (`lora_rank`). It also incorporates adaptive settings, such as adjusting dropout for smaller model variants to optimize performance.

*   **`LoRALinear`**: Lunaris integrates Low-Rank Adaptation (LoRA) through this class. `LoRALinear` layers are used in place of standard linear layers within the attention and feed-forward network components. This allows for efficient fine-tuning by significantly reducing the number of trainable parameters, as only the LoRA adapter weights are typically updated.

*   **`LunarisMind`**: This is the main class that encapsulates the entire Lunaris model. It orchestrates the various parts of the architecture, including:
    *   Token Embeddings: Input tokens are converted into dense vector representations.
    *   A stack of `TransformerDecoderBlock` layers.
    *   A final layer normalization step.
    *   A tied language modeling head, where the output linear layer shares weights with the token embedding layer. This is a common technique to reduce model size and improve performance.

*   **`TransformerDecoderBlock`**: Each decoder block is a fundamental building unit of the Lunaris model. It consists of:
    *   Layer Normalization: Applied before the self-attention and feed-forward sub-layers.
    *   `SelfAttention`: The attention mechanism.
    *   `FeedForward`: A position-wise feed-forward network.
    *   Dropout: Applied for regularization.
    *   Optional LayerScale: For larger model configurations, LayerScale is used to stabilize training by adaptively scaling the outputs of residual connections.

*   **`SelfAttention`**: This module implements multi-head self-attention.
    *   It uses `LoRALinear` for its query, key, value, and output projection layers.
    *   A key feature is its **custom ALiBi (Attention with Linear Biases) implementation**. ALiBi provides an alternative to traditional positional embeddings by directly biasing attention scores based on token distance. The custom implementation in Lunaris ensures correct ALiBi application even with a variable number of attention heads.
    *   Notably, **Flash Attention is explicitly disabled** if available. This decision is due to incompatibilities between standard Flash Attention implementations and the precise biasing required by the custom ALiBi mechanism. The model defaults to a PyTorch standard attention implementation to ensure the integrity of ALiBi.

*   **`FeedForward`**: This module is a standard position-wise feed-forward network, typically consisting of two linear transformations with an activation function in between.
    *   It uses `LoRALinear` for its linear layers.
    *   The activation function is configurable via `LunarisCodexConfig`, with options like SwiGLU or GeLU.

### Special Features

*   **ALiBi (Attention with Linear Biases)**: Lunaris employs a custom ALiBi implementation. Instead of adding explicit positional embeddings to the input tokens, ALiBi modifies attention scores based on the distance between query and key tokens. This method has been shown to improve extrapolation to longer sequence lengths. The implementation is carefully designed to work correctly across different numbers of attention heads.

*   **LoRA (Low-Rank Adaptation)**: LoRA is used in linear layers within the self-attention and feed-forward modules. This allows for more parameter-efficient fine-tuning. During fine-tuning, the original pre-trained model weights are kept frozen, and only the smaller LoRA adapter matrices are updated.

*   **Tied Embeddings**: The weights of the token embedding layer and the final language modeling head are shared. This reduces the total number of parameters in the model and can lead to improved performance.

*   **Configurable Architecture**: Many aspects of the model, such as its size (`d_model`, `n_layers`, `n_heads`), dropout rates, LoRA rank (`lora_rank`), and activation functions, are configurable through the `LunarisCodexConfig` class, allowing users to tailor the model to specific requirements.

This architecture aims to provide a robust and adaptable foundation for various natural language processing tasks, with a focus on efficient training and fine-tuning through LoRA and effective positional encoding via ALiBi.

---

## Getting Started

This section outlines the basic steps to get Lunaris Codex up and running. Note that training effective Large Language Models typically requires substantial datasets and computational resources.

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
Helper tools for data analysis, text cleaning, and custom BPE tokenization are available. Each utility is located in its own directory (e.g., `bpe_trainer/` for `BpeProcessor`, `text_cleaner/`, `data_analyzer/`) and includes a `README.md` with specific compilation and usage instructions. They can also be compiled using the main `Makefile` at the root of the repository (e.g., `make bpe_processor`).

*   **`BpeProcessor`**: (Located in `bpe_trainer/`) Trains BPE models from a corpus and tokenizes text using these custom models. <!-- MODIFIED -->
*   **`lunaris_text_cleaner`**: Cleans raw text files before tokenization. (Located in `text_cleaner/`)
*   **`lunaris_data_analyzer`**: Inspects and validates `.memmap` dataset files. (Located in `data_analyzer/`)

---

## Documentation & Wiki

For in-depth information, tutorials, and advanced guides, please visit the **[Lunaris Codex Project Wiki](https://github.com/MeryylleA/lunariscodex/wiki)**.  
Key pages include:
*   **[Home](https://github.com/MeryylleA/lunariscodex/wiki/Home)** (Start here for an overview of the Wiki content)
*   **[Dataset and Training Guidelines](https://github.com/MeryylleA/lunariscodex/wiki/Dataset-and-Training-Guidelines)**
*   [Data Preparation Pipeline](https://github.com/MeryylleA/lunariscodex/wiki/Data-Preparation-Pipeline) (`prepare_data.py`)
*   [Command-Line Arguments for Training](https://github.com/MeryylleA/lunariscodex/wiki/Command-Line-Arguments-for-Training) (`train.py`)
*   [Utility: BPE Processor](https://github.com/MeryylleA/lunariscodex/wiki/Utility:-BPE-Processor) **(Updated!)** <!-- MODIFIED & ASSUMING WIKI PAGE NAME CHANGE -->
*   [Utility: Lunaris Text Cleaner](https://github.com/MeryylleA/lunariscodex/wiki/Utility:-Lunaris-Text-Cleaner)
*   [Utility: Lunaris Data Analyzer](https://github.com/MeryylleA/lunariscodex/wiki/Utility:-Lunaris-Data-Analyzer)
*   [Tutorial: Training Your First Model](https://github.com/MeryylleA/lunariscodex/wiki/Training-Your-First-Model)
*   [Tutorial: Using the Lunaris-Data Dataset](https://github.com/MeryylleA/lunariscodex/wiki/Using-the-Lunaris-Data-Dataset)

---

## Roadmap

Our current focus and future plans include:
*   **Finalize and fully integrate the `BpeProcessor` (custom BPE tokenizer) with `prepare_data.py`, allowing native use of custom-trained tokenizers.** <!-- REPHRASED AND HIGHLIGHTED -->
*   **Implement detokenization functionality in `BpeProcessor`.** <!-- NEW/SPECIFIC -->
*   Enhancing `lunaris_data_analyzer` with token decoding capabilities.
*   Increasing unit test coverage for `model.py`.
*   Developing a dedicated evaluation script (`evaluate.py`).
*   Further enhancing `inference.py` (interactive mode, batch generation, etc.).
*   Expanding documentation: advanced tutorials, API reference.
*   (Ambitious) Releasing small to medium pretrained base models if resources permit.

---

## License

This project is licensed under the **MIT License**.  
Copyright (c) 2024-2025 **Francisco Antonio**

See [`LICENSE`](LICENSE) for more details.

---

## Contributing & Community

Developed by **Francisco Antonio** ([@MeryylleA](https://github.com/MeryylleA) on GitHub, [@Meryylle](https://x.com/a93918) on X/Twitter).

Lunaris Codex is an open-source endeavor. Contributions, feedback, bug reports, and feature requests are highly encouraged! Please see our [CONTRIBUTING.md](CONTRIBUTING.md) guidelines and join our community. (Discord: [Moon Cloud Services](https://discord.gg/JNsfzEwMtC) - Lunaris Codex has a dedicated section here)

Let's build something amazing together!

### Special Thanks
*   To the broader open-source AI community for their invaluable tools, research, and datasets.
*   To **Gemini (Google)** for extensive pair-programming sessions, architectural discussions, debugging support, and documentation assistance throughout the development of this project.

---

## Why "Lunaris"?

> *"Because great ideas are born in silence — and shine like the moon."*
