[![Lunaris Codex CI](https://github.com/MeryylleA/lunariscodex/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/MeryylleA/lunariscodex/actions/workflows/ci.yml)

# Lunaris Codex

**Lunaris Codex** is a highly flexible and customizable Transformer Decoder architecture designed for code generation and language modeling. Written entirely in PyTorch, it features modern optimizations like LoRA (Low-Rank Adaptation), optional FlashAttention, ALiBi (Attention with Linear Biases) positional biasing, and a comprehensive, configurable training and data preprocessing pipeline. This repository provides the full source code, enabling users to train their own decoder-only Large Language Models (LLMs) from scratch or efficiently fine-tune existing ones.

Our goal is to provide a clean, understandable, and powerful codebase that serves as an excellent starting point for researchers, students, and developers interested in building, training, and experimenting with state-of-the-art language models.

> **Note:** This project focuses on providing a robust, well-tested architecture and a complete training/data-processing toolkit. While the ambition for large-scale pretrained weights exists (targeting high-end GPU hardware like NVIDIA H100/GH200), the current release empowers users to train models of various sizes on custom datasets today. It is an ideal platform for learning, research, and building specialized models.

---

## Key Features

*   **Advanced Decoder-only Transformer Architecture (`model.py`):**
    *   Optimized for code and natural language tasks.
    *   Highly configurable: number of layers, hidden dimensions, attention heads, activation functions (SwiGLU, GELU), etc.
    *   Implements modern components such as Pre-Layer Normalization and LayerScale for enhanced training stability.
    *   Features tied input embedding and output language modeling head for improved parameter efficiency.
*   **Efficient Fine-tuning with LoRA (`LoRALinear`):**
    *   Built-in support for Low-Rank Adaptation via a custom `LoRALinear` layer, applicable to key projection layers.
    *   Easily toggled and configured through training arguments for parameter-efficient adaptation.
*   **Optimized Attention Mechanisms:**
    *   **ALiBi (Attention with Linear Biases):** Integrated for superior long-context handling and extrapolation, replacing traditional absolute positional embeddings.
    *   **Optional FlashAttention:** Support for the `flash-attn` library for significant speedups and memory savings on compatible NVIDIA GPUs.
    *   Includes a robust PyTorch-native manual attention implementation as a fallback, ensuring ALiBi and padding masks are correctly handled on CPU or other hardware.
*   **Versatile Data Preprocessing (`prepare_data.py`):**
    *   Comprehensive command-line interface (`argparse`) for full control over data sourcing, tokenization, and processing.
    *   Seamlessly processes structured datasets from Hugging Face Hub (e.g., the companion [Lunaris-Data dataset](https://huggingface.co/datasets/meryyllebr543/lunaris-data)) with custom input/target column mapping and formatting.
    *   Supports diverse local text file formats: line-by-line processing, chunking of large files, and glob pattern matching for multiple files.
    *   Flexible tokenizer loading from Hugging Face Hub by name or local path (including SentencePiece `.model` files and standard `tokenizer.json`).
    *   Automatic and configurable `pad_token_id` management.
    *   Efficiently saves processed data as memory-mapped NumPy files (`.memmap`) for rapid loading during training.
*   **Comprehensive Training Pipeline (`train.py`):**
    *   Extensive command-line configurability for all training aspects, including hyperparameters, optimization, and operational settings.
    *   Supports training from scratch and resuming from saved checkpoints.
    *   Features AdamW optimizer (with optional `fused` mode for CUDA), gradient clipping, and Learning Rate Schedulers (e.g., `ReduceLROnPlateau`).
    *   Automatic Mixed Precision (AMP) support (`fp16` or `bf16`) for CUDA devices.
    *   Robust checkpointing: saves and loads model state, optimizer, scheduler, configuration, and training arguments in PyTorch's `.pt` format. Includes "best model" saving based on validation loss.
    *   Detailed metrics logging (loss, perplexity, top-1 accuracy) for training and validation, with `tqdm` progress bars.
    *   Optional `torch.compile` support for further optimization (most effective on newer PyTorch versions and GPUs).
*   **C++ Utility Toolkit (in `data_analyzer/` and `text_cleaner/`):**
    *   **`lunaris_data_analyzer`**: A C++ tool for inspecting, validating, and gathering statistics from `.memmap` datasets, using `mmap` on Linux for efficiency.
    *   **`lunaris_text_cleaner`**: A C++ utility for performing various cleaning and normalization operations on raw text files (single or batch directory processing) before tokenization.
*   **Scalable and Tested:**
    *   Successfully tested with a ~3M parameter "toy" model trained end-to-end on CPU, demonstrating full pipeline functionality. The architecture is designed to be easily scaled.
*   **Continuous Integration:**
    *   Includes a GitHub Actions CI workflow to test the core data preparation, C++ utilities, and training pipeline, ensuring ongoing stability.
*   **CodeQL Security Scanning:**
    *   Integrated CodeQL for static analysis to help identify potential security vulnerabilities and code quality issues.

---

## Architecture Overview

Lunaris Codex implements a standard decoder-only Transformer architecture with several modern enhancements:
*   **Transformer Blocks:** A stack of `n_layers` identical decoder blocks using Pre-LayerNorm.
*   **Self-Attention:** Multi-Head Attention with integrated ALiBi. Features optional FlashAttention (CUDA) and a PyTorch-native fallback that correctly handles ALiBi and padding. LoRA can be applied to QKV and output projections.
*   **FeedForward Network (FFN):** Position-wise FFN with configurable SwiGLU/GELU activation and LoRA-compatible linear layers.
*   **Stability:** LayerScale is applied to sub-layer outputs.
*   **Embeddings:** Tied token embedding and language modeling head.

> The architecture is engineered for a balance of performance, modern features, code clarity, and extensive configurability, making it an excellent foundation for diverse NLP and code-related tasks.

---

## Getting Started

This section outlines the basic steps to get Lunaris Codex up and running. For more detailed guides, tutorials, and explanations, please refer to the **[Project Wiki](https://github.com/MeryylleA/lunariscodex/wiki)**.

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
    # For fish shell: source .venv/bin/activate.fish
    # For Windows: .venv\Scripts\activate
    ```
3.  **Install Dependencies:**
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```
    *See `requirements.txt` for core dependencies. For optional GPU acceleration with FlashAttention, install it separately as per its documentation.*

### 2. Data Preparation (`prepare_data.py`)

Prepare your dataset for training. You can use the [Lunaris-Data dataset](https://huggingface.co/datasets/meryyllebr543/lunaris-data), other Hugging Face Hub datasets, or your own local text files.

**Example: Preparing a sample of Lunaris-Data:**
```bash
python prepare_data.py \
    --data_source_type hf_dataset \
    --dataset_name_or_path meryyllebr543/lunaris-data \
    --hf_dataset_data_dir data \
    --hf_input_column input \
    --hf_target_column output \
    --hf_formatting_template "USER: {input} ASSISTANT: {target}" \
    --tokenizer_name_or_path bigcode/starcoder \
    --max_length 1024 \
    --add_special_tokens \
    --output_path ./processed_data/lunaris_data_sample.memmap \
    --max_examples 1000 
```
*For detailed usage and more examples, see the [[Data Preparation Pipeline]] page on our Wiki or run `python prepare_data.py --help`.*

### 3. Training (`train.py`)

Train your Lunaris Codex model using the prepared `.memmap` dataset.

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
*For a full list of training options and explanations, see the [[Command-Line Arguments for Training]] page on our Wiki or run `python train.py --help`.*

### 4. Using C++ Utilities (Optional)
Helper tools for data analysis and text cleaning are available in `data_analyzer/` and `text_cleaner/`. Each contains its own `README.md` with compilation and usage instructions.
*   **`lunaris_text_cleaner`**: Cleans raw text files before tokenization.
*   **`lunaris_data_analyzer`**: Inspects and validates `.memmap` dataset files.

---

## Documentation & Wiki

For in-depth information, tutorials, and advanced guides, please visit the **[Lunaris Codex Project Wiki](https://github.com/MeryylleA/lunariscodex/wiki)**.
---

## Roadmap

Our current focus and future plans include:
*   Refining and thoroughly testing the `inference.py` script.
*   Expanding documentation with more tutorials and API details.
*   Providing example configurations for training on common public datasets.
*   Benchmarking performance across different model sizes and hardware.
*   Exploring further optimizations like gradient checkpointing and deeper Hugging Face Hub integration.
*   (Ambitious) Releasing small to medium pretrained base models if resources permit.

---

## License

This project is licensed under the **MIT License**.
Copyright (c) 2024-2025 **Francisco Antonio**

See [`LICENSE`](LICENSE) for more details.

---

## Contributing & Community

Developed by **Francisco Antonio** ([@MeryylleA](https://github.com/MeryylleA) on GitHub, [@Meryylle](https://x.com/a93918) on X/Twitter).

Lunaris Codex is an open-source endeavor. Contributions, feedback, bug reports, and feature requests are highly encouraged! Please see our [CONTRIBUTING.md](CONTRIBUTING.md) guidelines and join our [GitHub Discussions](https://github.com/MeryylleA/lunariscodex/discussions).

Special thanks to the open-source AI community and to **Gemini** for extensive pair-programming, architectural discussions, and debugging support.

> Let's build something amazing together!

---

## Why "Lunaris"?

> *"Because great ideas are born in silence — and shine like the moon."*
