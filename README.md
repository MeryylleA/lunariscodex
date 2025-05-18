[![Lunaris Codex CI](https://github.com/MeryylleA/lunariscodex/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/MeryylleA/lunariscodex/actions/workflows/ci.yml)
[![Test prepare_data.py with Multiple Tokenizers](https://github.com/MeryylleA/lunariscodex/actions/workflows/test_prepare_data_tokenizers.yml/badge.svg?branch=main)](https://github.com/MeryylleA/lunariscodex/actions/workflows/test_prepare_data_tokenizers.yml)

# Lunaris Codex

**Lunaris Codex** is a highly flexible and customizable Transformer Decoder architecture designed for code generation and language modeling. Written entirely in PyTorch, it features state-of-the-art optimization techniques and includes modern components like LoRA, ALiBi, and a comprehensive training/data pipeline.

Our goal is to provide a clean, understandable, and powerful codebase that serves as an excellent starting point for researchers, students, and developers interested in building, training, and experimenting with advanced decoder-only models.

> **Note:** This project focuses on delivering a robust, well-tested architecture and a complete toolkit for training, data processing, and inference. While we aspire to release large-scale pretrained weights, our primary focus is on code quality and pipeline completeness.

---

## Key Features

* **Advanced Decoder-only Transformer Architecture (`model.py`):**
    * Optimized for both code and natural language tasks.
    * Highly configurable: number of layers, hidden dimensions, attention heads, activation functions (SwiGLU, GELU), and more.
    * Implements modern components such as Pre-Layer Normalization and LayerScale for better training stability.
    * Tied input embedding and output language modeling head for improved parameter efficiency.
* **Efficient Fine-tuning with LoRA (`LoRALinear`):**
    * Built-in support for Low-Rank Adaptation via a custom `LoRALinear` layer, applicable to key projection layers.
    * Easily toggled and configured through training arguments for parameter-efficient adaptation.
* **Optimized Attention Mechanisms:**
    * **ALiBi (Attention with Linear Biases):** Integrated for superior long-context handling and extrapolation, replacing traditional absolute positional embeddings.
    * **Optional FlashAttention:** Support for the `flash-attn` library for significant speedups and memory savings on compatible NVIDIA GPUs.
    * Includes a robust PyTorch-native manual attention implementation as a fallback. This ensures ALiBi and padding masks are correctly handled on CPU or other hardware.
* **Versatile Data Preprocessing (`prepare_data.py`):**
    * Comprehensive command-line interface (`argparse`) for full control over data sourcing, tokenization, and processing.
    * Seamlessly processes structured datasets from Hugging Face Hub (e.g., the companion [Lunaris-Data dataset](https://huggingface.co/datasets/meryyllebr543/lunaris-data)) with custom input/target columns and formatting.
    * Supports diverse local text file formats: line-by-line processing, chunking of large files, and glob pattern matching.
    * Flexible tokenizer loading from Hugging Face Hub by name or local path.
    * Automatic, configurable `pad_token_id` management with detailed logging.
    * Efficiently saves processed data as memory-mapped NumPy files (`.memmap`) for rapid loading during training.
* **Comprehensive Training Pipeline (`train.py`):**
    * Extensive command-line configurability for all training aspects.
    * Supports training from scratch and resuming from saved checkpoints.
    * Features AdamW optimizer, gradient clipping, and learning rate schedulers.
    * Automatic Mixed Precision (AMP) support (`fp16` or `bf16`) for CUDA.
    * Robust checkpointing: saves model state, optimizer, scheduler, model configuration, and training arguments. Includes "best model" saving based on validation loss.
    * Detailed metrics logging (loss, perplexity, top-1 accuracy) for training and validation.
    * Optional `torch.compile` support for further optimization.
* **Text Generation/Inference (`inference.py`):**
    * Provides a command-line script to load trained Lunaris Codex models from checkpoints.
    * Generates text autoregressively based on a user-provided prompt.
    * Supports configurable generation parameters (temperature, top-k, top-p, repetition penalty).
    * Completes the end-to-end workflow: data preparation → training → inference.
* **C++ Utility Toolkit (in `data_analyzer/` and `text_cleaner/`):**
    * **`lunaris_data_analyzer`:** A C++ tool for inspecting, validating, and gathering statistics from `.memmap` datasets, with configurable padding ID.
    * **`lunaris_text_cleaner`:** A C++ utility for performing various cleaning operations (including improved HTML cleaning) on raw text files.
* **Scalable and Tested:**
    * Successfully tested with a ~3M parameter "toy" model trained end-to-end on CPU, demonstrating full pipeline functionality from data preparation and cleaning through training to text generation.
* **Continuous Integration:**
    * Includes GitHub Actions CI workflows to test:
        * The core Python pipeline (`prepare_data.py`, `train.py`).
        * The C++ utilities (`lunaris_text_cleaner`, `lunaris_data_analyzer`).
        * `prepare_data.py` compatibility with multiple tokenizers in a separate workflow.
    * Ensures ongoing stability and catches regressions.
* **CodeQL Security Scanning:**
    * Integrated CodeQL for static analysis.

---

## Architecture Overview

Lunaris Codex implements a standard decoder-only Transformer architecture with several modern enhancements:
* **Transformer Blocks:** A stack of `n_layers` identical decoder blocks using Pre-LayerNorm.
* **Self-Attention:** Multi-Head Attention with integrated ALiBi. Features optional FlashAttention (CUDA) and a PyTorch-native fallback. LoRA can be applied.
* **FeedForward Network (FFN):** Position-wise FFN with configurable SwiGLU/GELU activation and LoRA-compatible linear layers.
* **Stability:** LayerScale is applied to sub-layer outputs.
* **Embeddings:** Tied token embedding and language modeling head.

> The architecture is engineered for a balance of performance, modern features, code clarity, and extensive configurability, making it an excellent foundation for diverse NLP and code-related tasks.

---

## Getting Started

This section outlines the basic steps to get Lunaris Codex up and running. For more detailed guides, tutorials, and explanations, please refer to the **[Project Wiki](https://github.com/MeryylleA/lunariscodex/wiki)**.

### 1. Installation & Setup

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/MeryylleA/lunariscodex.git
    cd lunariscodex
    ```
2. **Create a Python Virtual Environment:**  
   (Recommended: Python 3.10 or 3.11)
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # For Linux/macOS (bash/zsh)
    # For Fish shell: source .venv/bin/activate.fish
    # For Windows (cmd): .venv\Scripts\activate
    ```
3. **Install Dependencies:**
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
*For detailed usage and more examples, see the [Data Preparation Pipeline](https://github.com/MeryylleA/lunariscodex/wiki/Data-Preparation-Pipeline) page on our Wiki or run `python prepare_data.py --help`.*

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
*For a full list of training options and explanations, see the [Command-Line Arguments for Training](https://github.com/MeryylleA/lunariscodex/wiki/Command-Line-Arguments-for-Training) page on our Wiki or run `python train.py --help`.*

### 4. Running Inference (`inference.py`)

After training your model and saving a checkpoint, you can use `inference.py` to generate text.

**Example: Generating text with a trained model:**
```bash
python inference.py \
    --checkpoint_path ./checkpoints_tutorial/best_model.pt \
    --tokenizer_name_or_path bigcode/starcoder \
    --prompt "USER: Write a Python function that calculates factorial.\nASSISTANT:" \
    --max_new_tokens 100 \
    --temperature 0.7
```
*Replace paths and parameters as needed. Run `python inference.py --help` to see all options.*

### 5. Using C++ Utilities (Optional)
Helper tools for data analysis and text cleaning are available in `data_analyzer/` and `text_cleaner/`. Each contains its own `README.md` with compilation and usage instructions.
* **`lunaris_text_cleaner`:** Cleans raw text files before tokenization.
* **`lunaris_data_analyzer`:** Inspects and validates `.memmap` dataset files.

---

## Documentation & Wiki

For in-depth information, tutorials, and advanced guides, please visit the **[Lunaris Codex Project Wiki](https://github.com/MeryylleA/lunariscodex/wiki)**.

---

## Roadmap

Our current focus and future plans include:
* Further enhancing `inference.py` with features like interactive mode, batch generation, improved visual output, and advanced sampling techniques.
* Expanding documentation with more advanced tutorials and API reference details.
* Providing example configurations and scripts for training on common public datasets (e.g., SlimPajama, The Stack).
* Benchmarking performance and generation quality across different model sizes and hardware.
* Exploring further optimizations such as gradient checkpointing and advanced quantization techniques.
* (Ambitious) Releasing small to medium pretrained base models if resources permit.

---

## License

This project is licensed under the **MIT License**.  
Copyright (c) 2024-2025 **Francisco Antonio**

See [`LICENSE`](LICENSE) for more details.

---

## Contributing & Community

Developed by **Francisco Antonio** ([@MeryylleA](https://github.com/MeryylleA) on GitHub, [@Meryylle](https://x.com/a93918) on X/Twitter).

Lunaris Codex is an open-source endeavor. Contributions, feedback, bug reports, and feature requests are highly encouraged! Please see our [CONTRIBUTING.md](CONTRIBUTING.md) guidelines and join our community discussions.

Let's build something amazing together!

### Special Thanks
* To the broader open-source AI community for their invaluable tools, research, and datasets.
* To **Gemini (Google)** for extensive pair-programming sessions, architectural discussions, debugging support, and documentation assistance throughout the development of this project.

---

## Why "Lunaris"?

> *"Because great ideas are born in silence – and shine like the moon."*
