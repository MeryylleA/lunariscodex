---

# 🌙 Lunaris Codex (Updated)

**Lunaris Codex** is a highly flexible and customizable Transformer Decoder architecture designed for code generation and language modeling. Written entirely in PyTorch, it features modern optimizations like **LoRA**, optional **FlashAttention**, **ALiBi** positional biases, and a comprehensive, configurable training and data preprocessing pipeline. This repository provides the full source code, enabling users to train their own decoder-only LLMs from scratch or fine-tune existing ones.

> ⚠️ **Note:** This release focuses on providing a robust and well-tested architecture and training system. Pretrained weights on large-scale datasets are a future goal. The current codebase is excellent for research, learning, and training smaller to medium-sized models on custom data.

---

## ✨ Key Features

- ⚙️ **Decoder-only Transformer Architecture:** Optimized for code and natural language tasks, highly configurable (layers, dimensions, heads, etc.).
- 🚀 **LoRA (Low-Rank Adaptation):** Built-in support via `LoRALinear` layers for efficient fine-tuning of projection layers. Easily toggled via configuration.
- ⚡ **Optional FlashAttention:** Integration with `flash_attn` for significant speedups and memory savings on compatible NVIDIA GPUs. Includes a fallback to a custom PyTorch attention mechanism (with ALiBi support) for CPU or other hardware.
- 🧠 **ALiBi (Attention with Linear Biases):** Implemented for improved long-context handling and extrapolation capabilities, replacing traditional absolute positional embeddings.
- 🛠️ **Modern Transformer Components:** Includes SwiGLU activations, LayerNorm, LayerScale for training stability, and tied embedding/LM head.
- 📏 **Scalable by Design:** Default configuration yields ~183M parameters, but a ~3M parameter "toy" model trained successfully on CPU, showcasing a fully functional pipeline. The architecture can be easily scaled up or down.
- 📊 **Configurable Training Pipeline (`train.py`):**
    - Comprehensive argument parsing (`argparse`) for all training aspects.
    - Supports training from scratch and resuming from checkpoints.
    - AdamW optimizer (with optional `fused` mode for CUDA).
    - Gradient clipping and Learning Rate Schedulers (e.g., `ReduceLROnPlateau`).
    - Mixed Precision (AMP) support (`fp16`/`bf16`) for CUDA.
    - Checkpointing strategy via `epoch` or `steps`, saving model, optimizer, scheduler, config, and training args using PyTorch's `.pt` format (also saves best model based on validation loss).
    - Detailed logging of metrics: loss, perplexity, and top-1 accuracy for both training and validation.
    - Optional `torch.compile` support for further optimization (primarily GPU-focused).
- 💾 **Flexible Data Preprocessing (`prepare_data.py`):**
    - `argparse` for full configuration of data source, tokenizer, and processing parameters.
    - Supports various data sources:
        - Hugging Face Hub datasets (streaming or regular).
        - Local text files (one example per line).
        - Local large text files (processed in chunks).
    - Customizable tokenizer loading (from Hugging Face Hub or local path).
    - Handles `pad_token_id` automatically.
    - Tokenizes, truncates, and pads sequences to `max_length`.
    - Saves processed data efficiently as a memory-mapped NumPy file (`.memmap`) for fast loading during training.

---

## 🧱 Architecture Overview

Lunaris Codex implements a standard decoder-only Transformer with several modern enhancements:

- **Configurable number of Transformer Blocks** (default example: 10 layers, ~183M params; test example: 1 layer, ~3M params).
- **Configurable Hidden Size, Number of Attention Heads, and FeedForward Dimension.**
- **Self-Attention:**
    - Optional `flash_attn_func` for CUDA environments.
    - Fallback to a PyTorch-based manual attention implementation that correctly incorporates **ALiBi** and padding masks when FlashAttention is unavailable or running on CPU.
    - LoRA can be applied to `qkv_proj` and `output_proj` linear layers.
- **FeedForward Network (FFN):**
    - Configurable activation function (e.g., **SwiGLU**, GELU).
    - LoRA can be applied to `fc1` and `fc2` linear layers.
- **Positional Information:** **ALiBi** is used, eliminating the need for explicit positional embeddings and allowing for better extrapolation to sequence lengths unseen during training.
- **Normalization and Stability:** Pre-LayerNorm within blocks and **LayerScale** on residual connections.
- **Output Head:** Tied embedding and language modeling head weights for improved parameter efficiency and regularization.

> The architecture is designed for a balance of performance, flexibility, and clarity, making it a good base for experimentation and learning.

---

## 🧢 Training and Data Preparation

### 1. Data Preparation (`prepare_data.py`)

This script tokenizes your raw text/code data and saves it in an efficient memory-mapped format.

**Key Arguments:**
- `--data_source_type`: `hf_dataset`, `text_file_lines`, `text_file_chunks`.
- `--dataset_name_or_path`: HF dataset name or local file path.
- `--tokenizer_name_or_path`: HF tokenizer name or local path.
- `--max_length`: Sequence length for tokenization.
- `--output_path`: Where to save the `.memmap` file.
- `--add_special_tokens`: (Flag) Whether to add BOS/EOS.
- `--max_examples`: (Optional) Limit number of examples to process.

**Example Usage (Local Text File):**
```bash
python prepare_data.py \
    --data_source_type text_file_lines \
    --dataset_name_or_path ./my_code_corpus.txt \
    --tokenizer_name_or_path bigcode/starcoder \
    --max_length 1024 \
    --output_path ./processed_data/my_corpus.memmap
```

### 2. Training (`train.py`)

This script trains the Lunaris Codex model on a preprocessed `.memmap` dataset.

**Key Arguments (Selected):**
- `--memmap_file_train`, `--num_sequences_train`: Path and size of training data.
- `--memmap_file_val`, `--num_sequences_val`: (Optional) Path and size of validation data.
- `--tokenizer_name_or_path`: Must match the one used in `prepare_data.py`.
- `--dataset_max_length`, `--model_max_seq_len`: Sequence lengths.
- `--d_model`, `--n_layers`, `--n_heads`: Model dimensions.
- `--lora_rank`: Rank for LoRA (e.g., 32). Set to `0` for full fine-tuning/training.
- `--batch_size`, `--num_epochs`, `--learning_rate`.
- `--device`: `cuda` or `cpu`.
- `--checkpoint_dir`, `--resume_from_checkpoint`.
- `--mixed_precision_dtype`: `fp16` or `bf16` (for CUDA).
- `--use_torch_compile`: (Flag) To enable `torch.compile`.

**Example Usage (Training a small LoRA model on CPU):**
```bash
python train.py \
    --memmap_file_train ./processed_data/train.memmap \
    --num_sequences_train 6000 \
    --memmap_file_val ./processed_data/val.memmap \
    --num_sequences_val 200 \
    --tokenizer_name_or_path bigcode/starcoder \
    --dataset_max_length 32 \
    --model_max_seq_len 32 \
    --d_model 128 \
    --n_layers 2 \
    --n_heads 4 \
    --batch_size 16 \
    --num_epochs 3 \
    --lora_rank 8 \
    --device cpu \
    --checkpoint_dir ./checkpoints_small_cpu
```

---

## 🧠 What's Included (Current Status)

| Component                                     | Status                                   | Notes                                                                 |
|-----------------------------------------------|------------------------------------------|-----------------------------------------------------------------------|
| **Core Model Architecture (`model.py`)**        | ✅ Released                              | Decoder-only, ALiBi, LayerScale, SwiGLU                               |
| **LoRA Integration**                          | ✅ Released                              | `LoRALinear` class, configurable rank                                 |
| **FlashAttention Support (Optional)**         | ✅ Released                              | With fallback to PyTorch manual attention for CPU/compatibility       |
| **Data Preprocessing Pipeline (`prepare_data.py`)** | ✅ Released (Major Update)             | Supports HF datasets, local text files (lines/chunks), `argparse`       |
| **Training Pipeline (`train.py`)**            | ✅ Released (Major Update)             | `argparse`, checkpointing, LoRA/Full-FT, AMP, LR scheduler, validation |
| Dataset Preprocessing (Example Scripts)       | 🟡 Partial/Example                       | `prepare_data.py` is now very capable.                                |
| Inference System (`inference.py`)             | ⚠️ Needs Update/Testing                   | Original `inference.py` likely needs updates for new model/config.  |
| Pretrained Weights                            | ❌ Not Yet Released (Future Goal)        | Focus is on providing a solid training framework.                    |

---

## ⚠️ Roadmap (Revised)

- 🧬 **Release Example Configurations:** Provide example config files or command lines for training small/medium models on common public datasets (e.g., a subset of The Stack, TinyStories).
- 🧪 **Thorough Testing & Benchmarking:** Test training on various GPUs (if/when available) and CPUs to establish baseline performance.
- 📦 **Improve `inference.py`:** Update and test the inference script to work seamlessly with the new training pipeline and saved checkpoints. Add more generation options.
- 📖 **Hugging Face Integration:**
    - Publish the model architecture code to Hugging Face (can be done without weights).
    - Create a comprehensive model card.
    - (Future) Host example small pretrained/fine-tuned models.
- 🌐 **Enhanced Documentation:** Detailed API documentation for each module and class. Tutorials on training from scratch and fine-tuning.
- ✨ **Explore Further Optimizations:** Gradient checkpointing, DeepSpeed/FSDP (for very large models), more advanced schedulers.
- 🧪 **(Ambitious Future)** Release pretrained checkpoints if large-scale training resources become available.

---

## 📦 Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/MeryylleA/lunaris-codex.git 
    cd lunaris-codex
    ```
2.  Create and activate a Python virtual environment (e.g., with Python 3.10+):
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # or .venv\Scripts\activate for Windows
    ```
3.  Install dependencies:
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

**Core Dependencies (`requirements.txt` should include):**
- `torch` (version 2.0+ recommended for `scaled_dot_product_attention` and `torch.compile`)
- `transformers` (for tokenizers)
- `datasets` (for data loading utilities)
- `numpy`
- `tqdm` (for progress bars)
- `safetensors` (though current training checkpoints use `.pt`)
- `sentencepiece` (if using SentencePiece tokenizers)
- (Optional, for GPU) `flash-attn` (install separately if you have a compatible NVIDIA GPU and CUDA setup: `pip install flash-attn --no-build-isolation`)

---

## 📝 License

This project is licensed under the **MIT License**.  
Copyright (c) 2024-2025 **Francisco Antonio** (or your preferred copyright year)

See [`LICENSE`](LICENSE) for more details.

---

## 🌟 Credits

Developed by **Francisco Antonio** ([@MeryylleA](https://github.com/MeryylleA) on GitHub, [@Meryylle](https://x.com/a93918) on X/Twitter).

This project aims to provide a clean, understandable, and powerful base for building and training custom language models. Contributions, feedback, and stars ⭐ are highly welcome!

> For collaborations, dataset requests, or research interest, please open an issue/discussion on GitHub or contact via X/Twitter.

---

## 🌌 Why "Lunaris"?

> *"Because great ideas are born in silence — and shine like the moon."* 🌙

---
