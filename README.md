[![Lunaris Codex CI](https://github.com/MeryylleA/lunariscodex/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/MeryylleA/lunariscodex/actions/workflows/ci.yml)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/MeryylleA/lunariscodex)
[![codecov](https://codecov.io/gh/MeryylleA/lunariscodex/branch/main/graph/badge.svg?token=6FHOG5S0HQ)](https://codecov.io/gh/MeryylleA/lunariscodex)
![CodeRabbit Pull Request Reviews](https://img.shields.io/coderabbit/prs/github/MeryylleA/lunariscodex?utm_source=oss&utm_medium=github&utm_campaign=MeryylleA%2Flunariscodex&labelColor=171717&color=FF570A&link=https%3A%2F%2Fcoderabbit.ai&label=CodeRabbit+Reviews)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Lunaris Codex

**Lunaris Codex** is a highly flexible and customizable Transformer Decoder architecture designed for code generation and language modeling. Written entirely in PyTorch, it features state-of-the-art components and a complete pipeline for data preparation, training, and inference.

Our philosophy goes beyond fine-tuning existing models—we provide a robust foundation for building custom AI architectures from the ground up. Whether you're a researcher exploring novel architectures, a student learning about transformers, or a developer creating specialized language models, Lunaris Codex offers the tools and flexibility you need.

> **Note:** This project focuses on delivering a robust, well-tested architecture and a complete toolkit for training, data processing, and inference. While we aspire to release large-scale pretrained models, the main value here is the code, documentation, and reproducible pipeline.

---

## 🚀 What's New: Multi-GPU Training with DistributedDataParallel

**Major Update:** `train.py` now supports **DistributedDataParallel (DDP)** for efficient multi-GPU training! This groundbreaking enhancement enables:

- **Significantly faster pre-training** with automatic workload distribution across multiple GPUs
- **Seamless scaling** to larger datasets and model architectures
- **Production-ready distributed training** using PyTorch's battle-tested DDP framework
- **Successfully tested** on 2x RTX 4090 GPUs with substantial training speedups

Key DDP features implemented:
- **Automatic Process Management**: Handles distributed process group initialization and cleanup
- **Smart Device Allocation**: Each GPU process is automatically pinned to its designated device
- **Distributed Data Loading**: Uses `DistributedSampler` to ensure each GPU processes unique data shards
- **Synchronized Training**: Gradients are automatically synchronized across all GPUs
- **Robust Checkpointing**: DDP-aware model saving and loading with rank-based coordination
- **Aggregated Metrics**: Loss and performance metrics are properly averaged across all processes

---

## Key Features

*   **Advanced Decoder-only Transformer Architecture (`model.py`):**
    *   Optimized for both code and natural language tasks.
    *   Highly configurable: number of layers, hidden dimensions, attention heads, activation functions (SwiGLU, GELU), and more.
    *   Implements modern components such as Pre-Layer Normalization and LayerScale for better training stability.
    *   Features tied input embedding and output language modeling head for improved parameter efficiency.
    *   Implemented KV Caching for significantly optimized token generation during inference.

*   **Distributed Multi-GPU Training (`train.py`):**
    *   **DistributedDataParallel (DDP) support** for efficient multi-GPU training with linear scaling
    *   Comprehensive CLI configurability for all training aspects
    *   Supports training from scratch and resuming from checkpoints with enhanced DDP awareness
    *   AdamW optimizer with gradient clipping and distributed synchronization
    *   **Flexible Learning Rate Schedulers:** Supports `ReduceLROnPlateau` and `CosineAnnealingWarmRestarts`
    *   **Gradient Accumulation:** Compatible with DDP for simulating larger effective batch sizes
    *   Automatic Mixed Precision (AMP) support (`fp16` or `bf16`) optimized for multi-GPU setups
    *   **`torch.compile` support** for additional performance gains on compatible hardware

*   **Efficient Fine-tuning with LoRA (`LoRALinear`):**
    *   Built-in support for Low-Rank Adaptation via a custom `LoRALinear` layer.
    *   Easily toggled and configured for parameter-efficient adaptation.
    *   Fully compatible with DDP training for distributed fine-tuning.

*   **Optimized Attention Mechanisms:**
    *   **ALiBi (Attention with Linear Biases):** Integrated for superior long-context handling.
    *   **Optional FlashAttention:** Support for the `flash-attn` library for significant speedups on compatible NVIDIA GPUs.
    *   Robust PyTorch-native manual attention fallback ensuring correct ALiBi and padding mask handling.

*   **Versatile Data Preprocessing (`prepare_data.py` v0.3.0):**
    *   Comprehensive CLI for full control over data sourcing, tokenization, and processing.
    *   Processes Hugging Face Hub datasets with custom column mapping and formatting.
    *   Supports local text files (line-by-line, chunking, glob patterns).
    *   Flexible tokenizer loading with automatic `pad_token_id` management.
    *   Saves to efficient memory-mapped NumPy files (`.memmap`) optimized for distributed loading.

*   **Enhanced Text Generation/Inference (`inference.py` v0.3.9):**
    *   Rich, colorful CLI using the `rich` library for formatted outputs and progress indicators.
    *   Autoregressive text generation with configurable parameters (temperature, top-k, top-p, repetition penalty).
    *   Interactive mode (`--interactive`) and streaming generation (`--stream`).
    *   Integrated KV Caching for substantially faster inference.

*   **C++ Utility Toolkit:**
    *   **`BpeProcessor` (v0.2.0):** Trains BPE models from a corpus and tokenizes text using trained models.
    *   **`lunaris_data_analyzer` (v0.2.0):** Inspects `.memmap` datasets with configurable parameters.
    *   **`lunaris_text_cleaner` (v0.3.5):** Advanced text cleaning with multi-stage HTML processing.

*   **Production-Ready & Tested:**
    *   Full E2E pipeline demonstrated with successful multi-GPU training runs on high-end NVIDIA GPUs
    *   Comprehensive CI/CD with automated testing and dependency management
    *   Extensive documentation and community support

---

## Architecture Overview

Lunaris Codex implements a modern decoder-only Transformer architecture with cutting-edge enhancements:

*   **Transformer Blocks:** A stack of `n_layers` identical decoder blocks using Pre-LayerNorm
*   **Self-Attention:** Multi-Head Attention with integrated ALiBi and optional FlashAttention
*   **Feed-Forward Network (FFN):** Position-wise FFN with configurable SwiGLU/GELU activation
*   **Advanced Features:** LayerScale for stability, tied embeddings for efficiency, KV Caching for fast inference
*   **LoRA Integration:** Built-in support for parameter-efficient fine-tuning

The architecture balances performance, modern features, and code clarity, making it an excellent foundation for diverse NLP and code-related tasks.

---

## GPU Compatibility & Multi-GPU Training

Lunaris Codex is optimized for NVIDIA GPUs and supports both single-GPU and multi-GPU distributed training:

| GPU Architecture | Tested Models | DDP Status | Notes |
| :--------------- | :------------ | :--------- | :---- |
| **NVIDIA Blackwell** | RTX 5090 | ✅ **Verified** | Excellent DDP performance with `bf16` and `torch.compile` |
| **NVIDIA Ada Lovelace** | RTX 4090 (2x setup) | ✅ **Verified** | Successfully tested DDP training with significant speedups |
|                        | RTX 4070 Ti SUPER | ✅ **Verified** | Single-GPU verified, DDP expected to work |
| **Other Modern GPUs** | RTX 40/30-series | ⚙️ **Expected Compatible** | Should work with DDP based on PyTorch support |

**DDP Requirements:**
- NVIDIA GPUs with CUDA support
- PyTorch with distributed capabilities
- NCCL backend (automatically handled)
- Sufficient GPU memory for model replication across devices

---

## Getting Started

### 1. Installation & Setup

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/MeryylleA/lunariscodex.git
    cd lunariscodex
    ```

2.  **Create a Python Virtual Environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # Linux/macOS
    # Windows: .venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

### 2. Data Preparation

Prepare your dataset using `prepare_data.py`. For production training, use datasets with millions of examples:

```bash
python prepare_data.py \
    --data_source_type hf_dataset \
    --dataset_name_or_path meryyllebr543/lunaris-data \
    --tokenizer_name_or_path bigcode/starcoder \
    --output_path ./processed_data/lunaris_data.memmap \
    --hf_dataset_data_dir data \
    --hf_input_column input \
    --hf_target_column output \
    --hf_formatting_template "USER: {input}\nASSISTANT: {target}" \
    --max_length 1024 \
    --add_special_tokens \
    --overwrite_output
```

### 3. Multi-GPU Training with DDP

**🎯 Single-Node, Multi-GPU Training (Recommended):**

Launch distributed training using `torchrun` for optimal performance:

```bash
# Example: Training on a single machine with 2 GPUs
torchrun --standalone --nproc_per_node=2 train.py \
    --memmap_file_train ./processed_data/lunaris_data.memmap \
    --num_sequences_train 100000 \
    --tokenizer_name_or_path bigcode/starcoder \
    --dataset_max_length 1024 \
    --model_max_seq_len 1024 \
    --d_model 512 --n_layers 8 --n_heads 8 \
    --batch_size 8 \
    --accumulation_steps 4 \
    --num_epochs 3 \
    --learning_rate 5e-5 \
    --lr_scheduler_type cosine_warm_restarts \
    --mixed_precision_dtype bf16 \
    --use_torch_compile \
    --lora_rank 16 \
    --checkpoint_dir ./checkpoints_ddp \
    --save_every_n_epochs 1
```

**Key DDP Parameters:**
- `--nproc_per_node=N`: Number of GPUs to use (e.g., 2 for dual-GPU setup)
- `--standalone`: For single-node training (most common use case)
- `--batch_size`: **Per-GPU batch size** (effective batch size = batch_size × num_gpus × accumulation_steps)

**Single-GPU Training:**
For single-GPU or CPU training, use the standard command:
```bash
python train.py \
    --memmap_file_train ./processed_data/lunaris_data.memmap \
    --device cuda \
    # ... other arguments
```

**📋 Training Tips:**
- Use `python train.py --help` for a complete list of available options
- Monitor GPU memory usage and adjust `--batch_size` accordingly
- Enable `--mixed_precision_dtype bf16` for modern GPUs to save memory and increase speed
- Use `--use_torch_compile` for additional performance gains on supported hardware

### 4. Inference & Text Generation

Generate text with your trained model using the feature-rich `inference.py`:

```bash
python inference.py \
    --checkpoint_path ./checkpoints_ddp/best_model.pt \
    --tokenizer_name_or_path bigcode/starcoder \
    --prompt "USER: Write a Python function that calculates factorial.\nASSISTANT:" \
    --max_new_tokens 200 \
    --temperature 0.7 \
    --stream \
    --syntax_highlight python
```

**Advanced Features:**
- `--interactive`: Enter interactive chat mode
- `--stream`: Real-time token streaming
- `--syntax_highlight`: Code syntax highlighting
- `--prompt_file`: Load prompts from file
- `--output_file`: Save generated text
- 
---

## Documentation & Wiki

For comprehensive guides, tutorials, and API documentation, visit the **[Lunaris Codex Project Wiki](https://github.com/MeryylleA/lunariscodex/wiki)**.

**Essential Pages:**
- **[Home](https://github.com/MeryylleA/lunariscodex/wiki/Home)** - Start here for an overview
- **[Dataset and Training Guidelines](https://github.com/MeryylleA/lunariscodex/wiki/Dataset-and-Training-Guidelines)**
- **[Data Preparation Pipeline](https://github.com/MeryylleA/lunariscodex/wiki/Data-Preparation-Pipeline)**
- **[Command-Line Arguments for Training](https://github.com/MeryylleA/lunariscodex/wiki/Command-Line-Arguments-for-Training)**
- **[Tutorial: Training Your First Model](https://github.com/MeryylleA/lunariscodex/wiki/Training-Your-First-Model)**

---

## Roadmap

Our current development focus:

**Immediate Goals:**
- **Enhanced DDP Support**: Multi-node training capabilities and advanced distributed features
- **Model Architecture Improvements**: Additional attention mechanisms and architectural variants
- **Performance Optimizations**: Further integration with `torch.compile` and FlashAttention

**Upcoming Features:**
- **Evaluation Framework**: Comprehensive model evaluation and benchmarking tools
- **Advanced Inference**: Batch generation, speculative decoding, and deployment optimizations
- **Extended Tokenizer Support**: Full integration of custom BPE tokenizers with `prepare_data.py`
- **Production Tools**: Model quantization, ONNX export, and deployment utilities

**Long-term Vision:**
- Release of pretrained base models across various scales
- Community-driven model variants and specialized architectures
- Integration with popular ML frameworks and deployment platforms

---

## Contributing & Community

**Lunaris Codex is built by the community, for the community.** We welcome contributions of all kinds!

**Ways to Contribute:**
- 🐛 **Bug Reports**: Found an issue? Let us know!
- 💡 **Feature Requests**: Have ideas for improvements?
- 📖 **Documentation**: Help improve our guides and tutorials
- 🔧 **Code**: Submit pull requests for new features or fixes
- 💬 **Community**: Join discussions and help other users

**Get Involved:**
- **GitHub**: [github.com/MeryylleA/lunariscodex](https://github.com/MeryylleA/lunariscodex)
- **Discord**: [Moon Cloud Services](https://discord.gg/JNsfzEwMtC) (Lunaris Codex section)
- **Contributing Guide**: See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines

**Developed by:** Francisco Antonio ([@MeryylleA](https://github.com/MeryylleA) on GitHub, [@Meryylle](https://x.com/a93918) on X/Twitter)

---

## License

This project is licensed under the **MIT License**.  
Copyright (c) 2024-2025 **Francisco Antonio**

See [`LICENSE`](LICENSE) for complete terms.

---

## Special Thanks

- **The Open-Source AI Community** for their invaluable tools, research, and datasets
- **PyTorch Team** for the excellent distributed training framework that makes DDP possible
- **Gemini (Google)** for extensive development support, architectural discussions, and documentation assistance
- **Our Contributors** who help make Lunaris Codex better every day

---

## Why "Lunaris"?

> *"Because great ideas are born in silence — and shine like the moon."*

Just as the moon illuminates the darkness with reflected light, Lunaris Codex aims to illuminate the path forward in AI development by reflecting and building upon the best ideas in the field, making them accessible to everyone.
