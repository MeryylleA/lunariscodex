---

# Lunaris Codex

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Discord](https://img.shields.io/discord/1138864753915854898?label=Discord&logo=discord&color=7289DA)](https://discord.gg/JNsfzEwMtC)


**Lunaris Codex** is a state-of-the-art Transformer architecture and an optimized training script designed for building language models from scratch.

### Our Philosophy
This repository focuses on the essentials: high-quality, robust code for training powerful language models. We believe that a solid foundation is paramount. To support this, we handle the intensive work of data curation and provide ready-to-use, pre-processed datasets separately on the Hugging Face Hub. This allows you, the researcher or developer, to focus on what matters most: training and experimenting with models.

---

## Key Features

*   **Advanced Model Architecture (`model.py`):**
    *   A clean, decoder-only Transformer implementation featuring modern components like **Pre-Layer Normalization**, **SwiGLU** activation, and **Tied Embeddings** for training stability and parameter efficiency.
    *   **ALiBi (Attention with Linear Biases)** is integrated for superior long-context handling, removing the need for traditional positional embeddings.
    *   **LayerScale** is dynamically applied to stabilize training for larger model configurations.

*   **Flexible and Robust Training (`train.py`):**
    *   Supports both **full pre-training** and parameter-efficient **LoRA fine-tuning** out of the box.
    *   **Advanced Checkpointing System** with SHA256 integrity verification, automatic resumption from the latest or best checkpoint, and full state saving (model, optimizer, scheduler, args).
    *   **High-Performance Optimizations:** Full support for `bfloat16`/`fp16` mixed-precision, `torch.compile`, fused AdamW, and TF32 for maximum throughput on modern hardware.

*   **Rich and Interactive Inference (`inference.py`):**
    *   A polished command-line interface powered by the `rich` library for a superior user experience.
    *   **💬 Interactive Chat Mode:** Converse with your trained models directly in the terminal.
    *   **⚡️ Streaming Generation:** Watch the model generate text token-by-token in real-time.
    *   **Syntax Highlighting:** Formatted and colored code output for improved readability.
    *   **Flexible and User-Friendly:** Load prompts from files, save generation results to Markdown, and configure generation parameters on the fly.

---

## Architecture Overview

Lunaris Codex is engineered for a balance of performance, modern features, and code clarity. The table below summarizes its core components.

| Component | Implementation | Purpose |
| :--- | :--- | :--- |
| **Normalization** | Pre-LayerNorm | Ensures stable training gradients in deep networks. |
| **Positional Info**| ALiBi | Allows generalization to sequence lengths beyond the training window. |
| **FFN Activation**| SwiGLU | Provides better performance compared to standard ReLU/GeLU. |
| **Stabilization** | LayerScale | Adaptively scales residual connections to prevent exploding activations. |
| **Embeddings** | Tied Input/Output | Reduces parameter count and improves model quality. |
| **Fine-Tuning** | LoRA Layers | Enables highly efficient adaptation of pre-trained models. |

---

## Before You Begin: Hardware Requirements & Expectations

Training a large language model from scratch is a computationally intensive endeavor. We want to be transparent about what it takes to use Lunaris Codex effectively.

### Required Hardware

*   **GPU:** A modern GPU with significant VRAM and support for `bfloat16` (BF16) is **essential**.
    *   **Performance Baseline:** On our reference server (NVIDIA GH200 96GB), a full training epoch (processing 2.5B tokens) for the 1.2B parameter model takes **approximately 20 hours**.
    *   **Recommended Minimum:** To replicate this training, an NVIDIA A100 (40GB+) or a high-end consumer GPU like an RTX 4090 / 3090 (24GB) is a realistic starting point. Expect significantly longer training times on this hardware (e.g., 2-4 days per epoch on an A100).
    *   **VRAM:** Training the 1.2B parameter model requires **at least 24GB of VRAM**. For LoRA fine-tuning, 16GB may be sufficient.

*   **CPU, RAM, & Storage:**
    *   **CPU/RAM:** At least 8 CPU cores and 32GB of RAM are recommended to prevent data loading from becoming a bottleneck.
    *   **Storage:** A fast SSD is crucial. You will need space for the dataset (~5 GB) and multiple model checkpoints (5-10 GB each). Plan for at least **100 GB** of free space.

### Managing Expectations

1.  **Time is a Real Factor:** Training is measured in days, not hours. Please plan accordingly.
2.  **This is a Learning Tool:** The primary goal of this project is to provide a robust, understandable toolkit for you to learn the end-to-end process of building LLMs. The resulting models are highly capable, but do not expect them to compete with massive, proprietary models like GPT-4 after a single training run. The value is in the process, the code, and the knowledge you will gain.

---

## Lunaris Datasets for Training

To get you started, we provide pre-processed, tokenized, and curated datasets on the Hugging Face Hub. These are ready to be used directly with our `train.py` script.

*   **1. `smeryylle/lunaris-codex-2.5b-curriculum` (Recommended)**
    *   **Description:** 2.5 billion high-quality tokens curated from the Dolma dataset, arranged with a curriculum learning strategy. Ideal for pre-training models in the ~1B parameter range.
    *   **Hugging Face Link:** [meryyllebr543/lunaris-codex-2.5b-curriculum](https://huggingface.co/datasets/meryyllebr543/lunaris-codex-2.5b-curriculum)

*   *(More datasets focused on Code and Instruction Fine-Tuning are planned for the future.)*

---

## How to Train Your Own Lunaris Codex

Follow these three steps to begin your training journey.

### Step 1: Setup the Environment
```bash
# Clone the repository
git clone https://github.com/MeryylleA/lunariscodex.git
cd lunariscodex

# Create and activate a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Download the Data
We recommend using `git` with LFS to download our pre-processed dataset.

```bash
# Ensure you have git-lfs installed (https://git-lfs.com)
git lfs install

# Clone the dataset repository
git clone https://huggingface.co/datasets/meryyllebr543/lunaris-codex-2.5b-curriculum
```
This will download the `lunaris_codex_treino.npy` file into a local directory.

### Step 3: Train!

Here are a few common training scenarios.

**Example 1: Pre-training a 1.2B model from scratch**
This is the "golden path" for creating a new base model.
```bash
torchrun --standalone --nproc_per_node=1 train.py \
    --memmap_file_train ./lunaris-codex-2.5b-curriculum/lunaris_codex_treino.npy \
    --num_sequences_train 2441414 \
    --tokenizer_name_or_path allenai/OLMo-7B-0724-hf \
    --vocab_size 50280 \
    --d_model 2048 \
    --n_layers 16 \
    --n_heads 16 \
    --batch_size 16 \
    --learning_rate 3e-4 \
    --weight_decay 0.1 \
    --mixed_precision_dtype "bf16" \
    --use_torch_compile \
    --checkpoint_dir ./checkpoints/lunaris-1.2B-run1
```

**Example 2: Fine-tuning a pre-trained model with LoRA**
Use this to adapt a model for a specific task with very few trainable parameters.
```bash
python train.py \
    --resume_from_checkpoint ./checkpoints/lunaris-1.2B-run1/best_model.pt \
    --memmap_file_train ./path/to/your/sft_dataset.npy \
    --num_sequences_train 10000 \
    --lora_rank 16 \
    --num_epochs 1 \
    --learning_rate 1e-4 \
    --checkpoint_dir ./checkpoints/lunaris-1.2B-lora-finetune
```

**Example 3: Resuming an interrupted training run**
Simply point to the same `checkpoint_dir` and the script will automatically find the latest checkpoint and resume.
```bash
torchrun --standalone --nproc_per_node=1 train.py \
    --memmap_file_train ./lunaris-codex-2.5b-curriculum/lunaris_codex_treino.npy \
    --num_sequences_train 2441414 \
    # ... all other arguments must be IDENTICAL to the original run ...
    --checkpoint_dir ./checkpoints/lunaris-1.2B-run1
```

---

## Using Your Trained Model (`inference.py`)

Interact with your model using our feature-rich inference script.

**Example 1: Simple Generation**
```bash
python inference.py \
    --checkpoint_path ./checkpoints/lunaris-1.2B-run1/best_model.pt \
    --prompt "USER: What is ALiBi in Transformers?\nASSISTANT:"
```

**Example 2: ⚡️ Real-time Streaming Generation**
```bash
python inference.py \
    --checkpoint_path ./checkpoints/lunaris-1.2B-run1/best_model.pt \
    --prompt "USER: Write a short story about a robot who discovers music.\nASSISTANT:" \
    --stream
```

**Example 3: 💬 Interactive Chat Mode**
Start a conversation with your model. Use `/set`, `/clear`, and `/help` for in-session controls.
```bash
python inference.py \
    --checkpoint_path ./checkpoints/lunaris-1.2B-run1/best_model.pt \
    --interactive
```

---

## License

This project is licensed under the **MIT License**. See the [`LICENSE`](LICENSE) file for details.

## Contributing & Community

Developed by **Francisco Antonio** ([@MeryylleA](https://github.com/MeryylleA) on GitHub).

Lunaris Codex is an open-source project. Contributions, feedback, and bug reports are highly welcome! Please see our [`CONTRIBUTING.md`](CONTRIBUTING.md) guidelines for more information.

Join our community on Discord: [**Moon Cloud Services**](https://discord.gg/JNsfzEwMtC)

### Special Thanks
*   To the open-source AI community for their invaluable tools, research, and datasets.
*   To **Google Gemini** for extensive pair-programming sessions, architectural discussions, debugging, and documentation assistance.
*   Lambda AI, for supporting this project with large-scale training servers
