# Lunaris Codex

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Discord](https://img.shields.io/discord/1138864753915854898?label=Discord&logo=discord&color=7289DA)](https://discord.gg/JNsfzEwMtC)

A Note on Our Foundation: The architectural foundation of Lunaris Codex is proudly built upon Andrej Karpathy's nanoGPT. We chose nanoGPT for its brilliant simplicity and clarity, which aligns perfectly with our philosophy of providing a "hackable" and understandable base. This version, however, represents a significant evolution, integrating modern enhancements like RoPE, SwiGLU, and RMSNorm to push performance and capabilities far beyond the original.

**Lunaris Codex** is a streamlined, high-performance toolkit for pre-training powerful language models from scratch. This project provides a modern, Llama-style Transformer architecture and a robust, heavily optimized training script, designed for stability and maximum throughput.

### Our Philosophy
This repository is built on a simple, powerful idea: **provide a rock-solid, understandable foundation for creating strong base models.** We focus on clean, efficient, and well-documented code for the core tasks of model definition and training. This approach empowers researchers and developers to bring their own unique datasets to a proven, production-grade pipeline.

This new version marks a significant evolution of the project, moving to an architecture inspired by industry-leading models like Llama and Mistral, with a strong emphasis on modern best practices.

---

## Architecture Overview

Lunaris Codex is engineered for a balance of performance and clarity. Its architecture is based on the successful Llama-style models, ensuring state-of-the-art performance and training efficiency.

| Component | Implementation | Purpose |
| :--- | :--- | :--- |
| **Normalization** | **RMSNorm** | A simple, high-performance normalization layer that ensures stable training. |
| **Positional Info**| **RoPE (Rotary Positional Embeddings)** | Allows for excellent generalization to various sequence lengths without learned parameters. |
| **FFN Activation**| **SwiGLU** | Provides superior performance and expressiveness compared to standard ReLU/GeLU. |
| **Structure** | Pre-LayerNorm Decoder-Only | A stable and proven Transformer architecture for autoregressive language modeling. |
| **Embeddings** | Tied Input/Output | Reduces parameter count and improves model quality by sharing weights. |

---

## The Training Pipeline

Our `train.py` script is a feature-rich and resilient trainer, built to handle large-scale, long-running jobs.

*   **Engineered for Scale:** Designed to handle terabytes of data and train for days or weeks without interruption.
*   **Optimized Data Loading:** Features a memory-mapped `ShardDataset` class that efficiently handles massive datasets sharded into multiple `.npy` files.
*   **State-of-the-Art Performance:** Full support for `bfloat16`/`fp16` mixed-precision, `torch.compile`, and `DistributedDataParallel` (DDP).
*   **Robust LR Control:** A manual learning rate scheduler with linear warmup and cosine decay for full control and stable convergence.
*   **Resilient Checkpointing:** Automatically resumes from the latest checkpoint, saving the full training state (model, optimizer, step count) to ensure no progress is lost.

---

## Getting Started: The Lunaris Codex Workflow

Training your own model involves two main phases: **Data Preparation** and **Model Training**.

### Phase 1: Data Preparation (Your Task)

We believe that data is the soul of a model. To give you maximum flexibility, our training script is designed to consume a directory of tokenized data sharded into `.npy` files. We empower you to create your own datasets.

Here is a recommended, high-level guide to replicate our data preparation process:

1.  **Select Your Sources:** Choose high-quality, large-scale text corpora. A great starting point is a blend of:
    *   A filtered web crawl like `HuggingFaceFW/fineweb-edu`.
    *   An encyclopedic text source like `wikimedia/wikipedia`.

2.  **Train a Tokenizer:** Use a library like Hugging Face `tokenizers` to train a BPE tokenizer on a representative sample of your chosen datasets. Save the `tokenizer.json` file.

3.  **Tokenize and Shard the Data:**
    *   Write a script that **interleaves** your chosen datasets to create a homogeneous mixture. The `datasets.interleave_datasets` function is highly recommended for this. This is critical for training stability.
    *   Tokenize the text documents and append an End-of-Text token (e.g., `<|endoftext|>`).
    *   Concatenate the token IDs into a large buffer and periodically save it to disk as a shard file (e.g., `shard_00.npy`). A shard size of ~1 billion tokens is a good starting point.
    *   Ensure the final tokens are saved as a `numpy` array with `dtype=np.uint16` for memory efficiency.

### Phase 2: Model Training (Our Speciality)

This is where Lunaris Codex shines.

**1. Setup the Environment:**
```bash
# Clone the repository
git clone https://github.com/MeryylleA/lunariscodex.git
cd lunariscodex

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**2. Configure Your Training Run:**
Create a `train_config.yaml` file. This is where you define your model architecture, hyperparameters, and data paths. Start with our recommended stable configuration:

```yaml
# train_config.yaml
model:
  vocab_size: 65536      # Must match your tokenizer
  d_model: 1024
  n_layers: 32
  n_heads: 16
  max_seq_len: 1024
  dropout: 0.05

data_dir: "path/to/your/npy_shards/" # IMPORTANT: Point this to your data

learning_rate: 1.0e-4
max_steps: 103250        # Adjust based on your dataset size
warmup_steps: 10000

batch_size: 48           # Adjust based on your VRAM
gradient_accumulation_steps: 2

out_dir: "checkpoints/my-first-run"
wandb_project: "My-Lunaris-Project"
wandb_run_name: "run-1"
```

**3. Launch the Training!**
```bash
# For single-GPU or single-node multi-GPU training
torchrun --standalone --nproc_per_node=auto train.py train_config.yaml
```
The script will handle the rest, from compiling the model to logging on W&B and saving checkpoints.

---

## License & Community

This project is licensed under the **MIT License**.

Developed by **Francisco Antonio** ([@MeryylleA](https://github.com/MeryylleA) on GitHub).

Join our community on Discord for discussions, help, and to share your results: [**Moon Cloud Services**](https://discord.gg/JNsfzEwMtC)

### Special Thanks
*   To Andrej Karpathy for `nanoGPT`, which served as the inspirational and architectural starting point for this project.
*   To the open-source AI community for their invaluable tools, research, and datasets.
*   To **Google Gemini** for extensive pair-programming sessions, architectural discussions, debugging, and documentation assistance.
*   To **Lambda Labs** for supporting this project with the large-scale compute necessary for this research.
