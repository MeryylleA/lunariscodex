# Lunaris Codex

> **Note:** You are viewing an **experimental branch** of Lunaris Codex.  
> This version introduces a **Mixture-of-Experts (MoE) layer** in the style of the **Switch Transformer (k=1)**.  
> All MoE-related code lives in `model_moe.py` and is trained with `train_moe.py`.  
> These features are under active evaluation—expect breaking changes and rapid iteration.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Discord](https://img.shields.io/discord/1138864753915854898?label=Discord&logo=discord&color=7289DA)](https://discord.gg/JNsfzEwMtC)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/MeryylleA/lunariscodex)

A Note on Our Foundation: The architectural foundation of Lunaris Codex is proudly built upon Andrej Karpathy's nanoGPT. We chose nanoGPT for its brilliant simplicity and clarity, which aligns perfectly with our philosophy of providing a "hackable" and understandable base. This version, however, represents a significant evolution, integrating modern enhancements like **RoPE, Grouped Query Attention (GQA), Fused SwiGLU, Gradient Checkpointing**, and now **Mixture-of-Experts (MoE)** to push performance and capabilities far beyond the original.

**Lunaris Codex** is a streamlined, high-performance toolkit for pre-training powerful language models from scratch. This project provides a modern, Llama-style Transformer architecture and a robust, heavily optimized training script, designed for stability and maximum throughput.

### Our Philosophy
This repository is built on a simple, powerful idea: **provide a rock-solid, understandable foundation for creating strong base models.** We focus on clean, efficient, and well-documented code for the core tasks of model definition and training. This approach empowers researchers and developers to bring their own unique datasets to a proven, production-grade pipeline.

---

## Architecture Overview

Lunaris Codex is engineered for a balance of performance and clarity. Its architecture integrates several state-of-the-art features to ensure top-tier performance and training efficiency.

| Component | Implementation | Benefits & Considerations |
| :--- | :--- | :--- |
| **Normalization** | **RMSNorm** | A simpler, more efficient normalization technique than standard LayerNorm. It stabilizes training by normalizing activations based on their root mean square, using a single learnable gain parameter. |
| **Positional Info**| **RoPE (Rotary Positional Embeddings)** | Injects relative positional information, leading to excellent generalization across various sequence lengths. Achieved without any learned parameters, making it efficient. |
| **Attention** | **Grouped Query Attention (GQA)** | Drastically reduces the memory usage of the KV cache during inference by sharing Key/Value heads across groups of Query heads. This enables faster generation and larger context windows. |
| **FFN Activation**| **SwiGLU** | Offers improved performance over traditional activations like ReLU. It uses a gated linear unit, which allows the network to control the flow of information through the activation. |
| **Mixture-of-Experts (MoE)** | **Switch-style (k=1) Router + Expert FFNs** | **Benefits:** Achieves a much larger effective parameter count **without** increasing computation per forward pass. Each token is routed to exactly **one** expert (k=1), keeping FLOPs constant while unlocking model capacity. **Considerations:** Requires an auxiliary load-balancing loss to prevent expert collapse and ensure uniform utilization. |
| **Training** | **Gradient Checkpointing** | Massively reduces VRAM usage during training by recomputing activations during the backward pass instead of storing them. This allows for training larger models or using larger batch sizes, at the cost of a small compute overhead. |
| **Structure** | **Pre-LayerNorm Decoder-Only Transformer** | A standard, proven architecture for autoregressive language modeling. Pre-LayerNorm (applying normalization before attention/FFN) enhances training stability, especially for deep networks. |
| **Embeddings** | **Tied Input/Output Token Embeddings** | Significantly reduces the model's parameter count by sharing weights between the token embedding layer and the final output layer. Can also improve model quality and training efficiency. |

---

## The Training Pipeline

Our `train_moe.py` script is a feature-rich and resilient trainer, meticulously engineered to handle large-scale, long-running jobs with stability and efficiency. It is **specifically adapted** for the MoE architecture.

*   **Engineered for Scale:** Designed to process terabytes of data and sustain training for extended periods (days or weeks) without interruption.
*   **Optimized Data Loading (`ShardDataset`):** Employs a memory-mapped `ShardDataset` class, which efficiently streams data from massive datasets sharded into multiple `.npy` files. This approach minimizes RAM overhead while maximizing I/O throughput.
*   **High-Performance Training:**
    *   **Mixed-Precision:** Full support for `bfloat16` (preferred on compatible hardware) and `fp16` to accelerate training and reduce memory footprint.
    *   **`torch.compile`:** Integrates `torch.compile` for graph optimization, potentially speeding up model execution.
    *   **Distributed Training (`DDP`):** Leverages PyTorch's `DistributedDataParallel` (DDP) for efficient multi-GPU and multi-node training.
    *   **AdamW Optimizer:** Utilizes the AdamW optimizer, a standard choice for robust language model training.
    *   **Gradient Accumulation:** Allows for larger effective batch sizes by accumulating gradients over multiple steps.
    *   **Gradient Clipping:** Implements gradient clipping to prevent exploding gradients and stabilize training.
*   **Precise Learning Rate Control:** Features a learning rate scheduler with a linear warmup phase followed by a cosine decay. This allows for fine-grained control over the learning rate trajectory, crucial for stable convergence.
*   **Resilient Checkpointing:** Automatically resumes from the latest checkpoint if training is interrupted. Checkpoints save the complete training state (model weights, optimizer state, step count, epoch, and training configuration) to ensure no progress is lost and facilitate seamless continuation.
*   **Comprehensive Monitoring:** Integrates with Weights & Biases (W&B) for detailed experiment tracking (loss, perplexity, learning rate, gradient norms, **main_loss**, **aux_loss**, etc.) and provides informative console logging with progress bars via `tqdm`.

### Auxiliary Load-Balancing Loss (`aux_loss`)

Because only a **sparse subset** of experts is active for any given token, there is a risk that the gating network will **collapse**—always routing tokens to the same few experts.  
To prevent this, we add an **auxiliary loss** (`aux_loss`) that encourages uniform expert utilization.

*   **Mechanism:** The loss is computed as the dot product between (a) the **fraction of tokens dispatched to each expert** and (b) the **fraction of router probability mass assigned to each expert** (see `MixtureOfExperts.forward()` for the exact formula).
*   **Scaling:** The loss is multiplied by `config.aux_loss_weight` (default: `1e-2`, taken from the Switch Transformer paper) before being added to the **main cross-entropy loss**.
*   **Logging:** `train_moe.py` logs `main_loss`, `aux_loss`, and their sum separately in both the console and Weights & Biases, making it easy to verify that experts are being used evenly.

---

## Getting Started: The Lunaris Codex Workflow

Training your own model involves two main phases: **Data Preparation** and **Model Training**.

### Phase 1: Data Preparation (Your Task)

We believe that data is the soul of a model. While Lunaris Codex provides the engine, you provide the fuel. Our training script is designed to consume a directory of tokenized data sharded into `.npy` files. This phase is your responsibility, but here's a strong guide:

1.  **Select Your Sources:**
    *   **Quality & Scale:** Prioritize high-quality, large-scale text corpora.
    *   **Diversity:** A diverse dataset (e.g., combining web text, books, code, scientific articles, and conversational data) often leads to more robust and versatile models.
    *   **Examples:** Good starting points include filtered web crawls like `HuggingFaceFW/fineweb-edu` and encyclopedic sources like `wikimedia/wikipedia`. Consider what capabilities you want your model to have and select data accordingly.

2.  **Train a Tokenizer:**
    *   We strongly recommend using the Hugging Face `tokenizers` library to train a Byte Pair Encoding (BPE) tokenizer.
    *   Train it on a representative sample of your chosen datasets.
    *   Save the resulting `tokenizer.json` file. This file is crucial, as its vocabulary size will define `model.vocab_size` in your training configuration.

3.  **Tokenize and Shard the Data:** This is a critical step for stable and efficient training.
    *   **Interleaving Datasets:** Before tokenization, **interleave** your different data sources into a single, homogeneous mixture. The `datasets.interleave_datasets` function from the Hugging Face `datasets` library is highly recommended for this. *This step is crucial for training stability and preventing the model from overfitting to one data type early on.*
    *   **End-of-Text (EOT) Token:** Tokenize your text documents and, importantly, append a special End-of-Text token (e.g., `<|endoftext|>`, ensure this token is part of your tokenizer's vocabulary) to the end of each document. This helps the model learn document boundaries.
    *   **Concatenate and Shard:** Concatenate all tokenized token IDs into a very large buffer. Periodically save this buffer to disk as a shard file (e.g., `shard_0001.npy`, `shard_0002.npy`, etc.).
        *   **Shard Size:** A shard size of approximately 1 billion tokens per `.npy` file is a good starting point, balancing manageability and I/O efficiency.
        *   **Data Type:** Ensure the final token IDs in your `.npy` files are saved as `numpy` arrays with `dtype=np.uint16`. This significantly reduces disk space and memory usage, as token IDs rarely exceed 65535.

### Phase 2: Model Training (Our Speciality)

This is where Lunaris Codex shines. Follow these steps to launch your training run:

**1. Setup the Environment:**
```bash
# Clone the repository
git clone https://github.com/MeryylleA/lunariscodex.git
cd lunariscodex

# Create and activate a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**2. Configure Your Training Run:**
Create a `train_config_moe.yaml` file. This is where you define your model architecture, hyperparameters, and data paths. Below is a well-commented example configuration that shows how to use the new MoE features.

```yaml
# train_config_moe.yaml

# --- Model Configuration ---
model:
  vocab_size: 50304          # MUST match your tokenizer's vocab size
  d_model: 1024
  n_layers: 20
  n_heads: 16
  n_kv_heads: 4              # Enable GQA: n_heads must be divisible by n_kv_heads
  max_seq_len: 1024
  dropout: 0.0

  # --- MoE-specific settings ---
  n_experts: 8               # Number of expert FFNs in each MoE layer
  n_experts_per_token: 1     # Switch-style routing: 1 expert per token
  aux_loss_weight: 0.01      # Weight for the load-balancing auxiliary loss

# --- Data Configuration ---
data_dir: "path/to/your/npy_shards/"

# --- Optimizer & Scheduler ---
learning_rate: 2.0e-4
weight_decay: 0.1
warmup_steps: 2000
max_steps: 200000

# --- Training ---
batch_size: 32
gradient_accumulation_steps: 4
grad_clip: 1.0
compile_model: true

# --- I/O & Logging ---
out_dir: "checkpoints/lunaris-moe"
save_interval: 1000
log_interval: 20

# --- Weights & Biases (optional) ---
wandb_project: "lunaris-codex-moe"
wandb_run_name: "moe-8ex-gqa-20L"
```

**3. Launch the Training!**
```bash
# Single-GPU or single-node multi-GPU
torchrun --standalone --nproc_per_node=auto train_moe.py train_config_moe.yaml
```
The script will handle the rest: setting up distributed training, compiling the model, loading data, running the training loop, logging, and saving checkpoints.

---

## Best Practices for Pre-training

*   **Start Small, Iterate Fast:** Before committing to a multi-week run, validate your pipeline with a small model (`d_model=512`, `n_layers=8`, etc.) and a subset of data.
*   **Monitor Expert Utilization:** Watch the `aux_loss` curve in W&B. If it spikes or plateaus high, reduce `aux_loss_weight` or check data quality.
*   **Tune `aux_loss_weight` Carefully:** Too low → expert collapse. Too high → degraded language modeling performance. Typical values: `1e-3`–`1e-2`.
*   **All other best practices** (data quality, tokenizer alignment, learning-rate tuning, checkpointing) remain identical to the original README.

---

## Limitations

*   **Sparse Computation:** The current implementation uses a simple Python loop for token routing. While correct and readable, it is **not** optimized for maximum throughput. Future releases may include custom CUDA kernels or `torch.sparse` improvements.
*   **Expert Count & Memory:** While FLOPs per forward pass stay constant, **total parameter count scales linearly** with `n_experts`. Ensure you have enough GPU memory and CPU RAM to hold all expert weights.
*   **Fine-tuning & Downstream Tasks:** As with the base repository, fine-tuning scripts and built-in evaluation suites are not included. Users must adapt the pre-trained model to downstream tasks using external tooling.
*   **Resource Requirements:** MoE models can be **larger in memory** than dense models of the same computational budget. Budget your GPUs and storage accordingly.

---

## License & Community

This project is licensed under the **MIT License**.

Developed by **Francisco Antonio** ([@MeryylleA](https://github.com/MeryylleA) on GitHub).

Join our community on Discord for discussions, help, and to share your results: [**Moon Cloud Services**](https://discord.gg/JNsfzEwMtC)

### Special Thanks
*   To Andrej Karpathy for `nanoGPT`.
*   To the open-source AI community.
*   To Google for the Switch Transformer paper and design principles.
