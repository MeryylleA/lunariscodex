# Lunaris Codex - TPU Training Guide

This guide provides comprehensive instructions for pre-training Lunaris Codex models on Google Cloud TPUs using the `train_xla.py` script. This script is specifically optimized to leverage the power of TPUs for large-scale, efficient training via the PyTorch/XLA library.

## Introduction to TPU Training

Training on Tensor Processing Units (TPUs) offers significant advantages for large language models, primarily in terms of performance and cost-efficiency for massive-scale computation. PyTorch/XLA (Accelerated Linear Algebra) is the library that connects PyTorch to the XLA compiler, which in turn generates highly optimized code for TPU hardware.

The `train_xla.py` script is a refactored version of the main trainer, with key modifications for the TPU environment:

-   **XLA Integration:** Replaces CUDA-specific calls with their XLA equivalents (`xm.xla_device()`, `xm.optimizer_step()`, `xm.save()`, etc.).
-   **Distributed Training via `xmp.spawn`:** Uses the `xla_multiprocessing` library to spawn a separate Python process for each TPU core, handling data and model parallelism automatically.
-   **Optimized Data Loading:** The same high-performance `ShardDataset` is used, but it's now paired with a `DistributedSampler` configured for the XLA environment.
-   **Simplified Mixed Precision:** TPUs have first-class support for `bfloat16`, which is used by default to accelerate training and reduce memory usage without the need for manual loss scaling.
-   **No `torch.compile`:** The XLA backend is a Just-In-Time (JIT) compiler itself, so an additional `torch.compile` step is unnecessary.

---

## Environment Setup

To train on TPUs, you need to install a specific version of PyTorch and `torch_xla` that is compatible with your Cloud TPU environment.

**Recommended Installation:**
We strongly recommend using the following command to install the correct versions of the libraries. This ensures compatibility and stability.

```bash
pip install torch==2.6.0 'torch_xla[tpu]==2.6.0' -f https://storage.googleapis.com/libtpu-releases/index.html
```

Your full setup process will look like this:

```bash
# 1. Clone the repository
git clone https://github.com/MeryylleA/lunariscodex.git
cd lunariscodex

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install the required TPU libraries
pip install torch==2.6.0 'torch_xla[tpu]==2.6.0' -f https://storage.googleapis.com/libtpu-releases/index.html

# 4. Install other dependencies
pip install -r requirements.txt
```

---

## TPU Training Workflow

The workflow for TPU training is very similar to GPU training but with a few key differences in configuration and launch command.

### Phase 1: Data Preparation

This phase is identical to the main `README.md`. You must first prepare your dataset by:
1.  **Sourcing and cleaning** high-quality text data.
2.  **Training a BPE tokenizer** on your data.
3.  **Tokenizing and sharding** your dataset into a directory of `.npy` files.

Please refer to the "Phase 1: Data Preparation" section in the main `README.md` for a detailed guide on this process.

### Phase 2: Model Training on TPUs

This is where you use the `train_xla.py` script.

**1. Configure Your Training Run:**
Create a `train_config_xla.yaml` file. It is nearly identical to the GPU config, with a few minor changes.

```yaml
# train_config_xla.yaml

# --- Model Configuration ---
# This section remains the same as the GPU config.
model:
  vocab_size: 50304
  d_model: 1024
  n_layers: 20
  n_heads: 16
  max_seq_len: 1024
  dropout: 0.0
  n_kv_heads: 4 # For GQA

# --- Data Configuration ---
data_dir: "path/to/your/npy_shards/"

# --- Optimizer Configuration ---
learning_rate: 3.0e-4 # Learning rates might need slight adjustments for TPUs
weight_decay: 0.1
beta1: 0.9
beta2: 0.95

# --- Scheduler Configuration ---
warmup_steps: 2000
max_steps: 600000

# --- Training Configuration ---
# Note the key differences for TPU training:
batch_size: 16           # This is now the PER-CORE batch size.
                         # Effective batch size = batch_size * num_tpu_cores * gradient_accumulation_steps.
gradient_accumulation_steps: 2
grad_clip: 1.0
device: "xla"            # Set device to 'xla'
compile_model: false     # torch.compile is not needed with XLA

# --- I/O and Logging ---
out_dir: "checkpoints/my-tpu-run"
save_interval: 1000
log_interval: 20

# --- W&B Configuration ---
wandb_project: "lunaris-codex-tpu"
wandb_run_name: "tpu-experiment-001"
```

**2. Launch the Training:**
The `train_xla.py` script is launched directly with `python`. The script itself uses `xmp.spawn` to distribute the training across all available TPU cores.

```bash
# Launch the training script
python train_xla.py train_config_xla.yaml
```

The script will automatically detect the number of available TPU cores and spawn a training process for each one. The master process (core 0) will handle all logging and checkpoint saving.

---

## Hardware Recommendations

The choice of TPU hardware depends on your model size and desired batch size. Google Cloud offers different TPU configurations.

| Model Size (Parameters) | Recommended TPU Pod Slice | Why? |
| :--- | :--- | :--- |
| **~1B - 3B** | **TPU v3-8** or **v4-8** | A single TPU VM with 8 cores is a great starting point. It's cost-effective and powerful enough for models in this range. `v4` TPUs offer better performance and more memory (HBM) per core. |
| **7B - 13B** | **TPU v4-32** to **v4-64** | For 7B models, a 32-core pod slice provides a good balance of speed and capacity. Larger models around 13B will benefit from the increased memory and compute power of a 64-core slice. |
| **30B+** | **TPU v4-128** or larger | Training models of this scale requires a significant amount of compute. Larger pod slices (128 cores or more) are necessary to maintain reasonable training times and fit the model parameters and optimizer states in memory. |

**Key Considerations:**

*   **Batch Size:** The `batch_size` in your config is *per core*. A larger pod slice allows for a much larger global batch size, which is often crucial for training stability with very large models.
*   **Memory (HBM):** TPU v4 pods have more High-Bandwidth Memory (HBM) per core than v3 pods, which is critical for fitting larger models, activations, and optimizer states.
*   **Inter-Chip Interconnect (ICI):** Larger pod slices have faster inter-chip communication, which is essential for efficient model and data parallelism.

Always start with a smaller configuration to debug your pipeline before scaling up to a large, expensive pod slice.
