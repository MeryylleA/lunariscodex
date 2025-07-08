# Lunaris Codex

> **Note:** You are viewing the `experimental/hybrid-attention` branch of Lunaris Codex. This version introduces a state-of-the-art hybrid attention architecture for superior long-context performance. This feature is powerful but currently under evaluation.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Discord](https://img.shields.io/discord/1138864753915854898?label=Discord&logo=discord&color=7289DA)](https://discord.gg/JNsfzEwMtC)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/MeryylleA/lunariscodex)

A Note on Our Foundation: The architectural foundation of Lunaris Codex is proudly built upon Andrej Karpathy's nanoGPT. We chose nanoGPT for its brilliant simplicity and clarity, which aligns perfectly with our philosophy of providing a "hackable" and understandable base. This version, however, represents a revolutionary evolution, introducing the **RNope-SWA hybrid attention architecture** - a groundbreaking approach that interleaves two specialized types of attention layers to achieve unprecedented long-context performance and training efficiency.

**Lunaris Codex** is a streamlined, high-performance toolkit for pre-training powerful language models from scratch. This experimental branch features our cutting-edge hybrid attention architecture, designed to break the quadratic complexity barrier of traditional transformers while maintaining exceptional performance across both local and global context understanding.

### Our Philosophy
This repository is built on a simple, powerful idea: **provide a rock-solid, understandable foundation for creating strong base models that can handle massive contexts efficiently.** We focus on clean, efficient, and well-documented code for the core tasks of model definition and training. This approach empowers researchers and developers to bring their own unique datasets to a proven, production-grade pipeline capable of processing sequences far longer than traditional architectures.

This experimental version marks a quantum leap in the project's evolution, introducing a hybrid architecture that combines the best of both worlds: efficient local processing and powerful global reasoning capabilities.

---

## Architecture Overview: RNope-SWA Hybrid Attention

Lunaris Codex now features the **RNope-SWA** architecture, a revolutionary hybrid attention system inspired by "Rope to Nope and Back Again." This architecture interleaves two specialized layer types to achieve superior long-context performance while maintaining computational efficiency.

### The Hybrid Architecture

Our model alternates between two distinct types of transformer layers, each optimized for different aspects of language understanding:

| Layer Type | Implementation | Role & Benefits |
| :--- | :--- | :--- |
| **RoPE + Sliding Window Attention (SWA)** | **Local Specialist Layers** | **Role:** Handle local context, syntax, and short-term dependencies with exceptional efficiency. **Benefits:** Uses RoPE for relative positional encoding within a fixed window (e.g., 4096 tokens). Computational complexity of O(L×W) where W is window size, dramatically reducing memory usage for long sequences. Excels at local coherence, grammar, and immediate context understanding. |
| **NoPE + Full Causal Attention** | **Global Retriever Layers** | **Role:** Perform content-based global reasoning without positional bias. **Benefits:** Operates without positional embeddings (NoPE), using full attention to find and connect relevant information across the entire sequence based purely on semantic similarity. Enables powerful long-range reasoning and information retrieval capabilities. |

### Additional Architectural Features

| Component | Implementation | Benefits & Considerations |
| :--- | :--- | :--- |
| **Normalization** | **RMSNorm with Learnable Bias** | **Benefits:** More expressive than standard RMSNorm. The learnable bias allows the model to learn an optimal activation mean, enhancing stability by preventing mean-shift in deep networks. |
| **FFN Activation**| **SwiGLU with Fused Projection** | **Benefits:** Offers improved performance over traditional activations. The gate and up-projections are fused into a single linear layer, improving GPU performance by reducing kernel overhead. |
| **Training** | **Gradient Checkpointing** | **Benefits:** Massively reduces VRAM usage during training by recomputing activations during the backward pass instead of storing them. Essential for training large models with extended context lengths. |
| **Structure** | **Pre-LayerNorm Decoder-Only Transformer** | **Benefits:** A proven architecture for autoregressive language modeling. Pre-LayerNorm enhances training stability, especially crucial for the hybrid attention setup. |
| **Embeddings** | **Tied Input/Output Token Embeddings** | **Benefits:** Significantly reduces parameter count while improving model quality and training efficiency. |

### The Hybrid Advantage

This revolutionary architecture delivers several key advantages:

- **Efficient Long-Context Processing:** Most layers (RoPE+SWA) operate with linear complexity relative to window size, not full sequence length
- **Powerful Global Reasoning:** NoPE layers provide content-based attention across the entire sequence for sophisticated long-range understanding
- **Balanced Performance:** Local specialists handle immediate context while global retrievers manage long-term dependencies
- **Scalable Training:** Reduced computational burden enables training on much longer sequences than traditional architectures

---

## The Training Pipeline

Our `train.py` script is a feature-rich and resilient trainer, meticulously engineered to handle large-scale, long-running jobs with stability and efficiency, now optimized for the hybrid attention architecture.

*   **Engineered for Scale:** Designed to process terabytes of data and sustain training for extended periods (days or weeks) without interruption, with special optimizations for long-context training.
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
*   **Comprehensive Monitoring:** Integrates with Weights & Biases (W&B) for detailed experiment tracking (loss, perplexity, learning rate, etc.) and provides informative console logging with progress bars via `tqdm`.

---

## Getting Started: The Lunaris Codex Workflow

Training your own model involves two main phases: **Data Preparation** and **Model Training**.

### Phase 1: Data Preparation (Your Task)

We believe that data is the soul of a model. While Lunaris Codex provides the engine, you provide the fuel. Our training script is designed to consume a directory of tokenized data sharded into `.npy` files. This phase is your responsibility, but here's a strong guide:

1.  **Select Your Sources:**
    *   **Quality & Scale:** Prioritize high-quality, large-scale text corpora.
    *   **Diversity:** A diverse dataset (e.g., combining web text, books, code, scientific articles, and conversational data) often leads to more robust and versatile models.
    *   **Long-Context Considerations:** With the hybrid architecture's ability to handle extended sequences, consider including datasets with longer documents to fully leverage the model's capabilities.
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

# Switch to the experimental branch
git checkout experimental/hybrid-attention

# Create and activate a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**2. Configure Your Training Run:**
Create a `train_config.yaml` file. This is where you define your model architecture, hyperparameters, and data paths. Below is a well-commented example configuration that shows how to use the hybrid attention features.

```yaml
# train_config.yaml

# --- Model Configuration ---
# These parameters define the architecture of your LunarisCodex hybrid attention model.
model:
  vocab_size: 50304      # IMPORTANT: Must exactly match your tokenizer's vocabulary size.
  d_model: 1024          # Dimensionality of the model embeddings and hidden states.
  n_layers: 20           # Number of transformer blocks (layers).
  n_heads: 16            # Number of attention heads (must divide d_model).
  max_seq_len: 32768     # Maximum sequence length - leverage the hybrid architecture for long contexts!
  dropout: 0.0           # Dropout rate for regularization (0.0 to disable).

  # --- Hybrid Attention Configuration ---
  sliding_window_size: 4096  # Window size for RoPE+SWA layers. This defines the local attention
                             # window for efficient processing. Should be a power of 2.
  
  # --- Advanced Architectural Features ---
  n_kv_heads: 4          # Set to a value less than n_heads to enable Grouped Query Attention (GQA).
                         # `n_heads` must be divisible by `n_kv_heads`. e.g., 16 heads, 4 kv_heads.
                         # Omit this or set to null to default to standard Multi-Head Attention.
  
  use_gradient_checkpointing: true # Set to true to enable gradient checkpointing.
                                   # This saves a large amount of GPU memory at the cost of
                                   # slightly slower training iterations. Highly recommended for
                                   # large models or large batch sizes, especially with long contexts.

# --- Data Configuration ---
data_dir: "path/to/your/npy_shards/" # IMPORTANT: Point this to your directory of sharded .npy files.

# --- Optimizer Configuration ---
learning_rate: 2.0e-4    # Peak learning rate. Tune this carefully.
weight_decay: 0.1        # Weight decay for AdamW optimizer.

# --- Scheduler Configuration ---
warmup_steps: 2000       # Number of steps for linear learning rate warmup.
max_steps: 200000        # Total number of training steps.

# --- Training Configuration ---
batch_size: 16           # Per-GPU batch size. Reduced for long-context training.
gradient_accumulation_steps: 8 # Accumulates gradients over N micro-batches.
                         # Effective batch size = batch_size * num_gpus * gradient_accumulation_steps.
grad_clip: 1.0           # Gradient clipping value to prevent exploding gradients.
compile_model: true      # Whether to use torch.compile for potential speedups.

# --- I/O and Logging ---
out_dir: "checkpoints/hybrid-attention-run"   # Directory to save checkpoints.
save_interval: 1000      # Save a checkpoint every N steps.
log_interval: 20         # Log metrics to console/W&B every N steps.

# --- W&B (Weights & Biases) Configuration (Optional) ---
wandb_project: null                # Your W&B project name (e.g., "HybridAttentionExperiment"). Keep as null or remove to disable.
wandb_run_name: "rnope-swa-001"    # A name for this specific run (e.g., "hybrid-32k-context").
# wandb_entity: null               # Your W&B entity/team name (if applicable).
```

**3. Launch the Training!**
The `train.py` script uses `torchrun` for launching. This command is suitable for single-GPU, multi-GPU on a single node. For multi-node training, additional arguments for `torchrun` like `--nnodes`, `--node_rank`, `--rdzv_id`, `--rdzv_backend`, `--rdzv_endpoint` would be required.

```bash
# For single-GPU or single-node multi-GPU training:
# 'auto' will attempt to use all available GPUs on the node.
# If you want to specify the number of GPUs, use e.g., --nproc_per_node=2 for 2 GPUs.
torchrun --standalone --nproc_per_node=auto train.py train_config.yaml
```
The script will handle the rest: setting up distributed training (if applicable), compiling the model (if enabled), loading data, running the training loop, logging to the console and W&B (if configured), and saving checkpoints.

Good luck, and we're excited to see what you build with the revolutionary hybrid attention architecture of Lunaris Codex!

---

### Phase 3: Training on Google Cloud TPUs with `train_tpu.py` (Alternative)

For users with access to Google Cloud TPUs, Lunaris Codex provides `train_tpu.py`, a dedicated script optimized for training on these powerful accelerators using PyTorch/XLA. This offers an alternative to GPU-based training, often enabling larger scale experiments with the hybrid attention architecture.

**Prerequisites:**

*   **Google Cloud TPU Environment:** You must have a Google Cloud project with access to TPUs. This typically involves creating a TPU VM instance (e.g., a `ctpu` instance or a VM in a TPU Pod slice).
*   **PyTorch/XLA Installation:** PyTorch/XLA needs to be installed in your environment. Google Cloud TPU VMs often come pre-configured with PyTorch and PyTorch/XLA. If not, you'll need to follow the official PyTorch/XLA installation instructions for your specific TPU setup.
*   **Dependencies:** Ensure other dependencies from `requirements.txt` are installed in your TPU VM environment.

**1. Setup the Environment (on the TPU VM):**
Ensure you have cloned the repository and installed requirements on your TPU VM, similar to the GPU setup:
```bash
# Clone the repository (if not already done)
# git clone https://github.com/MeryylleA/lunariscodex.git
# cd lunariscodex

# Switch to the experimental branch
# git checkout experimental/hybrid-attention

# Create and activate a virtual environment (recommended)
# python3 -m venv .venv
# source .venv/bin/activate

# Install dependencies (ensure PyTorch/XLA is already provided or installed correctly)
pip install -r requirements.txt
```

**2. Configure Your Training Run:**
You will use the same `train_config.yaml` file as described in "Phase 2: Model Training".
*   **Data Path:** Ensure `data_dir` in your `train_config.yaml` points to the location of your sharded `.npy` files, accessible from the TPU VM.
*   **TPU Specifics:** The `train_tpu.py` script is designed to work with XLA. Configuration options like `device` or `compile_model: true` in your `train_config.yaml` will be ignored, as XLA handles device management and compilation.

**3. Launch the TPU Training:**
Unlike the GPU script that uses `torchrun`, the `train_tpu.py` script is typically launched directly with Python. The script itself uses `xmp.spawn` internally to distribute the training across all available TPU cores.

```bash
# Launch training on all available TPU cores
python train_tpu.py train_config.yaml
```
The script will automatically detect and utilize all TPU cores available to the VM. It will then proceed with data loading, model initialization on TPUs, and the training loop, logging progress and saving checkpoints as configured.

**How PyTorch/XLA Enables TPU Training:**

Google's TPUs require specialized software to interface with PyTorch code. This is where PyTorch/XLA comes in:

*   **PyTorch/XLA:** This is a Python package that allows PyTorch to run on XLA (Accelerated Linear Algebra) devices, including Google TPUs. It acts as a bridge between your PyTorch model and the TPU hardware.
*   **XLA Compilation:** When you run a PyTorch model with XLA, the XLA compiler takes your PyTorch operations and compiles them into highly optimized machine code specifically for the TPU architecture. This compilation step is key to achieving high performance on TPUs.
*   **Distributed Training:** The `torch_xla` library provides tools like `xmp.spawn` (used internally by `train_tpu.py`) to easily launch your training script across all TPU cores in a distributed manner. It handles the complexities of:
    *   **Data Parallelism:** Sharding the data and model replicas across TPU cores.
    *   **Gradient Synchronization:** Efficiently aggregating gradients from all cores during the backward pass using `xm.optimizer_step`.
    *   **Collective Operations:** Providing functions for synchronized actions like saving checkpoints (`xm.save`) or barrier synchronization (`xm.rendezvous`).
*   **Optimized Data Loading:** `train_tpu.py` uses `torch_xla.distributed.parallel_loader.MpDeviceLoader`. This specialized data loader efficiently transfers data batches to the appropriate TPU cores, minimizing data transfer bottlenecks.

By leveraging PyTorch/XLA, `train_tpu.py` abstracts away many of the low-level complexities of TPU programming, allowing you to focus on your model and training configuration.

---

## Best Practices for Pre-training with Hybrid Attention

Achieving optimal results when pre-training large language models with the hybrid attention architecture requires careful attention to various aspects of the process. Here are some best practices specific to the RNope-SWA architecture:

*   **Data Quality and Diversity**:
    *   **Foundation First**: The performance, capabilities, and biases of your model are overwhelmingly determined by your training data. "Garbage in, garbage out" very much applies.
    *   **Prioritize Quality**: Use high-quality, well-cleaned, and diverse datasets. Invest time in preprocessing your data, which can include deduplication, filtering inappropriate content, and ensuring consistent formatting.
    *   **Long-Context Advantage**: With the hybrid architecture's ability to handle extended sequences efficiently, include datasets with longer documents to fully leverage the model's capabilities.
    *   **Diversity Matters**: A model trained on diverse text (e.g., web crawls, books, code, scientific papers, conversational text) will generally be more robust and adaptable.

*   **Hybrid Architecture Specific Considerations**:
    *   **Sliding Window Size**: The `sliding_window_size` parameter is crucial. A value of 4096 is a good starting point, but you may experiment with values like 2048 or 8192 depending on your use case and available memory.
    *   **Layer Distribution**: The alternating pattern of RoPE+SWA and NoPE layers is automatically handled by the architecture. The balance provides optimal local and global processing.
    *   **Context Length**: Take advantage of the architecture's ability to handle long contexts by setting `max_seq_len` to values like 32768 or even higher if your hardware permits.

*   **Tokenizer Choice and Configuration**:
    *   **Domain Alignment**: Train a tokenizer that is appropriate for your target domain(s) and language(s). A well-suited tokenizer can significantly impact performance and convergence.
    *   **Vocabulary Size**: Remember that the `model.vocab_size` parameter in your `train_config.yaml` must exactly match the vocabulary size of your trained tokenizer.

*   **Choosing Model Size (`d_model`, `n_layers`, `n_heads`)**:
    *   **Memory Considerations**: The hybrid architecture is more memory-efficient for long contexts, but still consider your available compute resources.
    *   **Iterate and Scale**: Start with smaller model configurations to debug your pipeline, then scale up once you have a stable setup.

*   **Hyperparameter Tuning**:
    *   **Start with Defaults**: The provided `train_config.yaml` offers sensible starting points optimized for the hybrid architecture.
    *   **Batch Size Adjustment**: Long-context training may require smaller batch sizes due to memory constraints. Use `gradient_accumulation_steps` to maintain effective batch size.
    *   **Learning Rate**: The hybrid architecture may benefit from slightly different learning rate schedules. Monitor training carefully and adjust as needed.

*   **Monitoring Training Effectively**:
    *   **Use W&B**: Essential for tracking the complex dynamics of hybrid attention training.
    *   **Watch Memory Usage**: Monitor GPU memory usage closely, especially with long contexts.
    *   **Attention Patterns**: If possible, occasionally visualize attention patterns to ensure both local and global layers are functioning as expected.

*   **Understanding the Learning Rate Schedule**:
    *   **Extended Warmup**: The hybrid architecture may benefit from longer warmup periods to allow both attention types to stabilize.
    *   **Gradual Context Scaling**: Consider starting with shorter sequences and gradually increasing context length during training.

*   **Leveraging Gradient Accumulation**:
    *   **Essential for Long Contexts**: With extended sequences, gradient accumulation becomes crucial for maintaining effective batch sizes.
    *   **Memory Management**: Use gradient checkpointing aggressively to handle the memory requirements of long sequences.

*   **Regular Checkpointing**:
    *   **Critical for Long Training**: With the computational investment required for long-context training, frequent checkpointing is essential.

*   **Evaluation Strategy**:
    *   **Long-Context Benchmarks**: Evaluate on tasks that specifically test long-context understanding to validate the hybrid architecture's benefits.
    *   **Both Local and Global Tasks**: Test performance on both local coherence tasks and global reasoning tasks to ensure balanced capability.

*   **Start Small, Iterate Fast**:
    *   **Architecture Validation**: Before large-scale training, validate that the hybrid attention is working correctly with smaller experiments.
    *   **Context Length Scaling**: Start with moderate context lengths and gradually increase to find the optimal balance for your use case.

---

## Limitations

While Lunaris Codex with hybrid attention is designed to be a robust toolkit for pre-training advanced language models, it's important to understand its current scope and limitations:

*   **Experimental Nature**:
    *   The RNope-SWA hybrid attention architecture is cutting-edge and experimental. While promising, it may require additional tuning and optimization for specific use cases.
    *   Long-term stability and performance characteristics are still being evaluated across different domains and scales.

*   **Focus on Pre-training**:
    *   Lunaris Codex is primarily engineered for pre-training large language models from scratch. Its core script (`train.py`) and design philosophy are centered around this goal.
    *   The toolkit does not include built-in scripts or extensive, dedicated support for fine-tuning models on specific downstream tasks. Users wishing to fine-tune the pre-trained models will need to adapt the model code or integrate it with other frameworks.

*   **Data Preparation and Tokenization**:
    *   The crucial steps of data sourcing, cleaning, extensive preprocessing, and training a tokenizer are external to the `train.py` script.
    *   Users are responsible for these preliminary stages. The quality, diversity, and appropriateness of the data and tokenizer will significantly influence the final model's performance.

*   **Evaluation Beyond Pre-training Metrics**:
    *   The `train.py` script provides essential training metrics such as loss and perplexity, which are vital for monitoring the pre-training process.
    *   However, it does not have integrated evaluation pipelines for standard academic NLP benchmarks or long-context specific evaluations. Users will need to develop their own evaluation workflows.

*   **Resource Requirements**:
    *   Training capable large language models with extended context is inherently computationally intensive. While the hybrid architecture is more efficient than traditional full attention, it still demands significant resources:
        *   **High VRAM GPUs**: For larger models, longer contexts, and reasonable batch sizes.
        *   **Substantial Storage**: For datasets, tokenizers, and checkpoints.
        *   **Considerable Training Time**: Potentially days or weeks, depending on the model scale and context length.

*   **Advanced Parallelism and Features**:
    *   Currently, Lunaris Codex supports Distributed Data Parallel (DDP) for multi-GPU training.
    *   More advanced parallelism strategies (e.g., Fully Sharded Data Parallel - FSDP, Tensor Parallelism, Pipeline Parallelism) are not yet implemented but could be valuable for scaling the hybrid architecture.

---

## License & Community

This project is licensed under the **MIT License**.

Developed by **Francisco Antonio** ([@MeryylleA](https://github.com/MeryylleA) on GitHub).

Join our community on Discord for discussions, help, and to share your results with the hybrid attention architecture: [**Moon Cloud Services**](https://discord.gg/JNsfzEwMtC)

### Special Thanks
*   To Andrej Karpathy for `nanoGPT`, which served as the inspirational and architectural starting point for this project.
*   To the authors of "Rope to Nope and Back Again" for the theoretical foundation of the hybrid attention architecture.
*   To the open-source AI community for their invaluable tools, research, and datasets.
*   To **Google Gemini** for extensive pair-programming sessions, architectural discussions, debugging, and documentation assistance.
*   To **Lambda Labs** for supporting this project with the large-scale compute necessary for this research.
