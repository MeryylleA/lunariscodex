# Lunaris Codex

> **Note:** You are viewing an experimental branch of Lunaris Codex. This version includes state-of-the-art architectural features for enhanced performance and training efficiency. These features are currently under evaluation.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Discord](https://img.shields.io/discord/1138864753915854898?label=Discord&logo=discord&color=7289DA)](https://discord.gg/JNsfzEwMtC)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/MeryylleA/lunariscodex)

A Note on Our Foundation: The architectural foundation of Lunaris Codex is proudly built upon Andrej Karpathy's nanoGPT. We chose nanoGPT for its brilliant simplicity and clarity, which aligns perfectly with our philosophy of providing a "hackable" and understandable base. This version, however, represents a significant evolution, integrating modern enhancements like **RoPE, Grouped Query Attention (GQA), Fused SwiGLU, and Gradient Checkpointing** to push performance and capabilities far beyond the original.

**Lunaris Codex** is a streamlined, high-performance toolkit for pre-training powerful language models from scratch. This project provides a modern, Llama-style Transformer architecture and a robust, heavily optimized training script, designed for stability and maximum throughput.

### Our Philosophy
This repository is built on a simple, powerful idea: **provide a rock-solid, understandable foundation for creating strong base models.** We focus on clean, efficient, and well-documented code for the core tasks of model definition and training. This approach empowers researchers and developers to bring their own unique datasets to a proven, production-grade pipeline.

This new version marks a significant evolution of the project, moving to an architecture inspired by industry-leading models like Llama and Mistral, with a strong emphasis on modern best practices.

---

## Architecture Overview

Lunaris Codex is engineered for a balance of performance and clarity. Its architecture integrates several state-of-the-art features to ensure top-tier performance and training efficiency.

| Component | Implementation | Benefits & Considerations |
| :--- | :--- | :--- |
| **Normalization** | **RMSNorm with Learnable Bias** | **Benefits:** More expressive than standard RMSNorm. The learnable bias allows the model to learn an optimal activation mean, enhancing stability by preventing mean-shift in deep networks. |
| **Positional Info**| **RoPE (Rotary Positional Embeddings)** | **Benefits:** Injects relative positional information, leading to excellent generalization across various sequence lengths. Achieved without any learned parameters, making it efficient. |
| **Attention** | **Grouped Query Attention (GQA)** | **Benefits:** Drastically reduces the memory usage of the KV cache during inference by sharing Key/Value heads across groups of Query heads. This enables faster generation and larger context windows. |
| **FFN Activation**| **SwiGLU with Fused Projection** | **Benefits:** Offers improved performance over traditional activations. The gate and up-projections are fused into a single linear layer, improving GPU performance by reducing kernel overhead. |
| **Training** | **Gradient Checkpointing** | **Benefits:** Massively reduces VRAM usage during training by recomputing activations during the backward pass instead of storing them. This allows for training larger models or using larger batch sizes, at the cost of a small compute overhead. |
| **Structure** | **Pre-LayerNorm Decoder-Only Transformer** | **Benefits:** A standard, proven architecture for autoregressive language modeling. Pre-LayerNorm (applying normalization before attention/FFN) enhances training stability, especially for deep networks. |
| **Embeddings** | **Tied Input/Output Token Embeddings** | **Benefits:** Significantly reduces the model's parameter count by sharing weights between the token embedding layer and the final output layer. Can also improve model quality and training efficiency. |

---

## The Training Pipeline

Our `train.py` script is a feature-rich and resilient trainer, meticulously engineered to handle large-scale, long-running jobs with stability and efficiency.

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
*   **Comprehensive Monitoring:** Integrates with Weights & Biases (W&B) for detailed experiment tracking (loss, perplexity, learning rate, etc.) and provides informative console logging with progress bars via `tqdm`.

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
Create a `train_config.yaml` file. This is where you define your model architecture, hyperparameters, and data paths. Below is a well-commented example configuration that shows how to use the new features.

```yaml
# train_config.yaml

# --- Model Configuration ---
# These parameters define the architecture of your LunarisCodex model.
model:
  vocab_size: 50304      # IMPORTANT: Must exactly match your tokenizer's vocabulary size.
  d_model: 1024          # Dimensionality of the model embeddings and hidden states.
  n_layers: 20           # Number of transformer blocks (layers).
  n_heads: 16            # Number of attention heads (must divide d_model).
  max_seq_len: 1024     # Maximum sequence length the model can process.
  dropout: 0.0           # Dropout rate for regularization (0.0 to disable).

  # --- Advanced Architectural Features ---
  n_kv_heads: 4          # Set to a value less than n_heads to enable Grouped Query Attention (GQA).
                         # `n_heads` must be divisible by `n_kv_heads`. e.g., 16 heads, 4 kv_heads.
                         # Omit this or set to null to default to standard Multi-Head Attention.
  
  use_gradient_checkpointing: true # Set to true to enable gradient checkpointing.
                                   # This saves a large amount of GPU memory at the cost of
                                   # slightly slower training iterations. Highly recommended for
                                   # large models or large batch sizes.

# --- Data Configuration ---
data_dir: "path/to/your/npy_shards/" # IMPORTANT: Point this to your directory of sharded .npy files.

# --- Optimizer Configuration ---
learning_rate: 2.0e-4    # Peak learning rate. Tune this carefully.
weight_decay: 0.1        # Weight decay for AdamW optimizer.

# --- Scheduler Configuration ---
warmup_steps: 2000       # Number of steps for linear learning rate warmup.
max_steps: 200000        # Total number of training steps.

# --- Training Configuration ---
batch_size: 32           # Per-GPU batch size. Adjust based on your VRAM.
gradient_accumulation_steps: 4 # Accumulates gradients over N micro-batches.
                         # Effective batch size = batch_size * num_gpus * gradient_accumulation_steps.
grad_clip: 1.0           # Gradient clipping value to prevent exploding gradients.
compile_model: true      # Whether to use torch.compile for potential speedups.

# --- I/O and Logging ---
out_dir: "checkpoints/my-model-run"   # Directory to save checkpoints.
save_interval: 1000      # Save a checkpoint every N steps.
log_interval: 20         # Log metrics to console/W&B every N steps.

# --- W&B (Weights & Biases) Configuration (Optional) ---
wandb_project: null                # Your W&B project name (e.g., "MyLunarisProject"). Keep as null or remove to disable.
wandb_run_name: "experiment-001"   # A name for this specific run (e.g., "gqa-llama-run1").
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

Good luck, and we're excited to see what you build with Lunaris Codex!

---

## Best Practices for Pre-training

Achieving optimal results when pre-training large language models requires careful attention to various aspects of the process. Here are some best practices to consider when using Lunaris Codex:

*   **Data Quality and Diversity**:
    *   **Foundation First**: The performance, capabilities, and biases of your model are overwhelmingly determined by your training data. "Garbage in, garbage out" very much applies.
    *   **Prioritize Quality**: Use high-quality, well-cleaned, and diverse datasets. Invest time in preprocessing your data, which can include deduplication, filtering inappropriate content, and ensuring consistent formatting.
    *   **Diversity Matters**: A model trained on diverse text (e.g., web crawls, books, code, scientific papers, conversational text) will generally be more robust and adaptable.

*   **Tokenizer Choice and Configuration**:
    *   **Domain Alignment**: Train a tokenizer that is appropriate for your target domain(s) and language(s). A well-suited tokenizer can significantly impact performance and convergence.
    *   **Vocabulary Size**: Remember that the `model.vocab_size` parameter in your `train_config.yaml` must exactly match the vocabulary size of your trained tokenizer.

*   **Choosing Model Size (`d_model`, `n_layers`, `n_heads`)**:
    *   **Resource Balancing**: Select model parameters based on your available compute resources (GPU VRAM, total training time budget). Larger models require more memory and take longer to train.
    *   **Iterate and Scale**: It's often wise to start with smaller model configurations (e.g., fewer layers/heads, smaller `d_model`) to debug your entire pipeline, ensure data loading is correct, and iterate on hyperparameters quickly. Once you have a stable setup, you can scale up.

*   **Hyperparameter Tuning**:
    *   **Start with Defaults**: The `train_config.yaml` provides sensible starting points. However, optimal hyperparameters are often dataset-dependent.
    *   **Key Parameters**: If you're tuning, `learning_rate` is often the most critical. `batch_size` (and `gradient_accumulation_steps`) and `warmup_steps` are also commonly adjusted.
    *   **Systematic Approach**: Tune one or two hyperparameters at a time to understand their impact.

*   **Monitoring Training Effectively**:
    *   **Use W&B**: We strongly recommend enabling Weights & Biases logging (`wandb_project` in `train_config.yaml`). It provides invaluable insights into loss curves, learning rate schedules, gradient norms, hardware utilization, and more, helping you track progress and spot potential issues early.
    *   **Console Logs**: The `tqdm` progress bar provides immediate console feedback on loss and learning rate.
    *   **Watch for Issues**: Regularly check for signs of overfitting (training loss decreases but validation/downstream performance stagnates or degrades) or underfitting (training loss remains high).

*   **Understanding the Learning Rate Schedule**:
    *   **Warmup Phase**: The initial warmup period (controlled by `warmup_steps`) gradually increases the learning rate. This helps stabilize training in the early stages when weights are changing rapidly.
    *   **Decay Phase**: After warmup, the learning rate typically decays (cosine decay in this script) over the `max_steps`. This allows for finer adjustments as the model converges.
    *   **Setting `max_steps`**: Determine `max_steps` based on your dataset size and the number of epochs you want to train for (or total tokens you want the model to see). One epoch means the model has seen the entire dataset once.

*   **Leveraging Gradient Accumulation**:
    *   **Simulating Larger Batches**: If your per-GPU `batch_size` is limited by VRAM, `gradient_accumulation_steps` allows you to simulate a larger effective batch size (`batch_size * num_gpus * gradient_accumulation_steps`). This can improve training stability and performance.

*   **Regular Checkpointing**:
    *   **Don't Lose Progress**: For long training runs, frequent checkpointing (via `save_interval`) is crucial. It ensures that you can resume training with minimal loss of work in case of interruptions.

*   **Evaluation Strategy**:
    *   **Beyond Pre-training Loss**: While `train.py` focuses on minimizing the pre-training loss, the ultimate measure of a language model is its performance on downstream tasks.
    *   **Held-out Sets**: Maintain held-out validation datasets (not seen during training) to periodically evaluate your model's generalization. Consider using academic benchmarks relevant to your goals.

*   **Start Small, Iterate Fast**:
    *   **Debug Pipeline First**: Before launching a multi-day or multi-week training run on a massive dataset with a large model, conduct smaller experiments. Use a fraction of your data and a smaller model to verify your data loading, tokenization, training loop, and logging are all working correctly. This can save significant time and resources.

---

## Limitations

While Lunaris Codex is designed to be a robust toolkit for pre-training language models, it's important to understand its current scope and limitations:

*   **Focus on Pre-training**:
    *   Lunaris Codex is primarily engineered for pre-training large language models from scratch. Its core script (`train.py`) and design philosophy are centered around this goal.
    *   The toolkit does not include built-in scripts or extensive, dedicated support for fine-tuning models on specific downstream tasks (e.g., instruction tuning, sequence classification, question answering). Users wishing to fine-tune the pre-trained models will need to adapt the model code or integrate it with other frameworks (like Hugging Face Transformers).

*   **Data Preparation and Tokenization**:
    *   The crucial steps of data sourcing, cleaning, extensive preprocessing (beyond basic tokenization), and training a tokenizer are external to the `train.py` script.
    *   Users are responsible for these preliminary stages. The quality, diversity, and appropriateness of the data and tokenizer will significantly influence the final model's performance, and Lunaris Codex relies on the user to manage these aspects effectively.

*   **Evaluation Beyond Pre-training Metrics**:
    *   The `train.py` script provides essential training metrics such as loss and perplexity, which are vital for monitoring the pre-training process.
    *   However, it does not have integrated evaluation pipelines for standard academic NLP benchmarks (e.g., GLUE, SuperGLUE, MMLU) or support for custom downstream task evaluation. Users will need to develop their own evaluation workflows to assess model performance beyond pre-training.

*   **Resource Requirements**:
    *   Training capable large language models is inherently computationally intensive. While Lunaris Codex is optimized for efficiency, pre-training still demands significant resources, including:
        *   **High VRAM GPUs**: For larger models and batch sizes.
        *   **Substantial Storage**: For datasets, tokenizers, and checkpoints.
        *   **Considerable Training Time**: Potentially days or weeks, depending on the model scale and dataset size.
    *   Users should be prepared for these hardware and time commitments when undertaking ambitious pre-training projects.

*   **Advanced Parallelism and Features**:
    *   Currently, Lunaris Codex supports Distributed Data Parallel (DDP) for multi-GPU training.
    *   More advanced parallelism strategies (e.g., Fully Sharded Data Parallel - FSDP, Tensor Parallelism, Pipeline Parallelism) or features common in some large, established frameworks (like a dedicated model hub or extensive hyperparameter optimization suites) are not yet implemented. These could be areas for future development or community contributions.

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
