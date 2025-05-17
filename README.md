---

 🌙 Lunaris Codex

**Lunaris Codex** is a highly flexible and customizable Transformer Decoder architecture designed for code generation and language modeling. Written entirely in PyTorch, it features modern optimizations like **LoRA**, optional **FlashAttention**, **ALiBi** positional biases, and a comprehensive, configurable training and data preprocessing pipeline. This repository provides the full source code, enabling users to train their own decoder-only LLMs from scratch or fine-tune existing ones.

Our goal is to provide a clean, understandable, and powerful codebase that serves as an excellent starting point for researchers, students, and developers interested in building, training, and experimenting with state-of-the-art language models.

> ⚠️ **Note:** This project focuses on providing a robust and well-tested architecture and a complete training/data-processing toolkit. While the ambition for large-scale pretrained weights exists (targeting NVIDIA H100/GH200 class hardware), the current release empowers you to train models of various sizes on your custom datasets today. It's an ideal platform for learning, research, and building specialized models.

---

## ✨ Key Features

- ⚙️ **Decoder-only Transformer Architecture (`model.py`):**
    - Optimized for code and natural language tasks.
    - Highly configurable: number of layers, hidden dimensions, attention heads, activation functions, etc.
    - Implements modern components like SwiGLU activations, Pre-LayerNorm, and LayerScale for training stability.
    - Tied input embedding and output language modeling head for parameter efficiency.
- 🚀 **LoRA (Low-Rank Adaptation):**
    - Built-in support via a custom `LoRALinear` layer.
    - Easily apply efficient fine-tuning to model projection layers.
    - Configurable LoRA rank, toggleable via training arguments.
- ⚡ **Optional FlashAttention:**
    - Integration with `flash-attn` library for significant speedups and memory savings on compatible NVIDIA GPUs.
    - Graceful fallback to a PyTorch-native manual attention implementation if FlashAttention is unavailable or when running on CPU, ensuring broad compatibility.
- 🧠 **ALiBi (Attention with Linear Biases):**
    - Implemented for improved long-context handling and extrapolation capabilities, eliminating the need for traditional absolute positional embeddings.
    - Correctly integrated with both FlashAttention (causal mode) and the manual attention fallback.
- 💾 **Advanced Data Preprocessing (`prepare_data.py`):**
    - Full configuration via `argparse` for data sources, tokenizers, and all processing steps.
    - **Seamlessly processes structured datasets from Hugging Face Hub**, including the companion [Lunaris-Data dataset](https://huggingface.co/datasets/meryyllebr543/lunaris-data), allowing specification of input/target columns and custom formatting templates.
    - Supports loading diverse local text file formats: one example per line, chunking of large files, and glob patterns for ingesting multiple files.
    - Flexible tokenizer loading: use any tokenizer from Hugging Face Hub by name or provide a local path (supports SentencePiece `.model` files and standard `tokenizer.json`).
    - Automatic and configurable `pad_token_id` handling.
    - Efficiently saves tokenized, truncated, and padded data as memory-mapped NumPy files (`.memmap`) for lightning-fast loading during training.
- 📊 **Configurable Training Pipeline (`train.py`):**
    - Comprehensive command-line interface (`argparse`) for all training hyperparameters and operational settings.
    - Supports training from scratch and resuming from saved checkpoints.
    - AdamW optimizer with optional `fused` mode for CUDA.
    - Gradient clipping and Learning Rate Schedulers (e.g., `ReduceLROnPlateau`).
    - Mixed Precision (AMP) support (`fp16` or `bf16`) for CUDA devices.
    - Robust checkpointing strategy: save by `epoch` or `steps`, including model weights, optimizer/scheduler states, training arguments, and model configuration. Automatically saves the "best model" based on validation loss. Checkpoints use PyTorch's `.pt` format for broad compatibility.
    - Detailed logging of metrics: loss, perplexity, and top-1 accuracy for both training and validation sets, with `tqdm` progress bars.
    - Optional `torch.compile` support for further optimization (best results on newer PyTorch versions and GPUs).
- 📏 **Scalable by Design:**
    - Example default configuration for ~183M parameters.
    - Successfully tested a ~3M parameter "toy" model trained end-to-end on CPU, demonstrating the full pipeline functionality. The architecture can be easily scaled up or down by adjusting configuration parameters.

---

## 🧱 Architecture Overview

Lunaris Codex implements a standard, yet powerful, decoder-only Transformer architecture:

- **Transformer Blocks:** A stack of `n_layers` identical decoder blocks.
- **Self-Attention Module:**
    - Multi-Head Attention mechanism.
    - **ALiBi** is integrated by adding biases to the attention scores before softmax.
    - Optional FlashAttention for CUDA. If not used, a PyTorch manual attention loop is employed to ensure ALiBi and padding masks are correctly applied.
    - LoRA can be applied to QKV and output projection layers.
- **FeedForward Network (FFN):**
    - Position-wise FFN with configurable intermediate dimension and activation (SwiGLU or GELU).
    - LoRA can be applied to the linear layers within the FFN.
- **Normalization:** Pre-LayerNorm topology (normalization before attention/FFN sub-layers).
- **Stability:** LayerScale applied to the outputs of attention and FFN sub-layers before a residual dropout.
- **Embeddings:** Learns token embeddings. The same embedding matrix is tied to the final language modeling head.

> The design prioritizes a balance of cutting-edge features, performance, code clarity, and configurability, making Lunaris Codex an excellent foundation for various NLP tasks.

---

## 📚 Datasets & Preprocessing

Lunaris Codex is designed to be trained on diverse datasets. While you can use any text or code corpus, we are also proud to introduce **Lunaris-Data**, a high-quality dataset specifically curated for training advanced code generation models.

### 💎 Featured Dataset: [Lunaris-Data](https://huggingface.co/datasets/meryyllebr543/lunaris-data)

- **Curated Content:** Contains **74,000+ meticulously engineered examples** ideal for code synthesis, complex problem-solving, debugging, and system design tasks.
- **Generation:** Data was generated using a custom pipeline leveraging multiple advanced AI models: DeepSeek V3, GPT-4o Mini, and Codestral-25.01.
- **Rich Structure:** Each example includes `input` (detailed prompt), `output` (comprehensive solution with code and explanations, ~700-1200 tokens), `code` (extracted code snippet), `language`, and (for some examples) `context` fields.
- **Accessibility:** Hosted on the Hugging Face Hub in efficient Parquet format, organized into 75 batch files.
- **More Information:** Please visit the [**Lunaris-Data dataset card on Hugging Face**](https://huggingface.co/datasets/meryyllebr543/lunaris-data) for the full description, data structure, and usage guidelines.

### 1. Data Preparation (`prepare_data.py`)

This script is your first step. It takes raw text/code data from various sources, tokenizes it using your chosen tokenizer, and saves it into an efficient memory-mapped NumPy file (`.memmap`) format, ready for fast loading by `train.py`.

**Key Capabilities & Arguments:**
- **Data Sources (`--data_source_type`):**
    - `hf_dataset`: Load from Hugging Face Hub.
        - `--dataset_name_or_path`: e.g., `meryyllebr543/lunaris-data`.
        - `--hf_dataset_data_dir`: Subdirectory in the HF repo (e.g., `data` for Lunaris-Data, as its Parquet files are in `username/datasetname/data/`).
        - `--hf_input_column`, `--hf_target_column`, `--hf_formatting_template`: For structured datasets, define how to combine columns into a single training example (e.g., template `INSTRUCTION: {input}\nASSISTANT: {target}`).
        - `--hf_single_content_column`: For datasets with one main text column (e.g., `text`).
    - `text_file_lines`: Load from local text files, one example per line. Supports glob patterns (e.g., `"./my_code/**/*.py"`).
    - `text_file_chunks`: Load a single large local text file and split it into fixed-length token chunks.
- **Tokenizer (`--tokenizer_name_or_path`):** Specify a Hugging Face tokenizer name or a path to a local tokenizer.
- **Processing:**
    - `--max_length`: Target sequence length for tokenization and output memmap.
    - `--output_path`: Destination for the `.memmap` file.
    - `--add_special_tokens`: (Flag) Instructs the tokenizer to add BOS/EOS tokens if its configuration supports it.
    - `--max_examples`: (Optional) Limit the number of examples to process, useful for quick tests.
    - `--output_dtype`: `int16` or `int32` for token storage, to balance disk space and vocabulary size.

**Example: Preparing a 1k sample of Lunaris-Data for Training:**
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
    --output_path ./processed_data/lunaris_data_sample_1k.memmap \
    --max_examples 1000 
```

---

## 🧢 Training (`train.py`)

Once your data is prepared as a `.memmap` file, `train.py` handles the model training process. It's highly configurable via command-line arguments.

**Key Arguments (Selected - see `python train.py --help` for all):**
- **Dataset:** `--memmap_file_train`, `--num_sequences_train`, `--memmap_file_val`, `--num_sequences_val`, `--dataset_max_length`, `--dataset_dtype`.
- **Tokenizer:** `--tokenizer_name_or_path` (must match data preparation).
- **Model Architecture:** `--d_model`, `--n_layers`, `--n_heads`, `--model_max_seq_len`, `--lora_rank` (set to 0 for full training/fine-tuning).
- **Training Loop:** `--batch_size`, `--num_epochs`, `--learning_rate`, `--weight_decay`, `--grad_clip_norm`.
- **Operational:** `--device` (`cuda` or `cpu`), `--checkpoint_dir`, `--resume_from_checkpoint`, `--save_strategy` (`epoch` or `steps`), `--save_steps`, `--log_interval`.
- **Optimizations:** `--mixed_precision_dtype` (`fp16` or `bf16`), `--adam_fused`, `--use_torch_compile`, `--allow_tf32`, `--cudnn_benchmark`.

**Example: Training a small test model on CPU using the prepared Lunaris-Data sample:**
```bash
python train.py \
    --memmap_file_train ./processed_data/lunaris_data_sample_1k.memmap \
    --num_sequences_train 1000 \
    # Optional validation set:
    # --memmap_file_val ./processed_data/lunaris_data_val_sample.memmap \
    # --num_sequences_val 100 \
    --tokenizer_name_or_path bigcode/starcoder \
    --dataset_max_length 1024 \
    --dataset_dtype int32 \
    --model_max_seq_len 1024 \
    --d_model 256 \
    --n_layers 4 \
    --n_heads 4 \
    --batch_size 4 \
    --num_epochs 1 \
    --lora_rank 8 \
    --device cpu \
    --checkpoint_dir ./checkpoints_lunaris_data_cpu \
    --log_interval 10 
```

---

## 🧠 What's Included (Current Status `v0.2.0`)

| Component                                     | Status                                   | Notes                                                                 |
|-----------------------------------------------|------------------------------------------|-----------------------------------------------------------------------|
| **Core Model Architecture (`model.py`)**        | ✅ Released                              | Decoder-only, ALiBi, LayerScale, SwiGLU, Configurable                 |
| **LoRA Integration**                          | ✅ Released                              | `LoRALinear` class, integrated into Attention & FFN                   |
| **FlashAttention Support (Optional)**         | ✅ Released                              | With robust PyTorch manual attention fallback (ALiBi+Padding aware)  |
| **Data Preprocessing Pipeline (`prepare_data.py`)** | ✅ Released (Major Update)             | Supports HF Hub (structured/unstructured), local files (lines/chunks/glob), `argparse` |
| **Training Pipeline (`train.py`)**            | ✅ Released (Major Update)             | `argparse`, full checkpointing, LoRA/Full-FT, AMP, LR scheduler, validation, CPU/GPU |
| Example Dataset ([Lunaris-Data](https://huggingface.co/datasets/meryyllebr543/lunaris-data)) | ✅ Publicly Available on HF Hub         | 74k+ high-quality examples for code/tech tasks                     |
| Inference System (`inference.py`)             | ⚠️ Needs Update/Testing                   | Original `inference.py` requires updates for new model config & checkpointing.  |
| Pretrained Weights                            | ❌ Not Yet Released (Future Goal)        | Current focus is providing a solid training framework.               |

---

## 🗺️ Roadmap (Revised)

- 🧪 **Refine & Test `inference.py`:** Update the inference script to seamlessly load trained checkpoints (both full and LoRA-adapted) and provide flexible generation capabilities.
- 📖 **Comprehensive Documentation & Tutorials:**
    - Detailed API documentation for all modules and classes.
    - Tutorials:
        - Training Lunaris Codex from scratch on a custom dataset.
        - Fine-tuning Lunaris Codex (or other compatible models) using LoRA with Lunaris-Data.
        - Using `prepare_data.py` with various data sources.
- 🚀 **Example Training Runs & Configurations:** Provide example scripts/configs for training small-to-medium models on publicly available datasets (e.g., a subset of The Stack, TinyStories, or a sample of Lunaris-Data).
- 📊 **Benchmarking:** Establish baseline performance metrics (training speed, memory usage, perplexity on standard benchmarks) for different model sizes and hardware (CPU, and GPU when accessible).
- ✨ **Explore Further Optimizations:**
    - **Gradient Checkpointing:** Integrate for training larger models with limited VRAM.
    - **Advanced Schedulers & Optimizers.**
    - (Long-term) Explore distributed training (DeepSpeed/FSDP) if scaling to very large models.
- 🤗 **Hugging Face Hub Integration:**
    - Publish the core model architecture to the Hub for easy use with `AutoModel`.
    - Create a detailed model card for Lunaris Codex.
    - (Future) If small, useful checkpoints are trained, host them on the Hub.
- 🛠️ **(Ambitious)** Release pretrained base models if/when large-scale computational resources for training on 1B+ tokens (as originally envisioned for H100s) become available.

---

## 📦 Installation & Setup

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/MeryylleA/lunaris-codex.git 
    cd lunaris-codex
    ```
2.  **Create a Python Virtual Environment:**
    It's highly recommended to use a virtual environment (e.g., with Python 3.10, 3.11).
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # For Linux/macOS with bash/zsh
    # For fish shell: source .venv/bin/activate.fish
    # For Windows: .venv\Scripts\activate
    ```
3.  **Install Dependencies:**
    Upgrade pip and install from `requirements.txt`.
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

**Core Dependencies (ensure these are in `requirements.txt`):**
- `torch>=2.0` (PyTorch 2.0+ is recommended for `torch.compile` and `F.scaled_dot_product_attention`)
- `transformers` (for tokenizers and potentially loading other models)
- `datasets` (for Hugging Face dataset loading utilities)
- `numpy`
- `tqdm` (for progress bars)
- `sentencepiece` (required by many tokenizers, including StarCoder's if not using the HF wrapper directly for its file)
- `safetensors` (though current training checkpoints use PyTorch's `.pt` format, it's good for model weight distribution)

**Optional for NVIDIA GPU Acceleration:**
- `flash-attn`: For significantly faster attention. Install separately if you have a compatible NVIDIA GPU and CUDA setup (usually requires compilation):
  ```bash
  # Example, check flash-attn GitHub for a specific version compatible with your PyTorch/CUDA
  pip install flash-attn --no-build-isolation 
  ```

---

## 📝 License

This project is licensed under the **MIT License**.  
Copyright (c) 2024-2025 **Francisco Antonio** 

See [`LICENSE`](LICENSE) for more details.

---

## 🌟 Credits & Contribution

Developed by **Francisco Antonio** ([@MeryylleA](https://github.com/MeryylleA) on GitHub, [@Meryylle](https://x.com/a93918) on X/Twitter).

This project is a labor of love, aiming to demystify and provide accessible tools for building and training powerful language models. 

**Contributions are highly welcome!** Whether it's bug fixes, new features, documentation improvements, or sharing your training experiences and results, please feel free to open an issue or a pull request.

Special thanks to the open-source AI community and to **Gemini** for extensive pair-programming, architectural discussions, and debugging support throughout the recent development sprint.

> For collaborations, specific feature requests, or research interests, please connect via GitHub issues/discussions or X/Twitter. Let's build something amazing together!

---

## 🌌 Why "Lunaris"?

> *"Because great ideas are born in silence — and shine like the moon."* 🌙

---
