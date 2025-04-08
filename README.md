# 🌙 Lunaris Codex

**Lunaris Codex** is a custom Transformer Decoder architecture designed for code generation, written entirely in PyTorch with support for modern optimizations like **LoRA**, **FlashAttention**, and **ALiBi**. This repository includes the full source code for training and model definition — designed to be lightweight, scalable, and GPU-efficient.

> ⚠️ **Note:** This release includes only the architecture and training system. Pretrained weights will be released in the future, trained on a massive-scale dataset using NVIDIA **H100 SXM** and **GH200** GPUs.

---

## ✨ Features

- ⚙️ **Decoder-only Transformer**, optimized for code and natural language
- 🚀 **LoRA support** (Low-Rank Adaptation) for efficient fine-tuning
- ⚡ **FlashAttention integration** for fast attention on modern GPUs
- 🧠 **ALiBi** positional bias for long-context support
- 📏 ~183M parameters with default config (can be scaled easily)
- 🧪 Simple, scalable **training pipeline** with AMP, torch.compile, and safetensors

---

## 🧱 Architecture Overview

Lunaris Codex follows a modern decoder-only Transformer structure with several enhancements:

- **10 transformer blocks** (configurable)
- **768 hidden size**, **12 attention heads**
- **SwigLU activation** in the feedforward blocks
- **ALiBi** is used instead of absolute positional embeddings
- **LoRA** is applied to all projection layers (`qkv`, `output`, and feedforward)
- **FlashAttention** replaces traditional attention computation
- **LayerScale** stabilizes residual connections
- **Final tied embedding head** for efficiency and weight sharing

> All design decisions are optimized for training speed, memory efficiency, and compatibility with large-scale tokenizers like StarCoder.

---

## 🧢 Training

The repository includes a full training loop (`train.py`) compatible with custom datasets. The model uses:

- `torch.cuda.amp` (mixed precision)
- `AdamW` optimizer with fused operations
- Checkpointing using `.safetensors`
- Real-time metrics: **loss**, **perplexity**, and **top-1 accuracy**

The model is designed for training on **multi-GPU** setups with full support for TF32/BF16 performance on **NVIDIA A100**, **H100**, and **GH200**.

---

## 🧠 What's included

| Component                  | Status         |
|---------------------------|----------------|
| Model architecture         | ✅ Released    |
| LoRA integration           | ✅ Released    |
| FlashAttention support     | ✅ Released    |
| ALiBi + LayerScale         | ✅ Released    |
| Training pipeline          | ✅ Released    |
| Dataset preprocessing      | ❌ Private     |
| Inference system (CLI/UI)  | ❌ Private     |
| Pretrained weights         | 🔜 Coming soon |

---

## ⚠️ Roadmap

- 🧬 Release **pretrained checkpoints** trained on ~1B+ tokens of high-quality code
- 📦 Public **inference UI / API**
- 📖 Hugging Face model card & demo
- 🌐 Documentation for integrating with LangChain, Discord bots, or VSCode plugins

---

## 📦 Installation

```bash
pip install -r requirements.txt
```

Dependencies include:
- `torch`
- `transformers`
- `safetensors`
- `numpy`
- `datasets` (for custom extensions)

---

## 📝 License

This project is licensed under the **MIT License**.  
Copyright (c) 2025 **Francisco Antonio**

See [`LICENSE`](LICENSE) for more details.

---

## 🌟 Credits

Developed by [Francisco Antonio](https://github.com/MeryylleA)  
Feel free to star ⭐ the project and share feedback or contributions!

> For collaborations, dataset requests, or research interest: contact via GitHub or [@Meryylle](https://x.com/a93918)

---

## 🌌 Why "Lunaris"?

> *"Because great ideas are born in silence — and shine like the moon."* 🌙
