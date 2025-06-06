# 🚧 Lunaris Codex — [Under Maintenance]

> **Please note:** This repository is undergoing a complete redesign.

We are preparing a new phase of the project with fundamental improvements to the pipeline, datasets, and code structure.

## What's coming:

- 🧰 **Retirement of `prepare_data.py`**
The script will be deprecated and officially replaced by AllenAI's **Dolma Toolkit**, ensuring more powerful and scalable data collection and filtering.

- 🧠 **Improvements to `train.py`**
Internal updates to the training script, including:
- Native support for `.npy` files (replacing `.memmap`)
- New validations, fixes, and efficiency improvements
- Preparation for scaling larger models with greater stability

- 📦 **New SFT (Supervised Fine-Tuning) datasets**
Soon, refined SFT datasets will be available with:
- Very high-quality content
- Diversity of technical details
- Fully in English
- Inspired by the best practices of open source projects such as Mistral, OpenHermes, Zephyr, etc.

- 📚 **Documentation Rewrite**
The new README will include a complete guide:
- How to prepare your data with Dolma
- Training with `train.py`
- Details on LoRA, generation, architecture, and more

## Training Infrastructure

Pre-training of the next generation of Lunaris is being performed on an **NVIDIA GH200**
thanks to the support of [Lambda Cloud](https://lambda.ai/) through the **Lambda Research Grant Program**.
Thanks for your support — it is making this project possible.

---
Watch the progress soon in this repository.

In the meantime, feel free to explore the previous commits or get in touch.

— Francisco (Lunaris Creator Codex)
