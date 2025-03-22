# 🧬 Gemma in JAX/Flax

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/gemma_in_jax.ipynb)

This is a **pure JAX/Flax** implementation of **Gemma**, a decoder-only transformer model with **multi-query attention**, built for **TPU/GPU** training and inference. It includes training, evaluation, checkpointing, and text generation — fully from scratch using efficient JAX and Flax building blocks.

---

## 🚀 Highlights

- 🧠 **Decoder-only Transformer** with **Rotary Position Embeddings (RoPE)**, **RMSNorm**, **GeGLU**, and **Multi-Query Attention**.
- ⚡ Supports **TPU Multi-Device Training** using `pmap` and `jit`.
- 🧪 Includes **Tokenizer training** and full dataset preprocessing using Hugging Face Datasets.
- 💾 **Checkpointing & Restore** with Orbax.
- ✍️ **Text Generation** with temperature, top-k, and top-p sampling.
- ✅ **Minimal dependencies** and designed for easy extensibility.

---

## 🧱 Architecture Overview

| Component               | Details                                       |
| ----------------------- | --------------------------------------------- |
| **Attention**           | **Multi-query attention** with single KV head |
| **Positional Encoding** | Rotary embeddings (**RoPE**)                  |
| **MLP**                 | **GeGLU** activation                          |
| **Normalization**       | **RMSNorm**                                   |
| **Optimizer**           | **Optax** with warmup + cosine decay          |
| **Checkpointing**       | **Orbax CheckpointManager**                   |
| **Text Generation**     | Top-k, top-p, temperature sampling            |

---

## 📦 File Structure

- `gemma_in_jax.py` – Full model, training pipeline, generation, tokenizer, evaluation.
- `gemma_checkpoints/` – Checkpoints auto-saved here.
- `gemma_tokenizer.json` – Tokenizer auto-generated after training.
- `README.md` – This file.

---

## 📚 Example Usage

### 🛠️ 1. Install dependencies

```bash
pip install jax flax optax datasets tokenizers tqdm orbax-checkpoint
```

### 📖 2. Train the model

```bash
python gemma_in_jax.py --train
```

> Trains on Tiny Shakespeare with tokenizer training, checkpointing, and text generation samples.

### ✍️ 3. Generate text from a trained model

```bash
python gemma_in_jax.py --generate --prompt "The universe is vast" --max_tokens 100
```

---

## 🧪 Training Pipeline

- Prepares and tokenizes the dataset using Hugging Face Datasets + SentencePiece.
- Uses **warmup + cosine decay schedule** and **gradient clipping**.
- Includes **multi-device TPU support** via `pmap`.
- Saves checkpoints with **Orbax**, resumes from the latest automatically.
- Periodically prints **sample text generations** and evaluates with **loss & perplexity**.

---

## 📈 Evaluation

Evaluation uses average loss and perplexity on held-out validation splits:

```python
val_loss, perplexity = evaluate(model.apply, params, eval_data, config)
print(f"Validation Loss: {val_loss:.4f}, Perplexity: {perplexity:.2f}")
```

---

## 🔮 Future Improvements

- [x] Multi-device TPU training with `pmap`
- [x] Orbax checkpointing & resume
- [x] Tokenizer training and saving
- [ ] Add LoRA or QLoRA support
- [ ] Integrate streaming or chunked datasets
- [ ] Add inference API server

---

## 📜 License

Licensed under the **GPL-3.0** license. See the [LICENSE](./LICENSE) for full terms.

---

💡 **Contributions welcome!** Feel free to open issues, submit PRs, or experiment with enhancements.

---
