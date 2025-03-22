---

# 🦙 LLaMA 3 in JAX/Flax

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhyaneesh/awesome-jax-flax-llms/blob/main/models/llama3/llama3_in_jax.ipynb)

This is a **pure JAX/Flax** implementation of the **LLaMA 3** language model, optimized for **TPU/GPU** execution. It includes full training, evaluation, checkpointing, and text generation support — all built from scratch using efficient JAX primitives and Flax modules.

---

## 🚀 Highlights

- 🧠 **Decoder-only Transformer** architecture with RoPE, GQA, RMSNorm & SwiGLU.
- ⚡ **Optimized TPU/GPU Training** using `pmap` and `jit`.
- 🧪 **Tokenizer training** and **dataset preprocessing** with Hugging Face Datasets.
- 💾 **Checkpointing & Restore** using Orbax.
- ✍️ **Text Generation** with temperature, top-k, and top-p sampling.
- ✅ Minimal dependencies & portable design.

---

## 🧱 Architecture Overview

| Component | Details |
|----------|---------|
| **Attention** | Multi-head causal attention with **Grouped-Query Attention (GQA)** |
| **Position Encoding** | **Rotary Positional Embeddings (RoPE)** |
| **MLP** | Uses **SwiGLU** nonlinearity |
| **Normalization** | **RMSNorm** instead of LayerNorm |
| **Optimizer** | **Optax** with warmup, cosine decay & weight decay |
| **Checkpointing** | **Orbax CheckpointManager** with training resume & evaluation |
| **Text Generation** | Supports top-k, top-p, temperature sampling |

---

## 📦 File Structure

- `llama3_in_jax.py` – Complete model, tokenizer, training, evaluation, and generation code.
- `llama3_in_jax.ipynb` – Notebook interface for interactive training/inference.
- `README.md` – Project documentation (you’re reading it!).
- `llama_tokenizer.json` – Tokenizer saved after training (auto-generated).
- `llama_checkpoints/` – Checkpoints will be stored here (auto-generated).

---

## 📚 Example Usage

### 🛠️ 1. Install dependencies

```bash
pip install jax flax optax datasets tokenizers tqdm orbax-checkpoint
```

### 📖 2. Train the model

```bash
python llama3_in_jax.py
```

> Trains LLaMA on Tiny Shakespeare with tokenizer training, checkpointing, and periodic text sampling.

### 🧪 3. Generate text from trained model

```python
from llama3_in_jax import generate_text, LLaMA3, LLaMAConfig, load_checkpoint
from tokenizers import SentencePieceUnigramTokenizer

# Load model & tokenizer
model = LLaMA3(LLaMAConfig())
state, step = load_checkpoint("llama_checkpoints")
tokenizer = SentencePieceUnigramTokenizer("llama_tokenizer.json")

# Generate
prompt = "In a distant galaxy"
print(generate_text(model, state.params, tokenizer, prompt))
```

---

## 🧪 Training Pipeline

The training loop includes:

- **Tokenizer training** from raw HuggingFace datasets.
- **Gradient clipping**, **warmup**, and **cosine decay** scheduling.
- **pmap**-based TPU support with sharded dropout RNG.
- Periodic **evaluation** and **text generation**.
- **Orbax checkpointing** every N steps.

---

## 📈 Evaluation

After each epoch, the model is evaluated using average loss and perplexity over held-out samples.

```python
val_loss, perplexity = evaluate(model.apply, state.params, eval_data, config)
print(f"Validation Loss: {val_loss:.4f}, Perplexity: {perplexity:.2f}")
```

---

## 🧠 Future Improvements

- [x] Multi-device TPU training.
- [x] Tokenizer training + save/load support.
- [x] Checkpointing with Orbax.
- [ ] Distributed inference server (WIP).
- [ ] Add LoRA or QLoRA fine-tuning.

---

## 📜 License

Licensed under the **GPL-3.0** license. See the [LICENSE](../LICENSE) for details.

---

💡 **Contributions welcome** – feel free to open issues, submit PRs, or fork and experiment!

---
