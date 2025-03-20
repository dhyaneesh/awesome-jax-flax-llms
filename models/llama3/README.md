# 🦙 LLaMA 3 in JAX/Flax

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-repo/llama3_in_jax.ipynb)

This repository contains a **JAX/Flax implementation** of the **LLaMA 3** language model, optimized for execution on **TPUs and GPUs**. The implementation leverages **XLA compilation** and **Optax optimizers** for high-performance training.

## 🚀 Overview

The `llama3_in_jax.py` script provides a modular implementation of **LLaMA 3** using **JAX & Flax**, with key features including:

- **Autoregressive text generation** with an optimized decoding process.
- **Efficient TPU/GPU training** using JAX's `pmap` for parallelism.
- **Optimized Transformer architecture** with **Grouped-Query Attention (GQA)** and **Rotary Positional Embeddings (RoPE)**.
- **RMSNorm and SwiGLU activations** for stable training.

## 🧠 Model Architecture

LLaMA 3 is a **decoder-only transformer model** designed for efficiency and scalability. Key components include:

- **Multi-head Self-Attention with Grouped-Query Attention (GQA)**: Reduces memory and speeds up inference.
- **Rotary Positional Embeddings (RoPE)**: Improves long-context performance.
- **SwiGLU Activation Function**: Enhances non-linearity in feedforward layers.
- **Root Mean Square Normalization (RMSNorm)**: Replaces LayerNorm for stability.

## 🛠 Features

✅ **Pure JAX/Flax Implementation** (No PyTorch dependencies).  
✅ **Efficient TPU/GPU Training** via **pmap and vmap**.  
✅ **Custom Rotary Position Embeddings (RoPE) Implementation**.  
✅ **Grouped-Query Attention (GQA) for Faster Inference**.  
✅ **Optax-based Optimizer with Warmup & Weight Decay**.  
✅ **Minimal Dependencies & Hugging Face Dataset Support**.  

## 📌 Implementation Details

### 1️⃣ **Core Model Components**

The model is implemented with **Flax.linen**, featuring:

- **Custom RMSNorm** for normalization.
- **Precomputed Rotary Position Embeddings** for improved efficiency.
- **Multi-head Causal Self-Attention** with Grouped-Query Attention.
- **SwiGLU-based MLP Layers** for improved expressiveness.

```python
class LLaMACausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with Grouped-Query Attention."""
    config: LLaMAConfig
    
    def setup(self):
        self.wq = nn.Dense(self.config.dim)
        self.wk = nn.Dense(self.config.dim)
        self.wv = nn.Dense(self.config.dim)
        self.wo = nn.Dense(self.config.dim)
    
    def __call__(self, x, freqs_cis, mask=None):
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        q, k = apply_rotary_emb(q, k, freqs_cis)
        output = flash_attention(q, k, v, mask)
        return self.wo(output)
```

### 2️⃣ **Training Pipeline**

The model is trained using **Optax optimizers**, featuring:

- **Warmup Cosine Decay Learning Rate Scheduler**.
- **AdamW with Gradient Clipping**.
- **Cross-Entropy Loss Computation for Language Modeling**.

```python
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(
        learning_rate=lr_schedule,
        weight_decay=config.weight_decay
    )
)
```

### 3️⃣ **Parallel Training on TPU**

JAX’s **pmap** enables efficient TPU training by distributing computation across multiple devices.

```python
@jax.pmap
def train_step(state, batch):
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch['input_ids'])
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch['labels']).mean()
        return loss
    
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss
```

## 🏗 Setup & Usage

### **1️⃣ Install Dependencies**

Ensure you have JAX, Flax, and Optax installed:

```bash
pip install jax flax optax datasets transformers
```

### **2️⃣ Load a Hugging Face Dataset**

The script supports **Hugging Face Datasets**:

```python
from datasets import load_dataset
dataset = load_dataset("karpathy/tiny_shakespeare", split="train")
```

### **3️⃣ Train the Model**

Run the script to start training:

```bash
python llama3_in_jax.py
```

Or execute the Jupyter notebook in **Google Colab (with TPU runtime)**.

### **4️⃣ Generate Text**

Use the trained model for inference:

```python
prompt = "Once upon a time"
output = model.generate(input_ids, max_new_tokens=50, rng_key=rng_key)
print(output)
```

## 📖 Next Steps

- ✅ **Fine-tuning Support** for custom datasets.
- ✅ **Optimized Inference with XLA Compilation**.
- ✅ **Experiment with Different Tokenization Methods**.

## 📜 License

This project is licensed under the **GPL-3.0** license. See the [LICENSE](../LICENSE) file for details.

---

💡 Contributions & feedback are welcome! 🚀

