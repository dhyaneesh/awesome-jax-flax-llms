# 🦙 LLaMA 3 in JAX/Flax

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-repo/llama3_in_jax.ipynb)

This repository contains a **JAX/Flax implementation** of the **LLaMA 3** language model. Designed for efficient execution on **TPUs and GPUs**, this implementation utilizes JAX’s **XLA compilation** and **Optax optimizers** for high-performance training.

## 🚀 Overview

The `llama3_in_jax.py` script and accompanying notebook provide a modular implementation of **LLaMA 3** using **JAX & Flax**, with a focus on:

- **Scalable autoregressive text generation**.
- **Optimized parallel training** using JAX's **pmap** for **TPUs**.
- **Advanced optimizations** such as grouped-query attention (GQA) and Rotary Positional Embeddings (RoPE).

## 🧠 Understanding LLaMA 3

LLaMA 3 is a **decoder-only transformer model** optimized for efficiency and scalability. Key architectural highlights include:

- **Multi-head Self-Attention with Grouped-Query Attention (GQA)**: Reduces memory and speeds up inference.
- **Rotary Positional Embeddings (RoPE)**: Enables longer context lengths with minimal overhead.
- **SwiGLU Activation Function**: Used in feedforward layers for improved efficiency.
- **Root Mean Square Normalization (RMSNorm)**: Stabilizes training and replaces LayerNorm.

---

📌 **LLaMA 3 Model Architecture:**

**[Insert Model Diagram Here]**

---

## 🛠 Features

✅ **Pure JAX/Flax Implementation** (No PyTorch dependencies).  
✅ **Efficient TPU/ GPU Training** via **pmap and vmap**.  
✅ **Custom Rotary Position Embeddings (RoPE)** implementation.  
✅ **Grouped-Query Attention (GQA)** for faster inference.  
✅ **Optax-based Optimizer with Warmup & Weight Decay**.  
✅ **Minimal Dependencies & Hugging Face Dataset Support**.  

## 📌 Implementation Details

### 1️⃣ **Core Model Components**

The model is implemented with **Flax.linen**, featuring:

- **Custom RMSNorm** for stable training.
- **Precomputed Rotary Position Embeddings** to speed up inference.
- **Efficient Self-Attention Layer** with multi-query support.
- **SwiGLU-based MLP Layers** for enhanced non-linearity.

```python
class Llama3Attention(nn.Module):
    """Multi-head attention with RoPE and Grouped-Query Attention."""
    config: Llama3Config
    
    def setup(self):
        self.q_proj = nn.Dense(self.config.hidden_size)
        self.k_proj = nn.Dense(self.config.kv_dim)
        self.v_proj = nn.Dense(self.config.kv_dim)
        self.o_proj = nn.Dense(self.config.hidden_size)
    
    def __call__(self, hidden_states, position_ids=None, training=False):
        q, k, v = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)
        q, k = apply_rotary_pos_emb(q, k, self.config.freqs_cos, self.config.freqs_sin, position_ids)
        output = self.o_proj(q @ k.transpose(-2, -1) @ v)
        return output
```

### 2️⃣ **Training Pipeline**

The model is trained using **Optax optimizers**, featuring:

- **Warmup Cosine Decay Learning Rate Scheduler**.
- **AdamW with Gradient Clipping** for stable training.
- **Efficient Loss Computation using Softmax Cross-Entropy**.

```python
optimizer = optax.chain(
    optax.clip_by_global_norm(config.grad_clip),
    optax.adamw(
        learning_rate=lr_schedule,
        b1=config.beta1,
        b2=config.beta2,
        weight_decay=config.weight_decay
    )
)
```

### 3️⃣ **Parallel Training on TPU**

JAX’s **pmap** enables efficient TPU training by distributing computation across multiple devices.

```python
@partial(jax.pmap, axis_name='batch')
def train_step(state, batch, rng_keys):
    def loss_fn(params):
        logits, loss = state.apply_fn({'params': params}, batch['input_ids'], batch['labels'])
        return loss, logits
    
    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    grads = jax.lax.pmean(grads, axis_name='batch')
    new_state = state.apply_gradients(grads=grads)
    return new_state, {'loss': loss, 'perplexity': jnp.exp(loss)}
```

## 🏗 Setup & Usage

### **1️⃣ Install Dependencies**

Ensure you have JAX, Flax, and Optax installed:

```bash
pip install jax flax optax datasets transformers
```

### **2️⃣ Load a Hugging Face Dataset**

Modify the dataset loading section in the script:

```python
dataset_name = "wikitext"
dataset = load_dataset(dataset_name)
```

### **3️⃣ Train the Model**

Run the script to start training:

```bash
python llama3_in_jax.py
```

Or execute the Jupyter notebook step by step in **Google Colab (with TPU runtime)**.

### **4️⃣ Generate Text**

Run inference using the trained model:

```python
prompt = "Once upon a time"
generated_text = sample_from_model(state, prompt, encode, decode)
print(generated_text)
```

## 📖 Next Steps

- ✅ **Fine-tuning Support** for custom datasets.
- ✅ **Faster Inference with XLA Compilation**.
- ✅ **Experiment with Different Tokenization Methods**.

## 📜 License

This project is licensed under the **GPL-3.0** license. See the [LICENSE](../LICENSE) file for details.

---

💡 Contributions & feedback are welcome! 🚀
