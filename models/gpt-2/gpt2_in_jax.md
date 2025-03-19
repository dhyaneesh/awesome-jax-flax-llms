# 📝 GPT-2 in JAX/Flax

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhyaneesh/awesome-jax-flax-llms/blob/main/gpt2_in_jax.ipynb)

This folder contains a JAX/Flax implementation of the **GPT-2** language model. As part of the larger **awesome-jax-flax-llms** project, this implementation demonstrates how to efficiently train and run transformer-based models on **TPUs and GPUs** using JAX.

## 🚀 Overview

The `gpt2_in_jax.ipynb` notebook provides a clean and modular implementation of **GPT-2** using **JAX & Flax**, leveraging JAX’s **XLA compilation** and **Optax optimizers** for accelerated training. This implementation is designed for:

- Efficient **autoregressive text generation**.
- Scalable training with **JAX’s parallelization features**.
- Fine-tuning on **small to medium-scale datasets**.

## 🤖 Understanding GPT-2

GPT-2 (Generative Pre-trained Transformer 2) is an **autoregressive language model** that generates text by predicting the next token in a sequence. It is built using a stack of **Transformer decoder layers**, utilizing **self-attention mechanisms** to capture long-range dependencies. The key components of GPT-2 include:

- **Multi-head Self-Attention**: Enables the model to attend to different parts of the input sequence simultaneously.
- **Feedforward Neural Networks**: Processes the attended information to make predictions.
- **Positional Encoding**: Incorporates word order information since transformers lack built-in sequence awareness.
- **Layer Normalization & Dropout**: Improves stability and prevents overfitting.

---

![GPT-2 Architecture](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*YZTqlV51QyhX6VL9AV31eQ.png)

---

## 🛠 Features

- ✅ **Pure JAX/Flax implementation** of GPT-2.
- ✅ **Optimized for TPUs & GPUs** using JAX’s Just-In-Time (JIT) compilation.
- ✅ **Optax-based training** for efficient optimization.
- ✅ **Flexible model configuration**, allowing easy scaling.
- ✅ **Minimal dependencies**, making it lightweight & easy to extend.

## 📌 Notebook Details

The notebook includes:

- 📖 **Dataset Preparation**: Downloads and processes the *tinyshakespeare* dataset from GitHub.
- 🏗 **Model Definition**: GPT-2 architecture built using `Flax.linen`.
- 🎯 **Training Pipeline**: Implements loss computation, backpropagation, and optimization using `Optax`.
- 🏎 **Inference & Generation**: Generates text samples efficiently with autoregressive decoding.
- 📊 **Performance Evaluation**: Tracks training progress and visualizes loss curves.

## 🏗 Setup & Usage

### **1️⃣ Install Dependencies**

Ensure you have JAX and Flax installed. Run the following:

```bash
pip install jax flax optax datasets transformers
```

### **2️⃣ Run the Notebook**

Execute the `gpt2_in_jax.ipynb` notebook step by step in **Google Colab (with TPU runtime)** or a local Jupyter environment with GPU support.

### **3️⃣ Fine-tune GPT-2** (Optional)

To fine-tune on custom datasets, modify the training loop and load your dataset using **Hugging Face Datasets** or a custom data pipeline.

## 📖 Next Steps

- 🔄 **Enable longer context training** for improved coherence.
- 🪛 **Finetuning** the model using Hugging Face Datasets or a custom data pipeline.
- ⚡ Optimize inference with XLA caching.
- 📚 **Experiment with different tokenization methods**.

## 📜 License

This project is licensed under the **GPL-3.0** license. See the [LICENSE](../LICENSE) file for details.

---

💡 Contributions & feedback are welcome! 🚀

