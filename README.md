# 🚀 **Awesome JAX & Flax LLMs**  

Welcome to [awesome-jax-flax-llms](https://github.com/dhyaneesh/awesome-jax-flax-llms), a collection of open-source large language model (LLM) implementations built with **JAX & Flax**. This repository provides modular, efficient, and scalable implementations of transformer-based models, optimized for **high-speed TPU/GPU training** and **efficient inference**.  

> [!IMPORTANT]
> The implementations are for educational purposes only, which means it is not for any production but it covers all components of the models and can be changed to fit production needs.

## 🛠 **Features**  
- ✅ **Multiple LLM architectures implemented in JAX/Flax**  
- ✅ **Optimized for TPU acceleration with JAX’s XLA compiler**  
- ✅ **Highly modular & extensible codebase**  
- ✅ **Efficient training with Optax optimizers**  
- ✅ **Hugging Face support to train on various datasets**  
- ⏳ **Fine-tuning support (Coming Soon!)**  

## 📚 **Implemented Models**  

### ✅ **GPT-2 - JAX/Flax**  
A compact transformer-based language model implemented in **pure JAX/Flax**. This implementation leverages **XLA optimizations** for parallelism, making it efficient on **TPUs and GPUs**. It serves as the foundation for exploring JAX-based language modeling.  

📌 *Notebook: `models/gpt-2/gpt2_in_jax.ipynb`*  
📌 *Script: `models/gpt-2/train.py`*  

### ✅ **Llama 3 - JAX/Flax**  
An extension of the Llama series, incorporating **state-of-the-art optimizations** in JAX for handling **longer context windows** and **reduced memory footprint** with precision tuning. 

📌 *Notebook: `models/llama3/llama3_in_jax.ipynb`*  
📌 *Script: `models/llama3/llama3_in_jax.py`*  

### ⏳ **Gemma - JAX/Flax** (WIP)
A lightweight, open-weight transformer model from Google, implemented in **pure JAX/Flax**. This version emphasizes **modular design**, efficient **parameter sharing**, and **scalability** across TPUs/GPUs. Ideal for experimentation with **instruction tuning** and **alignment** techniques.  

⏳ *Notebook: `models/gemma/gemma_in_jax.ipynb`*  
📌 *Script: `models/gemma/gemma_in_jax.py`*  

Let me know if you want to add details like architecture size, tokenizer notes, or integration hooks.

### 📅 **DeepSeek-R1 - JAX/Flax (Coming Soon)**  
A **cutting-edge deep learning model** designed for **highly efficient semantic search**, leveraging **advanced transformer architectures** and **optimizations in JAX** for **faster retrieval and reduced computational costs**.

### 📅 **Mistral - JAX/Flax (Coming Soon)**  
A high-performance implementation of the **Mistral architecture**, featuring **dense & sparse mixture-of-expert layers**. This model will showcase **advanced TPU utilization** and optimized autoregressive decoding.  

---

## 📖 **Usage**  

### **Recommended Environment: Google Colab**  
These models are best run in **Google Colab**, which provides **free TPU support** for optimal performance.  

### **Running the Notebooks**  
Each model has its own Jupyter notebook. Navigate to the respective directories and open the notebook in **Google Colab** to explore the implementations.  

---

## 🔥 **Next Steps**  
- 🏗 **Fine-tuning Support**: Enabling training on custom datasets.  
- ⚡ **Larger Model Implementations**: Expanding the repo with more LLMs.  

## 📖 **References**

[1] HighPerfLLMs2024. Available: https://github.com/rwitten/HighPerfLLMs2024

[2] JAX Scaling Book. Available: https://jax-ml.github.io/scaling-book/


## 📜 **License**  
This project is licensed under the **GNU General Public License v3.0 (GPL-3.0)**. See the [LICENSE](LICENSE) file for details.  

---

💡 *Contributions are welcome! Feel free to submit issues and pull requests.*  

---
