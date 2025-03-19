---

# 🚀 **Awesome JAX & Flax LLMs**  

Welcome to [awesome-jax-flax-llms](https://github.com/dhyaneesh/awesome-jax-flax-llms), a curated collection of open-source large language model (LLM) implementations built with **JAX & Flax**. This repository provides modular, efficient, and scalable implementations of transformer-based models, optimized for **high-speed TPU/GPU training** and **efficient inference**.  

## 🛠 **Features**  
- ✅ **Multiple LLM architectures implemented in JAX/Flax**  
- ✅ **Optimized for TPU acceleration with JAX’s XLA compiler**  
- ✅ **Highly modular & extensible codebase**  
- ✅ **Efficient training with Optax optimizers**  
- ✅ **Hugging face support to train on various datasets**  
- ⏳ **Fine-tuning support (Coming Soon!)**  

## 📚 **Implemented Models**  

### ✅ **GPT-2 - JAX/Flax**  
A compact transformer-based language model implemented in **pure JAX/Flax**. This implementation leverages **XLA optimizations** for parallelism, making it efficient on **TPUs and GPUs**. It serves as the foundation for exploring JAX-based language modeling.  

📌 *Notebook: `models/gpt-2/gpt2_in_jax.ipynb`*  
📌 *Script: `models/gpt-2/train.py`*  

### ⏳ **Llama 2 - JAX (WIP)**  
An effort to bring **Meta’s Llama 2** to JAX, focusing on **memory-efficient attention** and **scalability for TPU-based pretraining**. This implementation aims to push **large-scale inference and training** in JAX environments.  

### ⏳ **Llama 3 - JAX (WIP)**  
An extension of the Llama series, incorporating **state-of-the-art optimizations** in JAX for handling **longer context windows** and **reduced memory footprint** with precision tuning.  


### 📅 **Mistral - JAX (Coming Soon)**  
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
- 🏆 **Performance Optimizations**: Enhancing TPU inference efficiency.  

## 📜 **License**  
This project is licensed under the **GNU General Public License v3.0 (GPL-3.0)**. See the [LICENSE](LICENSE) file for details.  

---

💡 *Contributions are welcome! Feel free to submit issues and pull requests.*  

---
