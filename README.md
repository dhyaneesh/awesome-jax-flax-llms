# Awesome JAX/Flax LLMs

Welcome to **Awesome JAX/Flax LLMs** – a curated collection of resources, libraries, projects, and tools focused on building, training, and deploying Large Language Models (LLMs) using [JAX](https://github.com/google/jax) and [Flax](https://github.com/google/flax). This repository aims to be your go-to guide for exploring the power of JAX's high-performance computing and Flax's flexible neural network ecosystem in the world of LLMs.

JAX brings composable function transformations (e.g., autodiff, JIT compilation, and vectorization) to Python+NumPy, while Flax provides a lightweight, flexible framework for neural networks. Together, they offer a cutting-edge environment for LLM research and development, optimized for GPUs, TPUs, and beyond.

Contributions are welcome! See the [Contributing](#contributing) section below.

---

## Table of Contents

- [Why JAX and Flax for LLMs?](#why-jax-and-flax-for-llms)
- [Libraries](#libraries)
- [Models and Projects](#models-and-projects)
- [Tutorials and Resources](#tutorials-and-resources)
- [Tools](#tools)
- [Contributing](#contributing)
- [License](#license)

---

## Why JAX and Flax for LLMs?

- **Performance**: JAX's XLA compilation accelerates computations on accelerators like GPUs and TPUs, making it ideal for scaling LLM training.
- **Flexibility**: Flax’s modular design lets you customize architectures and training loops without rigid frameworks.
- **Functional Paradigm**: Pure functions and explicit state management enable reproducible, debuggable LLM workflows.
- **Ecosystem**: A growing community of JAX-based tools (e.g., Optax, Equinox) enhances LLM development.

Whether you're pre-training a massive model, fine-tuning on a specific task, or experimenting with novel architectures, JAX and Flax provide the tools to push the boundaries of LLM research.

---

## Libraries

A selection of libraries tailored for LLMs in the JAX/Flax ecosystem:

- **[Flax](https://github.com/google/flax)** - A neural network library for JAX designed for flexibility, with support for transformers and LLMs.
- **[HuggingFace Transformers (Flax)](https://github.com/huggingface/transformers)** - Pretrained transformer models (e.g., BERT, GPT) with Flax support for natural language tasks.
- **[EasyLM](https://github.com/young-geng/EasyLM)** - A one-stop solution for pre-training, fine-tuning, evaluating, and serving LLMs in JAX/Flax.
- **[Optax](https://github.com/deepmind/optax)** - Gradient processing and optimization library for training LLMs efficiently.
- **[Equinox](https://github.com/patrick-kidger/equinox)** - Elegant neural networks in JAX with callable PyTrees and JIT/grad transformations.
- **[Levanter](https://github.com/stanford-crfm/levanter)** - Scalable, legible foundation models with named tensors in JAX.

---

## Models and Projects

Notable LLM implementations and projects built with JAX and Flax:

- **[DeepSeek-R1-Flax-1.5B-Distill](https://github.com/example/deepseek-r1-flax)** - Flax implementation of a distilled 1.5B parameter reasoning LLM.
- **[Performer](https://github.com/google-research/google-research)** - Flax implementation of the Performer architecture (linear transformers via FAVOR+).
- **[Whisper-JAX](https://github.com/sanchit-gandhi/whisper-jax)** - JAX/Flax implementation of OpenAI’s Whisper model, optimized for up to 70x speed-up on TPUs.
- **[Flax Models](https://github.com/matthias-wright/flaxmodels)** - Pretrained models (e.g., GPT-2) in Flax for easy integration.
- **[EasyDeL](https://github.com/erfanzar/EasyDeL)** - Streamlined training and serving of LLMs in JAX, with support for 8/6/4-bit inference.

---

## Tutorials and Resources

Learn to harness JAX and Flax for LLMs with these guides:

- **[JAX Documentation](https://jax.readthedocs.io/)** - Official docs for JAX’s core features (grad, jit, vmap, pmap).
- **[Flax Documentation](https://flax.readthedocs.io/)** - Guides on Flax NNX and Linen APIs for building models.
- **[Machine Learning with JAX](https://github.com/gordicaleksa/get-started-with-JAX)** - Tutorials covering JAX, Flax, and Haiku basics.
- **[JAX, Flax & Transformers Talks](https://huggingface.co)** - Hugging Face’s 3-day series on JAX/Flax and transformers.
- **[Simple Training Loop in JAX/Flax](https://github.com/soumik12345/jax-series)** - A hands-on example of training with JAX and Flax.

---

## Tools

Enhance your LLM workflow with these JAX/Flax-compatible tools:

- **[Chex](https://github.com/deepmind/chex)** - Utilities for writing and testing reliable JAX code.
- **[Penzai](https://github.com/google/penzai)** - Tools for legible, visualized, and editable neural network models in JAX.
- **[SafeJAX](https://github.com/TheCamusean/safejax)** - Serialize JAX/Flax model parameters with safetensors.
- **[JAXTyping](https://github.com/google/jaxtyping)** - Type annotations for shape and dtype checking in JAX.

---

## Contributing

We’d love your help to make this list even more awesome! To contribute:

1. Fork this repository.
2. Add your resource to the appropriate section in `README.md`.
3. Ensure it’s relevant to JAX, Flax, or LLMs.
4. Submit a pull request with a brief description of your addition.

Please follow the format: `- **[Name](URL)** - Short description.`

---

## License

This repository is licensed under the [GNU GPL 3 License](LICENSE). Feel free to use, share, and adapt this collection as you see fit.

---

*Last updated: April 01, 2025*

---
