import jax
import jax.numpy as jnp
from jax import random, lax, vmap
import flax.linen as nn
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

class TransformerConfig:
    """Configuration for the transformer model."""
    def __init__(
        self,
        vocab_size: int,
        block_size: int = 256,
        n_embed: int = 384,
        n_head: int = 6,
        n_layer: int = 6,
        dropout: float = 0.2,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.1,
        beta1: float = 0.9,
        beta2: float = 0.95,
        grad_clip: float = 1.0,
        warmup_steps: int = 2000,
        total_steps: int = 250000,
    ):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embed = n_embed
        self.n_head = n_head
        self.n_layer = n_layer
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.grad_clip = grad_clip
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with optimized implementation."""
    config: TransformerConfig

    def setup(self):
        config = self.config
        assert config.n_embed % config.n_head == 0

        # Key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Dense(
            3 * config.n_embed,
            kernel_init=nn.initializers.normal(stddev=0.02),
            bias_init=nn.initializers.zeros
        )

        # Output projection
        self.c_proj = nn.Dense(
            config.n_embed,
            kernel_init=nn.initializers.normal(stddev=0.02),
            bias_init=nn.initializers.zeros
        )

        # Causal mask to ensure attention only to previous tokens
        self.bias = jnp.tril(jnp.ones((config.block_size, config.block_size)))

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embed = config.n_embed
        self.head_dim = config.n_embed // config.n_head

    def __call__(self, x, training=False):
        B, T, C = x.shape  # batch size, sequence length, embedding dim

        # Calculate query, key, values for all heads in batch
        q, k, v = jnp.split(self.c_attn(x), 3, axis=-1)

        # Split into heads
        k = k.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        q = q.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)

        # Compute attention scores
        att = (q @ k.transpose(0, 1, 3, 2)) * (1.0 / jnp.sqrt(k.shape[-1]))
        att = jnp.where(self.bias[:T, :T] == 0, float('-inf'), att)
        att = jax.nn.softmax(att, axis=-1)
        att = self.attn_dropout(att, deterministic=not training)

        # Apply attention to values
        y = att @ v
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)

        # Output projection
        y = self.resid_dropout(self.c_proj(y), deterministic=not training)
        return y

class MLP(nn.Module):
    """Feed-forward network with GELU activation."""
    config: TransformerConfig

    def setup(self):
        self.c_fc = nn.Dense(
            4 * self.config.n_embed,
            kernel_init=nn.initializers.normal(stddev=0.02),
            bias_init=nn.initializers.zeros
        )
        self.c_proj = nn.Dense(
            self.config.n_embed,
            kernel_init=nn.initializers.normal(stddev=0.02),
            bias_init=nn.initializers.zeros
        )
        self.dropout = nn.Dropout(self.config.dropout)

    def __call__(self, x, training=False):
        x = self.c_fc(x)
        x = jax.nn.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x, deterministic=not training)
        return x

class Block(nn.Module):
    """Transformer block with pre-norm architecture."""
    config: TransformerConfig

    def setup(self):
        self.ln1 = nn.LayerNorm(epsilon=1e-5)
        self.ln2 = nn.LayerNorm(epsilon=1e-5)
        self.attn = CausalSelfAttention(self.config)
        self.mlp = MLP(self.config)

    def __call__(self, x, training=False):
        x = x + self.attn(self.ln1(x), training=training)
        x = x + self.mlp(self.ln2(x), training=training)
        return x

class GPT2(nn.Module):
    """GPT-2 language model."""
    config: TransformerConfig

    def setup(self):
        self.transformer = nn.ModuleList([Block(self.config) for _ in range(self.config.n_layer)])
        self.ln_f = nn.LayerNorm(epsilon=1e-5)
        self.wte = nn.Embed(
            self.config.vocab_size,
            self.config.n_embed,
            embedding_init=nn.initializers.normal(stddev=0.02)
        )
        self.wpe = nn.Embed(
            self.config.block_size,
            self.config.n_embed,
            embedding_init=nn.initializers.normal(stddev=0.02)
        )
        self.drop = nn.Dropout(self.config.dropout)

    def __call__(self, idx, training=False):
        b, t = idx.shape
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # Get token and position embeddings
        pos = jnp.arange(0, t, dtype=jnp.int32)
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb, deterministic=not training)

        # Forward through transformer blocks
        for block in self.transformer:
            x = block(x, training=training)

        x = self.ln_f(x)
        logits = jnp.matmul(x, self.wte.embedding.T)

        return logits

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None, rng_key=None):
        """Generate text using the model."""
        if rng_key is None:
            rng_key = jax.random.PRNGKey(0)

        for _ in range(max_new_tokens):
            # Crop context if needed
            idx_cond = idx if idx.shape[1] <= self.config.block_size else idx[:, -self.config.block_size:]

            # Forward pass
            logits = self(idx_cond, training=False)
            logits = logits[:, -1, :] / temperature

            # Apply top-k filtering
            if top_k is not None:
                v, _ = jax.lax.top_k(logits, min(top_k, logits.shape[-1]))
                logits = jnp.where(logits < v[..., [-1]], float('-inf'), logits)

            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = jax.lax.top_k(logits, k=logits.shape[-1])
                cumulative_probs = jnp.cumsum(jax.nn.softmax(sorted_logits, axis=-1))
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove = jnp.concatenate([jnp.zeros((1,), dtype=bool), sorted_indices_to_remove[:-1]])
                indices_to_remove = sorted_indices_to_remove[sorted_indices]
                logits = jnp.where(indices_to_remove, float('-inf'), logits)

            # Sample from the distribution
            probs = jax.nn.softmax(logits, axis=-1)
            next_token = jax.random.categorical(rng_key, probs)
            rng_key, _ = jax.random.split(rng_key)

            # Append sampled token to sequence
            idx = jnp.concatenate([idx, next_token[:, None]], axis=1)

        return idx 