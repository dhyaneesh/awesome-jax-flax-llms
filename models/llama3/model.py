import jax
import jax.numpy as jnp
from jax import random, lax, vmap
import flax.linen as nn
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

# Model Configuration
class LLaMAConfig:
    """Configuration for LLaMA model"""
    vocab_size: int = 32000
    dim: int = 512  # Hidden dimension
    n_layers: int = 8  # Number of transformer layers
    n_heads: int = 8  # Number of attention heads
    n_kv_heads: int = 4  # Number of key/value heads (for grouped-query attention)
    max_seq_len: int = 2048  # Maximum sequence length
    dropout_rate: float = 0.0  # Dropout rate

    # RoPE settings
    rope_theta: float = 10000.0  # Base for rotary embeddings

    # Generation settings
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.95

# RMS Normalization Layer
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    dim: int
    eps: float = 1e-5

    @nn.compact
    def __call__(self, x):
        weight = self.param('weight', nn.initializers.ones, (self.dim,))
        variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        x = x * jnp.reciprocal(jnp.sqrt(variance + self.eps))
        return x * weight

# Rotary Position Embeddings
def precompute_freqs_cis(dim: int, max_seq_len: int, theta: float = 10000.0):
    """Precompute the frequency tensor for complex exponentials (rotary embeddings)."""
    # Compute the frequencies for each feature dimension
    freqs = 1.0 / (theta ** (jnp.arange(0, dim // 2, dtype=jnp.float32) / dim))
    t = jnp.arange(max_seq_len, dtype=jnp.float32)
    # Create the frequency matrix by outer product
    freqs = jnp.outer(t, freqs)
    # Convert to complex exponentials
    return jnp.complex64(jnp.exp(1j * freqs))

def apply_rotary_emb(xq, xk, freqs_cis):
    """Apply rotary embeddings to the query and key tensors."""
    # Reshape inputs to isolate the last dimension into pairs for complex multiplication
    xq_r, xk_r = jnp.reshape(xq, (*xq.shape[:-1], -1, 2)), jnp.reshape(xk, (*xk.shape[:-1], -1, 2))

    # Convert to complex numbers
    xq_complex = jnp.complex64(xq_r[..., 0] + 1j * xq_r[..., 1])
    xk_complex = jnp.complex64(xk_r[..., 0] + 1j * xk_r[..., 1])

    # Reshape frequency cis for broadcasting
    freqs_cis = jnp.reshape(freqs_cis, (1, freqs_cis.shape[0], 1, freqs_cis.shape[1]))

    # Apply rotation through complex multiplication
    xq_out = xq_complex * freqs_cis
    xk_out = xk_complex * freqs_cis

    # Convert back to real tensor and reshape
    xq = jnp.stack([jnp.real(xq_out), jnp.imag(xq_out)], axis=-1).reshape(xq.shape)
    xk = jnp.stack([jnp.real(xk_out), jnp.imag(xk_out)], axis=-1).reshape(xk.shape)

    return xq, xk

@partial(jax.jit)
def flash_attention(q, k, v, mask=None, scale=None):
    """
    Optimized implementation of attention mechanism using JAX primitives
    for better compiler optimization and memory efficiency.
    """
    if scale is None:
        scale = 1.0 / jnp.sqrt(q.shape[-1])
    
    # Compute attention scores
    scores = jnp.matmul(q, k.transpose(0, 1, 3, 2)) * scale
    
    # Apply mask if provided
    if mask is not None:
        scores = jnp.where(mask == 0, float('-inf'), scores)
    
    # Apply softmax
    attn = jax.nn.softmax(scores, axis=-1)
    
    # Compute output
    out = jnp.matmul(attn, v)
    
    return out, attn

def swiglu(x, w1, w2, w3):
    """SwiGLU activation"""
    return jnp.matmul(x, w1) * jax.nn.silu(jnp.matmul(x, w2))

class LLaMACausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with support for grouped-query attention"""
    config: LLaMAConfig

    def setup(self):
        self.n_heads = self.config.n_heads
        self.n_kv_heads = self.config.n_kv_heads
        self.head_dim = self.config.dim // self.n_heads
        self.scale = self.head_dim ** -0.5

        # Projections for queries, keys, and values
        self.q_proj = nn.Dense(self.n_heads * self.head_dim)
        self.k_proj = nn.Dense(self.n_kv_heads * self.head_dim)
        self.v_proj = nn.Dense(self.n_kv_heads * self.head_dim)
        self.o_proj = nn.Dense(self.config.dim)

    def __call__(self, x, freqs_cis, mask=None, deterministic=True):
        batch_size, seq_len, _ = x.shape
        
        # Project queries, keys, and values
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for attention
        q = q.reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        
        # Apply rotary embeddings
        q, k = apply_rotary_emb(q, k, freqs_cis)
        
        # Repeat k,v heads if n_kv_heads < n_heads
        if self.n_kv_heads != self.n_heads:
            k = jnp.repeat(k, self.n_heads // self.n_kv_heads, axis=2)
            v = jnp.repeat(v, self.n_heads // self.n_kv_heads, axis=2)
        
        # Compute attention
        out, _ = flash_attention(q, k, v, mask, self.scale)
        
        # Reshape and project output
        out = out.reshape(batch_size, seq_len, -1)
        return self.o_proj(out)

class LLaMAMLP(nn.Module):
    """Feed-forward network with SwiGLU activation"""
    config: LLaMAConfig

    def setup(self):
        self.w1 = nn.Dense(self.config.dim * 4)
        self.w2 = nn.Dense(self.config.dim * 4)
        self.w3 = nn.Dense(self.config.dim)

    def __call__(self, x):
        return self.w3(swiglu(x, self.w1, self.w2, self.w3))

class LLaMABlock(nn.Module):
    """LLaMA transformer block"""
    config: LLaMAConfig

    def setup(self):
        self.attn = LLaMACausalSelfAttention(self.config)
        self.mlp = LLaMAMLP(self.config)
        self.norm1 = RMSNorm(self.config.dim)
        self.norm2 = RMSNorm(self.config.dim)

    def __call__(self, x, freqs_cis, mask=None, deterministic=True):
        # Pre-norm for attention
        h = x + self.attn(self.norm1(x), freqs_cis, mask, deterministic)
        # Pre-norm for MLP
        out = h + self.mlp(self.norm2(h))
        return out

class LLaMA3(nn.Module):
    """LLaMA language model"""
    config: LLaMAConfig

    def setup(self):
        self.embed_tokens = nn.Embed(self.config.vocab_size, self.config.dim)
        self.norm = RMSNorm(self.config.dim)
        
        # Create transformer blocks
        self.layers = [LLaMABlock(self.config) for _ in range(self.config.n_layers)]
        
        # Precompute rotary embeddings
        self.freqs_cis = precompute_freqs_cis(
            self.config.dim // self.config.n_heads,
            self.config.max_seq_len,
            self.config.rope_theta
        )

    def _tie_weights(self, params):
        """Tie the weights between the embedding and output projection."""
        params['embed_tokens']['embedding'] = params['lm_head']['kernel'].T
        return params

    def __call__(self, input_ids, deterministic=True, params=None):
        # Get input embeddings
        x = self.embed_tokens(input_ids)
        
        # Create causal mask
        seq_len = input_ids.shape[1]
        mask = jnp.triu(jnp.ones((seq_len, seq_len)), k=1)
        mask = jnp.where(mask, 0, 1)
        
        # Apply transformer blocks
        for layer in self.layers:
            x = layer(x, self.freqs_cis, mask, deterministic)
        
        # Final normalization
        x = self.norm(x)
        
        # Project to vocabulary
        logits = jnp.matmul(x, self.embed_tokens.embedding.T)
        
        return logits

    def generate(self, input_ids, max_new_tokens, rng_key, temperature=0.8, top_k=40, top_p=0.95):
        """Generate text using the model"""
        current_ids = input_ids
        for _ in range(max_new_tokens):
            # Get model predictions
            logits = self(current_ids, deterministic=True)
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < jnp.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits = jnp.where(indices_to_remove, float('-inf'), next_token_logits)
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = jax.lax.top_k(next_token_logits, k=next_token_logits.shape[-1])
                cumulative_probs = jnp.cumsum(jax.nn.softmax(sorted_logits, axis=-1))
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove = jnp.concatenate([jnp.zeros((1,), dtype=bool), sorted_indices_to_remove[:-1]])
                indices_to_remove = sorted_indices_to_remove[sorted_indices]
                next_token_logits = jnp.where(indices_to_remove, float('-inf'), next_token_logits)
            
            # Sample next token
            probs = jax.nn.softmax(next_token_logits, axis=-1)
            next_token = jax.random.categorical(rng_key, probs)
            
            # Append next token to sequence
            current_ids = jnp.concatenate([current_ids, next_token[:, None]], axis=1)
            
            # Update RNG key
            rng_key, _ = jax.random.split(rng_key)
        
        return current_ids 