import os
import math
import pickle
import jax
import jax.numpy as jnp
from jax import random, lax, vmap
import flax.linen as nn
import optax
from functools import partial
from tqdm import tqdm
from datasets import load_dataset
from tokenizers import SentencePieceUnigramTokenizer
from typing import Any, Callable, Dict, List, Optional, Tuple

# Check for TPU and set environment
os.environ['JAX_PLATFORM_NAME'] = 'tpu'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
print("JAX devices:", jax.devices())

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
    
    # Training settings
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 1000
    max_steps: int = 100000
    
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

# Flash Attention implementation (simplified for this example)
def flash_attention(q, k, v, mask=None):
    """Efficient attention implementation (conceptual - not actual Flash Attention)"""
    batch_size, num_heads, seq_len, head_dim = q.shape
    
    # Scaled dot-product attention
    scale = 1.0 / math.sqrt(head_dim)
    scores = jnp.matmul(q, jnp.swapaxes(k, -1, -2)) * scale
    
    # Apply mask if provided
    if mask is not None:
        scores = scores + mask
    
    # Apply softmax and compute weighted sum
    attn_weights = jax.nn.softmax(scores, axis=-1)
    output = jnp.matmul(attn_weights, v)
    
    return output

# SwiGLU Activation
def swiglu(x, w1, w2, w3):
    """SwiGLU activation function"""
    return jnp.dot(jax.nn.silu(jnp.dot(x, w3)) * jnp.dot(x, w1), w2)

# LLaMA Causal Self-Attention Module
class LLaMACausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with support for grouped-query attention"""
    config: LLaMAConfig
    
    def setup(self):
        config = self.config
        dim = config.dim
        n_heads = config.n_heads
        n_kv_heads = config.n_kv_heads
        head_dim = dim // n_heads
        
        # QKV projections
        self.wq = nn.Dense(n_heads * head_dim,
                          kernel_init=nn.initializers.variance_scaling(1.0, 'fan_in', 'normal'))
        self.wk = nn.Dense(n_kv_heads * head_dim,
                          kernel_init=nn.initializers.variance_scaling(1.0, 'fan_in', 'normal'))
        self.wv = nn.Dense(n_kv_heads * head_dim,
                          kernel_init=nn.initializers.variance_scaling(1.0, 'fan_in', 'normal'))
        
        # Output projection
        self.wo = nn.Dense(dim,
                          kernel_init=nn.initializers.variance_scaling(1.0, 'fan_in', 'normal'))
        
        # QK normalization for improved stability
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)
    
    def __call__(self, x, freqs_cis, mask=None, deterministic=True):
        B, T, C = x.shape
        config = self.config
        n_heads = config.n_heads
        n_kv_heads = config.n_kv_heads
        head_dim = C // n_heads
        
        # Linear projections
        q = self.wq(x).reshape(B, T, n_heads, head_dim)
        k = self.wk(x).reshape(B, T, n_kv_heads, head_dim)
        v = self.wv(x).reshape(B, T, n_kv_heads, head_dim)
        
        # Apply QK normalization
        q = jnp.swapaxes(self.q_norm(jnp.swapaxes(q, 1, 2)), 1, 2)
        k = jnp.swapaxes(self.k_norm(jnp.swapaxes(k, 1, 2)), 1, 2)
        
        # Apply rotary embeddings
        q, k = apply_rotary_emb(q, k, freqs_cis[:T])
        
        # Repeat k and v heads if n_heads > n_kv_heads (grouped-query attention)
        if n_heads > n_kv_heads:
            k = jnp.repeat(k, n_heads // n_kv_heads, axis=2)
            v = jnp.repeat(v, n_heads // n_kv_heads, axis=2)
        
        # Transpose tensors for attention computation (B, H, T, D)
        q, k, v = map(lambda x: jnp.swapaxes(x, 1, 2), (q, k, v))
        
        # Use flash attention (conceptually)
        output = flash_attention(q, k, v, mask)
        
        # Transpose output and project back to full dimension
        output = jnp.swapaxes(output, 1, 2).reshape(B, T, -1)
        output = self.wo(output)
        
        return output

# LLaMA MLP Module
class LLaMAMLP(nn.Module):
    """Feed-forward network with SwiGLU activation"""
    config: LLaMAConfig
    
    def setup(self):
        dim = self.config.dim
        hidden_dim = 4 * dim  # 4x expansion
        
        # Linear projections
        self.w1 = nn.Dense(hidden_dim,
                         kernel_init=nn.initializers.variance_scaling(1.0, 'fan_in', 'normal'))
        self.w2 = nn.Dense(dim,
                         kernel_init=nn.initializers.variance_scaling(1.0, 'fan_in', 'normal'))
        self.w3 = nn.Dense(hidden_dim,
                         kernel_init=nn.initializers.variance_scaling(1.0, 'fan_in', 'normal'))
    
    def __call__(self, x):
        return swiglu(x, self.w1.variables['params']['kernel'], 
                     self.w2.variables['params']['kernel'], 
                     self.w3.variables['params']['kernel'])

# LLaMA Transformer Block
class LLaMABlock(nn.Module):
    """LLaMA transformer block"""
    config: LLaMAConfig
    
    def setup(self):
        self.attention_norm = RMSNorm(self.config.dim)
        self.attention = LLaMACausalSelfAttention(self.config)
        self.ffn_norm = RMSNorm(self.config.dim)
        self.ffn = LLaMAMLP(self.config)
        self.dropout = nn.Dropout(self.config.dropout_rate)
    
    def __call__(self, x, freqs_cis, mask=None, deterministic=True):
        # Pre-norm for attention
        h = x + self.dropout(
            self.attention(self.attention_norm(x), freqs_cis, mask, deterministic),
            deterministic=deterministic
        )
        
        # Pre-norm for FFN
        out = h + self.dropout(
            self.ffn(self.ffn_norm(h)),
            deterministic=deterministic
        )
        
        return out

# Full LLaMA Model
class LLaMA3(nn.Module):
    """LLaMA language model"""
    config: LLaMAConfig
    
    def setup(self):
        config = self.config
        
        # Token embeddings
        self.token_embedding = nn.Embed(
            config.vocab_size, 
            config.dim,
            embedding_init=nn.initializers.normal(stddev=0.02)
        )
        
        # Transformer blocks
        self.blocks = [LLaMABlock(config) for _ in range(config.n_layers)]
        
        # Final layer norm
        self.norm_f = RMSNorm(config.dim)
        
        # Output projection (tied with embeddings)
        self.lm_head = nn.Dense(
            config.vocab_size,
            kernel_init=nn.initializers.normal(stddev=0.02),
            use_bias=False
        )
        
        # Pre-compute rotary embeddings
        self.freqs_cis = precompute_freqs_cis(
            config.dim // config.n_heads,
            config.max_seq_len,
            config.rope_theta
        )
    
    def __call__(self, input_ids, deterministic=True):
        B, T = input_ids.shape
        
        # Create causal attention mask
        mask = jnp.tril(
            jnp.ones((self.config.max_seq_len, self.config.max_seq_len))
        )
        mask = jnp.where(mask == 0, jnp.finfo(jnp.float32).min, 0.0)
        mask = mask[None, None, :T, :T]
        
        # Get embeddings
        h = self.token_embedding(input_ids)
        
        # Apply transformer blocks
        for block in self.blocks:
            h = block(h, self.freqs_cis, mask, deterministic)
        
        # Apply final normalization
        h = self.norm_f(h)
        
        # Get logits
        logits = self.lm_head(h)
        
        return logits
    
    def generate(self, input_ids, max_new_tokens, rng_key, temperature=0.8, top_k=40, top_p=0.95):
        """Generate text using the model"""
        B, T = input_ids.shape
        
        # Create initial output array
        output = input_ids
        
        # Generate tokens
        for i in range(max_new_tokens):
            # Keep the context within max sequence length
            curr_input = output[:, -self.config.max_seq_len:]
            
            # Get logits for the next token
            logits = self(curr_input, deterministic=True)[:, -1, :]
            
            # Apply temperature
            logits = logits / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                top_k_v, top_k_i = jax.lax.top_k(logits, top_k)
                indices_to_remove = jnp.broadcast_to(
                    jnp.arange(logits.shape[-1]) < top_k_i[:, -1:],
                    logits.shape
                )
                logits = jnp.where(indices_to_remove, logits, jnp.finfo(jnp.float32).min)
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = jnp.sort(logits, axis=-1, descending=True)
                cumulative_probs = jnp.cumsum(jax.nn.softmax(sorted_logits, axis=-1), axis=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep the first token above threshold
                sorted_indices_to_remove = jnp.roll(sorted_indices_to_remove, 1, axis=1)
                sorted_indices_to_remove = sorted_indices_to_remove.at[:, 0].set(False)
                
                # Scatter sorted tensors to original indexing
                indices_to_remove = jnp.zeros_like(logits, dtype=bool)
                indices_to_remove = indices_to_remove.at[jnp.arange(B)[:, None], sorted_indices].set(sorted_indices_to_remove)
                logits = jnp.where(indices_to_remove, jnp.finfo(jnp.float32).min, logits)
            
            # Sample from the filtered distribution
            rng_key, sample_key = random.split(rng_key)
            next_token = random.categorical(sample_key, logits, shape=(B,))
            
            # Append the sampled token to the sequence
            output = jnp.concatenate([output, next_token[:, None]], axis=1)
        
        return output

# Training State
class LLaMATrainState:
    """Custom training state for LLaMA model"""
    params: Dict
    opt_state: optax.OptState
    step: int
    
    @classmethod
    def create(cls, model, config, rng_key):
        """Initialize a new training state"""
        params = model.init(rng_key, jnp.ones((1, 1), dtype=jnp.int32))
        
        # Create optimizer with learning rate schedule
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=config.learning_rate,
            warmup_steps=config.warmup_steps,
            decay_steps=config.max_steps,
            end_value=config.learning_rate * 0.1
        )
        
        # AdamW optimizer with gradient clipping
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(
                learning_rate=lr_schedule,
                b1=0.9,
                b2=0.95,
                eps=1e-8,
                weight_decay=config.weight_decay
            )
        )
        
        opt_state = optimizer.init(params)
        
        return cls(
            params=params,
            opt_state=opt_state,
            step=0
        )
    
    def save(self, filepath):
        """Save training state to disk"""
        # Convert jax arrays to numpy for pickling
        numpy_state = jax.tree_map(lambda x: x.copy() if hasattr(x, 'copy') else x, self)
        with open(filepath, 'wb') as f:
            pickle.dump(numpy_state, f)
    
    @classmethod
    def load(cls, filepath):
        """Load training state from disk"""
        with open(filepath, 'rb') as f:
            numpy_state = pickle.load(f)
        # Convert numpy arrays back to jax arrays
        state = jax.tree_map(lambda x: jnp.array(x) if hasattr(x, 'shape') else x, numpy_state)
        return state

# Data preparation functions
def prepare_datasets(config):
    """Load and prepare datasets"""
    # Load datasets
    wiki_dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")
    
    # Initialize tokenizer
    tokenizer = SentencePieceUnigramTokenizer()
    
    # Train tokenizer if needed (simplified for this example)
    # In practice, you'd want to use a pre-trained tokenizer
    
    def tokenize_function(example):
        tokens = tokenizer.encode(example["text"]).ids
        return {"input_ids": tokens}
    
    # Tokenize dataset
    tokenized_dataset = wiki_dataset.map(
        tokenize_function,
        remove_columns=["text"],
        batched=True
    )
    
    return tokenized_dataset, tokenizer

def get_batch(key, data, config):
    """Create a batch of data for training"""
    batch_size = config.batch_size
    seq_len = config.max_seq_len
    
    # Generate random starting indices
    total_tokens = len(data["input_ids"])
    ix = random.randint(key, (batch_size,), 0, total_tokens - seq_len)
    
    # Vectorized operation to get input and target sequences
    x = vmap(lambda i: lax.dynamic_slice(jnp.array(data["input_ids"]), (i,), (seq_len,)))(ix)
    y = vmap(lambda i: lax.dynamic_slice(jnp.array(data["input_ids"]), (i + 1,), (seq_len,)))(ix)
    
    return x, y

# TPU initialization and setup
def initialize_tpu():
    """Initialize TPU devices if available"""
    devices = jax.devices()
    n_devices = len(devices)
    print(f"Found {n_devices} JAX devices: {devices}")
    
    # Check if TPUs are available
    if any(d.platform == 'tpu' for d in devices):
        print("TPU devices detected. Setting up for distributed training.")
        # Additional TPU-specific setup could go here
    else:
        print("No TPU devices found. Using available CPU/GPU.")
    
    return n_devices

def replicate_state_on_devices(state, n_devices):
    """Replicate training state across devices for pmap"""
    return jax.device_put_replicated(state, jax.local_devices()[:n_devices])

def step_device_rng_keys(rng, n_devices):
    """Create RNG keys for each device"""
    keys = random.split(rng, n_devices + 1)
    return keys[0], keys[1:]

# Training functions
def compute_loss(params, model, batch, rng_key):
    """Compute loss for a batch of data"""
    inputs, targets = batch
    
    # Forward pass
    logits = model.apply(params, inputs, deterministic=False, rngs={'dropout': rng_key})
    
    # Reshape for cross entropy
    logits = logits.reshape(-1, logits.shape[-1])
    targets = targets.reshape(-1)
    
    # Compute cross entropy loss
    loss = -jnp.mean(
        jnp.take_along_axis(
            jax.nn.log_softmax(logits),
            targets[:, None],
            axis=1
        )
    )
    
    return loss

def train_step(state, model, batch, rng_key, optimizer):
    """Single training step"""
    # Get gradient function
    loss_fn = lambda p: compute_loss(p, model, batch, rng_key)
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    
    # Apply gradients
    updates, new_opt_state = optimizer.update(grads, state.opt_state, state.params)
    new_params = optax.apply_updates(state.params, updates)
    
    # Update state
    new_state = LLaMATrainState(
        params=new_params,
        opt_state=new_opt_state,
        step=state.step + 1
    )
    
    return new_state, loss

# JIT-compiled training step for efficiency
train_step_jit = jax.jit(train_step, static_argnums=(1, 4))

# pmapped training step for multi-device training
def train_step_pmap(state, model, batch, rng_keys, optimizer):
    """Training step for pmap (parallel execution)"""
    return jax.pmap(
        lambda s, b, k: train_step(s, model, b, k, optimizer),
        axis_name='batch'
    )(state, batch, rng_keys)

def evaluate(params, model, eval_data, config, num_batches=50):
    """Evaluate model on validation data"""
    key = random.PRNGKey(42)
    total_loss = 0.0
    
    for i in range(num_batches):
        key, batch_key = random.split(key)
        batch = get_batch(batch_key, eval_data, config)
        loss = compute_loss(params, model, batch, random.PRNGKey(0))
        total_loss += loss
    
    avg_loss = total_loss / num_batches
    perplexity = jnp.exp(avg_loss)
    
    return avg_loss, perplexity

# Main training function
def train_llama(config, num_epochs=5, steps_per_epoch=1000, save_every=1000):
    """Train LLaMA model"""
    # Initialize TPU
    n_devices = initialize_tpu()
    
    # Setup model
    model = LLaMA3(config)
    rng_key = random.PRNGKey(42)
    
    # Create training state
    state = LLaMATrainState.create(model, config, rng_key)
    
    # Create optimizer
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.learning_rate,
        warmup_steps=config.warmup_steps,
        decay_steps=config.max_steps,
        end_value=config.learning_rate * 0.1
    )
    
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(
            learning_rate=lr_schedule,
            b1=0.9,
            b2=0.95,
            eps=1e-8,
            weight_decay=config.weight_decay
        )
    )
    
    # Prepare datasets
    train_dataset, tokenizer = prepare_datasets(config)
    
    # Set up for multi-device training if available
    if n_devices > 1:
        state = replicate_state_on_devices(state, n_devices)
    
    # Training loop
    rng_key = random.PRNGKey(0)
    step = 0
    total_steps = num_epochs * steps_per_epoch
    
    print(f"Starting training for {num_epochs} epochs ({total_steps} steps)")
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        for step_in_epoch in tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{num_epochs}"):
            # Get a batch of data
            rng_key, batch_key = random.split(rng_key)
            batch = get_batch(batch_key, train_dataset, config)
            
            # Training step
            if n_devices > 1:
                # Multi-device training
                rng_key, dropout_keys = step_device_rng_keys(rng_key, n_devices)
                state, loss = train_step_pmap(state, model, batch, dropout_keys, optimizer)
                # Average loss across devices
                loss = jnp.mean(loss)
            else:
                # Single device training
                rng_key, dropout_key = random.split(rng_key)
                state, loss = train_step_jit(state, model, batch, dropout_key, optimizer)
            
            epoch_loss += loss
            step += 1
            
            # Log progress
            if step % 100 == 0:
                print(f"Step {step}/{total_steps}, Loss: {loss:.4f}")
            
            # Save checkpoint
            if step % save_every == 0:
                if n_devices > 1:
                    # For multi-device, save only the first copy
                    save_state = jax.tree_map(lambda x: x[0], state)
                else:
                    save_state = state
                
                save_state.save(f"llama_checkpoint_step_{step}.pkl")
                
                # Generate sample text
                if n_devices > 1:
                    sample_params = jax.tree_map(lambda x: x[0], state.params)
                else:
                    sample_params = state.params
                
                prompt = tokenizer.encode("Once upon a time").ids
                prompt_tensor = jnp.array([prompt])
                
                sample_rng = random.PRNGKey(step)
                generated = model.apply(
                    {"params": sample_params},
                    prompt_tensor,
                    max_new_tokens=50,
                    rng_key=sample_rng,
                    temperature=config.temperature,
                    top_k=config.top_k,
                    top_p=config.top_p,
                    method=model.generate
                )
                
                generated_text = tokenizer.decode(generated[0].tolist())
                print(f"\nSample generation at step {step}:\n{generated_text}\n")
        
        # End of epoch
        avg_epoch_loss = epoch_loss / steps_per_epoch
        print(f"Epoch {epoch+1} complete. Average loss: {avg_epoch_loss:.4f}")
        
        # Evaluate on validation set
        if n_devices > 1:
            eval_params = jax.tree_map(lambda x: x[0], state.params)
        else:
            eval_params = state.params
        
        # Validation loss and perplexity
        val_loss, perplexity = evaluate(eval_params, model, train_dataset, config)
        print(f"Validation Loss: {val_loss:.4f}, Perplexity: {perplexity:.2f}")
    
    # Save final model
    if n_devices > 1:
        final_state = jax.tree_map(lambda x: x[0], state)
    else:
        final_state = state
    
    final_state.save("llama_final_model.pkl")
    print("Training complete. Final model saved.")
    
    return final_state

# Text generation function
def generate_text(model, params, tokenizer, prompt, max_new_tokens=100, temperature=0.8):
    """Generate text from a prompt"""
    prompt_tokens = tokenizer.encode(prompt).ids
    prompt_tensor = jnp.array([prompt_tokens])
    
    rng_key = random.PRNGKey(0)
    generated = model.apply(
        {"params": params},
        prompt_tensor,
        max_new_tokens=max_new_tokens,
        rng_key=rng_key,
        temperature=temperature,
        top_k=40,
        top_p=0.95,
        method=model.generate
    )
    
    generated_text = tokenizer.decode(generated[0].tolist())
    return generated_text

# Main entry point
if __name__ == "__main__":
    # Create configuration
    config = LLaMAConfig()
    
    # Train the model
    final_state = train_llama(config, num_epochs=5, steps_per_epoch=1000)
    
    # Generate some text
    model = LLaMA3(config)
    tokenizer = prepare_datasets(config)[1]
    
    prompt = "In a distant galaxy"
    generated_text = generate_text(model, final_state.params, tokenizer, prompt)
    
    print("\nGenerated text:")
    print(generated_text)
