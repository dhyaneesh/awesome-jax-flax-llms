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
from flax.training import train_state, orbax_utils
import orbax.checkpoint as ocp


# Check for TPU and set environment
os.environ['JAX_PLATFORM_NAME'] = 'tpu'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
print("JAX devices:", jax.devices())

# Model Configuration
# Update the GemmaConfig class with smaller values
class GemmaConfig:
    """Configuration for Gemma model"""
    vocab_size: int = 32000
    dim: int = 256  # Reduced from 512 to save memory
    n_layers: int = 4  # Reduced from 8 to save memory
    n_heads: int = 4  # Reduced from 8 to save memory
    n_kv_heads: int = 1  # For multi-query attention, use 1 KV head
    max_seq_len: int = 512  # Reduced from 2048 to save memory
    dropout_rate: float = 0.0  # Dropout rate

    # RoPE settings
    rope_theta: float = 10000.0  # Base for rotary embeddings

    # Training settings
    batch_size: int = 16  # Reduced from 16 to save memory
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 100
    max_steps: int = 10000

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
    # Compute scale if not provided
    if scale is None:
        scale = 1.0 / jnp.sqrt(q.shape[-1])

    # Compute attention scores with fused operation
    scores = jnp.einsum('bhid,bhjd->bhij', q, k) * scale

    # Apply causal mask if provided
    if mask is not None:
        scores = scores + mask

    # Stabilize softmax by subtracting max value
    scores_max = jnp.max(scores, axis=-1, keepdims=True)
    scores = scores - lax.stop_gradient(scores_max)

    # Apply softmax with higher precision
    attn_weights = jnp.exp(scores)
    attn_weights = attn_weights / jnp.sum(attn_weights, axis=-1, keepdims=True)

    # Compute attention output with fused operation
    output = jnp.einsum('bhij,bhjd->bhid', attn_weights, v)

    return output

# GeGLU Activation (replacing SwiGLU)
def geglu(x, w1, w2, w3):
    """GeGLU activation function using Flax modules"""
    return w2(jax.nn.gelu(w3(x), approximate=True) * w1(x))

# Gemma Causal Self-Attention Module (Multi-Query Attention)
class GemmaCausalSelfAttention(nn.Module):
    """Multi-query attention (single KV head shared across all query heads)"""
    config: GemmaConfig

    def setup(self):
        config = self.config
        dim = config.dim
        n_heads = config.n_heads
        n_kv_heads = config.n_kv_heads  # Typically 1 for multi-query attention
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

        # Apply rotary embeddings
        q, k = apply_rotary_emb(q, k, freqs_cis[:T])

        # Repeat k and v heads for multi-query attention
        if n_heads > n_kv_heads:
            k = jnp.repeat(k, n_heads // n_kv_heads, axis=2)
            v = jnp.repeat(v, n_heads // n_kv_heads, axis=2)

        # Transpose tensors for attention computation (B, H, T, D)
        q, k, v = map(lambda x: jnp.swapaxes(x, 1, 2), (q, k, v))

        # Use flash attention
        output = flash_attention(q, k, v, mask)

        # Transpose output and project back to full dimension
        output = jnp.swapaxes(output, 1, 2).reshape(B, T, -1)
        output = self.wo(output)

        return output

# Gemma MLP Module with GeGLU activation
class GemmaMLP(nn.Module):
    """Feed-forward network with GeGLU activation"""
    config: GemmaConfig

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
        return geglu(x, self.w1, self.w2, self.w3)

# Gemma Transformer Block
class GemmaBlock(nn.Module):
    """Gemma transformer block with pre-normalization"""
    config: GemmaConfig

    def setup(self):
        self.attention_norm = RMSNorm(self.config.dim)
        self.attention = GemmaCausalSelfAttention(self.config)
        self.ffn_norm = RMSNorm(self.config.dim)
        self.ffn = GemmaMLP(self.config)
        self.dropout = nn.Dropout(self.config.dropout_rate)

    def __call__(self, x, freqs_cis, mask=None, deterministic=True):
        # Pre-norm for attention (Gemma style)
        attn_input = self.attention_norm(x)
        attn_output = self.attention(attn_input, freqs_cis, mask, deterministic)
        h = x + self.dropout(attn_output, deterministic=deterministic)

        # Pre-norm for FFN (Gemma style)
        ffn_input = self.ffn_norm(h)
        ffn_output = self.ffn(ffn_input)
        out = h + self.dropout(ffn_output, deterministic=deterministic)

        return out

# Full Gemma Model
class Gemma(nn.Module):
    """Gemma language model"""
    config: GemmaConfig

    def setup(self):
        config = self.config

        # Token embeddings
        self.token_embedding = nn.Embed(
            config.vocab_size,
            config.dim,
            embedding_init=nn.initializers.normal(stddev=0.02)
        )

        # Transformer blocks
        self.blocks = [GemmaBlock(config) for _ in range(config.n_layers)]

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
                indices_to_remove = jnp.ones_like(logits, dtype=bool)
                indices_to_remove = indices_to_remove.at[jnp.arange(B)[:, None], top_k_i].set(False)
                logits = jnp.where(indices_to_remove, jnp.finfo(jnp.float32).min, logits)

            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_indices = jnp.argsort(logits, axis=-1)[:, ::-1]  # Sort indices in descending order
                sorted_logits = jnp.take_along_axis(logits, sorted_indices, axis=-1)  # Get sorted values

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
            next_token = random.categorical(sample_key, logits, axis=-1)

            # Append the sampled token to the sequence
            output = jnp.concatenate([output, next_token[:, None]], axis=1)

        return output

# Create initial training state with Flax TrainState
def create_train_state(model, config, rng_key):
    """Create initial training state."""
    # Initialize model parameters
    init_params = model.init(rng_key, jnp.ones((1, 1), dtype=jnp.int32))

    # Create learning rate schedule
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.learning_rate,
        warmup_steps=config.warmup_steps,
        decay_steps=config.max_steps,
        end_value=config.learning_rate * 0.1
    )

    # Create optimizer
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

    # Create and return train state - ensure parameters have consistent structure
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=init_params,
        tx=optimizer
    )

# Data preparation functions
def prepare_datasets(config):
    """Load and prepare datasets"""
    # Load datasets
    wiki_dataset = load_dataset("karpathy/tiny_shakespeare", split="train")

    # Print the dataset structure to understand its format
    print("Dataset structure:", list(wiki_dataset.features.keys()))
    print("First example:", wiki_dataset[0])

    # Initialize tokenizer
    tokenizer = SentencePieceUnigramTokenizer()

    # Train the tokenizer
    # First check what columns are actually available
    column_names = list(wiki_dataset.features.keys())
    text_column = "text" if "text" in column_names else column_names[0]

    # Get a sample of texts for training the tokenizer
    # (for efficiency, we don't need to use the entire dataset)
    sample_texts = [
        example[text_column] for example in wiki_dataset.select(range(min(10000, len(wiki_dataset))))
        if isinstance(example[text_column], str) and example[text_column].strip()
    ]

    # Train the tokenizer
    tokenizer.train_from_iterator(
        sample_texts,
        vocab_size=config.vocab_size,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
    )

    # Save the tokenizer for later use
    tokenizer.save("gemma_tokenizer.json")

    def tokenize_function(example):
        # Make sure we're accessing the right column
        text = example[text_column]
        if isinstance(text, str) and text.strip():
            tokens = tokenizer.encode(text).ids
            return {"input_ids": tokens}
        return {"input_ids": []}

    # Tokenize dataset
    tokenized_dataset = wiki_dataset.map(
        tokenize_function,
        remove_columns=column_names,  # Remove all original columns
        batched=False  # Process one example at a time for better error handling
    )

    # Filter out empty examples
    tokenized_dataset = tokenized_dataset.filter(lambda x: len(x["input_ids"]) > 0)

    # Convert to a format suitable for training
    # Flatten all token sequences into one long sequence
    all_tokens = []
    for example in tokenized_dataset:
        all_tokens.extend(example["input_ids"])

    # Create a single "dataset" with just input_ids for easier slicing during training
    flattened_dataset = {"input_ids": all_tokens}

    return flattened_dataset, tokenizer

def get_batch(key, data, config):
    """Create a batch of data for training"""
    batch_size = config.batch_size
    seq_len = config.max_seq_len

    # Generate random starting indices
    total_tokens = len(data["input_ids"]) - seq_len - 1  # -1 for target shifting

    # Make sure we have enough tokens
    if total_tokens <= 0:
        raise ValueError(f"Not enough tokens in dataset. Found {len(data['input_ids'])}, need at least {seq_len + 2}")

    # Generate batch_size random starting points
    ix = random.randint(key, (batch_size,), 0, total_tokens)

    # Create input and target sequences
    x = jnp.stack([jnp.array(data["input_ids"][i:i+seq_len]) for i in ix])
    y = jnp.stack([jnp.array(data["input_ids"][i+1:i+seq_len+1]) for i in ix])

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

# Improved RNG key handling for multi-device setup
def create_device_rng_keys(rng, n_devices):
    """Create RNG keys for each device with proper shape for pmap"""
    # Split the key into n_devices + 1 keys
    keys = random.split(rng, n_devices + 1)
    # Return the next key and the device keys reshaped for pmap
    return keys[0], keys[1:]

def evaluate(model_apply_fn, params, eval_data, config, num_batches=10):
    """Evaluate model on validation data"""
    key = random.PRNGKey(42)
    total_loss = 0.0
    total_tokens = 0

    for i in range(num_batches):
        key, batch_key = random.split(key)
        inputs, targets = get_batch(batch_key, eval_data, config)
        batch_size = inputs.shape[0]

        # Forward pass
        if isinstance(params, dict) and 'params' in params:
            logits = model_apply_fn(params, inputs, deterministic=True)
        else:
            logits = model_apply_fn({'params': params}, inputs, deterministic=True)

        # Compute loss
        logits = logits.reshape(-1, logits.shape[-1])
        targets_flat = targets.reshape(-1)
        
        # Compute cross entropy loss with higher numerical stability
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits, targets_flat
        ).mean()

        total_loss += loss * batch_size
        total_tokens += batch_size

    avg_loss = total_loss / total_tokens
    perplexity = jnp.exp(avg_loss)

    metrics = {
        'loss': avg_loss,
        'perplexity': perplexity
    }

    return metrics

# Training functions
def train_step(state, batch, dropout_rng):
    """Single training step with improved stability"""
    inputs, targets = batch

    def loss_fn(params):
        # Forward pass with dropout
        logits = state.apply_fn(
            {'params': params}, 
            inputs, 
            deterministic=False,
            rngs={'dropout': dropout_rng}
        )

        # Compute loss with improved numerical stability
        logits = logits.reshape(-1, logits.shape[-1])
        targets_flat = targets.reshape(-1)
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits, targets_flat
        ).mean()

        return loss

    # Get gradients
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params['params'])

    # Update state with gradient clipping
    new_state = state.apply_gradients(grads={'params': grads})

    metrics = {
        'loss': loss,
    }

    return new_state, metrics

train_step_pmap = jax.pmap(train_step, axis_name='batch')

# Main training function with corrected multi-device support
def train_gemma(config, num_epochs=5, steps_per_epoch=1000, save_every=1000):
    """Train Gemma model"""
    # Initialize TPU
    n_devices = initialize_tpu()

    # Setup model
    model = Gemma(config)
    rng_key = random.PRNGKey(42)

    # Create training state
    state = create_train_state(model, config, rng_key)

    # Create learning rate schedule
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.learning_rate,
        warmup_steps=config.warmup_steps,
        decay_steps=config.max_steps,
        end_value=config.learning_rate * 0.1
    )

    # Setup checkpointing
    checkpoint_dir = os.path.abspath("gemma_checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpointer = ocp.PyTreeCheckpointer()
    options = ocp.CheckpointManagerOptions(
        max_to_keep=3,
        create=True
    )
    checkpoint_manager = ocp.CheckpointManager(
        checkpoint_dir,
        checkpointer,
        options
    )

    # Replicate the state across devices for multi-device training
    if n_devices > 1:
        state = jax.device_put_replicated(state, jax.devices())

    # Prepare datasets
    train_dataset, tokenizer = prepare_datasets(config)

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
            batch_data = get_batch(batch_key, train_dataset, config)

            # Training step
            if n_devices > 1:
                # Calculate per-device batch size
                per_device_batch = config.batch_size // n_devices
                if per_device_batch == 0:
                    raise ValueError(f"Batch size {config.batch_size} is too small for {n_devices} devices")

                # Reshape batch data for multi-device training
                inputs, targets = batch_data
                inputs = inputs.reshape((n_devices, per_device_batch, inputs.shape[1]))
                targets = targets.reshape((n_devices, per_device_batch, targets.shape[1]))
                batch_data = (inputs, targets)

                # Create per-device RNG keys
                rng_key, dropout_keys = create_device_rng_keys(rng_key, n_devices)

                # Apply pmapped training step
                state, metrics = train_step_pmap(state, batch_data, dropout_keys)
                loss = metrics['loss'].mean()  # Average loss across devices
            else:
                # Single device training
                rng_key, dropout_key = random.split(rng_key)
                state, metrics = train_step(state, batch_data, dropout_key)
                loss = metrics['loss']

            # Get current learning rate
            current_lr = lr_schedule(step)

            epoch_loss += loss
            step += 1

            # Log progress
            if step % 100 == 0:
                print(f"Step {step}/{total_steps}, Loss: {loss:.4f}, LR: {current_lr:.6f}")

            # Save checkpoint
            if step % save_every == 0:
                if n_devices > 1:
                    # For multi-device, save only the first copy
                    save_state = jax.tree.map(lambda x: x[0], state)
                else:
                    save_state = state

                # Save checkpoint using Orbax
                save_args = orbax_utils.save_args_from_target(save_state)
                checkpoint_manager.save(step, save_state, save_kwargs={'save_args': save_args})
                print(f"Checkpoint saved at step {step}")

                # Generate sample text
                if n_devices > 1:
                    sample_params = jax.tree.map(lambda x: x[0], state.params)
                else:
                    sample_params = state.params

                prompt = tokenizer.encode("Once upon a time").ids
                prompt_tensor = jnp.array([prompt])

                sample_rng = random.PRNGKey(step)
                # Check if sample_params already has a 'params' key
                if 'params' in sample_params:
                    generated = model.apply(
                        sample_params,
                        prompt_tensor,
                        max_new_tokens=50,
                        rng_key=sample_rng,
                        temperature=config.temperature,
                        top_k=config.top_k,
                        top_p=config.top_p,
                        method=model.generate
                    )
                else:
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
            eval_params = jax.tree.map(lambda x: x[0], state.params)
        else:
            eval_params = state.params

        # Validation loss and perplexity
        val_metrics = evaluate(model.apply, eval_params, train_dataset, config)
        print(f"Validation Loss: {val_metrics['loss']:.4f}, Perplexity: {val_metrics['perplexity']:.4f}")

    # Training complete - save final model
    if n_devices > 1:
        final_state = jax.tree.map(lambda x: x[0], state)
    else:
        final_state = state
    
    # Save final checkpoint
    save_args = orbax_utils.save_args_from_target(final_state)
    checkpoint_manager.save(step, final_state, save_kwargs={'save_args': save_args})
    print(f"Final model saved at step {step}")
    
    # Return the trained model and tokenizer
    return final_state, tokenizer

# Function to load a saved model
def load_model(checkpoint_dir, config):
    """Load a saved model from checkpoint"""
    model = Gemma(config)
    
    # Create checkpointer
    checkpointer = ocp.PyTreeCheckpointer()
    
    # Initialize an empty state structure for restore
    rng_key = random.PRNGKey(0)
    init_state = create_train_state(model, config, rng_key)
    
    # Find latest checkpoint
    checkpoint_manager = ocp.CheckpointManager(checkpoint_dir, checkpointer)
    latest_step = checkpoint_manager.latest_step()
    
    if latest_step is None:
        raise ValueError(f"No checkpoints found in {checkpoint_dir}")
    
    # Restore from checkpoint
    restored_state = checkpoint_manager.restore(latest_step, init_state)
    print(f"Model restored from step {latest_step}")
    
    return model, restored_state

if __name__ == "__main__":
    # Initialize config
    config = GemmaConfig()
    
    # Train the model or load a pre-trained one
    import argparse
    import sys
    
    # Check if running in Jupyter/Colab
    in_jupyter = any('jupyter' in arg or 'colab' in arg for arg in sys.argv)
    
    if in_jupyter:
        # If in Jupyter/Colab, don't parse arguments
        print("Running in Jupyter/Colab environment")
        
        # Default behavior - you can modify this as needed
        print("Training new model...")
        final_state, tokenizer = train_gemma(config, num_epochs=3, steps_per_epoch=10)
        
    else:
        # Normal command line execution
        parser = argparse.ArgumentParser(description="Train or use a Gemma model")
        parser.add_argument("--train", action="store_true", help="Train a new model")
        parser.add_argument("--generate", action="store_true", help="Generate text with a trained model")
        parser.add_argument("--prompt", type=str, default="Once upon a time", help="Prompt for text generation")
        parser.add_argument("--max_tokens", type=int, default=100, help="Maximum tokens to generate")
        args = parser.parse_args()
        
        checkpoint_dir = os.path.abspath("gemma_checkpoints")
        
        if args.train:
            # Train new model
            print("Training new model...")
            final_state, tokenizer = train_gemma(config, num_epochs=3, steps_per_epoch=10)
            
        elif args.generate:
            # Load pre-trained model
            print("Loading pre-trained model...")
            try:
                model, state = load_model(checkpoint_dir, config)
                tokenizer = SentencePieceUnigramTokenizer.from_file("gemma_tokenizer.json")
                
                # Generate text
                prompt = tokenizer.encode(args.prompt).ids
                prompt_tensor = jnp.array([prompt])
                
                rng_key = random.PRNGKey(0)
                params = state.params
                if 'params' in params:
                    actual_params = params
                else:
                    actual_params = {'params': params}
                    
                generated = model.apply(
                    actual_params,
                    prompt_tensor,
                    max_new_tokens=args.max_tokens,
                    rng_key=rng_key,
                    temperature=config.temperature,
                    top_k=config.top_k,
                    top_p=config.top_p,
                    method=model.generate
                )
                
                generated_text = tokenizer.decode(generated[0].tolist())
                print(f"\nGenerated text:\n{generated_text}\n")
                
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Please train a model first or specify a valid checkpoint directory.")
        
        else:
            print("Please specify either --train or --generate")
