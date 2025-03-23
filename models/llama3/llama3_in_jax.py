import os
import math
import pickle
from functools import partial
from tqdm import tqdm
from typing import Any, Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import random, lax
import flax.linen as nn
import optax
from flax.training import train_state, orbax_utils
import orbax.checkpoint as ocp
from datasets import load_dataset
from tokenizers import SentencePieceUnigramTokenizer

# -----------------------------------------------------------------------------
# Environment Setup
# -----------------------------------------------------------------------------
os.environ['JAX_PLATFORM_NAME'] = 'tpu'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
print("JAX devices:", jax.devices())

# -----------------------------------------------------------------------------
# Model Configuration
# -----------------------------------------------------------------------------
class LLaMAConfig:
    """Configuration for LLaMA model"""
    vocab_size: int = 32000
    dim: int = 512            # Hidden dimension
    n_layers: int = 8         # Number of transformer layers
    n_heads: int = 8          # Number of attention heads
    n_kv_heads: int = 4       # Number of key/value heads (for grouped-query attention)
    max_seq_len: int = 2048   # Maximum sequence length
    dropout_rate: float = 0.0 # Dropout rate

    # RoPE settings
    rope_theta: float = 10000.0

    # Training settings
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 1000
    max_steps: int = 100000

    # Generation settings
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.95

# -----------------------------------------------------------------------------
# Utility Layers and Functions
# -----------------------------------------------------------------------------
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

def precompute_freqs_cis(dim: int, max_seq_len: int, theta: float = 10000.0):
    """Precompute rotary frequency tensor for complex exponentials."""
    freqs = 1.0 / (theta ** (jnp.arange(0, dim // 2, dtype=jnp.float32) / dim))
    t = jnp.arange(max_seq_len, dtype=jnp.float32)
    freqs = jnp.outer(t, freqs)
    return jnp.complex64(jnp.exp(1j * freqs))

def apply_rotary_emb(xq, xk, freqs_cis):
    """Apply rotary embeddings to query and key tensors."""
    xq_r = jnp.reshape(xq, (*xq.shape[:-1], -1, 2))
    xk_r = jnp.reshape(xk, (*xk.shape[:-1], -1, 2))
    xq_complex = jnp.complex64(xq_r[..., 0] + 1j * xq_r[..., 1])
    xk_complex = jnp.complex64(xk_r[..., 0] + 1j * xk_r[..., 1])
    freqs_cis = jnp.reshape(freqs_cis, (1, freqs_cis.shape[0], 1, freqs_cis.shape[1]))
    xq_out = xq_complex * freqs_cis
    xk_out = xk_complex * freqs_cis
    xq = jnp.stack([jnp.real(xq_out), jnp.imag(xq_out)], axis=-1).reshape(xq.shape)
    xk = jnp.stack([jnp.real(xk_out), jnp.imag(xk_out)], axis=-1).reshape(xk.shape)
    return xq, xk

@partial(jax.jit)
def flash_attention(q, k, v, mask=None, scale=None):
    """Optimized attention mechanism using JAX primitives."""
    batch_size, num_heads, seq_len, head_dim = q.shape
    if scale is None:
        scale = 1.0 / jnp.sqrt(head_dim)
    scores = jnp.einsum('bhid,bhjd->bhij', q, k) * scale
    if mask is not None:
        scores = scores + mask
    scores_max = jnp.max(scores, axis=-1, keepdims=True)
    scores = scores - lax.stop_gradient(scores_max)
    attn_weights = jnp.exp(scores)
    attn_weights = attn_weights / jnp.sum(attn_weights, axis=-1, keepdims=True)
    output = jnp.einsum('bhij,bhjd->bhid', attn_weights, v)
    return output

def swiglu(x, w1, w2, w3):
    """SwiGLU activation function."""
    return w2(jax.nn.silu(w3(x)) * w1(x))

# -----------------------------------------------------------------------------
# Model Modules
# -----------------------------------------------------------------------------
class LLaMACausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with support for grouped-query attention."""
    config: LLaMAConfig

    def setup(self):
        cfg = self.config
        head_dim = cfg.dim // cfg.n_heads
        self.wq = nn.Dense(cfg.n_heads * head_dim,
                           kernel_init=nn.initializers.variance_scaling(1.0, 'fan_in', 'normal'))
        self.wk = nn.Dense(cfg.n_kv_heads * head_dim,
                           kernel_init=nn.initializers.variance_scaling(1.0, 'fan_in', 'normal'))
        self.wv = nn.Dense(cfg.n_kv_heads * head_dim,
                           kernel_init=nn.initializers.variance_scaling(1.0, 'fan_in', 'normal'))
        self.wo = nn.Dense(cfg.dim,
                           kernel_init=nn.initializers.variance_scaling(1.0, 'fan_in', 'normal'))
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)

    def __call__(self, x, freqs_cis, mask=None, deterministic=True):
        B, T, C = x.shape
        cfg = self.config
        head_dim = C // cfg.n_heads

        q = self.wq(x).reshape(B, T, cfg.n_heads, head_dim)
        k = self.wk(x).reshape(B, T, cfg.n_kv_heads, head_dim)
        v = self.wv(x).reshape(B, T, cfg.n_kv_heads, head_dim)

        # Apply QK normalization
        q = jnp.swapaxes(self.q_norm(jnp.swapaxes(q, 1, 2)), 1, 2)
        k = jnp.swapaxes(self.k_norm(jnp.swapaxes(k, 1, 2)), 1, 2)

        q, k = apply_rotary_emb(q, k, freqs_cis[:T])

        if cfg.n_heads > cfg.n_kv_heads:
            k = jnp.repeat(k, cfg.n_heads // cfg.n_kv_heads, axis=2)
            v = jnp.repeat(v, cfg.n_heads // cfg.n_kv_heads, axis=2)

        q, k, v = map(lambda x: jnp.swapaxes(x, 1, 2), (q, k, v))
        output = flash_attention(q, k, v, mask)
        output = jnp.swapaxes(output, 1, 2).reshape(B, T, -1)
        output = self.wo(output)
        return output

class LLaMAMLP(nn.Module):
    """Feed-forward network with SwiGLU activation."""
    config: LLaMAConfig

    def setup(self):
        dim = self.config.dim
        hidden_dim = 4 * dim
        self.w1 = nn.Dense(hidden_dim,
                           kernel_init=nn.initializers.variance_scaling(1.0, 'fan_in', 'normal'))
        self.w2 = nn.Dense(dim,
                           kernel_init=nn.initializers.variance_scaling(1.0, 'fan_in', 'normal'))
        self.w3 = nn.Dense(hidden_dim,
                           kernel_init=nn.initializers.variance_scaling(1.0, 'fan_in', 'normal'))

    def __call__(self, x):
        return swiglu(x, self.w1, self.w2, self.w3)

class LLaMABlock(nn.Module):
    """Transformer block for LLaMA model."""
    config: LLaMAConfig

    def setup(self):
        self.attention_norm = RMSNorm(self.config.dim)
        self.attention = LLaMACausalSelfAttention(self.config)
        self.ffn_norm = RMSNorm(self.config.dim)
        self.ffn = LLaMAMLP(self.config)
        self.dropout = nn.Dropout(self.config.dropout_rate)

    def __call__(self, x, freqs_cis, mask=None, deterministic=True):
        h = x + self.dropout(
            self.attention(self.attention_norm(x), freqs_cis, mask, deterministic),
            deterministic=deterministic
        )
        out = h + self.dropout(
            self.ffn(self.ffn_norm(h)),
            deterministic=deterministic
        )
        return out

# -----------------------------------------------------------------------------
# Full LLaMA Model Definition
# -----------------------------------------------------------------------------
class LLaMA3(nn.Module):
    """LLaMA language model."""
    config: LLaMAConfig

    def setup(self):
        cfg = self.config
        self.token_embedding = nn.Embed(cfg.vocab_size, cfg.dim,
                                        embedding_init=nn.initializers.normal(stddev=0.02))
        self.blocks = [LLaMABlock(cfg) for _ in range(cfg.n_layers)]
        self.norm_f = RMSNorm(cfg.dim)
        self.lm_head = nn.Dense(cfg.vocab_size,
                                kernel_init=nn.initializers.normal(stddev=0.02),
                                use_bias=False)
        self.freqs_cis = precompute_freqs_cis(cfg.dim // cfg.n_heads, cfg.max_seq_len, cfg.rope_theta)

    def __call__(self, input_ids, deterministic=True):
        B, T = input_ids.shape
        mask = jnp.tril(jnp.ones((self.config.max_seq_len, self.config.max_seq_len)))
        mask = jnp.where(mask == 0, jnp.finfo(jnp.float32).min, 0.0)
        mask = mask[None, None, :T, :T]

        h = self.token_embedding(input_ids)
        for block in self.blocks:
            h = block(h, self.freqs_cis, mask, deterministic)
        h = self.norm_f(h)
        logits = self.lm_head(h)
        return logits

    def generate(self, input_ids, max_new_tokens, rng_key, temperature=0.8, top_k=40, top_p=0.95):
        B, T = input_ids.shape
        output = input_ids
        for _ in range(max_new_tokens):
            curr_input = output[:, -self.config.max_seq_len:]
            logits = self(curr_input, deterministic=True)[:, -1, :]
            logits = logits / temperature

            if top_k > 0:
                top_k_v, top_k_i = jax.lax.top_k(logits, top_k)
                indices_to_remove = jnp.broadcast_to(
                    jnp.arange(logits.shape[-1]) < top_k_i[:, -1:],
                    logits.shape
                )
                logits = jnp.where(indices_to_remove, logits, jnp.finfo(jnp.float32).min)

            if top_p < 1.0:
                sorted_indices = jnp.argsort(logits, axis=-1)[:, ::-1]
                sorted_logits = jnp.take_along_axis(logits, sorted_indices, axis=-1)
                cumulative_probs = jnp.cumsum(jax.nn.softmax(sorted_logits, axis=-1), axis=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove = jnp.roll(sorted_indices_to_remove, 1, axis=1)
                sorted_indices_to_remove = sorted_indices_to_remove.at[:, 0].set(False)
                indices_to_remove = jnp.zeros_like(logits, dtype=bool)
                indices_to_remove = indices_to_remove.at[jnp.arange(B)[:, None], sorted_indices].set(sorted_indices_to_remove)
                logits = jnp.where(indices_to_remove, jnp.finfo(jnp.float32).min, logits)

            rng_key, sample_key = random.split(rng_key)
            next_token = random.categorical(sample_key, logits, shape=(B,))
            output = jnp.concatenate([output, next_token[:, None]], axis=1)
        return output

# -----------------------------------------------------------------------------
# Training, Evaluation and Utility Functions
# -----------------------------------------------------------------------------
def create_train_state(model, config, rng_key):
    """Create initial training state."""
    init_params = model.init(rng_key, jnp.ones((1, 1), dtype=jnp.int32))
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
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=init_params,
        tx=optimizer
    )

def prepare_datasets(config):
    """Load and prepare the dataset and tokenizer."""
    wiki_dataset = load_dataset("karpathy/tiny_shakespeare", split="train")
    print("Dataset structure:", list(wiki_dataset.features.keys()))
    print("First example:", wiki_dataset[0])
    tokenizer = SentencePieceUnigramTokenizer()
    column_names = list(wiki_dataset.features.keys())
    text_column = "text" if "text" in column_names else column_names[0]
    sample_texts = [example[text_column] for example in wiki_dataset.select(range(min(10000, len(wiki_dataset))))
                    if isinstance(example[text_column], str) and example[text_column].strip()]
    tokenizer.train_from_iterator(
        sample_texts,
        vocab_size=config.vocab_size,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
    )
    tokenizer.save("llama_tokenizer.json")
    def tokenize_function(example):
        text = example[text_column]
        if isinstance(text, str) and text.strip():
            tokens = tokenizer.encode(text).ids
            return {"input_ids": tokens}
        return {"input_ids": []}
    tokenized_dataset = wiki_dataset.map(tokenize_function, remove_columns=column_names, batched=False)
    tokenized_dataset = tokenized_dataset.filter(lambda x: len(x["input_ids"]) > 0)
    all_tokens = []
    for example in tokenized_dataset:
        all_tokens.extend(example["input_ids"])
    flattened_dataset = {"input_ids": all_tokens}
    return flattened_dataset, tokenizer

def get_batch(key, data, config):
    """Generate a training batch."""
    batch_size = config.batch_size
    seq_len = config.max_seq_len
    total_tokens = len(data["input_ids"]) - seq_len - 1
    if total_tokens <= 0:
        raise ValueError(f"Not enough tokens in dataset. Found {len(data['input_ids'])}, need at least {seq_len + 2}")
    ix = random.randint(key, (batch_size,), 0, total_tokens)
    x = jnp.stack([jnp.array(data["input_ids"][i:i+seq_len]) for i in ix])
    y = jnp.stack([jnp.array(data["input_ids"][i+1:i+seq_len+1]) for i in ix])
    return x, y

def initialize_tpu():
    """Initialize TPU devices if available."""
    devices = jax.devices()
    n_devices = len(devices)
    print(f"Found {n_devices} JAX devices: {devices}")
    if any(d.platform == 'tpu' for d in devices):
        print("TPU devices detected. Setting up for distributed training.")
    else:
        print("No TPU devices found. Using available CPU/GPU.")
    return n_devices

def create_device_rng_keys(rng, n_devices):
    """Create RNG keys for each device."""
    keys = random.split(rng, n_devices + 1)
    return keys[0], keys[1:]

def train_step(state, batch, dropout_rng):
    """Single training step."""
    inputs, targets = batch
    def loss_fn(params):
        logits = state.apply_fn({"params": params}, inputs, deterministic=False, rngs={'dropout': dropout_rng})
        logits = logits.reshape(-1, logits.shape[-1])
        targets_flat = targets.reshape(-1)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets_flat).mean()
        return loss
    grad_fn = jax.value_and_grad(loss_fn)
    actual_params = state.params["params"] if "params" in state.params else state.params
    loss, grads = grad_fn(actual_params)
    wrapped_grads = {"params": grads} if "params" in state.params else grads
    new_state = state.apply_gradients(grads=wrapped_grads)
    return new_state, loss

train_step_jit = jax.jit(train_step)

def train_step_pmap_wrapper(state, batch, dropout_rng):
    return train_step(state, batch, dropout_rng)

train_step_pmap = jax.pmap(train_step_pmap_wrapper, axis_name='batch')

def evaluate(model_apply_fn, params, eval_data, config, num_batches=10):
    key = random.PRNGKey(42)
    total_loss = 0.0
    for _ in range(num_batches):
        key, batch_key = random.split(key)
        inputs, targets = get_batch(batch_key, eval_data, config)
        if isinstance(params, dict) and 'params' in params:
            logits = model_apply_fn(params, inputs, deterministic=True)
        else:
            logits = model_apply_fn({'params': params}, inputs, deterministic=True)
        logits = logits.reshape(-1, logits.shape[-1])
        targets_flat = targets.reshape(-1)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets_flat).mean()
        total_loss += loss
    avg_loss = total_loss / num_batches
    perplexity = jnp.exp(avg_loss)
    return avg_loss, perplexity

def train_llama(config, num_epochs=5, steps_per_epoch=1000, save_every=1000):
    """Train LLaMA model."""
    n_devices = initialize_tpu()
    model = LLaMA3(config)
    rng_key = random.PRNGKey(42)
    state = create_train_state(model, config, rng_key)
    if n_devices > 1:
        state = jax.device_put_replicated(state, jax.devices())
    train_dataset, tokenizer = prepare_datasets(config)
    checkpoint_dir = os.path.abspath("llama_checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpointer = ocp.PyTreeCheckpointer()
    options = ocp.CheckpointManagerOptions(max_to_keep=3, create=True)
    checkpoint_manager = ocp.CheckpointManager(checkpoint_dir, checkpointer, options)
    rng_key = random.PRNGKey(0)
    step = 0
    total_steps = num_epochs * steps_per_epoch
    print(f"Starting training for {num_epochs} epochs ({total_steps} steps)")
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for _ in tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{num_epochs}"):
            rng_key, batch_key = random.split(rng_key)
            batch_data = get_batch(batch_key, train_dataset, config)
            if n_devices > 1:
                per_device_batch = config.batch_size // n_devices
                if per_device_batch == 0:
                    raise ValueError(f"Batch size {config.batch_size} is too small for {n_devices} devices")
                inputs, targets = batch_data
                inputs = inputs.reshape((n_devices, per_device_batch, inputs.shape[1]))
                targets = targets.reshape((n_devices, per_device_batch, targets.shape[1]))
                batch_data = (inputs, targets)
                rng_key, dropout_keys = create_device_rng_keys(rng_key, n_devices)
                state, loss = train_step_pmap(state, batch_data, dropout_keys)
                loss = jnp.mean(loss)
            else:
                rng_key, dropout_key = random.split(rng_key)
                state, loss = train_step_jit(state, batch_data, dropout_key)
            epoch_loss += loss
            step += 1
            if step % 100 == 0:
                print(f"Step {step}/{total_steps}, Loss: {loss:.4f}")
            if step % save_every == 0:
                save_state = jax.tree_map(lambda x: x[0], state) if n_devices > 1 else state
                save_args = orbax_utils.save_args_from_target(save_state)
                checkpoint_manager.save(step, save_state, save_kwargs={'save_args': save_args})
                print(f"Checkpoint saved at step {step}")
                sample_params = jax.tree_map(lambda x: x[0], state.params) if n_devices > 1 else state.params
                prompt = tokenizer.encode("Once upon a time").ids
                prompt_tensor = jnp.array([prompt])
                sample_rng = random.PRNGKey(step)
                if 'params' in sample_params:
                    generated = model.apply(sample_params, prompt_tensor, max_new_tokens=50,
                                            rng_key=sample_rng, temperature=config.temperature,
                                            top_k=config.top_k, top_p=config.top_p, method=model.generate)
                else:
                    generated = model.apply({"params": sample_params}, prompt_tensor, max_new_tokens=50,
                                            rng_key=sample_rng, temperature=config.temperature,
                                            top_k=config.top_k, top_p=config.top_p, method=model.generate)
                generated_text = tokenizer.decode(generated[0].tolist())
                print(f"\nSample generation at step {step}:\n{generated_text}\n")
        avg_epoch_loss = epoch_loss / steps_per_epoch
        print(f"Epoch {epoch+1} complete. Average loss: {avg_epoch_loss:.4f}")
        eval_params = jax.tree_map(lambda x: x[0], state.params) if n_devices > 1 else state.params
        val_loss, perplexity = evaluate(model.apply, eval_params, train_dataset, config)
        print(f"Validation Loss: {val_loss:.4f}, Perplexity: {perplexity:.2f}")
    final_state = jax.tree_map(lambda x: x[0], state) if n_devices > 1 else state
    save_args = orbax_utils.save_args_from_target(final_state)
    checkpoint_manager.save(total_steps, final_state, save_kwargs={'save_args': save_args})
    print("Training complete. Final model saved.")
    return final_state

def load_checkpoint(checkpoint_dir, step=None):
    checkpoint_dir = os.path.abspath(checkpoint_dir)
    checkpointer = ocp.PyTreeCheckpointer()
    options = ocp.CheckpointManagerOptions(create=False)
    checkpoint_manager = ocp.CheckpointManager(checkpoint_dir, checkpointer, options)
    if step is None:
        step = checkpoint_manager.latest_step()
        if step is None:
            raise ValueError(f"No checkpoints found in {checkpoint_dir}")
    model = LLaMA3(LLaMAConfig())
    rng_key = random.PRNGKey(0)
    dummy_state = create_train_state(model, LLaMAConfig(), rng_key)
    restored_state = checkpoint_manager.restore(step, dummy_state)
    print(f"Restored checkpoint from step {step}")
    return restored_state, step

def generate_text(model, params, tokenizer, prompt, max_new_tokens=100, temperature=0.8):
    if not isinstance(prompt, str):
        prompt = str(prompt)
    prompt_tokens = tokenizer.encode(prompt).ids
    prompt_tensor = jnp.array([prompt_tokens])
    rng_key = random.PRNGKey(0)
    if 'params' in params:
        generated = model.apply(params, prompt_tensor, max_new_tokens=max_new_tokens,
                                rng_key=rng_key, temperature=temperature,
                                top_k=40, top_p=0.95, method=model.generate)
    else:
        generated = model.apply({"params": params}, prompt_tensor, max_new_tokens=max_new_tokens,
                                rng_key=rng_key, temperature=temperature,
                                top_k=40, top_p=0.95, method=model.generate)
    generated_text = tokenizer.decode(generated[0].tolist())
    return generated_text

# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    config = LLaMAConfig()
    checkpoint_dir = os.path.abspath("llama_checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    final_state = train_llama(config, num_epochs=5, steps_per_epoch=10, save_every=10)
    model = LLaMA3(config)
    tokenizer = prepare_datasets(config)[1]
    prompt = "In a distant galaxy"
    generated_text = generate_text(model, final_state.params, tokenizer, prompt)
    print("\nGenerated text:")
    print(generated_text)
