import optax
import flax.linen as nn
import jax.numpy as jnp
from typing import Any, Callable, Dict, Optional, Tuple, List
from flax.training import train_state
from functools import partial
import os
from datasets import load_dataset
from tqdm import tqdm

class Llama3Config:
    """Configuration for the Llama3 model."""
    def __init__(
        self,
        vocab_size: int,
        context_length: int = 4096,
        hidden_size: int = 4096,
        intermediate_size: int = 11008,  # 4 * hidden / 1.5 (Llama3 uses an unusual MLP expansion factor)
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,  # For grouped-query attention
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
        rope_scaling: Optional[Dict[str, float]] = None,
        dropout_rate: float = 0.0,
        attention_dropout: float = 0.0,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.1,
        beta1: float = 0.9,
        beta2: float = 0.95,
        grad_clip: float = 1.0,
        warmup_steps: int = 2000,
        total_steps: int = 250000,
    ):
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling if rope_scaling else {"type": "linear", "factor": 1.0}
        self.dropout_rate = dropout_rate
        self.attention_dropout = attention_dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.grad_clip = grad_clip
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        
        # Derived attributes
        self.head_dim = hidden_size // num_attention_heads
        self.kv_heads = num_key_value_heads
        self.kv_dim = num_key_value_heads * self.head_dim
        self.groups = num_attention_heads // num_key_value_heads

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    epsilon: float = 1e-6
    
    @nn.compact
    def __call__(self, x):
        # Shape of x: (batch_size, seq_len, hidden_size)
        # Calculate RMS
        variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        x = x * jax.lax.rsqrt(variance + self.epsilon)
        
        # Scale and shift
        scale = self.param('scale', nn.initializers.ones, (x.shape[-1],))
        return x * scale

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """Precompute the frequency tensor for complex exponentials (cos, sin) used in RoPE."""
    freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2)[: (dim // 2)] / dim))
    t = jnp.arange(end)
    freqs = jnp.outer(t, freqs)
    # Complex exponential: cos(x) + i sin(x) in JAX friendly way
    freqs_cos = jnp.cos(freqs)
    freqs_sin = jnp.sin(freqs)
    return freqs_cos, freqs_sin

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    """Apply rotary position embeddings to q and k tensors."""
    # q, k: (batch, seq_len, num_heads, head_dim)
    # cos, sin: (seq_len, head_dim/2)
    
    # If position_ids not provided, assume sequential
    if position_ids is None:
        position_ids = jnp.arange(q.shape[1])
    
    # Get the cos and sin for the positions we need
    cos = cos[position_ids]  # Shape: (seq_len, head_dim/2)
    sin = sin[position_ids]  # Shape: (seq_len, head_dim/2)
    
    # Reshape for broadcasting: (1, seq_len, 1, head_dim/2)
    cos = jnp.expand_dims(jnp.expand_dims(cos, 0), 2)
    sin = jnp.expand_dims(jnp.expand_dims(sin, 0), 2)
    
    # Split q and k into real and imaginary parts (even and odd indices)
    q_even = q[..., ::2]
    q_odd = q[..., 1::2]
    k_even = k[..., ::2]
    k_odd = k[..., 1::2]
    
    # Apply complex number multiplication
    q_out_even = q_even * cos - q_odd * sin
    q_out_odd = q_odd * cos + q_even * sin
    k_out_even = k_even * cos - k_odd * sin
    k_out_odd = k_odd * cos + k_even * sin
    
    # Interleave the even and odd parts back together
    q_out = jnp.zeros_like(q)
    k_out = jnp.zeros_like(k)
    q_out = q_out.at[..., ::2].set(q_out_even)
    q_out = q_out.at[..., 1::2].set(q_out_odd)
    k_out = k_out.at[..., ::2].set(k_out_even)
    k_out = k_out.at[..., 1::2].set(k_out_odd)
    
    return q_out, k_out

class Llama3Attention(nn.Module):
    """Multi-head attention with RoPE, multi-query attention option, and optimized implementation."""
    config: Llama3Config
    
    def setup(self):
        config = self.config
        hidden_size = config.hidden_size
        head_dim = config.head_dim
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Dense(
            config.num_attention_heads * head_dim,
            kernel_init=nn.initializers.normal(stddev=0.02),
            bias_init=nn.initializers.zeros,
            use_bias=False
        )
        
        # For grouped-query attention
        self.k_proj = nn.Dense(
            config.kv_dim,
            kernel_init=nn.initializers.normal(stddev=0.02),
            bias_init=nn.initializers.zeros,
            use_bias=False
        )
        
        self.v_proj = nn.Dense(
            config.kv_dim,
            kernel_init=nn.initializers.normal(stddev=0.02),
            bias_init=nn.initializers.zeros,
            use_bias=False
        )
        
        # Output projection
        self.o_proj = nn.Dense(
            hidden_size,
            kernel_init=nn.initializers.normal(stddev=0.02),
            bias_init=nn.initializers.zeros,
            use_bias=False
        )
        
        # Attention dropout
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        
        # For RoPE (Rotary Position Embedding)
        freqs_cos, freqs_sin = precompute_freqs_cis(
            head_dim, 
            config.context_length * 2,  # Doubled for potential RoPE scaling
            config.rope_theta
        )
        self.freqs_cos = freqs_cos
        self.freqs_sin = freqs_sin
        
    def __call__(self, hidden_states, attention_mask=None, position_ids=None, training=False):
        """
        hidden_states: (batch_size, seq_len, hidden_size)
        attention_mask: (batch_size, 1, seq_len, seq_len)
        position_ids: (batch_size, seq_len)
        """
        config = self.config
        batch_size, seq_length, _ = hidden_states.shape
        
        # Linear projections
        q = self.q_proj(hidden_states)  # (batch, seq_len, num_heads * head_dim)
        k = self.k_proj(hidden_states)  # (batch, seq_len, kv_dim)
        v = self.v_proj(hidden_states)  # (batch, seq_len, kv_dim)
        
        # Reshape for multi-head attention
        q = q.reshape(batch_size, seq_length, config.num_attention_heads, config.head_dim)
        k = k.reshape(batch_size, seq_length, config.num_key_value_heads, config.head_dim)
        v = v.reshape(batch_size, seq_length, config.num_key_value_heads, config.head_dim)
        
        # Apply RoPE to q and k
        q, k = apply_rotary_pos_emb(
            q, k, 
            self.freqs_cos[:seq_length], 
            self.freqs_sin[:seq_length],
            position_ids
        )
        
        # For grouped-query attention: repeat k and v for each query group
        if config.num_key_value_heads < config.num_attention_heads:
            k = jnp.repeat(k, config.groups, axis=2)  # (batch, seq_len, n_heads, head_dim)
            v = jnp.repeat(v, config.groups, axis=2)  # (batch, seq_len, n_heads, head_dim)
            
        # Transpose to (batch, num_heads, seq_len, head_dim)
        q = jnp.transpose(q, (0, 2, 1, 3))  # (batch, n_heads, seq_len, head_dim)
        k = jnp.transpose(k, (0, 2, 1, 3))  # (batch, n_heads, seq_len, head_dim)
        v = jnp.transpose(v, (0, 2, 1, 3))  # (batch, n_heads, seq_len, head_dim)
        
        # Compute attention scores
        scale = jnp.sqrt(config.head_dim)
        attention_scores = (q @ jnp.transpose(k, (0, 1, 3, 2))) / scale
        
        # Apply causal mask (if not provided, create one)
        if attention_mask is None:
            # Create causal mask that allows tokens to attend only to previous positions
            causal_mask = jnp.tril(jnp.ones((seq_length, seq_length)))
            attention_mask = jnp.reshape(causal_mask, (1, 1, seq_length, seq_length))
            
        # Apply attention mask
        attention_scores = jnp.where(
            attention_mask < 0.5, 
            jnp.full_like(attention_scores, -1e10), 
            attention_scores
        )
        
        # Softmax attention weights
        attention_weights = jax.nn.softmax(attention_scores, axis=-1)
        # Apply attention dropout
        attention_weights = self.attn_dropout(attention_weights, deterministic=not training)
        
        # Apply attention to values
        context_layer = attention_weights @ v  # (batch, n_heads, seq_len, head_dim)
        
        # Transpose and reshape back
        context_layer = jnp.transpose(context_layer, (0, 2, 1, 3))  # (batch, seq_len, n_heads, head_dim)
        context_layer = context_layer.reshape(batch_size, seq_length, -1)  # (batch, seq_len, hidden_size)
        
        # Output projection
        output = self.o_proj(context_layer)
        
        return output

class SwiGLU(nn.Module):
    """SwiGLU activation as used in Llama3."""
    config: Llama3Config
    
    def setup(self):
        config = self.config
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size
        
        # Gate and up projections
        self.gate_proj = nn.Dense(
            intermediate_size,
            kernel_init=nn.initializers.normal(stddev=0.02),
            bias_init=nn.initializers.zeros,
            use_bias=False
        )
        self.up_proj = nn.Dense(
            intermediate_size,
            kernel_init=nn.initializers.normal(stddev=0.02),
            bias_init=nn.initializers.zeros,
            use_bias=False
        )
        
        # Down projection
        self.down_proj = nn.Dense(
            hidden_size,
            kernel_init=nn.initializers.normal(stddev=0.02),
            bias_init=nn.initializers.zeros,
            use_bias=False
        )
        
        self.dropout = nn.Dropout(config.dropout_rate)
        
    def __call__(self, x, training=False):
        # Gate path with SwiGLU activation
        gate = self.gate_proj(x)
        gate = jax.nn.swish(gate)  # SwiGLU uses swish instead of GELU
        
        # Up projection path
        up = self.up_proj(x)
        
        # Combine with element-wise multiplication
        intermediate = gate * up
        
        # Down projection
        output = self.down_proj(intermediate)
        output = self.dropout(output, deterministic=not training)
        
        return output

class Llama3DecoderLayer(nn.Module):
    """Transformer decoder layer with pre-layer normalization design."""
    config: Llama3Config
    
    def setup(self):
        config = self.config
        
        # Pre-attention norm and attention
        self.input_layernorm = RMSNorm(epsilon=config.rms_norm_eps)
        self.self_attn = Llama3Attention(config)
        
        # Pre-FFN norm and feed-forward network
        self.post_attention_layernorm = RMSNorm(epsilon=config.rms_norm_eps)
        self.mlp = SwiGLU(config)
        
        # Residual dropout
        self.dropout = nn.Dropout(config.dropout_rate)
        
    def __call__(self, hidden_states, attention_mask=None, position_ids=None, training=False):
        # Self-attention block with residual connection
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states, 
            attention_mask=attention_mask, 
            position_ids=position_ids,
            training=training
        )
        hidden_states = self.dropout(hidden_states, deterministic=not training)
        hidden_states = residual + hidden_states
        
        # Feed-forward block with residual connection
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, training=training)
        hidden_states = residual + hidden_states
        
        return hidden_states

class Llama3Model(nn.Module):
    """Llama3 base model before final output layer."""
    config: Llama3Config
    
    def setup(self):
        config = self.config
        
        # Token embeddings
        self.embed_tokens = nn.Embed(
            config.vocab_size,
            config.hidden_size,
            embedding_init=nn.initializers.normal(stddev=0.02)
        )
        
        # Decoder layers
        self.layers = [Llama3DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        
        # Final layer norm
        self.norm = RMSNorm(epsilon=config.rms_norm_eps)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout_rate)
        
    def __call__(
        self, 
        input_ids, 
        attention_mask=None, 
        position_ids=None, 
        training=False
    ):
        # Get input embeddings
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = self.dropout(hidden_states, deterministic=not training)
        
        # Create causal attention mask if not provided
        if attention_mask is None:
            batch_size, seq_length = input_ids.shape
            attention_mask = jnp.ones((batch_size, seq_length))
            
        # Convert 2D mask to 4D mask for attention
        extended_attention_mask = attention_mask[:, None, None, :]
        # Adjust mask values: convert 0 to large negative value for softmax masking
        extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
        
        # Default position IDs if not provided
        if position_ids is None:
            position_ids = jnp.expand_dims(jnp.arange(input_ids.shape[1]), axis=0)
            position_ids = jnp.broadcast_to(position_ids, input_ids.shape)
        
        # Process through decoder layers
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=extended_attention_mask,
                position_ids=position_ids,
                training=training
            )
        
        # Final layer norm
        hidden_states = self.norm(hidden_states)
        
        return hidden_states

class Llama3ForCausalLM(nn.Module):
    """Llama3 model with language modeling head."""
    config: Llama3Config
    
    def setup(self):
        # Base model
        self.model = Llama3Model(self.config)
        
        # LM head
        self.lm_head = nn.Dense(
            self.config.vocab_size,
            kernel_init=nn.initializers.normal(stddev=0.02),
            bias_init=nn.initializers.zeros,
            use_bias=False
        )
        
        # For weight tying
        self.apply_weight_tying = True
        
    def _tie_weights(self, params):
        """Tie embedding weights with output layer if enabled."""
        if not self.apply_weight_tying:
            return params
        
        # Create a new parameter dictionary and update lm_head kernel
        new_params = params.copy()
        new_params['lm_head']['kernel'] = jnp.transpose(new_params['model']['embed_tokens']['embedding'])
        return new_params
        
    def __call__(
        self, 
        input_ids, 
        targets=None, 
        attention_mask=None, 
        position_ids=None, 
        training=False,
        params=None
    ):
        # Apply weight tying if enabled
        if params is not None and self.apply_weight_tying and not training:
            params = self._tie_weights(params)
            
        # Get transformer outputs
        hidden_states = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            training=training
        )
        
        # Project to vocabulary
        logits = self.lm_head(hidden_states)
        
        # If targets are provided, compute loss
        if targets is not None:
            # Cross-entropy loss for next token prediction
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits.reshape(-1, self.config.vocab_size),
                targets.reshape(-1)
            ).mean()
            return logits, loss
            
        return logits

class CustomTrainState(train_state.TrainState):
    """Custom train state with additional fields if needed."""
    pass

# ---------- Data Loading ----------
def get_batch(rng_key, data, batch_size, context_length):
    """Get a random batch of data for causal language modeling."""
    data_size = data.shape[0]
    
    # Safety check: ensure data is large enough
    if data_size <= context_length:
        raise ValueError(f"Data size ({data_size}) must be larger than context length ({context_length})")
        
    rng_key, split_key = jax.random.split(rng_key)
    
    # Generate random starting indices
    max_start_idx = data_size - context_length - 1
    indices = jax.random.randint(
        split_key,
        shape=(batch_size,),
        minval=0,
        maxval=max_start_idx
    )
    
    # Create input and target sequences
    idx = jnp.arange(context_length)
    offsets = indices[:, None]
    x_indices = offsets + idx
    y_indices = offsets + idx + 1  # Targets are shifted by 1
    
    x = jnp.take(data, x_indices, axis=0)
    y = jnp.take(data, y_indices, axis=0)
    
    return x, y, rng_key

def create_train_state(rng_key, config):
    """Initialize model and create training state."""
    model = Llama3ForCausalLM(config)
    
    # Initialize model parameters
    dummy_input = jnp.ones((8, 64), dtype=jnp.int32)
    params = model.init(rng_key, dummy_input, training=False)["params"]
    
    # Create optimizer with learning rate schedule
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.learning_rate,
        warmup_steps=config.warmup_steps,
        decay_steps=config.total_steps,
        end_value=config.learning_rate * 0.1,
    )
    
    # AdamW optimizer with weight decay and gradient clipping
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.grad_clip),
        optax.adamw(
            learning_rate=lr_schedule,
            b1=config.beta1,
            b2=config.beta2,
            weight_decay=config.weight_decay
        )
    )
    
    return CustomTrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer
    )

# Single device training step
@partial(jax.jit, static_argnums=(3,))
def train_step(state, batch, rng_key, training=True):
    """Single training step for model."""
    inputs, targets = batch
    
    # Use different dropout key for each step
    dropout_rng = jax.random.fold_in(rng_key, state.step)
    
    def loss_fn(params):
        logits, loss = state.apply_fn(
            {"params": params},
            inputs,
            targets=targets,
            training=training,
            rngs={"dropout": dropout_rng}
        )
        return loss, logits
    
    # Compute loss and gradients
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    
    # Update model parameters
    new_state = state.apply_gradients(grads=grads)
    
    metrics = {
        "loss": loss,
        "perplexity": jnp.exp(loss),
    }
    
    return new_state, metrics, logits

# TPU-optimized training step
@partial(jax.pmap, axis_name='batch', static_broadcasted_argnums=(3,))
def train_step_pmap(state, batch, rng_keys, training=True):
    """Training step with parallel processing for TPUs."""
    inputs, targets = batch
    
    # Use device-specific RNG key
    dropout_rng = jax.random.fold_in(rng_keys, jax.lax.axis_index('batch'))
    dropout_rng = jax.random.fold_in(dropout_rng, state.step)
    
    def loss_fn(params):
        logits, loss = state.apply_fn(
            {"params": params},
            inputs,
            targets=targets,
            training=training,
            rngs={"dropout": dropout_rng}
        )
        return loss, logits
    
    # Compute loss and gradients
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    
    # Average gradients across replicas
    grads = jax.lax.pmean(grads, axis_name='batch')
    loss = jax.lax.pmean(loss, axis_name='batch')
    
    # Update model parameters
    new_state = state.apply_gradients(grads=grads)
    
    metrics = {
        "loss": loss,
        "perplexity": jnp.exp(loss),
    }
    
    return new_state, metrics, logits

# Standard evaluation step
@partial(jax.jit, static_argnums=(2,))
def eval_step(state, batch, training=False):
    """Evaluation step for model."""
    inputs, targets = batch
    logits, loss = state.apply_fn(
        {"params": state.params},
        inputs,
        targets=targets,
        training=training,
    )
    metrics = {
        "loss": loss,
        "perplexity": jnp.exp(loss),
    }
    return metrics

# TPU-optimized evaluation step
@partial(jax.pmap, axis_name='batch', static_broadcasted_argnums=(2,))
def eval_step_pmap(state, batch, training=False):
    """Evaluation step with parallel processing for TPUs."""
    inputs, targets = batch
    logits, loss = state.apply_fn(
        {"params": state.params},
        inputs,
        targets=targets,
        training=training,
    )
    
    # Average loss across replicas
    loss = jax.lax.pmean(loss, axis_name='batch')
    
    metrics = {
        "loss": loss,
        "perplexity": jnp.exp(loss),
    }
    return metrics

@partial(jax.jit, static_argnums=(2, 3, 4))
def generate_step(params, idx, context_length, temperature=1.0, top_k=40, apply_fn=None, rng_key=None):
    """Single generation step with top-k sampling."""
    # Get the context (up to context_length)
    context_size = min(idx.shape[1], context_length)
    idx_cond = idx[:, -context_size:]
    
    # Get logits from the model
    logits = apply_fn({"params": params}, idx_cond, training=False)
    
    # Focus on last token logits
    logits = logits[:, -1, :] / temperature
    
    # Apply top-k sampling
    if top_k > 0:
        top_k = min(top_k, logits.shape[-1])
        topk_values, _ = jax.lax.top_k(logits, top_k)
        threshold = topk_values[:, -1]
        logits = jnp.where(logits < threshold[:, None], jnp.full_like(logits, -1e10), logits)
    
    # Sample from the distribution
    sample = jax.random.categorical(rng_key, logits, axis=-1)
    
    # Append new token to sequence
    return jnp.concatenate([idx, sample[:, None]], axis=1)

def generate(
    params,
    apply_fn,
    prompt_idx,
    rng_key,
    max_new_tokens=100,
    temperature=1.0,
    top_k=40,
    context_length=4096
):
    """Generate text with the model."""
    # Handle prompts longer than context length
    if prompt_idx.shape[1] > context_length:
        idx = prompt_idx[:, -context_length:]
        print(f"Warning: Prompt truncated to the last {context_length} tokens.")
    else:
        idx = prompt_idx
    
    # Generate tokens sequentially
    for _ in range(max_new_tokens):
        rng_key, next_key = jax.random.split(rng_key)
        idx = generate_step(
            params,
            idx,
            context_length,
            temperature=temperature,
            top_k=top_k,
            apply_fn=apply_fn,
            rng_key=next_key
        )
    
    return idx

# ---------- TPU Initialization and Multi-device Training ----------
def initialize_tpu():
    """Check if running on TPU and initialize."""
    if 'tpu' in jax.devices()[0].platform:
        print(f"Running on {jax.device_count()} TPU devices")
        return True
    else:
        print("Not running on TPU")
        return False

def replicate_state_on_devices(state):
    """Replicate state across all TPU devices."""
    return jax.device_put_replicated(state, jax.local_devices())

def prepare_batch_for_devices(batch, num_devices):
    """Prepare batch for multiple devices by reshaping."""
    x, y = batch
    
    # Calculate per-device batch size
    total_batch_size = x.shape[0]
    per_device_batch_size = total_batch_size // num_devices
    
    # Handle batch sizes not divisible by num_devices
    if total_batch_size % num_devices != 0:
        new_batch_size = per_device_batch_size * num_devices
        x = x[:new_batch_size]
        y = y[:new_batch_size]
    
    # Reshape for devices
    x = x.reshape((num_devices, per_device_batch_size) + x.shape[1:])
    y = y.reshape((num_devices, per_device_batch_size) + y.shape[1:])
    
    return x, y

def create_device_rng_keys(rng_key, num_devices):
    """Create separate RNG keys for each device."""
    return jax.random.split(rng_key, num_devices)

def step_device_rng_keys(rng_keys):
    """Update RNG keys for each device."""
    new_keys = jax.vmap(jax.random.split)(rng_keys)
    return new_keys[:, 0]

def save_checkpoint(state, step, checkpoint_dir="llama3_checkpoints"):
    """Save model checkpoint."""
    import pickle
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # If using TPU, get params from first device
    if isinstance(state.params, list):
        params = jax.tree_util.tree_map(lambda x: x[0], state.params)
    else:
        params = state.params
        
    # Save the parameters
    with open(os.path.join(checkpoint_dir, f"model_step_{step}.pkl"), "wb") as f:
        pickle.dump(params, f)
    
    print(f"Saved checkpoint at step {step}")

def load_checkpoint(checkpoint_path, state):
    """Load model checkpoint."""
    import pickle
    
    # Load parameters
    with open(checkpoint_path, "rb") as f:
        params = pickle.load(f)
    
    # Update state with loaded parameters
    return state.replace(params=params)

def evaluate_model(state, val_data, batch_size, context_length, rng_key, use_tpu, num_devices):
    """Evaluate model on validation data."""
    # Get validation batch
    x, y, rng_key = get_batch(rng_key, val_data, batch_size, context_length)
    
    if use_tpu:
        # Prepare batch for TPU
        x, y = prepare_batch_for_devices((x, y), num_devices)
        # Run evaluation step
        metrics = eval_step_pmap(state, (x, y), False)
        # Extract metrics from first device
        metrics = {k: v[0] for k, v in metrics.items()}
    else:
        # Run standard evaluation step
        metrics = eval_step(state, (x, y), False)
        
    return metrics, rng_key

def prepare_dataset(dataset_name, split="train", tokenizer=None):
    """
    Prepare a Hugging Face dataset for training.
    
    Args:
        dataset_name: str, name of the dataset on Hugging Face
        split: str, which split to use (default: "train")
        tokenizer: Optional pre-trained tokenizer
    
    Returns:
        train_data: encoded training data
        val_data: encoded validation data
        encode_fn: function to encode text
        decode_fn: function to decode indices
        vocab_size: size of vocabulary
    """
    # Load dataset
    dataset = load_dataset(dataset_name)
    
    # Get text from dataset
    if split in dataset:
        dataset_split = dataset[split]
    else:
        print(f"Split {split} not found, using 'train'")
        dataset_split = dataset["train"]
    
    if "text" in dataset_split.features:
        text_key = "text"
    else:
        # Try to find a text field
        text_fields = [k for k, v in dataset_split.features.items()
                      if v.dtype == 'string']
        if text_fields:
            text_key = text_fields[0]
        else:
            raise ValueError("Could not find text field in dataset")
    
    # Sample a subset of the dataset if it's too large
    if len(dataset_split) > 100000:
        dataset_split = dataset_split.select(range(100000))
        print(f"Dataset too large, sampling 100,000 examples")
    
    # Combine all texts
    texts = dataset_split[text_key]
    
    if tokenizer is None:
        # Simple character-level tokenization
        print("Using character-level tokenization")
        all_text = "\n".join(texts)
        chars = sorted(list(set(all_text)))
        vocab_size = len(chars)
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}
        
        # Create encode/decode functions
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l if i in itos])
        
        # Encode full text
        data = jnp.array(encode(all_text))
    else:
        # Use provided tokenizer
        print("Using provided tokenizer")
        tokenized = [tokenizer.encode(text) for text in texts]
        # Flatten all tokenized texts into a single sequence
        data = jnp.array([token for text in tokenized for token in text])
        vocab_size = tokenizer.vocab_size
        encode = tokenizer.encode
        decode = tokenizer.decode
    
    # Split into train/val
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    return train_data, val_data, encode, decode, vocab_size

def train_llama3(
    config,
    dataset_name,
    batch_size=32,
    max_steps=10000,
    eval_freq=500,
    checkpoint_freq=1000,
    checkpoint_dir="llama3_checkpoints",
    resume_from=None,
    tokenizer=None,
):
    """
    Train a Llama3 model on a given dataset.
    
    Args:
        config: Model configuration
        dataset_name: Name of the dataset on Hugging Face
        batch_size: Batch size for training
        max_steps: Maximum number of training steps
        eval_freq: How often to evaluate on validation set
        checkpoint_freq: How often to save checkpoints
        checkpoint_dir: Directory to save checkpoints
        resume_from: Optional checkpoint to resume from
        tokenizer: Optional pre-trained tokenizer
    """
    use_tpu = initialize_tpu()
    num_devices = jax.device_count()
    
    # Adjust batch size based on number of devices
    if use_tpu:
        global_batch_size = batch_size * num_devices
        print(f"Using TPU with {num_devices} devices, global batch size: {global_batch_size}")
    else:
        global_batch_size = batch_size
        print(f"Using {jax.local_device_count()} devices, batch size: {global_batch_size}")
    
    # Load dataset
    print(f"Loading dataset: {dataset_name}")
    train_data, val_data, encode, decode, vocab_size = prepare_dataset(dataset_name, tokenizer=tokenizer)
    
    # Update config with correct vocab size
    if config.vocab_size != vocab_size:
        print(f"Updating config vocab size from {config.vocab_size} to {vocab_size}")
        config.vocab_size = vocab_size
    
    # Initialize model
    print("Initializing model")
    rng_key = jax.random.PRNGKey(42)
    rng_key, init_key = jax.random.split(rng_key)
    state = create_train_state(init_key, config)
    
    # Resume from checkpoint if specified
    if resume_from:
        print(f"Resuming from checkpoint: {resume_from}")
        state = load_checkpoint(resume_from, state)
    
    # For TPU, replicate the state across devices
    if use_tpu:
        state = replicate_state_on_devices(state)
        # Create separate rng keys for each device
        device_rng_keys = create_device_rng_keys(rng_key, num_devices)
    
    # Training loop
    print(f"Beginning training for {max_steps} steps")
    for step in tqdm(range(max_steps)):
        try:
            # Get batch
            if use_tpu:
                # For TPU, create a batch for each device
                x, y, rng_key = get_batch(rng_key, train_data, global_batch_size, config.context_length)
                x, y = prepare_batch_for_devices((x, y), num_devices)
                
                # Update device RNG keys
                device_rng_keys = step_device_rng_keys(device_rng_keys)
                
                # Train step with parallel map
                state, metrics, _ = train_step_pmap(state, (x, y), device_rng_keys, True)
                
                # Extract metrics from first device
                metrics = {k: v[0] for k, v in metrics.items()}
            else:
                # Standard single-device training
                x, y, rng_key = get_batch(rng_key, train_data, global_batch_size, config.context_length)
                state, metrics, _ = train_step(state, (x, y), rng_key)
            
            # Print metrics occasionally
            if step % 100 == 0:
                print(f"Step {step}: loss = {metrics['loss']:.4f}, perplexity = {metrics['perplexity']:.4f}")
            
            # Evaluate on validation set
            if step > 0 and step % eval_freq == 0:
                print("Evaluating on validation set")
                val_metrics, rng_key = evaluate_model(
                    state, val_data, global_batch_size, config.context_length, 
                    rng_key, use_tpu, num_devices
                )
                print(f"Validation at step {step}: loss = {val_metrics['loss']:.4f}, "
                      f"perplexity = {val_metrics['perplexity']:.4f}")
            
            # Save checkpoint
            if step > 0 and step % checkpoint_freq == 0:
                save_checkpoint(state, step, checkpoint_dir)
        
        except Exception as e:
            print(f"Error during training step {step}: {e}")
            import traceback
            traceback.print_exc()
            break
    
    # Final evaluation
    try:
        print("Final evaluation")
        val_metrics, _ = evaluate_model(
            state, val_data, global_batch_size, config.context_length, 
            rng_key, use_tpu, num_devices
        )
        print(f"Final validation: loss = {val_metrics['loss']:.4f}, "
              f"perplexity = {val_metrics['perplexity']:.4f}")
    except Exception as e:
        print(f"Error during final evaluation: {e}")
    
    # Save final checkpoint
    save_checkpoint(state, max_steps, os.path.join(checkpoint_dir, "final"))
    
    return state, encode, decode

def sample_from_model(state, prompt, encode, decode, max_tokens=100, temperature=0.8, top_k=40, context_length=4096):
    """
    Generate text from the model given a prompt.
    
    Args:
        state: Model state
        prompt: Text prompt to continue from
        encode: Function to encode text to tokens
        decode: Function to decode tokens to text
        max_tokens: Maximum number of tokens to generate
        temperature: Temperature for sampling
        top_k: Number of top tokens to consider for sampling
        context_length: Maximum context length
    
    Returns:
        Generated text
    """
    print(f"Generating {max_tokens} tokens from prompt: '{prompt}'")
    
    # Encode the prompt
    prompt_tokens = encode(prompt)
    prompt_idx = jnp.array([prompt_tokens])
    
    # If using TPU, get params from first device
    if isinstance(state.params, list):
        params = jax.tree_util.tree_map(lambda x: x[0], state.params)
        apply_fn = state.apply_fn
    else:
        params = state.params
        apply_fn = state.apply_fn
    
    # Generate text
    rng_key = jax.random.PRNGKey(42)
    generated_idx = generate(
        params,
        apply_fn,
        prompt_idx,
        rng_key,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        context_length=context_length
    )
    
    # Decode the generated text
    generated_text = decode(generated_idx[0].tolist())
    
    return generated_text

def main():
    """Main function to demonstrate training and inference."""
    # Example configuration for a small model
    config = Llama3Config(
        vocab_size=50304,  # Will be updated based on dataset
        context_length=1024,
        hidden_size=1024,
        intermediate_size=2816,
        num_hidden_layers=8,
        num_attention_heads=8,
        num_key_value_heads=4,
        learning_rate=2e-4,
        warmup_steps=1000,
        total_steps=10000,
    )
    
    # Train the model
    dataset_name = "wikitext"  # Smaller dataset for demonstration
    state, encode, decode = train_llama3(
        config,
        dataset_name,
        batch_size=16,
        max_steps=5000,
        eval_freq=500,
        checkpoint_freq=1000,
    )
    
    # Generate text from the trained model
    prompt = "Once upon a time"
    generated_text = sample_from_model(state, prompt, encode, decode, max_tokens=200)
    
    print("Generated text:")
    print(generated_text)

if __name__ == "__main__":
    main()