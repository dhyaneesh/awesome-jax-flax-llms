if __name__ == "__main__":
    # Replace Jupyter notebook cell magic with regular Python code
    import subprocess
    subprocess.check_call(["pip", "install", "tqdm", "datasets"])

    import jax
    import optax
    import flax.linen as nn
    import jax.numpy as jnp
    from typing import Any, Callable, Dict, Optional, Tuple
    from flax.training import train_state
    from functools import partial
    import os
    from datasets import load_dataset
    from tqdm import tqdm

    dataset_name = "karpathy/tiny_shakespeare"
    """
    Other examples may include :
    [Note that these may take longer]

    dataset_name = "mindchain/wikitext2"  # WikiText dataset
    dataset_name = "SamuelYang/bookcorpus"  # BookCorpus dataset
    dataset_name = "oscar-corpus/oscar"  # OSCAR dataset
    """

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

    class CustomTrainState(train_state.TrainState):
        """Custom train state with additional fields if needed."""
        # Add any additional fields here if necessary
        pass

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
            # Instead of register_buffer, we'll make it a class variable
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

            # Reshape to (B, T, nh, hs)
            k = k.reshape(B, T, self.n_head, self.head_dim)
            q = q.reshape(B, T, self.n_head, self.head_dim)
            v = v.reshape(B, T, self.n_head, self.head_dim)

            # Transpose to (B, nh, T, hs)
            k = jnp.transpose(k, (0, 2, 1, 3))
            q = jnp.transpose(q, (0, 2, 1, 3))
            v = jnp.transpose(v, (0, 2, 1, 3))

            # Efficient scaled dot-product attention
            scale = jnp.sqrt(self.head_dim)
            att = (q @ jnp.transpose(k, (0, 1, 3, 2))) / scale  # (B, nh, T, T)

            # Causal mask to ensure attention only to past tokens
            mask = self.bias[:T, :T]
            # Use jnp.where with a mask for better TPU compatibility
            att = jnp.where(mask == 0, jnp.full_like(att, -1e10), att)  # (B, nh, T, T)

            # Softmax attention
            att = jax.nn.softmax(att, axis=-1)
            att = self.attn_dropout(att, deterministic=not training)

            # Combine heads
            y = att @ v  # (B, nh, T, hs)
            y = jnp.transpose(y, (0, 2, 1, 3))  # (B, T, nh, hs)
            y = y.reshape(B, T, C)  # (B, T, C)

            # Output projection
            y = self.resid_dropout(self.c_proj(y), deterministic=not training)
            return y


    class MLP(nn.Module):
        """MLP with better initialization and gelu activation."""
        config: TransformerConfig

        def setup(self):
            config = self.config
            self.c_fc = nn.Dense(
                4 * config.n_embed,
                kernel_init=nn.initializers.normal(stddev=0.02),
                bias_init=nn.initializers.zeros
            )
            # Use approximate=True for faster GELU on TPUs
            self.gelu = lambda x: jax.nn.gelu(x, approximate=True)
            self.c_proj = nn.Dense(
                config.n_embed,
                kernel_init=nn.initializers.normal(stddev=0.02),
                bias_init=nn.initializers.zeros
            )
            self.dropout = nn.Dropout(config.dropout)

        def __call__(self, x, training=False):
            x = self.c_fc(x)
            x = self.gelu(x)
            x = self.c_proj(x)
            x = self.dropout(x, deterministic=not training)
            return x


    class Block(nn.Module):
        """Transformer block with pre-layer normalization."""
        config: TransformerConfig

        def setup(self):
            self.ln_1 = nn.LayerNorm(epsilon=1e-5)
            self.attn = CausalSelfAttention(self.config)
            self.ln_2 = nn.LayerNorm(epsilon=1e-5)
            self.mlp = MLP(self.config)

        def __call__(self, x, training=False):
            # Pre-layer normalization design
            x = x + self.attn(self.ln_1(x), training=training)
            x = x + self.mlp(self.ln_2(x), training=training)
            return x


    class GPT(nn.Module):
        """GPT Language Model with improved implementation."""
        config: TransformerConfig

        def setup(self):
            config = self.config

            # Token and position embeddings
            self.wte = nn.Embed(
                config.vocab_size,
                config.n_embed,
                embedding_init=nn.initializers.normal(stddev=0.02)
            )
            self.wpe = nn.Embed(
                config.block_size,
                config.n_embed,
                embedding_init=nn.initializers.normal(stddev=0.02)
            )

            # Transformer blocks
            self.blocks = [Block(config) for _ in range(config.n_layer)]

            # Final layer norm and head
            self.ln_f = nn.LayerNorm(epsilon=1e-5)
            self.lm_head = nn.Dense(
                config.vocab_size,
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
            new_params['lm_head']['kernel'] = new_params['wte']['embedding']
            return new_params

        def __call__(self, idx, targets=None, training=False, params=None):
            config = self.config
            b, t = idx.shape

            # Apply weight tying if enabled (only during inference)
            if params is not None and self.apply_weight_tying and not training:
                params = self._tie_weights(params)

            # Get token and position embeddings
            token_emb = self.wte(idx)  # (b, t, n_embed)
            pos = jnp.arange(0, t, dtype=jnp.int32)
            pos_emb = self.wpe(pos)  # (t, n_embed)

            # Sum embeddings and apply dropout
            x = token_emb + pos_emb

            # Apply transformer blocks
            for block in self.blocks:
                x = block(x, training=training)

            # Apply final layer norm
            x = self.ln_f(x)

            # Project to vocabulary
            logits = self.lm_head(x)  # (b, t, vocab_size)

            # If targets are provided, compute loss
            if targets is not None:
                # Cross-entropy loss for next token prediction
                loss = optax.softmax_cross_entropy_with_integer_labels(
                    logits.reshape(-1, config.vocab_size),
                    targets.reshape(-1)
                ).mean()
                return logits, loss

            return logits

    # ---------- Data Loading ----------
    def get_batch(rng_key, data, batch_size, block_size):
        """Get a random batch of data with safety checks."""
        data_size = data.shape[0]

        # Safety check: ensure data is large enough
        if data_size <= block_size:
            raise ValueError(f"Data size ({data_size}) must be larger than block size ({block_size})")

        rng_key, split_key = jax.random.split(rng_key)

        # Use jax.random.randint for generating random indices
        max_start_idx = data_size - block_size - 1
        indices = jax.random.randint(
            split_key,
            shape=(batch_size,),
            minval=0,
            maxval=max_start_idx
        )

        # TPU-optimized data loading using vectorized operations
        idx = jnp.arange(block_size)
        offsets = indices[:, None]
        x_indices = offsets + idx
        y_indices = offsets + idx + 1

        x = jnp.take(data, x_indices, axis=0)
        y = jnp.take(data, y_indices, axis=0)

        return x, y, rng_key


    def create_train_state(rng_key, config):
        """Create initial training state."""
        model = GPT(config)

        # Initialize model parameters with a properly shaped input
        dummy_input = jnp.ones((8, 64), dtype=jnp.int32)  # Use batch size divisible by 8 for TPU
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

    # TPU-optimized training step with pmap support
    @partial(jax.pmap, axis_name='batch', static_broadcasted_argnums=(3,))
    def train_step_pmap(state, batch, rng_keys, training=True):
        """Single training step with parallel processing support for TPUs."""
        inputs, targets = batch

        # Use the device-specific RNG key and fold in the step for different dropout patterns
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


    # Standard training step for single device
    @partial(jax.jit, static_argnums=(3,))
    def train_step(state, batch, rng_key, training=True):
        """Single training step for single device."""
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


    # TPU-optimized evaluation step with pmap support
    @partial(jax.pmap, axis_name='batch', static_broadcasted_argnums=(2,))
    def eval_step_pmap(state, batch, training=False):
        """Evaluation step with parallel processing support for TPUs."""
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


    # Standard evaluation step for single device
    @partial(jax.jit, static_argnums=(2,))
    def eval_step(state, batch, training=False):
        """Evaluation step for single device."""
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

    # Fixed the static_argnums to not include rng_key
    @partial(jax.jit, static_argnums=(2, 3, 4))
    def generate_step(params, idx, block_size, temperature=1.0, top_k=40, apply_fn=None, rng_key=None):
        """Single generation step using top-k sampling."""
        # Take the last block_size tokens as context (or fewer if not enough)
        context_size = min(idx.shape[1], block_size)
        idx_cond = idx[:, -context_size:]

        # Get logits from the model
        logits = apply_fn({"params": params}, idx_cond, training=False)

        # Focus only on the last time step
        logits = logits[:, -1, :] / temperature

        # Optional top-k sampling using JAX's efficient operators
        if top_k > 0:
            top_k = min(top_k, logits.shape[-1])
            topk_values, _ = jax.lax.top_k(logits, top_k)
            threshold = topk_values[:, -1]
            logits = jnp.where(logits < threshold[:, None], jnp.full_like(logits, -1e10), logits)

        # Sample from the distribution
        sample = jax.random.categorical(rng_key, logits, axis=-1)

        # Append to the sequence
        return jnp.concatenate([idx, sample[:, None]], axis=1)


    def generate(
        params,
        apply_fn,
        prompt_idx,
        rng_key,
        max_new_tokens=100,
        temperature=1.0,
        top_k=40,
        block_size=256
    ):
        """Generate text using the model with proper handling of long prompts."""
        # Handle prompt that may be longer than block_size
        if prompt_idx.shape[1] > block_size:
            # Only keep the last block_size tokens of the prompt
            idx = prompt_idx[:, -block_size:]
            print(f"Warning: Prompt was truncated to the last {block_size} tokens due to context length limit.")
        else:
            idx = prompt_idx

        # Generate tokens one by one
        for _ in range(max_new_tokens):
            rng_key, next_key = jax.random.split(rng_key)
            idx = generate_step(
                params,
                idx,
                block_size,
                temperature=temperature,
                top_k=top_k,
                apply_fn=apply_fn,
                rng_key=next_key
            )

        return idx


    # ---------- TPU Initialization and Multi-device Training ----------
    def initialize_tpu():
        """Initialize TPU system."""
        # Check if running on TPU
        if 'tpu' in jax.devices()[0].platform:
            print(f"Running on {jax.device_count()} TPU devices")
            return True
        else:
            print("Not running on TPU")
            return False


    def replicate_state_on_devices(state):
        """Replicate state across all TPU devices."""
        # Broadcast the state to all devices
        state = jax.device_put_replicated(state, jax.local_devices())
        return state

    # Fixed to handle batch sizes not divisible by device count
    def prepare_batch_for_devices(batch, num_devices):
        """Prepare batch for multiple devices by reshaping."""
        x, y = batch

        # Get total batch size and calculate per-device batch size
        total_batch_size = x.shape[0]
        per_device_batch_size = total_batch_size // num_devices
        
        # Need to handle cases where batch size is not divisible by num_devices
        if total_batch_size % num_devices != 0:
            # Either pad or truncate to make it divisible
            new_batch_size = per_device_batch_size * num_devices
            x = x[:new_batch_size]
            y = y[:new_batch_size]

        # Reshape to (num_devices, per_device_batch_size, ...)
        x = x.reshape((num_devices, per_device_batch_size) + x.shape[1:])
        y = y.reshape((num_devices, per_device_batch_size) + y.shape[1:])

        return x, y


    def create_device_rng_keys(rng_key, num_devices):
        """Create separate RNG keys for each device."""
        # Split the main key into num_devices keys
        return jax.random.split(rng_key, num_devices)


    def step_device_rng_keys(rng_keys):
        """Update RNG keys for each device independently."""
        # Split each device's key to get a new key for each device
        new_keys = jax.vmap(jax.random.split)(rng_keys)
        # Each split returns 2 keys, keep the first one for each device
        return new_keys[:, 0]

    # Function to save model checkpoint
    def save_checkpoint(state, step, checkpoint_dir="checkpoints"):
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

    def evaluate_model(state, val_data, batch_size, block_size, rng_key, use_tpu, num_devices):
        """Evaluate model on validation data."""
        # Get validation batch
        x, y, rng_key = get_batch(rng_key, val_data, batch_size, block_size)
        
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

    def prepare_dataset(dataset_name, split="train"):
        """
        Prepare a Hugging Face dataset for training.

        Args:
            dataset_name: str, name of the dataset on Hugging Face
            split: str, which split to use (default: "train")

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
        if "text" in dataset[split].features:
            text_key = "text"
        else:
            # Try to find a text field
            text_fields = [k for k, v in dataset[split].features.items()
                          if v.dtype == 'string']
            if text_fields:
                text_key = text_fields[0]
            else:
                raise ValueError("Could not find text field in dataset")

        # Combine all texts
        text = "\n".join(dataset[split][text_key])

        # Create vocabulary
        chars = sorted(list(set(text)))
        vocab_size = len(chars)
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}

        # Create encode/decode functions
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])

        # Encode full text
        data = jnp.array(encode(text))

        # Split into train/val
        n = int(0.9 * len(data))
        train_data = data[:n]
        val_data = data[n:]

        return train_data, val_data, encode, decode, vocab_size

    use_tpu = initialize_tpu()
    num_devices = jax.device_count()

    # Load dataset from Hugging Face
    train_data, val_data, encode, decode, vocab_size = prepare_dataset(dataset_name)
    
    # Calculate reasonable total_steps based on dataset size
    data_size = len(train_data)
    
    # Better batch size calculation
    if use_tpu:
        # Scale batch size based on both device count and memory
        devices_to_use = min(num_devices, 8)
        # More conservative batch size for TPU
        batch_size = min(64 * devices_to_use, 512)  # Cap at 512
    else:
        batch_size = min(64, data_size // 10)  # Cap batch size at 64

    # Estimate total steps based on dataset size and batch size
    # Aim for ~100 epochs for small datasets, fewer for larger ones
    epochs = max(10, min(100, 1000000 // data_size))
    total_steps = (data_size * epochs) // batch_size

    config = TransformerConfig(
        vocab_size=vocab_size,
        block_size=min(256, data_size // 2),  # Ensure block_size isn't too large
        n_embed=384,
        n_head=6,
        n_layer=6,
        dropout=0.2,
        learning_rate=3e-4,
        total_steps=total_steps,
    )

    rng_key = jax.random.PRNGKey(42)
    rng_key, init_key = jax.random.split(rng_key)
    train_state = create_train_state(init_key, config)

    # For TPU, replicate the state across devices
    if use_tpu:
        train_state = replicate_state_on_devices(train_state)
        # Create separate rng keys for each device
        device_rng_keys = create_device_rng_keys(rng_key, num_devices)

    print(f"Training for {total_steps} steps with batch size {batch_size}")
    steps_to_run = min(5000, total_steps)  # Cap at 5000 steps for example

    # Add evaluation frequency and checkpoint frequency
    eval_freq = 500
    checkpoint_freq = 1000
    
    # Use tqdm for progress bar
    for step in tqdm(range(steps_to_run)):
        try:
            # Get batch
            if use_tpu:
                # For TPU, create a batch for each device
                x, y, rng_key = get_batch(rng_key, train_data, batch_size, config.block_size)
                x, y = prepare_batch_for_devices((x, y), num_devices)

                # Update device RNG keys correctly
                device_rng_keys = step_device_rng_keys(device_rng_keys)

                # Train step with parallel map
                train_state, metrics, _ = train_step_pmap(train_state, (x, y), device_rng_keys, True)

                # Extract metrics from first device (all are same due to pmean)
                metrics = {k: v[0] for k, v in metrics.items()}
            else:
                # Standard single-device training
                x, y, rng_key = get_batch(rng_key, train_data, batch_size, config.block_size)
                train_state, metrics, _ = train_step(train_state, (x, y), rng_key)

            # Print metrics occasionally
            if step % 100 == 0:
                print(f"Step {step}: loss = {metrics['loss']:.4f}, perplexity = {metrics['perplexity']:.4f}")
                
            # Evaluate on validation set
            if step > 0 and step % eval_freq == 0:
                val_metrics, rng_key = evaluate_model(
                    train_state, val_data, batch_size, config.block_size, 
                    rng_key, use_tpu, num_devices
                )
                print(f"Validation at step {step}: loss = {val_metrics['loss']:.4f}, "
                      f"perplexity = {val_metrics['perplexity']:.4f}")
                
            # Save checkpoint
            if step > 0 and step % checkpoint_freq == 0:
                save_checkpoint(train_state, step)

        except Exception as e:
            print(f"Error during training step {step}: {e}")
            break

    # Final evaluation
    try:
        val_metrics, _ = evaluate_model(
            train_state, val_data, batch_size, config.block_size, 
            rng_key, use_tpu, num_devices
        )
        print(f"Final validation: loss = {val_metrics['loss']:.4f}, "
              f"perplexity = {val_metrics['perplexity']:.4f}")
    except Exception as e:
        print(f"Error during final evaluation: {e}")

    # Save final checkpoint
    save_checkpoint(train_state, steps_to_run, checkpoint_dir="checkpoints_final")

    try:
        prompt = "ROMEO:"
        print(f"Generating text from prompt: '{prompt}'")
        prompt_idx = jnp.array([encode(prompt)])

        # If using TPU, get params from first device
        if use_tpu:
            params = jax.tree_util.tree_map(lambda x: x[0], train_state.params)
            apply_fn = train_state.apply_fn
        else:
            params = train_state.params
            apply_fn = train_state.apply_fn

        rng_key, gen_key = jax.random.split(rng_key)
        generated_idx = generate(
            params,
            apply_fn,
            prompt_idx,
            gen_key,
            max_new_tokens=500,
            temperature=0.8,
            top_k=40,
            block_size=config.block_size
        )

        generated_text = decode(generated_idx[0].tolist())
        print(generated_text)

    except Exception as e:
        print(f"Error during text generation: {e}")
