import os
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
from model import LLaMA3, LLaMAConfig

# Check for TPU and set environment
os.environ['JAX_PLATFORM_NAME'] = 'tpu'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
print("JAX devices:", jax.devices())

def create_train_state(model, config, rng_key):
    """Create and initialize the model's training state."""
    # Initialize model parameters
    dummy_input = jnp.ones((config.batch_size, config.max_seq_len), dtype=jnp.int32)
    variables = model.init(rng_key, dummy_input)
    
    # Create learning rate schedule
    learning_rate_fn = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.learning_rate,
        warmup_steps=config.warmup_steps,
        decay_steps=config.max_steps,
        end_value=0.0
    )
    
    # Create optimizer
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(
            learning_rate=learning_rate_fn,
            b1=0.9,
            b2=0.999,
            eps=1e-8,
            weight_decay=config.weight_decay
        )
    )
    
    # Create training state
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=optimizer
    )
    
    return state

def prepare_datasets(config):
    """Prepare the training and validation datasets."""
    # Load dataset (example using wikitext)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    
    # Initialize tokenizer
    tokenizer = SentencePieceUnigramTokenizer()
    tokenizer.train_from_iterator(
        dataset["train"]["text"],
        vocab_size=config.vocab_size,
        show_progress=True
    )
    
    def tokenize_function(example):
        # Tokenize text
        tokens = tokenizer.encode(example["text"])
        
        # Create input and target sequences
        input_ids = tokens.ids[:-1]
        target_ids = tokens.ids[1:]
        
        # Pad or truncate sequences
        if len(input_ids) > config.max_seq_len:
            input_ids = input_ids[:config.max_seq_len]
            target_ids = target_ids[:config.max_seq_len]
        else:
            padding = [0] * (config.max_seq_len - len(input_ids))
            input_ids.extend(padding)
            target_ids.extend(padding)
        
        return {
            "input_ids": input_ids,
            "target_ids": target_ids
        }
    
    # Process datasets
    train_dataset = dataset["train"].map(
        tokenize_function,
        remove_columns=dataset["train"].column_names
    )
    eval_dataset = dataset["validation"].map(
        tokenize_function,
        remove_columns=dataset["validation"].column_names
    )
    
    return train_dataset, eval_dataset, tokenizer

def get_batch(key, data, config):
    """Get a batch of data."""
    # Get random indices
    indices = jax.random.randint(key, (config.batch_size,), 0, len(data))
    
    # Get batch
    batch = {
        "input_ids": jnp.stack([data[i]["input_ids"] for i in indices]),
        "target_ids": jnp.stack([data[i]["target_ids"] for i in indices])
    }
    
    return batch

def initialize_tpu():
    """Initialize TPU devices."""
    try:
        jax.devices()
    except RuntimeError:
        print("No TPU devices found. Using CPU.")
        os.environ['JAX_PLATFORM_NAME'] = 'cpu'

def create_device_rng_keys(rng, n_devices):
    """Create RNG keys for each device."""
    return jax.random.split(rng, n_devices)

def evaluate(model_apply_fn, params, eval_data, config, num_batches=10):
    """Evaluate the model on the validation dataset."""
    total_loss = 0
    num_tokens = 0
    
    for _ in range(num_batches):
        # Get batch
        batch = get_batch(jax.random.PRNGKey(0), eval_data, config)
        
        # Forward pass
        logits = model_apply_fn(params, batch["input_ids"], deterministic=True)
        
        # Compute loss
        labels = jax.nn.one_hot(batch["target_ids"], config.vocab_size)
        loss = jnp.sum(
            jax.nn.cross_entropy_with_logits(logits, labels) * 
            (batch["target_ids"] != 0)  # Mask out padding tokens
        )
        
        total_loss += loss
        num_tokens += jnp.sum(batch["target_ids"] != 0)
    
    return total_loss / num_tokens

def train_step(state, batch, dropout_rng):
    """Perform a single training step."""
    def loss_fn(params):
        # Forward pass with dropout
        logits = state.apply_fn(
            params,
            batch["input_ids"],
            deterministic=False,
            rngs={"dropout": dropout_rng}
        )
        
        # Compute loss
        labels = jax.nn.one_hot(batch["target_ids"], state.params["embed_tokens"]["embedding"].shape[0])
        loss = jnp.mean(
            jax.nn.cross_entropy_with_logits(logits, labels) * 
            (batch["target_ids"] != 0)  # Mask out padding tokens
        )
        
        return loss
    
    # Compute gradients
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    
    # Update state
    new_state = state.apply_gradients(grads=grads)
    
    return new_state, loss

def train_llama(config, num_epochs=5, steps_per_epoch=1000, save_every=1000):
    """Train the LLaMA model."""
    # Initialize TPU
    initialize_tpu()
    
    # Create model
    model = LLaMA3(config)
    
    # Create RNG key
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
    
    # Create training state
    state = create_train_state(model, config, init_rng)
    
    # Prepare datasets
    train_dataset, eval_dataset, tokenizer = prepare_datasets(config)
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Training
        total_loss = 0
        for step in tqdm(range(steps_per_epoch)):
            # Get batch
            rng, batch_rng = jax.random.split(rng)
            batch = get_batch(batch_rng, train_dataset, config)
            
            # Training step
            rng, dropout_rng = jax.random.split(rng)
            state, loss = train_step(state, batch, dropout_rng)
            
            total_loss += loss
            
            # Save checkpoint
            if (step + 1) % save_every == 0:
                checkpoint_dir = f"checkpoints/llama3_epoch{epoch+1}_step{step+1}"
                os.makedirs(checkpoint_dir, exist_ok=True)
                orbax_checkpointer = ocp.StandardCheckpointer()
                orbax_checkpointer.save(
                    checkpoint_dir,
                    {"model": state.params},
                    force=True
                )
        
        # Print average loss
        avg_loss = total_loss / steps_per_epoch
        print(f"Average training loss: {avg_loss:.4f}")
        
        # Evaluation
        eval_loss = evaluate(state.apply_fn, state.params, eval_dataset, config)
        print(f"Validation loss: {eval_loss:.4f}")

def load_checkpoint(checkpoint_dir, step=None):
    """Load a checkpoint from disk."""
    orbax_checkpointer = ocp.StandardCheckpointer()
    restored = orbax_checkpointer.restore(checkpoint_dir)
    return restored["model"]

def generate_text(model, params, tokenizer, prompt, max_new_tokens=100, temperature=0.8):
    """Generate text from a prompt."""
    # Tokenize prompt
    input_ids = tokenizer.encode(prompt).ids
    input_ids = jnp.array(input_ids)[None, :]
    
    # Generate
    rng = jax.random.PRNGKey(0)
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        rng_key=rng,
        temperature=temperature
    )
    
    # Decode
    output_text = tokenizer.decode(output_ids[0].tolist())
    return output_text

if __name__ == "__main__":
    # Create config
    config = LLaMAConfig()
    
    # Train model
    train_llama(config) 