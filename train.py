import os
import sys
import tqdm
import jax
import jax.numpy as jnp
import jax.nn as nn
import haiku as hk
import flax.linen as nn
import optax
import jax.lax as lax
import math
import numpy as np
from transformer import Transformer
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

dataset = "shakespeare"
model_dir = "transformer.py"
batch_size = 32
seq_len = 32
d_model = 32
d_k = 64
h = 4
n_layers = 3
n_epochs = 10
lr = 1e-4
warmup_steps = 4000
max_steps = 20000
log_interval = 200

optimizer = optax.adam(learning_rate=lr)

# Download the dataset
url = "https://www.gutenberg.org/cache/epub/100/pg100.txt"

if not os.path.isdir("data"):
    os.makedirs("data")

logging.info("Downloading dataset...")
os.system(f"curl -L {url} >> data/data.txt")

with open("data/data.txt", "r") as f:
    text = f.read()

chars = list(set(text))
i2c = {i: c for i, c in enumerate(chars)}
c2i = {c: i for i, c in enumerate(chars)}

vocab_size = len(chars)
logging.info(f"Vocab size: {vocab_size}")

# test and train split 80/20
def prepare_data(text, seq_len, train_ratio=0.8):
    logging.info("Preparing data...")
    data = [c2i[c] for c in text]
    n = len(data)
    n_train = int(n * train_ratio)
    n_val = n - n_train

    train_data = data[:n_train]
    val_data = data[n_train:]

    n_train_batches = n_train // seq_len
    n_val_batches = n_val // seq_len

    train_data = train_data[:n_train_batches * seq_len]
    val_data = val_data[:n_val_batches * seq_len]

    train_data = np.array(train_data).reshape(n_train_batches, seq_len)
    val_data = np.array(val_data).reshape(n_val_batches, seq_len)

    logging.info("Data preparation complete.")
    return train_data, val_data

train_data, val_data = prepare_data(text, seq_len)

def get_batch(mode):
    logging.debug(f"Fetching batch for {mode} mode.")
    data = train_data if mode == "train" else val_data
    n_batches = data.shape[0]
    idx = np.random.randint(0, n_batches)
    x = np.stack([data[idx]])
    y = np.roll(x, -1, axis=1)
    return x, y

model = Transformer(n_layers, vocab_size, d_model, d_k, h)

def loss_fn(params, x, y):
    x = jax.nn.one_hot(x, vocab_size)
    y = jax.nn.one_hot(y, vocab_size)
    y_pred = model.apply(params, x)
    loss = nn.log_softmax(y_pred) * y
    loss = -jnp.sum(loss, axis=-1)
    return loss.mean()

params = model.init(jax.random.PRNGKey(0), jnp.ones((1, seq_len, vocab_size)))
opt_state = optimizer.init(params)

def count_params(params):
    return sum(jnp.prod(jnp.array(p.shape)) for p in jax.tree_util.tree_leaves(params))

num_params = count_params(params)
logging.info(f"Number of model parameters: {num_params}")

def train_step(params, opt_state, x, y):
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

def eval_step(params, x, y):
    x = jax.nn.one_hot(x, vocab_size)
    y_pred = model.apply(params, x)
    loss = nn.log_softmax(y_pred) * y
    loss = -jnp.sum(loss, axis=-1)
    return loss.mean()

logging.info("Starting training process...")
for epoch in range(n_epochs):
    logging.info(f"Starting epoch {epoch+1}/{n_epochs}")
    for step in range(max_steps):
        x, y = get_batch("train")
        params, opt_state, loss = train_step(params, opt_state, x, y)

        if step % log_interval == 0:
            x, y = get_batch("val")
            val_loss = eval_step(params, x, y)
            logging.info(f"Epoch: {epoch}, Step: {step}, Loss: {loss}, Val Loss: {val_loss}")
    logging.info(f"Epoch {epoch+1} complete. Loss: {loss}, Val Loss: {val_loss}")
logging.info("Training process complete.")
