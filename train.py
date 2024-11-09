import os
import sys
import tqdm
import jax
import jax.numpy as jnp
import jax.nn as nn
import haiku as hk
import flax.linen as nn
import jax.lax as lax
import math
import numpy as np
from transformer import Transformer


dataset = "shakespeare"
model_dir = "transformer.py"
batch_size = 32
seq_len = 128
d_model = 512
d_k = 64
h = 8
n_layers = 6
n_epochs = 10
lr = 1e-4
warmup_steps = 4000
max_steps = 20000
log_interval = 200

optimizer = flax.optim.Adam(learning_rate=lr)


# Download the dataset
url = "https://www.gutenberg.org/cache/epub/100/pg100.txt"

if not os.path.isdir("data"):
    os.makedirs("data")


os.system(f"curl -L {url} >> data/data.txt")

with open("data/data.txt", "r") as f:
    text = f.read()


chars = list(set(text))
i2c = {i: c for i, c in enumerate(chars)}
c2i = {c: i for i, c in enumerate(chars)}

vocab_size = len(chars)
print(vocab_size)


#test and train split 80/20
def prepare_data(text, seq_len, train_ratio=0.8):
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

    return train_data, val_data

train_data, val_data = prepare_data(text, seq_len)

def get_batch(mode):
    data = train_data if mode == "train" else val_data
    n_batches = data.shape[0]
    idx = np.random.randint(0, n_batches)
    x = np.stack([data[idx]])
    y = np.roll(x, -1, axis=1)
    return x, y

model = Transformer(n_layers, d_model, d_k, h)

def loss_fn(x, y):
    x = jax.nn.one_hot(x, vocab_size)
    y = jax.nn.one_hot(y, vocab_size)
    y_pred = model(x)
    loss = nn.log_softmax(y_pred) * y
    loss = -jnp.sum(loss, axis=-1)
    return loss

def train_step(optimizer, x, y):
    def loss_fn(params):
        x = jax.nn.one_hot(x, vocab_size)
        y = jax.nn.one_hot(y, vocab_size)
        y_pred = model(x)
        loss = nn.log_softmax(y_pred) * y
        loss = -jnp.sum(loss, axis=-1)
        return loss.mean()

    loss, grad = jax.value_and_grad(loss_fn)(optimizer.target)
    optimizer = optimizer.apply_gradient(grad)
    return optimizer, loss

def eval_step(params, x, y):
    x = jax.nn.one_hot(x, vocab_size)
    y = jax.nn.one_hot(y, vocab_size)
    y_pred = model(x)
    loss = nn.log_softmax(y_pred) * y
    loss = -jnp.sum(loss, axis=-1)
    return loss.mean()


for epoch in range(n_epochs):
    for step in range(max_steps):
        x, y = get_batch("train")
        optimizer, loss = train_step(optimizer, x, y)

        if step % log_interval == 0:
            x, y = get_batch("val")
            val_loss = eval_step(optimizer.target, x, y)
            print(f"Epoch: {epoch}, Step: {step}, Loss: {loss}, Val Loss: {val_loss}")
    print(f"Epoch: {epoch}, Loss: {loss}, Val Loss: {val_loss}")

    









