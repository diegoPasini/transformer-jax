import jax
import jax.numpy as jnp
import jax.nn as nn
import haiku as hk
import flax.linen as nn
import jax.lax as lax
import math
import numpy as np

## d_model
## h
## d_v
## d_k

# Jax Sharding
# Look into Jax jit
def scaled_dot_product_attention(q, k, v):
    # Replace 0 with -np.inf
    print(q.shape)
    print(k.shape)
    print(jnp.permute_dims(k, axes = (0, 2, 1)).shape)
    attention_matrix = q @ jnp.permute_dims(k, axes = (0, 2, 1))
    print(attention_matrix.shape)
    d_k = k.shape[0]
    attention_matrix = attention_matrix / jnp.sqrt(d_k)
    attention_matrix = jnp.triu(attention_matrix, k = 0)
    attention_matrix = jnp.where(attention_matrix == 0, -9e15, attention_matrix)
    attention_matrix = jax.nn.softmax(attention_matrix, axis = -1)
    print(f"attention_matrix.shape: {attention_matrix.shape}") 
    return attention_matrix @ v

class ScaledDotProductModule(nn.Module):
    d_model: int
    d_k: int

    def setup(self):
        weight_dim = self.d_model * self.d_k * 3
        self.wqkv = nn.Dense(features=weight_dim, use_bias=False)
    
    def __call__(self, x):
        print(x.shape)
        print(f"x {x}")
        qkv = self.wqkv(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        
        x = scaled_dot_product_attention(q, k, v)
        return x

class MultiHeadAttention(nn.Module):
    d_model: int
    d_k: int
    h: int

    def setup(self):
        self.linear = nn.Dense(features=self.d_model * self.d_k * self.h)
        self.heads = [ScaledDotProductModule(self.d_model, self.d_k) for _ in range(self.h)]
        self.ff1 = nn.Dense(features=self.d_model)
        self.ff2 = nn.Dense(features=self.d_model)

    @hk.transparent
    def feedforward(self, x):
        x = self.ff1(x)
        x = nn.relu(x)
        x = self.ff2(x)
        return x
    
    def __call__(self, x):
        residual = x
        attentions = [head(x) for head in self.heads]
        attentions = jnp.concatenate(attentions, axis=-1)
        x = self.linear(attentions)
        x = x + residual
        residual = x
        x = self.feedforward(x)
        x = x + residual

        return x

class PositionalEncoding(nn.Module):
    d_model : int         
    max_len : int = 5000  

    def setup(self):
        pe = np.zeros((self.max_len, self.d_model))
        position = np.arange(0, self.max_len, dtype=np.float32)[:,None]
        div_term = np.exp(np.arange(0, self.d_model, 2) * (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        pe = pe[None]
        self.pe = jax.device_put(pe)

    def __call__(self, x):
        x = x + self.pe[:, :x.shape[1]]
        return x

class Transformer(nn.Module):
    num_layers: int
    d_model: int
    d_k: int
    h: int

    def setup(self):
        self.attention_layers = [MultiHeadAttention(self.d_model, self.d_k, self.h) for _ in range(self.num_layers)]
        self.positional_encoder = PositionalEncoding(self.d_model)
        self.lin = nn.Dense(features=self.d_model)

    def __call__(self, output_embeddings):
        x = output_embeddings
        x = self.positional_encoder(x)
        for attention_layer in self.attention_layers:
            x = attention_layer(x)
        x = self.lin(x)
        x = nn.softmax(x)
        return x

def main():
    dummy_input = jnp.ones((1, 10, 512))
    transformer_model = Transformer(num_layers=6, d_model=512, d_k=64, h=8)
    rng = jax.random.PRNGKey(0)
    params = transformer_model.init(rng, dummy_input)
    output = transformer_model.apply(params, dummy_input)
    print("Transformer output shape:", output.shape)

if __name__ == "__main__":
    main()
