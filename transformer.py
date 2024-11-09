import jax
import jax.numpy as jnp
import jax.nn as nn
import haiku as hk
import flax.linen as nn
import jax.lax as lax
import math

## d_model
## h
## d_v
## d_k

# Jax Sharding
# Look into Jax jit
def scaled_dot_product_attention(q, k, v):
    # Replace 0 with -np.inf

    attention_matrix = jnp.dot(q, k.T)
    d_k = k.shape[0]
    attention_matrix = jax.nn.softmax(attention_matrix / jnp.sqrt(d_k), axis = 1)
    attention_matrix = jnp.triu(attention_matrix, k = -math.inf)
    return attention_matrix @ v


class ScaledDotProductModule(nn.Module):
    # weight dim : d_model x d_k
    def __init__(self, d_model, d_k):
        super().__init__(self)
        weight_dim = d_model * d_k
        self.w_q = flax.linen.Dense(features = weight_dim, use_bias = False)
        self.w_k = flax.linen.Dense(features = weight_dim, use_bias = False)
        self.w_v = flax.linen.Dense(features = weight_dim, use_bias = False)
    
    def __call__(self, x):

        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)
        
        x = scaled_dot_product_attention(q, k, v)


class MultiHeadAttention():
    def __init__(self, d_model, d_k, h):
        super().__init__(self)
        self.d_model = d_model
        self.d_k = d_k
        self.h = h
        self.linear = flax.linen.Dense(features = d_model * d_k * h)
        self.heads = nn.ModuleList([ScaledDotProductModule(d_model, d_k) for i in range(h)])
        self.ff1 = flax.linen.Dense(features = d_model)
        self.ff2 = flax.linen.Dense(features = d_model)


    @hk.transparent
    def feedforward(self, x):
        x = self.ff1(x)
        x = nn.relu(x)
        x = self.ff2(x)
        return x
    
    def __call__(self, x):
        residual = x
        attentions = jnp.empty_like((self.h, self.d_k))
        for i, l in enumerate(self.heads):
            attentions[i] = l(x)
        attentions = lax.concatenate(attentions, dimension = 1)
        x = self.linear(attentions)
        x = x + residual
        residual = x
        x = self.feedforward(x)
        x = x + residual

        return x

class PositionalEncoding():
    def __init__(self, d_model, dropout = 0.1, max_len = 5000):
        self.d_model = d_model
        self.dropout = dropout
        self.pe = jnp.zeros((max_len, d_model))
        position = jnp.arange(0, max_len).reshape(-1, 1)
        div_term = jnp.exp(jnp.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        self.pe[:, 0::2] = jnp.sin(position * div_term)
        self.pe[:, 1::2] = jnp.cos(position * div_term)
        self.pe = self.pe[None, :, :]
        self.register_buffer('pe', self.pe)


    def encode(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class Transformer():
    def __init__(self, num_layers, d_model, d_k, h):
        self.d_model = d_model
        self.d_k = d_k
        self.h = h
        self.num_layers = num_layers
        self.heads = nn.ModuleList([MultiHeadAttention(d_model, d_k, h) for i in range(num_layers)])
        # self.positional_encoder = PositionalEncoding()
        self.lin = flax.linen.Dense(features = d_model)

    def __call__(self, output_embeddings):
        x = output_embeddings
        for i, l in enumerate(self.heads):
            x = l(x)
        x = self.lin(x)
        x = nn.softmax(x)
        return x
        
    