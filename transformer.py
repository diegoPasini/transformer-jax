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
        attentions = jnp.empty_like((self.h, self.d_k))
        for i, l in enumerate(self.heads):
            attentions[i] = l(x)
        attentions = lax.concatenate(attentions, dimension = 1)
        x = self.linear(attentions)
        x = feedforward(self, x)
        return x

class PositionalEncoding():
    def __init__(self, d_model, ):
        self.d_model = d_model

    def encode(self, ):

class Transformer():
    def __init__(self, num_layers, d_model, d_k, h):
        self.d_model = d_model
        self.d_k = d_k
        self.h = h
        self.num_layers = num_layers
        self.heads = nn.ModuleList([MultiHeadAttention(d_model, d_k, h) for i in range(num_layers)])
        

    
    def __call__(self, ):
    
    