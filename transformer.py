import jax
import jax.numpy as jnp
# import jax.nn as nn
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
    attention_matrix = q @ jnp.permute_dims(k, axes=(0, 1, 3, 2))
    d_k = k.shape[-1]
    attention_matrix = attention_matrix / jnp.sqrt(d_k)
    attention_matrix = jnp.triu(attention_matrix, k=0)
    seq_len = q.shape[2]
    mask = jnp.tril(jnp.ones((seq_len, seq_len)))
    mask = mask[None, None, :, :]
    attention_matrix = jnp.where(
        mask == 0,
        jnp.full_like(attention_matrix, -1e9),
        attention_matrix
    )
    attention_matrix = jax.nn.softmax(attention_matrix, axis=-1)
    output = attention_matrix @ v
    return output

class MultiHeadAttentionModule(nn.Module):
    d_model: int
    d_k: int
    num_heads: int

    def setup(self):
        weight_dim = self.d_model * self.d_k * 3
        self.wqkv = nn.Dense(features=weight_dim, kernel_init=nn.initializers.xavier_uniform(), use_bias=False)
    
    def __call__(self, x):
        batch_size, seq_length, _ = x.shape
        qkv = self.wqkv(x)
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, -1)
        qkv = qkv.transpose(0, 2, 1, 3) 
        q, k, v = jnp.array_split(qkv, 3, axis=-1)
        x = scaled_dot_product_attention(q, k, v)
        #print(f"x1.shape: {x.shape}")
        x = x.transpose(0, 2, 1, 3) 
        # print(f"x2.shape: {x.shape}")
        x = x.reshape(batch_size, seq_length, self.d_model * self.d_k)
        # print(f"x3.shape: {x.shape}")
        # print(f"x.shape: {x.shape}")    
        return x

class TransformerLayer(nn.Module):
    d_model: int
    d_k: int
    head: int
    dropout_rate: float = 0.1

    def setup(self):
        self.linear = nn.Dense(features=self.d_model * self.d_k, kernel_init=nn.initializers.xavier_uniform())
        self.multi_head_attention = MultiHeadAttentionModule(self.d_model, self.d_k, self.head)
        self.dropout = nn.Dropout(rate=self.dropout_rate)
        self.layernorm1 = nn.LayerNorm()
        self.layernorm2 = nn.LayerNorm()
        self.ff1 = nn.Dense(features=self.d_model * self.d_k, kernel_init=nn.initializers.xavier_uniform())
        self.ff2 = nn.Dense(features=self.d_model * self.d_k, kernel_init=nn.initializers.xavier_uniform())

    @hk.transparent
    def feedforward(self, x):
        x = self.ff1(x)
        x = nn.relu(x)
        x = self.ff2(x)
        return x
    
    def __call__(self, x, training: bool = True):
        residual = x
        attention = self.multi_head_attention(x)
        x = self.linear(attention)
        x = x + self.dropout(x, deterministic=not training)
        x = x + residual
        x = self.layernorm1(x)
        residual = x
        x = self.feedforward(x)
        x = x + self.dropout(x, deterministic=not training)
        x = x + residual
        x = self.layernorm2(x)
        return x

class PositionalEncoding(nn.Module):
    d_model: int         
    max_len: int = 5000  

    def setup(self):
        pe = np.zeros((self.max_len, self.d_model))
        position = np.arange(0, self.max_len, dtype=np.float32)[:, None]
        div_term = np.exp(np.arange(0, self.d_model, 2) * (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        pe = pe[None]
        self.pe = self.sow('constants', 'pe', jax.device_put(pe))

    def __call__(self, x):
        pe = self.get_variable('constants', 'pe')
        x = x + self.pe[:, :x.shape[1]]
        return x

class Transformer(nn.Module):
    num_layers: int
    vocab_size: int
    d_model: int
    d_k: int
    h: int
    dropout_rate: float = 0.1

    def setup(self):
        self.embedding = nn.Embed(num_embeddings = self.vocab_size, features=self.d_model * self.d_k)
        self.attention_layers = [TransformerLayer(self.d_model, self.d_k, self.h, self.dropout_rate) for _ in range(self.num_layers)]
        self.positional_encoder = PositionalEncoding(self.d_model * self.d_k)
        self.lin = nn.Dense(features=self.vocab_size)
        self.dropout = nn.Dropout(rate=self.dropout_rate)
        self.layernorm = nn.LayerNorm()

    def __call__(self, output_embeddings, training: bool = True):
        # Ensure the input is of integer type
        output_embeddings = jnp.asarray(output_embeddings, dtype=jnp.int32)
        x = self.embedding(output_embeddings)
        x = self.positional_encoder(x)
        for attention_layer in self.attention_layers:
            x = attention_layer(x, training=training)
        x = self.lin(x)
        x = x + self.dropout(x, deterministic=not training)
        x = self.layernorm(x)
        return x

# def main():
#     dummy_input = jnp.ones((1, 10, 512))
#     transformer_model = Transformer(num_layers=6, d_model=512, d_k=64, h=8)
#     rng = jax.random.PRNGKey(0)
#     params = transformer_model.init(rng, dummy_input)
#     output = transformer_model.apply(params, dummy_input)
#     print("Transformer output shape:", output.shape)

# if __name__ == "__main__":
#     main()
