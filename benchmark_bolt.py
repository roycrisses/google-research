
import time
import jax
import jax.numpy as jnp
from performer.fast_attention.jax import fast_attention

def benchmark():
    batch_size = 32
    seq_len = 1024
    num_heads = 8
    d_model = 64
    nb_features = 256

    key = jax.random.PRNGKey(0)
    q = jax.random.normal(key, (batch_size, seq_len, num_heads, d_model))
    k = jax.random.normal(key, (batch_size, seq_len, num_heads, d_model))
    v = jax.random.normal(key, (batch_size, seq_len, num_heads, d_model))

    attn_fn = fast_attention.make_fast_softmax_attention(
        qkv_dim=d_model,
        nb_features=nb_features,
        unidirectional=False
    )

    # Warmup
    print("Warming up...")
    out = attn_fn(q, k, v)
    out.block_until_ready()

    print("Benchmarking...")
    start = time.time()
    for _ in range(10):
        out = attn_fn(q, k, v)
        out.block_until_ready()
    end = time.time()
    print(f"Time taken (Bidirectional): {(end - start) / 10:.4f}s")

    attn_fn_uni = fast_attention.make_fast_softmax_attention(
        qkv_dim=d_model,
        nb_features=nb_features,
        unidirectional=True
    )

    # Warmup
    print("Warming up (Unidirectional)...")
    out = attn_fn_uni(q, k, v)
    out.block_until_ready()

    print("Benchmarking (Unidirectional)...")
    start = time.time()
    for _ in range(10):
        out = attn_fn_uni(q, k, v)
        out.block_until_ready()
    end = time.time()
    print(f"Time taken (Unidirectional): {(end - start) / 10:.4f}s")

if __name__ == "__main__":
    benchmark()
