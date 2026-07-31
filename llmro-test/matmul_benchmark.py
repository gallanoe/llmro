import time
import jax
import jax.numpy as jnp


def bench_matmul(n=8192, dtype=jnp.bfloat16, iters=50):
    key = jax.random.key(0)
    a = jax.random.normal(key, (n, n), dtype)
    b = jax.random.normal(key, (n, n), dtype)

    mm = jax.jit(lambda x, y: x @ y)
    mm(a, b).block_until_ready()

    t0 = time.perf_counter()
    for _ in range(iters):
        out = mm(a, b)
    out.block_until_ready()
    dt = time.perf_counter() - t0

    return (2 * n**3 * iters) / dt / 1e12


def check_cudnn_attention(b=4, s=1025, h=12, d=64):
    key = jax.random.key(0)
    q, k, v = [
        jax.random.normal(key, (b, s, h, d), jnp.float32).astype(jnp.bfloat16)
        for _ in range(3)
    ]

    fn = jax.jit(
        lambda q, k, v: jax.nn.dot_product_attention(
            q, k, v, is_causal=True, implementation="cudnn"
        )
    )
    fn(q, k, v).block_until_ready()

    hlo = fn.lower(q, k, v).compile().as_text()
    return "cudnn" in hlo.lower()


if __name__ == "__main__":
    print(jax.devices())
    for dt in (jnp.bfloat16, jnp.float16, jnp.float32):
        print(f"{dt.__name__:>10}: {bench_matmul(dtype=dt):7.1f} TFLOPS")
    print("cuDNN attention:", check_cudnn_attention())
    print(jax.local_devices()[0].memory_stats())
