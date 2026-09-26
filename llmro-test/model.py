import os

from jax.nn import initializers

# MUST be set before jax initializes its allocator. Without this, XLA's default
# BFC allocator cannot obtain more than ~4.6 GB of the 32 GB card under WSL2 --
# it is not the driver (raw cuMemAlloc reaches 27 GB) and not the model.
# See DESCRIPTION.md §5.
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "cuda_async")

from dataclasses import dataclass
import numpy as np
from jax import Array
import jax
import optax
import jax.numpy as jnp
from flax import nnx
import flax.linen as nn

from typing import Optional, Tuple
from jaxtyping import DTypeLike, Float, Int


@dataclass(frozen=True)
class ModelConfig:
    seed: int
    vocab_size: int
    d_model: int
    n_heads: int
    d_hidden: int
    n_layers: int
    dtype: DTypeLike  # compute
    param_dtype: DTypeLike  # storage

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads


# RoPE
def rope_tables(
    cfg: ModelConfig,
    seq_len: int,
    base: float = 10_000.0,
) -> Tuple[Float[Array, "seq d_head"], Float[Array, "seq d_head"]]:
    """Generate RoPE table.

    Step 1: Calculate a ladder of decreasing angular velocities.
        To do this, we take the inverted power of some base value.
        The inverted geometric pattern is just convention - technically
        any ladder of frequencies is sufficient.
        The base value sets the spread of the frequency ladder. Higher/lower
        spread dedicates more/less to long-range distances.
    Step 2: Calculate angular displacement values. m[i, d] = ith token, head dimension d
        Hence, angular displacement is per-step angular displacement * timestamp i.e.
        m[i, d] = i * theta_d
    Step 3: Duplicate across the final axis for vectorized application of
        angular displacement

    Note: we use float32 to construct our tables to avoid numerical inaccuracies
    before the values are essentially "normalized" by sin/coa i.e. [-1,1] does not
    need higher fidelity representation.
    """
    # Calculate ladder of angular velocities

    pows = jnp.arange(0, cfg.d_head, 2, dtype=cfg.param_dtype) / cfg.d_head
    freqs = 1 / (base**pows)
    # Calculate table of angular displacements
    seq = jnp.arange(seq_len, dtype=cfg.param_dtype)
    angles = jnp.outer(seq, freqs)
    emb = jnp.concatenate((angles, angles), axis=-1)
    return jnp.cos(emb).astype(cfg.dtype), jnp.sin(emb).astype(cfg.dtype)


def apply_rope(
    x: Float[Array, "batch seq heads d_head"],
    cos: Float[Array, "seq d_head"],
    sin: Float[Array, "seq d_head"],
) -> Float[Array, "batch seq heads d_head"]:
    """Apply rotary embeddings

    Vectorized application of rotary embeddings uniformly across
    all tokens in each head. Assuming a fixed token and head:
    Step 1: Split latent dimension in half, treat as D / 2 vectors in R^2 by defining
    points x[i] and x[i + D / 2] := (a, b)
    Step 2: Calculate corresponding orthogonal vector (-b, a)
    Step 3: Apply rotations using v * cos(theta) + v_ort * sin(theta)

    A more comprehensive formulation: take a vector x in R^D.
    We then take D / 2 R^2 planar subspaces by taking pairs of canonical
    basis vectors. We then apply rotations along those planar subspaces
    on the projection of the vector x on those planar subspaces.

    Essentially - we are doing norm-preserving rotation to encode relative
    position information to overcome permutation invariance of attention.

    NOTE: this construction is largely arbitrary - we can take any pairs of
    canonical basis vectors. We can even try to encode relative positional signals
    using norm-preserving transformations in higher-dimensional subspaces - but those
    just decompose into independent planar rotations anyways. We just need at least
    two dimensions for nontrivial continuous rotations. "Why rotations"
    can't be easily explained in a docstring.
    """
    # Construct orthogonal vectors
    e1, e2 = jnp.split(x, 2, axis=-1)
    x_ort = jnp.concatenate((-e2, e1), axis=-1)
    # Broadcast tables across batch and heads and apply rotation
    cos, sin = cos[None, :, None, :], sin[None, :, None, :]
    return x * cos + x_ort * sin


# Custom Linear because they didn't port fp8 from Linen to NNX


# Need custom variable type to not be counted as parameter
class Fp8Meta(nnx.Variable):
    """Scaling state: checkpointed with the model, ignored by nnx.Optimizer(wrt=nnx.Param)."""


# TODO Later
class Fp8Linear(nnx.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        rngs: nnx.Rngs,
        param_dtype=jnp.float32,
        amax_history_len: int = 16,
    ):
        # Master weight: plain f32 Param. FP8 copies are made per call, never stored.
        self.kernel = nnx.Param(
            nnx.initializers.normal(0.02)(rngs.params(), (d_in, d_out), param_dtype)
        )

        # One scale + amax history per quantized tensor.
        self.x_scale = Fp8Meta(jnp.ones((), jnp.float32))  # input activations (E4M3)
        self.x_amax = Fp8Meta(jnp.zeros((amax_history_len,), jnp.float32))
        self.w_scale = Fp8Meta(jnp.ones((), jnp.float32))  # kernel (E4M3)
        self.w_amax = Fp8Meta(jnp.zeros((amax_history_len,), jnp.float32))
        self.g_scale = Fp8Meta(jnp.ones((), jnp.float32))  # output gradient (E5M2)
        self.g_amax = Fp8Meta(jnp.zeros((amax_history_len,), jnp.float32))

    def __call__(
        self, x: Float[Array, "batch seq din"]
    ) -> Float[Array, "batch seq dout"]:
        pass
        # Quantize to fp8
        # Rescale up


# SwiGLU
class SwiGLU(nnx.Module):
    def __init__(self, cfg: ModelConfig, rngs: nnx.Rngs):

        # bfloat16
        # self.u = nnx.Linear(
        #     cfg.d_model,
        #     cfg.d_hidden,
        #     use_bias=False,
        #     kernel_init=nnx.initializers.normal(0.02),
        #     param_dtype=cfg.param_dtype,
        #     dtype=cfg.dtype,
        #     rngs=rngs,
        # )

        self.u = nnx.bridge.ToNNX(
            nn.Dense(
                cfg.d_hidden,  # out features only; d_in is inferred
                use_bias=False,
                kernel_init=nnx.initializers.normal(0.02),
                param_dtype=cfg.param_dtype,
                dtype=cfg.dtype,
                dot_general_cls=nn.fp8_ops.Fp8DirectDotGeneralOp,
            ),
            rngs=rngs,
        )

        # bloatf16
        # self.g = nnx.Linear(
        #     cfg.d_model,
        #     cfg.d_hidden,
        #     use_bias=False,
        #     kernel_init=nnx.initializers.normal(0.02),
        #     param_dtype=cfg.param_dtype,
        #     dtype=cfg.dtype,
        #     rngs=rngs,
        # )

        self.g = nnx.bridge.ToNNX(
            nn.Dense(
                cfg.d_hidden,  # out features only; d_in is inferred
                use_bias=False,
                kernel_init=nnx.initializers.normal(0.02),
                param_dtype=cfg.param_dtype,
                dtype=cfg.dtype,
                dot_general_cls=nn.fp8_ops.Fp8DirectDotGeneralOp,
            ),
            rngs=rngs,
        )

        # bfloat16
        # self.d = nnx.Linear(
        #     cfg.d_hidden,
        #     cfg.d_model,
        #     use_bias=False,
        #     kernel_init=nnx.initializers.normal(0.02),
        #     param_dtype=cfg.param_dtype,
        #     dtype=cfg.dtype,
        #     rngs=rngs,
        # )

        self.d = nnx.bridge.ToNNX(
            nn.Dense(
                cfg.d_model,  # out features only; d_in is inferred
                use_bias=False,
                kernel_init=nnx.initializers.normal(0.02),
                param_dtype=cfg.param_dtype,
                dtype=cfg.dtype,
                dot_general_cls=nn.fp8_ops.Fp8DirectDotGeneralOp,
            ),
            rngs=rngs,
        )

    def __call__(
        self, x: Float[Array, "batch seq d_model"]
    ) -> Float[Array, "batch seq d_model"]:
        g_proj = self.g(x)
        u_proj = self.u(x)
        return self.d(u_proj * nnx.silu(g_proj))


# Attention
class Attention(nnx.Module):
    def __init__(self, cfg: ModelConfig, rngs: nnx.Rngs):
        self.cfg = cfg
        # bloatf16
        # self.qkv_proj = nnx.Linear(
        #     cfg.d_model,
        #     3 * cfg.d_model,
        #     use_bias=False,
        #     kernel_init=nnx.initializers.normal(0.02),
        #     param_dtype=cfg.param_dtype,
        #     dtype=cfg.dtype,
        #     rngs=rngs,
        # )
        self.qkv_proj = nnx.bridge.ToNNX(
            nn.Dense(
                3 * cfg.d_model,
                use_bias=False,
                kernel_init=nnx.initializers.normal(0.02),
                param_dtype=cfg.param_dtype,
                dtype=cfg.dtype,
                dot_general_cls=nn.fp8_ops.Fp8DirectDotGeneralOp,
            ),
            rngs=rngs,
        )
        self.q_norm = nnx.RMSNorm(
            cfg.d_head, param_dtype=cfg.param_dtype, dtype=cfg.dtype, rngs=rngs
        )
        self.k_norm = nnx.RMSNorm(
            cfg.d_head, param_dtype=cfg.param_dtype, dtype=cfg.dtype, rngs=rngs
        )
        # bfloat16
        # self.out_proj = nnx.Linear(
        #     cfg.d_model,
        #     cfg.d_model,
        #     use_bias=False,
        #     kernel_init=nnx.initializers.normal(0.02),
        #     param_dtype=cfg.param_dtype,
        #     dtype=cfg.dtype,
        #     rngs=rngs,
        # )
        self.out_proj = nnx.bridge.ToNNX(
            nn.Dense(
                cfg.d_model,
                use_bias=False,
                kernel_init=nnx.initializers.normal(0.02),
                param_dtype=cfg.param_dtype,
                dtype=cfg.dtype,
                dot_general_cls=nn.fp8_ops.Fp8DirectDotGeneralOp,
            ),
            rngs=rngs,
        )

    def __call__(
        self,
        x: Float[Array, "batch seq d_model"],
        cos: Float[Array, "seq d_head"],
        sin: Float[Array, "seq d_head"],
    ) -> Float[Array, "batch seq d_model"]:
        qkv_proj = self.qkv_proj(x)
        q, k, v = jnp.split(qkv_proj, 3, axis=-1)
        q = q.reshape(*x.shape[:-1], self.cfg.n_heads, self.cfg.d_head)
        k = k.reshape(*x.shape[:-1], self.cfg.n_heads, self.cfg.d_head)
        v = v.reshape(*x.shape[:-1], self.cfg.n_heads, self.cfg.d_head)

        q = apply_rope(self.q_norm(q), cos, sin)
        k = apply_rope(self.k_norm(k), cos, sin)

        out = jax.nn.dot_product_attention(
            q, k, v, is_causal=True, implementation="cudnn"
        )
        out = out.reshape(*x.shape)

        return self.out_proj(out)


class AttentionBlock(nnx.Module):
    def __init__(self, cfg: ModelConfig, rngs: nnx.Rngs):
        self.cfg = cfg
        self.attn = Attention(cfg, rngs)
        self.attn_norm = nnx.RMSNorm(
            cfg.d_model, param_dtype=cfg.param_dtype, dtype=cfg.dtype, rngs=rngs
        )
        self.swiglu = SwiGLU(cfg, rngs)
        self.mlp_norm = nnx.RMSNorm(
            cfg.d_model, param_dtype=cfg.param_dtype, dtype=cfg.dtype, rngs=rngs
        )

    def __call__(
        self,
        x: Float[Array, "batch seq d_model"],
        cos: Float[Array, "seq d_head"],
        sin: Float[Array, "seq d_head"],
    ) -> Float[Array, "batch seq d_model"]:
        x = x + self.attn_norm(self.attn(x, cos, sin))
        return x + self.mlp_norm(self.swiglu(x))


class LLM(nnx.Module):
    def __init__(self, cfg: ModelConfig, rngs: nnx.Rngs):
        self.embed = nnx.Embed(
            cfg.vocab_size,
            cfg.d_model,
            param_dtype=cfg.param_dtype,
            dtype=cfg.dtype,
            embedding_init=nnx.initializers.normal(0.02),
            rngs=rngs,
        )
        self.layers = nnx.List([AttentionBlock(cfg, rngs) for _ in range(cfg.n_layers)])
        self.norm_out = nnx.RMSNorm(
            cfg.d_model, param_dtype=cfg.param_dtype, dtype=cfg.dtype, rngs=rngs
        )

    def __call__(
        self,
        x: Int[Array, "batch seq"],
        cos: Float[Array, "seq d_head"],
        sin: Float[Array, "seq d_head"],
    ) -> Float[Array, "batch seq vocab"]:
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.norm_out(x)
        return self.embed.attend(x)


def generate(
    model: LLM,
    cfg: ModelConfig,
    prompt: Int[Array, "batch seq"],
    max_new_tokens: int,
    *,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    key: Optional[int] = None,
):
    """Autoregressive sampling. prompt: (batch, seq) int32."""

    if temperature != 0.0 and key is None:
        raise ValueError("key required when temperature > 0")

    tokens = prompt
    for _ in range(max_new_tokens):
        cos, sin = rope_tables(cfg, tokens.shape[1])
        logits = model(tokens, cos, sin)[:, -1, :]  # only the last position

        if temperature == 0.0:  # greedy
            nxt = jnp.argmax(logits, axis=-1)
        else:
            logits = logits.astype(jnp.float32) / temperature
            if top_k is not None:
                kth = jnp.sort(logits, axis=-1)[:, -top_k, None]
                logits = jnp.where(logits < kth, -jnp.inf, logits)
            key, sub = jax.random.split(key)  # type: ignore
            nxt = jax.random.categorical(sub, logits, axis=-1)

        tokens = jnp.concatenate([tokens, nxt[:, None]], axis=1)
    return tokens


if __name__ == "__main__":
    cfg = ModelConfig(
        seed=0,
        vocab_size=24_576,
        d_model=768,
        n_heads=12,
        d_hidden=2048,
        n_layers=12,
        dtype=jnp.bfloat16,
        param_dtype=jnp.float32,
    )
    # TODO: Refactor into TrainConfig
    batch = 8
    seq_len = 1024

    tokens = jnp.asarray(np.random.randint(0, cfg.vocab_size, (batch, seq_len + 1)))
    x = tokens[:, :-1]  # Everything but last
    y = tokens[:, 1:]  # Shift window by 1

    cos, sin = rope_tables(cfg, seq_len)
    model = LLM(cfg, nnx.Rngs(params=cfg.seed))

    n_params = sum(leaf.size for leaf in jax.tree.leaves(nnx.state(model, nnx.Param)))
    print(f"params: {n_params:,}")
    assert n_params == 11_112_448, f"debug config param count changed: {n_params:,}"

    optimizer = nnx.Optimizer(model, optax.adamw(1e-3), wrt=nnx.Param)
    print(
        f"ln(vocab) = {float(jnp.log(cfg.vocab_size)):.5f}  <-- where it should start"
    )
    for step in range(201):
        loss = train_step(model, optimizer, x, cos, sin, y)
        if step % 10 == 0:
            print(f"step: {step:>4}  loss {float(loss):.5f}")
