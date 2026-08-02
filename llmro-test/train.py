import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax import nnx
import orbax.checkpoint as ocp
import grain.python as g
from model import LLM, ModelConfig, rope_tables
import json
import hashlib
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from tensorboardX import SummaryWriter
from typing import Tuple
from jaxtyping import Array, Int, Float


class NpyDataSource(g.RandomAccessDataSource):
    def __init__(self, paths: list[Path], seq_len: int = 1024):
        super().__init__()
        self.paths = [str(p) for p in paths]
        self.seq_len = seq_len
        lens = []
        for p in self.paths:
            with open(p, "rb") as f:
                shape = (
                    np.lib.format.read_magic(f)
                    and np.lib.format.read_array_header_1_0(f)
                )[0]
            lens.append(shape[0])
        counts = [(n - 1) // seq_len for n in lens]
        self.cum = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        self._mm = None

    @property
    def mm(self):
        if self._mm is None:
            self._mm = [np.load(p, mmap_mode="r") for p in self.paths]
        return self._mm

    def __getstate__(self):
        return {**self.__dict__, "_mm": None}  # drop handles for pickling

    def __len__(self):
        return int(self.cum[-1])

    def __getitem__(self, i):
        i = int(i)
        f = int(np.searchsorted(self.cum, i, side="right") - 1)
        off = (i - int(self.cum[f])) * self.seq_len
        return np.asarray(self.mm[f][off : off + self.seq_len + 1], dtype=np.uint16)

    def __repr__(self):
        return f"TokenWindows(n={len(self.paths)}, seq_len={self.seq_len}, total={len(self)})"


@nnx.jit
def train_step(
    model: LLM,
    optimizer: nnx.Optimizer,
    x: Int[Array, "batch seq"],
    cos: Float[Array, "seq d_head"],
    sin: Float[Array, "seq d_head"],
    y: Int[Array, "batch seq"],
):
    def loss_fn(m: LLM):
        logits = m(x, cos, sin)
        return optax.softmax_cross_entropy_with_integer_labels(
            logits.astype(jnp.float32), y
        ).mean()

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss, optax.global_norm(grads)


@nnx.jit
def eval_step(
    model: LLM,
    x: Int[Array, "batch seq"],
    cos: Float[Array, "seq d_head"],
    sin: Float[Array, "seq d_head"],
    y: Int[Array, "batch seq"],
) -> Tuple[float, int]:
    logits = model(x, cos, sin)
    ce = optax.softmax_cross_entropy_with_integer_labels(logits.astype(jnp.float32), y)
    return ce.sum(), jnp.asarray(ce.size, jnp.float32)


def train():
    batch_size = 16
    accum_steps = 16
    seq_len = 1024
    n_steps = 40_000
    checkpoint_every = 1_000
    eval_every = 500

    CHECKPOINT_DIR = Path("./checkpoints/")
    RUNS_DIR = Path("./runs/")

    meta = json.loads(Path("./datasets/fineweb-edu-sample-10BT/meta.json").read_text())
    eval_bpt = meta["heldout_totals"]["bytes_per_token"]
    n_windows = meta["totals"]["base_windows"]

    eval_data = (
        g.MapDataset.source(
            NpyDataSource(
                paths=list(
                    Path("./datasets/fineweb-edu-sample-10BT/heldout").rglob("*")
                ),
                seq_len=seq_len,
            )
        )
        .map(lambda b: (b[:-1], b[1:]))
        .batch(batch_size=batch_size)
        .to_iter_dataset()
    )

    train_data = (
        g.MapDataset.source(
            NpyDataSource(
                paths=list(Path("./datasets/fineweb-edu-sample-10BT/base").rglob("*")),
                seq_len=seq_len,
            )
        )
        .shuffle(seed=42)
        .repeat()
        .map(lambda b: (b[:-1], b[1:]))
        .batch(batch_size=batch_size)
        .to_iter_dataset()
    )

    # Create model
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
    peak_lr = 1e-3
    warmup = int(0.02 * n_steps)
    decay = int(0.12 * n_steps)
    schedule = optax.join_schedules(
        [
            optax.linear_schedule(0, peak_lr, warmup),
            optax.constant_schedule(peak_lr),
            optax.linear_schedule(peak_lr, 0, decay),
        ],
        boundaries=[warmup, n_steps - decay],
    )

    cos, sin = rope_tables(cfg, seq_len)
    model = LLM(cfg, nnx.Rngs(params=cfg.seed))

    def decay_mask(state):
        return jax.tree.map_with_path(
            lambda path, _: "['kernel']" in jax.tree_util.keystr(path), state
        )

    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(schedule, b1=0.9, b2=0.95, weight_decay=0.1, mask=decay_mask),
    )
    tx = optax.MultiSteps(tx, every_k_schedule=accum_steps)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)  # type: ignore

    it = iter(train_data)
    writer = SummaryWriter()

    with ocp.CheckpointManager(CHECKPOINT_DIR.absolute()) as mgr:
        for step in tqdm(range(n_steps * accum_steps)):
            x, y = next(it)
            loss, grad_norm = train_step(model, optimizer, x, cos, sin, y)
            if step % accum_steps == 0:
                writer.add_scalar("train/loss", loss, step)
                writer.add_scalar("train/grad_norm", grad_norm, step)
            if step % (eval_every * accum_steps) == 0:
                nats, toks = 0.0, 0.0
                for i, (ex, ey) in enumerate(eval_data):
                    n, t = eval_step(model, ex, cos, sin, ey)
                    nats += float(n)
                    toks += float(t)
                bpb = (nats / np.log(2)) / (toks * eval_bpt)
                writer.add_scalar("eval/loss", nats / toks, step)
                writer.add_scalar("eval/bpb", bpb, step)
            if step % (checkpoint_every * accum_steps) == 0:
                ckpt = {
                    "model": nnx.state(model),
                    "opt": nnx.state(optimizer),
                    "step": step,
                    "grain": it.get_state(),
                }
                mgr.save(step, args=ocp.args.StandardSave(ckpt))


def load_checkpoint(path: Path):
    # TODO: Update to match actual training loop
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

    n_steps = 40_000
    peak_lr = 1e-3
    warmup = int(0.02 * n_steps)
    decay = int(0.12 * n_steps)
    schedule = optax.join_schedules(
        [
            optax.linear_schedule(0, peak_lr, warmup),
            optax.constant_schedule(peak_lr),
            optax.linear_schedule(peak_lr, 0, decay),
        ],
        boundaries=[warmup, n_steps - decay],
    )

    def decay_mask(state):
        return jax.tree.map_with_path(
            lambda path, _: "['kernel']" in jax.tree_util.keystr(path), state
        )

    tx = optax.adamw(schedule, b1=0.9, b2=0.95, weight_decay=0.1, mask=decay_mask)
    model = LLM(cfg, nnx.Rngs(params=cfg.seed))
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    with ocp.CheckpointManager(path.absolute()) as mgr:
        r = mgr.restore(
            mgr.latest_step(),
            args=ocp.args.StandardRestore(
                {
                    "model": nnx.state(model),
                    "opt": nnx.state(optimizer),
                    "step": 0,
                    "grain": None,
                }
            ),
        )
    nnx.update(model, r["model"])
    nnx.update(optimizer, r["opt"])


if __name__ == "__main__":
    train()
