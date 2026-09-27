"""Build the model and restore it from the Orbax checkpoints train.py writes.

Checkpoint layout (train.py): checkpoints/<run_id>/<step>/default, holding
    {"model": nnx.state(model), "opt": nnx.state(optimizer), "step": int, "grain": iterator state}
`step` is the micro-step index (accum loop), not the optimizer step.

Two ways back in:
    restore_model        weights (+ FP8 scales) only -- eval, SFT, chat
    restore_train_state  everything -- resuming a pretraining run
"""

from pathlib import Path
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from flax import nnx

from model import LLM, ModelConfig, rope_tables


def build_model(cfg: ModelConfig, seq_len: int) -> LLM:
    """Construct the LLM with every variable materialized.

    The linears are linen Dense behind nnx.bridge, which has no kernels until it
    sees an input (d_in is inferred). lazy_init runs one batch-1 forward pass to
    create them -- required before building an optimizer, counting params, or
    restoring, since all three need the full state tree.
    """
    model = LLM(cfg, nnx.Rngs(params=cfg.seed))
    cos, sin = rope_tables(cfg, seq_len)
    nnx.bridge.lazy_init(model, jnp.zeros((1, seq_len), jnp.int32), cos, sin)
    return model


def latest_step(ckpt_dir: Path) -> int:
    with ocp.CheckpointManager(Path(ckpt_dir).absolute()) as mgr:
        step = mgr.latest_step()
    if step is None:
        raise FileNotFoundError(f"no checkpoints in {ckpt_dir}")
    return step


def restore_model(ckpt_dir: Path, model: LLM, step: Optional[int] = None) -> int:
    """Load weights and FP8 scaling state into `model` in place. Returns the step.

    Only the "model" subtree is read (partial restore), so this works without
    rebuilding the pretraining optimizer -- SFT and chat build their own, or none.
    """
    ckpt_dir = Path(ckpt_dir).absolute()
    step = latest_step(ckpt_dir) if step is None else step
    target = {"model": nnx.state(model)}
    with ocp.CheckpointManager(ckpt_dir) as mgr:
        r = mgr.restore(
            step,
            args=ocp.args.PyTreeRestore(
                item=target,
                restore_args=ocp.checkpoint_utils.construct_restore_args(target),
                partial_restore=True,
            ),
        )
    nnx.update(model, r["model"])
    return step


def restore_train_state(
    ckpt_dir: Path,
    model: LLM,
    optimizer: nnx.Optimizer,
    grain_state: dict[str, Any],
    step: Optional[int] = None,
) -> tuple[int, dict[str, Any]]:
    """Load model, optimizer and data position for resuming. Returns (step, grain_state).

    `model` and `optimizer` must be built exactly as the saved run built them
    (same config, same optax chain) -- the tree has to match leaf for leaf.
    `grain_state` is any iterator's get_state(), used only as the shape to restore
    into. The saved `step` has already been trained on: resume the loop at step + 1,
    after it.set_state(returned_grain_state).
    """
    ckpt_dir = Path(ckpt_dir).absolute()
    step = latest_step(ckpt_dir) if step is None else step
    target = {
        "model": nnx.state(model),
        "opt": nnx.state(optimizer),
        "step": 0,
        "grain": grain_state,
    }
    with ocp.CheckpointManager(ckpt_dir) as mgr:
        r = mgr.restore(step, args=ocp.args.StandardRestore(target))
    nnx.update(model, r["model"])
    nnx.update(optimizer, r["opt"])
    # Orbax hands scalars back as numpy arrays; Grain's set_state wants plain ints.
    grain = jax.tree.map(lambda v: v.item() if isinstance(v, np.ndarray) else v, r["grain"])
    return int(r["step"]), grain
