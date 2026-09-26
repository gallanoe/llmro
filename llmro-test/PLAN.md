# Implementation Plan

`DESCRIPTION.md` is the spec: what we're building and why. This file is the **build plan**: the order to build it in, where each phase stands, and what's still open. Phase H hands over to `DESCRIPTION.md` §8 (the run itself).

*Last updated 2026-09-26.*

---

## Status at a glance

| Phase | What | Status |
|---|---|---|
| A | Environment gate | ✅ done |
| B | Vertical slice on fake data | ✅ done |
| C | Real tokenizer and data | ✅ done |
| D | Durability (checkpoint / resume) | 🟡 save works; **resume not built** |
| E | WSD schedule + anneal swap + eval | 🟡 built; edge cases open |
| — | FP8 training (added, not in original plan) | 🟡 built and verified on one step; first run in progress |
| F | SFT | ⬜ not started |
| G | Scale-up measurements | 🟡 partly done in B; Grain-in-loop MFU and Muon A/B open |
| H | The run | ⬜ blocked on D, E and F |

**Two environment facts that outrank everything below:**
- **`XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async` must be set** before JAX starts, or XLA can't allocate more than ~4.6 GB of the 32 GB card (`DESCRIPTION.md` §5).
- **Filter the anneal set on `int_score`, never the float `score`.** The wrong column fails silently and gives a pool 4.5× too small (`DESCRIPTION.md` §4).

---

## Where things stand (2026-09-26)

### Code
- **`model.py`**: the real 768/12 architecture. All five linears per block (`qkv_proj`, `out_proj`, SwiGLU `u`/`g`/`d`) are FP8 via linen `nn.Dense` + `Fp8DirectDotGeneralOp`, wrapped with `nnx.bridge.ToNNX`. Embeddings, logits and attention stay bf16. There's also an unfinished native `Fp8Linear` (its `__call__` is a stub) for a later hand-written version.
- **`train.py`**:
  - Grain pipelines for base, anneal and heldout.
  - WSD schedule; AdamW inside `MultiSteps` (16 × 16 × 1024 = 262k tokens/step).
  - FP8-aware `train_step`: `DiffState` over Param + OWG, Params to the optimizer, OWG written back with `nnx.update`.
  - Base loop then anneal loop on one global step counter.
  - bpb eval every 500 steps, Orbax save every 1,000.
  - A per-run `run_id` shared by `checkpoints/<run_id>/` and `runs/<run_id>/`.
- **Current config:** `n_steps = 14_000`, which is ~3.7B tokens, below the spec's 5B / ~19k steps. Decay is 12%, i.e. 1,680 steps and ~0.44B anneal tokens. Both pools fit without repeating.

### Runs
| Run | Reached | Result | Notes |
|---|---|---|---|
| `Aug01_*` (old) | ~21k opt steps | train loss 3.29, bpb 1.042 at ~6k | Logs and checkpoints didn't match; deleted. |
| `2026-09-26_051719` | ~28.5k opt steps, 8.6 h | train loss 3.03, **eval bpb 0.985** | Stopped early. Its config wasn't saved, so `n_steps` and bf16/FP8 are unknown. ~242k tok/s averaged over the whole run. |
| `2026-09-26_135412` | in progress | — | `n_steps = 14_000` |

---

## Open issues

Most important first.

1. **The last ≤1,000 steps are never saved or evaluated.** Evals and checkpoints only fire on step multiples, so the final anneal steps (the most valuable weights) are lost at the end of the run. Add one eval and one `mgr.save` after the anneal loop.
2. **There's no resume path.** `load_checkpoint` is stale: its optimizer lacks `clip` and `MultiSteps`, it doesn't `lazy_init`, it doesn't return the model, and nothing calls it. Resuming also needs to know whether it's in base or anneal (derive that from the step) and must pass `purge_step` to `SummaryWriter`.
3. **There's no eval at the switch.** The swap step (`n_steps - decay`) generally isn't a multiple of `eval_every`. Run one eval just before the anneal loop so the bpb curve has a point exactly at the swap.
4. **Run configs aren't saved.** Write `config.json` (the `TrainConfig`) into `checkpoints/<run_id>/` at launch. Without it, the 05:17 run can't be interpreted.
5. **`train/grad_norm` reads high.** It's the mean of micro-batch norms, not the norm of the accumulated gradient that clipping sees.
6. **Base and anneal loops are copies.** Every fix has to be made twice; one loop that swaps the iterator at the switch step would remove that.
7. **`model.py`'s `__main__` is broken:**
   - it calls `train_step`, which moved to `train.py` in c5449c3;
   - there's no `lazy_init`;
   - it asserts the debug param count against the real config.
8. **There's no debug config.** Everything is hardcoded to 768/12, so the spec's "debug until green" loop (§2) isn't available. That matters most for testing resume (D) and SFT (F).
9. **The spec's anneal mixture isn't done.** 3–5% smoltalk2 `Mid` instruction data is specified but not downloaded or tokenized. The anneal is currently pure FineWeb-Edu `int_score ≥ 4`.
10. **No compilation cache**, so every launch pays the full XLA compile time.

---

## The ordering principle

Build **vertically, not horizontally.** Don't finish the tokenizer, then the data pipeline, then the model, then the loop, each complete before the next. Instead, get a thin end-to-end path running on fake data first, then replace each fake piece with a real one.

The practical consequence: **loss collapsing on one fixed batch is a real milestone.** It proves forward, backward, optimizer, jit and generation all work, and it costs nothing to get wrong.

---

## A: Environment gate ✅

**Goal:** confirm JAX runs on sm_120 and measure what the card actually does, before building anything.

**Result:**
- **229.5 TFLOPS** sustained BF16, above the 209 on the spec sheet. That means BF16 with FP32 accumulate runs at full rate.
- `dot_product_attention(implementation='cudnn')` verifiably selects cuDNN.

**Lesson:** the gate should also have tried one ~20 GB allocation. Matmul benchmarks use small buffers, so they passed on a machine that couldn't allocate more than 4.6 GB. That went undiagnosed through all of A and most of B.

## B: Vertical slice on fake data ✅

**Goal:** the whole pipeline's shape, none of its content. The model (RoPE, SwiGLU, RMSNorm, reordered norms, QK-norm), AdamW, a train step and generation, all on `np.random.randint`.

**Result:**
- One fixed batch overfits from 10.16 to 0.001 in 200 steps.
- Param counts are exact: 11,112,448 (debug) and 103,829,760 (real).
- MFU is 0.58 on the real config: 214k tok/s at device batch 24. Batch 32 runs out of memory.

**Lesson:** overfit **one fixed** batch. Freshly drawn random tokens are unlearnable, so their loss stays at ln(vocab) even when the code is correct.

## C: Real tokenizer and real data ✅

**Order (forced):** chat template → BPE → tokenize corpus → Grain loader.

**Result:**
- ChatML special tokens (`<|im_start|>`, `<|im_end|>`) were reserved before training the 24k BPE.
- All 8 downloaded `sample-10BT` shards are tokenized to uint16, split into exclusive pools at tokenize time:

| Pool | Rule | Tokens |
|---|---|---|
| base | `int_score == 3`, minus heldout | 5.31B |
| anneal | `int_score >= 4` | 0.87B |
| heldout | 1 in 1000 base docs, by blake2b(id) | 5.5M |

**Notes:**
- More unique data is available if needed. `sample-10BT` holds ~10B GPT-2 tokens, and only part of it has been downloaded.
- Presorting before tokenization isn't needed. The split happens in the tokenize pass, and Grain shuffles windows anyway.

## D: Durability 🟡

**Goal:** a run can die at any point and resume with a continuous loss curve and the same data order.

**Done:**
- Orbax saves model, optimizer, step and Grain iterator state.
- Per-run directories.
- Orbax verified able to save the bridged FP8 state, RNG keys included.

**Left:**
- The resume path (open issue 2).
- `config.json` per run (issue 4).
- The compilation cache (issue 10).
- The kill-and-resume test itself: run, kill, resume, and check the curve is continuous and the data order replayed. Do it on the debug config, where it takes seconds.

## E: Schedule and anneal 🟡

**Goal:** a WSD schedule, with the data switching from base to anneal at decay onset, and heldout bpb to confirm the switch had an effect.

**Done:**
- The WSD schedule: 2% warmup, stable, 12% linear decay to 0.
- Base loop then anneal loop on one global step.
- Heldout bpb every 500 steps.

**Design decisions (settled):**
- **One optimizer throughout.** The anneal is the last stretch of the same run, not a new run. Don't reset Adam's state at the switch.
- **The switch step is fixed in advance** at `n_steps - decay`. Nothing in the loss triggers it.
- **Heldout stays the same across the switch.** It's the one consistent yardstick. An optional `eval/bpb_anneal` on a few held-back anneal windows would measure the gain on the anneal data directly.
- **Size the decay against the anneal pool:** `decay × 262k` must fit in 0.87B tokens, or the anneal repeats data. ~1.5 epochs is acceptable if needed; a cap on `decay` or more shards also works.

**Left:** open issues 1, 3 and 9.

## FP8 training 🟡 (added)

**Why:** the 5090 runs FP8 matmuls at ~3× BF16.

| Matmul (8192², measured) | TFLOPS | Kernel |
|---|---|---|
| bf16 | 226.7 | `cublas$lt$matmul` |
| e4m3 direct `dot_general` | ~700 | `cublas$lt$matmul$f8` |
| e4m3 → convert → × scale → bf16 dot | 223.4 | plain bf16: **the rewrite to FP8 didn't happen** |

**Design:**
- **FP8 covers the linears only:** ~75% of FLOPs (169.9M of 226.5M per token). Logits (17%) and attention (8%) stay bf16. The best-case end-to-end speedup is ~2×.
- **Formats:** E4M3 for weights and activations, E5M2 for gradients.
- **Scaling:** delayed per-tensor scaling, using Flax's amax history.
- **Master weights stay f32** (`param_dtype`); FP8 copies only exist during the matmul.
- **Use `Fp8DirectDotGeneralOp`,** not the deprecated `Fp8DotGeneralOp`. Its convert-then-scale pattern is the one that didn't turn into an FP8 kernel.
- **The OWG mechanism:** the new scales come back out of the backward pass as "gradients". So `train_step` must differentiate `nnx.Any(nnx.Param, OWG)` through `argnums=DiffState(...)`, send only Params to the optimizer, and write the OWG values back with `nnx.update`.

**Verified on the real model (one step):**
- 103,829,760 params, unchanged.
- 360 OWG leaves.
- 168 `cublas$lt$matmul$f8` calls in the compiled step.
- Loss decreases and gradients are nonzero.

**Traps:**
- **Without `argnums=diff`, every gradient silently rounds to zero.** The scales stay at 1.0 and gradients fall below E5M2's range. Measured: grad norm 0.0, 100% zeros, no crash.
- **Linen `Dense` needs `nnx.bridge.lazy_init(model, x, cos, sin)`** before the optimizer or the param count. It infers `d_in` from a sample input.
- **Checkpoints from before FP8 won't restore:** the state tree changed.

**Left:**
- Confirm end-to-end speed against bf16 on a real run.
- Check that eval under FP8 matches bf16 on the final checkpoint, before SFT and generation (which may run in bf16).
- The hand-written `Fp8Linear` is optional, a learning exercise.

## F: SFT ⬜

**Goal:** the second training mode. Apply the chat template and compute loss on assistant tokens only.

This phase has the most bugs, and they're silent. A wrong mask doesn't crash; it produces a model that writes the user's turns too and talks to itself. **The inspect test:** decode one batch and print each token next to whether it carries loss, then read it yourself.

Then generate. The debug model's output will be gibberish, but it should be gibberish **shaped like a reply**, not a continuation.

**At the end of F the pipeline is complete.** Everything after is scale.

## G: Scale-up 🟡

**Done in B:** MFU 0.58, the device-batch limit is 24, and 5B tokens fits the budget.

**Left:**
- **Re-measure throughput with Grain in the loop.** B used fake in-memory data. The 05:17 run averaged ~242k tok/s over its full wall time, but its config is unknown.
- **Re-check the best device batch under FP8.** Memory use changed. Keep the effective batch near 262k: 24 × 11 or 32 × 8.
- **Muon A/B** in the debug loop.

## H: The run ⬜

`DESCRIPTION.md` §8 takes over: launch, checkpoint regularly, watch the grad norm, anneal, SFT, talk to it.

**Before launching:**
- Clear open issues 1–4.
- Settle `n_steps`. The spec says ~19k (5B tokens). 40k (~10.5B, ~13 h in bf16) is viable, with ~1.7 epochs of base or more shards downloaded. The current 14k is ~3.7B.

Keep the log while it runs. It's what makes run two smarter than run one, not just longer.
