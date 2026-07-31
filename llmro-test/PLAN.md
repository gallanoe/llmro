# Implementation Plan

`DESCRIPTION.md` §8 is the **run** plan — what happens once the code exists and you launch. This is the **build** plan: how you get from an empty folder to code that can execute §8 at all. It ends where §8 begins.

## The ordering principle

Build **vertically, not horizontally.** The tempting structure is to finish the tokenizer, then the data pipeline, then the model, then the training loop — each complete before the next. Don't.

Instead: get a thin end-to-end path running on fake data as early as possible, then replace each fake piece with a real one. This follows directly from §2 of the spec — the debug config is the thing you live in, so reach it in the first sitting rather than the fifth. Horizontal ordering has you spending an evening on a tokenizer before knowing whether JAX can do a backward pass on this card.

The practical consequence: **loss going down on a fixed batch is a real milestone.** It proves forward, backward, optimizer, jit, donation, and generation all work, and it costs nothing to get wrong.

---

## Status

- **A — Environment gate: ✅ done.** 229.5 TFLOPS measured (above the 209 spec sheet), FP32-accumulate confirmed full rate, cuDNN attention verifiably selected.
- **B — Vertical slice: ✅ done.** Model, train step, and sampling all run; loss collapses on a fixed batch; param counts exact at 11,112,448 / 103,829,760; MFU 0.58 on the real config. Phase 2 of `DESCRIPTION.md` §8 got measured early and came in at nearly 2× the planning assumption.
- **Next: C — real tokenizer and real data.**

One environment fact from B that outranks everything else in this file: **`XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async` must be set** or XLA can't allocate past ~4.6 GB of the 32 GB card. See `DESCRIPTION.md` §5.

---

## A — Environment gate

Confirm JAX runs on sm_120 at all, and measure what the card actually does.

A small standalone benchmark is enough: allocate two large bf16 matrices, matmul in a loop, print achieved TFLOPS. Compare against the 209 spec-sheet figure. That single number answers the highest-leverage unknown in the spec — whether GB202 runs BF16 with FP32 accumulate at full or half rate — and therefore whether your run is 15 hours or 30. Separately, confirm `dot_product_attention(..., implementation='cudnn')` neither throws nor silently falls back to the XLA path.

This comes first because it's twenty minutes and it can invalidate the entire project. Nothing else is worth building until it passes.

**Result: 229.5 TFLOPS, full rate, cuDNN confirmed.** It resolved the good way.

**What this gate should have also included, in hindsight:** a single large allocation. Matmul benchmarks use small buffers and pass happily on a box that cannot allocate more than 4.6 GB — which is exactly the state this machine was in for all of Phase A and most of Phase B, undiagnosed. Add "allocate 20 GB in one array" to the environment gate; it's one line and it would have caught the allocator problem before it got mistaken for a model-level memory wall.

## B — Vertical slice on fake data

The whole pipeline's shape, none of its content.

Write the model — RoPE, SwiGLU, RMSNorm, reordered norm placement, QK-norm — a training step, AdamW via Optax, and a generation function. Feed it `np.random.randint`. Run the debug config for a few hundred steps.

**The milestone is overfitting a single fixed batch**, not "loss goes down on random tokens." Those are different tests and only the first one means anything. Freshly-generated random tokens are unlearnable by construction — the loss will sit at `ln(vocab) ≈ 10.1` forever no matter how correct your code is, because there is no mapping from input to target to find. Draw *one* batch, keep it, and train on it repeatedly: loss should start near 10.1 (the tied-embedding init at std 0.02 puts it there) and drive toward zero. Measured: 10.16 → 0.001 in 200 steps. That collapse is the signal that forward, backward, and the optimizer are all wired correctly.

This is where you fight JAX rather than machine learning: NNX idioms, `donate_argnums`, static shapes, recompilation, shape errors in attention. Fight it against data you don't care about. The param counter belongs here too, asserting against the ~11M and ~104M targets before either number has cost you anything.

Resist adding checkpointing, real data, or SFT. Those are later phases and they will make this one harder to debug.

## C — Real tokenizer and real data

Now the content, in an order that's counterintuitive but forced.

**Decide the chat template first** — the tokenizer needs its special tokens reserved, and discovering otherwise after tokenizing 5B tokens is the most expensive mistake available here. Then train the 24k BPE on a text sample, then tokenize the corpus to `uint16` shards, then write the Grain loader and swap out the fake data.

The corpus tokenization is a multi-hour CPU job — start it and go do something else. While you're touching the metadata anyway, check what score ≥ 4 actually yields, since the anneal budget depends on it.

The loss curve should now look qualitatively different from Phase B. Real text is learnable; random tokens aren't.

## D — Durability

Orbax async checkpointing, Grain's iterator state, and the compilation cache.

The debug config makes this testable in seconds rather than hours: run, kill the process, resume, look at whether the curve is continuous and the data order replayed. It's the least interesting phase in the plan and the one that saves the project when hour 11 of 15 dies to a power blip.

## E — Schedule and anneal

WSD, plus the data mixture swap that coincides with decay onset.

Mechanically this is two data sources and a switch at a step count, but it's the first time the schedule and the data pipeline have to agree with each other. The debug config runs the entire schedule in ~1,500 steps, so you can watch the whole shape — warmup, stable, decay — in a couple of minutes.

The held-out bits-per-byte eval loop lands here too, since a visible drop at the swap is how you know the swap actually happened.

## F — SFT

The second training mode: chat template applied, loss masked to assistant tokens only.

This has the highest bug density in the project, and its bugs are silent — a wrong mask doesn't crash, it just produces a model that hallucinates user turns and talks to itself. The decode-and-inspect test belongs here: print a batch's tokens alongside whether each position carries loss, and read it with your eyes.

Then generate. The debug model's output will be gibberish — it's 11M params on 50M tokens — but it should be gibberish *shaped like a reply* rather than a continuation. That distinction is the whole point of the phase.

**At the end of F you have a complete pipeline.** Everything after this is scale.

## G — Scale-up

Point the existing code at the real config.

Most of this already happened in Phase B, ahead of schedule: MFU is 0.58 against the 229.5 TFLOPS Phase A ceiling, the batch ceiling is device batch 24 (the spec's 128 was over by 5×), and the 5B-vs-3B decision resolves to **5B** with room to spare.

What's left for this phase is the part that needs real data: **re-measure MFU with Grain in the loop.** Every Phase B number is fake in-memory data with no dataloader, no checkpointing, no z-loss, and no gradient clipping — an optimistic ceiling by construction. A loader that doesn't prefetch is the classic way to give back 20 points of MFU, and it will not show up until the real pipeline is attached.

Then run the Muon A/B in the debug loop, which is a five-minute experiment now that everything else works.

Nothing new gets built here. You're turning knobs on code that already runs.

## H — The run

`DESCRIPTION.md` §8 takes over: launch, checkpoint every 30 minutes, watch the gradient norm, anneal, SFT, talk to it.

Keep the log while it runs. Run two is where μP, more tokens, and the depth-versus-width question live — and the log is the only thing that makes run two smarter than run one rather than just longer.
