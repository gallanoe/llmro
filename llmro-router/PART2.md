# Part 2: End-to-end bypass training pilot

Sep 28, 2026 · @Eero Gallano

## Goal

Build and validate end-to-end training of learned block bypasses for per-token routing on a frozen Qwen3.5-0.8B, as a pilot on a small dataset. Success means a working training loop whose trained bypasses cause less damage than regression-fitted bypasses under the same routing plans, with every correctness check passing.

This workspace starts from scratch: it builds the routed model, fits the initial bypasses, trains a router to produce plans, then trains the bypasses end to end. A later full run swaps in a much larger dataset produced separately; nothing else should need to change.

## Background

Per-token routing lets each token run or approximate each middle block of a frozen model; a small learned bypass stands in for any block the token doesn't run.

**Model structure (Qwen3.5-0.8B):**

- 24 layers in 6 **blocks**, each \[Gated DeltaNet, Gated DeltaNet, Gated DeltaNet, gated full attention\]: B0 = layers 0–3, B1 = 4–7, B2 = 8–11, B3 = 12–15, B4 = 16–19, B5 = 20–23.
- Hidden size 1024. Full attention: 8 query heads, 2 KV heads, head dimension 256. DeltaNet: 16 heads, head dimension 128, with a recurrent state and a short-convolution state. Vocabulary 248,320, tied with the embeddings. About 83M parameters per block.

**Definitions:**

| Term | Meaning |
| --- | --- |
| Routable blocks | B1–B4. B0 and B5 always run. |
| Plan | Per token, keep or bypass for each routable block: 16 plans. |
| Cost | Blocks executed per token = 2 + number kept. Stock is 6. |
| Bypass | `h_out = h_in + f_j(h_in / rms(h_in))`, a small MLP predicting the block's residual update from the token's own state. |
| Shadow writes | A token that bypasses block j still writes block j's cache entries (attention K/V, DeltaNet state and convolution inputs), computed from its unchanged input `h_in`, so later tokens never find gaps. |
| Damage | KL(stock ‖ routed) over the full vocabulary, nats per token. |

**Prior results** from a separate codebase, for orientation only (don't expect exact reproduction):

| Condition, \~5 blocks per token | Damage |
| --- | --- |
| Plain skipping (identity) under per-token routing | 0.15–0.16 |
| Bypass MLP (hidden 1024) fitted by regression | 0.067–0.072 |
| Oracle bypass: exact block output per token, in-block writes still shadows | \~0.03 |

- With plain skipping, most damage lands on tokens that skipped nothing: skipping tokens' under-processed states corrupt the cache writes everyone reads, mostly DeltaNet states. Bypasses mainly reduce that.
- At \~4.6 blocks per token, fitted bypasses gave 0.15–0.17 damage.
- Rank-16 linear bypasses were worse than skipping; a full-rank linear bypass gave about 0.08.
- A router trained on labels from runs where every token uses the same plan underpredicts real mixed-plan damage by about 2×, but routes comparably to better-labeled routers.

## Setup

The pipeline has five stages, run in order on one RTX 5090 (32 GB) with the base model frozen throughout.

**Environment:** PyTorch, transformers with Qwen3.5 support, and the fast DeltaNet kernels (flash-linear-attention, causal-conv1d). Load the model text-only, bf16, multi-token-prediction head disabled. Wrap the Hugging Face layer modules with a custom forward loop rather than forking the model code; DeltaNet layers must take and return their state explicitly.

**Pilot data** (built here, since the large dataset comes from a separate task):

- About 300 of the model's own thinking-mode generations from diverse prompts (math, code, chat, knowledge), truncated to 2,048 tokens. Drop any sequence with verbatim looping in its first 2,048 tokens: flag it if at least 50% of the 20-token n-grams in its last 512 tokens occurred earlier.
- 50 WikiText-103 test passages of 1,024 tokens.
- Split by sequence, 70/15/15 train/validation/test, stratified by source, fixed seed. That's roughly 400k tokens.

**Stages:**

1. **Routed model.** Per-token routing with shadow writes and a bypass hook (see the next section), with the tests in the last section passing.
2. **Fit initial bypasses.** For each routable block, collect stock pairs on train: the block's input `h_in` and output `h_out` from a normal forward pass. Fit `f_j` to `Δ = h_out − h_in` from `ĥ = h_in / rms(h_in)` (no learned scale): `f(ĥ) = W₂·SiLU(W₁ĥ + b₁) + b₂`, hidden size 1024, `W₂` zero-initialized. MSE loss, AdamW (lr 1e-3, weight decay 0.01), batches of 4,096 tokens, early stopping on validation MSE (patience 5, max 200 epochs).
3. **Labels and router.** For each of the 16 plans, run every sequence with every token on that plan and the fitted bypasses in place, under real shadow-write execution, and record per-token damage. Train the router on these labels (below), then choose λ on validation so that average cost hits 5.0 and 4.5.
4. **End-to-end training** of the bypasses under the router's fixed plans (variants below).
5. **Evaluation** on test.

**Router:**

- **Input:** each token's hidden state after layer 3 (B0's output) from the stock pass. B0 always runs, so this input is the same under every plan, and plans don't shift as bypasses train.
- **Model:** Linear(1024, 256) → ReLU → Dropout(0.1) → Linear(256, 16), predicting standardized `log(KL + 1e-4)` for each plan. The all-keep plan has KL 0 by construction; include it as a constant rather than a learned output. Standardize inputs with train statistics; AdamW (lr 1e-3, weight decay 1e-4), batch 1,024, early stopping on validation loss.
- **Plan choice:** each token takes `argmin over plans of predicted_kl + λ·cost`, where `predicted_kl = max(exp(pred) − 1e-4, 0)`. Compute plans once per sequence and hold them fixed through training and evaluation.

## Training-time execution

One parallel, teacher-forced forward pass over each full sequence, with a per-token keep mask for each routable block, gives shadow writes for free:

```python
h = B0(h)
for j in (1, 2, 3, 4):
    keep = plan[:, j]                         # [T] bool
    h_in, h_cur = h, h
    for layer in block[j]:
        x = where(keep, h_cur, h_in)          # bypassing tokens feed their unchanged input
        y = layer(x)                          # full layer over ALL tokens
        h_cur = where(keep, y, h_cur)         # only kept tokens take the layer output
    h = where(keep, h_cur, h_in + f[j](h_in / rms(h_in)))
h = B5(h)
logits = lm_head(final_norm(h))
```

- Every layer processes every token. A bypassing token's attention K/V and DeltaNet contributions come from `h_in`: those are its shadow writes. Its own output from those layers is discarded.
- Layer pre-norms, rotary positions, the short convolution, and the DeltaNet recurrence all operate on `x` exactly as in the stock forward pass. No cache during training.
- **Gradients into `f_j`** flow through the bypassing token's own later computation and through later tokens' reads of its writes in later blocks. Its in-block shadow writes come from `h_in`, so `f_j` gets no gradient through them.
- **Frozen layers** need activation gradients only, no weight gradients. Checkpoint activations per block.
- **Teacher:** the same model with every block kept and adapters disabled, run without gradients. No second copy of the weights.
- **Loss:** mean over tokens of KL(stock ‖ routed). Compute log-softmax in fp32 over chunks of 512 positions; never materialize full-vocabulary logits for a whole sequence.
- **Optional local loss** (weight μ = 0.1): for bypassing tokens, MSE between `f_j`'s output and the catch-up target, the stock block output minus the token's actual input.

**Shadow-write adapters** (variant B only): LoRA (rank 32, up-projection zero-initialized) on the write projections of layers 2–4 of each routable block, applied only to tokens bypassing that block. Layer 1 needs none, because its shadow write is already exact.

- Attention layer: the K and V projections.
- DeltaNet layers: the q/k/v input projection and the write-strength (β) and decay projections.

A bypassing token's own output from those layers is discarded, so these adapters change only what other tokens read. Their gradient comes entirely through later tokens.

## Variants and training defaults

Train four variants on the router's plans at target cost 5.0, and variant A again at 4.5.

| Variant | Trainable | Initialization | Question |
| --- | --- | --- | --- |
| A | Bypass MLPs, hidden 1024 | Regression fit | Does end-to-end training beat the fit? |
| B | A + shadow-write adapters | Regression fit; adapters at zero | Can in-block writes be improved too? |
| C | Bypass MLPs, hidden 4096 | Regression fit at that size | Is bypass capacity the limit? |
| D | Bypass MLPs, hidden 1024 | Output layer zero (no regression fit) | How much does the fit initialization matter? |

**Defaults:**

- AdamW, learning rate 3e-4, weight decay 0.01, 2% warmup, cosine decay. fp32 master weights for trainable parameters, bf16 activations.
- About 64k tokens per step (e.g. 32 sequences of 2,048 tokens, with gradient accumulation as needed).
- Up to 1,500 steps, evaluating validation KL every 50 steps and keeping the best checkpoint. The pilot set is small, so expect overfitting and rely on early stopping; that's acceptable for a pilot.
- Rough cost: about 5 GFLOP per token (routed forward and backward through frozen layers, plus the teacher pass), so a few seconds per step and a few hours per variant.

**Log every run:** training KL; validation KL for all tokens, kept-everywhere tokens, and bypassing tokens; bypass output norms per block; learning rate.

## Evaluation

The headline is the paired difference in test damage between trained and regression-fitted bypasses, at target costs 5.0 and 4.5.

**Conditions** (same test sequences, same router plans, achieved cost reported):

| Condition | What a bypassing token does |
| --- | --- |
| SKIP | Identity (`f = 0`), shadow writes |
| FIT | Regression-fitted bypass |
| A, B, C, D | Trained variants |

**Metrics:**

- Mean damage (KL), split into all tokens, kept-everywhere tokens (cost-6 plan), and bypassing tokens.
- NLL of the actual next token minus stock NLL; top-1 agreement with stock.
- Breakdowns by source (self-generated vs. WikiText) and, for self-generated text, by region (prompt, thinking, answer).
- 95% sequence-level bootstrap intervals, resampling sequences within source; paired bootstrap for differences.

**Diagnostics:**

- Training, validation, and test damage over steps, to show overfitting.
- Each block's bypass fit quality (R² of `Δ` on stock pairs) before and after training: does end-to-end training trade raw fit for usefulness?
- Cost: bypass parameters and FLOPs per token as a fraction of one block.

## Checks, deliverables, and the full run

No result counts until these checks pass; assert them in code and report them.

| Check | Must hold |
| --- | --- |
| Native equivalence | The all-keep plan, adapters disabled, reproduces stock logits exactly (damage 0) |
| Zero bypass | With `f = 0`, the routed model equals plain skipping exactly |
| Uniform plans | With every token on the same plan and `f = 0`, results equal a simple implementation that removes those blocks for the whole sequence |
| Layer-1 shadow writes | In a bypassed block, the first layer's stored entries equal a real visit's for the same input |
| Teacher | The teacher pass equals stock bitwise |
| Gradient gating | With the all-keep plan, every bypass and adapter gradient is exactly zero |
| Step 0 | The first training step's loss matches the FIT condition's damage on the same batch |

Optional, but needed later for generation-based evaluation: token-by-token decoding with an explicit per-block cache, where each layer exposes `run` (full computation) and `write_only` (shadow write). Check it against the parallel path on 3 sequences, to within bf16 and chunked-vs-recurrent kernel noise.

**Deliverables:**

1. Code: routed model, bypass fitting, labeling, router, trainer, evaluation.
2. Checkpoints holding only trained components (bypasses, adapters, router), plus router λ values.
3. Per-token results for every condition.
4. A markdown report: setup, sanity checks, headline table, training curves, diagnostics, deviations from this spec, and limitations.

**The full run later:** swap in the large self-generated dataset (about 300M tokens, from the separate data task), rebuild router labels on more data, train on 50–100M tokens, and evaluate on a larger test split. Everything else stays the same.

