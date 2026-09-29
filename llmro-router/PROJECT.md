# Per-Token Adaptive Depth for a Pretrained Hybrid LLM

Retrofitting learned block bypasses and per-token routing onto Qwen3.5: findings to date and research plan

---

## 1. Summary

### The idea

Not every token needs the same amount of computation. Predicting a closing bracket or the next piece of a
multi-token word is nearly free; the token that carries the answer to a math problem is not. Standard
transformers still spend the same depth on both.

This project adds **per-token adaptive depth** to a pretrained language model without retraining it. The middle
of the network is split into blocks. For every token, a small learned **router** decides, per block, whether the
token runs the block normally or takes a cheap shortcut. The base model's weights stay frozen; only small added
components are trained.

### What the experiments so far established

Seven zero-shot experiments on Qwen3.5-0.8B (no model training) shaped the design:

1. **The premise holds.** On the model's own generated text, about half of all tokens tolerate less computation
   with essentially no change to their prediction. Tolerance tracks how confident the model is.
2. **A router can see it early.** The hidden state after the first block predicts which tokens can take
   shortcuts (AUROC 0.93 with a small MLP), well beyond token identity alone.
3. **The main obstacle is contamination.** If a token simply skips a block, its hidden state enters every later
   layer under-processed, and its cache writes there are off. Every other token reads those writes. Most routing
   damage lands on tokens that skipped nothing, through the shared caches, overwhelmingly the recurrent state of
   the linear-attention (DeltaNet) layers.
4. **Fixing the cause works better than fixing the symptom.** Linear corrections to the corrupted cache writes
   recovered only about a quarter of the damage at a realistic size. Replacing "skip" with a **bypass**, a
   ~2M-parameter MLP that predicts what the block would have done to the token's hidden state, halves the damage
   at about 2% of a block's compute.

### Current design in one paragraph

Qwen3.5-0.8B stays frozen. Its four middle blocks are routable. Per token, a router reading the first block's
output chooses, for each routable block, **keep** (run it) or **bypass** (run a small learned approximation of
it). Tokens that bypass a block still write that block's cache entries from their unchanged hidden state, so the
cache is always dense. The bypasses, a set of small cache-write adapters, and eventually the router are trained
end to end to match the stock model's predictions under real per-token execution.

### Goals

| Track | Goal | Compute budget |
|---|---|---|
| **E: efficiency** | Same quality, fewer blocks per token | Below the native 6 blocks per token |
| **Q: quality** | Better answers, same compute | 6 blocks per token on average, with repeats paid for by bypasses |

Track E is the current focus. Track Q depends on an unresolved question: whether running a block twice can help
at all (§4.6).

### Related work

- **Per-input layer programs on frozen models:** CoLa (Li et al., 2025) showed that skipping and repeating
  pretrained layers per input can shorten computation and fix wrong answers. DR.LLM (Heakl et al., 2025) and POLAR
  (ICML 2026) train routers over such programs with frozen layers. POLAR found that predicting the whole program
  up front beats deciding layer by layer.
- **Adaptive or recurrent depth trained from scratch:** Mixture-of-Depths (Raposo et al., 2024),
  Mixture-of-Recursions (Bae et al., 2025), Ouro (Zhu et al., 2025), Huginn (Geiping et al., 2025).
- **Early exit:** CALM (Schuster et al., 2022), LayerSkip (Elhoushi et al., 2024).
- **Converting pretrained models to recursive ones:** Relaxed Recursive Transformers (Bae et al., 2024), which
  used 15–60B tokens of full-parameter uptraining plus layer-wise LoRA.
- **Cache problems under adaptive depth:** CHASE trains looped models to tolerate cache holes left by early exit;
  MELT (Vendrell et al., 2026) trains a shared, gated cache across loops.
- **Layer roles:** middle layers are robust to deletion and reordering while the first and last layers are fragile
  (Lad, Gurnee & Tegmark, 2024; Gromov et al., 2024).

---

## 2. Findings to date

All experiments use Qwen3.5-0.8B in bf16 with no weights trained, teacher-forced evaluation, and **damage**
measured as KL(stock ‖ modified) over the full vocabulary, in nats per token. Unless stated, test numbers are on a
held-out split of 38 sequences (59,740 positions): 22 of the model's own thinking-mode generations ("selfgen"),
8 math generations, and 8 WikiText passages. Intervals, where given, are 95% sequence-level bootstraps.

### 2.1 Headline progression

Damage at about 5 blocks per token (stock uses 6), under two routers. "Old" is a probe trained on whole-sequence
labels (§2.3); "clean" is a probe trained on clean labels (§2.6).

| Condition | Old router | Clean router |
|---|---|---|
| Simulated (per-token damage looked up from whole-sequence runs) | 0.083 | — |
| **Real per-token routing, plain skip** | **0.161** | **0.153** |
| Plain skip + best realistic linear write correction (rank 32, DeltaNet) | 0.139 | — |
| **Real per-token routing, bypass (MLP, hidden 1024)** | **0.072** | **0.067** |
| Oracle bypass (exact per-token block output; in-block writes still shadows) | 0.030 | 0.034 |
| Plain skip, every cache write replaced with the stock model's | 0.047 | 0.011 |
| Bypass, every cache write replaced with the stock model's | 0.016 | 0.007 |
| Static reference: skip B2 for every token (cost 5.0) | 0.280 | 0.280 |

The last three rows use the stock model's cache writes, which aren't available at inference; they are ceilings,
not achievable results.

### 2.2 Block tolerance (experiment 001)

Each block skipped or repeated for every token (WikiText):

- **B0 and B5 are untouchable.** Skipping either sends perplexity from 23.4 to roughly 26,000–55,000. Even
  repeating them costs about 2 nats of KL.
- **B1–B4 tolerate skipping:** KL 0.22–1.07, perplexity 28–72. B2 is cheapest, B4 most expensive.
- **Repeats barely hurt** (KL 0.1–0.2), but no repeat beats stock on average.

### 2.3 Per-token tolerance (experiment 002)

All 81 skip/keep/repeat plans over B1–B4, each applied to a whole sequence, with per-token damage recorded:

- At a tolerance of 0.05 nats, **47% of selfgen tokens** tolerate a cheaper plan; the average cheapest sufficient
  plan costs 5.01 blocks. Human-written text is much less compressible (WikiText: 80% of tokens need all 6 blocks).
- **Cheapness tracks confidence:** tokens with stock next-token entropy below 0.01 nats average about 3.5 blocks;
  above 1 nat, nearly 6.
- Cheap token types: punctuation, word continuations, digits. Content-word starts and function words are expensive.
- **Repeats add nothing to efficiency:** restricting to skip/keep gives average cost 5.02 vs. 5.01 with all plans.
- Block decisions are mildly dependent (total correlation 0.18 nats against a joint entropy of about 2 nats),
  strongest between adjacent blocks.
- **Generation problem:** at an 8,192-token budget, 45 of 50 math generations and 78 of 150 general generations
  never finished and ended in verbatim loops. Only 5 of 50 math problems were solved.

These whole-sequence damage tables turned out to be poor labels for real per-token routing (§2.4–§2.6).

### 2.4 Can a router see tolerance early? (experiment 003)

Probes on hidden states, predicting per-token tolerance from experiment 002:

- From the **output of B0** (layer 3), a small MLP reaches AUROC 0.93 for "some cheaper plan is free"; token
  identity alone reaches 0.78–0.80.
- A simulated router built from a probe predicting damage per plan closes 65% of the gap between the best static
  plan and a perfect per-token oracle at 5 blocks per token.
- Reading one block later (layer 7) helps only slightly (68%) and would forfeit routing B1. The router stays after B0.
- Reading the layer-3 state through the output layer ("logit lens") is useless (AUROC 0.62) and costs 1.6× a block.

### 2.5 Real per-token routing and contamination (experiments 004, 005)

**Real routing (004).** Per-token plans, with **shadow writes**: a token that skips a block still writes that
block's cache entries from its unchanged hidden state, so later tokens never find gaps.

- Real damage is about **twice the simulated value** (0.161 vs. 0.083 at 5 blocks), though still well below
  skipping B2 for everyone (0.280).
- **Tokens that skipped nothing are hurt** (0.153 vs. 0 in simulation). Tokens that skipped do about as well as
  simulated.
- Perfect per-token choices from the whole-sequence tables fall from 0.008 simulated to 0.147 real.

**Which writes carry the damage (005).** Selectively replacing cache writes with the stock model's:

- Replacing only **deviated tokens'** writes (tokens that skipped something at or before that layer) removes about
  **70%** of the damage (0.161 → 0.047). Tokens that kept every block then have exactly zero damage.
- **Downstream writes** (in blocks run after a skip) matter more than the skipped block's own shadow writes.
- **DeltaNet writes dominate:** fixing only those removes 0.089, versus 0.027 for attention writes. DeltaNet's
  recurrent state blends every token's writes into one running summary, so a bad write can't be ignored later.
- Once deviated tokens' writes are native, everyone else's are automatically native too. The fix only needs to
  target deviated tokens.
- In a clean context, skipping is far cheaper than whole-sequence measurements suggested: skipping B2 for every
  token costs about 0.04 instead of 0.25 on a sample sequence.

### 2.6 Linear write corrections and clean labels (experiment 006)

**Write corrections.** Closed-form, low-rank linear maps from a deviated token's actual input to the stock model's
write, applied where a LoRA on the write projections would act:

- At a realistic size (rank 32 per layer and role, 6.3M parameters), they close **25%** of the DeltaNet-fixable
  damage at 5 blocks and 16% at 4.5.
- Even a full-rank correction per skip pattern (646M parameters, nearly the decoder's size) closes only 46%; fit
  in-sample on the test data itself, 65%. **Native writes aren't a linear function of the drifted hidden state.**
- Corrections must act wherever a LoRA would, including the token's own computation. Correcting only what others
  read makes skipping tokens worse.

**Clean labels.** With every cache write native, a token's damage depends only on its own plan:

- Skipping B1 or B2 is nearly free (0.076 and 0.056); skipping B3 or B4 is not (0.39–0.48).
- The average cheapest sufficient plan drops to 4.16 blocks on selfgen and 3.95 on math.
- A probe retrained on these labels routes far better in a clean context: 0.011 damage at 5.1 blocks.
  Simulation and real execution match exactly in that setting.

### 2.7 Block bypasses (experiment 007)

Replacing "skip" (identity) with a small per-token map predicting the block's residual update,
`h_out ≈ h_in + f(h_in)`, fitted on stock pairs:

- **Best bypass:** an MLP with hidden size 1024, about 2.1M parameters per block (2.3% of a block's matmul FLOPs).
  It reduces damage at ~5 blocks per token from 0.161 to **0.072** (old router) and from 0.153 to **0.067** (clean
  router); at ~4.6 blocks, from 0.32–0.36 to 0.15–0.17. The NLL gap to stock halves (+0.18 → +0.09) and top-1
  agreement rises from 0.87 to 0.91.
- **Fit quality predicts damage:** across families, damage falls steadily as the bypass explains more of the block's
  update. The best bypass explains 61%. Low-rank linear bypasses (rank 16) are worse than skipping.
- **Headroom remains:** an oracle bypass (exact block output per token) reaches 0.030–0.034. Refitting linear
  bypasses on the inputs tokens actually carry during routing helps under the clean router (0.081 → 0.073).
- **Mostly a contamination fix:** nearly all of the bypass's benefit is reduced harm to other tokens. About 0.06 of
  contamination remains at 5 blocks. About half of it is the skipped block's own shadow writes, which even a
  perfect bypass leaves in place; the rest is bypass error carried downstream.
- Chaining four per-layer linear maps (to improve in-block writes) was worse than one block-level map.
- Restricting routing to B1 and B2 is worse than routing over all four blocks at every matched cost.
- Per-token routing adds value on top of the bypass: bypassing B2 for every token costs 0.134, versus 0.067 for
  per-token routing at the same compute.
- A router retrained on bypass labels from whole-sequence runs was not better (0.082); those labels again
  predicted about half the real mixed-plan damage.

### 2.8 What the findings imply

- **"Skip" should mean "run a learned bypass."**
- **Labels from whole-sequence runs are unreliable** for real per-token routing: they ignore how plans interact
  through the caches. The router ultimately has to be trained under real mixed execution.
- **Remaining damage** comes from in-block shadow writes and bypass error. Both are targets for training.
- **Efficiency upside at 0.8B is modest.** The output layer (248k-word vocabulary, tied with the embeddings) reads
  about 0.5 GB per token, versus about 1 GB for all six blocks. Saving one block per token cuts weight reads by
  roughly a tenth before bypass overhead. The efficiency case rests mainly on larger models, where the output
  layer is a smaller share.

---

## 3. Design

### 3.1 Base model

| Property | Qwen3.5-0.8B |
|---|---|
| Layers | 24, as 6 × [Gated DeltaNet ×3 → gated full attention] |
| Hidden size | 1024 |
| Full attention | 8 query heads, 2 KV heads, head dimension 256, sigmoid output gate |
| Gated DeltaNet | 16 heads, head dimension 128; fixed-size recurrent state plus short-convolution state |
| FFN | SwiGLU, intermediate size 3584 |
| Vocabulary | 248,320, embeddings tied with the output layer |
| Parameters | ~0.8B: ~0.25B embeddings, ~83M per 4-layer block |

Loaded text-only (no vision encoder), multi-token-prediction head disabled, weights frozen. Qwen3.5-2B has the same
24-layer layout and is the second rung of the size ladder.

### 3.2 Blocks and plans

- Six blocks B0–B5, each one [DeltaNet, DeltaNet, DeltaNet, attention] group: B0 = layers 0–3, B1 = 4–7,
  B2 = 8–11, B3 = 12–15, B4 = 16–19, B5 = 20–23.
- **B0 and B5 always run.** B1–B4 are routable.
- **Track E actions:** keep or bypass, giving 16 plans per token. Cost = blocks executed = 2 + number kept, from 2
  to 6. Bypass cost is reported separately.
- **Track Q actions** (if pursued): keep, bypass, or repeat (run twice, at most once per block).

### 3.3 Bypass

For routable block j with input `h_in`:

```
h_out ≈ h_in + f_j( h_in / rms(h_in) )
f_j(x) = W₂ · SiLU(W₁ x + b₁) + b₂        # hidden size 1024, W₂ zero-initialized
```

- About 2.1M parameters per block; about 2.3% of a block's matmul FLOPs.
- Initialized by regression on the block's stock residual update, then trained end to end (§4.3).
- **Limitation:** a real block mixes information across tokens; a bypass sees only the token's own state. It
  captures the part of the block's effect that is predictable per token.

### 3.4 Cache

- One slot per block: the attention layer's KV cache plus three DeltaNet recurrent and convolution states. Track E
  uses 6 slots. With repeats, a second slot per routable block is added (10 in total).
- **Every token writes every slot.** A kept block writes normally. A bypassed block gets **shadow writes**: the
  token computes that block's cache entries from its unchanged input `h_in`, using only the cheap projections
  (K/V for attention; q, k, v, convolution, write strength β, decay, and state update for DeltaNet).
- After a bypass, the token continues from `h_in + f(h_in)`, and its later writes come from that state.
- Shadow writes are exact at each block's first layer and approximate at layers 2–4.
- **Memory:** 12 KB of attention KV per token for the stock layout, plus ~19 MB of fixed DeltaNet state per sequence.

### 3.5 Router

- **Input:** each token's hidden state after B0.
- **Current form:** a small MLP (hidden size 256) predicting damage, `log(KL + 1e-4)`, for each of the 16 plans.
  Each token takes `argmin(predicted damage + λ · cost)`. λ sets the compute budget and can be changed at inference
  without retraining.
- **Initialization:** trained on clean labels (each plan's damage with all cache writes native), which gave the
  best real results so far.
- **Final training under real execution** (open decision, §4.4):
  - (a) **Joint straight-through Gumbel training** with per-block keep/bypass heads, made sequential so each head
    sees earlier blocks' choices. Because the bypass branch is cheap, computing both branches per block during
    training costs little more than the keep path.
  - (b) **Iterative relabeling:** measure damage under the current policy's mixed execution and refit the
    damage-predicting router.
- Compute the router in fp32, or add a small decision margin: in token-by-token decoding, bf16 noise flipped about
  0.5% of decisions.

### 3.6 Adapters

All zero-initialized, all inactive on native visits, so the all-keep plan is exactly the stock model.

| Component | Where | Purpose | Status |
|---|---|---|---|
| **Bypass MLP** | Replaces a bypassed block's residual update | Keeps the token's own state, and therefore its later writes, close to native | Core |
| **Shadow-write adapter** | Write projections of layers 2–4 of a bypassed block, rank ~32 | Reduces the in-block shadow-write contamination that remains even with a perfect bypass | To test (experiment 008) |
| **Downstream write adapter** | Write projections in blocks run after a bypass | Residual downstream contamination | Low priority: linear versions were weak |

**Local targets are free.** The teacher pass (the same frozen model, all blocks kept, adapters off) computes every
token's stock hidden states and cache writes, which gives direct regression targets for bypasses and write adapters
in addition to the end-to-end loss.

### 3.7 Training objective

```
L = KL(stock ‖ routed)                        # every token, under real per-token execution with shadow writes
  + λ_budget · (E[cost] − budget)             # when the router is trained; λ_budget by dual ascent
  + μ · L_local                               # optional regression of bypass outputs and writes to stock targets
  [+ CE on verified reasoning traces]         # Track Q only
```

The teacher needs no second copy of the weights: it is the same model with adapters disabled and every block kept.

### 3.8 Excluded, and why

| Idea | Reason |
|---|---|
| Plain skipping (identity) | Contamination; the bypass halves damage at ~2% of a block's compute |
| Linear write correction as the main fix | Native writes aren't linearly recoverable (25% at rank 32, <50% at any size) |
| Sharing one cache slot between a block and its repeat | Saves only 8 KB per token here, breaks native equivalence, and pretrained models collapse under it unless trained for it |
| Routing only B1 and B2 | Worse than routing all four blocks at every matched cost |
| Repeats for Track E | Add nothing to efficiency |
| Logit lens as router input | Uninformative and more expensive than a block |
| Router labels from whole-sequence runs | Underpredict real mixed-plan damage by about 2× |

---

## 4. Plan

### 4.1 Status

Frozen-model investigation is complete (experiments 001–007). The next phase is training small components with
the base model frozen.

### 4.2 Unblock training data (priority)

End-to-end training needs far more text than the ~400k tokens used so far, and thinking-mode generation at 0.8B is
unstable.

1. **Fix generation:**
   - Try the model card's recommended thinking-mode settings with a presence or repetition penalty.
   - Try shorter budgets, with loop detection that truncates degenerate tails.
   - Compare accuracy with thinking disabled on the same math problems.
   - Check whether Qwen3.5-2B finishes reliably.
2. **Broad self-generated set:** ~300M tokens of the base model's own responses to diverse prompts (chat, code,
   math, general knowledge), filtered for loops. The native model provides dense per-token supervision, so no
   labels are needed.
3. **Verified reasoning set (Track Q only):** Nemotron-Math-v2 problems. Sample 8 times each, keep problems solved
   1–6 times, keep correct traces. Nemotron-Math-v2 (December 2025) predates Qwen3.5's small models (March 2026),
   so it is used for training only.

### 4.3 Experiment 008: end-to-end bypass training

Train bypasses (and optionally shadow-write adapters) on KL to the stock model under real per-token routing with
shadow writes. The router is held fixed: the clean-label router's plans, at target costs 5.0 and 4.5.

| Variant | Trainable components |
|---|---|
| A | Bypass MLPs (hidden 1024), initialized from the regression fit |
| B | A + shadow-write adapters (rank 32, layers 2–4 of each bypassed block) |
| C | A with larger bypasses (hidden 4096) |
| D | A initialized from zero (checks how much the regression initialization matters) |

- **Targets:** damage at 5 blocks from 0.067 toward the oracle-bypass level (~0.03); variant B is the only one that
  can go below it.
- **Defaults:**
  - AdamW, learning rate 3e-4 with cosine decay
  - Sequences of 4,096 tokens, ~64k tokens per step
  - 50–100M training tokens
  - Gradient checkpointing per block
  - Optional local regression loss, weight 0.1
- **Compute:** roughly 5 GFLOP per token: the routed forward and backward through the frozen layers, plus a
  no-gradient teacher pass. That's a few hours per variant on one RTX 5090.
- **Evaluation:** same held-out split and metrics as experiment 007, plus damage split into kept-everywhere and
  bypassing tokens, and an own-damage vs. contamination split via native-write substitution.

### 4.4 Router training under real execution

After 008, train the router jointly with the bypasses using one of the two approaches in §3.5, under a compute
budget enforced by a Lagrange multiplier. The budget schedule starts at 6 blocks and lowers linearly to the target
over the first half of training. Compare against the fixed clean-label router.

### 4.5 Efficiency evaluation

- Budget sweep at 5.5, 5.0, 4.5, and 4.0 blocks per token.
- Three efficiency metrics:
  - blocks per token
  - weight bytes read per token, counting bypasses and shadow-write projections
  - measured batch-1 decoding speed, where skipped blocks simply don't run apart from shadow writes and the bypass
- Profile where decoding time actually goes at this size: per-layer kernel-launch overhead may matter more than
  memory reads.

### 4.6 Track Q: can repeats help?

In experiment 002, some plan beat stock by more than 0.1 nats for 46% of tokens, versus 9% for one random plan. But
picking the best of 80 perturbations always looks good, and every plan was worse than stock on average. The fair
test is best-of-80 random hidden-state perturbations with the same KL as the repeat plans. If noise wins as often
as repeats, Track Q has no zero-shot support. If repeats clearly win, add repeat slots and train at budget 6 with the
verified-trace loss.

### 4.7 Evaluation

- **Clean evaluation set:**
  - Tier A: 1,000–2,000 procedurally generated problems (BeyondBench-style algorithmic tasks and GSM-Symbolic-style
    templates with fresh values), calibrated at the generator level so the base model with thinking enabled scores
    20–60%, then frozen with a fixed seed.
  - Tier B: 200–400 human-written problems published after March 2026.
  - Tier C: MathArena competitions after the model's release, for larger models.
- **General ability:** MMLU-Pro, IFEval, perplexity on held-out text, with the router active.
- **Generation, not just teacher forcing:** measure accuracy and drift in free generation, where routing errors can
  compound.
- **Reporting:** blocks per token and bytes read per token next to every accuracy number. All results are reported;
  no success thresholds are pre-registered.

### 4.8 Scaling

Rerun the winning configuration on Qwen3.5-2B (same layout). Then choose a third model: 4B or 9B on one RTX 5090
(9B has 32 layers, 8 blocks, 6 routable), or 27B on rented hardware.

### 4.9 Risks and open questions

| Risk | Signal or mitigation |
|---|---|
| Training doesn't move bypasses much beyond the regression fit | Variant D vs. A; the local loss; larger bypasses |
| In-block shadow writes remain the floor | Variant B; oracle-bypass comparison |
| Router training under mixed execution is unstable | Start from the clean-label router; budget schedule; compare (a) and (b) |
| Generation data stays degenerate at 0.8B | Loop filtering; thinking-off data; generate data with 2B |
| Teacher-forced gains don't carry over to free generation | Generation-based evaluation (§4.7) |
| Efficiency gains are small at 0.8B | Expected from the output-layer share; the efficiency case rests on 2B+ |
| Probes and bypasses fit on ~250 sequences don't generalize | Refit on the broad self-generated set |

---

## 5. Code

### 5.1 Current repository

The investigation code is organized as a library plus one directory per experiment:

```
qwen35-investigate/
├── pyproject.toml                  # uv-managed
├── src/qwen35_investigate/
│   ├── pertoken.py                 # per-token routing: parallel teacher-forced path (routed_forward, routed_blocks),
│   │                               #   token-by-token path with a cache (routed_decode), DecoderLayer.write_only
│   │                               #   (shadow writes), perfect-cache mode, ProbeRouter
│   ├── substitution.py             # replacing selected cache writes with the stock model's: routed_blocks_sub,
│   │                               #   substituted_decode, NativeWrites, deviation_masks, actual_writes
│   ├── delta_rule.py               # chunked DeltaNet readouts of the state before each position's update
│   ├── write_correction.py         # closed-form low-rank write corrections
│   └── bypass.py                   # bypass families and fitting
├── experiments/
│   └── NNN_<name>/                 # 001–007, e.g. 002_per_token_routing, 004_pertoken_shadow_writes,
│       ├── README.md               #   005_write_substitution, 006_write_correction, 007_block_bypass
│       ├── run.py                  # staged pipeline (--stage)
│       ├── report.py
│       └── results/                # reports and plots tracked; large intermediates git-ignored
└── data/activations/               # extracted hidden states (git-ignored)
```

Conventions:

- Every experiment has a README with question, setup, results, deviations, and limitations.
- Runs are reproducible from their scripts.
- Every result is backed by sanity checks that must reproduce earlier experiments exactly.

### 5.2 Additions for the training phase

```
src/qwen35_investigate/
├── model/
│   ├── routed.py                   # one routed model, three modes: fixed plan per sequence, hard per-token,
│   │                               #   straight-through Gumbel (all branches computed)
│   └── cache.py                    # per-block slots; separate repeat slots if Track Q proceeds
├── router/
│   ├── damage_router.py            # predicts damage per plan; argmin + λ·cost
│   └── gumbel_router.py            # sequential per-block heads for joint training
├── adapters/
│   ├── bypass.py                   # trainable bypass MLPs (initialized from fitted ones)
│   ├── shadow.py                   # shadow-write LoRA on write projections of layers 2–4
│   └── control.py                  # adapters_disabled(): the teacher is the same object
├── data/
│   ├── selfgen.py                  # broad self-generated set via vLLM, with loop filtering
│   ├── verified.py                 # rejection-sampled correct traces (Track Q)
│   ├── procedural/                 # clean evaluation generators (Tier A)
│   ├── packing.py
│   └── manifest.py                 # shards with generator version, model, seed
├── train/
│   ├── losses.py                   # KL to stock, budget (Lagrangian), local regression, CE
│   ├── schedules.py                # Gumbel temperature, budget, entropy bonus
│   └── trainer.py                  # separate optimizer groups for router and adapters
└── eval/
    ├── harness.py                  # clean tiers; MMLU-Pro and IFEval via lm-evaluation-harness; perplexity
    ├── generate.py                 # free generation with per-token routing
    ├── efficiency.py               # blocks/token, bytes/token, batch-1 decoding speed
    └── diagnostics.py              # native-write substitution, oracle bypass, routing patterns
```

Design rules carried forward:

1. **Wrap the Hugging Face implementation; don't fork it.** A custom forward loop decides which layer runs and which
   cache slot it uses. DeltaNet layers take and return their state explicitly and call the same fast kernels.
2. **Every layer exposes `run` and `write_only`,** so shadow writes and routing are built from two calls.
3. **The teacher is the same object** with adapters disabled and all blocks kept: no second weight copy, no drift.
4. **Evaluate correctness and speed separately.** Batched evaluation computes every branch and selects; speed is
   measured with batch-1 decoding and true skipping. vLLM is used only to generate data from the base model.
5. **Checkpoints hold only trained components** (router, bypasses, adapters): tens of MB.

### 5.3 Tests that gate the training code

Most of these already exist as experiment sanity checks and should become unit tests:

| Test | Asserts |
|---|---|
| Native equivalence | The all-keep plan with adapters disabled reproduces stock logits exactly |
| Uniform-plan equivalence | With one plan for all tokens, per-token routing reproduces whole-sequence routing exactly |
| Parallel vs. token-by-token | Teacher-forced and token-by-token paths agree to within bf16 and chunked-vs-recurrent kernel noise, for fixed plans |
| Shadow writes | `write_only` stores exactly what a full `run` would for the same input; layer-1 shadow writes equal real writes |
| Zero bypass | A bypass with `f = 0` reproduces plain skipping exactly |
| Substitution | With all writes native, kept-everywhere tokens have exactly zero damage |
| Adapter gating | Every adapter contributes exactly zero on native visits |

### 5.4 Tooling

- **Environment:** uv, PyTorch, transformers with Qwen3.5 support, flash-linear-attention and causal-conv1d
- **Data generation:** vLLM (base model only)
- **Evaluation:** lm-evaluation-harness with a custom model wrapper
- **Testing:** pytest
- **Tracking:** per-run `metrics.jsonl` plus Weights & Biases or a local logger
- **Storage:** large intermediates (extracted activations, per-token results) stay git-ignored; reports and plots
  are tracked

---

## Glossary

| Term | Meaning |
|---|---|
| **Block** | Four consecutive layers: three Gated DeltaNet layers, then one full-attention layer. Qwen3.5-0.8B has six (B0–B5). |
| **Routable block** | B1–B4, which each token can keep or bypass. B0 and B5 always run. |
| **Plan** | One token's actions for B1–B4: 16 keep/bypass plans (81 with repeats). |
| **Cost** | Blocks executed per token; stock is 6. |
| **Damage** | KL(stock ‖ modified) over the full vocabulary, in nats per token. |
| **Skip** | Bypassing a block with the identity: the hidden state passes through unchanged. |
| **Bypass** | A small learned MLP predicting a block's residual update from the token's own hidden state. |
| **Shadow write** | The cache entries a token writes for a block it didn't run, computed from its unchanged input. |
| **Deviated token** | At a given layer, a token that skipped or bypassed that block or an earlier one. |
| **Native write** | The cache entry the stock model would have written at that position and layer. |
| **Contamination** | Damage to other tokens from reading deviated tokens' off-native cache writes. |
| **Whole-sequence labels** | Per-token damage measured with every token on the same plan. |
| **Clean labels** | Per-token damage measured with every cache write native, so it depends only on the token's own plan. |
| **Oracle bypass** | A bypass that reproduces each token's exact block output in context; the ceiling for per-token approximation while in-block writes remain shadows. |
| **Teacher** | The same frozen model with adapters off and all blocks kept, used as the training target. |
| **Straight-through Gumbel** | Training discrete choices with hard random samples in the forward pass and gradients of the smooth version in the backward pass. |
| **Track E / Track Q** | The efficiency goal and the quality goal. |

## References

- Bae et al., 2024. *Relaxed Recursive Transformers: Effective Parameter Sharing with Layer-wise LoRA.* arXiv:2410.20672.
- Bae et al., 2025. *Mixture-of-Recursions.*
- Elhoushi et al., 2024. *LayerSkip.*
- Geiping et al., 2025. *Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach* (Huginn). arXiv:2502.05171.
- Gromov et al., 2024. *The Unreasonable Ineffectiveness of the Deeper Layers.*
- Heakl et al., 2025. *DR.LLM.*
- Lad, Gurnee & Tegmark, 2024. *The Remarkable Robustness of LLMs: Stages of Inference?*
- Li et al., 2025. *CoLa.* (Full title to be filled in.)
- POLAR, ICML 2026. (Full citation to be filled in.)
- Raposo et al., 2024. *Mixture-of-Depths.*
- Schuster et al., 2022. *Confident Adaptive Language Modeling* (CALM).
- Vendrell et al., 2026. *Memory-Efficient Looped Transformer: Decoupling Compute from Memory in Looped Language Models* (MELT). arXiv:2605.07721.
- Zhu et al., 2025. *Scaling Latent Reasoning via Looped Language Models* (Ouro). arXiv:2510.25741.
- *CHASE: Cache-Hole-Adapted Skip Exit for Looped State-Space Language Models.* arXiv:2607.10110.
- *Depth-adaptive Inference of Looped Language Models via Continuous Depth Batching.* arXiv:2608.09444.
- *BeyondBench: Contamination-Resistant Evaluation of Reasoning in Language Models.* arXiv:2509.24210.
- *MathArena: Evaluating LLMs on Uncontaminated Math Competitions.* arXiv:2505.23281.
- *LiveBench: A Challenging, Contamination-Free LLM Benchmark.* arXiv:2406.19314.
- Mirzadeh et al., 2024. *GSM-Symbolic.*
- NVIDIA. *Nemotron-Math-v2* (Hugging Face dataset).
- Qwen Team. *Qwen3.5* model cards (Hugging Face).
