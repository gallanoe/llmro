# Project Spec: 360M LLM Pretraining Run

**Goal:** train a GPT-2-medium-class language model end to end — data → tokenizer → pretraining → anneal → SFT → eval — on a single A100 in under 72 hours, in JAX.

**Success criterion:** not a benchmark number. A model you can talk to, a pipeline you understand line by line, and a written record of what each decision cost you. Beating GPT-2 (2019) on CORE is table stakes at this budget, not an achievement.

---

## 1. Budget

| Item | Value |
|---|---|
| Hardware | 1× A100 80GB (SXM or PCIe) |
| Rate | ~$1.79/hr median, ~$1.09/hr at cheapest provider |
| Wall clock | ~51h at MFU 0.40; ~68h at MFU 0.30 |
| Compute cost | $92–122 on-demand, ~$56–75 at floor, ~half again on spot |
| Total FLOPs | 6ND × 1.08 ≈ 2.3e19 |

Book 4 days, not 3. The marginal hour is $1.79; a rushed run is worth far less than that.

**Why A100 over the cheaper 5090:** consumer Blackwell (sm_120) is the roughest edge of the jaxlib/cuDNN support matrix. Ampere is the most trodden CUDA target JAX has. You paid ~$45 to not spend day one on wheel archaeology.

---

## 2. Model

Target ~360M params. Keep GPT-2 medium's *shape*, replace its 2019 internals.

| Parameter | Value | Note |
|---|---|---|
| Layers | 24 | |
| d_model | 1024 | |
| Heads | 16 (head_dim 64) | MHA; GQA optional but pointless at this size |
| MLP | SwiGLU, hidden 3072 | 3 matrices, not 2 |
| Positional | RoPE | not learned embeddings |
| Norm | RMSNorm, no bias | |
| Norm placement | reordered (post-block, pre-residual-add) | OLMo 2 style |
| QK-norm | yes, RMSNorm on Q and K before RoPE | |
| Vocab | 32,768 (custom BPE) | see §3 |
| Context | 1024 | |
| Embeddings | tied input/output | |
| Biases | none anywhere | |

Rough param count: 24 × (4.19M attn + 9.44M MLP) + 33.6M embeddings ≈ **361M**.

**Verify this with an actual param counter before launching.** Off-by-one on the MLP hidden dim moves the total by 10M and silently changes your token/param ratio.

### Why these choices

- **QK-norm is the highest-value single change.** Its real benefit isn't spike avoidance, it's that it lets you run a much higher learning rate. In a controlled SmolLM-360M ablation (almost exactly your size): at 3e-5 and 3e-4 it made no consistent difference, but at 1e-3 the model *without* QK-norm blew up to final loss 6.334 while the model *with* it converged to 2.496 — the best of all six configs.
- **Reordered norm and QK-norm ship together.** OLMo 2's ablations found neither helps in isolation; together they improve both the growth and the spikiness of the gradient norm. Don't cherry-pick one.
- **32k vocab over 50,257.** GPT-2's vocab would be 51M params (14% of the model), and with tying it's your largest single matmul. A 32k BPE trained on your actual corpus compresses better and puts more of the budget in the layers.

---

## 3. Data

### Pretraining corpus — 10B tokens

Use the pre-mixed, pre-shuffled HF Smol-Data blends. They remove the mixture-tuning variable entirely, which is not a variable you want on run one.

- **Primary:** `HuggingFaceFW/finepdfs_edu_50BT-dclm_30BT-fineweb_edu_20BT-shuffled` — a tested 100B-token mix; take a 9B slice.
- **Fallback if you want single-source simplicity:** the FineWeb-Edu 10BT sample.

All ODC-By. Note that ODC-By licenses the *annotations*, not the underlying scraped text — fine for a personal project, not a provenance guarantee.

### Anneal corpus — final ~1B tokens

**This is the decision that cannot be retrofitted.** Reserve the last ~10% of the token budget for a different mixture, trained during LR decay:

- higher-quality web (FineWeb-Edu score 4+, or FinePDFs-Edu)
- instruction-formatted data at ~3–5% — pull from the `Mid` subset of `HuggingFaceTB/smoltalk2`
- optionally a small math/code slice (FineMath, RefineCode)

This is what OLMo calls Dolmino and nanochat calls midtraining. It changes your LR schedule and requires a second data pipeline, so build it before launch.

### Tokenizer

Train a 32k BPE on ~2–4B characters sampled from your actual pretraining mix. Do this first — everything downstream depends on it, and retokenizing 10B tokens is a multi-hour job you don't want to discover you need on day two.

### Pre-flight

Tokenize the full 10B to local NVMe as uint16 shards (~20GB) **before the GPU clock starts.** Do not pay A100-hours to stream and tokenize. Verify shard count, total token count, and that a random shard decodes to sensible text.

---

## 4. Training stack (JAX)

Take the dependency list straight from MaxText, which is the closest thing to an official answer:

| Layer | Choice |
|---|---|
| Core | `jax[cuda12]` |
| Model | **Flax NNX** (not linen — linen is legacy) |
| Optimizer | Optax |
| Checkpointing | Orbax (async) |
| Data loading | Grain |
| Attention | `jax.nn.dot_product_attention(..., implementation='cudnn')` |

**Do not use MaxText itself.** It's TPU-first and built for sharding across pods; the config surface will bury you at 360M on one GPU. It's explicitly a reference implementation to fork — read its attention and optimizer code, don't inherit its launcher.

**Do not write your own attention kernel.** The cuDNN backend skips computing the non-causal regions entirely, where the XLA path materializes a mask tensor and applies it to the logits. Confirm cuDNN is actually being selected — don't assume the flag took.

**Grain matters more than it looks.** On a rented box you want resume to replay the exact data order. Grain gives checkpointable iterator state for free; a hand-rolled loader won't.

### JAX-specific traps

- `donate_argnums` on params and opt_state, or XLA keeps a duplicate of everything.
- Static shapes only. Any batch/seq change triggers full recompile. Drop the ragged last batch.
- `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` on a dedicated box (default preallocates 75%).
- No autocast. Cast to bf16 at matmul boundaries manually; keep params and Optax state fp32.
- Set `JAX_COMPILATION_CACHE_DIR` so restarts don't re-pay tens of seconds of compile.

---

## 5. Precision

**BF16 compute, FP32 master weights, FP32 optimizer states.** Not negotiable, not clever, correct.

Explicitly **not** NVFP4: it requires Blackwell (A100 has no FP4 path at all), only the linear-layer GEMMs get accelerated, the recipe needs 2D block scaling + Random Hadamard Transforms + stochastic rounding + the last 15% of layers held in BF16, and the residual loss gap *grows* as models get smaller. NVIDIA measured ~3× more zero-valued weight gradients under NVFP4 than BF16 at the same token horizon — a silent degradation invisible in a loss curve. Wrong tool, wrong scale, wrong hardware.

Note that "pure" BF16 is what's unstable; keeping the master weights and optimizer in FP32 is what buys the stability. Don't let a memory-saving impulse talk you out of it — you have 80GB and need under 6.

---

## 6. Optimization

| Setting | Value |
|---|---|
| Optimizer | AdamW, β=(0.9, 0.95), wd 0.1 (no wd on norms/embeddings) |
| Schedule | **WSD** (warmup–stable–decay), not cosine |
| Warmup | ~2% of steps |
| Decay | final ~10–15%, to ~0 — this segment *is* the anneal phase |
| Peak LR | start at 1e-3; QK-norm is what makes this survivable |
| Grad clip | global norm 1.0 |
| Seq len | 1024 |
| Tokens/optimizer step | ~524k (device batch 32 × grad accum 16) |
| Total steps | ~19,000 |

WSD over cosine specifically because the decay segment and the anneal data swap are the same event. Cosine makes that awkward.

Also apply: **z-loss** on output logits, and **init all params at mean 0, std 0.02** (OLMo 2 switched to this from scaled init for exactly this stability reason).

### Memory check

360M params × (2 bf16 + 2 grad + 4 fp32 master + 8 Adam states) ≈ **5.8GB**. Everything else on the 80GB card is activations. You have enormous headroom — use a large device batch and minimize grad accumulation, which is itself worth a few points of MFU.

---

## 7. Execution phases

| # | Phase | Duration | Gate |
|---|---|---|---|
| 0 | Tokenizer train + full corpus tokenization | 2–4h CPU | Shards decode correctly; token count matches |
| 1 | Throughput smoke test (200 steps) | 20 min | **MFU ≥ 0.35 — see below** |
| 2 | Resume test: kill at step 500, restart | 15 min | Loss curve continuous; data order replays |
| 3 | Pretraining, stable phase | ~45h | No sustained grad-norm growth |
| 4 | Anneal / decay phase | ~6h | Loss drops visibly at the mixture swap |
| 5 | SFT | 2–4h | Model answers a question instead of continuing it |
| 6 | Eval + writeup | 2h | — |

### The Phase 1 gate

Compute achieved MFU directly: `tokens_per_sec × 6 × 361e6 × 1.08 / 312e12`

- **≥ 0.35** → proceed.
- **0.25–0.35** → profile with `jax.profiler.trace`, check the TensorBoard op breakdown. Usually attention silently fell back to the XLA path, or the dataloader isn't prefetching.
- **< 0.25** → something structural is wrong. Find it on $2 of GPU time, not $120.

This 20-minute measurement is the highest-value thing you do all week. JAX on a single NVIDIA GPU is a second-class path relative to TPU; do not assume 0.40.

### Checkpointing

Every ~15 minutes, async via Orbax. If you're on a community-tier or spot instance, preemption over a 50-hour run is likely, not hypothetical. **Test resume by actually killing the process** — an untested resume path is not a resume path.

---

## 8. Post-training

Two to four GPU-hours. Negligible against 51, and it's the difference between a text continuer and something you can talk to.

**Primary dataset: SmolTalk** (~1.1M records, built specifically for small-LM training). The reason it exists is your exact problem — the SmolLM2 team found their base model beat other 1–2B models, but fine-tuning on MagPie-Pro or OpenHermes-2.5 left it behind those models' post-trained versions. Small models respond differently to SFT data than 8B+ models do.

Alternatives: **TuluTalk** (808k, Magpie-filtered merge of Tulu 3 and SmolTalk, ~23% smaller than SmolTalk with better benchmarks) or the full **Tulu 3 SFT mixture** (939k, ODC-BY, designed for 8B+ Llamas).

Filter to ~100–200k examples, 2–3 epochs, ~300–500M tokens.

### Cut ruthlessly

- **Reasoning traces / OpenThoughts.** A 360M model can't learn CoT; you'll teach it to emit long confident garbage. Take `no_think` variants everywhere.
- **Safety/jailbreak subsets.** Tulu 3 spends 100k rows (>10%) on refusal behavior this model can't execute coherently.
- **Multilingual, tool use, 64k-context subsets.** All present, all wasted capacity.
- **RL of any kind.** RLVR needs a policy good enough that the reward signal separates good rollouts from bad. Yours isn't.

### Optional: DPO on UltraFeedback

SmolLM2 tested UltraFeedback, UltraInteract, Capybara, and ORCA; UltraFeedback was most consistently effective. Expect marginal capability gains at 360M — do it to learn the DPO machinery, not for the score.

### The actual bug surface

Chat template and loss masking. You train on assistant tokens only; user turns and template scaffolding are masked out. Get it wrong and the model hallucinates user turns and talks to itself.

**Write a test that decodes a batch and prints which token positions carry loss. Eyeball it before launching.**

---

## 9. Evaluation

- **During training:** held-out loss / bits-per-byte on a slice of the pretraining distribution the model never sees. This is your real signal; benchmarks are too noisy at this scale.
- **Base model:** the DCLM CORE suite, for comparability against the nanochat leaderboard. GPT-2's CORE is 0.256525 — that's the floor to clear, not the target.
- **Post-SFT:** talk to it. Twenty minutes of manual conversation tells you more than any 360M benchmark number.

**Reference points:** nanochat's current record is 1.65h on 8×H100 (~13 GPU-hours) to clear GPT-2 CORE; its $1000 tier is 41.6h on 8×H100 (~333 GPU-hours) at depth 32. Your ~51 A100-hours ≈ 17 H100-equivalent hours. You are in the same neighborhood as the speedrun, with a much longer horizon on a smaller model — so you should clear GPT-2 comfortably and the interesting question is by how much.

---

## 10. Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| JAX MFU below target | **High** | Phase 1 gate; cuDNN attention; profile early |
| Spot/community preemption | Medium–High | 15-min Orbax checkpoints, tested resume |
| Loss spikes / divergence | Low | QK-norm + reordered norm + z-loss + clip 1.0 |
| Chat template / loss mask bug | **High** | Decode-and-inspect test before SFT |
| Tokenizer redo mid-project | Low | Do it in Phase 0, verify before tokenizing 10B |
| Param count ≠ intended | Medium | Count params, assert against target, before launch |

---

## 11. Open decisions

- Provider: Lambda/RunPod (stable, ~$1.79) vs. Vast/Thunder (~$1.09, preemption risk). Given a tested resume path, the cheap tier is defensible.
- On-demand vs. spot — spot roughly halves the bill and a 50-hour run is exactly the workload where that compounds.
- Whether to swap AdamW for Muon. Higher token-efficiency, but adds the attention-logit-explosion failure mode that MuonClip/QK-Clip exists to solve. Not for run one.
- Exact anneal mixture ratios — worth one small ablation if you have hours left over.

---

## Unverified numbers in this spec

Flagged so you check rather than inherit:

- **MFU estimates (0.30–0.42)** are mine, not measured. The whole Phase 1 gate exists because of this.
- **RedPajama-v2 ~30T tokens** and **FineMath ~34B (3+ subset)** — recalled, not verified against the HF cards.
- **Param count 361M** — arithmetic above, not run through a counter.
- **A100 dense BF16 = 312 TFLOPS** — spec sheet, achievable ceiling only.
