# Project Spec: ~100M LLM Pretraining Run (local)

**Goal:** a working end-to-end pipeline — data → tokenizer → pretraining → anneal → SFT → talk to it — on a single RTX 5090 I already own, in JAX.

**Success criterion:** the pipeline is the deliverable; the model is the receipt. Every stage runs, and I understand each one line by line. The model should be *recognizably an LLM* — grammatical, locally coherent, answering in the shape of an answer after SFT. Not a good model. A real one. Keep a running log of what each decision cost.

**Relationship to `llmro-mini`:** same project shape, ~1/34th the compute. That spec assumes a rented A100 and 2.3e19 FLOPs. This one assumes hardware I own and can re-run, which changes the currency from dollars to attempts — and that changes almost every number below.

---

## 1. Budget

| Item | Value |
|---|---|
| Hardware | 1× RTX 5090, 32GB GDDR7 (owned) |
| Peak BF16 dense | **229.5 TFLOPS measured** (Phase A; spec sheet said ~209) |
| Planning MFU | ~~0.30 assumed~~ → **0.58 measured** (Phase B, real config, fake data) |
| Real run | **103,829,760 params** (counted) × 5B tokens |
| Total FLOPs | 6ND × 1.08 ≈ **3.4e18** |
| Wall clock | **~6.5h** at the measured 214k tok/s |
| Cost | electricity |

**Phase A resolved the two headline unknowns.** Measured 229.5 TFLOPS sustained on large BF16 matmuls — *above* the 209 spec-sheet figure, which settles the FP32-accumulate question: GB202 runs BF16 with FP32 accumulate at **full rate**, not half. Every wall-clock figure below is therefore the good case, not the bad one. `implementation='cudnn'` also verifiably selects cuDNN.

**Phase B measured the real config end to end** (fake data, no dataloader/checkpointing/z-loss/clipping, so treat as an optimistic ceiling):

| Device batch | Tokens/step | ms/step | tok/s | TFLOP/s (6N, total N) | MFU | Peak mem |
|---|---|---|---|---|---|---|
| 8 | 8,192 | 40.3 | 203,356 | 126.7 | 55.2% | 6.43 GB |
| 16 | 16,384 | 82.1 | 199,545 | 124.3 | 54.2% | 10.51 GB |
| **24** | 24,576 | 114.5 | **214,621** | 133.7 | **58.3%** | 14.72 GB |
| 32 | — | — | — | — | **OOM** (even at mem_fraction 0.92) | — |

At 214k tok/s, 5B tokens is **~6.5 hours**, not 15. Counting non-embedding params instead of total (the stricter convention) gives MFU 47.7% — still far past the 0.30 gate either way, so the §8 Phase 2 decision resolves to "proceed at 5B tokens" on both conventions.

Time is flexible, so the binding constraint isn't hours — it's **how many attempts I get**. Design for re-runnability over peak quality.

**Why not 3 hours.** An earlier draft of this spec targeted 3h. At MFU 0.30 that caps `N × D` at ~1e17, which puts the compute-optimal point at ~70M params on 1.5B tokens. That model would be Chinchilla-correct and disappointing to talk to. Coherence at this scale tracks *total tokens seen* far more than parameter count — GPT-2 small is recognizably an LLM at 124M params because it saw 10B tokens, four times past Chinchilla. So: overtrain deliberately, and spend the wall clock.

---

## 2. Two configs, one pipeline

**This is the central structural idea of the project.** Every bug worth finding is config-independent: chat-template loss masking, resume replaying data order, a shard that decodes to garbage, a param counter disagreeing with arithmetic. Find them in a 2-minute loop, not a 15-hour one.

| | Debug | Real |
|---|---|---|
| Params | **11,112,448** (counted) | **103,829,760** (counted) |
| d_model / layers / heads | 256 / 6 / 4 | 768 / 12 / 12 |
| SwiGLU hidden | 704 | 2048 |
| Tokens | 50M | 5B |
| D/N | 4.5 | 48 |
| Tokens/step | 32k | 262k |
| Device batch × accum | 32 × 1 | **16 × 16** (see §7) |
| Steps | ~1,500 | ~19,000 |
| Runtime | **~45 s** measured | **~6.5h** measured |

Same tokenizer, same shards, same code path, same phase sequence — only the config differs. The debug config runs *all* stages including anneal, SFT, and generation.

Its batch is deliberately small so it takes ~1,500 steps rather than ~190: that's enough to actually exercise warmup → stable → decay instead of dying in warmup. **Its loss is not meaningful** — at d256 the embedding is 57% of parameters. It is a plumbing test, not a small experiment.

Run the debug config until every phase is green. Then launch the real one, once.

---

## 3. Model

Keep GPT-2 small's *shape*, replace its 2019 internals.

| Parameter | Value | Note |
|---|---|---|
| Layers | 12 | |
| d_model | 768 | |
| Heads | 12 (head_dim 64) | MHA; GQA is an inference optimization, pointless here |
| MLP | SwiGLU, hidden 2048 | 3 matrices, not 2 |
| Positional | RoPE | not learned embeddings |
| Norm | RMSNorm, no bias | |
| Norm placement | reordered (post-block, pre-residual-add) | OLMo 2 style |
| QK-norm | yes, RMSNorm on Q and K before RoPE | |
| Vocab | 24,576 (custom BPE) | see §4 |
| Context | 1024 | |
| Embeddings | tied input/output | |
| Biases | none anywhere | |
| Dropout | **0** | |

Param count: 12 × (2.36M attn + 4.72M MLP) + 18.9M embeddings ≈ **104M**.

**Counted in Phase B: 103,829,760 exactly** (debug config: 11,112,448). The arithmetic held.

One correction to the table above: the 2.36M per-layer attention figure is `4 × 768 × 768` = 2,359,296, which **omits the QK-norm scales** — the real number is 2,359,424, i.e. +128/layer (two RMSNorm vectors of `d_head` = 64). Immaterial to the total, but it's the kind of 128 that makes a param-counter assertion fail and sends you hunting for a real bug.

### Why these choices

- **d768 × 12L is the most trodden shape in existence.** Every reference implementation, known-good loss curve, and debugging anecdote lives at GPT-2 small's dimensions. The deliverable here is a working pipeline — spending the novelty budget on architecture is a bad trade. Modern internals, 2019 skeleton.
- **QK-norm is the highest-value single change.** Its real benefit isn't spike avoidance, it's that it permits a much higher learning rate. In a controlled SmolLM-360M ablation: at 3e-5 and 3e-4 it made no consistent difference, but at 1e-3 the model *without* QK-norm blew up to final loss 6.334 while the model *with* it converged to 2.496.
- **Reordered norm and QK-norm ship together.** OLMo 2's ablations found neither helps in isolation; together they improve both the growth and the spikiness of the gradient norm. Don't cherry-pick one.
- **24k vocab.** At d768 a 32k tied embedding is 25M params — 20% of the model doing lookup instead of thinking. 24k is the middle: better compression than 16k, meaningfully cheaper than 32k. Low stakes but **irreversible** — retokenizing 5B tokens is a multi-hour job. Decide it in Phase 0 and don't revisit.
- **Dropout 0.** GPT-2 used 0.1. At D/N ≈ 48 you are nowhere near the regime where regularization helps; inheriting that default would actively cost you.

---

## 4. Data

### Pretraining corpus — 5B tokens

**`HuggingFaceFW/fineweb-edu`, config `sample-10BT`.** Verified sizes from the HF datasets-server:

| Dataset | Parquet | Raw | Rows | Tokens |
|---|---|---|---|---|
| **`sample-10BT`** | **28.5 GB** | 48.1 GB | 9.7M | 10B |
| `sample-100BT` | 286 GB | 479 GB | 97.3M | 100B |
| `finepdfs…shuffled` (llmro-mini's primary) | 268 GB | 479 GB | 56.1M | 100B |

llmro-mini's premix is 268 GB — sensible when renting a box and using 9B of its 100B tokens, absurd here where under 5% would be touched. `sample-10BT` needs ~half its shards pulled (~14 GB) to cover 5B tokens.

- **The `-Edu` filter is the highest-value data decision at this scale.** It is the thing that makes a 100M model coherent on a small budget. Take FineWeb-Edu, not raw FineWeb.
- **"10BT" means 10B *GPT-2* tokens, not yours.** A 24k BPE compresses worse than GPT-2's 50k (+~10% tokens) but will be trained on FineWeb-Edu rather than Reddit outbound links (−some). These roughly cancel; expect 10B ± 15% of your tokens. Either way there's ~2× headroom.
- **Consequence to accept:** FineWeb-Edu is expository and textbook-ish. The base model will sound like a Wikipedia article and be weak on dialogue register. SFT is therefore not optional — it is load-bearing for the stated goal.

### Anneal corpus — final ~600M tokens

**This is the decision that cannot be retrofitted.** Reserve the last ~12% of the token budget for a different mixture, trained during LR decay.

The good news: **it comes free from the same download.** `sample-10BT` carries `score` and `int_score` columns, and FineWeb-Edu is only filtered at score ≥ 3 — so the anneal set is carved out of shards already on disk by filtering **score ≥ 4**. Add instruction-formatted data at ~3–5% from the `Mid` subset of `HuggingFaceTB/smoltalk2`.

**Verified in Phase C** with a full metadata pass over shard 0 (726,000 docs, 0.755B GPT-2 tokens):

| `int_score` | Docs | Share of tokens |
|---|---|---|
| 3 | 629,506 | 86.11% |
| 4 | 95,929 | 13.82% |
| 5 | 565 | 0.07% |

**score ≥ 4 = 13.89% of tokens.** The guess of "~1–1.5B from a 10B sample" was right, at the top of its range — the full sample yields ~1.53B of our tokens. Mean doc length measured 1040 GPT-2 tokens against the assumed 1030, so the intra-document-masking argument in the unverified-numbers list stands.

Budget check at the 8 shards actually pulled: **6.29B our-tokens total → 0.87B anneal pool, 5.41B base.** Against a requirement of 600M anneal + 4.4B base, that fits with room. Do not drop below 7 shards.

One thing the arithmetic assumes: that score ≥ 4 docs are *carved out* of the base corpus rather than used in both. If you leave them in the pretraining mix as well, the anneal stops being a distribution shift and the §10 "bpb drops visibly at the swap" signal disappears.

### Tokenizer

24k byte-level BPE, trained on 2–4B characters sampled from the actual pretraining shards.

**Reserve the chat-template special tokens now.** The classic failure: train the tokenizer, tokenize 5B tokens, pretrain for 15 hours, reach SFT, discover there is no `<|im_start|>`/`<|im_end|>`, and choose between hacking in out-of-vocab tokens or retokenizing the corpus. This means **the chat template must be decided before the tokenizer is trained** — a weirdly early dependency, and the single most expensive ordering mistake available in this project.

### Disk

| | Size |
|---|---|
| Parquet download (~half the shards) | ~14 GB |
| Tokenized `uint16` shards, 5B tokens | **10 GB** |
| Tokenizer training sample | ~3 GB, from the same shards |
| Peak transient / persistent | ~30 GB / 10 GB |

`uint16` works because vocab 24,576 < 65,536. Tokenize to local NVMe **before the training clock starts**; verify shard count, total token count, and that a random shard decodes to sensible text. Delete the parquet once shards verify.

---

## 5. Training stack (JAX)

| Layer | Choice |
|---|---|
| Core | `jax[cuda13]` (what's actually installed; jax/jaxlib 0.11.0) |
| Model | **Flax NNX** (not linen — linen is legacy) |
| Optimizer | Optax |
| Checkpointing | Orbax (async) |
| Data loading | Grain |
| Attention | `jax.nn.dot_product_attention(..., implementation='cudnn')` |

Dependency list taken from MaxText, which is the closest thing to an official answer. **Do not use MaxText itself** — it's TPU-first and built for sharding across pods. Read its attention and optimizer code; don't inherit its launcher.

**Do not write your own attention kernel.** The cuDNN backend skips computing the non-causal regions entirely, where the XLA path materializes a mask tensor and applies it to the logits.

**Grain matters more than it looks.** You will restart this run, and you want resume to replay the exact data order. Grain gives checkpointable iterator state for free; a hand-rolled loader won't.

### The sm_120 problem

llmro-mini contains a paragraph arguing *for* the A100 precisely because consumer Blackwell is the roughest edge of the jaxlib/cuDNN support matrix. That risk hasn't gone away — it's been deliberately accepted, because the hardware is free and the A100 isn't. It is now the **top entry in the risk register**, not a footnote.

Concretely, two things must be confirmed before anything else happens: that JAX sees the card at all, and that `implementation='cudnn'` actually selects cuDNN rather than silently falling back to XLA. Don't assume the flag took.

### `XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async` is mandatory on this box

**Found in Phase B, and it is the single highest-impact environment fact in this document.** With jax/jaxlib 0.11.0 under WSL2 (driver 610.47), XLA's default BFC allocator cannot obtain more than **~4.6 GB** of the 32 GB card. Measured, in a fresh process each time:

| Single allocation | default BFC | `cuda_async` |
|---|---|---|
| 4 GB | OK | OK |
| 5 GB | **fail** | OK |
| 12 GB | **fail** | OK |
| 28 GB | **fail** | OK |

This is not the card, the driver, or the model. Raw `cuMemAlloc` through `libcuda` reaches 27 GB in the same environment, and `nvidia-smi` reports 28.7 GB free throughout — BFC's arena growth is what fails, logging `CUDA_ERROR_OUT_OF_MEMORY` while the driver is perfectly willing. `XLA_PYTHON_CLIENT_PREALLOCATE`, `TF_GPU_ALLOCATOR`, and `XLA_PYTHON_CLIENT_ALLOCATOR=platform` all make no difference; only `cuda_async` does.

**Set it in the environment before every run.** Without it the real config OOMs at device batch 8 and the project looks memory-bound when it isn't — which is exactly the wrong conclusion to draw, and it silently costs you a factor of ~3 in batch size.

### Other JAX-specific traps

- `donate_argnums` on params and opt_state, or XLA keeps a duplicate of everything.
- Static shapes only. Any batch/seq change triggers full recompile. Drop the ragged last batch.
- `XLA_PYTHON_CLIENT_MEM_FRACTION` — the 0.75 default is *not* the binding constraint here; raising it to 0.85 or 0.92 did not make device batch 32 fit. Fix the allocator first, then don't bother with this knob.
- No autocast. Cast to bf16 at matmul boundaries manually; keep params and Optax state fp32.
- Set `JAX_COMPILATION_CACHE_DIR` — with a sub-minute debug loop, recompiling every iteration is most of your runtime.

---

## 6. Precision

**BF16 compute, FP32 master weights, FP32 optimizer states.** Not negotiable, not clever, correct.

At 104M params that's 104M × (2 bf16 + 2 grad + 4 fp32 master + 8 Adam) ≈ **1.7 GB**. "Pure" BF16 is what's unstable, and keeping master weights and optimizer in FP32 is what buys the stability.

**Correction to the original claim that "memory is not a constraint at this scale."** It is — just not because of the weights. At device batch 24 the measured peak is 14.72 GB against 1.7 GB of weights and optimizer state; the balance is activations and, dominantly, the output logits (§7). The 1.7 GB figure is still correct and still leaves no reason to compromise on FP32 masters — that part of the argument holds. But don't read it as "there's 30 GB of headroom," because there isn't.

Explicitly **not** NVFP4. The 5090 *does* have an FP4 path — unlike the A100, so llmro-mini's reasoning doesn't transfer — but the conclusion is unchanged and stronger: only the linear-layer GEMMs get accelerated, the recipe needs 2D block scaling + Random Hadamard Transforms + stochastic rounding + the last 15% of layers held in BF16, and the residual loss gap *grows* as models get smaller. This model is 5× smaller than the ones that argument was measured on.

---

## 7. Optimization

| Setting | Value |
|---|---|
| Optimizer | AdamW, β=(0.9, 0.95), wd 0.1 (no wd on norms/embeddings) |
| Schedule | **WSD** (warmup–stable–decay), not cosine |
| Warmup | 2% ≈ 380 steps |
| Decay | final 12% ≈ 2,300 steps ≈ **600M tokens** — this segment *is* the anneal phase |
| Peak LR | 1e-3; QK-norm is what makes this survivable |
| Grad clip | global norm 1.0 |
| Seq len | 1024 |
| Tokens/optimizer step | ~262k (**device batch 16 × grad accum 16** — measured, see below) |
| Total steps | ~19,000 |

The decay segment and the anneal data swap are the same event; WSD makes that natural and cosine makes it awkward. Note the decay budget (600M tokens) and the anneal corpus budget (§4) are the same number by construction — if one moves, move both.

Also apply **z-loss** on output logits, and **init all params at mean 0, std 0.02** (OLMo 2 switched to this from scaled init for exactly this stability reason).

### Batch size — measured, and the original estimate was 5× too high

The draft called for device batch 128 × 1024 seq, derived from ~26 GB usable at a rough ~20 bytes per token per layer per dim. **That estimate was wrong by more than 5×.** Measured ceiling on the real config is **device batch 24**; batch 32 OOMs even at mem_fraction 0.92.

Why the rule of thumb missed it: it scales with *layers*, and the dominant term here doesn't. The output logits are `batch × seq × vocab` — at batch 24 that's 1.2 GB in bf16, and the `.astype(float32)` before cross-entropy materializes a second full-width copy, with the backward pass needing a third. At the draft's batch 128 the logits alone would want ~32 GB. No amount of per-layer activation accounting sees this, because it has nothing to do with depth.

**Use device batch 16 × grad accum 16** = 262,144 tokens/step exactly. Batch 24 is marginally faster per token (214k vs 200k tok/s) but 256 sequences doesn't divide by 24; batch 32 would divide evenly but doesn't fit. The throughput cost of choosing 16 is ~7%.

If you want the batch back, **chunked cross-entropy** is the fix — compute the loss over vocab blocks and never materialize the full fp32 logits tensor. Gradient rematerialization is the wrong knob here; it targets per-layer activations, which are not what's full.

### Muon: the one upgrade worth trying

llmro-mini defers Muon ("not for run one"). **Reverse that here**, as a Phase 2b A/B rather than a baked-in default:

- It is the main driver of the nanoGPT speedrun records, which are set at almost exactly this config — ~124M params on FineWeb. Unusually direct evidence.
- **`optax.contrib.muon` exists** (optax 0.2.8), so it's an import, not a Newton-Schulz implementation to debug. This is the fact that flips the cost/benefit.
- Its known failure mode is attention logit explosion — which is what MuonClip/QK-Clip was invented for, and what QK-norm already defends against. The insurance is pre-paid.
- The 2-minute debug loop makes it a five-minute experiment instead of a commitment.

Caveat: Muon applies to 2D hidden matrices only. Embeddings, norms, and the output head stay on AdamW, and that hybrid is where the implementation bugs live. **Get AdamW green first, then swap.**

Held for run two: **μP**, to tune LR on the debug config and transfer to the real one. It fits the two-config structure almost too well, but it's fiddly and a silent μP bug looks exactly like "the model is just mediocre."

---

## 8. Execution phases

| # | Phase | Duration | Gate |
|---|---|---|---|
| 0a | ~~**JAX on sm_120**~~ | **DONE** | ✅ card seen; cuDNN verifiably selected; 229.5 TFLOPS |
| 0b | Chat template decided, tokenizer trained | 1h | Special tokens present in vocab |
| 0c | Tokenize corpus; check score ≥ 4 yield | 2–4h CPU | Shards decode; token count matches; anneal set ≥ 600M |
| 1 | **Debug config, all stages end to end** | ~10 min/loop | Every phase runs; SFT loss mask correct; model emits text |
| 2 | ~~Throughput measurement, real config~~ | **DONE early** | ✅ MFU 0.58; param count exact; batch ceiling 24 |
| 2b | Muon A/B on debug config | 10 min | Pick optimizer |
| 3 | Resume test: kill at step 500, restart | 15 min | Loss curve continuous; data order replays |
| 4 | Pretraining, stable phase | **~6h** | No sustained grad-norm growth |
| 5 | Anneal / decay phase | ~2h | Held-out bpb drops visibly at the mixture swap |
| 6 | SFT | ~15 min | Model answers a question instead of continuing it |
| 7 | Talk to it; write up | 1h | — |

**Phase 0a comes before everything.** It is 20 minutes and it can invalidate the entire project. Do not train a tokenizer before knowing JAX runs on this card.

**Phase 1 is where the time actually goes**, and that's correct. Every iteration of it is a bug that doesn't cost 15 hours to discover.

### The Phase 2 measurement

Compute achieved MFU directly: `tokens_per_sec × 6 × 103.8e6 / 229.5e12`

- **≥ 0.30** → proceed at 5B tokens.
- **0.20–0.30** → proceed, but drop to 3B tokens (D/N ≈ 29) or accept the longer clock.
- **< 0.20** → something structural is wrong. Profile with `jax.profiler.trace`. Usually attention silently fell back to the XLA path, or the dataloader isn't prefetching.

**Resolved in Phase B: 0.58** (0.48 non-embedding). Proceed at 5B tokens, ~6.5h. Re-run this measurement once Grain is in the loop — the gate above still applies to the *real* pipeline, and a dataloader that doesn't prefetch is exactly the failure this catches.

### Checkpointing

Every ~30 minutes, async via Orbax. There's no preemption risk on your own box, but there are power cuts, crashes, and the near-certainty that you'll want to restart from somewhere. **Test resume by actually killing the process** — an untested resume path is not a resume path.

---

## 9. Post-training

~15 GPU-minutes. Negligible against 15 hours, and it's the difference between a text continuer and something you can talk to. Given a FineWeb-Edu-only pretraining diet, it's also what supplies the dialogue register the base model has never seen.

**Primary dataset: SmolTalk** (~1.1M records, built specifically for small-LM training). It exists because of exactly this problem — the SmolLM2 team found their base model beat other 1–2B models, but fine-tuning on MagPie-Pro or OpenHermes-2.5 left it behind those models' post-trained versions. Small models respond differently to SFT data than 8B+ models do.

Filter to **~50k examples, 2–3 epochs, ~50M tokens** — scaled down from llmro-mini's 100–200k, proportionate to a 104M model on 5B tokens.

### Cut ruthlessly

- **Reasoning traces / OpenThoughts.** A 104M model cannot learn CoT; you'll teach it to emit long confident garbage. Take `no_think` variants everywhere.
- **Safety/jailbreak subsets.** Refusal behavior this model cannot execute coherently.
- **Multilingual, tool use, long-context subsets.** All present, all wasted capacity.
- **RL of any kind.** RLVR needs a policy good enough that the reward signal separates good rollouts from bad. Yours isn't. DPO likewise — at 104M it would be machinery-learning, not capability, and that's a run-two project.

### The actual bug surface

Chat template and loss masking. You train on assistant tokens only; user turns and template scaffolding are masked out. Get it wrong and the model hallucinates user turns and talks to itself.

**Write a test that decodes a batch and prints which token positions carry loss. Eyeball it.** This runs in the debug config in seconds — there is no excuse for finding this bug at hour 15.

---

## 10. Evaluation

**No benchmark harness.** Deliberately.

- **During training:** held-out loss / bits-per-byte on a FineWeb-Edu slice the model never sees. This is the real signal.
- **At the mixture swap:** bpb should drop visibly when the anneal data comes in. If it doesn't, the swap didn't happen — check the loader before blaming the data.
- **Post-SFT:** talk to it for twenty minutes.

llmro-mini targets the DCLM CORE suite against GPT-2's 0.256525. That's dropped here: CORE is an average of ~22 multiple-choice benchmarks whose value is *comparability to other people's runs*, and at 104M most of its component tasks sit near chance. The number would be noise dressed as a score, and wiring up the harness is hours that Phase 1 wants.

### What "recognizably an LLM" actually means

Set expectations now so the result isn't read as failure:

**Expect:** grammatical English indefinitely. Topical coherence for roughly a paragraph, then drift. Confident, unreliable facts. After SFT, answers shaped like answers — which is the single most satisfying moment in the project.

**Do not expect:** multi-turn memory, reasoning, reliable facts, or instruction nuance.

---

## 11. Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| ~~**JAX/cuDNN broken on sm_120**~~ | **RESOLVED** | Phase A: JAX sees the card, cuDNN verifiably selected |
| ~~FP32-accumulate is half-rate on GB202~~ | **RESOLVED** | Phase A: 229.5 TFLOPS measured, full rate |
| ~~MFU below target~~ | **RESOLVED** | Phase B: 0.58 measured (0.48 non-embedding) vs 0.30 gate |
| ~~Param count ≠ intended~~ | **RESOLVED** | Phase B: 103,829,760 counted, matches |
| ~~Batch size doesn't fit 32GB~~ | **CONFIRMED, mitigated** | Real ceiling is device batch 24, not 128; use 16 × accum 16 |
| Chat template / loss mask bug | **High** | Decode-and-inspect test in debug config |
| **Allocator regressions / env drift** | **Medium** | `XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async` is load-bearing (§5) — if a jaxlib upgrade or a fresh shell drops it, everything OOMs at 1/3 the batch and looks like a model bug |
| score ≥ 4 yields < 600M anneal tokens | Medium | Check in Phase C; widen toward score ≥ 3 |
| Special tokens missing from tokenizer | Medium | Decide chat template *before* training the tokenizer |
| Loss spikes / divergence | Low | QK-norm + reordered norm + z-loss + clip 1.0 |
| Muon hybrid-optimizer bugs | Low | AdamW baseline green first; A/B in debug config |

---

## 12. Open decisions

- **Muon vs AdamW** — resolved empirically at Phase 2b, not now.
- **5B vs 3B tokens** — resolved by the Phase 2 MFU measurement.
- **Exact anneal mixture ratios** — the 3–5% instruction fraction is inherited, not tuned. Cheap to ablate in the debug config if curious.
- **Depth vs width** — 16L × d640 likely beats 12L × d768 slightly at equal params, but narrower matmuls cost MFU on a 5090. Probably a wash; not worth the risk budget on run one.
- **μP** — run two.
- **Whether the debug config lives in the repo as a permanent test target.** It should.

---

## Unverified numbers in this spec

Flagged so they get checked rather than inherited. This list is longer than llmro-mini's because more of this is estimated.

**Resolved by Phases A and B** (struck through, kept for the log):

- ~~**RTX 5090 BF16 dense = 209.5 TFLOPS**~~ → **229.5 measured**, above spec sheet.
- ~~**Whether GB202 runs BF16 with FP32 accumulate at full or half rate.**~~ → **Full rate.** This was flagged as the highest-leverage unknown in the document; it resolved the good way.
- ~~**MFU 0.30**~~ → **0.58 measured** on the real config (0.48 counting non-embedding params only). Wall clock drops from ~15h to ~6.5h.
- ~~**Param count 104M**~~ → **103,829,760 counted.**
- ~~**Activation memory coefficient (~20 bytes/token/layer/dim)**~~ → **the rule of thumb was wrong by >5×** for this model, because the binding term is the `batch × seq × vocab` logits tensor, which doesn't scale with layers at all. Device batch 24 is the measured ceiling, not 128.

**Still unverified:**

- **Muon's efficiency gain** — inferred from published speedrun results at a similar scale, not measured here.
- **Whether the ~6.5h figure survives a real dataloader.** All Phase B throughput is on fake in-memory data with no Grain, no checkpointing, no z-loss, and no gradient clipping. Treat it as an optimistic ceiling; re-measure in Phase G.
- **Whether the BFC allocator ceiling is a jaxlib 0.11.0 bug, a WSL2 limitation, or driver 610.47 specifically.** Not diagnosed further — `cuda_async` works, and root-causing it isn't on the critical path. Worth re-testing on any upgrade.
- ~~**score ≥ 4 anneal yield**~~ → **13.89% of tokens**, measured over shard 0's full 726k docs. ~1.53B from the full 10B sample; the guess held.
- ~~**Mean document length ≈ 1030 tokens**~~ → **1040 measured.** The intra-document-masking argument stands.
- **Token yield of `sample-10BT` under a 24k BPE** — the ±15% cancellation argument is still reasoning, not measurement. This is now the *only* unmeasured number in the data budget, and it resolves the moment you encode shard 0: compare the real output count against `token_count.sum()`, which is the GPT-2 baseline sitting right there in the parquet.

*Dataset sizes in §4 are verified — pulled live from the HuggingFace datasets-server, not recalled.*
