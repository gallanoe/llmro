# Per-Token Adaptive Depth for a Pretrained Hybrid LLM

Retrofitting skip / run-once / run-twice routing onto Qwen3.5 with LoRA adaptation

---

## 1. Summary

### The idea

Not every token needs the same amount of computation. After "sat on", predicting "the" is nearly free; the token that carries the answer to a math problem is not. Standard transformers still spend the same depth on both.

This project adds **per-token adaptive depth** to a pretrained language model. The middle of the network is split into blocks, and for every token a small learned **router** chooses, per block, one of three actions:

- **skip:** the block is bypassed; the token's hidden state passes through unchanged
- **keep:** the block runs once, as in the original model
- **repeat:** the block runs twice, feeding its own output back in

Easy tokens can skip blocks to save compute. Hard tokens can repeat blocks to get more of it.

### Why retrofit instead of training from scratch

Architectures with adaptive or recurrent depth (looped transformers, Mixture-of-Recursions, Mixture-of-Depths) are usually trained from scratch, and the literature suggests training the capability in from the start beats bolting it onto an existing model. Training from scratch is out of reach on a single GPU. The central hypothesis here is that **LoRA adaptation can substitute for from-scratch training**: small trainable corrections, placed exactly where routing pushes a layer off its usual input distribution, can let a pretrained model tolerate skipping and repeating.

Whether LoRA alone is enough is treated as something to measure, not assume (see the capacity ladder in §3.8).

### Two goals

| Track | Goal | Compute budget |
|---|---|---|
| **E: efficiency** | Same quality, fewer blocks per token | Below the native 6 blocks per token |
| **Q: quality** | Better answers, same compute | Exactly 6 blocks per token on average; repeats must be paid for by skips |

### What makes this design distinctive

1. **Hybrid-attention base model.** Qwen3.5 interleaves linear-attention (Gated DeltaNet) layers with full-attention layers. Routing is done at the granularity of one interleave group, so every block ends in a full-attention layer.
2. **Dense cache via shadow writes.** A token that skips a block still writes that block's cache entries, computed cheaply from its unchanged hidden state. Later tokens never find gaps in the cache.
3. **Native equivalence by construction.** When every token keeps every block, the model is exactly the base model: all adapters are gated off, and the base model doubles as the distillation teacher.

### Related work

- **Per-input layer programs on frozen models:** CoLa (Li et al., 2025) showed that skipping and repeating pretrained layers per input can both shorten computation and fix wrong answers. DR.LLM (Heakl et al., 2025) and POLAR (ICML 2026) train routers for this with frozen layers. POLAR found that predicting the whole program up front beats deciding layer by layer, and that useful programs are mostly contiguous segments of one to four layers.
- **Adaptive depth trained from scratch:** Mixture-of-Depths (Raposo et al., 2024), Mixture-of-Recursions (Bae et al., 2025), Ouro (Zhu et al., 2025), Huginn (Geiping et al., 2025).
- **Early exit:** CALM (Schuster et al., 2022), LayerSkip (Elhoushi et al., 2024).
- **Converting pretrained models to recursive ones:** Relaxed Recursive Transformers (Bae et al., 2024) converted Gemma into a looped model with layer-wise LoRA. Their conversion used 15–60B tokens of full-parameter uptraining in addition to LoRA.
- **Layer roles:** middle layers are robust to deletion and reordering while the first and last layers are fragile (Lad, Gurnee & Tegmark, 2024; Gromov et al., 2024). This motivates keeping the first and last blocks fixed.

---

## 2. Design

### 2.1 Base model: Qwen3.5-0.8B

| Property | Value |
|---|---|
| Layers | 24, arranged as 6 × [Gated DeltaNet ×3 → gated full attention] |
| Hidden size | 1024 |
| Full attention | 8 query heads, 2 KV heads, head dimension 256 (rotary on 64 dims), sigmoid output gate |
| Gated DeltaNet | 16 heads, head dimension 128 |
| FFN | SwiGLU, intermediate size 3584 |
| Vocabulary | 248,320, input and output embeddings tied |
| Parameters | ~0.8B total: ~0.25B embeddings, ~0.5B in the 24 layers (~85M per 4-layer block) |

The model is loaded text-only (no vision encoder) with the multi-token-prediction head disabled. Qwen3.5-2B has the same 24-layer, interval-4 layout, so it is the natural second rung of the size ladder.

### 2.2 Blocks and routing plans

The 24 layers form six blocks, B0–B5, each one [DeltaNet, DeltaNet, DeltaNet, full attention] group.

- **B0 and B5 are fixed.** Every token always runs them once. The first and last layers of a transformer do specialized work (turning tokens into features, turning features into predictions) and are fragile under perturbation.
- **B1–B4 are routable.** Each takes one of three actions per token: skip, keep, or repeat (at most one repeat).

A **routing plan** for one token is four actions, one per routable block. There are 3⁴ = **81 possible plans per token**. The number of block executions per token ranges from 2 (skip all four) to 10 (repeat all four); the native model uses 6.

| Block executions | 2 | 3 | 4 | 5 | **6** | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|
| Number of plans | 1 | 4 | 10 | 16 | **19** | 16 | 10 | 4 | 1 |

Execution for one token:

```
h ← B0(h)
for j in 1..4:
    skip:    h ← h
    keep:    h ← Bj(h)
    repeat:  h ← Bj′(Bj(h))      # Bj′ = second pass: same weights as Bj (plus repeat adapters), own cache slot
h ← B5(h)
logits ← LMHead(h)
```

### 2.3 The cache: ten static slots with shadow writes

Each block execution needs its own cache: one full-attention KV cache and three DeltaNet recurrent states (each with a short-convolution state). A repeat pass reads and writes separately from the first pass, because its inputs, and therefore its keys, values, and states, differ.

**Static layout.** The cache always has 10 slots, one per possible block visit:

```
B0 | B1 | B1′ | B2 | B2′ | B3 | B3′ | B4 | B4′ | B5
```

Every sequence has the same cache shape regardless of routing, which keeps batching simple.

**Every token writes every slot.** This is the key design choice. With per-token routing, a later token that runs block j needs every earlier token's entry at block j, including tokens that skipped it. Each visit is therefore one of two kinds:

- **Full visit:** the block runs normally. It updates the hidden state and writes its cache entries.
- **Shadow write:** the block is skipped. The hidden state passes through unchanged, but the token still computes the slot's cache entries from that unchanged hidden state, using only the cheap projections:
  - full-attention layer: K and V
  - DeltaNet layers: q, k, v, the short convolution, the write strength β, the decay α, and the recurrent state update

  The expensive parts (attention readout, output projection, FFN) are skipped. A shadow write costs roughly a quarter of a block.

Repeat slots follow the same rule. A token that does not repeat block j writes shadow entries into slot Bj′, computed from its hidden state after Bj.

**Shadow writes are exact for the first layer of each block.** That layer's input is the block's input whether the block runs or not. Layers 2–4 see an input missing the earlier layers' processing, so their shadow entries are approximations. Shadow adapters (§2.5) correct them.

**One useful consequence:** when every token uses the same plan, nobody reads the shadow entries, so the design reduces exactly to per-sequence routing. The frozen-model measurements in §3.3 rely on this.

**Memory**

| Configuration | Attention KV per token | Fixed DeltaNet state per sequence |
|---|---|---|
| Stock Qwen3.5-0.8B (6 attention layers) | 12 KB | ~19 MB |
| Static 10-slot design (10 attention layers) | 20 KB | ~31 MB |

Hybrid models have very small caches: a dense model like Llama-3-8B uses about 128 KB per token. Below roughly 1.5k tokens, the fixed DeltaNet state is the larger share.

**Excluded: sharing one cache slot between a block and its repeat.** Looped-transformer work sometimes shares one KV cache across passes. Huginn tolerates it zero-shot; Ouro collapses under it unless trained for it (MELT; Vendrell et al., 2026). Here it would save only 8 KB per token (20 KB back to 12 KB). It would also break native equivalence, since every token's context would change whenever any token repeats, and it adds a confound when interpreting results. It is out of scope.

### 2.4 Router

- **Input:** each token's hidden state after B0. B0's attention is causal, so the router only uses context up to and including the current token.
- **Architecture:** a two-layer MLP (hidden size 256) with four heads of three logits each (skip, keep, repeat per routable block).
- **Initialization:** strongly biased toward keep, so training begins at exactly the base model's behavior.
- **Training: straight-through Gumbel-softmax.** For each token and block, sample a hard choice with Gumbel noise and use it in the forward pass. In the backward pass, use the gradient of the soft (temperature-τ) sample. Computing that gradient requires all three branches' outputs, so during training every routable block computes skip, keep, and repeat for every token:

  ```
  h_out = y_skip·h + y_keep·B(h) + y_repeat·B′(B(h))
  ```

  In the forward pass only one term is nonzero. In the backward pass, each option's logit moves according to how its branch would have changed the loss.
- **Inference:** no noise; argmax per head; only the chosen branch runs.
- **Optional warm start** from frozen-model labels (§3.3).

### 2.5 Adapters

All adapters are LoRA modules with zero-initialized up-projections, so each starts as a no-op. All are gated off on native visits, which is why the all-keep plan equals the base model exactly.

There are three kinds of off-native computation, each with its own adapter type:

| Type | When it applies | What is off-distribution | Scope |
|---|---|---|---|
| **Jump** | Full visit to the first layer of a block entered right after a skipped block | Input is missing one or more blocks of processing | All linear layers of that layer |
| **Repeat** | Full visit on the second pass through a block | Input has already been through this block | All linear layers of the block |
| **Shadow** | Shadow writes, layers 2–4 of a block | Projections see an unrefined hidden state, **and other tokens read the result** | Only the matrices used for cache writes |

**Soft gating during training.** Under straight-through Gumbel, gates are hard in the forward pass and soft in the backward pass:

- The jump adapter's delta is scaled by the probability that the previous block was skipped for this token.
- The repeat adapter is active on the repeat branch, scaled by `y_repeat`.
- For shadow writes, the gate applies to *which entry is written*: `y_skip · shadow + (1 − y_skip) · full`. The shadow LoRA is always active inside the write-only computation itself.

At inference every gate is exactly 0 or 1.

**Adapter variants** (compared in §3.6):

| ID | Name | Full-visit adapters | Shadow adapters |
|---|---|---|---|
| V1 | Block LoRA | One LoRA per routable block, on for any off-native visit (jump or repeat) | None |
| V2 | Block LoRA + Shadow | As V1 | One per block, layers 2–4 |
| V3 | Typed LoRA + Shadow | Separate jump and repeat LoRAs per block (8 sets: B1 repeat only, B2–B4 both, B5 jump only) | One per block, layers 2–4 |

Shadow adapters are shared between a block's normal slot and its repeat slot. They are split only if their imitation losses diverge.

### 2.6 Shadow adapter training

**Which matrices**

| Layer in block | Shadow LoRA | Matrices |
|---|---|---|
| 1 (DeltaNet) | None; already exact | none |
| 2, 3 (DeltaNet) | Yes | Input projections for q, k, v, β, α (fused in the reference implementation) |
| 4 (full attention) | Yes | K and V projections |

That is 12 shadow adapters (4 routable blocks × layers 2–4). At rank 32 they total about 3M parameters.

**Two training signals**

1. **Local imitation loss.** Because training computes all three branches for every token, the entries a real visit *would* have written are always available. The shadow learns to match them:

   ```
   shadow write:  K̃ = (W_k + ΔW_k) · norm(h_block_input)
   real write:    K  =  W_k · norm(h_layer_input)        # from the keep (or repeat) branch

   L_shadow = Σ_layers normalized_error(shadow_write, stopgrad(real_write))
   ```

   - Error measure: cosine distance for unit-normalized vectors (DeltaNet q and k); variance-normalized MSE for values; MSE on pre-sigmoid β and α. Attention keys are compared after their normalization, i.e., as stored.
   - `stopgrad` on the target keeps the real path from being pulled toward the shadow.
   - The shadow's input is detached for this loss, so it trains only the shadow LoRA and never pushes upstream layers.
   - It is computed for **all tokens**, not only those that skipped. This gives far more training data and keeps shadows accurate as router decisions shift.

2. **End-to-end signal.** Shadow entries are read by later tokens, so the main distillation loss backpropagates through those reads into the shadow LoRA. This teaches what readers actually need, which may differ from exact imitation.

**Schedule**

1. **Closed-form initialization.** On a calibration batch with everything running natively, fit the least-squares linear map from block input to real write for each layer, then truncate the difference from the original weights to rank 32 with SVD. A LoRA is exactly a low-rank linear correction, so this is the optimal starting point for the local objective.
2. **Pre-fit.** Train only the shadow LoRAs on the local loss for 10–20M tokens with the router off and everything native. This needs no Gumbel branches and is cheap.
3. **Joint training.** Both signals stay active while the compute budget drops.

### 2.7 Training objective

```
L = KL(native ‖ routed)                       # match the base model, every token
  + λ · (E[compute] − budget)                  # budget constraint, λ adjusted automatically
  + μ · L_shadow                               # shadow imitation, μ = 0.1
  [+ CE on verified reasoning traces]          # Track Q only
```

- **Teacher:** the same model with all adapters off and every block kept, run without gradients. No second copy of the weights is needed.
- **Expected compute** per token: `E[compute] = 2 + Σ_j (p_keep,j + 2·p_repeat,j)`, in block executions (the 2 counts B0 and B5).
- **Budget enforcement:** a Lagrange multiplier updated by dual ascent: `λ ← max(0, λ + η·(Ē − budget))` for Track E, two-sided for Track Q. The budget becomes a constraint rather than a weight to tune.

### 2.8 Efficiency ceiling at 0.8B

The output layer (248,320 × 1024, tied with the embeddings) reads about 0.5 GB per generated token in bf16, compared with about 1 GB for all six blocks. Even if every token skips all four routable blocks, and shadow writes still happen, bytes read per token fall by only about a third. The ceiling rises with model size, as the output layer becomes a smaller share of the weights.

---

## 3. Plan

### 3.1 Scope

- **Thinking mode is enabled throughout:** training data generation, training, and evaluation. Final models also get one evaluation pass with thinking disabled. That costs evaluation time only; no extra training runs.
- **No success thresholds are pre-registered.** The study is exploratory, and all results are reported.
- **Out of scope for now:**
  - KV-cache sharing between passes (§2.3)
  - Comparisons against other methods, such as static layer pruning with LoRA healing
  - A router-only arm with frozen layers (the frozen-model measurements in §3.3 serve as the no-adaptation reference)

### 3.2 Steps

| Step | Work | Output | Estimated time |
|---|---|---|---|
| 0 | Environment setup: PyTorch, transformers with Qwen3.5 support, flash-linear-attention and causal-conv1d (verify Blackwell support), vLLM. Load the model text-only, MTP off. Measure real generation and training throughput. | Working environment; real speed numbers that replace every estimate below | ~1 day |
| 1 | Build the modified model: blocks, 10-slot cache, shadow writes, plan execution in all three modes. Pass the core tests (§4.4). | A model that runs any routing plan | ~1 week |
| 2 | Build and freeze the evaluation set; score the base model | Frozen eval set and baselines | 2–4 days |
| 3 | Frozen-model measurements (§3.3) | Headroom estimates, per-token map, warm-up labels. **Decision point.** | A few days |
| 4 | Generate training data (§3.4) | Broad set, verified reasoning set | 1–2 days, overlaps step 5 |
| 5 | Build training: router, adapters, losses, schedules, shadow pre-fit; smoke test on a few million tokens | A validated training loop | ~1 week |
| 6 | Staged experiments (§3.6) | Trained routers and adapters | ~2 weeks of GPU time |
| 7 | Evaluation and analysis (§3.5) | Results | A few days |
| 8 | Rerun the winning settings on Qwen3.5-2B; then choose a larger model | Scaling evidence | 1–2 weeks |

Total: roughly 6–8 weeks for 0.8B through 2B. Most of the uncertainty is in step 1 (the most error-prone code) and in the real throughput measured in step 0.

### 3.3 Frozen-model measurements (step 3)

These need no training and establish whether the idea has room to work.

1. **All 81 plans, applied per sequence.** Run every plan on the evaluation set, with each plan applied to all tokens of a sequence (§2.3 explains why shadows then go unread). Compare the best plan per problem against the rate at which a random plan fixes a wrong answer by chance. Computation is shared across plans with a common prefix of block decisions. Plans form a 3-ary tree over four decisions, which cuts block executions from ~490 to ~200 per sequence for teacher-forced passes (~2.4×).
2. **Per-token map.** On teacher-forced sequences from the broad training set, record for every token the cheapest plan whose next-token distribution stays within ε of native (measured by `KL_t(native ‖ plan)`).
3. **Inspection.** Check which tokens come out cheap. If function words, closing brackets, and multi-token word continuations tolerate short plans while numbers and answer tokens want more compute, the per-token premise holds.

**Decision point:** if nearly all tokens need about the same compute, per-token routing has little to exploit, and the plan is revisited before any training.

**Warm-up labels.** From the per-token map, build soft labels over the 81 plans:

```
p_t(P) ∝ exp( −[ metric_t(P) + λ_c · (cost(P) − 6) ] / T )
```

- Metric: `KL_t(native ‖ P)` for Track E; cross-entropy on verified traces for Track Q, where plans may beat native.
- T is chosen so the best plan receives about half the probability mass on average.
- Labels are marginalized to per-block distributions over {skip, keep, repeat} to match the router's four heads.
- The router is pretrained on them with cross-entropy, adapters untouched.

**Known bias in these labels.** Each measurement uses one plan for every token, so a token's label assumes all earlier tokens took the same plan. This is pessimistic about skipping for easy tokens surrounded by hard ones, and optimistic about repeats. A more faithful "native-context" labeling would run one token along each plan while earlier tokens stay native. For attention that is straightforward; for DeltaNet it requires querying the native state just before each position (feasible by shifting queries by one position in the chunked kernel). It is built only if the warm start proves important.

### 3.4 Data

| Dataset | Contents | Size | Used by |
|---|---|---|---|
| **Broad self-generated set** | Qwen3.5-0.8B's own responses, thinking enabled, to diverse prompts (chat, code, math, general knowledge) | ~300M tokens | All training (distillation target); per-token map |
| **Verified reasoning set** | Nemotron-Math-v2 problems. Sample the base model 8 times each, keep problems solved 1–6 times, keep the correct traces | Depends on yield | Track Q correct-answer loss |
| **Warm-up set** | A few thousand sequences from the broad set, labeled via §3.3 | ~5k sequences | Router warm start |

- **Why self-generated data:** it matches the post-trained model's own distribution and how it is used at inference, which keeps the adapters' job small. The native model provides free, dense supervision on every token.
- **Why the base model's own correct traces rather than dataset-provided traces:** Nemotron-Math-v2 traces come from a much larger model (gpt-oss-120b). Imitating them would teach style rather than capability. Rejection sampling from the base model keeps traces in-distribution while verifying correctness.
- **Contamination:** Nemotron-Math-v2 (December 2025) predates Qwen3.5's small models (March 2026), and its problems derive from public forums (AoPS, Math StackExchange). It is used for training only; evaluation uses a separate clean set (§3.5). All training sets are deduplicated against the evaluation set.

### 3.5 Evaluation

**Clean evaluation set**

| Tier | Source | Size | Role |
|---|---|---|---|
| A | Procedurally generated problems, freshly instanced: algorithmic tasks (BeyondBench-style) and templated word problems with new numbers and entities (GSM-Symbolic-style) | 1,000–2,000 | Primary, statistically powered result |
| B | Human-written problems published after March 2026 at accessible difficulty (e.g., post-release LiveBench math, spring and summer 2026 contests) | 200–400 | Directional check on real problems |
| C | MathArena competitions after the model's release | As available | Larger models only |

Rules:

- **Calibrate difficulty at the generator level, not per item.** Tune generator settings so the base model (thinking enabled) averages 20–60% accuracy. Selecting individual items by base-model performance biases comparisons through regression to the mean.
- **Generate once with a fixed seed, fingerprint, and freeze** before any routing experiment.
- **Report tiers separately.**
- **Sample size:** detecting a ~4-point paired accuracy difference at ~80% power needs roughly 750–1,000 problems, which is why Tier A carries the main result.

**General-ability checks** (no degradation from routing): MMLU-Pro (knowledge), IFEval (instruction following), and perplexity on held-out ordinary text. These run with the router active, because per-token routing affects every token of every input.

**Efficiency metrics** (reported with every accuracy number):

1. **Blocks per token:** full block executions, from router decisions.
2. **Weight bytes read per token:** full visits count all block weights; shadow writes count only their projection matrices; the output layer is included. This is the best proxy for decoding speed, which is memory-bandwidth-bound.
3. **Measured decoding speed** at batch size 1. With one token at a time, skipped blocks simply do not run apart from their shadow writes, so plain PyTorch shows the real speedup without custom kernels. Prompt processing with mixed per-token paths would need custom kernels and is not measured.

**Diagnostics**

- **Perfect-cache check:** at evaluation, replace every shadow entry with the real entry (computed by actually running the block) and measure how much of the gap to the base model closes. This is an upper bound on what better shadow writes could gain.
- **Routing patterns:** which tokens skip, which repeat, and how that relates to base-model next-token entropy.
- **Shadow imitation error per layer:** expected to increase from layer 2 to layer 4.
- **Random-routing control:** random per-token plans at the same average compute, confirming the router learns something beyond compute allocation.

### 3.6 Experiments

The full cross of adapter variants × warm start × budget settings would be 18 runs times seeds. Experiments are staged instead, each stage using the previous stage's winner.

| Stage | Configurations | Settings | Question |
|---|---|---|---|
| 1. Adapter variant | V1, V2, V3 | Track E, budget 5, cold start, 100M tokens, 1 seed | Which adapter setup works |
| 2. Warm start | Best variant × {cold, warm} × {Track E @ 5, Track Q @ 6} | 300M tokens, 3 seeds each | Whether frozen-model labels help, and on which track |
| 3. Budget sweep | Best variant + winning initialization | Track E at budgets 4, 4.5, 5, 5.5 | The quality-versus-compute curve |

About 11 configurations in total. Stage 2 uses three seeds because the warm-start effect may be small.

**Comparing warm and cold starts**

- Final quality at matched budget, paired on the same evaluation items, across seeds
- Training tokens needed to reach a fixed distillation loss or accuracy
- Stability: collapse events (router going all-keep or all-skip) during the budget ramp
- Bias persistence: agreement between the warm-started router and its labels over training. High agreement combined with worse results than cold start means the label bias is sticking.

If warm and cold tie, the warm start is dropped and the frozen-model measurements remain purely diagnostic. If the warm start clearly helps, the native-context labeling (§3.3) is built and added as a third arm.

### 3.7 Default hyperparameters

| Setting | Default | Rationale |
|---|---|---|
| Router | 2-layer MLP, hidden 256, on B0 output; keep logit strongly favored at init | Training starts at the base model's behavior, so nothing breaks before adapters learn |
| Gumbel temperature | 1.0 → 0.3 over the first 60% of training, then held; hard samples in the forward pass throughout | High early gives smooth gradients and exploration; lower later aligns the gradient with hard inference-time choices; below ~0.2 gradients become noisy |
| Learning rates | LoRA 2e-4, router 5e-5; 2% warmup, cosine decay | Standard LoRA rate; slower router avoids locking in choices before adapters adapt |
| Router entropy bonus | 0.01, decaying to 0 by 30% of training | Prevents early collapse to all-keep or all-skip |
| Budget schedule | Track E: 6 → target linearly over the first 50%, then held. Track Q: fixed at 6 | Adapters and shadow writes need time before heavy skipping; the hold lets the model settle at the evaluated budget |
| Budget enforcement | Lagrange multiplier, dual ascent | Hits the target without hand-tuning a penalty weight |
| Distillation loss | KL(native ‖ routed) on every token, temperature 1 | Standard distillation; penalizes dropping probability the base model assigns |
| Track Q mix | Alternating 50/50 batches: CE on verified traces, KL on the broad set | CE drives improvement; KL anchors against drift. Adjust if general-ability checks slip |
| Shadow loss weight | 0.1, per-layer errors normalized to target scale | Guides without dominating; the end-to-end loss decides what matters |
| Full-visit LoRA | Rank 16 (α = 32) on all linear layers of the block | Common fine-tuning size; enough capacity for distribution-shift corrections without overwhelming routing |
| Shadow LoRA | Rank 32 on write projections only | Harder job (predicting one to three layers ahead) on small matrices, so higher rank is cheap |
| Sequence length / batch | 4,096 tokens; ~64k tokens per step | Thinking traces are long; ~4,600 steps per 300M-token run leaves room for the schedules |
| Warm-up label temperature | Top plan receives ~50% of probability mass on average | Soft enough for near-ties, sharp enough to be informative |
| Warm-start training | Router only, one epoch over the labels, learning rate 1e-3 | Supervised pretraining of a small fresh network |

### 3.8 LoRA capacity ladder

The claim that LoRA can stand in for from-scratch training is tested directly. If the gap between routed and native loss stops shrinking during training, move up one rung:

1. Rank 16 full-visit and rank 32 shadow adapters (the defaults)
2. Higher rank (64–256)
3. Unfreeze the routable blocks, with the KL-to-native loss as an anchor
4. Full uptraining, the regime used by Relaxed Recursive Transformers

**Token-budget check:** train V2 on a few hundred million tokens and plot loss on off-native visits against native loss. A gap that keeps shrinking means data is the bottleneck. One that plateaus early means capacity is.

### 3.9 Compute budget (single RTX 5090, 32 GB)

| Item | Estimate |
|---|---|
| Training cost | ~8 GFLOP per token: three branches per routable block, the teacher pass, the large output layer |
| Tokens per run | 300M for main runs; 100M for screening runs |
| Time per 300M-token run | ~12–24 hours, assuming 25–35% of peak throughput |
| Broad set generation | ~6 hours |
| Verified set generation | ~9 hours |
| Frozen-model measurements | Hours (warm-up labels) to several hours (81 plans on the evaluation set with thinking) |
| Full 0.8B experiment plan | ~2 weeks of GPU time |
| 2B follow-up | ~1–2 weeks |

These are estimates until the throughput measurements in step 0. Qwen3.5-0.8B and 2B fit comfortably in 32 GB. For the third rung: 4B is easy; 9B needs QLoRA and is tight with three-branch training, and its 32 layers form 8 blocks, 6 routable (729 plans instead of 81); 27B requires rented hardware.

### 3.10 Risks and open questions

| Risk | Mitigation or signal |
|---|---|
| Little per-token variation in required compute | Detected before training by the per-token map (§3.3) |
| Shadow writes too inaccurate, especially at layer 4 | Perfect-cache check; raise shadow rank; fallback is the sparse-cache approach with cache-hole-adaptation training (CHASE) |
| Router collapse | Keep-biased init, entropy bonus, slow router learning rate, budget schedule |
| Gap between soft training and hard inference | Temperature annealing; always evaluate with hard choices |
| Router input (B0 output) too weak a signal | Check correlation between router decisions and base-model next-token entropy before blaming training |
| LoRA capacity insufficient | Capacity ladder (§3.8) |
| Limited efficiency headroom at 0.8B | Output-layer ceiling (§2.8); expected to improve at 2B and beyond |
| Warm-up label bias | Cold-start arm; bias-persistence tracking |
| Throughput estimates off | Measured in step 0; screening runs at 100M tokens |

**Deferred decisions**

- Third-rung model (4B, 9B, or 27B), after the 0.8B and 2B results
- Specific Tier B sources and their licensing
- Lazy repeat-slot shadows: skip shadow writes into Bj′ until some token actually repeats block j (a compute optimization)
- Native-context warm-up labels, only if the warm start matters

---

## 4. Code

### 4.1 Guiding structure

The center of the codebase is **one model class that runs in three modes, sharing all core code**:

1. **Fixed plan:** one routing plan for every token of a sequence. Used for frozen-model measurements and testing.
2. **Hard per-token:** router argmax per token. Used for evaluation and inference.
3. **Gumbel:** all three branches computed per routable block, straight-through Gumbel selection. Used for training.

Data generation, measurements, training, and evaluation are separate packages around it.

### 4.2 Repository layout

```
llmro-router/
├── pyproject.toml              # dependencies, managed with uv
├── configs/                    # one YAML file fully describes each run
│   ├── model/                  #   base model, block grouping
│   ├── stage0/                 #   frozen-model measurement settings
│   ├── train/                  #   one file per experiment
│   └── eval/
├── src/routed/
│   ├── model/
│   │   ├── layers.py           # wraps each Qwen3.5 layer with run() and write_only()
│   │   ├── blocks.py           # groups layers into 6 blocks; runs a block as a unit
│   │   ├── cache.py            # static 10-slot cache: attention KV + DeltaNet states per slot
│   │   ├── plan.py             # routing plans; mapping from block visits to cache slots
│   │   └── routed_model.py     # full forward pass; the three execution modes
│   ├── router/
│   │   ├── router.py           # MLP with four 3-way heads
│   │   └── gumbel.py           # straight-through Gumbel sampling, temperature handling
│   ├── adapters/
│   │   ├── lora.py             # LoRA module, zero init, SVD init support
│   │   ├── jump.py
│   │   ├── repeat.py
│   │   ├── shadow.py
│   │   └── control.py          # adapters_disabled() context manager (teacher mode)
│   ├── data/
│   │   ├── procedural/         # Tier A evaluation generators
│   │   ├── selfgen.py          # broad self-generated set via vLLM
│   │   ├── verified.py         # sampling, verification, and filtering of Nemotron-Math-v2
│   │   ├── packing.py          # tokenization and packing into 4,096-token sequences
│   │   └── manifest.py         # dataset shards with generator version, model, seed
│   ├── stage0/
│   │   ├── enumerate.py        # all 81 plans with shared-prefix computation
│   │   └── token_map.py        # per-token cheapest plans; warm-up labels
│   ├── train/
│   │   ├── losses.py           # KL to native, budget (Lagrangian), shadow imitation, CE
│   │   ├── schedules.py        # temperature, budget, entropy-bonus schedules
│   │   ├── shadow_prefit.py    # least-squares + SVD init; shadow-only pre-fit
│   │   ├── warmstart.py        # router pretraining on warm-up labels
│   │   └── trainer.py          # loop; separate optimizer groups for router and adapters
│   └── eval/
│       ├── harness.py          # clean eval tiers; MMLU-Pro and IFEval via lm-evaluation-harness; perplexity
│       ├── generate.py         # batched generation for correctness evaluation
│       ├── efficiency.py       # blocks/token, bytes/token, batch-1 decoding speed
│       └── diagnostics.py      # perfect-cache check, routing patterns, shadow error
├── scripts/                    # thin CLI entry points; no logic
│   ├── generate_data.py
│   ├── build_eval.py
│   ├── run_stage0.py
│   ├── train.py
│   └── evaluate.py
├── tests/
└── experiments/                # run outputs; git-ignored except summaries
```

### 4.3 Design decisions

1. **Wrap the Hugging Face implementation; do not fork it.** Load the official Qwen3.5 weights and layer modules, and drive them from a custom forward loop that decides which layer runs and which cache slot it uses. The stock cache assumes one entry per physical layer, which repeats break, so `cache.py` replaces it entirely. The DeltaNet layer forward is likely reimplemented thinly so it takes and returns its recurrent and convolution state explicitly, while calling the same fast kernels.

2. **Every layer wrapper exposes two operations.**
   - `run(h, slot, positions) → (h_out, slot)`: full computation.
   - `write_only(h, slot, positions) → slot`: shadow write. Projections and state update only; the hidden state is untouched.

   Blocks and the routed model are built entirely from these two calls.

3. **The teacher is the same object.** `adapters_disabled()` plus the all-keep plan gives the base model: no second weight copy in memory, and no possibility of teacher drift.

4. **Correctness and speed are evaluated separately.**
   - Correctness: batched generation in which every token computes all branches and keeps the router's choice. Wasteful but correct, and fast enough for the full evaluation set.
   - Speed: batch-1 decoding with true skipping.

   vLLM cannot run the modified model, so it is used only to generate data from the base model.

5. **Runs are reproducible from a single file.** Each run stores its YAML config, git commit, random seeds, a metrics log, and a checkpoint containing **only the router and adapters** (tens of MB; base weights are never re-saved). Datasets are written as shards with a manifest. The evaluation set is frozen and fingerprinted.

### 4.4 Tests

These gate everything built on top of the model code.

| Test | Asserts |
|---|---|
| `test_native_equivalence` | The all-keep plan with adapters disabled reproduces the base model's logits (within floating-point tolerance). **The most important test; nothing else is built until it passes.** |
| `test_decode_consistency` | Token-by-token generation matches whole-sequence processing, for arbitrary per-token plans |
| `test_cache_slots` | Repeats and skips read and write the correct slots; repeat passes never touch first-pass slots |
| `test_shadow_first_layer` | Shadow writes at layer 1 of each block exactly match real writes |
| `test_uniform_plan_equivalence` | With one plan for all tokens, results match a per-sequence implementation with no shadow writes |
| `test_gumbel_branches` | Each branch computed in Gumbel mode matches the hard per-token mode for the same choice |
| `test_adapter_gating` | All adapter contributions are exactly zero on native visits |

### 4.5 Build order

| Plan step | Code |
|---|---|
| 1. Modified model | `model/`, `tests/` |
| 2. Evaluation set | `data/procedural/`, `eval/harness.py`, `eval/generate.py` |
| 3. Frozen-model measurements | `stage0/` |
| 4. Training data | `data/selfgen.py`, `data/verified.py`, `data/packing.py`, `data/manifest.py` |
| 5. Training | `router/`, `adapters/`, `train/` |
| 7. Evaluation | `eval/efficiency.py`, `eval/diagnostics.py` |

### 4.6 Tooling

- **Environment:** uv; PyTorch; transformers with Qwen3.5 support; flash-linear-attention and causal-conv1d for DeltaNet kernels
- **Data generation:** vLLM (base model only)
- **Evaluation:** lm-evaluation-harness with a custom model wrapper for MMLU-Pro and IFEval
- **Testing:** pytest
- **Experiment tracking:** Weights & Biases or a local logger writing `metrics.jsonl` per run

---

## Glossary

| Term | Meaning |
|---|---|
| **Block** | One group of four consecutive layers: three Gated DeltaNet layers followed by one full-attention layer. Qwen3.5-0.8B has six (B0–B5). |
| **Routable block** | B1–B4, which can be skipped, kept, or repeated per token. B0 and B5 always run once. |
| **Routing plan** | The four actions (skip, keep, repeat) for one token's routable blocks; 81 possibilities. |
| **Native** | Every block kept exactly once, i.e., the original model's computation. |
| **Off-native visit** | A block execution whose input differs from what the original model would give it: a repeat pass, or the first layer after a skipped block. |
| **Cache slot** | Storage for one block visit: one full-attention KV cache plus three DeltaNet recurrent and convolution states. The static design has ten. |
| **Full visit** | A block execution that updates the hidden state and writes its cache entries. |
| **Shadow write** | The cache entries a skipping token still writes, computed from its unchanged hidden state using only the cheap projections. |
| **Jump adapter** | LoRA on the first layer of a block entered right after a skipped block. |
| **Repeat adapter** | LoRA on a block's second pass. |
| **Shadow adapter** | LoRA on the write projections of layers 2–4, correcting shadow writes toward what a real visit would have written. |
| **Straight-through Gumbel** | Training trick for discrete choices: hard random samples in the forward pass, gradients of the smooth (softmax) version in the backward pass. |
| **Teacher** | The base model's predictions (adapters off, all blocks kept), used as the distillation target. |
| **Budget** | Target average number of block executions per token (native = 6). |
| **Warm start** | Pretraining the router on labels derived from frozen-model measurements before joint training. |
| **Track E / Track Q** | The efficiency goal (fewer blocks, same quality) and the quality goal (better answers, same compute). |

## References

- Bae et al., 2024. *Relaxed Recursive Transformers: Effective Parameter Sharing with Layer-wise LoRA.* arXiv:2410.20672.
- Bae et al., 2025. *Mixture-of-Recursions.*
- Elhoushi et al., 2024. *LayerSkip.*
- Geiping et al., 2025. *Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach* (Huginn). arXiv:2502.05171.
- Gromov et al., 2024. *The Unreasonable Ineffectiveness of the Deeper Layers.*
- Heakl et al., 2025. *DR.LLM.*
- Lad, Gurnee & Tegmark, 2024. *The Remarkable Robustness of LLMs: Stages of Inference?*
- Li et al., 2025. *CoLa: chain-of-layers test-time depth adaptation.*
- POLAR, ICML 2026. *Up-front layer-program prediction for pretrained LLMs.*
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
