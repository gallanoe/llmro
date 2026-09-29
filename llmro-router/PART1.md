# Part 1: Fixing thinking-mode generation and building the training data

Sep 28, 2026 · @Eero Gallano

## Goal

Produce a large, clean corpus of Qwen3.5-0.8B's own outputs (about 300M tokens) for training small components added to the frozen model. The blocker: at this size, thinking-mode generation often never finishes and falls into verbatim loops. This task first finds generation settings that finish reliably, then generates the corpus, and optionally a verified reasoning set.

The corpus trains a per-token router and small MLPs that stand in for skipped blocks. The frozen stock model's next-token distributions are the training targets, so no labels are needed. What matters is that the corpus looks like what the model produces at inference: its own responses, with thinking enabled. Answer correctness matters only for the optional verified set.

## Background

- **Model:** Qwen3.5-0.8B from Hugging Face, loaded text-only (no vision encoder), bf16. It is a hybrid of Gated DeltaNet linear-attention layers and periodic full-attention layers. Thinking mode is enabled through the chat template (`enable_thinking=True`); the model writes its reasoning inside a thinking section, then the final answer.
- **Hardware:** one RTX 5090 (32 GB) with 128 GB system RAM.
- **Generation engine:** vLLM. Confirm it supports Qwen3.5's hybrid architecture; fall back to Hugging Face `generate` if not.

**What went wrong before** (a separate setup, thinking enabled, 8,192-token budget, roughly the model card's recommended thinking-mode sampling: temperature 0.6, top-p 0.95, top-k 20):

| Prompt set | Generations | Hit the token budget | Final answer correct |
| --- | --- | --- | --- |
| General prompts (chat, code, math, knowledge) | 150 | 78 | — |
| Math problems | 50 | 45 | 5 |

Generations that hit the budget ended in verbatim loops: a median of 93–100% of 20-token n-grams in their last 3,000 tokens repeated earlier text. Their first 2,048 tokens were mostly clean (median 1% repeated). Exact settings from that run weren't recorded, so re-measure the baseline here.

## Stage A: find generation settings that finish reliably

Pick the mildest settings that pass the criteria below; if none pass, stop and report, because the fallback is the project owner's decision.

**Calibration prompts** (fixed seed, never reused in the corpus):

- 200 math word problems with known answers: a mix at roughly GSM8K difficulty and MATH levels 1–3.
- 200 general prompts: chat, coding, explanations, open-ended writing.

**Metrics per configuration:**

| Metric | Definition |
| --- | --- |
| Finish rate | Fraction of generations that end on their own (end-of-sequence after the answer) before the token budget |
| Loop rate | Fraction flagged by the loop detector below |
| Math accuracy | Final answer matches the reference, checked with a math-equivalence checker (e.g. math-verify) |
| Length | Median and 90th-percentile generated tokens, thinking and answer separately |
| Throughput | Generated tokens per second on the 5090 |

**Loop detector:** a generation loops if at least 50% of the 20-token n-grams in any 1,024-token window (after the first 1,024 tokens) already occurred earlier in the same generation. Record the onset: the first window that crosses the threshold. Validate it by hand on 30 flagged and 30 unflagged generations and report false positives.

**Configurations,** one factor at a time from the baseline, then combine the best two:

1. Baseline: the model card's recommended thinking-mode settings, 8,192-token budget.
2. Presence penalty 0.5, 1.0, 1.5.
3. Repetition penalty 1.05, 1.10.
4. Temperature 0.6, 0.8, 1.0 (top-p 0.95).
5. Token budget 2,048, 4,096, 8,192, with the best settings so far.
6. Thinking disabled (the model card's non-thinking settings).
7. Reference: Qwen3.5-2B with the best 0.8B settings, to see whether the larger model finishes reliably.

**Pass criteria:** finish rate ≥ 90% and loop rate ≤ 5% on both prompt sets, with math accuracy no worse than the baseline. Penalties change the model's own distribution, so among passing settings prefer the smallest penalty and report its size.

**If nothing passes at 0.8B,** report the best configuration plus the numbers for three fallbacks, and stop:

- Truncate each generation at its loop onset and keep only the clean prefix.
- Generate with thinking disabled.
- Generate with Qwen3.5-2B.

## Stage B: generate the broad set

Generate about 300M tokens (prompt plus response) with the settings chosen in Stage A. Run a 30M-token pilot first and report throughput, filtering losses, and the category mix before committing to the full run.

**Prompts:**

- Use prompts only, never a dataset's own responses.
- Target mix: about 35% math and reasoning, 25% code, 25% general chat and instructions, 15% knowledge and explanatory writing.
- Sources: openly licensed instruction-prompt collections (for example, the prompts in NVIDIA's Nemotron post-training datasets). Record each source and its license.
- Deduplicate exactly and near-exactly (MinHash on prompt text).
- Remove prompts overlapping common evaluation sets (GSM8K test, MATH test, MMLU-Pro, IFEval), which may be used for evaluation later.

**Generation:** thinking enabled unless Stage A concluded otherwise, with the chosen sampling settings and budget.

**Filtering:**

- Keep generations that finish on their own.
- For generations the loop detector flags, keep the text before the loop onset if at least 256 response tokens remain; otherwise drop them.
- Cap each sample at 4,096 tokens (the training sequence length), keeping the first 4,096.
- Record every sample's flags (finished, looped, onset, truncated) rather than silently dropping them.

## Stage C (optional): verified reasoning set

Build this only after Stage B, and only if time allows. It supports a later quality-focused track, not the main training.

1. Take problems and reference answers from [Nemotron-Math-v2](https://huggingface.co/datasets/nvidia/Nemotron-Math-v2). Ignore its solution traces, which come from a much larger model.
2. Sample the base model 8 times per problem with the Stage A settings, and check each final answer against the reference.
3. Keep problems solved 1–6 times out of 8, and keep every correct trace for those problems.
4. Report problems attempted, the distribution of solve counts (0/8 through 8/8), the number of correct traces kept, and total tokens.

Nemotron-Math-v2 (released December 2025) predates Qwen3.5's small models (March 2026), so use it for training only, never for evaluation.

## Output format, storage, and splits

Write sharded Parquet (about 100k samples per shard) plus a manifest, and split by sample, never by token.

**Per-sample fields:**

| Field | Contents |
| --- | --- |
| `id` | Stable sample id |
| `source`, `category` | Prompt source and one of math, code, chat, knowledge |
| `messages` | The conversation in chat format |
| `token_ids` | The full sequence, tokenized with the model's chat template |
| `regions` | Token offsets for prompt, thinking section, and final answer |
| `settings_id` | Which generation configuration produced it |
| `finished`, `looped`, `loop_onset`, `truncated` | Filtering flags |
| `verified`, `correct` | Stage C only |

**Manifest** (`manifest.json`): model and tokenizer revisions, generation settings, prompt sources and licenses, seeds, sample and token counts per category and split, and vLLM and transformers versions.

**Splits:** 98% train, 1% validation, 1% test, stratified by category, fixed seed. Stage A's calibration prompts stay out of every split.

## Checks, acceptance, and deliverables

The task is done when Stage A has a passing configuration (or a documented failure with fallback numbers) and the broad set exists with its manifest and splits.

**Sanity checks (report each):**

- Decoding and re-tokenizing reproduces `token_ids`; report the mismatch rate.
- In every finished sample, the region offsets line up with the chat template's thinking-section markers.
- Loop-detector precision from the hand check in Stage A.
- Category mix within ±5 percentage points of the targets; token totals per category.
- No prompt appears in more than one split, and none overlaps the excluded evaluation sets.

**Acceptance:**

- A Stage A configuration meeting the pass criteria, or a stop with the three fallbacks measured.
- At least 250M tokens in the broad set after filtering (or the 30M pilot, if stopped there).

**Deliverables:**

1. Generation and filtering code, with unit tests for the loop detector.
2. A Stage A report in markdown: every configuration against every metric, the chosen settings, and a few loop examples before and after the fix.
3. The dataset shards, manifest, and split files.
4. The Stage C set and its report, if built.
