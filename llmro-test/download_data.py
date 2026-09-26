"""Download raw data: FineWeb-Edu shards for pretraining, SmolTalk2 for SFT.

Phase C, step 1 (FineWeb-Edu) and phase F (SmolTalk2). Neither depends on a
decision made elsewhere, so both can run while other work is in progress.

FineWeb-Edu sizing (DESCRIPTION.md sec 4): `sample-10BT` is 14 parquet shards,
28.5 GB, ~10B *GPT-2* tokens. A 24k BPE compresses worse than GPT-2's 50k --
roughly +10% tokens for the same text -- so a shard is worth ~786M of our
tokens, and the 5B-token budget needs ~6.4 shards. Default is 8 for margin
against the +/-15% uncertainty sec 4 flags on that conversion.

SmolTalk2 (DESCRIPTION.md sec 9): only the `no_think` SFT splits that fit a
104M model -- see SMOLTALK_SPLITS for what is in and why the rest is out.
~1.6 GB. The ~50k-example subsample happens at tokenize time, not here.

Downloads are resumable: hf_hub_download skips files already complete in the
local dir, so re-running after an interruption costs one HEAD per file.

    uv run python download_data.py                          # 8 shards -> ./data/fineweb-edu
    uv run python download_data.py --shards 14              # everything
    uv run python download_data.py --dataset smoltalk2      # SFT splits -> ./data/smoltalk2
    uv run python download_data.py --verify-only            # re-check what's on disk
    uv run python download_data.py --dataset smoltalk2 --verify-only
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

FINEWEB_REPO = "HuggingFaceFW/fineweb-edu"
FINEWEB_PREFIX = "sample/10BT"

# sec 4: "10BT" counts GPT-2 tokens. Ours run ~10% higher for the same text.
GPT2_TOKENS_TOTAL = 10_000_000_000
BPE_INFLATION = 1.10

SMOLTALK_REPO = "HuggingFaceTB/smoltalk2"
SMOLTALK_PREFIX = "SFT"

# Short, English, general-assistant conversations. Row counts as of 2026-09.
SMOLTALK_SPLITS = (
    "smoltalk_smollm3_smol_magpie_ultra_no_think",  # 407k  the bulk: general instruction following
    "smoltalk_smollm3_everyday_conversations_no_think",  # 2k  greetings / small talk
    "smoltalk_smollm3_systemchats_30k_no_think",  # 34k  follows a system prompt
    "tulu_3_sft_personas_instruction_following_no_think",  # 30k  constraint following
    "smoltalk_smollm3_smol_rewrite_no_think",  # 53k  rewrite tasks
    "smoltalk_smollm3_smol_summarize_no_think",  # 96k  summarization
    "smoltalk_smollm3_explore_instruct_rewriting_no_think",  # 30k  more rewriting
)
# Deliberately left out (sec 9, "Cut ruthlessly"):
#   OpenThoughts3_*, Mixture_of_Thoughts_*   distilled reasoning / math / code
#   OpenHermes_2.5_*                          small models underperform on it (sec 9)
#   LongAlign_*                               64k context; we train at 1024
#   smoltalk_multilingual_*                   non-English
#   hermes_function_calling_*, xlam_traces_*  tool use
#   table_gpt_*                               tabular niche
#   every *_think split                       reasoning traces
# The `Mid` config is not the instruction data sec 4 assumed for the anneal --
# it is reasoning traces only (Llama-Nemotron reasoning, OpenThoughts3).


def list_parquet(repo: str, prefix: str) -> list[str]:
    from huggingface_hub import HfApi

    files = sorted(
        f
        for f in HfApi().list_repo_files(repo, repo_type="dataset")
        if f.startswith(prefix) and f.endswith(".parquet")
    )
    if not files:
        sys.exit(f"no parquet files under {prefix} in {repo} -- did the repo layout change?")
    return files


def fineweb_files(n_shards: int) -> list[str]:
    files = list_parquet(FINEWEB_REPO, FINEWEB_PREFIX)
    if n_shards > len(files):
        sys.exit(f"asked for {n_shards} shards but only {len(files)} exist")
    return files[:n_shards]


def smoltalk_files() -> list[str]:
    files = list_parquet(SMOLTALK_REPO, SMOLTALK_PREFIX)
    picked = []
    for split in SMOLTALK_SPLITS:
        # Shards are named "<split>-00000-of-0000N.parquet". Match on the dash so
        # a split whose name prefixes another's can't pull in the wrong files.
        mine = [f for f in files if f.startswith(f"{SMOLTALK_PREFIX}/{split}-")]
        if not mine:
            sys.exit(f"split {split!r} not found in {SMOLTALK_REPO} -- did the repo layout change?")
        picked += mine
    return picked


def download(repo: str, files: list[str], out: Path, workers: int) -> list[Path]:
    from huggingface_hub import hf_hub_download

    def one(fname: str) -> Path:
        return Path(
            hf_hub_download(
                repo_id=repo,
                filename=fname,
                repo_type="dataset",
                local_dir=out,
            )
        )

    # Parallel across files; each file still shows its own progress bar.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, files))


def verify(paths: list[Path]) -> None:
    """Confirm each parquet is readable and report what we actually got.

    Reads footer metadata only -- no row groups -- so this is seconds, not
    minutes. A truncated download fails here rather than hours into
    tokenization.
    """
    import pyarrow.parquet as pq

    w = max(len(p.name) for p in paths)
    print(f"\n{'file':<{w}} {'size':>8} {'rows':>12}  columns")
    print("-" * (w + 50))
    total_rows = total_bytes = 0
    bad = []
    for p in paths:
        try:
            md = pq.ParquetFile(p).metadata
        except Exception as e:  # truncated, corrupt, or still partial
            bad.append((p.name, str(e).splitlines()[0][:50]))
            print(f"{p.name:<{w}} {'--':>8} {'UNREADABLE':>12}")
            continue
        size = p.stat().st_size
        total_rows += md.num_rows
        total_bytes += size
        cols = ", ".join(md.schema.names[:4])
        print(f"{p.name:<{w}} {size / 1e9:>7.2f}G {md.num_rows:>12,}  {cols}...")

    if bad:
        print("\nFAILED:")
        for name, err in bad:
            print(f"  {name}: {err}")
        sys.exit("re-run to resume the incomplete downloads")

    print("-" * (w + 50))
    print(f"{len(paths)} files  {total_bytes / 1e9:.1f} GB  {total_rows:,} rows")


def fineweb_estimate(n: int) -> None:
    frac = n / 14
    gpt2_tokens = GPT2_TOKENS_TOTAL * frac
    our_tokens = gpt2_tokens * BPE_INFLATION

    print(f"\nEstimated yield (sec 4 arithmetic, NOT measured):")
    print(f"  ~{gpt2_tokens / 1e9:.1f}B GPT-2 tokens  ->  ~{our_tokens / 1e9:.1f}B at 24k BPE")
    print(f"  budget is 5B: {'OK' if our_tokens > 5.5e9 else 'TIGHT -- consider more shards'}")
    print(
        "\nBoth numbers are estimates until you tokenize. The +/-15% on the "
        "inflation factor\nis the unverified part -- confirm against the real "
        "count after encoding shard 0."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=("fineweb-edu", "smoltalk2"), default="fineweb-edu")
    ap.add_argument("--shards", type=int, default=8, help="fineweb-edu only: how many of the 14 to pull (default: 8, ~17 GB)")
    ap.add_argument("--out", type=Path, default=None, help="default: data/<dataset>")
    ap.add_argument("--workers", type=int, default=4, help="parallel file downloads")
    ap.add_argument("--verify-only", action="store_true", help="check what is already on disk")
    args = ap.parse_args()

    fineweb = args.dataset == "fineweb-edu"
    repo = FINEWEB_REPO if fineweb else SMOLTALK_REPO
    prefix = FINEWEB_PREFIX if fineweb else SMOLTALK_PREFIX
    out = args.out or Path("data") / args.dataset
    out.mkdir(parents=True, exist_ok=True)

    if args.verify_only:
        have = sorted((out / prefix).glob("*.parquet"))
        if not have:
            sys.exit(f"nothing found under {out / prefix}")
        verify(have)
        if fineweb:
            fineweb_estimate(len(have))
        return

    files = fineweb_files(args.shards) if fineweb else smoltalk_files()
    print(f"{repo}  ->  {out}")
    print(f"{len(files)} files, {args.workers} at a time\n")

    paths = download(repo, files, out, args.workers)
    verify(paths)
    if fineweb:
        fineweb_estimate(len(paths))
        print(f"\nDelete the parquet only after your uint16 shards verify (sec 4).")


if __name__ == "__main__":
    main()
