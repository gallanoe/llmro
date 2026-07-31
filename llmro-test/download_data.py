"""Download FineWeb-Edu `sample-10BT` shards for pretraining.

Phase C, step 1. This is off the critical path -- it depends on no decision
made elsewhere, so it should be running while the chat template and tokenizer
are being sorted out.

Sizing (DESCRIPTION.md sec 4): `sample-10BT` is 14 parquet shards, 28.5 GB, ~10B
*GPT-2* tokens. A 24k BPE compresses worse than GPT-2's 50k -- roughly +10%
tokens for the same text -- so a shard is worth ~786M of our tokens, and the
5B-token budget needs ~6.4 shards. Default is 8 for margin against the +/-15%
uncertainty sec 4 flags on that conversion.

Downloads are resumable: hf_hub_download skips files already complete in the
local dir, so re-running after an interruption costs one HEAD per shard.

    uv run python download_data.py                 # 8 shards -> ./data/fineweb-edu
    uv run python download_data.py --shards 14     # everything
    uv run python download_data.py --verify-only   # re-check what's on disk
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = "HuggingFaceFW/fineweb-edu"
PREFIX = "sample/10BT"

# sec 4: "10BT" counts GPT-2 tokens. Ours run ~10% higher for the same text.
GPT2_TOKENS_TOTAL = 10_000_000_000
BPE_INFLATION = 1.10


def shard_files(n_shards: int) -> list[str]:
    from huggingface_hub import HfApi

    files = sorted(
        f
        for f in HfApi().list_repo_files(REPO, repo_type="dataset")
        if f.startswith(PREFIX) and f.endswith(".parquet")
    )
    if not files:
        sys.exit(f"no parquet files under {PREFIX} in {REPO} -- did the repo layout change?")
    if n_shards > len(files):
        sys.exit(f"asked for {n_shards} shards but only {len(files)} exist")
    return files[:n_shards]


def download(files: list[str], out: Path, workers: int) -> list[Path]:
    from huggingface_hub import hf_hub_download

    def one(fname: str) -> Path:
        return Path(
            hf_hub_download(
                repo_id=REPO,
                filename=fname,
                repo_type="dataset",
                local_dir=out,
            )
        )

    # Parallel across shards; each file still shows its own progress bar.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, files))


def verify(paths: list[Path]) -> None:
    """Confirm each parquet is readable and report what we actually got.

    Reads footer metadata only -- no row groups -- so this is seconds, not
    minutes. A truncated download fails here rather than three hours into
    tokenization.
    """
    import pyarrow.parquet as pq

    print(f"\n{'shard':<28} {'size':>8} {'rows':>12}  columns")
    print("-" * 78)
    total_rows = total_bytes = 0
    bad = []
    for p in paths:
        try:
            md = pq.ParquetFile(p).metadata
        except Exception as e:  # truncated, corrupt, or still partial
            bad.append((p.name, str(e).splitlines()[0][:50]))
            print(f"{p.name:<28} {'--':>8} {'UNREADABLE':>12}")
            continue
        size = p.stat().st_size
        total_rows += md.num_rows
        total_bytes += size
        cols = ", ".join(md.schema.names[:4])
        print(f"{p.name:<28} {size / 1e9:>7.2f}G {md.num_rows:>12,}  {cols}...")

    if bad:
        print("\nFAILED:")
        for name, err in bad:
            print(f"  {name}: {err}")
        sys.exit("re-run to resume the incomplete downloads")

    n = len(paths)
    frac = n / 14
    gpt2_tokens = GPT2_TOKENS_TOTAL * frac
    our_tokens = gpt2_tokens * BPE_INFLATION

    print("-" * 78)
    print(f"{n} shards  {total_bytes / 1e9:.1f} GB  {total_rows:,} rows")
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
    ap.add_argument("--shards", type=int, default=8, help="how many of the 14 to pull (default: 8, ~17 GB)")
    ap.add_argument("--out", type=Path, default=Path("data/fineweb-edu"))
    ap.add_argument("--workers", type=int, default=4, help="parallel shard downloads")
    ap.add_argument("--verify-only", action="store_true", help="check what is already on disk")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    if args.verify_only:
        have = sorted((args.out / PREFIX).glob("*.parquet"))
        if not have:
            sys.exit(f"nothing found under {args.out / PREFIX}")
        verify(have)
        return

    files = shard_files(args.shards)
    print(f"{REPO}  ->  {args.out}")
    print(f"{len(files)} shards, ~{len(files) * 2.04:.0f} GB, {args.workers} at a time\n")

    paths = download(files, args.out, args.workers)
    verify(paths)

    print(f"\nDelete the parquet only after your uint16 shards verify (sec 4).")


if __name__ == "__main__":
    main()
