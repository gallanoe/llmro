"""Write meta.json and carve the held-out eval slice.

Held-out is taken as the TAIL of each base shard. Nothing is rewritten: the
tail is copied into heldout/ and meta.json records a shorter `usable` length
for each base shard. The loader must read `usable`, not the file length --
that is what keeps the eval slice unseen.

    uv run python build_meta.py
"""

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path("datasets/fineweb-edu-sample-10BT")
TOKENIZER = Path("tokenizer.json")
SEQ_LEN = 1024
VOCAB_SIZE = 24_576
HELDOUT_FRACTION = 0.001  # ~0.1% of base, tail of each shard


def npy_len(path: Path) -> int:
    """Read the row count from the .npy header without touching the data."""
    with open(path, "rb") as f:
        np.lib.format.read_magic(f)
        shape, _, _ = np.lib.format.read_array_header_1_0(f)
    return int(shape[0])


def main() -> None:
    tok_sha = hashlib.sha256(TOKENIZER.read_bytes()).hexdigest()

    base = sorted((ROOT / "base").glob("*.npy"))
    anneal = sorted((ROOT / "anneal").glob("*.npy"))
    if not base:
        raise SystemExit(f"no shards under {ROOT / 'base'}")

    heldout_dir = ROOT / "heldout"
    heldout_dir.mkdir(exist_ok=True)

    meta = {
        "tokenizer_sha256": tok_sha,
        "seq_len": SEQ_LEN,
        "vocab_size": VOCAB_SIZE,
        "dtype": "uint16",
        "eos_id": 0,
        "note": "base shards: read only the first `usable` tokens. The tail is heldout.",
        "base": {},
        "anneal": {},
        "heldout": {},
    }

    for p in base:
        n = npy_len(p)
        # round the split to a window boundary so no training window straddles it
        n_hold = max(SEQ_LEN + 1, int(n * HELDOUT_FRACTION))
        usable = ((n - n_hold) // SEQ_LEN) * SEQ_LEN

        src = np.load(p, mmap_mode="r")
        out = heldout_dir / p.name
        np.save(out, np.asarray(src[usable:]))

        meta["base"][p.name] = {"tokens": n, "usable": usable}
        meta["heldout"][p.name] = {"tokens": n - usable}

    for p in anneal:
        meta["anneal"][p.name] = {"tokens": npy_len(p)}

    meta["totals"] = {
        "base_usable": sum(v["usable"] for v in meta["base"].values()),
        "anneal": sum(v["tokens"] for v in meta["anneal"].values()),
        "heldout": sum(v["tokens"] for v in meta["heldout"].values()),
    }
    meta["totals"]["base_windows"] = sum(
        v["usable"] // SEQ_LEN for v in meta["base"].values()
    )

    (ROOT / "meta.json").write_text(json.dumps(meta, indent=2))

    t = meta["totals"]
    print(f"tokenizer sha256 : {tok_sha[:16]}...")
    print(f"base   (usable)  : {t['base_usable'] / 1e9:.3f}B  ({t['base_windows']:,} windows)")
    print(f"anneal           : {t['anneal'] / 1e9:.3f}B")
    print(f"heldout          : {t['heldout'] / 1e6:.1f}M")
    print(f"wrote {ROOT / 'meta.json'} and {len(base)} files under {heldout_dir}")


if __name__ == "__main__":
    main()
