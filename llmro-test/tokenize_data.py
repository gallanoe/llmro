"""Tokenize FineWeb-Edu parquet shards into flat uint16 token streams.

Each document is routed to exactly one of three mutually exclusive pools:

    anneal   int_score >= 4          the sec 7 decay-phase mixture
    heldout  1/1000 of the rest      the sec 10 bits-per-byte eval slice
    base     everything else         the stable phase

Because the split happens here, no downstream loader needs a `usable` bound to
respect -- there is no way to leak the eval set by forgetting a field.

Documents stop being the unit on the way out: each pool becomes one flat
uint16 array per input shard, documents concatenated with EOS between them.
Training reads fixed windows that span document boundaries freely.

Writes meta.json at the end (token counts, tokenizer hash, heldout byte
count). Run this and nothing else.

    uv run python tokenize_data.py
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer
from tqdm import tqdm

TOKENIZER = Path("tokenizer.json")
PARQUET_DIR = Path("data/fineweb-edu/sample/10BT")
OUT = Path("datasets/fineweb-edu-sample-10BT")

BATCH_SIZE = 1_000
SEQ_LEN = 1024
VOCAB_SIZE = 24_576
EOS = 0
POOLS = ("base", "anneal", "heldout")

# ~1/HELDOUT_IN of *base* documents are reserved for eval. Carved from base
# rather than the whole corpus so the eval slice matches the distribution being
# trained on during the stable phase.
HELDOUT_IN = 1_000


def is_heldout(doc_id: str) -> bool:
    """Deterministic, order-independent assignment keyed on the document id.

    A counter would also be deterministic but would shift if shards were
    reordered or the job re-run partially; hashing the id will not.
    """
    h = hashlib.blake2b(doc_id.encode(), digest_size=8).digest()
    return int.from_bytes(h, "big") % HELDOUT_IN == 0


def npy_len(path: Path) -> int:
    """Row count from the .npy header, without reading the data."""
    with open(path, "rb") as f:
        np.lib.format.read_magic(f)
        shape, _, _ = np.lib.format.read_array_header_1_0(f)
    return int(shape[0])


def main() -> None:
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    shards = sorted(PARQUET_DIR.glob("*.parquet"))
    if not shards:
        raise SystemExit(f"no parquet under {PARQUET_DIR} -- run download_data.py")
    for pool in POOLS:
        (OUT / pool).mkdir(parents=True, exist_ok=True)

    def pack(docs: list[str]) -> np.ndarray:
        return np.concatenate(
            [t.ids + [EOS] for t in tokenizer.encode_batch_fast(docs)]
        )

    heldout_bytes = 0

    for i, path in enumerate(shards):
        f = pq.ParquetFile(path)
        parts = {p: [] for p in POOLS}
        n_batches = int(np.ceil(f.metadata.num_rows / BATCH_SIZE))

        for batch in tqdm(
            f.iter_batches(batch_size=BATCH_SIZE, columns=["text", "int_score", "id"]),
            total=n_batches,
            desc=f"shard {i}",
        ):
            split = {p: [] for p in POOLS}
            for text, score, doc_id in zip(
                batch["text"].to_pylist(),
                batch["int_score"].to_pylist(),
                batch["id"].to_pylist(),
            ):
                if score >= 4:
                    split["anneal"].append(text)
                elif is_heldout(doc_id):
                    split["heldout"].append(text)
                    heldout_bytes += len(text.encode("utf-8"))
                else:
                    split["base"].append(text)

            for pool in POOLS:
                if split[pool]:
                    parts[pool].append(pack(split[pool]))

        for pool in POOLS:
            arr = np.concatenate(parts[pool]).astype(np.uint16)
            np.save(OUT / pool / f"{i:03d}.npy", arr)
            print(f"  {pool:<8} {len(arr):>12,} tokens")

    # ---- metadata -------------------------------------------------------
    meta = {
        "tokenizer_sha256": hashlib.sha256(TOKENIZER.read_bytes()).hexdigest(),
        "seq_len": SEQ_LEN,
        "vocab_size": VOCAB_SIZE,
        "dtype": "uint16",
        "eos_id": EOS,
        "heldout_rule": f"1/{HELDOUT_IN} of base documents, by blake2b(id)",
    }
    for pool in POOLS:
        meta[pool] = {
            p.name: {"tokens": npy_len(p)} for p in sorted((OUT / pool).glob("*.npy"))
        }
    meta["totals"] = {p: sum(v["tokens"] for v in meta[p].values()) for p in POOLS}
    meta["totals"]["base_windows"] = meta["totals"]["base"] // SEQ_LEN

    heldout_tokens = meta["totals"]["heldout"]
    meta["heldout_totals"] = {
        "tokens": heldout_tokens,
        "bytes": heldout_bytes,
        "bytes_per_token": heldout_bytes / heldout_tokens,
    }

    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))

    t = meta["totals"]
    print()
    print(f"tokenizer sha256 : {meta['tokenizer_sha256'][:16]}...")
    for pool in POOLS:
        print(f"{pool:<9}        : {t[pool] / 1e9:8.4f}B tokens")
    print(f"base windows     : {t['base_windows']:,}")
    print(
        f"heldout bytes    : {heldout_bytes:,} "
        f"({meta['heldout_totals']['bytes_per_token']:.4f} bytes/token)"
    )
    print(f"wrote {OUT / 'meta.json'}")


if __name__ == "__main__":
    main()
