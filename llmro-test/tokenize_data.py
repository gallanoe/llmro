import pyarrow.parquet as pq
import numpy as np
from tqdm import tqdm
from tokenizers import Tokenizer, models, pre_tokenizers, decoders

tokenizer = Tokenizer.from_file("./tokenizer.json")

files = [
    pq.ParquetFile(f)
    for f in [
        "./data/fineweb-edu/sample/10BT/000_00000.parquet",
        "./data/fineweb-edu/sample/10BT/001_00000.parquet",
        "./data/fineweb-edu/sample/10BT/002_00000.parquet",
        "./data/fineweb-edu/sample/10BT/003_00000.parquet",
        "./data/fineweb-edu/sample/10BT/004_00000.parquet",
        "./data/fineweb-edu/sample/10BT/005_00000.parquet",
        "./data/fineweb-edu/sample/10BT/006_00000.parquet",
        "./data/fineweb-edu/sample/10BT/007_00000.parquet",
    ]
]

# Save as a single dataset file
BATCH_SIZE = 1_000

for i, f in enumerate(files):
    base, anneal = [], []
    n_batches = np.ceil(f.metadata.num_rows / BATCH_SIZE)
    for batch in tqdm(
        f.iter_batches(batch_size=BATCH_SIZE, columns=["text", "int_score"]),
        total=n_batches,
    ):
        ba, bb = [], []
        for text, score in zip(
            batch["text"].to_pylist(), batch["int_score"].to_pylist()
        ):
            if score >= 4:
                ba.append(text)
            else:
                bb.append(text)
        anneal.append(
            np.concatenate([t.ids + [0] for t in tokenizer.encode_batch_fast(ba)])
        )
        base.append(
            np.concatenate([t.ids + [0] for t in tokenizer.encode_batch_fast(bb)])
        )
    base = np.concatenate(base).astype(np.uint16)
    anneal = np.concatenate(anneal).astype(np.uint16)

    np.save(f"./datasets/fineweb-edu-sample-10BT/base/{i:03d}.npy", base)
    np.save(f"./datasets/fineweb-edu-sample-10BT/anneal/{i:03d}.npy", anneal)
