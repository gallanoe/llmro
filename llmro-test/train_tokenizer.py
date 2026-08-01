import pyarrow.parquet as pq
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors

# CHECKED BEFOREHAND
n_samples = 726_000

tokenizer_shard = pq.ParquetFile("./data/fineweb-edu/sample/10BT/000_00000.parquet")


def create_iterator(file: pq.ParquetFile, batch_size=1000):
    def iterator():
        for batch in file.iter_batches(batch_size=batch_size, columns=["text"]):
            for text in batch["text"].to_pylist():
                yield text

    return iterator


tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()
# tokenizer.post_processor = processors.TemplateProcessing(
#     single="$A <|endoftext|>", special_tokens=[("<|endoftext|>", eos_id)]
# )
trainer = trainers.BpeTrainer(
    vocab_size=24_576,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    special_tokens=["<|endoftext|>", "<|im_start|>", "<|im_end|>"],
)
tokenizer.train_from_iterator(
    create_iterator(tokenizer_shard)(), trainer=trainer, length=n_samples
)
tokenizer.save("tokenizer.json")
