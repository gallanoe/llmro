"""The chat format. Defined once, here -- tokenize_sft.py and chat.py both use it.

ChatML, with the special tokens reserved when the tokenizer was trained:

    <|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n{content}<|im_end|>\n ... <|endoftext|>

Loss mask (DESCRIPTION.md sec 9): 1 on assistant content and its <|im_end|>,
0 on everything else. The <|im_end|> is what teaches the model to stop; the
"<|im_start|>assistant\n" header is 0 because at inference we write it ourselves.

    uv run python template.py     # self-test + prints a masked example
"""

from pathlib import Path

from tokenizers import Tokenizer

TOKENIZER = Path(__file__).parent / "tokenizer.json"
ROLES = ("system", "user", "assistant")


class ChatTemplate:
    def __init__(self, tokenizer: Tokenizer):
        self.tok = tokenizer

        def special(name: str) -> int:
            i = tokenizer.token_to_id(name)
            if i is None:
                raise ValueError(
                    f"{name} missing from tokenizer -- wrong tokenizer.json?"
                )
            return i

        self.im_start = special("<|im_start|>")
        self.im_end = special("<|im_end|>")
        self.eos = special("<|endoftext|>")
        self._newline = self._encode("\n")
        self._headers = {r: self._header(r) for r in ROLES}  # constant; encode once

    @classmethod
    def load(cls, path: Path = TOKENIZER) -> "ChatTemplate":
        return cls(Tokenizer.from_file(str(path)))

    def _fast_batch_encode(self, text: list[str]) -> list[list[int]]:
        return [
            e.ids for e in self.tok.encode_batch_fast(text, add_special_tokens=False)
        ]

    def fast_batch_render(
        self, messages: list[list[dict]]
    ) -> list[tuple[list[int], list[int]]]:
        """render() over many conversations, with one tokenizer call for all of them.

        Same output as [render(c) for c in messages], ~17x faster: every message
        body goes to Rust in a single parallel batch instead of one call each.
        Pieces are still encoded separately, so mask boundaries stay exact.
        """
        bodies = iter(
            self._fast_batch_encode([m["content"] for c in messages for m in c])
        )

        out = []
        for convo in messages:
            ids: list[int] = []
            mask: list[int] = []
            for m in convo:
                head = self._headers[m["role"]]
                body = next(bodies) + [self.im_end]
                train = int(m["role"] == "assistant")
                ids += head + body + self._newline
                mask += [0] * len(head) + [train] * len(body) + [0] * len(self._newline)
            ids.append(self.eos)
            mask.append(0)
            out.append((ids, mask))
        return out

    def _encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False).ids

    def _header(self, role: str) -> list[int]:
        return [self.im_start] + self._encode(f"{role}\n")

    def render(self, messages: list[dict]) -> tuple[list[int], list[int]]:
        """A full conversation -> (ids, mask), same length. For training.

        Each piece is tokenized on its own and the ids concatenated. Tokenizing
        the joined string instead would let BPE merge across a role boundary,
        and then no token cleanly belongs to the header or the content.
        """
        self._check(messages)
        if messages[-1]["role"] != "assistant":
            raise ValueError(
                "render() needs a conversation ending in an assistant turn"
            )
        ids: list[int] = []
        mask: list[int] = []
        for m in messages:
            head = self._header(m["role"])
            body = self._encode(m["content"]) + [self.im_end]
            train = int(m["role"] == "assistant")
            ids += head + body + self._newline
            mask += [0] * len(head) + [train] * len(body) + [0] * len(self._newline)
        ids.append(self.eos)  # document separator when packed, as in pretraining
        mask.append(0)
        return ids, mask

    def prompt(self, messages: list[dict]) -> list[int]:
        """History up to a user turn -> ids ending in the assistant header. For inference.

        Same bytes as render() up to that point, so the model sees exactly the
        prefix it was trained on. Sample from here and stop at self.im_end.
        """
        self._check(messages)
        if messages[-1]["role"] == "assistant":
            raise ValueError("prompt() is for generating the next assistant turn")
        ids: list[int] = []
        for m in messages:
            ids += (
                self._header(m["role"])
                + self._encode(m["content"])
                + [self.im_end]
                + self._newline
            )
        return ids + self._header("assistant")

    def _check(self, messages: list[dict]) -> None:
        if not messages:
            raise ValueError("empty conversation")
        for i, m in enumerate(messages):
            if m["role"] not in ROLES:
                raise ValueError(f"message {i}: unknown role {m['role']!r}")
            if m["role"] == "system" and i != 0:
                raise ValueError(f"message {i}: system is only allowed first")

    def show(self, ids: list[int], mask: list[int]) -> str:
        """Decode with loss-carrying spans in [[...]] -- the eyeball test (sec 9)."""
        out, span, cur = [], [], None
        for i, m in zip(ids, mask):
            if m != cur and span:
                text = self.tok.decode(span, skip_special_tokens=False)
                out.append(f"[[{text}]]" if cur else text)
                span = []
            cur = m
            span.append(i)
        if span:
            text = self.tok.decode(span, skip_special_tokens=False)
            out.append(f"[[{text}]]" if cur else text)
        return "".join(out)


if __name__ == "__main__":
    t = ChatTemplate.load()
    convo = [
        {"role": "system", "content": "You are concise."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "It is 4."},
        {"role": "user", "content": "And 3+3?"},
        {"role": "assistant", "content": "6."},
    ]
    ids, mask = t.render(convo)
    dec = lambda x: t.tok.decode(x, skip_special_tokens=False)

    assert len(ids) == len(mask)
    expected = (
        "<|im_start|>system\nYou are concise.<|im_end|>\n"
        "<|im_start|>user\nWhat is 2+2?<|im_end|>\n"
        "<|im_start|>assistant\nIt is 4.<|im_end|>\n"
        "<|im_start|>user\nAnd 3+3?<|im_end|>\n"
        "<|im_start|>assistant\n6.<|im_end|>\n<|endoftext|>"
    )
    assert dec(ids) == expected, f"round trip:\n{dec(ids)!r}"

    p = t.prompt(convo[:-1])
    assert ids[: len(p)] == p, (
        "prompt() is not a prefix of render() -- train/inference mismatch"
    )

    trained = [i for i, m in zip(ids, mask) if m]
    assert dec(trained) == "It is 4.<|im_end|>6.<|im_end|>", dec(trained)

    batch = [convo, convo[1:3], convo[1:]]
    assert t.fast_batch_render(batch) == [t.render(c) for c in batch], (
        "fast_batch_render() disagrees with render()"
    )

    print(
        "ok: round trip, prompt is a prefix of render, mask covers exactly the replies\n"
    )
    print(t.show(ids, mask))
