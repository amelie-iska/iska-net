from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from tqdm.auto import tqdm

from iska_reasoner.graph.schema import GraphExample, graph_source_tokens
from iska_reasoner.utils.io import read_jsonl, write_jsonl


SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<GRAPH>", "<SEP>", "<POS>"]


@dataclass
class GraphVocab:
    token_to_id: dict[str, int]

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<PAD>"]

    @property
    def unk_id(self) -> int:
        return self.token_to_id["<UNK>"]

    def encode(self, token: str) -> int:
        return self.token_to_id.get(token, self.unk_id)

    def decode(self, idx: int) -> str:
        inv = self.id_to_token
        return inv.get(int(idx), "<UNK>")

    @property
    def id_to_token(self) -> dict[int, str]:
        return {idx: tok for tok, idx in self.token_to_id.items()}

    def to_dict(self) -> dict[str, object]:
        return {"token_to_id": self.token_to_id}

    @classmethod
    def from_dict(cls, row: dict[str, object]) -> "GraphVocab":
        return cls(token_to_id={str(k): int(v) for k, v in dict(row["token_to_id"]).items()})

    def save(self, path: str | Path) -> None:
        write_jsonl(path, [self.to_dict()])

    @classmethod
    def load(cls, path: str | Path) -> "GraphVocab":
        rows = list(read_jsonl(path))
        if not rows:
            raise ValueError(f"Empty vocab file: {path}")
        return cls.from_dict(rows[0])


def example_tokens(example: GraphExample) -> Iterable[str]:
    source, _, _, _ = graph_source_tokens(example)
    yield from source
    yield "<SEP>"
    yield "<POS>"
    yield from example.target_tokens


def read_extra_tokens(paths: Iterable[str | Path] | None) -> list[str]:
    tokens: list[str] = []
    if not paths:
        return tokens
    for path_like in paths:
        path = Path(path_like)
        if not path.exists():
            continue
        if path.suffix == ".jsonl":
            for row in read_jsonl(path):
                if "token" in row:
                    tokens.append(str(row["token"]))
                elif "tokens" in row and isinstance(row["tokens"], list):
                    tokens.extend(str(tok) for tok in row["tokens"])
                elif "token_to_id" in row:
                    tokens.extend(str(tok) for tok in dict(row["token_to_id"]).keys())
        else:
            tokens.extend(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return tokens


def build_vocab(
    examples: Iterable[GraphExample],
    min_freq: int = 1,
    max_size: int | None = None,
    extra_tokens: Iterable[str] | None = None,
    total: int | None = None,
    progress_desc: str | None = None,
) -> GraphVocab:
    counts: Counter[str] = Counter()
    iterator = examples
    if progress_desc:
        iterator = tqdm(examples, total=total, desc=progress_desc, unit="ex")
    for example in iterator:
        counts.update(example_tokens(example))
    tokens = list(SPECIAL_TOKENS)
    protected = set(tokens)
    if extra_tokens:
        for token in extra_tokens:
            tok = str(token)
            if tok and tok not in protected:
                tokens.append(tok)
                protected.add(tok)
    candidates = [tok for tok, freq in counts.most_common() if freq >= min_freq and tok not in protected]
    if max_size is not None:
        candidates = candidates[: max(0, max_size - len(tokens))]
    tokens.extend(candidates)
    return GraphVocab(token_to_id={tok: idx for idx, tok in enumerate(tokens)})
