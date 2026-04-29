from __future__ import annotations

import json
import random
import re
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from iska_reasoner.data.vocab import GraphVocab
from iska_reasoner.graph.schema import GraphExample, graph_source_numeric_features, graph_source_tokens
from iska_reasoner.topology import topology_feature_tensor


KIND_TO_ID = {
    "pad": 0,
    "special": 1,
    "node": 2,
    "edge": 3,
    "position": 4,
    "target": 5,
}


NUMERIC_NODE_TYPES = {
    "coordinate_3d",
    "molecule_property",
    "material_property",
    "audio_duration",
    "binding_affinity",
    "assay_value",
    "protein_coordinate",
    "ligand_coordinate",
    "temperature",
}


def extract_numeric_values(example: GraphExample, max_values: int) -> tuple[list[float], list[float]]:
    values: list[float] = []
    if max_values <= 0:
        return [], []
    for node in example.nodes:
        if node.type not in NUMERIC_NODE_TYPES and not any(isinstance(v, (int, float)) for v in node.features.values()):
            continue
        for value in node.features.values():
            if isinstance(value, (int, float)) and len(values) < max_values:
                values.append(float(value))
        if len(values) >= max_values:
            break
        for match in re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", node.value):
            values.append(float(match))
            if len(values) >= max_values:
                break
        if len(values) >= max_values:
            break
    mask = [1.0] * len(values)
    if len(values) < max_values:
        values.extend([0.0] * (max_values - len(values)))
        mask.extend([0.0] * (max_values - len(mask)))
    return values[:max_values], mask[:max_values]


@dataclass
class EncodedExample:
    token_ids: list[int]
    kind_ids: list[int]
    endpoint_ids: list[tuple[int, int]]
    identifier_ids: list[int]
    numeric_features: list[list[float]]
    slot_ids: list[int]
    labels: list[int]
    task: str
    example_id: str


class GraphJsonlDataset(Dataset[GraphExample]):
    def __init__(self, path: str | Path, preload: bool = False, transform: Callable[[GraphExample], GraphExample] | None = None):
        self.path = Path(path)
        self.preload = preload
        self.transform = transform
        self._handle = None
        self.examples: list[GraphExample] | None = None
        self.offsets: array[int] = array("Q")
        if preload:
            from iska_reasoner.utils.io import read_jsonl

            self.examples = [GraphExample.from_dict(row) for row in read_jsonl(self.path)]
        else:
            self._build_offsets()
        if len(self) == 0:
            raise ValueError(f"No examples in {self.path}")

    def _build_offsets(self) -> None:
        file_size = self.path.stat().st_size if self.path.exists() else 0
        show_progress = file_size >= 64 * 1024 * 1024
        with self.path.open("rb") as handle:
            with tqdm(
                total=file_size,
                desc=f"index/{self.path.name}",
                unit="B",
                unit_scale=True,
                disable=not show_progress,
                leave=False,
            ) as pbar:
                while True:
                    offset = handle.tell()
                    line = handle.readline()
                    if not line:
                        break
                    if line.strip():
                        self.offsets.append(offset)
                    pbar.update(len(line))

    def _row_at(self, index: int) -> dict[str, Any]:
        if self._handle is None or self._handle.closed:
            self._handle = self.path.open("rb")
        self._handle.seek(int(self.offsets[index]))
        line = self._handle.readline()
        return json.loads(line.decode("utf-8"))

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_handle"] = None
        return state

    def __del__(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle is not None and not handle.closed:
            handle.close()

    def __len__(self) -> int:
        return len(self.examples) if self.examples is not None else len(self.offsets)

    def __getitem__(self, index: int) -> GraphExample:
        if self.examples is not None:
            example = self.examples[index]
        else:
            example = GraphExample.from_dict(self._row_at(index))
        return self.transform(example) if self.transform is not None else example


def select_order(example: GraphExample, rng: random.Random, mode: str = "sample") -> list[int]:
    n = len(example.target_tokens)
    if n == 0:
        return []
    if not example.decoder_orders:
        return list(range(n))
    if mode == "first":
        return example.decoder_orders[0]
    if mode == "reverse":
        return list(reversed(example.decoder_orders[0]))
    return rng.choice(example.decoder_orders)


def encode_example(
    example: GraphExample,
    vocab: GraphVocab,
    order: list[int],
    max_source_tokens: int,
    max_target_tokens: int,
    max_seq_len: int | None = None,
) -> EncodedExample:
    source_tokens, source_kinds, endpoints, identifiers = graph_source_tokens(example)
    source_numeric_features = graph_source_numeric_features(example)
    if max_seq_len is not None:
        target_count = min(len(order), max_target_tokens)
        reserved = 1 + 2 * target_count
        source_budget = max(1, min(max_source_tokens, max_seq_len - reserved))
    else:
        source_budget = max_source_tokens
    source_tokens = source_tokens[:source_budget]
    source_kinds = source_kinds[:source_budget]
    endpoints = endpoints[:source_budget]
    identifiers = identifiers[:source_budget]
    source_numeric_features = source_numeric_features[:source_budget]

    token_ids = [vocab.encode(tok) for tok in source_tokens]
    kind_ids = [KIND_TO_ID.get(kind, KIND_TO_ID["special"]) for kind in source_kinds]
    endpoint_ids = list(endpoints)
    identifier_ids = list(identifiers)
    numeric_features = list(source_numeric_features)
    slot_ids = [0 for _ in token_ids]
    labels = [-100 for _ in token_ids]

    token_ids.append(vocab.encode("<SEP>"))
    kind_ids.append(KIND_TO_ID["special"])
    endpoint_ids.append((0, 0))
    identifier_ids.append(0)
    numeric_features.append([0.0] * 4)
    slot_ids.append(0)
    labels.append(-100)

    for reveal_rank, target_idx in enumerate(order[:max_target_tokens], start=1):
        token = example.target_tokens[target_idx]
        token_ids.append(vocab.encode("<POS>"))
        kind_ids.append(KIND_TO_ID["position"])
        endpoint_ids.append((0, 0))
        identifier_ids.append(0)
        numeric_features.append([0.0] * 4)
        slot_ids.append(min(target_idx + 1, max_target_tokens))
        labels.append(vocab.encode(token))

        token_ids.append(vocab.encode(token))
        kind_ids.append(KIND_TO_ID["target"])
        endpoint_ids.append((0, 0))
        identifier_ids.append(0)
        numeric_features.append([0.0] * 4)
        slot_ids.append(min(target_idx + 1, max_target_tokens))
        labels.append(-100)

    return EncodedExample(
        token_ids=token_ids,
        kind_ids=kind_ids,
        endpoint_ids=endpoint_ids,
        identifier_ids=identifier_ids,
        numeric_features=numeric_features,
        slot_ids=slot_ids,
        labels=labels,
        task=example.task,
        example_id=example.id,
    )


class RandomOrderCollator:
    def __init__(
        self,
        vocab: GraphVocab,
        max_source_tokens: int = 128,
        max_target_tokens: int = 64,
        max_seq_len: int = 256,
        max_numeric_targets: int = 0,
        order_mode: str = "sample",
        seed: int = 17,
    ):
        self.vocab = vocab
        self.max_source_tokens = max_source_tokens
        self.max_target_tokens = max_target_tokens
        self.max_seq_len = max_seq_len
        self.max_numeric_targets = max_numeric_targets
        self.order_mode = order_mode
        self.rng = random.Random(seed)

    def __call__(self, examples: list[GraphExample]) -> dict[str, Any]:
        encoded = [
            encode_example(
                ex,
                self.vocab,
                select_order(ex, self.rng, self.order_mode),
                self.max_source_tokens,
                self.max_target_tokens,
                self.max_seq_len,
            )
            for ex in examples
        ]
        seq_len = min(max(len(ex.token_ids) for ex in encoded), self.max_seq_len)
        batch = len(encoded)

        input_ids = torch.full((batch, seq_len), self.vocab.pad_id, dtype=torch.long)
        kind_ids = torch.zeros((batch, seq_len), dtype=torch.long)
        slot_ids = torch.zeros((batch, seq_len), dtype=torch.long)
        endpoint_ids = torch.zeros((batch, seq_len, 2), dtype=torch.long)
        identifier_ids = torch.zeros((batch, seq_len), dtype=torch.long)
        source_numeric_features = torch.zeros((batch, seq_len, 4), dtype=torch.float32)
        labels = torch.full((batch, seq_len), -100, dtype=torch.long)
        attention_mask = torch.zeros((batch, seq_len), dtype=torch.bool)
        topology_features = topology_feature_tensor(examples)
        numeric_targets = torch.zeros((batch, self.max_numeric_targets), dtype=torch.float32)
        numeric_mask = torch.zeros((batch, self.max_numeric_targets), dtype=torch.float32)

        for row, ex in enumerate(encoded):
            n = min(len(ex.token_ids), seq_len)
            input_ids[row, :n] = torch.tensor(ex.token_ids[:n], dtype=torch.long)
            kind_ids[row, :n] = torch.tensor(ex.kind_ids[:n], dtype=torch.long)
            slot_ids[row, :n] = torch.tensor(ex.slot_ids[:n], dtype=torch.long)
            endpoint_ids[row, :n] = torch.tensor(ex.endpoint_ids[:n], dtype=torch.long)
            identifier_ids[row, :n] = torch.tensor(ex.identifier_ids[:n], dtype=torch.long)
            source_numeric_features[row, :n] = torch.tensor(ex.numeric_features[:n], dtype=torch.float32)
            labels[row, :n] = torch.tensor(ex.labels[:n], dtype=torch.long)
            attention_mask[row, :n] = True
            values, value_mask = extract_numeric_values(examples[row], self.max_numeric_targets)
            if values:
                numeric_targets[row] = torch.tensor(values, dtype=torch.float32)
                numeric_mask[row] = torch.tensor(value_mask, dtype=torch.float32)

        # Causal mask: True means disallowed for nn.Transformer.
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        return {
            "input_ids": input_ids,
            "kind_ids": kind_ids,
            "slot_ids": slot_ids,
            "endpoint_ids": endpoint_ids,
            "identifier_ids": identifier_ids,
            "source_numeric_features": source_numeric_features,
            "labels": labels,
            "attention_mask": attention_mask,
            "causal_mask": causal_mask,
            "topology_features": topology_features,
            "numeric_targets": numeric_targets,
            "numeric_mask": numeric_mask,
            "tasks": [ex.task for ex in examples],
            "example_ids": [ex.id for ex in examples],
            "examples": examples,
        }
