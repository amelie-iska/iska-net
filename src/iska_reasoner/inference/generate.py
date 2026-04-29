from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from iska_reasoner.data.dataset import RandomOrderCollator, encode_example
from iska_reasoner.data.vocab import GraphVocab
from iska_reasoner.graph.orders import build_orders
from iska_reasoner.graph.schema import GraphExample
from iska_reasoner.models.random_order_tokengt import RandomOrderTokenGT, RandomOrderTokenGTConfig
from iska_reasoner.training.checkpointing import load_checkpoint
from iska_reasoner.utils.logging import get_device


def load_model_for_inference(checkpoint: str | Path, vocab_path: str | Path, device_name: str = "cuda") -> tuple[RandomOrderTokenGT, GraphVocab, torch.device]:
    vocab = GraphVocab.load(vocab_path)
    payload = torch.load(checkpoint, map_location="cpu")
    model_cfg = dict(payload.get("config", {}).get("model", {}))
    model_cfg["vocab_size"] = len(vocab.token_to_id)
    model = RandomOrderTokenGT(RandomOrderTokenGTConfig(**model_cfg))
    load_checkpoint(checkpoint, model)
    device = get_device(device_name)
    model.to(device)
    model.eval()
    return model, vocab, device


@torch.no_grad()
def complete_graph_tokens(
    model: RandomOrderTokenGT,
    vocab: GraphVocab,
    example: GraphExample,
    device: torch.device,
    max_steps: int = 16,
    sample: bool = False,
    temperature: float = 1.0,
    max_source_tokens: int = 128,
) -> list[str]:
    if not example.target_tokens:
        # Unknown target at inference: allocate generic slots to predict.
        example.target_tokens = ["<UNK>" for _ in range(max_steps)]
    example.decoder_orders = build_orders(example.target_tokens, seed=0, n_random=0)
    order = list(range(min(max_steps, len(example.target_tokens))))
    generated: list[str] = []

    for step in range(len(order)):
        tmp = GraphExample(
            id=example.id,
            task=example.task,
            nodes=example.nodes,
            edges=example.edges,
            target_tokens=generated + ["<UNK>"],
            metadata=example.metadata,
            decoder_orders=[list(range(len(generated) + 1))],
        )
        encoded = encode_example(tmp, vocab, list(range(len(generated) + 1)), max_source_tokens, max_steps)
        collator = RandomOrderCollator(vocab=vocab, max_source_tokens=max_source_tokens, max_target_tokens=max_steps, max_seq_len=model.cfg.max_seq_len, order_mode="first")
        batch = collator([tmp])
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        out = model(**{k: batch[k] for k in [
            "input_ids",
            "kind_ids",
            "slot_ids",
            "endpoint_ids",
            "identifier_ids",
            "source_numeric_features",
            "attention_mask",
            "causal_mask",
        ]})
        pos_mask = batch["kind_ids"].eq(4) & batch["attention_mask"]
        last_pos = torch.where(pos_mask[0])[0][-1]
        logits = out["logits"][0, last_pos] / max(temperature, 1e-6)
        if sample:
            token_id = torch.multinomial(torch.softmax(logits, dim=-1), 1).item()
        else:
            token_id = torch.argmax(logits).item()
        token = vocab.decode(token_id)
        if token in {"<PAD>", "<SEP>", "<POS>", "<GRAPH>"}:
            break
        generated.append(token)
    return generated
