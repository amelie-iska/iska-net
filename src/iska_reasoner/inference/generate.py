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
            coordinate_targets=example.coordinate_targets,
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


@torch.no_grad()
def predict_coordinate_records(
    model: RandomOrderTokenGT,
    vocab: GraphVocab,
    example: GraphExample,
    device: torch.device,
    target_tokens: list[str] | None = None,
    max_source_tokens: int = 128,
    max_target_tokens: int = 64,
) -> list[dict[str, Any]]:
    """Predict continuous coordinate means for already chosen COORD records.

    This helper does not choose the symbolic records. It reads an existing
    graph-token candidate, runs the same `<POS>` slots through the model, and
    returns the optional continuous coordinate head predictions for coordinate
    records. The ordinary autoregressive decoder therefore remains responsible
    for deciding whether a coordinate record exists and which frame/atom/axis it
    names.
    """
    if not bool(model.cfg.coordinate_head_enabled):
        return []
    tokens = list(target_tokens if target_tokens is not None else example.target_tokens)
    if not tokens:
        return []
    tmp = GraphExample(
        id=example.id,
        task=example.task,
        nodes=example.nodes,
        edges=example.edges,
        target_tokens=tokens,
        metadata=example.metadata,
        decoder_orders=[list(range(len(tokens)))],
        coordinate_targets=example.coordinate_targets,
    )
    collator = RandomOrderCollator(
        vocab=vocab,
        max_source_tokens=max_source_tokens,
        max_target_tokens=max_target_tokens,
        max_seq_len=model.cfg.max_seq_len,
        order_mode="first",
    )
    batch = collator([tmp])
    batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
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
    mean = out.get("coordinate_mean")
    logvar = out.get("coordinate_logvar")
    if mean is None or logvar is None:
        return []
    pos_indices = torch.where(batch["kind_ids"][0].eq(4) & batch["attention_mask"][0])[0]
    results: list[dict[str, Any]] = []
    for pos_idx, token in zip(pos_indices.tolist(), tokens[: len(pos_indices)], strict=False):
        if not token.startswith("COORD:"):
            continue
        xyz = mean[0, pos_idx].detach().cpu().tolist()
        sigma = torch.exp(0.5 * logvar[0, pos_idx]).mul(max(float(model.cfg.coordinate_target_scale), 1e-6)).detach().cpu().tolist()
        results.append(
            {
                "token": token,
                "x": float(xyz[0]),
                "y": float(xyz[1]),
                "z": float(xyz[2]),
                "sigma_x": float(sigma[0]),
                "sigma_y": float(sigma[1]),
                "sigma_z": float(sigma[2]),
            }
        )
    return results


@torch.no_grad()
def predict_uma_coordinate_frame(
    model: RandomOrderTokenGT,
    vocab: GraphVocab,
    example: GraphExample,
    device: torch.device,
    target_tokens: list[str] | None = None,
    max_source_tokens: int = 128,
    max_target_tokens: int = 64,
    max_uma_coordinate_atoms: int = 64,
) -> dict[str, Any]:
    """Predict one all-atom Cartesian frame from source-side UMA query slots.

    The frame comes from the optional continuous coordinate head reading
    `UMA_COORD_QUERY:*` source slots. These slots are derived from sequence,
    BioSELFIES/SELFIES, SMILES, or atom symbols and do not require structure
    labels in the input row.
    """
    if not bool(model.cfg.coordinate_head_enabled) or max_uma_coordinate_atoms <= 0:
        return {"atoms": [], "coordinates": [], "symbols": []}
    tokens = list(target_tokens if target_tokens is not None else example.target_tokens)
    if not tokens:
        tokens = ["<UNK>"]
    tmp = GraphExample(
        id=example.id,
        task=example.task,
        nodes=example.nodes,
        edges=example.edges,
        target_tokens=tokens,
        metadata=example.metadata,
        decoder_orders=[list(range(len(tokens)))],
        coordinate_targets=example.coordinate_targets,
    )
    collator = RandomOrderCollator(
        vocab=vocab,
        max_source_tokens=max_source_tokens,
        max_target_tokens=max_target_tokens,
        max_seq_len=model.cfg.max_seq_len,
        max_uma_coordinate_atoms=max_uma_coordinate_atoms,
        order_mode="first",
    )
    batch = collator([tmp])
    batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
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
    mean = out.get("coordinate_mean")
    if mean is None:
        return {"atoms": [], "coordinates": [], "symbols": []}
    mask = batch["uma_coordinate_query_mask"][0].to(dtype=torch.bool)
    indices = torch.where(mask)[0].tolist()
    symbols = list((batch.get("uma_coordinate_symbols") or [[]])[0])[: len(indices)]
    coords = mean[0, indices, :].detach().cpu().tolist()
    atoms = [
        {
            "element": str(symbol or "C"),
            "name": f"{str(symbol or 'C')[:2]}{idx + 1}",
            "residue": "UGM",
            "residue_index": 1,
        }
        for idx, symbol in enumerate(symbols)
    ]
    return {"atoms": atoms, "coordinates": [[float(v) for v in xyz[:3]] for xyz in coords], "symbols": symbols}
