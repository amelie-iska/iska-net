#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch.utils.data import DataLoader

from iska_reasoner.data.dataset import GraphJsonlDataset, RandomOrderCollator
from iska_reasoner.data.vocab import GraphVocab
from iska_reasoner.inference.generate import load_model_for_inference
from iska_reasoner.utils.config import load_config
from iska_reasoner.validation.evaluate import evaluate_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a trained checkpoint.")
    parser.add_argument("--config", action="append", help="YAML config path. Values are used as defaults.")
    parser.add_argument("--checkpoint")
    parser.add_argument("--vocab")
    parser.add_argument("--data")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device")
    parser.add_argument("--max-batches", type=int)
    args = parser.parse_args()
    cfg = load_config(args.config) if args.config else {}
    val_cfg = cfg.get("validation", {})
    checkpoint = args.checkpoint or val_cfg.get("checkpoint")
    vocab_path = args.vocab or val_cfg.get("vocab")
    data_path = args.data or val_cfg.get("data_path") or cfg.get("data", {}).get("val_path") or cfg.get("data", {}).get("train_path")
    if not checkpoint or not vocab_path or not data_path:
        raise SystemExit("Provide --checkpoint/--vocab/--data or --config with validation/data paths")
    batch_size = int(args.batch_size or val_cfg.get("batch_size", 8))
    max_batches = args.max_batches if args.max_batches is not None else val_cfg.get("max_batches")
    max_batches = int(max_batches) if max_batches is not None else None
    device_name = args.device or val_cfg.get("device", "cuda")
    model, vocab, device = load_model_for_inference(checkpoint, vocab_path, device_name)
    dataset = GraphJsonlDataset(data_path)
    collator = RandomOrderCollator(vocab=vocab, order_mode="first", max_seq_len=model.cfg.max_seq_len, max_numeric_targets=model.cfg.numeric_dim)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
    hidden_topology_cfg = val_cfg.get("hidden_topology") or cfg.get("hidden_topology")
    metrics = evaluate_model(
        model,
        loader,
        device,
        prefix="validation/",
        max_batches=max_batches,
        hidden_topology_cfg=hidden_topology_cfg,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
