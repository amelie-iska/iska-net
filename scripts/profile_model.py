#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from iska_reasoner.models.random_order_tokengt import RandomOrderTokenGT, RandomOrderTokenGTConfig
from iska_reasoner.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile local RandomOrderTokenGT parameter counts and memory.")
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--vocab-size", type=int, default=2048)
    args = parser.parse_args()
    cfg = load_config(args.config)
    model_cfg = dict(cfg["model"])
    model_cfg["vocab_size"] = args.vocab_size
    model = RandomOrderTokenGT(RandomOrderTokenGTConfig(**model_cfg))
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    dtype_bytes = 2 if torch.cuda.is_available() else 4
    metrics = {
        "profile/parameter_count": params,
        "profile/trainable_parameter_count": trainable,
        "profile/estimated_param_memory_mb": params * dtype_bytes / (1024**2),
        "profile/device": "cuda" if torch.cuda.is_available() else "cpu",
        "profile/config": model_cfg,
    }
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
