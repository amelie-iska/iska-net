from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from iska_reasoner.utils.io import ensure_dir


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    step: int,
    config: dict[str, Any],
) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "step": step,
        "config": config,
    }
    torch.save(payload, path)


def load_checkpoint(path: str | Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None) -> int:
    payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["model"], strict=False)
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    return int(payload.get("step", 0))
