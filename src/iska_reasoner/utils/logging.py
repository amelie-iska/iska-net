from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .io import ensure_dir


def setup_logging(log_dir: str | Path | None = None, name: str = "iska_reasoner") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    if log_dir is not None:
        ensure_dir(log_dir)
        file_handler = logging.FileHandler(Path(log_dir) / "run.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(prefer: str = "cuda") -> torch.device:
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class WandbLogger:
    def __init__(self, cfg: dict[str, Any], run_name: str | None = None):
        self.enabled = bool(cfg.get("enabled", False))
        self._run = None
        if self.enabled:
            try:
                import wandb

                mode = os.environ.get("WANDB_MODE", cfg.get("mode", "online"))
                env_tags = _string_tags(os.environ.get("WANDB_TAGS", "").split(","))
                cfg_tags = _string_tags(cfg.get("tags") or [])
                self._run = wandb.init(
                    project=os.environ.get("WANDB_PROJECT", cfg.get("project", "iska-ugm")),
                    entity=os.environ.get("WANDB_ENTITY") or cfg.get("entity"),
                    name=run_name or os.environ.get("WANDB_NAME") or cfg.get("run_name"),
                    mode=mode,
                    group=os.environ.get("WANDB_GROUP") or cfg.get("group"),
                    job_type=os.environ.get("WANDB_JOB_TYPE") or cfg.get("job_type"),
                    tags=[*cfg_tags, *env_tags],
                    config=cfg.get("config"),
                    resume=os.environ.get("WANDB_RESUME", cfg.get("resume")),
                    id=os.environ.get("WANDB_RUN_ID") or cfg.get("id"),
                )
            except Exception as exc:  # pragma: no cover - online auth/network dependent
                self.enabled = False
                print(f"[wandb] disabled after init failure: {exc}")

    def log(self, metrics: dict[str, float], step: int) -> None:
        if self._run is not None:
            import wandb

            wandb.log(metrics, step=step)

    def finish(self) -> None:
        if self._run is not None:
            import wandb

            wandb.finish()


def _string_tags(tags: Any) -> list[str]:
    if isinstance(tags, str):
        tags = [tags]
    out: list[str] = []
    for tag in tags or []:
        if tag is None:
            continue
        text = str(tag).strip()
        if text:
            out.append(text)
    return out
