#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any


def _enabled(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _float_or_none(value: str | None) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _event_metrics(args: argparse.Namespace) -> dict[str, Any]:
    try:
        stage_index = int(args.stage_id)
    except ValueError:
        stage_index = -1
    status_ok = 1.0 if args.status in {"", "start", "ok", "dry_run"} else 0.0
    metrics: dict[str, Any] = {
        "runner/stage_index": stage_index,
        "runner/event_count": 1.0,
        f"runner/events/{args.event}": 1.0,
        "runner/status_ok": status_ok,
    }
    seconds = _float_or_none(args.seconds)
    if seconds is not None:
        metrics["runner/seconds"] = seconds
        metrics[f"runner/{args.event}_seconds"] = seconds
    if args.command_index is not None:
        metrics["runner/command_index"] = float(args.command_index)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Log full-runner stage and command events to W&B.")
    parser.add_argument("--stage-id", required=True)
    parser.add_argument("--stage-name", required=True)
    parser.add_argument("--event", required=True)
    parser.add_argument("--status", default="")
    parser.add_argument("--seconds", default="")
    parser.add_argument("--run-dir", default=os.environ.get("RUN_DIR", ""))
    parser.add_argument("--command", default="")
    parser.add_argument("--command-index", type=int)
    args = parser.parse_args()

    if not _enabled(os.environ.get("WANDB_ENABLED", "0")):
        return

    os.environ.setdefault("WANDB_SILENT", "true")
    os.environ.setdefault("WANDB_CONSOLE", "off")

    try:
        import wandb
    except Exception as exc:  # pragma: no cover - depends on optional install
        print(f"[wandb] unavailable for runner event logging: {exc}")
        return

    project = os.environ.get("WANDB_PROJECT", "iska-ugm")
    entity = os.environ.get("WANDB_ENTITY") or None
    mode = os.environ.get("WANDB_MODE", "online")
    group = os.environ.get("WANDB_GROUP") or os.environ.get("RUN_ID") or "full-training"
    run_id = os.environ.get("WANDB_RUN_ID") or f"{os.environ.get('RUN_ID', 'local')}-runner"
    name = os.environ.get("WANDB_NAME") or f"{group}-runner"
    tags = [tag for tag in os.environ.get("WANDB_TAGS", "full-runner,shell-stage").split(",") if tag]

    try:
        init_kwargs = {
            "project": project,
            "entity": entity,
            "id": run_id,
            "name": name,
            "group": group,
            "job_type": "runner",
            "mode": mode,
            "tags": tags,
            "settings": wandb.Settings(silent=True),
        }
        if mode != "offline":
            init_kwargs["resume"] = "allow"
        run = wandb.init(
            **init_kwargs,
        )
        metrics = _event_metrics(args)
        wandb.log(metrics)
        summary = {
            "last_event": args.event,
            "last_stage_id": args.stage_id,
            "last_stage_name": args.stage_name,
            "last_status": args.status,
            "last_event_unix": time.time(),
        }
        if args.run_dir:
            summary["run_dir"] = str(Path(args.run_dir))
        if args.command:
            summary["last_command"] = args.command[:500]
        run.summary.update(summary)
        wandb.finish()
    except Exception as exc:  # pragma: no cover - online auth/network dependent
        print(f"[wandb] runner event skipped: {exc}")


if __name__ == "__main__":
    main()
