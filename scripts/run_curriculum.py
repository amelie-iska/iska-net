#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml
from tqdm.auto import tqdm


def _command_for_stage(stage: dict[str, Any], conda_env: str) -> list[str] | None:
    kind = stage.get("kind")
    if kind not in {"train", "ablation"} or not stage.get("configs"):
        return None
    cmd = ["conda", "run", "--no-capture-output", "-n", conda_env, "python", "scripts/train_stage.py"]
    for config in stage["configs"]:
        cmd.extend(["--config", str(config)])
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or dry-run an ordered UGM curriculum.")
    parser.add_argument("--curriculum", default="config/curriculum/ugm_sequence_first_curriculum.yaml")
    parser.add_argument("--start-at", default="00")
    parser.add_argument("--stop-after", default="99")
    parser.add_argument("--conda-env", default="tokengt")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-dir", default="logs/curriculum")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.curriculum).read_text(encoding="utf-8"))
    stages = cfg.get("curriculum", {}).get("stages", [])
    log_dir = Path(args.log_dir) / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    log_dir.mkdir(parents=True, exist_ok=True)
    status_path = log_dir / "status.jsonl"
    selected = [stage for stage in stages if str(args.start_at) <= str(stage.get("id")) <= str(args.stop_after)]
    for stage in tqdm(selected, desc="curriculum", unit="stage"):
        stage_id = str(stage.get("id"))
        name = str(stage.get("name"))
        cmd = _command_for_stage(stage, args.conda_env)
        event = {"stage_id": stage_id, "name": name, "kind": stage.get("kind"), "command": cmd, "dry_run": args.dry_run}
        if cmd is None:
            event["status"] = "skipped_no_command"
            status_path.open("a", encoding="utf-8").write(json.dumps(event, sort_keys=True) + "\n")
            continue
        if args.dry_run:
            event["status"] = "dry_run"
            print(" ".join(cmd))
            status_path.open("a", encoding="utf-8").write(json.dumps(event, sort_keys=True) + "\n")
            continue
        started = time.time()
        log_path = log_dir / f"{stage_id}_{name}.log"
        with log_path.open("w", encoding="utf-8") as handle:
            proc = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, text=True)
        event["seconds"] = round(time.time() - started, 3)
        event["status"] = "ok" if proc.returncode == 0 else f"failed:{proc.returncode}"
        event["log"] = str(log_path)
        status_path.open("a", encoding="utf-8").write(json.dumps(event, sort_keys=True) + "\n")
        if proc.returncode != 0:
            raise SystemExit(proc.returncode)
    print(json.dumps({"status_path": str(status_path), "stages": len(selected)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

