#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iska_reasoner.utils.config import load_config
from iska_reasoner.utils.io import ensure_dir


DEFAULT_VARIANTS = [
    ("baseline", "config/train/graph_state_ablation_baseline_tiny.yaml"),
    ("topology", "config/train/graph_state_ablation_topology_tiny.yaml"),
    ("tropical", "config/train/graph_state_ablation_tropical_tiny.yaml"),
    ("topo_tropical", "config/train/graph_state_ablation_topo_tropical_tiny.yaml"),
]


def _last_metrics(metrics_path: Path) -> dict[str, Any]:
    if not metrics_path.exists():
        return {}
    last: dict[str, Any] = {}
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            last = json.loads(line)
    return last


def _variant_output_dir(configs: list[str]) -> Path:
    cfg = load_config(configs)
    return Path(cfg["run"]["output_dir"])


def run_variant(name: str, model_config: str, data_config: str, train_config: str, dry_run: bool = False) -> dict[str, Any]:
    configs = [model_config, data_config, train_config]
    output_dir = _variant_output_dir(configs)
    command = [sys.executable, "scripts/train_stage.py"]
    for config in configs:
        command.extend(["--config", config])
    if dry_run:
        return {"variant": name, "command": command, "output_dir": str(output_dir), "dry_run": True}
    started = time.perf_counter()
    result = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True)
    elapsed = time.perf_counter() - started
    metrics = _last_metrics(output_dir / "metrics.jsonl")
    return {
        "variant": name,
        "command": command,
        "output_dir": str(output_dir),
        "returncode": result.returncode,
        "elapsed_seconds": elapsed,
        "stdout_tail": result.stdout.splitlines()[-20:],
        "stderr_tail": result.stderr.splitlines()[-20:],
        "last_metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the graph-state topology/persistence/tropical ablation suite.")
    parser.add_argument("--model-config", default="config/model/tiny_tokengt.yaml")
    parser.add_argument("--data-config", default="config/data/synthetic_graphs.yaml")
    parser.add_argument("--variant", action="append", help="Variant name=config path. Defaults to the four tiny ablations.")
    parser.add_argument("--output", default="outputs/graph_state_ablation/summary.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    variants = DEFAULT_VARIANTS
    if args.variant:
        variants = []
        for item in args.variant:
            if "=" not in item:
                raise SystemExit("--variant must be name=config_path")
            name, path = item.split("=", 1)
            variants.append((name, path))

    results = []
    for name, train_config in tqdm(variants, desc="graph-state-ablation", unit="variant"):
        row = run_variant(name, args.model_config, args.data_config, train_config, dry_run=args.dry_run)
        results.append(row)
        if row.get("returncode", 0) != 0:
            break

    summary = {
        "model_config": args.model_config,
        "data_config": args.data_config,
        "variants": [name for name, _ in variants],
        "results": results,
        "quality_metrics": [
            "loss",
            "token_accuracy",
            "verifier/*",
            "gflownet/reward_mean",
        ],
        "diversity_metrics": [
            "tropical/logit_entropy",
            "gflownet/unique_terminal_states",
            "gflownet/action_entropy",
            "hidden_topology/distogram_entropy_mean",
        ],
        "resource_metrics": [
            "elapsed_seconds",
            "grad_norm",
            "lr",
        ],
    }
    output = Path(args.output)
    ensure_dir(output.parent)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    failures = [row for row in results if row.get("returncode", 0) != 0]
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
