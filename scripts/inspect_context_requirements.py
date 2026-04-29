#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from array import array
from collections import defaultdict
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _summary_totals(data_dir: Path | None, paths: list[Path]) -> dict[str, int]:
    if not data_dir:
        return {}
    summary_path = data_dir / "summary.json"
    if not summary_path.exists():
        return {}
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    counts = summary.get("counts") or {}
    return {path.stem: int(counts[path.stem]) for path in paths if path.stem in counts}


def _lengths(row: dict[str, Any]) -> tuple[int, int, int]:
    source = 1 + len(row.get("nodes", [])) + len(row.get("edges", []))
    target = len(row.get("target_tokens", []))
    sequence = source + 1 + 2 * target
    return source, target, sequence


def _percentile(sorted_values: array[int], q: float) -> int:
    if not sorted_values:
        return 0
    q = min(1.0, max(0.0, q))
    idx = int(round(q * (len(sorted_values) - 1)))
    return int(sorted_values[idx])


class LengthAccumulator:
    def __init__(self) -> None:
        self.source = array("I")
        self.target = array("I")
        self.sequence = array("I")
        self.max_record: dict[str, Any] = {}

    def add(self, source: int, target: int, sequence: int, record: dict[str, Any]) -> None:
        self.source.append(source)
        self.target.append(target)
        self.sequence.append(sequence)
        if not self.max_record or sequence > int(self.max_record["model_sequence_tokens_untruncated"]):
            self.max_record = {
                **record,
                "source_graph_tokens": source,
                "target_tokens": target,
                "model_sequence_tokens_untruncated": sequence,
            }

    def summarize(self, quantiles: list[float]) -> dict[str, Any]:
        source = array("I", self.source)
        target = array("I", self.target)
        sequence = array("I", self.sequence)
        source = array("I", sorted(source))
        target = array("I", sorted(target))
        sequence = array("I", sorted(sequence))
        examples = len(sequence)
        return {
            "examples": examples,
            "source_graph_tokens": {
                "max": int(source[-1]) if source else 0,
                "mean": float(sum(source) / examples) if examples else 0.0,
                "quantiles": {str(q): _percentile(source, q) for q in quantiles},
            },
            "target_tokens": {
                "max": int(target[-1]) if target else 0,
                "mean": float(sum(target) / examples) if examples else 0.0,
                "quantiles": {str(q): _percentile(target, q) for q in quantiles},
            },
            "model_sequence_tokens_untruncated": {
                "max": int(sequence[-1]) if sequence else 0,
                "mean": float(sum(sequence) / examples) if examples else 0.0,
                "quantiles": {str(q): _percentile(sequence, q) for q in quantiles},
            },
            "max_record": self.max_record,
        }


def _context_recommendation(summary: dict[str, Any], multiplier: float) -> dict[str, Any]:
    global_summary = summary["global"]
    max_source = int(global_summary["source_graph_tokens"]["max"])
    max_target = int(global_summary["target_tokens"]["max"])
    max_sequence = int(global_summary["model_sequence_tokens_untruncated"]["max"])
    recommended = int(max(1, round(max_sequence * multiplier)))
    return {
        "context_multiplier": float(multiplier),
        "max_source_tokens_required": max_source,
        "max_target_tokens_required": max_target,
        "max_model_sequence_tokens_untruncated": max_sequence,
        "recommended_max_seq_len": recommended,
        "recommended_model_config": {"model": {"max_seq_len": recommended}},
        "recommended_data_config": {
            "data": {
                "max_source_tokens": max_source,
                "max_target_tokens": max_target,
                "max_seq_len": recommended,
            }
        },
    }


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml

        text = yaml.safe_dump(payload, sort_keys=False)
    except Exception:
        text = json.dumps(payload, indent=2, sort_keys=True)
    path.write_text(text, encoding="utf-8")


def inspect_paths(paths: list[Path], split_totals: dict[str, int], quantiles: list[float], progress_every: int, context_multiplier: float) -> dict[str, Any]:
    global_acc = LengthAccumulator()
    by_split: dict[str, LengthAccumulator] = defaultdict(LengthAccumulator)
    by_dataset: dict[str, LengthAccumulator] = defaultdict(LengthAccumulator)
    errors = 0
    total_expected = sum(split_totals.values()) if split_totals else None
    with tqdm(total=total_expected, desc="context/all", unit="ex", position=0) as all_pbar:
        for path in paths:
            split = path.stem
            with path.open("r", encoding="utf-8") as handle:
                tqdm.write(f"[context] start {path} expected_examples={split_totals.get(split)}")
                with tqdm(total=split_totals.get(split), desc=f"context/{path.name}", unit="ex", position=1, leave=True) as split_pbar:
                    for line in handle:
                        if not line.strip():
                            continue
                        try:
                            row = json.loads(line)
                            source, target, sequence = _lengths(row)
                            dataset = str((row.get("metadata") or {}).get("source_dataset") or "unknown")
                            record = {
                                "id": row.get("id"),
                                "split": split,
                                "source_dataset": dataset,
                                "task": row.get("task"),
                            }
                            global_acc.add(source, target, sequence, record)
                            by_split[split].add(source, target, sequence, record)
                            by_dataset[dataset].add(source, target, sequence, record)
                        except Exception as exc:  # pragma: no cover - diagnostic path.
                            errors += 1
                            tqdm.write(f"[context] error {path}:{errors}: {exc!r}")
                        finally:
                            split_pbar.update(1)
                            all_pbar.update(1)
                        if len(global_acc.sequence) % progress_every == 0:
                            postfix = {
                                "max_seq": int(max(global_acc.sequence)) if global_acc.sequence else 0,
                                "max_source": int(max(global_acc.source)) if global_acc.source else 0,
                                "max_target": int(max(global_acc.target)) if global_acc.target else 0,
                                "datasets": len(by_dataset),
                                "errors": errors,
                            }
                            split_pbar.set_postfix(postfix, refresh=False)
                            all_pbar.set_postfix(postfix, refresh=False)
                tqdm.write(f"[context] done {path}")
    result = {
        "global": global_acc.summarize(quantiles),
        "by_split": {key: acc.summarize(quantiles) for key, acc in sorted(by_split.items())},
        "by_dataset": {key: acc.summarize(quantiles) for key, acc in sorted(by_dataset.items())},
        "errors": errors,
    }
    result["context_recommendation"] = _context_recommendation(result, context_multiplier)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect untruncated context lengths for graph JSONL data.")
    parser.add_argument("--data-dir", type=Path, help="Directory containing train.jsonl/val.jsonl/test.jsonl.")
    parser.add_argument("--path", action="append", type=Path, help="Specific JSONL path. May be repeated.")
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    parser.add_argument("--write-context-config", type=Path, help="Optional YAML override containing model.max_seq_len and data no-truncation caps.")
    parser.add_argument("--quantile", action="append", type=float, help="Quantile to report. May be repeated.")
    parser.add_argument("--progress-every", type=int, default=100000)
    parser.add_argument("--context-multiplier", type=float, default=2.0, help="Recommended context multiplier over the largest untruncated row.")
    args = parser.parse_args()

    paths: list[Path] = []
    if args.data_dir:
        for name in ("train.jsonl", "val.jsonl", "test.jsonl"):
            candidate = args.data_dir / name
            if candidate.exists():
                paths.append(candidate)
    if args.path:
        paths.extend(args.path)
    if not paths:
        raise SystemExit("Provide --data-dir or at least one --path")

    quantiles = args.quantile or [0.5, 0.9, 0.95, 0.99, 0.995, 0.999, 1.0]
    split_totals = _summary_totals(args.data_dir, paths)
    result = inspect_paths(paths, split_totals=split_totals, quantiles=quantiles, progress_every=args.progress_every, context_multiplier=args.context_multiplier)
    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    if args.write_context_config:
        rec = result["context_recommendation"]
        _write_yaml(
            args.write_context_config,
            {
                **rec["recommended_model_config"],
                **rec["recommended_data_config"],
                "context_audit": {
                    "context_multiplier": rec["context_multiplier"],
                    "max_model_sequence_tokens_untruncated": rec["max_model_sequence_tokens_untruncated"],
                    "max_source_tokens_required": rec["max_source_tokens_required"],
                    "max_target_tokens_required": rec["max_target_tokens_required"],
                },
            },
        )
    print(text)


if __name__ == "__main__":
    main()
