#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tqdm.auto import tqdm

from iska_reasoner.data.splits import SplitReport, assign_split_for_policy
from iska_reasoner.graph.schema import GraphExample
from iska_reasoner.utils.io import ensure_dir, read_jsonl


def _summary_totals(input_dir: Path) -> dict[str, int]:
    summary_path = input_dir / "summary.json"
    if not summary_path.exists():
        return {}
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    counts = summary.get("counts") or summary.get("split_sizes") or {}
    return {str(key): int(value) for key, value in counts.items() if isinstance(value, int)}


def resplit(
    input_dir: Path,
    output_dir: Path,
    val_ratio: float,
    test_ratio: float,
    split_policy: str,
    progress_every: int,
) -> dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    input_paths = [path for path in [input_dir / "train.jsonl", input_dir / "val.jsonl", input_dir / "test.jsonl"] if path.exists()]
    totals = _summary_totals(input_dir)
    expected_total = sum(totals.get(path.stem, 0) for path in input_paths) or None
    handles = {
        "train": (output_dir / "train.jsonl").open("w", encoding="utf-8"),
        "val": (output_dir / "val.jsonl").open("w", encoding="utf-8"),
        "test": (output_dir / "test.jsonl").open("w", encoding="utf-8"),
    }
    counts = {"train": 0, "val": 0, "test": 0}
    invalid = 0
    report = SplitReport(policy=split_policy)
    try:
        with tqdm(total=expected_total, desc=f"resplit/{input_dir.name}", unit="ex") as pbar:
            for input_path in input_paths:
                for row in read_jsonl(input_path):
                    try:
                        example = GraphExample.from_dict(row)
                    except Exception:
                        invalid += 1
                        pbar.update(1)
                        continue
                    split, group_key = assign_split_for_policy(example, split_policy, val_ratio, test_ratio)
                    example.metadata.setdefault("curation", {})
                    example.metadata["curation"]["split_policy"] = split_policy
                    example.metadata["curation"]["split_group_key"] = group_key
                    handles[split].write(json.dumps(example.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
                    counts[split] += 1
                    report.add(split, group_key)
                    pbar.update(1)
                    total = sum(counts.values()) + invalid
                    if progress_every and total % progress_every == 0:
                        pbar.set_postfix({"invalid": invalid, **counts}, refresh=False)
    finally:
        for handle in handles.values():
            handle.close()

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "split_policy": split_policy,
        "counts": counts,
        "split_sizes": counts,
        "total": sum(counts.values()),
        "invalid_rows": invalid,
        "split_report": report.to_dict(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Resplit graph JSONL files with row-hash or entity-aware split keys.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--val-ratio", type=float, default=0.01)
    parser.add_argument("--test-ratio", type=float, default=0.01)
    parser.add_argument("--split-policy", choices=["row_hash", "entity"], default="entity")
    parser.add_argument("--progress-every", type=int, default=10000)
    args = parser.parse_args()
    summary = resplit(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        split_policy=args.split_policy,
        progress_every=args.progress_every,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
