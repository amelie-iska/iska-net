#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm


def count_nonblank_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    total_bytes = path.stat().st_size
    count = 0
    with path.open("rb") as handle:
        with tqdm(total=total_bytes, desc=f"integrity/{path.name}", unit="B", unit_scale=True) as pbar:
            while True:
                line = handle.readline()
                if not line:
                    break
                if line.strip():
                    count += 1
                pbar.update(len(line))
    return count


def _expected_counts(summary: dict[str, Any]) -> dict[str, int]:
    counts = summary.get("counts")
    if isinstance(counts, dict):
        return {str(split): int(counts.get(split, 0) or 0) for split in ("train", "val", "test")}
    split_sizes = summary.get("split_sizes")
    if isinstance(split_sizes, dict):
        return {str(split): int(split_sizes.get(split, 0) or 0) for split in ("train", "val", "test")}
    return {split: int(summary.get(split, 0) or 0) for split in ("train", "val", "test")}


def inspect_integrity(data_dir: Path) -> dict[str, Any]:
    summary_path = data_dir / "summary.json"
    summary: dict[str, Any] = {}
    if summary_path.exists() and summary_path.stat().st_size:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = _expected_counts(summary)
    actual = {split: count_nonblank_jsonl(data_dir / f"{split}.jsonl") for split in ("train", "val", "test")}
    expected_total = int(summary.get("total") or sum(expected.values()))
    actual_total = sum(actual.values())
    mismatches = []
    for split in ("train", "val", "test"):
        if expected.get(split, 0) and expected[split] != actual[split]:
            mismatches.append({"split": split, "expected": expected[split], "actual": actual[split]})
    if expected_total and expected_total != actual_total:
        mismatches.append({"split": "total", "expected": expected_total, "actual": actual_total})
    return {
        "data_dir": str(data_dir),
        "summary_path": str(summary_path),
        "summary_exists": summary_path.exists(),
        "expected": expected,
        "actual": actual,
        "expected_total": expected_total,
        "actual_total": actual_total,
        "ok": not mismatches and actual_total > 0,
        "mismatches": mismatches,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify graph JSONL split counts against summary.json before training.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output")
    parser.add_argument("--warn-only", action="store_true")
    args = parser.parse_args()

    result = inspect_integrity(Path(args.data_dir))
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)
    if not result["ok"] and not args.warn_only:
        print(
            "Dataset integrity check failed. Rerun graphification to completion before training.",
            file=sys.stderr,
        )
        raise SystemExit(2)


if __name__ == "__main__":
    main()
