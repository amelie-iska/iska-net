#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from tqdm.auto import tqdm


STATIC_PREFIXES = (
    "ALL_ATOM_CARTESIAN:",
    "ALL_ATOM_CONTACT:",
    "CARTESIAN_ATOM:",
    "CARTESIAN_FRAME:",
    "CONTACT_PATCH:",
    "ESM_CONTACT:",
    "JACOBIAN_CONTACT:",
)

DYNAMICS_PREFIXES = (
    "TOKEN_MOTION:",
    "UMA_",
    "UMA:",
    "ORACLE:",
    "INTERNAL_COORD:",
    "ADAPTIVE_PATCH:",
    "AFFINITY_CONTACT:",
    "COMPLEX_CONTACT:",
    "PPI_CONTACT:",
)


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def _tokens(row: dict[str, Any]) -> list[str]:
    return [str(tok) for tok in row.get("target_tokens") or []]


def _has_prefix(row: dict[str, Any], prefixes: tuple[str, ...]) -> bool:
    return any(token.startswith(prefixes) for token in _tokens(row))


def _stable_score(row: dict[str, Any], salt: str) -> int:
    text = str(row.get("id") or row.get("metadata", {}).get("row_hash") or json.dumps(row, sort_keys=True)[:512])
    return int(hashlib.sha1(f"{salt}\t{text}".encode("utf-8")).hexdigest()[:16], 16)


def _select_rows(input_dir: Path, target_rows: int, mode: str) -> list[dict[str, Any]]:
    candidates: list[tuple[int, dict[str, Any]]] = []
    split_paths = [input_dir / "train.jsonl", input_dir / "val.jsonl", input_dir / "test.jsonl"]
    total_hint = sum(path.stat().st_size for path in split_paths if path.exists())
    with tqdm(total=total_hint or None, desc=f"subset/{mode}/scan", unit="B", unit_scale=True) as pbar:
        for path in split_paths:
            if not path.exists():
                continue
            before = 0
            with path.open("rb") as raw:
                for raw_line in raw:
                    before += len(raw_line)
                    try:
                        row = json.loads(raw_line.decode("utf-8"))
                    except Exception:
                        continue
                    static_ok = _has_prefix(row, STATIC_PREFIXES)
                    dynamics_ok = static_ok and _has_prefix(row, DYNAMICS_PREFIXES)
                    if mode == "static" and static_ok:
                        candidates.append((_stable_score(row, mode), row))
                    elif mode == "structure_dynamics" and dynamics_ok:
                        candidates.append((_stable_score(row, mode), row))
            pbar.update(before)
    candidates.sort(key=lambda item: item[0])
    return [row for _score, row in candidates[:target_rows]]


def _split_rows(rows: list[dict[str, Any]], val_ratio: float, test_ratio: float) -> dict[str, list[dict[str, Any]]]:
    splits = {"train": [], "val": [], "test": []}
    for row in rows:
        score = _stable_score(row, "split") / float(0xFFFFFFFFFFFFFFFF)
        if score < test_ratio:
            splits["test"].append(row)
        elif score < test_ratio + val_ratio:
            splits["val"].append(row)
        else:
            splits["train"].append(row)
    if rows and not splits["train"]:
        splits["train"].append(rows[0])
    return splits


def build_subset(input_dir: Path, output_dir: Path, target_rows: int, mode: str, val_ratio: float, test_ratio: float) -> dict[str, Any]:
    rows = _select_rows(input_dir, target_rows, mode)
    splits = _split_rows(rows, val_ratio, test_ratio)
    counts = {split: _write_jsonl(output_dir / f"{split}.jsonl", split_rows) for split, split_rows in splits.items()}
    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "mode": mode,
        "target_rows": target_rows,
        "selected_rows": len(rows),
        "split_counts": counts,
        "static_prefixes": STATIC_PREFIXES,
        "dynamics_prefixes": DYNAMICS_PREFIXES if mode == "structure_dynamics" else [],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build static-structure and structure-dynamics phase subsets from a curated graph corpus.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--static-output-dir", required=True)
    parser.add_argument("--dynamics-output-dir", required=True)
    parser.add_argument("--static-target-rows", type=int, default=25000)
    parser.add_argument("--dynamics-target-rows", type=int, default=2500)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--summary", default="")
    args = parser.parse_args()

    static_summary = build_subset(
        Path(args.input_dir),
        Path(args.static_output_dir),
        args.static_target_rows,
        "static",
        args.val_ratio,
        args.test_ratio,
    )
    dynamics_summary = build_subset(
        Path(args.input_dir),
        Path(args.dynamics_output_dir),
        args.dynamics_target_rows,
        "structure_dynamics",
        args.val_ratio,
        args.test_ratio,
    )
    summary = {"static_structure": static_summary, "structure_dynamics": dynamics_summary}
    if args.summary:
        Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
