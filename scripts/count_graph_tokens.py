#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iska_reasoner.graph.schema import GraphExample, graph_source_tokens


def _new_counts() -> dict[str, Any]:
    return {
        "examples": 0,
        "source_graph_tokens": 0,
        "target_tokens": 0,
        "separator_tokens": 0,
        "position_query_tokens": 0,
        "target_reveal_tokens": 0,
        "model_sequence_tokens_untruncated": 0,
        "supervised_prediction_tokens": 0,
    }


def _add_counts(dst: dict[str, Any], src: dict[str, int]) -> None:
    for key, value in src.items():
        dst[key] = int(dst.get(key, 0)) + int(value)


def _count_example(row: dict[str, Any]) -> dict[str, int]:
    example = GraphExample.from_dict(row)
    source_tokens, _, _, _ = graph_source_tokens(example)
    target_count = len(example.target_tokens)
    return {
        "examples": 1,
        "source_graph_tokens": len(source_tokens),
        "target_tokens": target_count,
        "separator_tokens": 1,
        "position_query_tokens": target_count,
        "target_reveal_tokens": target_count,
        "model_sequence_tokens_untruncated": len(source_tokens) + 1 + 2 * target_count,
        "supervised_prediction_tokens": target_count,
    }


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


def _line_totals(paths: list[Path]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for path in paths:
        with path.open("rb") as handle:
            totals[path.stem] = sum(1 for _ in handle)
    return totals


def count_paths(paths: list[Path], split_totals: dict[str, int] | None = None, progress_every: int = 100000) -> dict[str, Any]:
    summary = _new_counts()
    by_split: dict[str, dict[str, Any]] = {}
    by_task: dict[str, dict[str, Any]] = defaultdict(_new_counts)
    by_dataset: dict[str, dict[str, Any]] = defaultdict(_new_counts)
    errors: Counter[str] = Counter()
    split_totals = split_totals or {}
    global_total = sum(split_totals.values()) if split_totals else None

    with tqdm(total=global_total, desc="count/all", unit="ex", position=0) as global_pbar:
        for path in paths:
            split = path.stem
            by_split.setdefault(split, _new_counts())
            tqdm.write(f"[count] start {path} expected_examples={split_totals.get(split)}")
            with path.open("r", encoding="utf-8") as handle:
                with tqdm(total=split_totals.get(split), desc=f"count/{path.name}", unit="ex", position=1, leave=True) as split_pbar:
                    for line in handle:
                        if not line.strip():
                            continue
                        try:
                            row = json.loads(line)
                            counts = _count_example(row)
                        except Exception as exc:  # pragma: no cover - diagnostics path.
                            errors[f"{path}:{exc.__class__.__name__}"] += 1
                            split_pbar.update(1)
                            global_pbar.update(1)
                            continue
                        _add_counts(summary, counts)
                        _add_counts(by_split[split], counts)
                        task = str(row.get("task") or "unknown")
                        _add_counts(by_task[task], counts)
                        dataset = str((row.get("metadata") or {}).get("source_dataset") or "unknown")
                        _add_counts(by_dataset[dataset], counts)
                        split_pbar.update(1)
                        global_pbar.update(1)
                        if int(summary["examples"]) % progress_every == 0:
                            postfix = {
                                "source": int(summary["source_graph_tokens"]),
                                "target": int(summary["target_tokens"]),
                                "seq": int(summary["model_sequence_tokens_untruncated"]),
                                "datasets": len(by_dataset),
                                "errors": sum(errors.values()),
                            }
                            split_pbar.set_postfix(postfix, refresh=False)
                            global_pbar.set_postfix(postfix, refresh=False)
            tqdm.write(f"[count] done {path}: {by_split[split]}")

    return {
        **summary,
        "by_split": dict(sorted(by_split.items())),
        "by_task": dict(sorted(by_task.items())),
        "by_dataset": dict(sorted(by_dataset.items())),
        "errors": dict(errors),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Count graph/source/target tokens in graph JSONL data.")
    parser.add_argument("--data-dir", type=Path, help="Directory containing train.jsonl/val.jsonl/test.jsonl.")
    parser.add_argument("--path", action="append", type=Path, help="Specific JSONL path. May be repeated.")
    parser.add_argument("--output", type=Path, help="Optional JSON summary output path.")
    parser.add_argument("--progress-every", type=int, default=100000)
    parser.add_argument(
        "--max-model-sequence-tokens-total",
        type=int,
        help="Fail after writing the summary if untruncated model-sequence tokens exceed this budget.",
    )
    parser.add_argument(
        "--line-counts",
        action="store_true",
        help="Pre-count JSONL lines when no summary.json is available, so progress bars have totals.",
    )
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

    split_totals = _summary_totals(args.data_dir, paths)
    if args.line_counts and len(split_totals) < len(paths):
        split_totals.update({key: value for key, value in _line_totals(paths).items() if key not in split_totals})

    result = count_paths(paths, split_totals=split_totals, progress_every=args.progress_every)
    if args.max_model_sequence_tokens_total is not None:
        total_tokens = int(result.get("model_sequence_tokens_untruncated", 0))
        result["max_model_sequence_tokens_total"] = int(args.max_model_sequence_tokens_total)
        result["within_model_sequence_token_budget"] = total_tokens <= int(args.max_model_sequence_tokens_total)
    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)
    if args.max_model_sequence_tokens_total is not None and not result["within_model_sequence_token_budget"]:
        raise SystemExit(
            "model_sequence_tokens_untruncated="
            f"{result['model_sequence_tokens_untruncated']} exceeds "
            f"--max-model-sequence-tokens-total={args.max_model_sequence_tokens_total}"
        )


if __name__ == "__main__":
    main()
