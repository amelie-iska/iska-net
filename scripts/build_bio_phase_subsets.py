#!/usr/bin/env python
from __future__ import annotations

import argparse
import heapq
import hashlib
import json
from itertools import count
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


_REF_HANDLES: dict[str, Any] = {}


def _source_ref_path(row: dict[str, Any]) -> str | None:
    if isinstance(row, dict) and row.get("__jsonl_ref__") and row.get("path"):
        return str(row["path"])
    return None


def _matches_source_path(path: str | None, include_patterns: tuple[str, ...], exclude_patterns: tuple[str, ...]) -> bool:
    if path is None:
        return True
    if include_patterns and not any(pattern in path for pattern in include_patterns):
        return False
    if exclude_patterns and any(pattern in path for pattern in exclude_patterns):
        return False
    return True


def _resolve_jsonl_ref(row: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(row, dict) or not row.get("__jsonl_ref__"):
        return row
    path = str(row.get("path") or "")
    offset = row.get("offset")
    if not path or offset is None:
        return row
    handle = _REF_HANDLES.get(path)
    if handle is None or handle.closed:
        handle = Path(path).open("rb")
        _REF_HANDLES[path] = handle
    handle.seek(int(offset))
    line = handle.readline()
    if not line:
        return row
    try:
        return json.loads(line.decode("utf-8"))
    except Exception:
        return row


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield _resolve_jsonl_ref(json.loads(line))


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def _write_jsonl_ref(path: Path, offset: int, raw_line: bytes) -> dict[str, Any]:
    return {
        "__jsonl_ref__": True,
        "path": str(path.resolve()),
        "offset": int(offset),
        "sha1": hashlib.sha1(raw_line.strip()).hexdigest(),
    }


def _tokens(row: dict[str, Any]) -> list[str]:
    return [str(tok) for tok in row.get("target_tokens") or []]


def _has_prefix(row: dict[str, Any], prefixes: tuple[str, ...]) -> bool:
    return any(token.startswith(prefixes) for token in _tokens(row))


def _stable_score(row: dict[str, Any], salt: str) -> int:
    text = str(row.get("id") or row.get("metadata", {}).get("row_hash") or json.dumps(row, sort_keys=True)[:512])
    return int(hashlib.sha1(f"{salt}\t{text}".encode("utf-8")).hexdigest()[:16], 16)


def _matching_ref_source_paths(input_dir: Path, include_patterns: tuple[str, ...], exclude_patterns: tuple[str, ...]) -> list[Path]:
    if not include_patterns:
        return []
    split_paths = [input_dir / "train.jsonl", input_dir / "val.jsonl", input_dir / "test.jsonl"]
    paths: set[Path] = set()
    for split_path in split_paths:
        if not split_path.exists():
            continue
        with split_path.open("rb") as raw:
            for raw_line in raw:
                try:
                    row = json.loads(raw_line.decode("utf-8"))
                except Exception:
                    continue
                ref_path = _source_ref_path(row)
                if ref_path and _matches_source_path(ref_path, include_patterns, exclude_patterns):
                    paths.add(Path(ref_path))
    return sorted(paths)


def _iter_candidate_rows(
    input_dir: Path,
    include_patterns: tuple[str, ...],
    exclude_patterns: tuple[str, ...],
) -> Iterable[tuple[dict[str, Any], dict[str, Any], int, str]]:
    source_paths = _matching_ref_source_paths(input_dir, include_patterns, exclude_patterns)
    if source_paths:
        total_hint = sum(path.stat().st_size for path in source_paths if path.exists())
        with tqdm(total=total_hint or None, desc="subset/source_scan", unit="B", unit_scale=True) as pbar:
            for source_path in source_paths:
                if not source_path.exists():
                    continue
                pending = 0
                with source_path.open("rb") as raw:
                    while True:
                        offset = raw.tell()
                        raw_line = raw.readline()
                        if not raw_line:
                            break
                        pending += len(raw_line)
                        try:
                            row = json.loads(raw_line.decode("utf-8"))
                        except Exception:
                            continue
                        yield row, _write_jsonl_ref(source_path, offset, raw_line), len(raw_line), "source"
                        if pending >= 16 * 1024 * 1024:
                            pbar.update(pending)
                            pending = 0
                if pending:
                    pbar.update(pending)
        return

    split_paths = [input_dir / "train.jsonl", input_dir / "val.jsonl", input_dir / "test.jsonl"]
    total_hint = sum(path.stat().st_size for path in split_paths if path.exists())
    with tqdm(total=total_hint or None, desc="subset/curated_scan", unit="B", unit_scale=True) as pbar:
        for path in split_paths:
            if not path.exists():
                continue
            pending = 0
            with path.open("rb") as raw:
                for raw_line in raw:
                    pending += len(raw_line)
                    try:
                        output_row = json.loads(raw_line.decode("utf-8"))
                    except Exception:
                        continue
                    ref_path = _source_ref_path(output_row)
                    if not _matches_source_path(ref_path, include_patterns, exclude_patterns):
                        continue
                    row = _resolve_jsonl_ref(output_row)
                    yield row, output_row, len(raw_line), "curated"
                    if pending >= 16 * 1024 * 1024:
                        pbar.update(pending)
                        pending = 0
            if pending:
                pbar.update(pending)


def _select_modes(
    input_dir: Path,
    target_rows_by_mode: dict[str, int],
    source_path_include: tuple[str, ...] = (),
    source_path_exclude: tuple[str, ...] = (),
) -> tuple[dict[str, list[tuple[int, dict[str, Any]]]], dict[str, Any]]:
    heaps: dict[str, list[tuple[int, int, int, int, dict[str, Any]]]] = {mode: [] for mode in target_rows_by_mode}
    serial = count()
    scanned_rows = 0
    matched_rows = {mode: 0 for mode in target_rows_by_mode}
    scan_sources: dict[str, int] = {}
    for row, output_row, _byte_count, scan_source in _iter_candidate_rows(input_dir, source_path_include, source_path_exclude):
        scanned_rows += 1
        scan_sources[scan_source] = scan_sources.get(scan_source, 0) + 1
        static_ok = _has_prefix(row, STATIC_PREFIXES)
        dynamics_ok = static_ok and _has_prefix(row, DYNAMICS_PREFIXES)
        for mode, target_rows in target_rows_by_mode.items():
            if target_rows <= 0:
                continue
            if (mode == "static" and not static_ok) or (mode == "structure_dynamics" and not dynamics_ok):
                continue
            matched_rows[mode] += 1
            rank_score = _stable_score(row, mode)
            split_score = _stable_score(row, "split")
            item = (-rank_score, next(serial), rank_score, split_score, output_row)
            heap = heaps[mode]
            if len(heap) < target_rows:
                heapq.heappush(heap, item)
            elif rank_score < -heap[0][0]:
                heapq.heapreplace(heap, item)

    selected = {
        mode: [(item[3], item[4]) for item in sorted(heap, key=lambda item: item[2])]
        for mode, heap in heaps.items()
    }
    stats = {
        "scanned_rows": scanned_rows,
        "matched_rows": matched_rows,
        "scan_sources": scan_sources,
        "source_path_include": list(source_path_include),
        "source_path_exclude": list(source_path_exclude),
    }
    return selected, stats


def _split_rows(rows: list[tuple[int, dict[str, Any]]], val_ratio: float, test_ratio: float) -> dict[str, list[dict[str, Any]]]:
    splits = {"train": [], "val": [], "test": []}
    for split_score, row in rows:
        score = split_score / float(0xFFFFFFFFFFFFFFFF)
        if score < test_ratio:
            splits["test"].append(row)
        elif score < test_ratio + val_ratio:
            splits["val"].append(row)
        else:
            splits["train"].append(row)
    if rows and not splits["train"]:
        splits["train"].append(rows[0][1])
    return splits


def _write_subset(
    input_dir: Path,
    output_dir: Path,
    target_rows: int,
    mode: str,
    rows: list[tuple[int, dict[str, Any]]],
    val_ratio: float,
    test_ratio: float,
    scan_stats: dict[str, Any],
) -> dict[str, Any]:
    splits = _split_rows(rows, val_ratio, test_ratio)
    counts = {split: _write_jsonl(output_dir / f"{split}.jsonl", split_rows) for split, split_rows in splits.items()}
    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "mode": mode,
        "target_rows": target_rows,
        "selected_rows": len(rows),
        "split_counts": counts,
        "scan_stats": scan_stats,
        "static_prefixes": STATIC_PREFIXES,
        "dynamics_prefixes": DYNAMICS_PREFIXES if mode == "structure_dynamics" else [],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def build_subset(
    input_dir: Path,
    output_dir: Path,
    target_rows: int,
    mode: str,
    val_ratio: float,
    test_ratio: float,
    source_path_include: tuple[str, ...] = (),
    source_path_exclude: tuple[str, ...] = (),
) -> dict[str, Any]:
    selected, scan_stats = _select_modes(
        input_dir,
        {mode: target_rows},
        source_path_include=source_path_include,
        source_path_exclude=source_path_exclude,
    )
    return _write_subset(input_dir, output_dir, target_rows, mode, selected[mode], val_ratio, test_ratio, scan_stats)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build static-structure and structure-dynamics phase subsets from a curated graph corpus.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--static-output-dir", required=True)
    parser.add_argument("--dynamics-output-dir", required=True)
    parser.add_argument("--static-target-rows", type=int, default=25000)
    parser.add_argument("--dynamics-target-rows", type=int, default=2500)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument(
        "--source-path-include",
        action="append",
        default=[],
        help="Only dereference/index source rows whose JSONL source path contains this substring. Repeatable.",
    )
    parser.add_argument(
        "--source-path-exclude",
        action="append",
        default=[],
        help="Skip dereferencing/indexing source rows whose JSONL source path contains this substring. Repeatable.",
    )
    parser.add_argument("--summary", default="")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    include = tuple(args.source_path_include or ())
    exclude = tuple(args.source_path_exclude or ())
    selected, scan_stats = _select_modes(
        input_dir,
        {"static": args.static_target_rows, "structure_dynamics": args.dynamics_target_rows},
        source_path_include=include,
        source_path_exclude=exclude,
    )
    static_summary = _write_subset(
        input_dir,
        Path(args.static_output_dir),
        args.static_target_rows,
        "static",
        selected["static"],
        args.val_ratio,
        args.test_ratio,
        scan_stats,
    )
    dynamics_summary = _write_subset(
        input_dir,
        Path(args.dynamics_output_dir),
        args.dynamics_target_rows,
        "structure_dynamics",
        selected["structure_dynamics"],
        args.val_ratio,
        args.test_ratio,
        scan_stats,
    )
    summary = {"static_structure": static_summary, "structure_dynamics": dynamics_summary}
    if args.summary:
        Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    for handle in _REF_HANDLES.values():
        handle.close()


if __name__ == "__main__":
    main()
