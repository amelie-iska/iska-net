#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tqdm.auto import tqdm

from iska_reasoner.data.graphify import graphify_rows
from iska_reasoner.data.multimodal import graphify_multimodal, iter_synthetic_multimodal_examples
from iska_reasoner.data.phase_policy import SEQUENCE_ONLY
from iska_reasoner.utils.io import ensure_dir, read_jsonl, write_jsonl


def iter_csv(path: Path, limit: int | None = None) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t" if path.suffix.lower() in {".tsv", ".tab"} else ",")
        for i, row in enumerate(reader):
            if limit is not None and i >= limit:
                break
            yield dict(row)


def iter_json_or_jsonl(path: Path, limit: int | None = None) -> Iterable[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        for i, row in enumerate(read_jsonl(path)):
            if limit is not None and i >= limit:
                break
            yield row
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload if isinstance(payload, list) else payload.get("rows", [])
    for i, row in enumerate(rows):
        if limit is not None and i >= limit:
            break
        if isinstance(row, dict):
            yield row


def iter_fasta(path: Path, limit: int | None = None) -> Iterable[dict[str, Any]]:
    header = ""
    seq: list[str] = []
    emitted = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if seq:
                yield {"prompt": header, "protein_sequence": "".join(seq), "task": "function_description"}
                emitted += 1
                if limit is not None and emitted >= limit:
                    return
            header = line[1:]
            seq = []
        else:
            seq.append(line)
    if seq and (limit is None or emitted < limit):
        yield {"prompt": header, "protein_sequence": "".join(seq), "task": "function_description"}


def iter_rows(paths: list[Path], limit: int | None) -> Iterable[dict[str, Any]]:
    emitted = 0
    for path in paths:
        remaining = None if limit is None else max(0, limit - emitted)
        if remaining == 0:
            return
        suffix = path.suffix.lower()
        if suffix in {".fa", ".fasta", ".faa", ".fna"}:
            iterator = iter_fasta(path, remaining)
        elif suffix in {".csv", ".tsv", ".tab"}:
            iterator = iter_csv(path, remaining)
        else:
            iterator = iter_json_or_jsonl(path, remaining)
        for row in iterator:
            yield row
            emitted += 1
            if limit is not None and emitted >= limit:
                return


def collect_paths(inputs: list[str] | None, input_dirs: list[str] | None, patterns: list[str]) -> list[Path]:
    paths = [Path(item) for item in inputs or []]
    for directory in input_dirs or []:
        base = Path(directory)
        if not base.exists():
            continue
        for pattern in patterns:
            paths.extend(sorted(base.rglob(pattern)))
    return sorted(dict.fromkeys(path for path in paths if path.exists() and path.is_file()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare neutral multimodal graph-to-graph rows.")
    parser.add_argument("--input", action="append", help="CSV/TSV/JSON/JSONL/FASTA source file.")
    parser.add_argument("--input-dir", action="append", help="Directory scanned recursively for CSV/TSV/JSON/JSONL/FASTA files.")
    parser.add_argument("--pattern", action="append", default=["*.jsonl", "*.json", "*.csv", "*.tsv", "*.fa", "*.fasta", "*.faa", "*.fna"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-name", default="local_multimodal_graph_to_graph")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--synthetic-if-empty", action="store_true")
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--molecular-input-policy", choices=["sequence_only", "allow_structure"], default=SEQUENCE_ONLY)
    parser.add_argument("--geometric-features", action="store_true", help="Add string-derived 2D molecular descriptors. Off by default; does not read structure files.")
    args = parser.parse_args()

    if args.synthetic:
        rows = list(iter_synthetic_multimodal_examples(count=args.count, seed=args.seed))
    else:
        paths = collect_paths(args.input, args.input_dir, args.pattern)
        rows = list(tqdm(iter_rows(paths, args.limit), desc="multimodal/rows")) if paths else []
        if not rows and args.synthetic_if_empty:
            rows = list(iter_synthetic_multimodal_examples(count=args.count, seed=args.seed))
        if not rows:
            raise SystemExit("--input/--input-dir is required unless --synthetic or --synthetic-if-empty is set")
    if args.molecular_input_policy == SEQUENCE_ONLY and not args.geometric_features:
        graphs = list(tqdm(graphify_rows(rows, args.dataset_name), total=len(rows), desc="multimodal/graphify"))
    else:
        graphs = [
            graphify_multimodal(
                row,
                idx,
                args.dataset_name,
                molecular_input_policy=args.molecular_input_policy,
                geometric_features=args.geometric_features,
            ).to_dict()
            for idx, row in enumerate(tqdm(rows, desc="multimodal/graphify"))
        ]
    ensure_dir(Path(args.output).parent)
    count = write_jsonl(args.output, graphs)
    print(json.dumps({"rows": len(rows), "graphs": count, "output": args.output}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
