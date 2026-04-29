#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tqdm.auto import tqdm

from iska_reasoner.data.phase_policy import actual_structure_file_source, graph_structure_violations, sanitize_graph_example_for_sequence_only
from iska_reasoner.graph.schema import GraphExample
from iska_reasoner.utils.io import read_jsonl


def inspect_policy(
    paths: list[Path],
    sequence_only_molecules: bool = False,
    forbid_actual_structure_files: bool = True,
    sanitize_sequence_only: bool = False,
    max_examples: int | None = None,
) -> dict[str, object]:
    counts: Counter[str] = Counter()
    violations: list[dict[str, object]] = []
    scanned = 0
    for path in paths:
        rows = read_jsonl(path)
        iterator = tqdm(rows, desc=f"policy/{path.name}", unit="ex")
        for row in iterator:
            if max_examples is not None and scanned >= max_examples:
                break
            scanned += 1
            try:
                ex = GraphExample.from_dict(row)
            except Exception as exc:
                violations.append({"path": str(path), "row": scanned, "kind": "invalid_graph", "detail": str(exc)})
                continue
            if sanitize_sequence_only and sequence_only_molecules:
                ex = sanitize_graph_example_for_sequence_only(ex)
            counts[ex.task] += 1
            source_path = actual_structure_file_source(ex)
            if forbid_actual_structure_files and source_path:
                violations.append({"path": str(path), "id": ex.id, "kind": "actual_structure_file_source", "detail": source_path})
            if sequence_only_molecules:
                found = graph_structure_violations(ex)
                if found:
                    violations.append({"path": str(path), "id": ex.id, "kind": "sequence_only_violation", "detail": found[:32], "count": len(found)})
    return {
        "ok": not violations,
        "scanned": scanned,
        "task_counts": dict(counts),
        "sequence_only_molecules": sequence_only_molecules,
        "forbid_actual_structure_files": forbid_actual_structure_files,
        "sanitize_sequence_only": sanitize_sequence_only,
        "violation_count": len(violations),
        "violations": violations[:200],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check dataset rows against molecular training phase policy.")
    parser.add_argument("--data-dir", help="Directory containing train/val/test JSONL files.")
    parser.add_argument("--input", action="append", help="JSONL file to inspect. May be repeated.")
    parser.add_argument(
        "--sequence-only-molecules",
        action="store_true",
        help=(
            "Reject coordinate/PDB/trajectory/force/energy structure supervision. "
            "String-derived SMILES/SELFIES atom and bond graph records remain allowed."
        ),
    )
    parser.add_argument("--sanitize-sequence-only", action="store_true", help="Apply the training-time sequence-only graph sanitizer before checking.")
    parser.add_argument("--allow-actual-structure-files", action="store_true", help="Do not reject rows sourced from PDB/mmCIF/SDF/trajectory files.")
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--output", help="Write JSON report.")
    args = parser.parse_args()

    paths = [Path(p) for p in args.input or []]
    if args.data_dir:
        base = Path(args.data_dir)
        paths.extend(path for path in [base / "train.jsonl", base / "val.jsonl", base / "test.jsonl"] if path.exists())
    if not paths:
        raise SystemExit("Provide --data-dir or --input.")

    result = inspect_policy(
        paths,
        sequence_only_molecules=args.sequence_only_molecules,
        forbid_actual_structure_files=not args.allow_actual_structure_files,
        sanitize_sequence_only=args.sanitize_sequence_only,
        max_examples=args.max_examples,
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text)
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
