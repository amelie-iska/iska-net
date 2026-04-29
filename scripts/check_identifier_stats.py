#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tqdm.auto import tqdm

from iska_reasoner.graph.schema import GraphExample, graph_source_tokens
from iska_reasoner.utils.io import read_jsonl


def inspect_paths(paths: list[Path], limit: int | None = None) -> dict[str, Any]:
    examples = 0
    invalid = 0
    max_identifier = 0
    max_nodes = 0
    max_edges = 0
    node_edge_overlap = 0
    missing_node_identifiers = 0
    missing_edge_identifiers = 0
    for path in paths:
        iterator = read_jsonl(path)
        for row in tqdm(iterator, desc=f"ids/{path.name}", unit="ex"):
            if limit is not None and examples >= limit:
                break
            try:
                example = GraphExample.from_dict(row)
            except Exception:
                invalid += 1
                continue
            _, kinds, _, identifiers = graph_source_tokens(example)
            node_ids = {identifier for kind, identifier in zip(kinds, identifiers) if kind == "node"}
            edge_ids = {identifier for kind, identifier in zip(kinds, identifiers) if kind == "edge"}
            if example.nodes and not node_ids:
                missing_node_identifiers += 1
            if example.edges and not edge_ids:
                missing_edge_identifiers += 1
            if node_ids.intersection(edge_ids):
                node_edge_overlap += 1
            max_identifier = max(max_identifier, max(identifiers, default=0))
            max_nodes = max(max_nodes, len(example.nodes))
            max_edges = max(max_edges, len(example.edges))
            examples += 1
    return {
        "examples": examples,
        "invalid_rows": invalid,
        "max_identifier": max_identifier,
        "max_nodes": max_nodes,
        "max_edges": max_edges,
        "node_edge_identifier_overlap_examples": node_edge_overlap,
        "missing_node_identifier_examples": missing_node_identifiers,
        "missing_edge_identifier_examples": missing_edge_identifiers,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check vertex/edge structural identifier coverage in graph JSONL data.")
    parser.add_argument("--input", action="append", help="Specific JSONL path. Defaults to train/val/test under --data-dir.")
    parser.add_argument("--data-dir")
    parser.add_argument("--output")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.input:
        paths = [Path(path) for path in args.input]
    elif args.data_dir:
        data_dir = Path(args.data_dir)
        paths = [path for path in [data_dir / "train.jsonl", data_dir / "val.jsonl", data_dir / "test.jsonl"] if path.exists()]
    else:
        raise SystemExit("Provide --input or --data-dir")
    result = inspect_paths(paths, limit=args.limit)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
