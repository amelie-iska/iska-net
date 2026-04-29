from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from tqdm.auto import tqdm

from iska_reasoner.data.splits import SplitReport, assign_split_for_policy
from iska_reasoner.graph.schema import GraphExample
from iska_reasoner.topology import summarize_graph
from iska_reasoner.utils.io import ensure_dir, read_jsonl, write_jsonl


def graph_hash(example: GraphExample) -> str:
    payload = {
        "task": example.task,
        "nodes": sorted((node.type, node.value) for node in example.nodes),
        "edges": sorted((edge.src, edge.dst, edge.type) for edge in example.edges),
        "target_tokens": sorted(example.target_tokens),
    }
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def normalized_text(example: GraphExample) -> str:
    pieces = [example.task]
    pieces.extend(node.value for node in example.nodes if node.value)
    pieces.extend(example.target_tokens)
    text = " ".join(pieces).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^\w\u0590-\u05ff]+", " ", text)).strip()


def shingles(text: str, k: int = 5) -> set[str]:
    tokens = text.split()
    if len(tokens) < k:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i : i + k]) for i in range(len(tokens) - k + 1)}


def minhash_signature(shingle_set: set[str], num_perm: int = 32) -> tuple[int, ...]:
    if not shingle_set:
        return tuple([0] * num_perm)
    signature = []
    for i in range(num_perm):
        best = None
        salt = f"mh{i}:"
        for shingle in shingle_set:
            value = int(hashlib.blake2b((salt + shingle).encode("utf-8"), digest_size=8).hexdigest(), 16)
            if best is None or value < best:
                best = value
        signature.append(int(best or 0))
    return tuple(signature)


def jaccard_from_signatures(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return sum(1 for x, y in zip(a, b) if x == y) / len(a)


def license_allowed(example: GraphExample, blocked_patterns: list[str]) -> bool:
    if not blocked_patterns:
        return True
    text = " ".join(
        str(value)
        for value in [
            example.metadata.get("license"),
            example.metadata.get("source_license"),
            example.metadata.get("upstream_license"),
        ]
        if value
    ).lower()
    return not any(pattern.lower() in text for pattern in blocked_patterns)


def contamination_terms(paths: list[str | Path] | None) -> set[str]:
    terms: set[str] = set()
    if not paths:
        return terms
    for path in paths:
        p = Path(path)
        if not p.exists():
            continue
        if p.suffix == ".jsonl":
            for row in read_jsonl(p):
                try:
                    ex = GraphExample.from_dict(row)
                    terms.add(normalized_text(ex))
                except Exception:
                    continue
        else:
            for line in p.read_text(encoding="utf-8").splitlines():
                line = re.sub(r"\s+", " ", line.strip().lower())
                if line:
                    terms.add(line)
    return terms


def quality_score(example: GraphExample) -> float:
    topo = summarize_graph(example)
    score = 0.0
    score += min(1.0, topo.node_count / 16.0)
    score += min(1.0, topo.edge_count / max(1.0, topo.node_count))
    score += min(1.0, len(example.target_tokens) / 8.0)
    score += 0.5 if any(node.type in {"tool_call", "lean_proof", "smiles", "solution"} for node in example.nodes) else 0.0
    score += 0.25 if example.decoder_orders else 0.0
    return float(score)


def split_name(example_hash: str, val_ratio: float, test_ratio: float) -> str:
    bucket = int(example_hash[:8], 16) / 0xFFFFFFFF
    if bucket < test_ratio:
        return "test"
    if bucket < test_ratio + val_ratio:
        return "val"
    return "train"


def curate_files(
    input_paths: list[str | Path],
    output_dir: str | Path,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    min_quality: float = 0.0,
    near_dedup_threshold: float = 1.01,
    blocked_license_patterns: list[str] | None = None,
    contamination_paths: list[str | Path] | None = None,
    split_policy: str = "row_hash",
) -> dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    seen: set[str] = set()
    splits: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    before = 0
    invalid = 0
    duplicate = 0
    near_duplicate = 0
    low_quality = 0
    license_blocked = 0
    contaminated = 0
    scores: list[float] = []
    task_counts: Counter[str] = Counter()
    signatures: list[tuple[int, ...]] = []
    blocked_license_patterns = blocked_license_patterns or []
    contamination = contamination_terms(contamination_paths)
    split_report = SplitReport(policy=split_policy)

    for input_path in input_paths:
        rows = list(read_jsonl(input_path))
        for row in tqdm(rows, desc=f"curate/{Path(input_path).name}"):
            before += 1
            try:
                example = GraphExample.from_dict(row)
            except Exception:
                invalid += 1
                continue
            h = graph_hash(example)
            if h in seen:
                duplicate += 1
                continue
            if not license_allowed(example, blocked_license_patterns):
                license_blocked += 1
                continue
            norm = normalized_text(example)
            if norm in contamination or any(term and term in norm for term in contamination if len(term) > 80):
                contaminated += 1
                continue
            sig = minhash_signature(shingles(norm))
            if near_dedup_threshold <= 1.0 and any(jaccard_from_signatures(sig, prev) >= near_dedup_threshold for prev in signatures):
                near_duplicate += 1
                continue
            seen.add(h)
            signatures.append(sig)
            score = quality_score(example)
            if score < min_quality:
                low_quality += 1
                continue
            example.metadata["curation"] = {"hash": h, "quality_score": score}
            split, split_group_key = assign_split_for_policy(example, split_policy, val_ratio, test_ratio)
            example.metadata["curation"]["split_policy"] = split_policy
            example.metadata["curation"]["split_group_key"] = split_group_key
            splits[split].append(example.to_dict())
            split_report.add(split, split_group_key)
            scores.append(score)
            task_counts[example.task] += 1

    for split, rows in splits.items():
        write_jsonl(output_dir / f"{split}.jsonl", rows)
    summary = {
        "input_rows": before,
        "invalid_rows": invalid,
        "duplicates_removed": duplicate,
        "near_duplicates_removed": near_duplicate,
        "low_quality_removed": low_quality,
        "license_blocked": license_blocked,
        "contamination_removed": contaminated,
        "kept_rows": sum(len(rows) for rows in splits.values()),
        "split_sizes": {split: len(rows) for split, rows in splits.items()},
        "quality_mean": mean(scores) if scores else 0.0,
        "quality_min": min(scores) if scores else 0.0,
        "quality_max": max(scores) if scores else 0.0,
        "task_counts": dict(task_counts),
        "split_report": split_report.to_dict(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Curate graph JSONL into deduplicated train/val/test splits.")
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--min-quality", type=float, default=0.0)
    parser.add_argument("--near-dedup-threshold", type=float, default=1.01, help="Enable MinHash near-dedup at <=1.0, e.g. 0.9.")
    parser.add_argument("--blocked-license-pattern", action="append", default=[])
    parser.add_argument("--contamination", action="append", help="Text or JSONL file to filter against.")
    parser.add_argument("--split-policy", choices=["row_hash", "entity"], default="row_hash")
    args = parser.parse_args()
    summary = curate_files(
        args.input,
        args.output_dir,
        args.val_ratio,
        args.test_ratio,
        args.min_quality,
        near_dedup_threshold=args.near_dedup_threshold,
        blocked_license_patterns=args.blocked_license_pattern,
        contamination_paths=args.contamination,
        split_policy=args.split_policy,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
