from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, TextIO

from tqdm.auto import tqdm

from iska_reasoner.data.splits import (
    EXPLICIT_SPLIT_KEYS,
    SplitReport,
    _clean,
    _molecule_key,
    _sequence_key,
    assign_split_for_policy,
    split_name_from_key,
)
from iska_reasoner.graph.schema import GraphExample
from iska_reasoner.topology import summarize_graph
from iska_reasoner.utils.io import ensure_dir, read_jsonl


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


def _write_jsonl_row(handle: TextIO, row: dict[str, Any], *, sort_keys: bool = True) -> None:
    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=sort_keys))
    handle.write("\n")


def _write_jsonl_line(handle: TextIO, line: str) -> None:
    handle.write(line.rstrip("\n"))
    handle.write("\n")


def _write_jsonl_ref(handle: TextIO, path: str | Path, offset: int, row_hash: str) -> None:
    _write_jsonl_row(
        handle,
        {
            "__jsonl_ref__": True,
            "path": str(Path(path).resolve()),
            "offset": int(offset),
            "sha1": row_hash,
        },
        sort_keys=True,
    )


def _ensure_trailing_newline(path: Path) -> None:
    """Make append-resume safe after an interrupted write."""
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("rb+") as handle:
        handle.seek(-1, 2)
        if handle.read(1) != b"\n":
            handle.write(b"\n")


def _cleanup_split_indexes(output_dir: Path, split: str) -> None:
    for suffix in (".offsets.u64", ".offsets.meta.json"):
        (output_dir / f"{split}.jsonl{suffix}").unlink(missing_ok=True)


def _iter_jsonl_lines(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield line


def _iter_jsonl_lines_with_offsets(path: str | Path):
    with Path(path).open("rb") as handle:
        while True:
            offset = handle.tell()
            raw_line = handle.readline()
            if not raw_line:
                break
            if raw_line.strip():
                yield offset, raw_line.decode("utf-8", errors="replace").strip()


def _line_dedup_hash(line: str) -> str:
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        return hashlib.sha1(line.encode("utf-8")).hexdigest()
    if isinstance(row, dict) and row.get("__jsonl_ref__") and row.get("sha1"):
        return str(row["sha1"])
    return hashlib.sha1(line.encode("utf-8")).hexdigest()


def _curate_resume_path(output_dir: Path) -> Path:
    return output_dir / ".curate_resume_state.json"


def _curate_resume_config(
    input_paths: list[str | Path],
    val_ratio: float,
    test_ratio: float,
    split_policy: str,
    dedup_key: str,
    quality_mode: str,
    fast_copy: bool,
    index_only: bool = False,
) -> dict[str, Any]:
    return {
        "input_paths": [str(Path(path)) for path in input_paths],
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "split_policy": split_policy,
        "dedup_key": dedup_key,
        "quality_mode": quality_mode,
        "fast_copy": fast_copy,
        "index_only": index_only,
    }


def _load_curate_resume_state(path: Path, config: dict[str, Any]) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if state.get("version") != 1 or state.get("config") != config:
        return None
    return state


def _write_curate_resume_state(
    path: Path,
    config: dict[str, Any],
    processed_rows: dict[str, int],
    counters: dict[str, int],
) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(
            {
                "version": 1,
                "config": config,
                "processed_rows": processed_rows,
                "counters": counters,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    tmp.replace(path)


def _load_fast_resume_outputs(temp_paths: dict[str, Path]) -> tuple[set[str], dict[str, int]]:
    seen: set[str] = set()
    split_counts = {split: 0 for split in temp_paths}
    for split, path in temp_paths.items():
        if not path.exists():
            continue
        for line in _iter_jsonl_lines(path):
            seen.add(_line_dedup_hash(line))
            split_counts[split] += 1
    return seen, split_counts


def _metadata_first(metadata: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = metadata.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _first_node_value(row: dict[str, Any], node_types: set[str]) -> str:
    for node in row.get("nodes") or []:
        if node.get("type") in node_types and node.get("value"):
            return str(node["value"]).strip()
    return ""


def _row_split_key(row: dict[str, Any], split_policy: str, row_hash: str) -> str:
    if split_policy == "row_hash":
        return f"row_hash:{row_hash}"
    if split_policy != "entity":
        raise ValueError(f"Unknown split policy {split_policy!r}; expected row_hash or entity")

    metadata = row.get("metadata") or {}
    for key in EXPLICIT_SPLIT_KEYS:
        value = metadata.get(key)
        if value is not None and str(value).strip():
            return f"{key}:{_clean(value)}"

    protein_sequence = _metadata_first(metadata, "protein_sequence", "sequence", "aa_sequence") or _first_node_value(
        row, {"protein_sequence"}
    )
    if protein_sequence:
        return f"protein_seq:{_sequence_key(protein_sequence, alphabet='protein')}"

    rna_sequence = _metadata_first(metadata, "rna_sequence", "rna") or _first_node_value(row, {"rna_sequence", "rna"})
    if rna_sequence:
        family = _metadata_first(metadata, "rfam_family", "rna_family", "family")
        if family:
            return f"rna_family:{_clean(family)}"
        return f"rna_seq:{_sequence_key(rna_sequence, alphabet='rna')}"

    dna_sequence = _metadata_first(metadata, "dna_sequence", "dna") or _first_node_value(row, {"dna_sequence", "dna"})
    if dna_sequence:
        return f"dna_seq:{_sequence_key(dna_sequence, alphabet='dna')}"

    smiles = _metadata_first(metadata, "smiles", "ligand_smiles", "selfies") or _first_node_value(
        row, {"smiles", "selfies"}
    )
    if smiles:
        return _molecule_key(smiles, metadata)

    graph_seed = _metadata_first(metadata, "graph_generator_seed", "generator_seed")
    if graph_seed:
        return f"graph_seed:{_clean(graph_seed)}"

    for key in ("doi", "arxiv_id", "paper_id", "document_id", "source_document", "title", "record_id", "set_id"):
        value = metadata.get(key)
        if value is not None and str(value).strip():
            return f"document:{_clean(value)}"
    for node in row.get("nodes") or []:
        if node.get("type") in {"doi", "arxiv_id", "paper_id", "document_id", "title", "hebrew_title"} and node.get(
            "value"
        ):
            return f"document:{_clean(node['value'])}"
    return f"row_hash:{row_hash}"


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
    dedup_key: str = "graph_hash",
    quality_mode: str = "full",
    fast_copy: bool = False,
    index_only: bool = False,
    resume: bool = False,
    resume_state_every: int = 10000,
) -> dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    seen: set[str] = set()
    split_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    before = 0
    invalid = 0
    duplicate = 0
    near_duplicate = 0
    low_quality = 0
    license_blocked = 0
    contaminated = 0
    score_count = 0
    score_sum = 0.0
    score_min: float | None = None
    score_max: float | None = None
    task_counts: Counter[str] = Counter()
    signatures: list[tuple[int, ...]] = []
    near_dedup_enabled = near_dedup_threshold <= 1.0
    blocked_license_patterns = blocked_license_patterns or []
    contamination = contamination_terms(contamination_paths)
    split_report = SplitReport(policy=split_policy)

    fast_row_mode = dedup_key == "row_hash" and quality_mode == "none" and not near_dedup_enabled
    if fast_copy and not fast_row_mode:
        raise ValueError("--fast-copy requires --dedup-key row_hash, --quality-mode none, and near-dedup disabled")
    if index_only and not fast_row_mode:
        raise ValueError("--index-only requires --dedup-key row_hash, --quality-mode none, and near-dedup disabled")
    if resume and not fast_row_mode:
        raise ValueError("--resume is supported only in row-hash/no-quality fast curation mode")

    temp_paths = {split: output_dir / f".{split}.jsonl.tmp" for split in split_counts}
    resume_path = _curate_resume_path(output_dir)
    resume_config = _curate_resume_config(
        input_paths,
        val_ratio,
        test_ratio,
        split_policy,
        dedup_key,
        quality_mode,
        fast_copy,
        index_only,
    )
    resume_state = _load_curate_resume_state(resume_path, resume_config) if resume else None
    processed_rows = {str(Path(path)): 0 for path in input_paths}
    resume_outputs_exist = False
    if resume_state is not None:
        processed_rows.update({str(key): int(value) for key, value in resume_state.get("processed_rows", {}).items()})
        counters = resume_state.get("counters", {})
        before = int(counters.get("input_rows", 0))
        invalid = int(counters.get("invalid_rows", 0))
        duplicate = int(counters.get("duplicates_removed", 0))
        near_duplicate = int(counters.get("near_duplicates_removed", 0))
        low_quality = int(counters.get("low_quality_removed", 0))
        license_blocked = int(counters.get("license_blocked", 0))
        contaminated = int(counters.get("contamination_removed", 0))
        seen, split_counts = _load_fast_resume_outputs(temp_paths)
        score_count = sum(split_counts.values())
        score_sum = float(score_count)
        score_min = 1.0 if score_count else None
        score_max = 1.0 if score_count else None
    elif resume and fast_row_mode and any(path.exists() and path.stat().st_size > 0 for path in temp_paths.values()):
        seen, split_counts = _load_fast_resume_outputs(temp_paths)
        resume_outputs_exist = bool(seen)
        score_count = sum(split_counts.values())
        score_sum = float(score_count)
        score_min = 1.0 if score_count else None
        score_max = 1.0 if score_count else None
    handles: dict[str, TextIO] = {}

    def save_resume_state() -> None:
        if not resume or not fast_row_mode:
            return
        _write_curate_resume_state(
            resume_path,
            resume_config,
            processed_rows,
            {
                "input_rows": before,
                "invalid_rows": invalid,
                "duplicates_removed": duplicate,
                "near_duplicates_removed": near_duplicate,
                "low_quality_removed": low_quality,
                "license_blocked": license_blocked,
                "contamination_removed": contaminated,
            },
        )

    try:
        for split, temp_path in temp_paths.items():
            mode = "a" if resume_state is not None or resume_outputs_exist else "w"
            if mode == "a":
                _ensure_trailing_newline(temp_path)
            handles[split] = temp_path.open(mode, encoding="utf-8")

        if fast_row_mode:
            if blocked_license_patterns:
                raise ValueError("Fast row curation does not support blocked license filtering.")
            if contamination:
                raise ValueError("Fast row curation does not support contamination filtering.")
            if min_quality > 0:
                raise ValueError("Fast row curation does not support positive --min-quality.")
            for input_path in input_paths:
                input_key = str(Path(input_path))
                skip_rows = processed_rows.get(input_key, 0) if resume_state is not None else 0
                rows_seen_for_input = 0
                iterator = (
                    _iter_jsonl_lines_with_offsets(input_path)
                    if index_only
                    else ((0, line) for line in _iter_jsonl_lines(input_path))
                )
                for offset, line in tqdm(iterator, desc=f"curate/{Path(input_path).name}", unit="row"):
                    rows_seen_for_input += 1
                    if rows_seen_for_input <= skip_rows:
                        continue
                    before += 1
                    processed_rows[input_key] = rows_seen_for_input
                    h = hashlib.sha1(line.encode("utf-8")).hexdigest()
                    if h in seen:
                        duplicate += 1
                        if before % resume_state_every == 0:
                            save_resume_state()
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        invalid += 1
                        if before % resume_state_every == 0:
                            save_resume_state()
                        continue
                    seen.add(h)
                    split_group_key = _row_split_key(row, split_policy, h)
                    split = split_name_from_key(split_group_key, val_ratio, test_ratio)
                    if index_only:
                        _write_jsonl_ref(handles[split], input_path, offset, h)
                    elif fast_copy:
                        _write_jsonl_line(handles[split], line)
                    else:
                        metadata = row.setdefault("metadata", {})
                        metadata["curation"] = {
                            "hash": h,
                            "quality_score": 1.0,
                            "split_policy": split_policy,
                            "split_group_key": split_group_key,
                        }
                        _write_jsonl_row(handles[split], row, sort_keys=False)
                    split_counts[split] += 1
                    split_report.add(split, split_group_key)
                    score_count += 1
                    score_sum += 1.0
                    score_min = 1.0 if score_min is None else min(score_min, 1.0)
                    score_max = 1.0 if score_max is None else max(score_max, 1.0)
                    task_counts[str(row.get("task", "unknown"))] += 1
                    if before % resume_state_every == 0:
                        save_resume_state()
                processed_rows[input_key] = rows_seen_for_input
                save_resume_state()
        else:
            for input_path in input_paths:
                for row in tqdm(read_jsonl(input_path), desc=f"curate/{Path(input_path).name}", unit="row"):
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
                    if near_dedup_enabled:
                        sig = minhash_signature(shingles(norm))
                        if any(jaccard_from_signatures(sig, prev) >= near_dedup_threshold for prev in signatures):
                            near_duplicate += 1
                            continue
                        signatures.append(sig)
                    seen.add(h)
                    score = quality_score(example) if quality_mode == "full" else 1.0
                    if score < min_quality:
                        low_quality += 1
                        continue
                    example.metadata["curation"] = {"hash": h, "quality_score": score}
                    split, split_group_key = assign_split_for_policy(example, split_policy, val_ratio, test_ratio)
                    example.metadata["curation"]["split_policy"] = split_policy
                    example.metadata["curation"]["split_group_key"] = split_group_key
                    _write_jsonl_row(handles[split], example.to_dict())
                    split_counts[split] += 1
                    split_report.add(split, split_group_key)
                    score_count += 1
                    score_sum += score
                    score_min = score if score_min is None else min(score_min, score)
                    score_max = score if score_max is None else max(score_max, score)
                    task_counts[example.task] += 1
    except Exception:
        for handle in handles.values():
            handle.close()
        if resume and fast_row_mode:
            save_resume_state()
        else:
            for temp_path in temp_paths.values():
                temp_path.unlink(missing_ok=True)
        raise
    else:
        for handle in handles.values():
            handle.close()
        for split, temp_path in temp_paths.items():
            temp_path.replace(output_dir / f"{split}.jsonl")
            _cleanup_split_indexes(output_dir, split)
        resume_path.unlink(missing_ok=True)

    summary = {
        "input_rows": before,
        "invalid_rows": invalid,
        "duplicates_removed": duplicate,
        "near_duplicates_removed": near_duplicate,
        "low_quality_removed": low_quality,
        "license_blocked": license_blocked,
        "contamination_removed": contaminated,
        "kept_rows": sum(split_counts.values()),
        "split_sizes": split_counts,
        "quality_mean": score_sum / score_count if score_count else 0.0,
        "quality_min": score_min if score_min is not None else 0.0,
        "quality_max": score_max if score_max is not None else 0.0,
        "task_counts": dict(task_counts),
        "split_report": split_report.to_dict(),
        "index_only": index_only,
    }
    summary_tmp = output_dir / ".summary.json.tmp"
    summary_tmp.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    summary_tmp.replace(output_dir / "summary.json")
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
    parser.add_argument(
        "--dedup-key",
        choices=["graph_hash", "row_hash"],
        default="graph_hash",
        help="Use canonical graph hashes for strict dedup or raw JSONL row hashes for fast pre-normalized corpora.",
    )
    parser.add_argument(
        "--quality-mode",
        choices=["full", "none"],
        default="full",
        help="Use full topology quality scoring or skip quality scoring for pre-normalized corpora.",
    )
    parser.add_argument(
        "--fast-copy",
        action="store_true",
        help="In row-hash/no-quality mode, copy original JSONL rows to splits without injecting curation metadata.",
    )
    parser.add_argument(
        "--index-only",
        action="store_true",
        help="In row-hash/no-quality mode, write source path/byte-offset references instead of copying full rows.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume interrupted row-hash/no-quality curation from temp split files and a progress state file.",
    )
    parser.add_argument(
        "--resume-state-every",
        type=int,
        default=10000,
        help="Write resumable fast-curation progress after this many input rows.",
    )
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
        dedup_key=args.dedup_key,
        quality_mode=args.quality_mode,
        fast_copy=args.fast_copy,
        index_only=args.index_only,
        resume=args.resume,
        resume_state_every=args.resume_state_every,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
