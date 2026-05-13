from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from tqdm.auto import tqdm

from iska_reasoner.data.bioselfies import bioselfies_from_modalities
from iska_reasoner.data.graphify import graphify_rows
from iska_reasoner.graph.schema import Edge, GraphExample, Node
from iska_reasoner.utils.config import load_yaml
from iska_reasoner.utils.io import ensure_dir


def _manifest_items(manifest_path: Path, datasets: set[str] | None) -> list[dict[str, Any]]:
    manifest = load_yaml(manifest_path)
    items = []
    for item in manifest.get("datasets", []):
        if item.get("method", "hf_rows") != "hf_rows" or item.get("manifest_only", False):
            continue
        if item.get("full_training_enabled", True) is False:
            continue
        if datasets and item["name"] not in datasets:
            continue
        items.append(item)
    return items


def _item_row_cap(item: dict[str, Any], max_rows_per_dataset: int | None) -> int | None:
    item_cap = item.get("full_training_max_rows")
    caps: list[int] = []
    if max_rows_per_dataset is not None:
        caps.append(int(max_rows_per_dataset))
    if item_cap is not None:
        caps.append(int(item_cap))
    return min(caps) if caps else None


def _parquet_paths(raw_full_dir: Path, item: dict[str, Any]) -> list[Path]:
    name = item["name"]
    config = item.get("config", "default")
    split = item.get("split", "train")
    return sorted((raw_full_dir / name / config / split).glob("*.parquet"))


def _parquet_num_rows(path: Path) -> int | None:
    try:
        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception:
        return None


def _dataset_row_total(paths: list[Path]) -> int | None:
    total = 0
    for path in paths:
        rows = _parquet_num_rows(path)
        if rows is None:
            return None
        total += rows
    return total


def _effective_total(row_total: int | None, max_rows: int | None, remaining_budget: int | None) -> int | None:
    if row_total is None:
        if max_rows is None and remaining_budget is None:
            return None
        candidates = [value for value in (max_rows, remaining_budget) if value is not None]
        return min(candidates) if candidates else None
    effective = row_total
    if max_rows is not None:
        effective = min(effective, max_rows)
    if remaining_budget is not None:
        effective = min(effective, remaining_budget)
    return max(0, effective)


def _split_postfix(counts: dict[str, int], total: int, dataset_total: int, errors: int = 0) -> dict[str, int]:
    return {
        "all": total,
        "dataset": dataset_total,
        "train": counts["train"],
        "val": counts["val"],
        "test": counts["test"],
        "errors": errors,
    }


def _split_name(example_id: str, val_ratio: float, test_ratio: float) -> str:
    bucket = int(hashlib.sha1(example_id.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    if bucket < test_ratio:
        return "test"
    if bucket < test_ratio + val_ratio:
        return "val"
    return "train"


def _text_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (list, tuple, set)):
        return "; ".join(_text_value(item) for item in value if _text_value(item))
    return str(value)


def _first_text(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    lowered = {str(key).lower(): key for key in row}
    for key in keys:
        if key in row:
            text = _text_value(row.get(key)).strip()
            if text:
                return text
        original = lowered.get(key.lower())
        if original is not None:
            text = _text_value(row.get(original)).strip()
            if text:
                return text
    return ""


def _longest_sequence_like_text(row: dict[str, Any]) -> str:
    best = ""
    for key, value in row.items():
        key_text = str(key).lower()
        if any(skip in key_text for skip in ("description", "function", "comment", "organism", "name", "text")):
            continue
        text = re.sub(r"\s+", "", _text_value(value)).upper()
        if len(text) > len(best) and re.fullmatch(r"[A-Z*.-]+", text or ""):
            best = text
    return best


def _compact_length_bin(length: int) -> str:
    for bound in (16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192):
        if length <= bound:
            return f"le_{bound}"
    return "gt_8192"


def _target_safe(text: str, max_len: int = 96) -> str:
    cleaned = re.sub(r"\s+", "_", text.strip())
    cleaned = re.sub(r"[^A-Za-z0-9_.:+\\-\\[\\]/]", "_", cleaned)
    return cleaned[:max_len] if cleaned else "unknown"


def _infer_compact_modality(dataset_name: str, row: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    name = dataset_name.lower()
    metadata: dict[str, Any] = {}
    if "pubchem" in name or "selfies" in name or _first_text(row, ("selfies", "SELFIES", "molecule_selfies")):
        text = _first_text(row, ("selfies", "SELFIES", "molecule_selfies", "canonical_selfies"))
        if not text:
            text = _first_text(row, ("smiles", "SMILES", "canonical_smiles", "CanonicalSMILES"))
            metadata["source_string_type"] = "smiles"
        else:
            metadata["source_string_type"] = "selfies"
        return "selfies", text, metadata

    if "rfam" in name or "rna" in name:
        sequence = _first_text(row, ("rna_sequence", "sequence", "Sequence", "seq", "Seq", "nucleotide_sequence"))
        return "rna", sequence or _longest_sequence_like_text(row), metadata

    if "dna" in name or "coding" in name:
        sequence = _first_text(row, ("dna_sequence", "sequence", "Sequence", "seq", "Seq", "nucleotide_sequence"))
        return "dna", sequence or _longest_sequence_like_text(row), metadata

    sequence = _first_text(
        row,
        (
            "protein_sequence",
            "aa_sequence",
            "rep_sequence",
            "representative_sequence",
            "sequence",
            "Sequence",
            "seq",
        ),
    )
    return "protein", sequence or _longest_sequence_like_text(row), metadata


def _compact_bioselfies_row(row: dict[str, Any], modality: str, sequence_text: str, max_chars: int) -> str:
    if modality == "protein":
        local = {"protein_sequence": sequence_text}
    elif modality == "dna":
        local = {"dna_sequence": sequence_text}
    elif modality == "rna":
        local = {"rna_sequence": sequence_text}
    else:
        local = {"selfies": sequence_text}
    return bioselfies_from_modalities(local, max_len=max_chars)


def _compact_bio_scale_graph(
    row: dict[str, Any],
    dataset_name: str,
    local_idx: int,
    max_sequence_chars: int,
) -> dict[str, Any]:
    modality, raw_text, extra_metadata = _infer_compact_modality(dataset_name, row)
    raw_text = re.sub(r"\s+", "", raw_text) if modality != "selfies" else raw_text.strip()
    full_length = len(raw_text)
    clipped = raw_text[:max_sequence_chars]
    bioselfies = _compact_bioselfies_row(row, modality, clipped, max_sequence_chars)
    row_hash = hashlib.sha1(
        json.dumps(row, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8", errors="replace")
    ).hexdigest()[:16]
    example_id = f"{dataset_name}:compact:{local_idx}:{row_hash}"

    function_text = _first_text(
        row,
        (
            "function_description",
            "Function [CC]",
            "function",
            "description",
            "annotation",
            "protein_name",
            "Protein names",
            "product",
        ),
    )
    family = _first_text(row, ("family", "family_id", "rfam_acc", "accession", "Entry", "id"))

    sequence_preview = clipped[:512]
    nodes = [
        Node("domain", "bio_scale_domain", modality, {"compact": True}),
        Node(
            "sequence",
            f"{modality}_sequence_compact",
            sequence_preview,
            {
                "length": full_length,
                "clipped_length": len(clipped),
                "preview_length": len(sequence_preview),
                "max_sequence_chars": max_sequence_chars,
            },
        ),
        Node(
            "bioselfies",
            "bioselfies_compact",
            bioselfies,
            {
                "length": len(bioselfies),
                "modality": modality,
                "tokenizer": "bioselfies",
            },
        ),
    ]
    edges = [
        Edge("domain", "sequence", "has_compact_sequence"),
        Edge("sequence", "bioselfies", "serializes_to_bioselfies"),
    ]
    target_tokens = [
        "BIO_SCALE:compact",
        "UGM:tokenizer:bioselfies",
        f"BIOSEQ:modality:{modality}",
        f"BIOSEQ:length_bin:{_compact_length_bin(full_length)}",
    ]
    if modality == "selfies":
        target_tokens.append("BIOSEQ:selfies_input:8192_ready")
    if family:
        nodes.append(Node("family", "bio_sequence_family", family[:256], {"source": "row"}))
        edges.append(Edge("domain", "family", "has_family_annotation"))
        target_tokens.append(f"BIOSEQ:family:{_target_safe(family)}")
    if function_text:
        nodes.append(Node("function", "function_description", function_text[:1024], {"source": "row"}))
        edges.append(Edge("sequence", "function", "has_function_description"))
        target_tokens.append("BIOSEQ:function_description_present")

    metadata = {
        "source_dataset": dataset_name,
        "row_hash": row_hash,
        "modalities": [modality, "bioselfies"],
        "bio_scale_compact": True,
        "bioselfies_enabled": True,
        "input_representation": "bioselfies",
        "modality": modality,
        "sequence_length": full_length,
        "sequence_clipped_length": len(clipped),
        **extra_metadata,
    }
    example = GraphExample(
        id=example_id,
        task="bio_sequence_scale_pretraining",
        nodes=nodes,
        edges=edges,
        target_tokens=target_tokens,
        decoder_orders=[list(range(len(target_tokens)))],
        metadata=metadata,
    )
    example.validate()
    return example.to_dict()


def _graphify_compact_bio_scale_rows(
    rows: list[dict[str, Any]],
    dataset_name: str,
    start_idx: int,
    max_sequence_chars: int,
):
    for offset, row in enumerate(rows):
        yield _compact_bio_scale_graph(row, dataset_name, start_idx + offset, max_sequence_chars)


def graphify_full_parquet_manifest(
    manifest_path: str | Path,
    raw_full_dir: str | Path,
    output_dir: str | Path,
    max_rows_per_dataset: int | None,
    row_budget: int | None,
    val_ratio: float,
    test_ratio: float,
    batch_size: int,
    datasets: set[str] | None = None,
    progress_every: int = 10000,
    nested_progress: bool = True,
    bio_scale_compact: bool = False,
    bio_scale_max_sequence_chars: int = 8192,
) -> dict[str, Any]:
    manifest = Path(manifest_path)
    raw_full = Path(raw_full_dir)
    out_dir = ensure_dir(output_dir)
    items = _manifest_items(manifest, datasets)
    path_map = {item["name"]: _parquet_paths(raw_full, item) for item in items}
    source_rows = {name: _dataset_row_total(paths) for name, paths in path_map.items()}
    global_total: int | None = 0
    remaining_budget = row_budget
    for item in items:
        name = item["name"]
        effective = _effective_total(source_rows[name], _item_row_cap(item, max_rows_per_dataset), remaining_budget)
        if effective is None:
            global_total = None
            break
        global_total += effective
        if remaining_budget is not None:
            remaining_budget = max(0, remaining_budget - effective)

    handles = {
        "train": (out_dir / "train.jsonl").open("w", encoding="utf-8"),
        "val": (out_dir / "val.jsonl").open("w", encoding="utf-8"),
        "test": (out_dir / "test.jsonl").open("w", encoding="utf-8"),
    }
    counts = {"train": 0, "val": 0, "test": 0}
    per_dataset: dict[str, int] = {}
    per_dataset_limits: dict[str, int | None] = {}
    per_dataset_errors: dict[str, int] = {}
    total = 0
    errors = 0
    try:
        with tqdm(total=global_total, desc="graphify/full/all", unit="ex", position=0) as global_pbar:
            for item in items:
                name = item["name"]
                paths = path_map[name]
                if not paths:
                    tqdm.write(f"[graphify] skip {name}: no parquet files found under {raw_full}")
                    per_dataset[name] = 0
                    per_dataset_errors[name] = 0
                    continue
                if row_budget is not None and total >= row_budget:
                    tqdm.write("[graphify] row budget reached; stopping before remaining datasets")
                    break

                dataset_remaining_budget = None if row_budget is None else max(0, row_budget - total)
                item_row_cap = _item_row_cap(item, max_rows_per_dataset)
                per_dataset_limits[name] = item_row_cap
                dataset_total = _effective_total(source_rows[name], item_row_cap, dataset_remaining_budget)
                written_for_dataset = 0
                errors_for_dataset = 0
                start_idx = 0
                compact_note = f" compact_bioselfies={bio_scale_compact}"
                tqdm.write(
                    f"[graphify] start {name}: files={len(paths)} source_rows={source_rows[name]} "
                    f"target_rows={dataset_total} batch_size={batch_size}{compact_note}"
                )
                with tqdm(
                    total=dataset_total,
                    desc=f"graphify/full/{name}",
                    unit="ex",
                    position=1,
                    leave=True,
                ) as dataset_pbar:
                    for path_index, path in enumerate(paths, start=1):
                        if (item_row_cap is not None and written_for_dataset >= item_row_cap) or (
                            row_budget is not None and total >= row_budget
                        ):
                            break
                        parquet = pq.ParquetFile(path)
                        file_rows = int(parquet.metadata.num_rows)
                        with tqdm(
                            total=file_rows,
                            desc=f"parquet/{name}/{path.name}",
                            unit="row",
                            position=2,
                            leave=False,
                            disable=not nested_progress,
                        ) as file_pbar:
                            for batch in parquet.iter_batches(batch_size=batch_size):
                                rows = batch.to_pylist()
                                remaining = len(rows)
                                if item_row_cap is not None:
                                    remaining = min(remaining, item_row_cap - written_for_dataset)
                                if row_budget is not None:
                                    remaining = min(remaining, row_budget - total)
                                if remaining <= 0:
                                    break
                                rows = rows[:remaining]
                                graph_iter = (
                                    _graphify_compact_bio_scale_rows(
                                        rows,
                                        name,
                                        start_idx=start_idx,
                                        max_sequence_chars=bio_scale_max_sequence_chars,
                                    )
                                    if bio_scale_compact
                                    else graphify_rows(rows, name, start_idx=start_idx)
                                )
                                for graph in graph_iter:
                                    try:
                                        split = _split_name(str(graph["id"]), val_ratio, test_ratio)
                                        handles[split].write(json.dumps(graph, ensure_ascii=False, sort_keys=True) + "\n")
                                        counts[split] += 1
                                        total += 1
                                        written_for_dataset += 1
                                        if written_for_dataset % progress_every == 0:
                                            postfix = _split_postfix(counts, total, written_for_dataset, errors)
                                            postfix["file"] = path_index
                                            dataset_pbar.set_postfix(postfix, refresh=False)
                                            global_pbar.set_postfix(postfix, refresh=False)
                                    except Exception as exc:  # pragma: no cover - diagnostics path.
                                        errors += 1
                                        errors_for_dataset += 1
                                        tqdm.write(f"[graphify] error {name}:{path.name}:{start_idx}: {exc!r}")
                                        continue
                                    finally:
                                        dataset_pbar.update(1)
                                        global_pbar.update(1)
                                start_idx += len(rows)
                                file_pbar.update(len(rows))
                        dataset_pbar.set_postfix(_split_postfix(counts, total, written_for_dataset, errors), refresh=True)
                per_dataset[name] = written_for_dataset
                per_dataset_errors[name] = errors_for_dataset
                tqdm.write(
                    f"[graphify] done {name}: written={written_for_dataset} errors={errors_for_dataset} "
                    f"cumulative={total} split_counts={counts}"
                )
                if row_budget is not None and total >= row_budget:
                    break
    finally:
        for handle in handles.values():
            handle.close()
    summary = {
        "manifest": str(manifest),
        "raw_full_dir": str(raw_full),
        "output_dir": str(out_dir),
        "max_rows_per_dataset": max_rows_per_dataset,
        "row_budget": row_budget,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "counts": counts,
        "total": total,
        "per_dataset": per_dataset,
        "per_dataset_limits": per_dataset_limits,
        "per_dataset_errors": per_dataset_errors,
        "per_dataset_source_rows": source_rows,
        "parquet_files": {name: [str(path) for path in paths] for name, paths in path_map.items()},
        "errors": errors,
        "bio_scale_compact": bio_scale_compact,
        "bio_scale_max_sequence_chars": bio_scale_max_sequence_chars,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Graphify full selected HF parquet snapshots from the dataset manifest.")
    parser.add_argument("--manifest", default="data/manifests/datasets.yaml")
    parser.add_argument("--raw-full-dir", default="data/raw_hf_full")
    parser.add_argument("--output-dir", default="data/processed/real_full_selected_mix")
    parser.add_argument(
        "--max-rows-per-dataset",
        type=int,
        default=None,
        help="Optional per-dataset row cap. Omit for full available parquet rows.",
    )
    parser.add_argument("--row-budget", type=int)
    parser.add_argument("--val-ratio", type=float, default=0.01)
    parser.add_argument("--test-ratio", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--dataset", action="append", help="Optional manifest dataset name. May be repeated.")
    parser.add_argument("--progress-every", type=int, default=10000)
    parser.add_argument("--no-nested-progress", action="store_true", help="Disable per-parquet-file progress bars.")
    parser.add_argument(
        "--bio-scale-compact",
        action="store_true",
        help="Write compact BioSELFIES/sequence rows for million-scale bio sequence pretraining.",
    )
    parser.add_argument(
        "--bio-scale-max-sequence-chars",
        type=int,
        default=8192,
        help="Maximum raw sequence/SELFIES characters retained per compact bio-scale row.",
    )
    args = parser.parse_args()
    summary = graphify_full_parquet_manifest(
        manifest_path=args.manifest,
        raw_full_dir=args.raw_full_dir,
        output_dir=args.output_dir,
        max_rows_per_dataset=args.max_rows_per_dataset,
        row_budget=args.row_budget,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        batch_size=args.batch_size,
        datasets=set(args.dataset) if args.dataset else None,
        progress_every=args.progress_every,
        nested_progress=not args.no_nested_progress,
        bio_scale_compact=args.bio_scale_compact,
        bio_scale_max_sequence_chars=args.bio_scale_max_sequence_chars,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
