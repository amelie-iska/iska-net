#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tqdm.auto import tqdm

from iska_reasoner.data.graphify import graphify_rows
from iska_reasoner.utils.io import ensure_dir, write_jsonl


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str) and value.strip().startswith("["):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else [value]
        except Exception:
            return [value]
    return [value]


def _iter_local_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _iter_hf_omg(split: str) -> Iterable[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError("The datasets package is required for streaming tattabio/OMG") from exc
    ds = load_dataset("tattabio/OMG", split=split, streaming=True)
    for row in ds:
        yield dict(row)


def _row_bucket(row: dict[str, Any], bucket_count: int) -> str:
    cds = _as_list(row.get("CDS_seqs") or row.get("cds_seqs"))
    igs = _as_list(row.get("IGS_seqs") or row.get("igs_seqs"))
    orientations = _as_list(row.get("CDS_orientations") or row.get("cds_orientations"))
    aa_len = sum(len(str(seq)) for seq in cds)
    igs_len = sum(len(str(seq)) for seq in igs)
    has_minus = any(str(item).lower() in {"false", "0", "-", "minus"} for item in orientations)
    source = ""
    ids = _as_list(row.get("CDS_ids") or row.get("cds_ids") or row.get("IGS_ids") or row.get("igs_ids"))
    if ids:
        source = str(ids[0]).split("|", 1)[0]
    source_hash = int(hashlib.sha1(source.encode("utf-8")).hexdigest()[:8], 16) % max(1, bucket_count)
    return "|".join(
        [
            f"cds{min(len(cds), 12)}",
            f"igs{min(len(igs), 12)}",
            f"aa{min(9, aa_len // 500)}",
            f"iglen{min(9, igs_len // 250)}",
            "minus" if has_minus else "plus_only",
            f"src{source_hash}",
        ]
    )


def _valid_row(row: dict[str, Any], require_intergenic: bool, min_cds: int) -> bool:
    cds = [seq for seq in _as_list(row.get("CDS_seqs") or row.get("cds_seqs")) if str(seq).strip()]
    igs = [seq for seq in _as_list(row.get("IGS_seqs") or row.get("igs_seqs")) if str(seq).strip()]
    if len(cds) < min_cds:
        return False
    if require_intergenic and not igs:
        return False
    return True


def diverse_subsample(rows: Iterable[dict[str, Any]], *, target_rows: int, scan_limit: int, bucket_count: int, per_bucket: int, require_intergenic: bool, min_cds: int, seed: str) -> list[dict[str, Any]]:
    buckets: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    scanned = 0
    for row in tqdm(rows, total=scan_limit, desc="omg/diverse_scan", unit="row"):
        scanned += 1
        if scanned > scan_limit:
            break
        if not _valid_row(row, require_intergenic=require_intergenic, min_cds=min_cds):
            continue
        bucket = _row_bucket(row, bucket_count)
        key_material = json.dumps(row.get("CDS_ids") or row.get("IGS_ids") or row, sort_keys=True, default=str)[:4096]
        score = hashlib.sha256(f"{seed}|{bucket}|{key_material}".encode("utf-8")).hexdigest()
        bucket_rows = buckets.setdefault(bucket, [])
        bucket_rows.append((score, row))
        bucket_rows.sort(key=lambda item: item[0])
        del bucket_rows[per_bucket:]
        if sum(len(items) for items in buckets.values()) >= target_rows and len(buckets) >= max(4, target_rows // max(1, per_bucket)):
            break
    selected = [item for bucket in sorted(buckets) for item in buckets[bucket]]
    selected.sort(key=lambda item: item[0])
    return [row for _, row in selected[:target_rows]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a diverse, intergenic-retaining subsample of tattabio/OMG and graphify it.")
    parser.add_argument("--input-jsonl", help="Optional local OMG-style JSONL for tests/offline preparation. If omitted, streams tattabio/OMG from Hugging Face.")
    parser.add_argument("--raw-output", default="data/processed/omg_diverse_subsample/raw.jsonl")
    parser.add_argument("--graph-output", default="data/processed/omg_diverse_subsample/all.jsonl")
    parser.add_argument("--split", default="train")
    parser.add_argument("--target-rows", type=int, default=10000)
    parser.add_argument("--scan-limit", type=int, default=500000)
    parser.add_argument("--bucket-count", type=int, default=256)
    parser.add_argument("--per-bucket", type=int, default=4)
    parser.add_argument("--min-cds", type=int, default=3)
    parser.add_argument("--allow-no-intergenic", action="store_true")
    parser.add_argument("--seed", default="ugm-omg-diverse-v1")
    parser.add_argument("--dataset-name", default="omg_diverse_mixed_intergenic_subsample")
    args = parser.parse_args()

    rows = _iter_local_jsonl(Path(args.input_jsonl)) if args.input_jsonl else _iter_hf_omg(args.split)
    selected = diverse_subsample(
        rows,
        target_rows=args.target_rows,
        scan_limit=args.scan_limit,
        bucket_count=args.bucket_count,
        per_bucket=args.per_bucket,
        require_intergenic=not args.allow_no_intergenic,
        min_cds=args.min_cds,
        seed=args.seed,
    )
    raw_path = Path(args.raw_output)
    graph_path = Path(args.graph_output)
    ensure_dir(raw_path.parent)
    raw_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected), encoding="utf-8")
    ensure_dir(graph_path.parent)
    count = write_jsonl(graph_path, tqdm(graphify_rows(iter(selected), args.dataset_name), total=len(selected), desc="omg/graphify", unit="ex"))
    print(json.dumps({"selected_rows": len(selected), "graphs": count, "raw_output": str(raw_path), "graph_output": str(graph_path)}, indent=2))


if __name__ == "__main__":
    main()
