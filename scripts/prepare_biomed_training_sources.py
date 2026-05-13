#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm


UNIPROT_FIELDS = [
    "accession",
    "reviewed",
    "protein_name",
    "gene_names",
    "organism_name",
    "organism_id",
    "sequence",
    "ec",
    "go_id",
    "keyword",
    "ft_binding",
    "ft_act_site",
    "ft_dna_bind",
    "cc_subcellular_location",
    "cc_cofactor",
    "cc_catalytic_activity",
    "cc_subunit",
]


def _count_data_rows(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    with path.open("rb") as handle:
        rows = sum(1 for _line in handle)
    return max(0, rows - 1)


def _atomic_text_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    return Path(name)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _atomic_text_path(path)
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def download_uniprot_features(
    output: Path,
    *,
    query: str,
    min_existing_rows: int,
    limit: int | None,
    force: bool,
    timeout: int,
) -> dict[str, Any]:
    existing_rows = _count_data_rows(output)
    if not force and existing_rows >= min_existing_rows and (limit is None or existing_rows >= limit):
        return {
            "output": str(output),
            "rows": existing_rows,
            "skipped": True,
            "reason": f"existing file has >= {min_existing_rows} rows",
            "source": "UniProtKB REST reviewed feature export",
        }

    params = {
        "query": query,
        "format": "tsv",
        "compressed": "false",
        "fields": ",".join(UNIPROT_FIELDS),
    }
    url = "https://rest.uniprot.org/uniprotkb/stream?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(url, headers={"User-Agent": "iska-net-biomed-source-prep/1.0"})
    tmp = _atomic_text_path(output)
    rows = 0
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, tmp.open("wb") as handle:
            progress = tqdm(desc="source/uniprot_features", unit="row")
            for line_no, line in enumerate(response):
                handle.write(line)
                if line_no > 0:
                    rows += 1
                    progress.update(1)
                    if limit is not None and rows >= limit:
                        break
            progress.close()
        if rows <= 0:
            raise RuntimeError("UniProt feature export produced no data rows")
        if rows < min_existing_rows and (limit is None or limit >= min_existing_rows):
            raise RuntimeError(
                f"UniProt feature export produced {rows} rows, below required minimum {min_existing_rows}"
            )
        tmp.replace(output)
    except Exception:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise
    return {
        "output": str(output),
        "rows": rows,
        "skipped": False,
        "source": "UniProtKB REST reviewed feature export",
        "query": query,
        "limit": limit,
        "fields": UNIPROT_FIELDS,
        "corresponds_to": ["naturelm_uniprot_sprot"],
    }


def export_binding_affinity_parquet(
    parquet_paths: list[Path],
    output: Path,
    *,
    min_existing_rows: int,
    force: bool,
    batch_size: int,
    limit: int | None,
) -> dict[str, Any]:
    existing_rows = _count_data_rows(output)
    if not force and existing_rows >= min_existing_rows and limit is None:
        return {
            "output": str(output),
            "rows": existing_rows,
            "skipped": True,
            "reason": f"existing file has >= {min_existing_rows} rows",
            "source": "jglaser/binding_affinity local parquet",
        }

    try:
        import pyarrow.parquet as pq
    except Exception as exc:  # pragma: no cover - environment guard
        raise RuntimeError("pyarrow is required to export full binding_affinity_public parquet rows") from exc

    missing = [str(path) for path in parquet_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing binding affinity parquet files: {missing}")

    total_rows = sum(pq.ParquetFile(path).metadata.num_rows for path in parquet_paths)
    if limit is not None:
        total_rows = min(total_rows, limit)
    tmp = _atomic_text_path(output)
    rows = 0
    columns = ["seq", "smiles", "smiles_can", "affinity_uM", "neg_log10_affinity_M", "affinity"]
    fieldnames = [
        "protein_sequence",
        "ligand_smiles",
        "affinity",
        "affinity_type",
        "affinity_units",
        "affinity_uM",
        "neg_log10_affinity_M",
        "binding_affinity_score",
        "interaction_type",
        "source_dataset",
    ]
    try:
        with tmp.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
            writer.writeheader()
            progress = tqdm(total=total_rows, desc="source/binding_affinity_public", unit="row")
            for parquet_path in parquet_paths:
                parquet = pq.ParquetFile(parquet_path)
                for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
                    for row in batch.to_pylist():
                        if limit is not None and rows >= limit:
                            break
                        protein_sequence = str(row.get("seq") or "").strip()
                        ligand_smiles = str(row.get("smiles_can") or row.get("smiles") or "").strip()
                        affinity_uM = row.get("affinity_uM")
                        neg_log10 = row.get("neg_log10_affinity_M")
                        score = row.get("affinity")
                        if not protein_sequence or not ligand_smiles or affinity_uM is None:
                            continue
                        writer.writerow(
                            {
                                "protein_sequence": protein_sequence,
                                "ligand_smiles": ligand_smiles,
                                "affinity": affinity_uM,
                                "affinity_type": "Kd_or_activity_proxy",
                                "affinity_units": "uM",
                                "affinity_uM": affinity_uM,
                                "neg_log10_affinity_M": neg_log10,
                                "binding_affinity_score": score,
                                "interaction_type": "protein_ligand",
                                "source_dataset": "binding_affinity_public",
                            }
                        )
                        rows += 1
                        progress.update(1)
                    if limit is not None and rows >= limit:
                        break
                if limit is not None and rows >= limit:
                    break
            progress.close()
        if rows <= 0:
            raise RuntimeError("Binding affinity export produced no data rows")
        tmp.replace(output)
    except Exception:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise
    return {
        "output": str(output),
        "rows": rows,
        "skipped": False,
        "source": "jglaser/binding_affinity local parquet",
        "source_parquet": [str(path) for path in parquet_paths],
        "corresponds_to": ["binding_affinity_public"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare full local UniProt feature and binding-affinity inputs from sources already used by the training corpus."
    )
    parser.add_argument("--uniprot-output", default="data/local/uniprot_features.tsv")
    parser.add_argument("--affinity-output", default="data/local/complex_affinity.tsv")
    parser.add_argument(
        "--binding-affinity-parquet",
        action="append",
        help="Local parquet from the full selected binding_affinity_public source. May be repeated.",
    )
    parser.add_argument("--summary", default="data/local/biomed_training_sources.summary.json")
    parser.add_argument("--uniprot-query", default="reviewed:true")
    parser.add_argument("--min-uniprot-rows", type=int, default=100000)
    parser.add_argument("--limit-uniprot", type=int)
    parser.add_argument("--min-affinity-rows", type=int, default=1000000)
    parser.add_argument("--affinity-batch-size", type=int, default=65536)
    parser.add_argument("--limit-affinity", type=int)
    parser.add_argument("--skip-uniprot", action="store_true")
    parser.add_argument("--skip-affinity", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()

    summary: dict[str, Any] = {"ok": True, "sources": {}}
    if not args.skip_uniprot:
        summary["sources"]["uniprot_features"] = download_uniprot_features(
            Path(args.uniprot_output),
            query=args.uniprot_query,
            min_existing_rows=args.min_uniprot_rows,
            limit=args.limit_uniprot,
            force=args.force,
            timeout=args.timeout,
        )
    if not args.skip_affinity:
        binding_affinity_parquet = args.binding_affinity_parquet or ["data/raw_hf_full/binding_affinity_public/default/train/0000.parquet"]
        summary["sources"]["complex_affinity"] = export_binding_affinity_parquet(
            [Path(path) for path in binding_affinity_parquet],
            Path(args.affinity_output),
            min_existing_rows=args.min_affinity_rows,
            force=args.force,
            batch_size=args.affinity_batch_size,
            limit=args.limit_affinity,
        )
    _write_json(Path(args.summary), summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
