#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MODALITY_DATASETS = {
    "molecule": ("pubchem10m_selfies_train",),
    "rna": ("rfam_sequence_train", "rnacentral_8192_sequence_train"),
    "dna": ("dna_coding_regions_train",),
    "protein_function_text": ("uniprot_function_text_train",),
}


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _source_rows(summary: dict[str, Any], datasets: tuple[str, ...]) -> int:
    source = summary.get("per_dataset_source_rows") or {}
    return sum(int(source.get(name) or 0) for name in datasets)


def _written_rows(summary: dict[str, Any], datasets: tuple[str, ...]) -> int:
    written = summary.get("per_dataset") or {}
    return sum(int(written.get(name) or 0) for name in datasets)


def check_targets(
    *,
    protein_summary_path: Path,
    bio_sequence_summary_path: Path,
    target_rows: int,
    allow_source_limited: set[str],
) -> dict[str, Any]:
    protein_summary = _read_json(protein_summary_path)
    bio_summary = _read_json(bio_sequence_summary_path)
    protein_rows = int(
        ((protein_summary.get("sources") or {}).get("uniprot_features") or {}).get("rows") or 0
    )

    modalities: dict[str, dict[str, Any]] = {
        "protein": {
            "datasets": ["UniProtKB REST feature stream"],
            "source_rows": protein_rows,
            "written_rows": protein_rows,
        }
    }
    for modality, datasets in MODALITY_DATASETS.items():
        modalities[modality] = {
            "datasets": list(datasets),
            "source_rows": _source_rows(bio_summary, datasets),
            "written_rows": _written_rows(bio_summary, datasets),
        }

    failures: list[str] = []
    warnings: list[str] = []
    for modality, payload in modalities.items():
        written = int(payload["written_rows"])
        source = int(payload["source_rows"])
        payload["target_rows"] = target_rows
        payload["target_met"] = written >= target_rows
        payload["source_limited"] = source < target_rows and written >= source
        if written >= target_rows:
            continue
        if modality in allow_source_limited and payload["source_limited"]:
            warnings.append(
                f"{modality} wrote {written:,} rows, below target {target_rows:,}, because source rows are {source:,}"
            )
        else:
            failures.append(
                f"{modality} wrote {written:,} rows, below target {target_rows:,} (source rows: {source:,})"
            )

    return {
        "ok": not failures,
        "target_rows_per_modality": target_rows,
        "modalities": modalities,
        "warnings": warnings,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check bio-scale modality row targets after graphification.")
    parser.add_argument("--protein-summary", required=True)
    parser.add_argument("--bio-sequence-summary", required=True)
    parser.add_argument("--target-rows", type=int, default=3_000_000)
    parser.add_argument("--allow-source-limited", default="dna,protein_function_text")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    allow_source_limited = {
        item.strip()
        for item in args.allow_source_limited.split(",")
        if item.strip()
    }
    summary = check_targets(
        protein_summary_path=Path(args.protein_summary),
        bio_sequence_summary_path=Path(args.bio_sequence_summary),
        target_rows=args.target_rows,
        allow_source_limited=allow_source_limited,
    )
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
