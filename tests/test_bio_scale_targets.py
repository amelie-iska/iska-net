from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_bio_phase_subsets import build_subset
from scripts.check_bio_scale_targets import check_targets


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _write_indexed_jsonl(source_path: Path, index_path: Path, rows: list[dict]) -> None:
    source_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    refs = []
    with source_path.open("wb") as handle:
        for row in rows:
            offset = handle.tell()
            line = json.dumps(row, sort_keys=True).encode("utf-8") + b"\n"
            handle.write(line)
            refs.append(
                {
                    "__jsonl_ref__": True,
                    "path": str(source_path.resolve()),
                    "offset": offset,
                    "sha1": hashlib.sha1(line.rstrip(b"\n")).hexdigest(),
                }
            )
    index_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in refs), encoding="utf-8")


def test_bio_phase_subsets_select_static_and_dynamics_rows(tmp_path: Path):
    data = tmp_path / "curated"
    rows = [
        {"id": "s0", "target_tokens": ["ALL_ATOM_CARTESIAN:protein:CA"], "nodes": [], "edges": []},
        {
            "id": "d0",
            "target_tokens": ["ALL_ATOM_CONTACT:protein:CA:CB", "INTERNAL_COORD:protein_phi"],
            "nodes": [],
            "edges": [],
        },
        {
            "id": "d1",
            "target_tokens": ["CARTESIAN_ATOM:dna:P", "AFFINITY_CONTACT:strong"],
            "nodes": [],
            "edges": [],
        },
        {"id": "plain", "target_tokens": ["ANSWER:no-structure"], "nodes": [], "edges": []},
    ]
    _write_jsonl(data / "train.jsonl", rows)
    _write_jsonl(data / "val.jsonl", [])
    _write_jsonl(data / "test.jsonl", [])

    static_summary = build_subset(data, tmp_path / "static", 10, "static", 0.0, 0.0)
    dynamics_summary = build_subset(data, tmp_path / "dynamics", 10, "structure_dynamics", 0.0, 0.0)

    assert static_summary["selected_rows"] == 3
    assert dynamics_summary["selected_rows"] == 2
    static_rows = (tmp_path / "static" / "train.jsonl").read_text(encoding="utf-8")
    dynamics_rows = (tmp_path / "dynamics" / "train.jsonl").read_text(encoding="utf-8")
    assert "plain" not in static_rows
    assert "INTERNAL_COORD:protein_phi" in dynamics_rows
    assert "AFFINITY_CONTACT:strong" in dynamics_rows


def test_bio_phase_subsets_dereference_index_only_rows(tmp_path: Path):
    data = tmp_path / "curated"
    rows = [
        {"id": "plain", "target_tokens": ["ANSWER:no-structure"], "nodes": [], "edges": []},
        {
            "id": "d0",
            "target_tokens": ["ALL_ATOM_CONTACT:protein:CA:CB", "PPI_CONTACT:strong"],
            "nodes": [],
            "edges": [],
        },
    ]
    _write_indexed_jsonl(tmp_path / "source.jsonl", data / "train.jsonl", rows)
    _write_jsonl(data / "val.jsonl", [])
    _write_jsonl(data / "test.jsonl", [])

    dynamics_summary = build_subset(data, tmp_path / "dynamics", 10, "structure_dynamics", 0.0, 0.0)

    assert dynamics_summary["selected_rows"] == 1
    dynamics_rows = (tmp_path / "dynamics" / "train.jsonl").read_text(encoding="utf-8")
    assert "d0" in dynamics_rows
    assert "__jsonl_ref__" not in dynamics_rows


def test_check_bio_scale_targets_warns_only_for_source_limited_dna(tmp_path: Path):
    protein_summary = {
        "sources": {
            "uniprot_features": {
                "rows": 3_000_000,
            }
        }
    }
    bio_summary = {
        "per_dataset": {
            "uniprot_uniref50_sequence_train": 3_000_000,
            "pubchem10m_selfies_train": 3_000_000,
            "rfam_sequence_train": 1_500_000,
            "rnacentral_8192_sequence_train": 1_500_000,
            "dna_coding_regions_train": 1_677_609,
            "uniprot_function_text_train": 464_395,
        },
        "per_dataset_source_rows": {
            "uniprot_uniref50_sequence_train": 60_315_044,
            "pubchem10m_selfies_train": 9_999_999,
            "rfam_sequence_train": 20_051_822,
            "rnacentral_8192_sequence_train": 7_340_032,
            "dna_coding_regions_train": 1_677_609,
            "uniprot_function_text_train": 464_395,
        },
    }
    protein_path = tmp_path / "protein.json"
    bio_path = tmp_path / "bio.json"
    protein_path.write_text(json.dumps(protein_summary), encoding="utf-8")
    bio_path.write_text(json.dumps(bio_summary), encoding="utf-8")

    summary = check_targets(
        protein_summary_path=protein_path,
        bio_sequence_summary_path=bio_path,
        target_rows=3_000_000,
        allow_source_limited={"dna", "protein_function_text"},
    )

    assert summary["ok"] is True
    assert summary["modalities"]["protein"]["target_met"] is True
    assert summary["modalities"]["molecule"]["target_met"] is True
    assert summary["modalities"]["rna"]["target_met"] is True
    assert summary["modalities"]["dna"]["source_limited"] is True
    assert summary["warnings"]
