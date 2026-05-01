#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iska_reasoner.data.dataset import GraphJsonlDataset
from iska_reasoner.data.vocab import GraphVocab


SCOPED_SEARCH_DIRS = ["README.md", "planning", "src", "scripts", "config", "tests", "assets"]
REQUIRED_FILES = [
    "data/processed/reference_tokens/naturelm_unigenx_tokens.txt",
    "data/processed/reference_tokens/motif_graph_tokens.txt",
    "data/processed/reference_tokens/motif_graph_tokens.summary.json",
    "data/processed/reference_tokens/multimodal_graph_tokens.txt",
    "data/raw_motifs/public/prosite.dat",
    "data/raw_motifs/public/interpro_entries.json",
    "data/raw_motifs/public/cath-names.txt",
    "data/raw_motifs/public/rfam-family.txt.gz",
    "data/processed/multimodal_graphs/train.jsonl",
    "data/processed/multimodal_graphs/val.jsonl",
    "data/processed/multimodal_graphs/test.jsonl",
    "data/external_repos/fairchem/README.md",
    "data/external_repos/Tropical-Attention/README.md",
    "config/inference/multimodal_tiny_inference.yaml",
    "config/model/tiny_tokengt_tropical.yaml",
    "config/train/graph_pretrain_tropical_attention_tiny.yaml",
    "config/train/overrides/tropical_attention_backend.yaml",
    "config/model/overrides/tropical_attention_backend.yaml",
    "config/model/overrides/hybrid_flash_mhta_backend.yaml",
    "config/generated/real_full_selected_context_compact.yaml",
    "config/train/multimodal_phase2_tiny.yaml",
    "config/train/multimodal_oracle_gflownet_tiny.yaml",
    "config/validate/multimodal_validation.yaml",
    "config/validate/multimodal_test.yaml",
    "config/validate/multimodal_gflownet_validation.yaml",
    "config/train/multimodal_phase2_4090.yaml",
    "config/train/multimodal_oracle_gflownet_4090.yaml",
    "config/train/structure_dynamics_4090.yaml",
    "config/train/structure_dynamics_oracle_gflownet_4090.yaml",
    "config/validate/structure_dynamics_validation.yaml",
    "config/validate/structure_dynamics_test.yaml",
    "config/validate/structure_dynamics_gflownet_validation.yaml",
    "config/inference/structure_dynamics_inference.yaml",
    "scripts/check_dataset_integrity.py",
    "scripts/run_full_training_sequence.sh",
    "scripts/run_full_phase1_phase2_training.sh",
    "scripts/validate_dataset_catalog.py",
    "scripts/build_motif_vocab.py",
    "scripts/download_uma_weights.py",
    "scripts/prepare_structure_dynamics_sources.py",
    "planning/DATASET-CATALOG-SPLITS.md",
    "planning/DATASET-CATALOG-IMPLEMENTATION-PLAN.md",
    "planning/DATASET-CATALOG-STATUS.md",
    "data/manifests/dataset_catalog_status.json",
    "planning/STRUCTURE-DYNAMICS-TRAINING.md",
    "planning/UGM-FULL-DATASET-DIFF-AUDIT.md",
    "planning/UGM-FULL-DATASET-IMPLEMENTATION-PLAN.md",
    "planning/UGM-SYNTHETIC-FAUX-CODE-AUDIT.md",
    "planning/PLAN-H.md",
    "README.md",
]
REQUIRED_VOCAB_TOKENS = [
    "UGM:graph_to_graph",
    "UGM:oracle:uma_feedback",
    "UGM:decoder:random_order_ar",
    "AA:W",
    "SELFIES:[=O]",
    "BOND:phosphodiester",
    "TEMP:300K",
    "PDB:MODEL",
    "FORCE:dir:px",
    "SEQ_MOTIF:core:coiled_coil",
    "SEQ_MOTIF:interpro:IPR000001",
    "SEQ_MOTIF:prosite:PS00001",
    "SEQ_MOTIF:rfam:RF00001",
    "STRUCT_MOTIF:cath:1.10.8.10",
    "STRUCT_MOTIF:core:cath_domain",
    "STRUCT_DERIVED_SEQ_MOTIF:cath:1.10.8.10",
    "STRUCT_DERIVED_SEQ_MOTIF:core:contact_patch_sequence",
]
DISALLOWED_TERMS = ["R" + "O" + "A" + "R", "r" + "o" + "a" + "r", "iska-" + "r" + "o" + "a" + "r"]


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _terminology_hits(root: Path) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for item in SCOPED_SEARCH_DIRS:
        path = root / item
        paths = [path] if path.is_file() else [p for p in path.rglob("*") if p.is_file()]
        for file_path in paths:
            if "__pycache__" in file_path.parts or file_path.suffix in {".pyc", ".pyo"}:
                continue
            text = _read_text(file_path)
            for term in DISALLOWED_TERMS:
                if term in text:
                    hits.append({"path": str(file_path.relative_to(root)), "term": term})
    return hits


def assess(root: Path) -> dict[str, Any]:
    files = [{"path": path, "exists": (root / path).exists()} for path in REQUIRED_FILES]
    train_count = val_count = test_count = 0
    try:
        train_count = len(GraphJsonlDataset(root / "data/processed/multimodal_graphs/train.jsonl"))
        val_count = len(GraphJsonlDataset(root / "data/processed/multimodal_graphs/val.jsonl"))
        test_count = len(GraphJsonlDataset(root / "data/processed/multimodal_graphs/test.jsonl"))
    except Exception:
        pass
    vocab_tokens: list[str] = []
    vocab_path = root / "outputs/multimodal_phase2_tiny/vocab.jsonl"
    if vocab_path.exists():
        vocab = GraphVocab.load(vocab_path)
        vocab_tokens = [tok for tok in REQUIRED_VOCAB_TOKENS if tok in vocab.token_to_id]
    reference_text = _read_text(root / "data/processed/reference_tokens/multimodal_graph_tokens.txt")
    reference_tokens = [tok for tok in REQUIRED_VOCAB_TOKENS if tok in reference_text.splitlines()]
    motif_summary: dict[str, Any] = {}
    motif_summary_path = root / "data/processed/reference_tokens/motif_graph_tokens.summary.json"
    if motif_summary_path.exists():
        try:
            motif_summary = json.loads(motif_summary_path.read_text(encoding="utf-8"))
        except Exception:
            motif_summary = {}
    dataset_catalog_status: dict[str, Any] = {}
    catalog_status_path = root / "data/manifests/dataset_catalog_status.json"
    if catalog_status_path.exists():
        try:
            dataset_catalog_status = json.loads(catalog_status_path.read_text(encoding="utf-8"))
        except Exception:
            dataset_catalog_status = {}
    terminology_hits = _terminology_hits(root)
    checkpoint_exists = (root / "outputs/multimodal_phase2_tiny/checkpoint_final.pt").exists()
    gflownet_checkpoint_exists = (root / "outputs/multimodal_oracle_gflownet_tiny/gflownet_final.pt").exists()
    docs = _read_text(root / "README.md") + "\n" + _read_text(root / "planning/PLAN-H.md")
    docs += "\n" + _read_text(root / "planning/STRUCTURE-DYNAMICS-TRAINING.md")
    doc_mentions = all(
        term in docs
        for term in [
            "Universal Graph Model",
            "NatureLM",
            "UniGenX",
            "FairChem",
            "oracle-feedback GFlowNet",
            "multimodal",
            "structure/dynamics",
            "motif vocabulary",
        ]
    )
    catalog_ready = bool(dataset_catalog_status.get("ready"))
    motif_summary_ok = (
        motif_summary.get("tokens", 0) >= 100000
        and motif_summary.get("records", 0) >= 70000
        and motif_summary.get("by_source", {}).get("interpro", 0) >= 50000
        and motif_summary.get("by_source", {}).get("prosite", 0) >= 2000
        and motif_summary.get("by_source", {}).get("rfam", 0) >= 4000
        and motif_summary.get("by_source", {}).get("cath", 0) >= 10000
    )
    ready = (
        all(item["exists"] for item in files)
        and train_count > 0
        and val_count > 0
        and test_count > 0
        and len(reference_tokens) == len(REQUIRED_VOCAB_TOKENS)
        and motif_summary_ok
        and catalog_ready
        and not terminology_hits
        and checkpoint_exists
        and doc_mentions
    )
    return {
        "ready_to_roll": ready,
        "required_files": files,
        "multimodal_train_examples": train_count,
        "multimodal_val_examples": val_count,
        "multimodal_test_examples": test_count,
        "reference_vocab_required_tokens": reference_tokens,
        "trained_vocab_required_tokens": vocab_tokens,
        "motif_summary": motif_summary,
        "motif_summary_ok": motif_summary_ok,
        "dataset_catalog_ready": catalog_ready,
        "dataset_catalog_summary": dataset_catalog_status.get("summary", {}),
        "phase2_checkpoint_exists": checkpoint_exists,
        "gflownet_checkpoint_exists": gflownet_checkpoint_exists,
        "terminology_hits": terminology_hits,
        "docs_cover_ugm_methodology": doc_mentions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Assess UGM ready-to-roll project state.")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    print(json.dumps(assess(Path(args.root).resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
