#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib
import json
import shutil
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm


DEFAULT_MODULES = [
    "wandb",
    "datasets",
    "transformers",
    "rdkit",
    "ase",
    "soundfile",
    "torchaudio",
    "ripser",
    "gudhi",
]


def check_module(name: str) -> dict[str, Any]:
    try:
        mod = importlib.import_module(name)
        return {"name": name, "available": True, "version": str(getattr(mod, "__version__", ""))}
    except Exception as exc:
        return {"name": name, "available": False, "error": repr(exc)}


def path_status(path: str) -> dict[str, Any]:
    p = Path(path)
    return {"path": path, "exists": p.exists(), "is_file": p.is_file(), "is_dir": p.is_dir()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Check local readiness for training, validation, and inference.")
    parser.add_argument("--json", action="store_true", help="Emit compact JSON only.")
    args = parser.parse_args()

    modules = [check_module(name) for name in tqdm(DEFAULT_MODULES, desc="readiness/modules", disable=args.json)]
    paths = [
        path_status("data/external_repos/sfm"),
        path_status("data/external_repos/unigenx"),
        path_status("data/external_repos/fairchem"),
        path_status("data/processed/reference_tokens/naturelm_unigenx_tokens.txt"),
        path_status("data/processed/reference_tokens/motif_graph_tokens.txt"),
        path_status("data/processed/reference_tokens/motif_graph_tokens.summary.json"),
        path_status("data/processed/reference_tokens/multimodal_graph_tokens.txt"),
        path_status("data/raw_motifs/public/prosite.dat"),
        path_status("data/raw_motifs/public/interpro_entries.json"),
        path_status("data/raw_motifs/public/cath-names.txt"),
        path_status("data/raw_motifs/public/rfam-family.txt.gz"),
        path_status("config/train/science_sft_4090.yaml"),
        path_status("config/train/multimodal_phase2_tiny.yaml"),
        path_status("config/train/multimodal_phase2_4090.yaml"),
        path_status("config/train/structure_dynamics_4090.yaml"),
        path_status("scripts/check_dataset_integrity.py"),
        path_status("scripts/validate_dataset_catalog.py"),
        path_status("scripts/download_uma_weights.py"),
        path_status("scripts/check_uma_oracle.py"),
        path_status("planning/DATASET-CATALOG-SPLITS.md"),
        path_status("planning/DATASET-CATALOG-IMPLEMENTATION-PLAN.md"),
        path_status("planning/DATASET-CATALOG-STATUS.md"),
        path_status("data/manifests/dataset_catalog_status.json"),
        path_status("scripts/prepare_structure_dynamics_sources.py"),
        path_status("config/train/overrides/wandb_online.yaml"),
        path_status("planning/LICENSE-REVIEW.md"),
        path_status("planning/STRUCTURE-DYNAMICS-TRAINING.md"),
    ]
    summary: dict[str, Any] = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "torch_version": torch.__version__,
        "modules": modules,
        "paths": paths,
        "lean_available": shutil.which("lean") is not None,
        "elan_available": shutil.which("elan") is not None,
    }
    summary["ready_python_optionals"] = all(item["available"] for item in modules)
    summary["ready_reference_data"] = all(item["exists"] for item in paths[:11])

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
