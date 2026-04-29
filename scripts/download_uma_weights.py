#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iska_reasoner.graph.schema import GraphExample, Node
from iska_reasoner.oracles import fairchem_repo_status, score_uma_oracle_candidate


def _prepend_fairchem_src(repo: Path) -> None:
    src = repo / "src"
    if src.exists() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _size(path: str | Path) -> int:
    return Path(path).stat().st_size


def _download_or_resolve(repo: Path, model_name: str, cache_dir: str | None) -> dict[str, Any]:
    if cache_dir:
        os.environ["FAIRCHEM_CACHE_DIR"] = cache_dir
    _prepend_fairchem_src(repo)
    from fairchem.core._config import CACHE_DIR  # type: ignore
    from fairchem.core.calculate import pretrained_mlip  # type: ignore

    checkpoint_path = pretrained_mlip.pretrained_checkpoint_path_from_name(model_name)
    refs: dict[str, dict[str, Any]] = {}
    for ref_type in ("atom_refs", "form_elem_refs"):
        try:
            ref_data = pretrained_mlip.get_reference_energies(model_name, ref_type, cache_dir=CACHE_DIR)
            model_checkpoint = pretrained_mlip._MODEL_CKPTS.checkpoints[model_name]
            file_data = getattr(model_checkpoint, ref_type)
            refs[ref_type] = {
                "available": True,
                "filename": file_data["filename"],
                "subfolder": file_data["subfolder"],
                "keys": sorted(str(key) for key in ref_data.keys())[:20],
            }
        except Exception as exc:
            refs[ref_type] = {
                "available": False,
                "error": f"{exc.__class__.__name__}: {exc}",
            }
    return {
        "cache_dir": CACHE_DIR,
        "model_name": model_name,
        "checkpoint_path": checkpoint_path,
        "checkpoint_exists": Path(checkpoint_path).exists(),
        "checkpoint_size_bytes": _size(checkpoint_path),
        "references": refs,
    }


def _score_smoke(repo: Path, model_name: str, task_name: str, device: str, smiles: str, temperature: float) -> dict[str, Any]:
    example = GraphExample(
        id="uma_download_smoke",
        task="uma_download_smoke",
        nodes=[
            Node(id="smiles", type="smiles", value=smiles),
            Node(id="temperature", type="temperature", value=f"{temperature:.3f}K", features={"kelvin": temperature}),
        ],
        edges=[],
        target_tokens=["SMILES:" + smiles, "UGM:oracle:uma_feedback"],
        metadata={"smiles": smiles, "temperature": temperature},
    )
    result = score_uma_oracle_candidate(
        example,
        ["SMILES:" + smiles, "UGM:oracle:uma_feedback"],
        backend="fairchem",
        strict=True,
        repo_path=repo,
        model_name=model_name,
        task_name=task_name,
        device=device,
    )
    return {
        "available": result.available,
        "reward": result.reward,
        "score": result.score,
        "energy_ev": result.energy_ev,
        "force_rms_ev_per_a": result.force_rms_ev_per_a,
        "atom_count": result.atom_count,
        "message": result.message,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Download or verify FairChem/UMA weights used by UGM oracle stages.")
    parser.add_argument("--repo", default="data/external_repos/fairchem", help="Local FairChem repository clone.")
    parser.add_argument("--model-name", default="uma-s-1p2")
    parser.add_argument("--task-name", default="omol")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-dir", default="", help="Optional FAIRCHEM_CACHE_DIR override.")
    parser.add_argument("--score-smoke", action="store_true", help="After resolving weights, run one strict ASE/FairChem score.")
    parser.add_argument("--smiles", default="CCO")
    parser.add_argument("--temperature", type=float, default=325.0)
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    status = fairchem_repo_status(repo)
    if not status.get("exists") or not status.get("importable"):
        raise SystemExit(json.dumps({"fairchem": status, "ok": False}, indent=2, sort_keys=True))
    resolved = _download_or_resolve(repo, args.model_name, args.cache_dir or None)
    payload: dict[str, Any] = {
        "ok": True,
        "fairchem": status,
        "weights": resolved,
        "score_smoke": None,
    }
    if args.score_smoke:
        payload["score_smoke"] = _score_smoke(repo, args.model_name, args.task_name, args.device, args.smiles, args.temperature)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
