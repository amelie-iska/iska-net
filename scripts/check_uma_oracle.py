#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iska_reasoner.graph.schema import GraphExample, Node
from iska_reasoner.oracles import fairchem_repo_status, score_uma_oracle_candidate


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the UGM FairChem/UMA oracle wiring.")
    parser.add_argument("--repo", default="data/external_repos/fairchem")
    parser.add_argument("--smiles", default="", help="Optional SMILES candidate to score.")
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--backend", choices=["fairchem", "proxy"], default="fairchem")
    parser.add_argument("--strict", action="store_true", help="Raise if FairChem/UMA scoring cannot run.")
    parser.add_argument("--model-name", default="uma-s-1p2")
    parser.add_argument("--task-name", default="omol")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    status = fairchem_repo_status(args.repo)
    payload: dict[str, object] = {"fairchem": status}
    if args.smiles:
        example = GraphExample(
            id="uma_check",
            task="multimodal_oracle_check",
            nodes=[
                Node(id="task", type="task", value="score candidate with UMA"),
                Node(id="smiles", type="smiles", value=args.smiles),
                Node(id="temperature", type="temperature", value=f"{args.temperature:.3f}K", features={"kelvin": args.temperature}),
            ],
            edges=[],
            target_tokens=["SMILES:" + args.smiles, "UGM:oracle:uma_feedback"],
            metadata={"smiles": args.smiles, "temperature": args.temperature},
        )
        result = score_uma_oracle_candidate(
            example,
            ["SMILES:" + args.smiles, "UGM:oracle:uma_feedback"],
            backend=args.backend,
            strict=args.strict,
            repo_path=args.repo,
            model_name=args.model_name,
            task_name=args.task_name,
            device=args.device,
        )
        payload["score"] = asdict(result)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
