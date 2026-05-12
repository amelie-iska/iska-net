#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iska_reasoner.inference.categorical_jacobian import (
    compute_glm2_hidden_state_jacobian_contacts,
    load_categorical_jacobian_contacts,
)


def _parse_spans(text: str | None) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    if not text:
        return spans
    for item in text.split(","):
        parts = item.split(":")
        if len(parts) < 3:
            continue
        spans.append((int(parts[0]), int(parts[1]), parts[2]))
    return spans


def main() -> None:
    parser = argparse.ArgumentParser(description="Build contact priors from cached or computed categorical-Jacobian maps.")
    parser.add_argument("--input", help="Cached JSON with either {'contacts': ...} or {'scores': matrix, 'spans': ...}.")
    parser.add_argument("--sequence", help="Optional gLM2 mixed-modality sequence for approximate hidden-state Jacobian computation.")
    parser.add_argument("--spans", help="Comma-separated start:end:label spans for inter-element contact typing.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="tattabio/gLM2_650M")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top-k", type=int, default=256)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--min-separation", type=int, default=6)
    parser.add_argument("--max-positions", type=int, default=256)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    if args.input:
        result = load_categorical_jacobian_contacts(args.input, top_k=args.top_k, min_score=args.min_score, min_separation=args.min_separation)
    elif args.sequence:
        result = compute_glm2_hidden_state_jacobian_contacts(
            args.sequence,
            spans=_parse_spans(args.spans),
            model_name=args.model,
            device=args.device,
            top_k=args.top_k,
            max_positions=args.max_positions,
            strict=args.strict,
        )
    else:
        raise SystemExit("Provide either --input or --sequence")
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"available": result.available, "contacts": len(result.contacts), "output": str(path), "message": result.message}, indent=2))


if __name__ == "__main__":
    main()
