#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import yaml


def _max_stat(summary: dict, key: str) -> int:
    value = summary.get("global", {}).get(key, {}).get("max", 0)
    return int(value or 0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a 2x context YAML override from an existing context_requirements.json summary.")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context-multiplier", type=float, default=2.0)
    args = parser.parse_args()

    data = json.loads(args.summary.read_text(encoding="utf-8"))
    max_seq = _max_stat(data, "model_sequence_tokens_untruncated")
    max_source = _max_stat(data, "source_graph_tokens")
    max_target = _max_stat(data, "target_tokens")
    recommended_seq = max(1, int(math.ceil(max_seq * args.context_multiplier)))
    output = {
        "model": {"max_seq_len": recommended_seq},
        "data": {
            "max_source_tokens": max_source,
            "max_target_tokens": max_target,
            "max_seq_len": recommended_seq,
        },
        "context_audit": {
            "context_multiplier": float(args.context_multiplier),
            "max_model_sequence_tokens_untruncated": max_seq,
            "max_source_tokens_required": max_source,
            "max_target_tokens_required": max_target,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(output, sort_keys=False), encoding="utf-8")
    print(json.dumps({"output": str(args.output), **output["context_audit"]}, indent=2))


if __name__ == "__main__":
    main()
