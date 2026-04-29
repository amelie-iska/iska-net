#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def module_exists(path: Path) -> bool:
    return path.exists() and importlib.util.spec_from_file_location(path.stem, path) is not None


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe upstream TokenGT/Fairseq checkout readiness.")
    parser.add_argument("--tokengt-dir", default="tokengt")
    parser.add_argument("--output")
    args = parser.parse_args()
    root = Path(args.tokengt_dir)
    large = root / "large-scale-regression" / "tokengt"
    fairseq = root / "large-scale-regression" / "fairseq"
    report = {
        "tokengt_dir": str(root),
        "exists": root.exists(),
        "graph_encoder_exists": module_exists(large / "modules" / "tokengt_graph_encoder.py"),
        "tokenizer_exists": module_exists(large / "modules" / "tokenizer.py"),
        "fairseq_submodule_exists": fairseq.exists(),
        "fairseq_python_package_exists": (fairseq / "fairseq").exists(),
        "ready_for_upstream_training": root.exists() and fairseq.exists() and (fairseq / "fairseq").exists(),
        "recommendation": "Use local compact model unless Fairseq submodule and dependencies are installed.",
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

