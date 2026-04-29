#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iska_reasoner.data.reference_repos import combined_reference_tokens, read_git_commit, write_tokens


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract SFM NatureLM and UniGenX reference tokens for GraphVocab extension.")
    parser.add_argument("--sfm-dir", default="data/external_repos/sfm")
    parser.add_argument("--unigenx-dir", default="data/external_repos/unigenx")
    parser.add_argument("--output", default="data/processed/reference_tokens/naturelm_unigenx_tokens.txt")
    args = parser.parse_args()

    sfm = Path(args.sfm_dir)
    unigenx = Path(args.unigenx_dir)
    tokens = combined_reference_tokens(sfm if sfm.exists() else None, unigenx if unigenx.exists() else None)
    count = write_tokens(tokens, args.output)
    summary = {
        "output": args.output,
        "token_count": count,
        "sfm_dir": str(sfm),
        "sfm_commit": read_git_commit(sfm) if sfm.exists() else "",
        "unigenx_dir": str(unigenx),
        "unigenx_commit": read_git_commit(unigenx) if unigenx.exists() else "",
    }
    Path(args.output).with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

