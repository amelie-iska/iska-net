#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iska_reasoner.utils.io import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge JSONL files without changing row contents.")
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = []
    for path in args.input:
        rows.extend(read_jsonl(path))
    count = write_jsonl(args.output, rows)
    print(f"Wrote {count} rows to {args.output}")


if __name__ == "__main__":
    main()

