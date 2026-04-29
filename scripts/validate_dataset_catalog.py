#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iska_reasoner.data.catalog import build_catalog_status, write_catalog_markdown


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate dataset catalog implementation status against local files.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--manifest", default="data/manifests/datasets.yaml")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--raw-full-dir", default="data/raw_hf_full")
    parser.add_argument("--full-dir", default="data/processed/real_full_selected_mix")
    parser.add_argument("--audit", default="data/manifests/dataset_capacity_audit.json")
    parser.add_argument("--output-json", default="data/manifests/dataset_catalog_status.json")
    parser.add_argument("--output-md", default="planning/DATASET-CATALOG-STATUS.md")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--warn-only", action="store_true", help="Write reports but do not exit nonzero on readiness failures.")
    args = parser.parse_args()

    status = build_catalog_status(
        root=args.root,
        manifest_path=args.manifest,
        raw_dir=args.raw_dir,
        raw_full_dir=args.raw_full_dir,
        full_dir=args.full_dir,
        audit_path=args.audit,
        show_progress=not args.no_progress,
    )
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_catalog_markdown(status, args.output_md)
    print(json.dumps(status["summary"], indent=2, sort_keys=True))
    print(f"wrote {output_json}")
    print(f"wrote {args.output_md}")
    if not status["ready"] and not args.warn_only:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
