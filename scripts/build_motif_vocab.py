#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iska_reasoner.data.motifs import download_public_motif_sources, write_motif_vocabulary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build complete UGM sequence/structure motif token vocabulary.")
    parser.add_argument("--output", default="data/processed/reference_tokens/motif_graph_tokens.txt")
    parser.add_argument("--summary", default="data/processed/reference_tokens/motif_graph_tokens.summary.json")
    parser.add_argument("--source", action="append", default=[], help="Local PROSITE/InterPro/CATH/Rfam/motif rows source file.")
    parser.add_argument("--download-public", action="store_true", help="Download public PROSITE, InterPro, CATH, and Rfam motif metadata.")
    parser.add_argument("--cache-dir", default="data/raw_motifs/public")
    parser.add_argument("--skip-interpro-download", action="store_true")
    args = parser.parse_args()

    sources = [Path(path) for path in args.source]
    downloaded = {}
    if args.download_public:
        downloaded = download_public_motif_sources(args.cache_dir, include_interpro=not args.skip_interpro_download)
        sources.extend(downloaded.values())
    summary = write_motif_vocabulary(args.output, sources, summary_path=args.summary)
    summary["downloaded"] = {key: str(value) for key, value in downloaded.items()}
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
