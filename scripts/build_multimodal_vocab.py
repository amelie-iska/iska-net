#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iska_reasoner.data.motifs import download_public_motif_sources, write_motif_vocabulary
from iska_reasoner.data.multimodal import write_multimodal_reference_tokens


def main() -> None:
    parser = argparse.ArgumentParser(description="Write neutral multimodal graph-to-graph reference tokens.")
    parser.add_argument("--output", default="data/processed/reference_tokens/multimodal_graph_tokens.txt")
    parser.add_argument("--motif-output", default="data/processed/reference_tokens/motif_graph_tokens.txt")
    parser.add_argument("--motif-summary", default="data/processed/reference_tokens/motif_graph_tokens.summary.json")
    parser.add_argument("--motif-source", action="append", default=[], help="Local PROSITE/InterPro/CATH/Rfam/motif rows file.")
    parser.add_argument("--download-public-motifs", action="store_true", help="Download public PROSITE, InterPro, CATH, and Rfam motif metadata before building tokens.")
    parser.add_argument("--motif-cache-dir", default="data/raw_motifs/public")
    parser.add_argument("--skip-interpro-download", action="store_true", help="When downloading public motifs, skip paginated InterPro API fetch.")
    args = parser.parse_args()
    motif_sources = [Path(path) for path in args.motif_source]
    downloaded = {}
    if args.download_public_motifs:
        downloaded = download_public_motif_sources(args.motif_cache_dir, include_interpro=not args.skip_interpro_download)
        motif_sources.extend(downloaded.values())
    motif_summary = write_motif_vocabulary(args.motif_output, motif_sources, summary_path=args.motif_summary)
    count = write_multimodal_reference_tokens(args.output, extra_motif_paths=motif_sources)
    print(
        json.dumps(
            {
                "tokens": count,
                "output": args.output,
                "motif_output": args.motif_output,
                "motif_summary": motif_summary,
                "downloaded": {key: str(value) for key, value in downloaded.items()},
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
