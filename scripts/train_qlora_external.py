#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Opt-in QLoRA/PEFT SFT entry point for external Hugging Face LMs.")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    try:
        import bitsandbytes  # noqa: F401
        import peft  # noqa: F401
        import transformers  # noqa: F401
        import trl  # noqa: F401
    except Exception as exc:
        raise SystemExit(
            "QLoRA external training requires optional packages: transformers, peft, trl, bitsandbytes. "
            f"Install them in a dedicated environment first. Import error: {exc!r}"
        )

    if args.dry_run:
        print(json.dumps({"model_id": args.model_id, "train_jsonl": args.train_jsonl, "output_dir": args.output_dir, "status": "dependencies_available"}, indent=2))
        return

    raise SystemExit(
        "External QLoRA is intentionally opt-in and model-specific. Dependencies are present, but this script stops before "
        "launching a costly run. Add tokenizer/template mapping for the selected model and invoke TRL SFTTrainer here."
    )


if __name__ == "__main__":
    main()

