#!/usr/bin/env python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iska_reasoner.training.stage_runner import run_training_stage
from iska_reasoner.utils.config import load_config


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train a configured stage.")
    parser.add_argument("--config", action="append", required=True, help="YAML config path. May be passed multiple times.")
    args = parser.parse_args()
    run_training_stage(load_config(args.config))


if __name__ == "__main__":
    main()
