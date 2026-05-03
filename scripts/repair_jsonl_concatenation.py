#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm


def _offset_cache_paths(path: Path) -> tuple[Path, Path]:
    return (
        path.with_name(f"{path.name}.offsets.u64"),
        path.with_name(f"{path.name}.offsets.meta.json"),
    )


def _decode_concatenated_json(line: str) -> list[Any]:
    decoder = json.JSONDecoder()
    objects: list[Any] = []
    pos = 0
    length = len(line)
    while pos < length:
        while pos < length and line[pos].isspace():
            pos += 1
        if pos >= length:
            break
        obj, end = decoder.raw_decode(line, pos)
        objects.append(obj)
        pos = end
    return objects


def repair_jsonl(path: Path, *, dry_run: bool = False) -> dict[str, int | str | bool]:
    if not path.exists():
        raise FileNotFoundError(path)

    tmp_path = path.with_name(f".{path.name}.repair.tmp")
    input_lines = 0
    output_lines = 0
    repaired_lines = 0
    split_objects = 0
    total_bytes = path.stat().st_size

    with path.open("r", encoding="utf-8", errors="replace") as source:
        sink = None if dry_run else tmp_path.open("w", encoding="utf-8")
        try:
            with tqdm(total=total_bytes, desc=f"repair/{path.name}", unit="B", unit_scale=True) as pbar:
                for line in source:
                    input_lines += 1
                    pbar.update(len(line.encode("utf-8", errors="replace")))
                    stripped = line.strip()
                    if not stripped:
                        continue

                    # The interrupted-resume failure mode appends one JSON object
                    # immediately after another, yielding `}{` in a single JSONL row.
                    if "}{" not in stripped:
                        if sink is not None:
                            sink.write(stripped)
                            sink.write("\n")
                        output_lines += 1
                        continue

                    objects = _decode_concatenated_json(stripped)
                    if len(objects) <= 1:
                        if sink is not None:
                            sink.write(stripped)
                            sink.write("\n")
                        output_lines += 1
                        continue

                    repaired_lines += 1
                    split_objects += len(objects)
                    output_lines += len(objects)
                    if sink is not None:
                        for obj in objects:
                            sink.write(json.dumps(obj, ensure_ascii=False, sort_keys=False))
                            sink.write("\n")
        finally:
            if sink is not None:
                sink.close()

    if not dry_run:
        tmp_path.replace(path)
        for cache_path in _offset_cache_paths(path):
            cache_path.unlink(missing_ok=True)
    else:
        tmp_path.unlink(missing_ok=True)

    return {
        "path": str(path),
        "dry_run": dry_run,
        "input_lines": input_lines,
        "output_lines": output_lines,
        "repaired_lines": repaired_lines,
        "split_objects": split_objects,
        "line_delta": output_lines - input_lines,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repair JSONL files where an interrupted resume appended two JSON objects onto one line."
    )
    parser.add_argument("--path", action="append", required=True, help="JSONL path to repair. May be repeated.")
    parser.add_argument("--dry-run", action="store_true", help="Scan and report without replacing files.")
    args = parser.parse_args()

    results = [repair_jsonl(Path(path), dry_run=args.dry_run) for path in args.path]
    print(json.dumps({"ok": True, "results": results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
