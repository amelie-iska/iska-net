from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from iska_reasoner.utils.config import load_yaml
from iska_reasoner.utils.io import ensure_dir, write_jsonl

DATASET_SERVER = "https://datasets-server.huggingface.co"


def _get_json(endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
    query = urllib.parse.urlencode(params)
    url = f"{DATASET_SERVER}/{endpoint}?{query}"
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def _download(url: str, path: Path, expected_size: int | None = None, force: bool = False) -> str:
    ensure_dir(path.parent)
    if path.exists() and not force and (expected_size is None or path.stat().st_size == expected_size):
        return "reused"
    tmp = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as response:
        total = expected_size
        if total is None:
            header = response.headers.get("Content-Length")
            total = int(header) if header and header.isdigit() else None
        with tmp.open("wb") as handle, tqdm(total=total, unit="B", unit_scale=True, desc=path.name) as pbar:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                pbar.update(len(chunk))
    tmp.replace(path)
    return "downloaded"


def _selected_parquet_files(item: dict[str, Any]) -> list[dict[str, Any]]:
    payload = _get_json("parquet", {"dataset": item["dataset_id"]})
    config = item.get("config", "default")
    split = item.get("split")
    files = [
        file
        for file in payload.get("parquet_files", [])
        if file.get("config") == config and file.get("split") == split and file.get("url")
    ]
    max_files = item.get("full_training_max_parquet_files")
    if max_files is not None:
        files = files[: int(max_files)]
    max_bytes = item.get("full_training_max_parquet_bytes")
    if max_bytes is not None:
        selected: list[dict[str, Any]] = []
        total = 0
        for file in files:
            size = int(file.get("size") or 0)
            if selected and total + size > int(max_bytes):
                break
            selected.append(file)
            total += size
        files = selected
    return files


def download_selected_splits(
    manifest_path: str | Path,
    out_dir: str | Path,
    dataset_name: str | None = None,
    max_total_bytes: int | None = None,
    force: bool = False,
) -> list[Path]:
    manifest = load_yaml(manifest_path)
    out_base = ensure_dir(out_dir)
    jobs: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    total_bytes = 0
    for item in manifest.get("datasets", []):
        name = item["name"]
        if dataset_name and name != dataset_name:
            continue
        if item.get("method", "hf_rows") != "hf_rows" or item.get("manifest_only", False):
            continue
        if item.get("full_training_enabled", True) is False:
            continue
        if not item.get("dataset_id"):
            continue
        files = _selected_parquet_files(item)
        total_bytes += sum(int(file.get("size") or 0) for file in files)
        jobs.append((item, files))
    free_bytes = shutil.disk_usage(Path.cwd()).free
    if max_total_bytes is not None and total_bytes > max_total_bytes:
        raise RuntimeError(f"Selected parquet files require {total_bytes} bytes, above max_total_bytes={max_total_bytes}")
    if total_bytes > free_bytes:
        raise RuntimeError(f"Selected parquet files require {total_bytes} bytes, but only {free_bytes} bytes are free")

    written: list[Path] = []
    for item, files in jobs:
        name = item["name"]
        config = item.get("config", "default")
        split = item.get("split", "train")
        target_dir = ensure_dir(out_base / name / config / split)
        statuses = []
        for file in files:
            target = target_dir / str(file.get("filename", "data.parquet"))
            status = _download(str(file["url"]), target, expected_size=int(file.get("size") or 0), force=force)
            statuses.append({"path": str(target), "status": status, "size": file.get("size"), "url": file.get("url")})
            written.append(target)
        write_jsonl(
            out_base / name / "PROVENANCE.jsonl",
            [
                {
                    "name": name,
                    "dataset_id": item.get("dataset_id"),
                    "config": config,
                    "split": split,
                    "method": "hf_parquet_selected_split",
                    "license": item.get("license"),
                    "downloaded_at": dt.datetime.utcnow().isoformat() + "Z",
                    "files": statuses,
                    "total_bytes": sum(int(file.get("size") or 0) for file in files),
                }
            ],
        )
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Download full selected HF parquet splits from the dataset manifest.")
    parser.add_argument("--manifest", default="data/manifests/datasets.yaml")
    parser.add_argument("--out-dir", default="data/raw_hf_full")
    parser.add_argument("--dataset")
    parser.add_argument("--max-total-gib", type=float, default=32.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    max_total_bytes = int(args.max_total_gib * 1024**3) if args.max_total_gib is not None else None
    paths = download_selected_splits(args.manifest, args.out_dir, args.dataset, max_total_bytes=max_total_bytes, force=args.force)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
