from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
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


def fetch_rows(dataset: str, config: str, split: str, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    page = 100
    pbar = tqdm(total=limit, desc=f"acquire/{dataset}:{split}")
    while len(rows) < limit:
        length = min(page, limit - len(rows))
        payload = _get_json("rows", {"dataset": dataset, "config": config, "split": split, "offset": offset, "length": length})
        page_rows = [row["row"] for row in payload.get("rows", [])]
        if not page_rows:
            break
        rows.extend(page_rows)
        offset += len(page_rows)
        pbar.update(len(page_rows))
        if len(page_rows) < length:
            break
    pbar.close()
    return rows


def _drop_fields(rows: list[dict[str, Any]], drop_fields: list[str]) -> list[dict[str, Any]]:
    if not drop_fields:
        return rows
    drop = set(drop_fields)
    return [{key: value for key, value in row.items() if key not in drop} for row in rows]


def clone_repo(repo_url: str, target_dir: Path, ref: str | None = None, recurse_submodules: bool = False) -> Path:
    repo_dir = target_dir / "repo"
    if repo_dir.exists():
        return repo_dir
    cmd = ["git", "clone", "--depth", "1"]
    if recurse_submodules:
        cmd.append("--recurse-submodules")
    if ref:
        cmd.extend(["--branch", ref])
    cmd.extend([repo_url, str(repo_dir)])
    subprocess.run(cmd, check=True)
    return repo_dir


def _synthetic_rows(name: str, count: int) -> list[dict[str, Any]]:
    if name == "ugm_multimodal_synthetic":
        from iska_reasoner.data.multimodal import iter_synthetic_multimodal_examples

        return list(iter_synthetic_multimodal_examples(count=max(1, count)))
    return []


def acquire_from_manifest(
    manifest_path: str | Path,
    out_dir: str | Path,
    dataset_name: str | None,
    limit: int | None,
    dry_run: bool = False,
    fail_fast: bool = False,
) -> list[Path]:
    manifest = load_yaml(manifest_path)
    out_base = ensure_dir(out_dir)
    written: list[Path] = []
    datasets = manifest.get("datasets", [])
    for item in datasets:
        name = item["name"]
        if dataset_name and name != dataset_name:
            continue
        method = item.get("method", "hf_rows")
        default_limit = int(item.get("default_limit", 50))
        n = int(limit or default_limit)
        target_dir = ensure_dir(out_base / name)
        meta = {
            "name": name,
            "dataset_id": item.get("dataset_id"),
            "config": item.get("config"),
            "split": item.get("split"),
            "license": item.get("license"),
            "method": method,
            "limit": n,
            "drop_fields": item.get("drop_fields", []),
            "acquired_at": dt.datetime.utcnow().isoformat() + "Z",
        }
        try:
            if dry_run or item.get("manifest_only", False):
                write_jsonl(target_dir / "PROVENANCE.jsonl", [meta | {"dry_run": True, "manifest_only": bool(item.get("manifest_only", False))}])
                continue
            if method == "hf_rows":
                rows = fetch_rows(item["dataset_id"], item.get("config", "default"), item["split"], n)
                rows = _drop_fields(rows, [str(field) for field in item.get("drop_fields", [])])
            elif method == "git_clone":
                repo_dir = clone_repo(
                    str(item["repo_url"]),
                    target_dir,
                    ref=item.get("ref"),
                    recurse_submodules=bool(item.get("recurse_submodules", False)),
                )
                write_jsonl(target_dir / "PROVENANCE.jsonl", [meta | {"repo_url": item.get("repo_url"), "repo_dir": str(repo_dir)}])
                written.append(repo_dir)
                continue
            elif method == "local_generated":
                rows = _synthetic_rows(name, n)
            elif method == "synthetic":
                rows = []
            else:
                raise ValueError(f"Unsupported acquisition method {method} for {name}")
            path = target_dir / f"{item.get('split', 'train')}.jsonl"
            write_jsonl(path, rows)
            write_jsonl(target_dir / "PROVENANCE.jsonl", [meta | {"rows": len(rows)}])
            written.append(path)
        except Exception as exc:
            write_jsonl(
                target_dir / "PROVENANCE.jsonl",
                [
                    meta
                    | {
                        "error": str(exc),
                        "error_type": exc.__class__.__name__,
                        "rows": 0,
                    }
                ],
            )
            if fail_fast:
                raise
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Acquire small dataset samples from manifests.")
    parser.add_argument("--manifest", default="data/manifests/datasets.yaml")
    parser.add_argument("--out-dir", default="data/raw")
    parser.add_argument("--dataset")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on the first acquisition error instead of recording it and continuing.")
    args = parser.parse_args()
    paths = acquire_from_manifest(args.manifest, args.out_dir, args.dataset, args.limit, args.dry_run, fail_fast=args.fail_fast)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
