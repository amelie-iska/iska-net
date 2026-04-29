#!/usr/bin/env python
from __future__ import annotations

import argparse
import fnmatch
import json
import sys
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tqdm.auto import tqdm

from iska_reasoner.utils.config import load_yaml
from iska_reasoner.utils.io import ensure_dir


HF_API = "https://huggingface.co/api/models"
HF_RESOLVE = "https://huggingface.co"

def fetch_model_info(repo_id: str) -> dict[str, Any]:
    url = f"{HF_API}/{urllib.parse.quote(repo_id, safe='/')}"
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def clone_or_update_github(repo_url: str, target: Path, ref: str | None = None) -> Path:
    if target.exists():
        if (target / ".git").exists():
            subprocess.run(["git", "-C", str(target), "fetch", "--depth", "1", "origin", ref or "main"], check=False)
        return target
    ensure_dir(target.parent)
    cmd = ["git", "clone", "--depth", "1"]
    if ref:
        cmd.extend(["--branch", ref])
    cmd.extend([repo_url, str(target)])
    subprocess.run(cmd, check=True)
    return target


def github_repo_metadata(entry: dict[str, Any], out_base: Path) -> dict[str, Any]:
    repo_url = entry["repo_url"]
    local_dir = Path(entry.get("local_dir") or out_base)
    repo_dir = clone_or_update_github(repo_url, local_dir, entry.get("ref"))
    commit = ""
    try:
        proc = subprocess.run(["git", "-C", str(repo_dir), "rev-parse", "HEAD"], text=True, capture_output=True, timeout=5, check=True)
        commit = proc.stdout.strip()
    except Exception:
        pass
    files = []
    for path in sorted(p for p in repo_dir.rglob("*") if p.is_file() and ".git" not in p.parts):
        rel = path.relative_to(repo_dir).as_posix()
        try:
            size = path.stat().st_size
        except OSError:
            size = None
        files.append({"filename": rel, "size": size})
    return {
        "repo_id": entry["name"],
        "provider": "github",
        "repo_url": repo_url,
        "local_dir": str(repo_dir),
        "commit": commit,
        "license": entry.get("license"),
        "files": files,
        "notes": entry.get("notes", ""),
    }


def siblings(info: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in info.get("siblings", []):
        name = item.get("rfilename") or item.get("path")
        if not name:
            continue
        rows.append(
            {
                "filename": name,
                "size": item.get("size"),
                "lfs": item.get("lfs"),
            }
        )
    return sorted(rows, key=lambda row: row["filename"])


def matches_any(filename: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(filename, pattern) for pattern in patterns)


def download_file(repo_id: str, filename: str, out_path: Path) -> None:
    url = f"{HF_RESOLVE}/{repo_id}/resolve/main/{urllib.parse.quote(filename, safe='/')}"
    ensure_dir(out_path.parent)
    with urllib.request.urlopen(url, timeout=120) as response, out_path.open("wb") as f:
        total = int(response.headers.get("content-length") or 0)
        with tqdm(total=total if total > 0 else None, unit="B", unit_scale=True, desc=f"model/{filename}") as pbar:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                pbar.update(len(chunk))


def main() -> None:
    parser = argparse.ArgumentParser(description="Record GitHub or Hugging Face model/reference repo metadata and optionally download small selected HF files.")
    parser.add_argument("--manifest", default="data/manifests/model_repos.yaml")
    parser.add_argument("--repo-name", default="unigenx")
    parser.add_argument("--out-dir", default="data/external_models")
    parser.add_argument("--download", action="store_true", help="Download selected files within --max-file-mb.")
    parser.add_argument("--pattern", action="append", help="Glob pattern to download. Defaults to manifest patterns.")
    parser.add_argument("--max-file-mb", type=float, help="Maximum remote file size to download.")
    args = parser.parse_args()

    manifest = load_yaml(args.manifest)
    entries = {entry["name"]: entry for entry in manifest.get("model_repos", [])}
    if args.repo_name not in entries:
        raise SystemExit(f"Unknown repo name {args.repo_name}. Available: {sorted(entries)}")
    entry = entries[args.repo_name]
    provider = entry.get("provider", "hf_model")
    if provider == "github":
        out_base = ensure_dir(Path(args.out_dir) / entry["name"])
        metadata = github_repo_metadata(entry, out_base)
        (out_base / "FILES.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
        summary = {"metadata_path": str(out_base / "FILES.json"), "downloaded": [], "skipped": [], "file_count": len(metadata["files"])}
        (out_base / "SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    repo_id = entry["repo_id"]
    info = fetch_model_info(repo_id)
    files = siblings(info)
    out_base = ensure_dir(Path(args.out_dir) / entry["name"])
    metadata = {
        "repo_id": repo_id,
        "license": entry.get("license"),
        "sha": info.get("sha"),
        "private": info.get("private"),
        "gated": info.get("gated"),
        "downloads": info.get("downloads"),
        "likes": info.get("likes"),
        "tags": info.get("tags", []),
        "files": files,
        "notes": entry.get("notes", ""),
    }
    (out_base / "FILES.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    patterns = args.pattern or entry.get("default_download_patterns", [])
    max_file_mb = float(args.max_file_mb if args.max_file_mb is not None else entry.get("default_max_file_mb", 1))
    downloaded = []
    skipped = []
    if args.download:
        for file_info in files:
            filename = file_info["filename"]
            size = file_info.get("size")
            size_mb = (float(size) / (1024**2)) if isinstance(size, (int, float)) else None
            if not matches_any(filename, patterns):
                continue
            if size_mb is not None and size_mb > max_file_mb:
                skipped.append({"filename": filename, "reason": "too_large", "size_mb": size_mb})
                continue
            target = out_base / filename
            try:
                download_file(repo_id, filename, target)
                downloaded.append({"filename": filename, "path": str(target)})
            except Exception as exc:
                skipped.append({"filename": filename, "reason": "download_failed", "error": repr(exc)})
    summary = {"metadata_path": str(out_base / "FILES.json"), "downloaded": downloaded, "skipped": skipped, "file_count": len(files)}
    (out_base / "SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
