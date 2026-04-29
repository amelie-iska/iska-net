from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from iska_reasoner.utils.config import load_yaml
from iska_reasoner.utils.io import ensure_dir

DATASET_SERVER = "https://datasets-server.huggingface.co"
DEFAULT_MIN_FREE_AFTER_BYTES = 512 * 1024**3


def _get_json(endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
    query = urllib.parse.urlencode(params)
    url = f"{DATASET_SERVER}/{endpoint}?{query}"
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def _human_bytes(value: int | None) -> str:
    if value is None:
        return "unknown"
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    size = float(value)
    for unit in units:
        if abs(size) < 1024.0 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{value} B"


def _path_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for root, _, files in os.walk(path):
        for file in files:
            try:
                total += (Path(root) / file).stat().st_size
            except OSError:
                continue
    return total


def _memory_info() -> dict[str, int | None]:
    info: dict[str, int | None] = {"mem_total_bytes": None, "mem_available_bytes": None, "swap_total_bytes": None, "swap_free_bytes": None}
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return info
    values: dict[str, int] = {}
    for line in meminfo.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, rest = line.split(":", 1)
        parts = rest.strip().split()
        if len(parts) >= 2 and parts[1].lower() == "kb":
            values[key] = int(parts[0]) * 1024
    info["mem_total_bytes"] = values.get("MemTotal")
    info["mem_available_bytes"] = values.get("MemAvailable")
    info["swap_total_bytes"] = values.get("SwapTotal")
    info["swap_free_bytes"] = values.get("SwapFree")
    return info


def _gpu_info() -> list[dict[str, Any]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
    except Exception:
        return []
    gpus = []
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        name, total_mib, free_mib = parts
        try:
            total = int(total_mib) * 1024**2
            free = int(free_mib) * 1024**2
        except ValueError:
            total = None
            free = None
        gpus.append({"name": name, "memory_total_bytes": total, "memory_free_bytes": free})
    return gpus


def _remote_hf_size(item: dict[str, Any]) -> dict[str, Any]:
    dataset_id = item.get("dataset_id")
    if not dataset_id:
        return {"available": False, "error": "missing dataset_id"}
    try:
        payload = _get_json("size", {"dataset": dataset_id})
    except Exception as exc:
        return {"available": False, "error": str(exc), "error_type": exc.__class__.__name__}
    size = payload.get("size", {})
    dataset = size.get("dataset", {}) or {}
    splits = size.get("splits", []) or []
    config = item.get("config", "default")
    split = item.get("split")
    selected_split = None
    for candidate in splits:
        if candidate.get("config") == config and candidate.get("split") == split:
            selected_split = candidate
            break
    return {
        "available": True,
        "partial": bool(payload.get("partial", False)),
        "dataset_num_rows": dataset.get("num_rows") or dataset.get("estimated_num_rows"),
        "dataset_num_bytes_original_files": dataset.get("num_bytes_original_files"),
        "dataset_num_bytes_parquet_files": dataset.get("num_bytes_parquet_files"),
        "dataset_num_bytes_memory": dataset.get("num_bytes_memory"),
        "split_num_rows": selected_split.get("num_rows") if selected_split else None,
        "split_num_bytes_parquet_files": selected_split.get("num_bytes_parquet_files") if selected_split else None,
        "split_num_bytes_memory": selected_split.get("num_bytes_memory") if selected_split else None,
        "split_found": selected_split is not None,
        "failed": payload.get("failed", []),
        "pending": payload.get("pending", []),
    }


def _entry_action(item: dict[str, Any], remote: dict[str, Any], disk_free: int, min_free_after: int) -> str:
    method = item.get("method", "hf_rows")
    if item.get("manifest_only", False):
        if method == "local_file":
            return "skip_local_user_provided"
        return "skip_manifest_only_large_or_restricted"
    if method == "local_file":
        return "skip_local_user_provided"
    if method == "git_clone":
        return "clone_or_reuse_git_repo"
    if method == "local_generated":
        return "generate_project_synthetic_rows"
    if method != "hf_rows":
        return "unsupported_method"
    limit = int(item.get("default_limit", 0))
    if limit <= 0:
        return "skip_no_default_limit"
    split_bytes = remote.get("split_num_bytes_parquet_files") if remote.get("available") else None
    if split_bytes is not None and split_bytes + min_free_after <= disk_free:
        return "download_manifest_sample_full_split_disk_feasible"
    return "download_manifest_sample_only"


def build_audit(manifest_path: Path, raw_dir: Path, hf_full_dir: Path, min_free_after: int) -> dict[str, Any]:
    manifest = load_yaml(manifest_path)
    disk = shutil.disk_usage(Path.cwd())
    entries = []
    known_full_hf_bytes = 0
    known_full_hf_count = 0
    local_raw_bytes = 0
    local_hf_full_bytes = _path_size(hf_full_dir)
    local_hf_full_file_count = len([path for path in hf_full_dir.glob("**/*.parquet")]) if hf_full_dir.exists() else 0
    for item in manifest.get("datasets", []):
        name = item["name"]
        target_dir = raw_dir / name
        local_bytes = _path_size(target_dir)
        local_raw_bytes += local_bytes
        remote = _remote_hf_size(item) if item.get("method", "hf_rows") == "hf_rows" and item.get("dataset_id") else {"available": False}
        split_bytes = remote.get("split_num_bytes_parquet_files") if remote.get("available") else None
        if isinstance(split_bytes, int):
            known_full_hf_bytes += split_bytes
            known_full_hf_count += 1
        entries.append(
            {
                "name": name,
                "method": item.get("method", "hf_rows"),
                "stage": item.get("stage"),
                "dataset_id": item.get("dataset_id"),
                "config": item.get("config"),
                "split": item.get("split"),
                "license": item.get("license"),
                "default_limit": int(item.get("default_limit", 0)),
                "manifest_only": bool(item.get("manifest_only", False)),
                "local_path": str(target_dir),
                "local_bytes": local_bytes,
                "local_human": _human_bytes(local_bytes),
                "remote": remote,
                "recommended_action": _entry_action(item, remote, disk.free, min_free_after),
            }
        )
    return {
        "created_at": dt.datetime.utcnow().isoformat() + "Z",
        "manifest": str(manifest_path),
        "raw_dir": str(raw_dir),
        "hf_full_dir": str(hf_full_dir),
        "capacity": {
            "disk_total_bytes": disk.total,
            "disk_used_bytes": disk.used,
            "disk_free_bytes": disk.free,
            "disk_total_human": _human_bytes(disk.total),
            "disk_free_human": _human_bytes(disk.free),
            "min_free_after_bytes": min_free_after,
            "min_free_after_human": _human_bytes(min_free_after),
            **_memory_info(),
            "gpus": _gpu_info(),
        },
        "summary": {
            "manifest_entries": len(entries),
            "local_raw_bytes": local_raw_bytes,
            "local_raw_human": _human_bytes(local_raw_bytes),
            "local_hf_full_bytes": local_hf_full_bytes,
            "local_hf_full_human": _human_bytes(local_hf_full_bytes),
            "local_hf_full_file_count": local_hf_full_file_count,
            "known_hf_split_bytes_total": known_full_hf_bytes,
            "known_hf_split_human": _human_bytes(known_full_hf_bytes),
            "known_hf_split_count": known_full_hf_count,
            "unknown_or_non_hf_entries": len(entries) - known_full_hf_count,
        },
        "entries": entries,
    }


def write_markdown(audit: dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    cap = audit["capacity"]
    gpus = cap.get("gpus") or []
    gpu_text = ", ".join(f"{gpu['name']} ({_human_bytes(gpu.get('memory_total_bytes'))} total)" for gpu in gpus) or "none detected"
    lines = [
        "# Dataset Capacity Audit",
        "",
        f"Created: `{audit['created_at']}`",
        "",
        "## Local Capacity",
        "",
        f"- Disk free: {_human_bytes(cap['disk_free_bytes'])} / {_human_bytes(cap['disk_total_bytes'])}.",
        f"- Reserved free-space floor: {_human_bytes(cap['min_free_after_bytes'])}.",
        f"- RAM available: {_human_bytes(cap.get('mem_available_bytes'))} / {_human_bytes(cap.get('mem_total_bytes'))}.",
        f"- Swap free: {_human_bytes(cap.get('swap_free_bytes'))} / {_human_bytes(cap.get('swap_total_bytes'))}.",
        f"- GPU: {gpu_text}.",
        "",
        "## Summary",
        "",
        f"- Manifest entries: {audit['summary']['manifest_entries']}.",
        f"- Current `data/raw` footprint covered by the manifest: {audit['summary']['local_raw_human']}.",
        f"- Current full selected HF parquet footprint: {audit['summary']['local_hf_full_human']} across {audit['summary']['local_hf_full_file_count']} files.",
        f"- Known HF selected-split size total: {audit['summary']['known_hf_split_human']} across {audit['summary']['known_hf_split_count']} entries.",
        "- Full-source downloads are not automatic; manifest-only and local-file entries remain provenance or user-provided paths.",
        "",
        "## Entry Status",
        "",
        "| Dataset | Method | Limit | Local | Remote selected split | Action |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for entry in audit["entries"]:
        remote = entry["remote"]
        remote_bytes = remote.get("split_num_bytes_parquet_files") if remote.get("available") else None
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{entry['name']}`",
                    str(entry["method"]),
                    str(entry["default_limit"]),
                    entry["local_human"],
                    _human_bytes(remote_bytes),
                    f"`{entry['recommended_action']}`",
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit dataset manifest sizes against local machine capacity.")
    parser.add_argument("--manifest", default="data/manifests/datasets.yaml")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--hf-full-dir", default="data/raw_hf_full")
    parser.add_argument("--output-json", default="data/manifests/dataset_capacity_audit.json")
    parser.add_argument("--output-md", default="planning/DATASET-CAPACITY-AUDIT.md")
    parser.add_argument("--min-free-after-gib", type=float, default=512.0)
    args = parser.parse_args()
    min_free_after = int(args.min_free_after_gib * 1024**3)
    audit = build_audit(Path(args.manifest), Path(args.raw_dir), Path(args.hf_full_dir), min_free_after)
    output_json = Path(args.output_json)
    ensure_dir(output_json.parent)
    output_json.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(audit, Path(args.output_md))
    print(json.dumps(audit["summary"], indent=2, sort_keys=True))
    print(f"wrote {output_json}")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()
