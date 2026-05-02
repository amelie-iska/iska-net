from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from iska_reasoner.utils.config import load_yaml
from iska_reasoner.utils.io import ensure_dir


LOCAL_SOURCE_LINKS = {
    "chembl_local_export": "https://www.ebi.ac.uk/chembl/",
    "bindingdb_local_export": "https://www.bindingdb.org/",
    "naturelm_pubchem_local": "https://pubchem.ncbi.nlm.nih.gov/",
    "naturelm_uniprot_local": "https://www.uniprot.org/",
    "uniprot_features_local_export": "https://www.uniprot.org/help/downloads",
    "naturelm_refseq_local": "https://www.ncbi.nlm.nih.gov/refseq/",
    "naturelm_materials_project_local": "https://materialsproject.org/",
    "pdbbind_docking_local": "http://www.pdbbind.org.cn/",
    "biomolecular_complex_affinity_local": "https://www.ebi.ac.uk/intact/",
    "ec_protein_generation_local": "https://enzyme.expasy.org/",
}

REFERENCE_TOKEN_PATHS = {
    "naturelm_unigenx": "data/processed/reference_tokens/naturelm_unigenx_tokens.txt",
    "motif": "data/processed/reference_tokens/motif_graph_tokens.txt",
    "motif_summary": "data/processed/reference_tokens/motif_graph_tokens.summary.json",
    "multimodal": "data/processed/reference_tokens/multimodal_graph_tokens.txt",
}

DEFAULT_PROCESSED_CORPORA = {
    "curated_graphs": "data/processed/curated_graphs",
    "hebrew_mix": "data/processed/hebrew_mix",
    "science_mix": "data/processed/science_mix",
    "multimodal_graphs": "data/processed/multimodal_graphs",
    "real_4090_mix": "data/processed/real_4090_mix",
    "real_full_selected_mix": "data/processed/real_full_selected_mix",
}


def human_bytes(value: int | None) -> str:
    if value is None:
        return "unknown"
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    size = float(value)
    for unit in units:
        if abs(size) < 1024.0 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{value} B"


def path_size(path: Path) -> int:
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


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.stat().st_size:
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def count_text_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return sum(1 for line in handle if line.strip())


def source_link(item: dict[str, Any]) -> str:
    if item.get("dataset_id"):
        return f"https://huggingface.co/datasets/{item['dataset_id']}"
    if item.get("repo_url"):
        return str(item["repo_url"]).removesuffix(".git")
    return LOCAL_SOURCE_LINKS.get(str(item.get("name")), "local/user-provided")


def selected_parquet_paths(raw_full_dir: Path, item: dict[str, Any]) -> list[Path]:
    name = str(item["name"])
    config = str(item.get("config", "default"))
    split = str(item.get("split", "train"))
    return sorted((raw_full_dir / name / config / split).glob("*.parquet"))


def load_full_corpus_state(full_dir: Path) -> dict[str, Any]:
    summary = read_json(full_dir / "summary.json")
    integrity = read_json(full_dir / "integrity.json")
    token_counts = read_json(full_dir / "token_counts.json")
    counts = summary.get("counts") if isinstance(summary.get("counts"), dict) else {}
    per_dataset = summary.get("per_dataset") if isinstance(summary.get("per_dataset"), dict) else {}
    return {
        "path": str(full_dir),
        "summary_exists": bool(summary),
        "integrity_exists": bool(integrity),
        "token_counts_exists": bool(token_counts),
        "integrity_ok": bool(integrity.get("ok")),
        "counts": counts,
        "total": int(summary.get("total") or sum(int(v or 0) for v in counts.values())),
        "per_dataset": {str(k): int(v or 0) for k, v in per_dataset.items()},
        "token_counts": token_counts,
    }


def inspect_reference_sources(root: Path) -> dict[str, Any]:
    refs: dict[str, Any] = {}
    for key, rel in REFERENCE_TOKEN_PATHS.items():
        path = root / rel
        if key.endswith("summary"):
            payload = read_json(path)
            refs[key] = {"path": rel, "exists": path.exists(), **payload}
        else:
            refs[key] = {"path": rel, "exists": path.exists(), "tokens": count_text_lines(path)}
    motif_summary = refs.get("motif_summary", {})
    refs["ready"] = (
        refs.get("naturelm_unigenx", {}).get("tokens", 0) >= 1
        and refs.get("motif", {}).get("tokens", 0) >= 100000
        and refs.get("multimodal", {}).get("tokens", 0) >= 100000
        and motif_summary.get("records", 0) >= 70000
        and motif_summary.get("tokens", 0) >= 100000
    )
    return refs


def inspect_processed_corpus(path: Path) -> dict[str, Any]:
    summary = read_json(path / "summary.json")
    integrity = read_json(path / "integrity.json")
    token_counts = read_json(path / "token_counts.json")
    split_sizes = summary.get("split_sizes") if isinstance(summary.get("split_sizes"), dict) else summary.get("counts")
    if not isinstance(split_sizes, dict):
        split_sizes = {}
    return {
        "path": str(path),
        "exists": path.exists(),
        "summary_exists": bool(summary),
        "integrity_exists": bool(integrity),
        "integrity_ok": integrity.get("ok") if integrity else None,
        "token_counts_exists": bool(token_counts),
        "kept_examples": int(summary.get("kept_rows") or summary.get("total") or sum(int(v or 0) for v in split_sizes.values())),
        "split_sizes": {str(k): int(v or 0) for k, v in split_sizes.items()},
    }


def _remote_bytes_from_audit(audit: dict[str, Any], name: str) -> int | None:
    for entry in audit.get("entries", []) or []:
        if entry.get("name") == name:
            remote = entry.get("remote") or {}
            value = remote.get("split_num_bytes_parquet_files")
            return int(value) if isinstance(value, int) else None
    return None


def classify_manifest_entry(
    item: dict[str, Any],
    root: Path,
    raw_dir: Path,
    raw_full_dir: Path,
    full_state: dict[str, Any],
    audit: dict[str, Any],
) -> dict[str, Any]:
    name = str(item["name"])
    method = str(item.get("method", "hf_rows"))
    manifest_only = bool(item.get("manifest_only", False))
    raw_path = raw_dir / name
    parquet_paths = selected_parquet_paths(raw_full_dir, item) if method == "hf_rows" else []
    parquet_bytes = sum(path.stat().st_size for path in parquet_paths if path.exists())
    full_examples = int(full_state.get("per_dataset", {}).get(name, 0))
    remote_bytes = _remote_bytes_from_audit(audit, name)
    status = "unknown"
    required_for_public_full = False
    error = False
    deferred = False

    if method == "hf_rows":
        if manifest_only:
            status = "deferred_manifest_only_or_restricted"
            deferred = True
        elif full_examples > 0 and full_state.get("integrity_ok"):
            status = "included_full_public_corpus"
            required_for_public_full = True
        elif parquet_paths:
            status = "raw_parquet_available_not_graphified"
            required_for_public_full = True
            error = True
        else:
            status = "missing_public_hf_parquet"
            required_for_public_full = True
            error = True
    elif method == "git_clone":
        repo_exists = (raw_path / "repo").exists()
        status = "git_source_available" if repo_exists else "git_source_missing"
        deferred = not repo_exists
    elif method == "local_generated":
        generated_exists = raw_path.exists() or (root / "data/processed/multimodal_graphs/summary.json").exists()
        status = "generated_source_available" if generated_exists else "generated_source_missing"
        error = not generated_exists
    elif method == "local_file":
        usable_files = [path for path in raw_path.glob("**/*") if path.is_file() and path.name != "PROVENANCE.jsonl"] if raw_path.exists() else []
        status = "local_user_export_available" if usable_files else "deferred_local_user_export_required"
        deferred = not usable_files
    else:
        status = "unsupported_manifest_method"
        error = True

    return {
        "name": name,
        "method": method,
        "stage": item.get("stage"),
        "license": item.get("license"),
        "manifest_only": manifest_only,
        "link": source_link(item),
        "upstream_split": "/".join(str(part) for part in [item.get("config"), item.get("split")] if part),
        "remote_selected_split_bytes": remote_bytes,
        "remote_selected_split_human": human_bytes(remote_bytes),
        "raw_path": str(raw_path),
        "raw_bytes": path_size(raw_path),
        "raw_human": human_bytes(path_size(raw_path)),
        "parquet_file_count": len(parquet_paths),
        "parquet_bytes": parquet_bytes,
        "parquet_human": human_bytes(parquet_bytes),
        "full_graph_examples": full_examples,
        "required_for_public_full": required_for_public_full,
        "status": status,
        "error": error,
        "deferred": deferred,
        "notes": item.get("notes", ""),
    }


def build_catalog_status(
    root: str | Path = ".",
    manifest_path: str | Path = "data/manifests/datasets.yaml",
    raw_dir: str | Path = "data/raw",
    raw_full_dir: str | Path = "data/raw_hf_full",
    full_dir: str | Path = "data/processed/real_full_selected_mix",
    audit_path: str | Path = "data/manifests/dataset_capacity_audit.json",
    processed_corpora: dict[str, str] | None = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    root_path = Path(root)
    manifest = load_yaml(root_path / manifest_path)
    raw = root_path / raw_dir
    raw_full = root_path / raw_full_dir
    full_state = load_full_corpus_state(root_path / full_dir)
    audit = read_json(root_path / audit_path)
    entries = []
    dataset_items = manifest.get("datasets", [])
    iterator = tqdm(dataset_items, desc="catalog/manifest", unit="dataset", disable=not show_progress)
    for item in iterator:
        entries.append(classify_manifest_entry(item, root_path, raw, raw_full, full_state, audit))
    refs = inspect_reference_sources(root_path)
    corpora = {
        name: inspect_processed_corpus(root_path / rel)
        for name, rel in (processed_corpora or DEFAULT_PROCESSED_CORPORA).items()
    }
    public_errors = [entry for entry in entries if entry["required_for_public_full"] and entry["error"]]
    deferred = [entry for entry in entries if entry["deferred"]]
    errors = []
    if not full_state.get("integrity_ok"):
        errors.append("full_selected_public_corpus_integrity_not_ok")
    if not full_state.get("token_counts_exists"):
        errors.append("full_selected_public_corpus_token_counts_missing")
    if not refs.get("ready"):
        errors.append("reference_vocabularies_not_ready")
    errors.extend(f"public_dataset_not_ready:{entry['name']}:{entry['status']}" for entry in public_errors)
    ready = not errors
    return {
        "ready": ready,
        "errors": errors,
        "full_corpus": full_state,
        "references": refs,
        "manifest_entries": entries,
        "deferred_entries": deferred,
        "processed_corpora": corpora,
        "summary": {
            "manifest_entries": len(entries),
            "public_full_entries": sum(1 for entry in entries if entry["required_for_public_full"]),
            "included_full_public_entries": sum(1 for entry in entries if entry["status"] == "included_full_public_corpus"),
            "deferred_entries": len(deferred),
            "errors": len(errors),
            "full_examples": full_state.get("total", 0),
            "full_train_examples": full_state.get("counts", {}).get("train", 0),
            "full_val_examples": full_state.get("counts", {}).get("val", 0),
            "full_test_examples": full_state.get("counts", {}).get("test", 0),
        },
    }


def write_catalog_markdown(status: dict[str, Any], output: str | Path) -> None:
    output_path = Path(output)
    ensure_dir(output_path.parent)
    full = status["full_corpus"]
    refs = status["references"]
    token_counts = full.get("token_counts", {})
    lines = [
        "# Dataset Catalog Implementation Status",
        "",
        "Generated from `scripts/validate_dataset_catalog.py`.",
        "",
        "## Readiness",
        "",
        f"- Ready: `{str(status['ready']).lower()}`",
        f"- Errors: {len(status['errors'])}",
        f"- Deferred local/restricted entries: {len(status['deferred_entries'])}",
    ]
    if status["errors"]:
        lines.extend(["", "### Errors", ""])
        lines.extend(f"- `{error}`" for error in status["errors"])
    lines.extend(
        [
            "",
            "## Full Selected Public Corpus",
            "",
            f"- Path: `{full['path']}`",
            f"- Integrity OK: `{str(full['integrity_ok']).lower()}`",
            f"- Summary exists: `{str(full['summary_exists']).lower()}`",
            f"- Token counts exist: `{str(full['token_counts_exists']).lower()}`",
            f"- Examples: {full.get('total', 0):,}",
            f"- Train/validation/test: {int(full.get('counts', {}).get('train', 0)):,} / "
            f"{int(full.get('counts', {}).get('val', 0)):,} / {int(full.get('counts', {}).get('test', 0)):,}",
            f"- Source graph tokens: {int(token_counts.get('source_graph_tokens', 0)):,}",
            f"- Target graph tokens: {int(token_counts.get('target_tokens', 0)):,}",
            f"- Untruncated model-sequence graph tokens: {int(token_counts.get('model_sequence_tokens_untruncated', 0)):,}",
            "",
            "## Reference Vocabularies",
            "",
            "| Source | Exists | Size |",
            "|---|---:|---:|",
            f"| NatureLM + UniGenX tokens | `{str(refs['naturelm_unigenx']['exists']).lower()}` | {refs['naturelm_unigenx'].get('tokens', 0):,} tokens |",
            f"| Motif tokens | `{str(refs['motif']['exists']).lower()}` | {refs['motif'].get('tokens', 0):,} tokens |",
            f"| Motif records | `{str(refs['motif_summary']['exists']).lower()}` | {refs['motif_summary'].get('records', 0):,} records |",
            f"| Multimodal tokens | `{str(refs['multimodal']['exists']).lower()}` | {refs['multimodal'].get('tokens', 0):,} tokens |",
            "",
            "## Manifest Entry Status",
            "",
            "| Dataset | Method | Status | Link | Full graph examples | Raw/parquet size | Split |",
            "|---|---|---|---|---:|---:|---|",
        ]
    )
    for entry in status["manifest_entries"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{entry['name']}`",
                    str(entry["method"]),
                    f"`{entry['status']}`",
                    f"<{entry['link']}>" if str(entry["link"]).startswith("http") else str(entry["link"]),
                    f"{entry['full_graph_examples']:,}",
                    entry["parquet_human"] if entry["parquet_file_count"] else entry["raw_human"],
                    str(entry.get("upstream_split") or ""),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Processed Corpora", "", "| Corpus | Exists | Examples | Split sizes | Integrity |", "|---|---:|---:|---|---:|"])
    for name, corpus in status["processed_corpora"].items():
        split_text = ", ".join(f"{split}={count:,}" for split, count in sorted(corpus["split_sizes"].items()))
        integrity = corpus["integrity_ok"]
        integrity_text = "n/a" if integrity is None else str(integrity).lower()
        lines.append(f"| `{name}` | `{str(corpus['exists']).lower()}` | {corpus['kept_examples']:,} | {split_text} | `{integrity_text}` |")
    lines.extend(
        [
            "",
            "## Deferred Entries",
            "",
            "Deferred entries are expected blockers, not silent failures. They require credentials, upstream review, local user-provided exports, or a deliberate acquisition path before they can be counted as complete training data.",
            "",
        ]
    )
    if status["deferred_entries"]:
        for entry in status["deferred_entries"]:
            lines.append(f"- `{entry['name']}`: `{entry['status']}`; link: {entry['link']}")
    else:
        lines.append("- None.")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
