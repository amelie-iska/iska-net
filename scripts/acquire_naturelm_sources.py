#!/usr/bin/env python
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqdm.auto import tqdm

from iska_reasoner.data.splits import SplitReport, assign_split_for_policy
from iska_reasoner.data.graphify import graphify_rows
from iska_reasoner.graph.schema import GraphExample
from iska_reasoner.utils.io import ensure_dir, write_jsonl
from iska_reasoner.utils.logging import WandbLogger, setup_logging
from scripts.prepare_science_sources import iter_rows


LOGGER = logging.getLogger("naturelm_acquire")


@dataclass(frozen=True)
class SourceFile:
    url: str
    md5_url: str | None = None
    filename: str | None = None
    size_hint: str | None = None


@dataclass(frozen=True)
class NatureLMSource:
    name: str
    dataset_name: str
    kind: str
    raw_dir: str
    files: tuple[SourceFile, ...] = ()
    source_url: str = ""
    description: str = ""
    large_download: bool = False
    large_prepare: bool = False
    requires_env: str | None = None
    prepare: bool = True
    metadata_only: bool = False
    tags: tuple[str, ...] = field(default_factory=tuple)


PUBCHEM_EXTRAS = "https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras"
UNIPROT_COMPLETE = "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete"
REFSEQ_RELEASE = "https://ftp.ncbi.nlm.nih.gov/refseq/release"


SOURCES: dict[str, NatureLMSource] = {
    "pubchem_cid_smiles": NatureLMSource(
        name="pubchem_cid_smiles",
        dataset_name="naturelm_pubchem_local",
        kind="pubchem",
        raw_dir="naturelm_pubchem_local",
        source_url=f"{PUBCHEM_EXTRAS}/",
        description="Complete PubChem CID to SMILES export.",
        files=(
            SourceFile(
                url=f"{PUBCHEM_EXTRAS}/CID-SMILES.gz",
                md5_url=f"{PUBCHEM_EXTRAS}/CID-SMILES.gz.md5",
                size_hint="1.4 GB compressed on 2026-04-29",
            ),
        ),
        large_prepare=True,
        tags=("public_core", "pubchem"),
    ),
    "pubchem_drug_names": NatureLMSource(
        name="pubchem_drug_names",
        dataset_name="naturelm_pubchem_drug_names",
        kind="pubchem",
        raw_dir="naturelm_pubchem_local",
        source_url=f"{PUBCHEM_EXTRAS}/",
        description="PubChem drug-name table for compound-name reconstruction.",
        files=(
            SourceFile(
                url=f"{PUBCHEM_EXTRAS}/Drug-Names.tsv.gz",
                md5_url=f"{PUBCHEM_EXTRAS}/Drug-Names.tsv.gz.md5",
                size_hint="837 KB compressed on 2026-04-29",
            ),
        ),
        tags=("public_core", "pubchem"),
    ),
    "uniprot_sprot": NatureLMSource(
        name="uniprot_sprot",
        dataset_name="naturelm_uniprot_sprot",
        kind="uniprot",
        raw_dir="naturelm_uniprot_local",
        source_url=f"{UNIPROT_COMPLETE}/",
        description="Reviewed UniProtKB/Swiss-Prot FASTA.",
        files=(
            SourceFile(
                url=f"{UNIPROT_COMPLETE}/uniprot_sprot.fasta.gz",
                size_hint="89 MB compressed on 2026-04-29",
            ),
        ),
        tags=("public_core", "uniprot"),
    ),
    "uniprot_varsplic": NatureLMSource(
        name="uniprot_varsplic",
        dataset_name="naturelm_uniprot_varsplic",
        kind="uniprot",
        raw_dir="naturelm_uniprot_local",
        source_url=f"{UNIPROT_COMPLETE}/",
        description="UniProtKB/Swiss-Prot variant splice isoform FASTA.",
        files=(
            SourceFile(
                url=f"{UNIPROT_COMPLETE}/uniprot_sprot_varsplic.fasta.gz",
                size_hint="8.2 MB compressed on 2026-04-29",
            ),
        ),
        tags=("public_core", "uniprot"),
    ),
    "uniprot_trembl": NatureLMSource(
        name="uniprot_trembl",
        dataset_name="naturelm_uniprot_trembl",
        kind="uniprot",
        raw_dir="naturelm_uniprot_local",
        source_url=f"{UNIPROT_COMPLETE}/",
        description="Full unreviewed UniProtKB/TrEMBL FASTA.",
        files=(
            SourceFile(
                url=f"{UNIPROT_COMPLETE}/uniprot_trembl.fasta.gz",
                size_hint="49 GB compressed on 2026-04-29",
            ),
        ),
        large_download=True,
        large_prepare=True,
        tags=("large_full", "uniprot"),
    ),
    "refseq_release_metadata": NatureLMSource(
        name="refseq_release_metadata",
        dataset_name="naturelm_refseq_release_metadata",
        kind="refseq",
        raw_dir="naturelm_refseq_local",
        source_url=f"{REFSEQ_RELEASE}/",
        description="RefSeq release README, release number, and installed-file manifest.",
        files=(
            SourceFile(url=f"{REFSEQ_RELEASE}/README"),
            SourceFile(url=f"{REFSEQ_RELEASE}/RELEASE_NUMBER"),
            SourceFile(url=f"{REFSEQ_RELEASE}/release-catalog/release234.files.installed"),
        ),
        prepare=False,
        metadata_only=True,
        tags=("public_core", "refseq"),
    ),
    "refseq_viral_protein": NatureLMSource(
        name="refseq_viral_protein",
        dataset_name="naturelm_refseq_viral_protein",
        kind="refseq",
        raw_dir="naturelm_refseq_local",
        source_url=f"{REFSEQ_RELEASE}/viral/",
        description="RefSeq viral protein FASTA slice.",
        files=(
            SourceFile(
                url=f"{REFSEQ_RELEASE}/viral/viral.1.protein.faa.gz",
                size_hint="101 MB compressed on 2026-04-29",
            ),
        ),
        tags=("public_core", "refseq"),
    ),
    "refseq_mitochondrion_protein": NatureLMSource(
        name="refseq_mitochondrion_protein",
        dataset_name="naturelm_refseq_mitochondrion_protein",
        kind="refseq",
        raw_dir="naturelm_refseq_local",
        source_url=f"{REFSEQ_RELEASE}/mitochondrion/",
        description="RefSeq mitochondrion protein FASTA slice.",
        files=(
            SourceFile(
                url=f"{REFSEQ_RELEASE}/mitochondrion/mitochondrion.1.protein.faa.gz",
                size_hint="33 MB compressed on 2026-04-29",
            ),
        ),
        tags=("public_core", "refseq"),
    ),
    "materials_project": NatureLMSource(
        name="materials_project",
        dataset_name="naturelm_materials_project_local",
        kind="materials_project",
        raw_dir="naturelm_materials_project_local",
        source_url="https://docs.materialsproject.org/downloading-data/using-the-api",
        description="Materials Project API export. Requires MP_API_KEY and optional mp-api installation.",
        requires_env="MP_API_KEY",
        prepare=False,
        metadata_only=True,
        tags=("public_core", "materials_project", "api_gated"),
    ),
}


def _utc_now() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _filename_from_url(url: str) -> str:
    return Path(urllib.parse.urlparse(url).path).name


def _content_length(url: str) -> int | None:
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            value = response.headers.get("content-length")
    except Exception:
        value = None
    try:
        return int(value) if value else None
    except ValueError:
        return None


def _download_text(url: str, timeout: int = 120) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def _parse_md5(text: str) -> str | None:
    for token in text.replace("*", " ").split():
        if len(token) == 32 and all(ch in "0123456789abcdefABCDEF" for ch in token):
            return token.lower()
    return None


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _download_file(file: SourceFile, target_dir: Path, overwrite: bool = False) -> dict[str, Any]:
    filename = file.filename or _filename_from_url(file.url)
    target = target_dir / filename
    total = _content_length(file.url)
    if target.exists() and target.stat().st_size > 0 and not overwrite:
        return {
            "url": file.url,
            "path": str(target),
            "bytes": target.stat().st_size,
            "skipped_existing": True,
            "md5_ok": None,
        }

    ensure_dir(target_dir)
    tmp = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(file.url, headers={"User-Agent": "iska-reasoner-dataset-acquirer/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response, tmp.open("wb") as handle:
        if total is None:
            try:
                total = int(response.headers.get("content-length") or 0) or None
            except ValueError:
                total = None
        with tqdm(total=total, desc=f"download/{filename}", unit="B", unit_scale=True) as pbar:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                pbar.update(len(chunk))
    tmp.replace(target)

    md5_expected = None
    md5_actual = None
    md5_ok = None
    if file.md5_url:
        md5_text = _download_text(file.md5_url)
        (target_dir / f"{filename}.md5").write_text(md5_text, encoding="utf-8")
        md5_expected = _parse_md5(md5_text)
        if md5_expected:
            md5_actual = _md5(target)
            md5_ok = md5_actual == md5_expected
            if not md5_ok:
                raise ValueError(f"MD5 mismatch for {target}: expected {md5_expected}, got {md5_actual}")

    return {
        "url": file.url,
        "path": str(target),
        "bytes": target.stat().st_size,
        "skipped_existing": False,
        "md5_expected": md5_expected,
        "md5_actual": md5_actual,
        "md5_ok": md5_ok,
    }


def _split_name(example_id: str, val_ratio: float, test_ratio: float) -> str:
    bucket = int(hashlib.sha1(example_id.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    if bucket < test_ratio:
        return "test"
    if bucket < test_ratio + val_ratio:
        return "val"
    return "train"


def _write_api_marker(source: NatureLMSource, raw_dir: Path) -> dict[str, Any]:
    payload = {
        "source": source.name,
        "dataset_name": source.dataset_name,
        "status": "auth_required",
        "requires_env": source.requires_env,
        "env_present": bool(os.environ.get(source.requires_env or "")),
        "source_url": source.source_url,
        "description": source.description,
        "created_at": _utc_now(),
    }
    ensure_dir(raw_dir)
    marker = raw_dir / "AUTH_REQUIRED.json"
    marker.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"marker": str(marker), **payload}


def acquire_sources(
    sources: Iterable[NatureLMSource],
    raw_base: Path,
    include_large: bool = False,
    overwrite: bool = False,
    logger: logging.Logger | None = None,
    wandb_logger: WandbLogger | None = None,
) -> dict[str, Any]:
    acquired: dict[str, Any] = {}
    logger = logger or LOGGER
    for source in sources:
        logger.info("Acquiring source %s (%s)", source.name, source.source_url)
        raw_dir = ensure_dir(raw_base / source.raw_dir)
        source_record: dict[str, Any] = {
            "name": source.name,
            "dataset_name": source.dataset_name,
            "kind": source.kind,
            "source_url": source.source_url,
            "description": source.description,
            "started_at": _utc_now(),
            "files": [],
            "skipped": False,
        }
        if source.requires_env:
            source_record["auth"] = _write_api_marker(source, raw_dir)
            source_record["skipped"] = True
            source_record["skip_reason"] = (
                f"missing {source.requires_env}"
                if not os.environ.get(source.requires_env)
                else "api export is credentialed; use Materials Project API export before local graphification"
            )
        elif source.large_download and not include_large:
            source_record["skipped"] = True
            source_record["skip_reason"] = "large download; rerun with --include-large"
        else:
            for file in source.files:
                source_record["files"].append(_download_file(file, raw_dir, overwrite=overwrite))
        source_record["finished_at"] = _utc_now()
        write_jsonl(raw_dir / f"PROVENANCE.{source.name}.jsonl", [source_record])
        _append_jsonl(raw_dir / "PROVENANCE.jsonl", source_record)
        acquired[source.name] = source_record
        if wandb_logger is not None:
            bytes_total = sum(int(file_info.get("bytes") or 0) for file_info in source_record.get("files", []))
            wandb_logger.log(
                {
                    f"naturelm/acquire/{source.name}/bytes": float(bytes_total),
                    f"naturelm/acquire/{source.name}/files": float(len(source_record.get("files", []))),
                    f"naturelm/acquire/{source.name}/skipped": float(bool(source_record.get("skipped"))),
                },
                step=len(acquired),
            )
        if source_record.get("skipped"):
            logger.info("Skipped source %s: %s", source.name, source_record.get("skip_reason", "metadata-only"))
        else:
            logger.info("Acquired source %s with %d file(s)", source.name, len(source_record["files"]))
    return acquired


def prepare_sources_to_splits(
    sources: Iterable[NatureLMSource],
    acquired: dict[str, Any],
    output_dir: Path,
    raw_base: Path,
    val_ratio: float,
    test_ratio: float,
    graph_limit_per_source: int | None = None,
    prepare_large: bool = False,
    split_policy: str = "entity",
    logger: logging.Logger | None = None,
    wandb_logger: WandbLogger | None = None,
) -> dict[str, Any]:
    out_dir = ensure_dir(output_dir)
    logger = logger or LOGGER
    progress_path = out_dir / "progress.jsonl"
    handles = {
        "train": (out_dir / "train.jsonl").open("w", encoding="utf-8"),
        "val": (out_dir / "val.jsonl").open("w", encoding="utf-8"),
        "test": (out_dir / "test.jsonl").open("w", encoding="utf-8"),
    }
    counts = {"train": 0, "val": 0, "test": 0}
    per_source: dict[str, Any] = {}
    split_report = SplitReport(policy=split_policy)
    total = 0
    try:
        for source in sources:
            record = acquired.get(source.name, {})
            logger.info("Preparing source %s", source.name)
            if record.get("skipped") or not source.prepare or source.metadata_only:
                per_source[source.name] = {"prepared": 0, "skipped": True, "reason": record.get("skip_reason") or "metadata-only"}
                _append_jsonl(progress_path, {"time": _utc_now(), "source": source.name, **per_source[source.name]})
                if wandb_logger is not None:
                    wandb_logger.log({f"naturelm/prepare/{source.name}/skipped": 1.0}, step=total)
                continue
            if source.large_prepare and not prepare_large:
                per_source[source.name] = {"prepared": 0, "skipped": True, "reason": "large preparation; rerun with --prepare-large"}
                _append_jsonl(progress_path, {"time": _utc_now(), "source": source.name, **per_source[source.name]})
                if wandb_logger is not None:
                    wandb_logger.log({f"naturelm/prepare/{source.name}/skipped": 1.0}, step=total)
                continue
            paths = [Path(file_info["path"]) for file_info in record.get("files", []) if file_info.get("path")]
            if not paths:
                per_source[source.name] = {"prepared": 0, "skipped": True, "reason": "no raw files"}
                _append_jsonl(progress_path, {"time": _utc_now(), "source": source.name, **per_source[source.name]})
                if wandb_logger is not None:
                    wandb_logger.log({f"naturelm/prepare/{source.name}/skipped": 1.0}, step=total)
                continue
            source_count = 0
            rows = iter_rows(paths, source.kind, graph_limit_per_source)
            graphs = graphify_rows(rows, source.dataset_name)
            with tqdm(total=graph_limit_per_source, desc=f"prepare/{source.name}", unit="ex") as pbar:
                for graph_row in graphs:
                    graph = GraphExample.from_dict(graph_row)
                    split, split_group_key = assign_split_for_policy(graph, split_policy, val_ratio, test_ratio)
                    graph.metadata.setdefault("curation", {})
                    graph.metadata["curation"]["split_policy"] = split_policy
                    graph.metadata["curation"]["split_group_key"] = split_group_key
                    handles[split].write(json.dumps(graph.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
                    counts[split] += 1
                    split_report.add(split, split_group_key)
                    source_count += 1
                    total += 1
                    pbar.update(1)
                    if source_count % 10000 == 0:
                        pbar.set_postfix({"source": source_count, "total": total, **counts}, refresh=False)
            per_source[source.name] = {"prepared": source_count, "skipped": False, "paths": [str(path) for path in paths]}
            _append_jsonl(progress_path, {"time": _utc_now(), "source": source.name, **per_source[source.name], "counts": dict(counts)})
            if wandb_logger is not None:
                wandb_logger.log(
                    {
                        f"naturelm/prepare/{source.name}/examples": float(source_count),
                        "naturelm/prepare/examples_total": float(total),
                        "naturelm/prepare/train_examples": float(counts["train"]),
                        "naturelm/prepare/val_examples": float(counts["val"]),
                        "naturelm/prepare/test_examples": float(counts["test"]),
                    },
                    step=total,
                )
            logger.info("Prepared %s examples from %s", source_count, source.name)
    finally:
        for handle in handles.values():
            handle.close()
    summary = {
        "created_at": _utc_now(),
        "output_dir": str(out_dir),
        "raw_base": str(raw_base),
        "counts": counts,
        "total": total,
        "split_sizes": counts,
        "per_source": per_source,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "graph_limit_per_source": graph_limit_per_source,
        "prepare_large": prepare_large,
        "split_report": split_report.to_dict(),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info("Prepared NatureLM public-source splits: %s", counts)
    return summary


def _select_sources(names: list[str], tags: list[str]) -> list[NatureLMSource]:
    if not names and not tags:
        tags = ["public_core"]
    selected: list[NatureLMSource] = []
    for source in SOURCES.values():
        if names and source.name not in names:
            continue
        if tags and not set(tags).intersection(source.tags):
            continue
        selected.append(source)
    unknown = sorted(set(names) - set(SOURCES))
    if unknown:
        raise SystemExit(f"Unknown source(s): {unknown}. Available: {sorted(SOURCES)}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Acquire and prepare official NatureLM-style public science sources.")
    parser.add_argument("--source", action="append", default=[], help=f"Source name. Available: {', '.join(sorted(SOURCES))}")
    parser.add_argument("--tag", action="append", default=[], help="Source tag selector. Defaults to public_core when no source is specified.")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--output-dir", default="data/processed/naturelm_public_sources")
    parser.add_argument("--include-large", action="store_true", help="Allow very large downloads such as UniProt TrEMBL.")
    parser.add_argument("--prepare", action="store_true", help="Graphify acquired sources into deterministic train/val/test splits.")
    parser.add_argument("--prepare-large", action="store_true", help="Allow high-cardinality preparation such as full PubChem CID-SMILES.")
    parser.add_argument("--graph-limit-per-source", type=int, help="Optional source-level cap for smoke tests. Omit for full selected preparation.")
    parser.add_argument("--val-ratio", type=float, default=0.01)
    parser.add_argument("--test-ratio", type=float, default=0.01)
    parser.add_argument("--split-policy", choices=["row_hash", "entity"], default="entity")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-dir", default="logs/naturelm_acquisition", help="Directory for run.log and run_summary.json.")
    parser.add_argument("--wandb", action="store_true", help="Log acquisition summary metrics to Weights & Biases.")
    parser.add_argument("--wandb-project", default="iska-ugm")
    parser.add_argument("--wandb-run-name", default="naturelm-public-sources")
    parser.add_argument("--wandb-mode", default=os.environ.get("WANDB_MODE", "offline"))
    args = parser.parse_args()

    logger = setup_logging(args.log_dir, name="naturelm_acquire")

    selected = _select_sources(args.source, args.tag)
    if args.dry_run:
        print(json.dumps({"selected": [source.name for source in selected], "sources": [asdict(source) for source in selected]}, indent=2, sort_keys=True))
        return

    raw_base = Path(args.raw_dir)
    wandb = WandbLogger(
        {
            "enabled": args.wandb,
            "project": args.wandb_project,
            "run_name": args.wandb_run_name,
            "mode": args.wandb_mode,
            "job_type": "dataset_acquisition",
            "tags": ["naturelm", "sfm", "unigenx", "dataset"],
            "config": vars(args),
        }
    )
    acquired = acquire_sources(
        selected,
        raw_base,
        include_large=args.include_large,
        overwrite=args.overwrite,
        logger=logger,
        wandb_logger=wandb,
    )
    result: dict[str, Any] = {"acquired": acquired}
    if args.prepare:
        result["prepared"] = prepare_sources_to_splits(
            selected,
            acquired,
            output_dir=Path(args.output_dir),
            raw_base=raw_base,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            graph_limit_per_source=args.graph_limit_per_source,
            prepare_large=args.prepare_large,
            split_policy=args.split_policy,
            logger=logger,
            wandb_logger=wandb,
        )
    if args.log_dir:
        ensure_dir(args.log_dir)
        (Path(args.log_dir) / "run_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.wandb and "prepared" in result:
        prepared = result["prepared"]
        wandb.log(
            {
                "naturelm/examples_total": float(prepared.get("total", 0)),
                "naturelm/train_examples": float(prepared.get("counts", {}).get("train", 0)),
                "naturelm/val_examples": float(prepared.get("counts", {}).get("val", 0)),
                "naturelm/test_examples": float(prepared.get("counts", {}).get("test", 0)),
            },
            step=0,
        )
    wandb.finish()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
