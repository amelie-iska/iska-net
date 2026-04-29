from __future__ import annotations

import json
from pathlib import Path

from iska_reasoner.data.catalog import build_catalog_status, human_bytes, source_link, write_catalog_markdown


def test_human_bytes_formats_values():
    assert human_bytes(None) == "unknown"
    assert human_bytes(0) == "0.0 B"
    assert human_bytes(2048) == "2.0 KB"


def test_source_link_prefers_hf_and_repo_urls():
    assert source_link({"name": "x", "dataset_id": "org/name"}) == "https://huggingface.co/datasets/org/name"
    assert source_link({"name": "x", "repo_url": "https://github.com/a/b.git"}) == "https://github.com/a/b"
    assert source_link({"name": "chembl_local_export"}) == "https://www.ebi.ac.uk/chembl/"


def test_catalog_status_distinguishes_ready_public_and_deferred_local(tmp_path: Path):
    manifest = tmp_path / "data/manifests/datasets.yaml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        """
datasets:
  - name: public_ready
    dataset_id: org/public
    config: default
    split: train
    method: hf_rows
    stage: unit
  - name: local_missing
    method: local_file
    manifest_only: true
    split: train
    stage: unit_local
""".strip()
        + "\n",
        encoding="utf-8",
    )
    full = tmp_path / "data/processed/real_full_selected_mix"
    full.mkdir(parents=True)
    (full / "summary.json").write_text(
        json.dumps({"counts": {"train": 1, "val": 0, "test": 0}, "total": 1, "per_dataset": {"public_ready": 1}}),
        encoding="utf-8",
    )
    (full / "integrity.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
    (full / "token_counts.json").write_text(
        json.dumps({"source_graph_tokens": 4, "target_tokens": 1, "model_sequence_tokens_untruncated": 7}),
        encoding="utf-8",
    )
    refs = tmp_path / "data/processed/reference_tokens"
    refs.mkdir(parents=True)
    (refs / "naturelm_unigenx_tokens.txt").write_text("<protein>\n", encoding="utf-8")
    (refs / "motif_graph_tokens.txt").write_text("TOK\n" * 100000, encoding="utf-8")
    (refs / "multimodal_graph_tokens.txt").write_text("TOK\n" * 100000, encoding="utf-8")
    (refs / "motif_graph_tokens.summary.json").write_text(json.dumps({"records": 70000, "tokens": 100000}), encoding="utf-8")

    status = build_catalog_status(
        root=tmp_path,
        manifest_path=manifest.relative_to(tmp_path),
        audit_path="missing.json",
        show_progress=False,
    )

    assert status["ready"] is True
    entries = {entry["name"]: entry for entry in status["manifest_entries"]}
    assert entries["public_ready"]["status"] == "included_full_public_corpus"
    assert entries["local_missing"]["status"] == "deferred_local_user_export_required"
    assert len(status["deferred_entries"]) == 1

    output = tmp_path / "planning/status.md"
    write_catalog_markdown(status, output)
    text = output.read_text(encoding="utf-8")
    assert "public_ready" in text
    assert "local_missing" in text
