import json
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.count_graph_tokens import count_paths
from scripts.graphify_full_parquet_manifest import graphify_full_parquet_manifest


def test_count_graph_tokens_with_split_totals(tmp_path: Path):
    path = tmp_path / "train.jsonl"
    row = {
        "id": "ex0",
        "task": "unit",
        "nodes": [
            {"id": "a", "type": "prompt", "value": "hello", "features": {}},
            {"id": "b", "type": "answer", "value": "world", "features": {}},
        ],
        "edges": [{"src": "a", "dst": "b", "type": "supports", "features": {}}],
        "target_tokens": ["ANSWER:world", "CLAIM:unit"],
        "decoder_orders": [[0, 1]],
        "metadata": {"source_dataset": "unit_dataset"},
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    summary = count_paths([path], split_totals={"train": 1}, progress_every=1)

    assert summary["examples"] == 1
    assert summary["source_graph_tokens"] == 4
    assert summary["target_tokens"] == 2
    assert summary["model_sequence_tokens_untruncated"] == 9
    assert summary["by_dataset"]["unit_dataset"]["examples"] == 1


def test_graphify_full_parquet_manifest_uses_real_totals(tmp_path: Path):
    manifest = tmp_path / "datasets.yaml"
    raw = tmp_path / "raw_hf_full"
    out = tmp_path / "processed"
    dataset_dir = raw / "gsm8k_main_train" / "main" / "train"
    dataset_dir.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "question": ["one plus one?", "two plus two?"],
                "answer": ["#### 2", "#### 4"],
            }
        ),
        dataset_dir / "0000.parquet",
    )
    manifest.write_text(
        """
datasets:
  - name: gsm8k_main_train
    dataset_id: openai/gsm8k
    config: main
    split: train
    method: hf_rows
    default_limit: 0
    stage: math_reasoning
""".strip()
        + "\n",
        encoding="utf-8",
    )

    summary = graphify_full_parquet_manifest(
        manifest_path=manifest,
        raw_full_dir=raw,
        output_dir=out,
        max_rows_per_dataset=None,
        row_budget=None,
        val_ratio=0.0,
        test_ratio=0.0,
        batch_size=1,
        progress_every=1,
        nested_progress=False,
    )

    assert summary["total"] == 2
    assert summary["counts"]["train"] == 2
    assert summary["per_dataset"]["gsm8k_main_train"] == 2
    assert summary["per_dataset_source_rows"]["gsm8k_main_train"] == 2
    assert (out / "summary.json").exists()


def test_graphify_full_parquet_manifest_honors_manifest_row_cap(tmp_path: Path):
    manifest = tmp_path / "datasets.yaml"
    raw = tmp_path / "raw_hf_full"
    out = tmp_path / "processed"
    dataset_dir = raw / "rfam_sequence_train" / "default" / "train"
    dataset_dir.mkdir(parents=True)
    pq.write_table(
        pa.table({"sequence": ["AUGC", "GGUU"], "family": ["RF00001", "RF00002"]}),
        dataset_dir / "0000.parquet",
    )
    manifest.write_text(
        """
datasets:
  - name: rfam_sequence_train
    dataset_id: multimolecule/rfam
    config: default
    split: train
    method: hf_rows
    default_limit: 0
    stage: rna_sequence_pretraining
    full_training_max_rows: 1
""".strip()
        + "\n",
        encoding="utf-8",
    )

    summary = graphify_full_parquet_manifest(
        manifest_path=manifest,
        raw_full_dir=raw,
        output_dir=out,
        max_rows_per_dataset=None,
        row_budget=None,
        val_ratio=0.0,
        test_ratio=0.0,
        batch_size=2,
        progress_every=1,
        nested_progress=False,
    )

    assert summary["total"] == 1
    assert summary["per_dataset"]["rfam_sequence_train"] == 1
    assert summary["per_dataset_limits"]["rfam_sequence_train"] == 1


def test_graphify_full_parquet_manifest_compact_bio_scale_rows(tmp_path: Path):
    manifest = tmp_path / "datasets.yaml"
    raw = tmp_path / "raw_hf_full"
    out = tmp_path / "processed"
    dataset_dir = raw / "rnacentral_8192_sequence_train" / "default" / "train"
    dataset_dir.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "sequence": ["AUGCAUGC" * 200],
                "accession": ["URS0000000001"],
                "description": ["RNAcentral compact-row smoke entry"],
            }
        ),
        dataset_dir / "0000.parquet",
    )
    manifest.write_text(
        """
datasets:
  - name: rnacentral_8192_sequence_train
    dataset_id: rnacentral/example
    config: default
    split: train
    method: hf_rows
    default_limit: 0
    stage: rna_sequence_pretraining
""".strip()
        + "\n",
        encoding="utf-8",
    )

    summary = graphify_full_parquet_manifest(
        manifest_path=manifest,
        raw_full_dir=raw,
        output_dir=out,
        max_rows_per_dataset=None,
        row_budget=None,
        val_ratio=0.0,
        test_ratio=0.0,
        batch_size=1,
        progress_every=1,
        nested_progress=False,
        bio_scale_compact=True,
        bio_scale_max_sequence_chars=8192,
    )

    row = json.loads((out / "train.jsonl").read_text(encoding="utf-8"))
    node_by_type = {node["type"]: node for node in row["nodes"]}
    assert summary["bio_scale_compact"] is True
    assert summary["total"] == 1
    assert row["task"] == "bio_sequence_scale_pretraining"
    assert row["metadata"]["input_representation"] == "bioselfies"
    assert row["metadata"]["modalities"] == ["rna", "bioselfies"]
    assert "UGM:tokenizer:bioselfies" in row["target_tokens"]
    assert "BIOSEQ:modality:rna" in row["target_tokens"]
    assert node_by_type["bioselfies_compact"]["value"].startswith("[RNA:A][RNA:U]")
    assert node_by_type["rna_sequence_compact"]["features"]["preview_length"] <= 512


def test_count_graph_tokens_budget_guard_writes_summary_then_fails(tmp_path: Path):
    path = tmp_path / "train.jsonl"
    output = tmp_path / "token_counts.json"
    row = {
        "id": "ex0",
        "task": "unit",
        "nodes": [
            {"id": "a", "type": "prompt", "value": "hello", "features": {}},
            {"id": "b", "type": "answer", "value": "world", "features": {}},
        ],
        "edges": [{"src": "a", "dst": "b", "type": "supports", "features": {}}],
        "target_tokens": ["ANSWER:world", "CLAIM:unit"],
        "decoder_orders": [[0, 1]],
        "metadata": {"source_dataset": "unit_dataset"},
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/count_graph_tokens.py"),
            "--path",
            str(path),
            "--output",
            str(output),
            "--max-model-sequence-tokens-total",
            "8",
            "--progress-every",
            "1",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["model_sequence_tokens_untruncated"] == 9
    assert summary["within_model_sequence_token_budget"] is False
