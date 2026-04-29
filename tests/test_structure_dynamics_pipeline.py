from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.check_dataset_integrity import inspect_integrity
from scripts.prepare_structure_dynamics_sources import parse_pdb
from iska_reasoner.data.graphify import graphify_rows
from iska_reasoner.tools import multimodal_metrics_for_example, multimodal_oracle_reward
from iska_reasoner.graph.schema import GraphExample


def test_parse_pdb_to_structure_dynamics_row_and_graph(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("UGM_UMA_BACKEND", "proxy")
    pdb = tmp_path / "toy.pdb"
    pdb.write_text(
        "\n".join(
            [
                "MODEL        1",
                "ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00  0.00           N",
                "ATOM      2  CA  GLY A   1       1.450   0.100   0.000  1.00  0.00           C",
                "ATOM      3  O   GLY A   1       2.100   1.000   0.000  1.00  0.00           O",
                "ENDMDL",
                "MODEL        2",
                "ATOM      1  N   GLY A   1       0.100   0.000   0.000  1.00  0.00           N",
                "ATOM      2  CA  GLY A   1       1.500   0.200   0.000  1.00  0.00           C",
                "ATOM      3  O   GLY A   1       2.200   1.100   0.100  1.00  0.00           O",
                "ENDMDL",
                "CONECT    1    2",
                "CONECT    2    3",
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    row = parse_pdb(pdb)
    assert row["task"] == "conformer_trajectory"
    assert len(row["atoms"]) == 3
    assert len(row["frames"]) == 2
    assert len(row["bonds"]) == 2

    graph_row = next(graphify_rows([row], "local_structure_dynamics_graph_to_graph"))
    ex = GraphExample.from_dict(graph_row)
    assert "UGM:modality:all_atom" in ex.target_tokens
    assert "UGM:modality:trajectory" in ex.target_tokens
    assert "UGM:serializer:pdb" in ex.target_tokens
    assert any(tok.startswith("COORD:x:") for tok in ex.target_tokens)
    assert any(edge.type == "molecular_bond" for edge in ex.edges)
    metrics = multimodal_metrics_for_example(ex)
    assert metrics["multimodal/all_atom_present_rate"] == 1.0
    assert metrics["multimodal/trajectory_present_rate"] == 1.0
    assert metrics["multimodal/frame_count_mean"] == 2.0
    assert multimodal_oracle_reward(ex, ex.target_tokens) >= 0.45


def test_dataset_integrity_detects_stale_summary(tmp_path: Path):
    data_dir = tmp_path / "graphs"
    data_dir.mkdir()
    for split in ("train", "val", "test"):
        (data_dir / f"{split}.jsonl").write_text("{}\n", encoding="utf-8")
    (data_dir / "summary.json").write_text(
        json.dumps({"counts": {"train": 2, "val": 1, "test": 1}, "total": 4}),
        encoding="utf-8",
    )
    result = inspect_integrity(data_dir)
    assert result["ok"] is False
    assert {"split": "train", "expected": 2, "actual": 1} in result["mismatches"]
    assert {"split": "total", "expected": 4, "actual": 3} in result["mismatches"]


def test_dataset_integrity_accepts_matching_summary(tmp_path: Path):
    data_dir = tmp_path / "graphs"
    data_dir.mkdir()
    for split in ("train", "val", "test"):
        (data_dir / f"{split}.jsonl").write_text("{}\n", encoding="utf-8")
    (data_dir / "summary.json").write_text(
        json.dumps({"counts": {"train": 1, "val": 1, "test": 1}, "total": 3}),
        encoding="utf-8",
    )
    result = inspect_integrity(data_dir)
    assert result["ok"] is True
