from __future__ import annotations

from pathlib import Path

from scripts.check_dataset_policy import inspect_policy
from iska_reasoner.data.graphify import graphify_unigenx
from iska_reasoner.data.multimodal import graphify_multimodal
from iska_reasoner.data.phase_policy import ALLOW_STRUCTURE, graph_structure_violations, sanitize_graph_example_for_sequence_only
from iska_reasoner.utils.io import write_jsonl


def test_sequence_only_policy_accepts_selfies_sequence_rows(tmp_path: Path):
    ex = graphify_unigenx(
        {"smiles": "CO", "atomic_symbols": ["C", "O"], "pos": [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]]},
        0,
        "unigenx_qm9_train",
    )
    path = tmp_path / "seq.jsonl"
    write_jsonl(path, [ex.to_dict()])
    result = inspect_policy([path], sequence_only_molecules=True)
    assert result["ok"] is True


def test_sequence_only_policy_rejects_structure_rows(tmp_path: Path):
    ex = graphify_unigenx(
        {"smiles": "CO", "atomic_symbols": ["C", "O"], "pos": [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]]},
        0,
        "unigenx_qm9_train",
        molecular_input_policy=ALLOW_STRUCTURE,
    )
    path = tmp_path / "structure.jsonl"
    write_jsonl(path, [ex.to_dict()])
    result = inspect_policy([path], sequence_only_molecules=True)
    assert result["ok"] is False
    assert result["violation_count"] >= 1


def test_sequence_only_sanitizer_strips_legacy_coordinate_rows():
    ex = graphify_unigenx(
        {"smiles": "CO", "atomic_symbols": ["C", "O"], "pos": [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]], "gap": 1.5},
        0,
        "unigenx_qm9_train",
        molecular_input_policy=ALLOW_STRUCTURE,
    )
    clean = sanitize_graph_example_for_sequence_only(ex)
    assert graph_structure_violations(clean) == []
    assert any(node.type == "smiles" for node in clean.nodes)
    assert not any(node.type in {"atom_symbol", "coordinate_3d", "molecule_property"} for node in clean.nodes)
    assert not any(token.startswith("PROPERTY:") for token in clean.target_tokens)
    assert "ANSWER:molecule_sequence" in clean.target_tokens


def test_sequence_only_policy_allows_oracle_temperature_rows(tmp_path: Path):
    ex = graphify_multimodal(
        {
            "prompt": "Score a candidate.",
            "selfies": "[C][O]",
            "temperature": 333.3,
            "oracle": {"name": "uma", "reward_bin": "medium"},
        },
        0,
        "local_multimodal_graph_to_graph",
    )
    path = tmp_path / "oracle.jsonl"
    write_jsonl(path, [ex.to_dict()])
    result = inspect_policy([path], sequence_only_molecules=True)
    assert result["ok"] is True
