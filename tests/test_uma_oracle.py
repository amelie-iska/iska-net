from __future__ import annotations

import pytest

from iska_reasoner.graph.schema import GraphExample, Node
from iska_reasoner.oracles import fairchem_repo_status, score_uma_coordinate_candidate, score_uma_oracle_candidate


def _example(smiles: str = "CCO") -> GraphExample:
    return GraphExample(
        id="uma_test",
        task="multimodal_oracle_test",
        nodes=[
            Node(id="task", type="task", value="score with UMA"),
            Node(id="smiles", type="smiles", value=smiles),
            Node(id="temperature", type="temperature", value="325K", features={"kelvin": 325.0}),
        ],
        edges=[],
        target_tokens=["SMILES:" + smiles, "UGM:oracle:uma_feedback"],
        metadata={"smiles": smiles, "temperature": 325.0},
    )


def test_fairchem_repo_is_cloned_and_importable():
    status = fairchem_repo_status("data/external_repos/fairchem")
    assert status["exists"]
    assert status["is_git_repo"]
    assert status["src_exists"]
    assert status["importable"], status.get("error")
    assert "uma-s-1p2" in status["available_models"]


def test_uma_proxy_backend_is_explicit_test_only():
    result = score_uma_oracle_candidate(_example(), ["SMILES:CCO", "UGM:oracle:uma_feedback"], backend="proxy", proxy_reward=0.5)
    assert result.backend == "proxy"
    assert result.available
    assert result.reward == pytest.approx(0.5)
    assert "tests and smoke runs" in result.message


def test_uma_coordinate_proxy_returns_energy_and_forces():
    result = score_uma_coordinate_candidate(
        ["C", "O"],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        temperature_k=350.0,
        backend="proxy",
    )
    assert result.backend == "proxy"
    assert result.available
    assert result.energy_per_atom_ev is not None
    assert result.force_rms_ev_per_a is not None
    assert result.forces_ev_per_a is not None
    assert len(result.forces_ev_per_a) == 2


def test_fairchem_backend_requires_real_candidate_and_reports_unavailable_without_fallback():
    result = score_uma_oracle_candidate(_example("not_smiles"), ["SMILES:not_smiles"], backend="fairchem", strict=False)
    assert result.backend == "fairchem"
    assert not result.available
    assert result.reward == 0.0
    assert result.message


def test_fairchem_strict_mode_raises_on_bad_candidate():
    with pytest.raises(Exception):
        score_uma_oracle_candidate(_example("not_smiles"), ["SMILES:not_smiles"], backend="fairchem", strict=True)
