import json

import torch

from iska_reasoner.gflownet.trainer import _candidate_vocab, _structure_dynamics_reward, _temperature_diversity_bonuses
from iska_reasoner.gflownet.trajectory import GraphSetPolicy, TrajectoryBalanceLoss, sample_trajectories
from iska_reasoner.graph.schema import GraphExample, Node


def test_trajectory_balance_smoke():
    target_mask = torch.tensor([[1, 1, 0, 0], [1, 0, 1, 0]], dtype=torch.float32)
    policy = GraphSetPolicy(num_actions=4, hidden_dim=16)
    loss_fn = TrajectoryBalanceLoss()
    traj = sample_trajectories(policy, target_mask, max_steps=3, epsilon=0.1)
    loss = loss_fn(traj)
    assert loss.isfinite()
    loss.backward()
    assert loss_fn.log_z.grad is not None


def test_temperature_diversity_bonus_prefers_high_temperature_terminal_variation():
    examples = [
        GraphExample(
            id="hot0",
            task="multimodal",
            nodes=[Node(id="temperature", type="temperature", value="400K", features={"kelvin": 400.0})],
            edges=[],
            target_tokens=[],
        ),
        GraphExample(
            id="hot1",
            task="multimodal",
            nodes=[Node(id="temperature", type="temperature", value="400K", features={"kelvin": 400.0})],
            edges=[],
            target_tokens=[],
        ),
        GraphExample(
            id="cool",
            task="multimodal",
            nodes=[Node(id="temperature", type="temperature", value="300K", features={"kelvin": 300.0})],
            edges=[],
            target_tokens=[],
        ),
    ]
    terminal = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    bonuses, metrics = _temperature_diversity_bonuses(examples, terminal, weight=0.1)
    assert bonuses[0] > 0.0
    assert bonuses[1] > 0.0
    assert bonuses[2] == 0.0
    assert metrics["high_temperature_unique_terminal_states"] == 2.0
    assert metrics["high_temperature_terminal_hamming"] > 0.0


def test_structure_dynamics_gflownet_reward_prefers_oracle_action_coverage():
    example = GraphExample(
        id="dyn",
        task="structure_dynamics_proxy",
        nodes=[],
        edges=[],
        target_tokens=[
            "UGM:task:structure_dynamics_proxy",
            "UGM:oracle:uma_feedback",
            "SEQ_STRUCT_DYN_PROXY:temperature_conditioned",
            "INTERNAL_COORD:protein_phi",
            "ADAPTIVE_PATCH:residue_atom_patch",
            "CONTACT_PATCH:hbond",
            "TOKEN_MOTION:uma:refine:b32",
        ],
    )
    verifier = type("Verifier", (), {"reward": 0.5})()
    rich_reward, metrics = _structure_dynamics_reward(
        example,
        [
            "UGM:task:structure_dynamics_proxy",
            "UGM:oracle:uma_feedback",
            "SEQ_STRUCT_DYN_PROXY:temperature_conditioned",
            "INTERNAL_COORD:protein_phi",
            "ADAPTIVE_PATCH:residue_atom_patch",
            "CONTACT_PATCH:hbond",
            "TOKEN_MOTION:uma:refine:b32",
        ],
        verifier,
    )
    sparse_reward, _ = _structure_dynamics_reward(example, ["UGM:task:structure_dynamics_proxy"], verifier)
    assert rich_reward > sparse_reward
    assert metrics["structure_dynamics_internal_rate"] == 1.0
    assert metrics["structure_dynamics_patch_rate"] == 1.0
    assert metrics["structure_dynamics_contact_rate"] == 1.0


def test_structure_dynamics_candidate_vocab_filters_to_dynamics_tokens(tmp_path):
    path = tmp_path / "graphs.jsonl"
    rows = [
        GraphExample(
            id="a",
            task="structure_dynamics_proxy",
            nodes=[],
            edges=[],
            target_tokens=[
                "ANSWER:plain",
                "INTERNAL_COORD:protein_phi",
                "ADAPTIVE_PATCH:residue_atom_patch",
                "UGM:oracle:uma_feedback",
            ],
        ).to_dict(),
        GraphExample(
            id="b",
            task="structure_dynamics_proxy",
            nodes=[],
            edges=[],
            target_tokens=["CONTACT_PATCH:hbond", "TOKEN_MOTION:uma:refine:b32"],
        ).to_dict(),
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    from iska_reasoner.data.dataset import GraphJsonlDataset

    dataset = GraphJsonlDataset(path)
    candidates, _ = _candidate_vocab(dataset, 16, mode="structure_dynamics")
    assert "ANSWER:plain" not in candidates
    assert "INTERNAL_COORD:protein_phi" in candidates
    assert "ADAPTIVE_PATCH:residue_atom_patch" in candidates
    assert "CONTACT_PATCH:hbond" in candidates


def test_structure_dynamics_candidate_vocab_derives_biomed_candidates_when_rows_are_legacy(tmp_path):
    path = tmp_path / "legacy_biomed.jsonl"
    row = GraphExample(
        id="legacy",
        task="biomolecular_complex_affinity",
        nodes=[
            Node(id="protein", type="protein_sequence", value="MKT"),
            Node(id="ligand", type="smiles", value="CCO"),
            Node(id="temperature", type="temperature", value="330K", features={"kelvin": 330.0}),
        ],
        edges=[],
        target_tokens=["BIOMED:complex_affinity", "ANSWER:legacy"],
    ).to_dict()
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    from iska_reasoner.data.dataset import GraphJsonlDataset

    dataset = GraphJsonlDataset(path)
    candidates, _ = _candidate_vocab(dataset, 64, mode="structure_dynamics")
    assert "ANSWER:legacy" not in candidates
    assert "UGM:task:structure_dynamics_proxy" in candidates
    assert "ALL_ATOM_CARTESIAN:enabled" in candidates
    assert "CARTESIAN_ATOM:protein:CA" in candidates
    assert "CARTESIAN_ATOM:ligand:heavy_atom" in candidates
