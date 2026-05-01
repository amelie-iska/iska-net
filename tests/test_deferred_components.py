import json
import wave
from pathlib import Path

import torch

from iska_reasoner.data.audio import extract_audio_features
from iska_reasoner.data.curate import curate_files
from iska_reasoner.data.dataset import RandomOrderCollator, extract_numeric_values
from iska_reasoner.data.graphify import graphify_bioactivity, graphify_local_audio, graphify_protein_ec, graphify_protein_ligand_docking, graphify_unigenx
from iska_reasoner.data.vocab import build_vocab
from iska_reasoner.gflownet.trajectory import GraphEditActionSpace, GraphSetPolicy, SubtrajectoryBalanceLoss, TrajectoryBalanceLoss, sample_trajectories
from iska_reasoner.models.random_order_tokengt import RandomOrderTokenGT, RandomOrderTokenGTConfig
from iska_reasoner.topology import ADVANCED_TOPOLOGY_FEATURE_NAMES, summarize_graph_advanced
from iska_reasoner.tropical import TropicalAttention, activation_cell_signature, tropical_max_spanning_arborescence
from iska_reasoner.utils.io import write_jsonl


def test_advanced_topology_and_tropical_modules():
    ex = graphify_bioactivity({"smiles": "CCO", "protein_sequence": "MKT", "standard_value": "12", "standard_units": "nM"}, 0, "bindingdb_local")
    metrics = summarize_graph_advanced(ex)
    assert set(ADVANCED_TOPOLOGY_FEATURE_NAMES) <= set(metrics)
    assert metrics["persistent_laplacian_scales"] >= 0

    attn = TropicalAttention(hidden_dim=8, num_heads=2, hard=True)
    out, weights = attn(torch.randn(2, 4, 8), mask=torch.ones(2, 4, dtype=torch.bool))
    assert out.shape == (2, 4, 8)
    assert weights.shape == (2, 2, 4, 4)
    sig = activation_cell_signature(out)
    assert sig.unique_cells > 0
    tree = tropical_max_spanning_arborescence(["a", "b", "c"], {("a", "b"): 2.0, ("b", "c"): 1.0, ("a", "c"): 0.1}, root="a")
    assert tree


def test_sequence_only_unigenx_sanitizes_structure_fields_and_uses_ar_model_forward():
    ex = graphify_unigenx(
        {"smiles": "C", "atomic_symbols": ["C"], "pos": [[0.1, 0.2, 0.3]], "gap": 1.5},
        0,
        "unigenx_qm9_train",
    )
    values, mask = extract_numeric_values(ex, 8)
    assert sum(mask) == 0
    assert not any(node.type in {"atom", "atom_symbol", "coordinate", "energy", "force"} for node in ex.nodes)
    assert any(node.type in {"smiles", "selfies"} for node in ex.nodes)
    vocab = build_vocab([ex])
    collator = RandomOrderCollator(vocab, max_numeric_targets=0)
    batch = collator([ex])
    model = RandomOrderTokenGT(RandomOrderTokenGTConfig(vocab_size=len(vocab.token_to_id), hidden_dim=32, num_layers=1, num_heads=4, ffn_dim=64))
    out = model(
        input_ids=batch["input_ids"],
        kind_ids=batch["kind_ids"],
        slot_ids=batch["slot_ids"],
        endpoint_ids=batch["endpoint_ids"],
        attention_mask=batch["attention_mask"],
        causal_mask=batch["causal_mask"],
        labels=batch["labels"],
    )
    assert out["loss"].isfinite()
    assert "numeric_diffusion_loss" not in out


def test_science_graphifiers_cover_protein_docking_and_bioactivity():
    protein = graphify_protein_ec({"protein_sequence": "MKTLL", "ec_number": "1.1.1.1"}, 0, "local_ec")
    assert any(node.type == "ec_number" for node in protein.nodes)
    docking = graphify_protein_ligand_docking(
        {"ligand_smiles": "CCO", "protein_sequence": "MKT", "pocket_atoms": ["C"], "pocket_coords": [[0, 1, 2]], "ligand_coords": [[1, 2, 3]], "affinity": "7.1"},
        0,
        "pdbbind_docking_local",
    )
    assert any(node.type == "protein_coordinate" for node in docking.nodes)
    assert any(node.type == "ligand_coordinate" for node in docking.nodes)
    bio = graphify_bioactivity({"smiles": "CCO", "target_sequence": "MKT", "standard_type": "Ki", "standard_value": "10", "standard_units": "nM"}, 0, "chembl_local")
    assert bio.task == "biomed_bioactivity"


def test_audio_feature_extraction_and_graphification(tmp_path: Path):
    wav_path = tmp_path / "tiny.wav"
    with wave.open(str(wav_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(8000)
        handle.writeframes(b"\x00\x00" * 800)
    features = extract_audio_features(wav_path)
    assert features.available
    ex = graphify_local_audio({"instruction": "מה נשמע?", "output": "ציפור", "task": "identify", "local_audio_path": str(wav_path)}, 0, "local_audio_reasoning")
    assert any(node.type == "audio_features" for node in ex.nodes)


def test_curation_near_dedup_license_and_contamination(tmp_path: Path):
    ex1 = graphify_bioactivity({"smiles": "CCO", "protein_sequence": "MKT", "standard_value": "12"}, 0, "bindingdb_local")
    ex2 = graphify_bioactivity({"smiles": "CCO", "protein_sequence": "MKT ", "standard_value": "12"}, 1, "bindingdb_local")
    ex2.id = "near_dup"
    ex2.metadata["license"] = "CC-BY"
    ex3 = graphify_bioactivity({"smiles": "N=C=O", "protein_sequence": "AAA", "standard_value": "99"}, 2, "bindingdb_local")
    ex3.metadata["license"] = "blocked-license"
    input_path = tmp_path / "in.jsonl"
    write_jsonl(input_path, [ex1.to_dict(), ex2.to_dict(), ex3.to_dict()])
    summary = curate_files([input_path], tmp_path / "out", near_dedup_threshold=0.8, blocked_license_patterns=["blocked"])
    assert summary["kept_rows"] == 1
    assert summary["near_duplicates_removed"] >= 1
    assert summary["license_blocked"] == 1


def test_gflownet_context_backward_subtrajectory_and_edit_space():
    target_mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.float32)
    context = torch.randn(1, 7)
    policy = GraphSetPolicy(num_actions=4, hidden_dim=16, context_dim=7)
    backward = GraphSetPolicy(num_actions=4, hidden_dim=16, context_dim=7)
    traj = sample_trajectories(policy, target_mask, max_steps=3, epsilon=0.1, backward_policy=backward, context=context)
    tb = TrajectoryBalanceLoss()(traj)
    stb = SubtrajectoryBalanceLoss()(traj)
    assert tb.isfinite() and stb.isfinite()
    space = GraphEditActionSpace(["A", "B"])
    assert space.decode(0) == ("ADD", "A")
    assert space.decode(len(space.actions) - 1) == ("STOP", None)
