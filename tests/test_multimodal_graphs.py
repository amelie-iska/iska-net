from __future__ import annotations

import os
import subprocess
from pathlib import Path

import torch

from iska_reasoner.data.bioselfies import add_bioselfies_graph, bioselfies_from_modalities
from iska_reasoner.data.dataset import RandomOrderCollator, coordinate_targets_by_index, extract_numeric_values
from iska_reasoner.data.graphify import graphify_rows
from iska_reasoner.data.multimodal import (
    BOND_TYPES,
    graphify_multimodal,
    multimodal_reference_tokens,
    records_to_multimodel_pdb,
    records_to_xyz_trajectory,
    write_mdtraj_trajectory,
)
from iska_reasoner.data.phase_policy import ALLOW_STRUCTURE, graph_structure_violations
from iska_reasoner.data.motifs import (
    build_motif_vocabulary,
    derive_structure_sequence_motifs_from_atoms,
    parse_cath_names,
    parse_interpro_json,
    parse_prosite_dat,
    parse_rfam_family,
)
from iska_reasoner.data.vocab import build_vocab
from iska_reasoner.graph.orders import build_orders, oracle_enabling_order, scientific_graph_order
from iska_reasoner.inference.generate import predict_uma_coordinate_frame
from iska_reasoner.models.random_order_tokengt import RandomOrderTokenGT, RandomOrderTokenGTConfig
from iska_reasoner.tools import multimodal_metrics_for_example, multimodal_oracle_reward, verify_example_tokens
from iska_reasoner.training.uma_coordinate import uma_coordinate_head_oracle_loss, uma_internal_coordinate_head_oracle_loss


def _row():
    return {
        "prompt": "Generate structure records.",
        "task": "conformer_trajectory",
        "protein_sequence": "MKTW",
        "sequence_motifs": [{"source": "prosite", "accession": "PS00001", "name": "Toy motif"}],
        "sequence_motifs_from_structure": [{"source": "cath", "accession": "1.10.8.10", "name": "Safe CATH sequence motif"}],
        "structure_motifs": [{"source": "cath", "accession": "1.10.8.10", "name": "Toy CATH motif"}],
        "structure_derived_sequence_motifs": [{"source": "local_structure", "accession": "GLY-GLY:contact_low"}],
        "selfies": "[C][=O][O]",
        "dna_sequence": "ATGC",
        "temperature": 315.5,
        "atoms": [
            {"element": "C", "name": "C1"},
            {"element": "O", "name": "O1"},
        ],
        "bonds": [{"src": 0, "dst": 1, "bond_type": "double"}],
        "distances": [[0, 1, 1.25]],
        "frames": [[[0.0, 0.0, 0.0], [1.25, 0.0, 0.0]]],
        "energy": -7.5,
        "forces": [[0.0, 0.0, 0.0], [0.2, -0.1, 0.0]],
        "function_description": "Toy target.",
    }


def test_multimodal_reference_vocab_contains_required_families():
    tokens = set(multimodal_reference_tokens())
    assert "UGM:graph_to_graph" in tokens
    assert "AA:W" in tokens
    assert "SELFIES:[=O]" in tokens
    assert "BOND:aromatic" in tokens
    assert "BOND:phosphodiester" in tokens
    assert "TEMP:300K" in tokens
    assert "TEMP:CONTINUOUS" in tokens
    assert "TEMP_BIN:310_320K" in tokens
    assert "PDB:MODEL" in tokens
    assert "FORCE:dir:px" in tokens
    assert "SEQ_MOTIF:core:coiled_coil" in tokens
    assert "SEQ_MOTIF_FROM_STRUCTURE:core:contact_patch_sequence" in tokens
    assert "STRUCT_MOTIF:core:cath_domain" in tokens
    assert "STRUCT_DERIVED_SEQ_MOTIF:core:contact_patch_sequence" in tokens
    assert "ATTN_BIN:sequence_to_motion:b63" in tokens
    assert "ATTN_COARSE:sequence_to_motion:critical" in tokens
    assert "TOKEN_COUPLING:uma:sequence_oracle:b63" in tokens
    assert "UMA_INFLUENCE:uma:trajectory_physics:b63" in tokens
    assert "TOKEN_MOTION:uma:temperature_scaled:b59" in tokens
    assert "UMA_TRAJ_BIN:temperature_scaled:b59" in tokens
    assert "SEQ_STRUCT_DYN_PROXY:temperature_conditioned" in tokens
    assert "UGM:oracle:uma_feedback" in tokens
    assert "peptide" in BOND_TYPES
    assert "UGM:tokenizer:bioselfies" in tokens
    assert "BIOSELFIES:[AA:A]" in tokens
    assert "BIOSELFIES:[RNA:A]" in tokens
    assert "HYBRID:atom_patch" in tokens
    assert "HBOND:candidate" in tokens
    assert "TORSION:sidechain_chi" in tokens
    assert "INTERNAL_COORD:protein_phi" in tokens
    assert "INTERNAL_COORD_QUERY:protein_phi" in tokens
    assert "ADAPTIVE_PATCH:residue_atom_patch" in tokens
    assert "CONTACT_PATCH:hbond" in tokens
    assert "SEQ_STRUCT_DYN_PROXY:all_atom_cartesian" in tokens
    assert "ALL_ATOM_CARTESIAN:enabled" in tokens
    assert "CARTESIAN_ATOM:protein:CA" in tokens
    assert "CARTESIAN_ATOM:ligand:heavy_atom" in tokens
    assert "CARTESIAN_FRAME:temperature_conditioned" in tokens


def test_bioselfies_decoder_is_total_and_graph_valid():
    nodes = []
    edges = []
    result = add_bioselfies_graph(
        nodes,
        edges,
        "[AA:M][AA:K][LINK:peptide][AA:T][CHAIN:break][RNA:A][RNA:U][HBOND:candidate][PATCH:open][UNK:???]",
    )
    assert result.residue_count == 3
    assert result.base_count == 2
    assert result.warnings
    assert "UGM:tokenizer:bioselfies" in result.target_tokens
    assert "AA:M" in result.target_tokens
    assert "RNA:U" in result.target_tokens
    assert "BOND:peptide" in result.target_tokens
    assert "HBOND:candidate" in result.target_tokens
    assert "HYBRID:open_patch" in result.target_tokens
    assert "BIOSELFIES:UNKNOWN" in result.target_tokens
    node_ids = {node.id for node in nodes}
    assert all(edge.src in node_ids and edge.dst in node_ids for edge in edges)


def test_bioselfies_only_graphification_stays_sequence_only_and_feeds_uma_queries():
    row = {
        "prompt": "Use BioSELFIES-only graph tokens for oracle-guided dynamics.",
        "task": "structure_dynamics_proxy",
        "input_representation": "bioselfies",
        "protein_sequence": "MKT",
        "rna_sequence": "AUG",
        "temperature": 355.0,
        "oracle": {"name": "uma"},
    }
    assert bioselfies_from_modalities(row).startswith("[AA:M][AA:K][AA:T]")
    ex = graphify_multimodal(row, 11, "local_multimodal_graph_to_graph")
    assert ex.metadata["bioselfies_enabled"] is True
    assert ex.metadata["bioselfies_only"] is True
    assert "bioselfies" in ex.metadata["modalities"]
    assert "UGM:tokenizer:bioselfies" in ex.target_tokens
    assert "BIOSELFIES:[AA:M]" in ex.target_tokens
    assert "BIOSELFIES:[RNA:A]" in ex.target_tokens
    assert not any(node.id == "protein" for node in ex.nodes)
    assert not graph_structure_violations(ex)

    vocab = build_vocab([ex], extra_tokens=multimodal_reference_tokens())
    collator = RandomOrderCollator(
        vocab,
        max_source_tokens=64,
        max_target_tokens=32,
        max_seq_len=160,
        max_uma_coordinate_atoms=8,
        order_mode="first",
    )
    batch = collator([ex])
    assert batch["uma_coordinate_query_mask"].sum().item() == 8
    assert batch["uma_coordinate_symbols"][0][:4] == ["N", "C", "C", "O"]


def test_multimodal_graphification_defaults_to_sequence_only():
    ex = graphify_multimodal(_row(), 0, "local_multimodal_graph_to_graph")
    assert ex.task == "multimodal_graph_to_graph"
    assert any(node.type == "amino_acid" and node.value == "W" for node in ex.nodes)
    assert any(node.type == "selfies_token" for node in ex.nodes)
    assert any(node.type == "dna_base" for node in ex.nodes)
    assert any(node.type == "sequence_motif" for node in ex.nodes)
    assert any(node.type == "sequence_motif_from_structure_vocab" for node in ex.nodes)
    assert any(node.type == "attention_coupling_bin" for node in ex.nodes)
    assert any(node.type == "token_motion_prior" for node in ex.nodes)
    assert any(node.type == "sequence_structure_dynamics_proxy" for node in ex.nodes)
    assert "SEQ_MOTIF:prosite:PS00001" in ex.target_tokens
    assert "SEQ_MOTIF_FROM_STRUCTURE:cath:1.10.8.10" in ex.target_tokens
    assert "UGM:oracle:uma_feedback" in ex.target_tokens
    assert "UGM:task:structure_dynamics_proxy" in ex.target_tokens
    assert any(tok.startswith("ATTN_BIN:sequence_to_motion:") for tok in ex.target_tokens)
    assert any(tok.startswith("TOKEN_COUPLING:uma:temperature_oracle:") for tok in ex.target_tokens)
    assert any(tok.startswith("TOKEN_MOTION:uma:") for tok in ex.target_tokens)
    assert "SEQ_STRUCT_DYN_PROXY:no_structure_file" in ex.target_tokens
    assert "INTERNAL_COORD:protein_phi" in ex.target_tokens
    assert "ADAPTIVE_PATCH:residue_atom_patch" in ex.target_tokens
    assert "CONTACT_PATCH:hbond" in ex.target_tokens
    assert "SEQ_STRUCT_DYN_PROXY:all_atom_cartesian" in ex.target_tokens
    assert "ALL_ATOM_CARTESIAN:enabled" in ex.target_tokens
    assert "CARTESIAN_ATOM:protein:CA" in ex.target_tokens
    assert not graph_structure_violations(ex)
    assert ex.metadata["ignored_structure_fields"]
    assert ex.metadata["temperature"] == 315.5
    assert ex.metadata["temperature_anchor"] == "325K"
    assert "TEMP:CONTINUOUS" in ex.target_tokens
    assert "TEMP_ANCHOR:325K" in ex.target_tokens
    assert "TEMP_BIN:310_320K" in ex.target_tokens
    temp = next(node for node in ex.nodes if node.type == "temperature")
    assert temp.features["kelvin_norm"] == 0.155
    metrics = multimodal_metrics_for_example(ex)
    assert metrics["multimodal/modality_count_mean"] >= 3
    assert metrics["multimodal/bond_type_coverage_rate"] == 0.0
    assert metrics["multimodal/attention_bin_count_mean"] > 0
    assert metrics["multimodal/uma_coupling_bin_count_mean"] > 0
    assert metrics["multimodal/uma_influence_bin_count_mean"] > 0
    assert metrics["multimodal/token_motion_prior_count_mean"] > 0
    assert metrics["multimodal/sequence_structure_dynamics_proxy_rate"] == 1.0
    assert not any(node.type == "coordinate_3d" for node in ex.nodes)


def test_four_modality_sequence_only_oracle_coupling_and_motion_records():
    row = {
        "prompt": "Predict temperature-conditioned structure-dynamics proxy records from sequences only.",
        "task": "structure_dynamics_proxy",
        "selfies": "[C][N][O]",
        "protein_sequence": "MKTWY",
        "rna_sequence": "AUGCGU",
        "dna_sequence": "ATGCGT",
        "temperature": 342.25,
        "function_description": "Toy function evidence.",
        "sequence_motifs": [{"source": "interpro", "accession": "IPR000001"}],
        "sequence_motifs_from_structure": [{"source": "cath", "accession": "1.10.8.10"}],
        "oracle": {"name": "uma"},
    }
    ex = graphify_multimodal(row, 3, "local_multimodal_graph_to_graph")
    modalities = set(ex.metadata["modalities"])
    assert {"selfies", "protein", "rna", "dna"}.issubset(modalities)
    assert not graph_structure_violations(ex)
    for modality in ["selfies", "protein", "rna", "dna"]:
        assert f"SEQ_STRUCT_DYN_PROXY:input:{modality}" in ex.target_tokens
    assert "SEQ_MOTIF_FROM_STRUCTURE:cath:1.10.8.10" in ex.target_tokens
    assert any(tok.startswith("ATTN_BIN:temperature_to_oracle:") for tok in ex.target_tokens)
    assert any(tok.startswith("TOKEN_COUPLING:uma:sequence_oracle:") for tok in ex.target_tokens)
    assert any(tok.startswith("TOKEN_MOTION:uma:") for tok in ex.target_tokens)
    assert "SEQ_STRUCT_DYN_PROXY:temperature_conditioned" in ex.target_tokens
    assert "UGM:oracle:uma_feedback" in ex.target_tokens


def test_uma_fine_bins_are_stage_gated():
    plain = graphify_multimodal(
        {
            "task": "function_description",
            "protein_sequence": "MKTWYV",
            "temperature": 333.0,
            "function_description": "Binds nucleotide cofactors in a sequence-grounded toy annotation.",
        },
        4,
        "local_multimodal_graph_to_graph",
    )
    stage_only_prefixes = (
        "ATTN_BIN:",
        "ATTN_COARSE:",
        "TOKEN_COUPLING:uma:",
        "UMA_INFLUENCE:uma:",
        "TOKEN_MOTION:uma:",
        "UMA_TRAJ_BIN:",
        "SEQ_STRUCT_DYN_PROXY:",
        "UGM:oracle:uma_feedback",
    )
    assert "TEMP:CONTINUOUS" in plain.target_tokens
    assert any(tok.startswith("ANSWER:Binds nucleotide cofactors") for tok in plain.target_tokens)
    assert not any(tok.startswith(stage_only_prefixes) for tok in plain.target_tokens)
    assert not any(node.type in {"attention_coupling_bin", "uma_influence_bin", "token_motion_prior"} for node in plain.nodes)

    staged = graphify_multimodal(
        {
            "task": "function_description",
            "protein_sequence": "MKTWYV",
            "temperature": 333.0,
            "function_description": "Binds nucleotide cofactors in a sequence-grounded toy annotation.",
            "enable_uma_binning": True,
        },
        5,
        "local_multimodal_graph_to_graph",
    )
    assert "UGM:oracle:uma_feedback" in staged.target_tokens
    assert any(tok.startswith("ATTN_BIN:") for tok in staged.target_tokens)
    assert any(tok.startswith("UMA_INFLUENCE:uma:") for tok in staged.target_tokens)
    assert any(tok.startswith("TOKEN_MOTION:uma:") for tok in staged.target_tokens)


def test_protrek_style_function_description_rows_are_sequence_only_graphs():
    row = {
        "protein_sequence": "MKTWYV",
        "function_description": "Binds nucleotide cofactors in a sequence-grounded toy annotation.",
        "sequence_motifs_from_structure": [{"source": "cath", "accession": "1.10.8.10"}],
    }
    graph_row = next(graphify_rows([row], "protrek_sequence_function"))
    ex = graphify_multimodal(
        {
            "protein_sequence": row["protein_sequence"],
            "function_description": row["function_description"],
            "sequence_motifs_from_structure": row["sequence_motifs_from_structure"],
            "temperature": 333.0,
            "oracle": {"name": "uma"},
        },
        0,
        "local_multimodal_graph_to_graph",
    )
    assert graph_row["task"] == "unigenx_ec_protein_generation"
    assert any(node["type"] == "function_description" for node in graph_row["nodes"])
    assert any(node["type"] == "sequence_motif_from_structure_vocab" for node in graph_row["nodes"])
    assert any(tok.startswith("ANSWER:Binds nucleotide cofactors") for tok in graph_row["target_tokens"])
    assert "SEQ_MOTIF_FROM_STRUCTURE:cath:1.10.8.10" in graph_row["target_tokens"]
    assert "UGM:task:function_description" in graph_row["target_tokens"]
    assert not graph_structure_violations(ex)
    assert "SEQ_STRUCT_DYN_PROXY:function_grounded" in ex.target_tokens
    assert any(tok.startswith("ATTN_BIN:function_to_reason:") for tok in ex.target_tokens)


def test_uniprot_binding_site_and_feature_rows_become_sequence_grounded_graphs():
    row = {
        "Entry": "P12345",
        "Sequence": "MKTWYV",
        "Protein names": "Toy kinase",
        "Gene Names": "toyK",
        "Organism": "Example organism",
        "Gene Ontology IDs": "GO:0005524; GO:0004672",
        "Keywords": "ATP-binding; Kinase",
        "Binding site": [{"type": "binding site", "ligand": "ATP", "position": 4, "description": "ATP binding"}],
        "Subcellular location [CC]": "Cytoplasm",
        "Cofactor": "Mg2+",
        "Catalytic activity": "ATP + substrate = ADP + product",
        "function_description": "Binds ATP and phosphorylates substrate proteins.",
    }
    graph_row = next(graphify_rows([row], "uniprot_features_local_export"))
    ex = graphify_multimodal(
        {
            "task": "structure_dynamics_proxy",
            "protein_sequence": row["Sequence"],
            "temperature": 330.0,
            "oracle": {"name": "uma"},
        },
        0,
        "local_multimodal_graph_to_graph",
    )
    assert graph_row["task"] == "unigenx_ec_protein_generation"
    assert "UGM:tokenizer:bioselfies" in graph_row["target_tokens"]
    assert "UNIPROT:feature:binding_site" in graph_row["target_tokens"]
    assert "SEQ_STRUCT_DYN_PROXY:all_atom_cartesian" in graph_row["target_tokens"]
    assert "ALL_ATOM_CARTESIAN:enabled" in graph_row["target_tokens"]
    assert "CARTESIAN_ATOM:protein:CA" in graph_row["target_tokens"]
    assert any(tok.startswith("UNIPROT:go:") for tok in graph_row["target_tokens"])
    assert any(tok.startswith("UNIPROT:keyword:") for tok in graph_row["target_tokens"])
    assert any(node["type"] == "uniprot_feature" for node in graph_row["nodes"])
    assert any(edge["type"] == "marks_binding_site" for edge in graph_row["edges"])
    assert not graph_structure_violations(ex)


def test_biomolecular_complex_affinity_rows_support_non_ligand_complexes():
    row = {
        "protein_sequence_a": "MKTWYV",
        "protein_sequence_b": "ACDEFG",
        "Kd": "12",
        "units": "nM",
        "interaction_type": "protein_protein",
        "temperature": "310K",
        "pH": "7.4",
    }
    graph_row = next(graphify_rows([row], "biomolecular_complex_affinity_local"))
    assert graph_row["task"] == "biomolecular_complex_affinity"
    assert "BIOMED:complex_affinity" in graph_row["target_tokens"]
    assert "COMPLEX:component:protein" in graph_row["target_tokens"]
    assert "COMPLEX:interaction:protein_protein" in graph_row["target_tokens"]
    assert "UGM:tokenizer:bioselfies" in graph_row["target_tokens"]
    assert "SEQ_STRUCT_DYN_PROXY:input:protein" in graph_row["target_tokens"]
    assert "CARTESIAN_ATOM:protein:CA" in graph_row["target_tokens"]
    assert any(tok.startswith("AFFINITY:Kd:12") for tok in graph_row["target_tokens"])
    assert any(node["type"] == "binding_affinity" for node in graph_row["nodes"])
    assert any(edge["type"] == "forms_complex_with" for edge in graph_row["edges"])


def test_biomed_annotations_affinity_direct_script_dry_run(tmp_path):
    (tmp_path / "uniprot.tsv").write_text(
        "Entry\tSequence\tProtein names\nP12345\tMKTWYV\tToy kinase\n",
        encoding="utf-8",
    )
    (tmp_path / "affinity.tsv").write_text(
        "protein_sequence_a\tprotein_sequence_b\tKd\tunits\nMKTWYV\tACDEFG\t12\tnM\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env.update(
        {
            "DRY_RUN": "1",
            "RUN_ID": "pytest-biomed-direct",
            "LOG_ROOT": str(tmp_path / "logs"),
            "DATA_DIR": str(tmp_path / "processed"),
            "UNIPROT_GRAPH_JSONL": str(tmp_path / "uniprot" / "all.jsonl"),
            "AFFINITY_GRAPH_JSONL": str(tmp_path / "affinity" / "all.jsonl"),
            "UNIPROT_FEATURES_INPUTS": str(tmp_path / "uniprot.tsv"),
            "AFFINITY_INPUTS": str(tmp_path / "affinity.tsv"),
            "TRAIN_PHASES": "all",
            "WANDB_CONFIG": "config/train/overrides/wandb_offline.yaml",
        }
    )
    result = subprocess.run(
        ["bash", "scripts/train_biomed_annotations_affinity_direct.sh"],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
    assert "Direct UniProt + affinity training" in result.stdout
    assert "config/train/biomed_annotations_affinity_250m.yaml" in result.stdout
    assert "config/train/biomed_annotations_affinity_gflownet_sft_4090.yaml" in result.stdout
    assert "config/train/biomed_annotations_affinity_structure_dynamics_gflownet_4090.yaml" in result.stdout


def test_biomed_annotations_affinity_direct_script_rejects_placeholder_inputs(tmp_path):
    env = os.environ.copy()
    env.update(
        {
            "DRY_RUN": "1",
            "RUN_ID": "pytest-biomed-placeholder",
            "LOG_ROOT": str(tmp_path / "logs"),
            "DATA_DIR": str(tmp_path / "processed"),
            "UNIPROT_GRAPH_JSONL": str(tmp_path / "uniprot" / "all.jsonl"),
            "AFFINITY_GRAPH_JSONL": str(tmp_path / "affinity" / "all.jsonl"),
            "UNIPROT_FEATURES_INPUTS": "/path/to/uniprot_features.tsv",
            "AFFINITY_INPUTS": "/path/to/complex_affinity.tsv",
            "WANDB_CONFIG": "config/train/overrides/wandb_offline.yaml",
        }
    )
    result = subprocess.run(
        ["bash", "scripts/train_biomed_annotations_affinity_direct.sh"],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "placeholder path" in result.stderr
    assert "Replace it with a real local" in result.stderr


def test_multimodal_graphification_allows_structure_only_when_explicit():
    ex = graphify_multimodal(_row(), 0, "local_multimodal_graph_to_graph", molecular_input_policy=ALLOW_STRUCTURE)
    assert any(node.type == "structure_motif" for node in ex.nodes)
    assert any(node.type == "structure_derived_sequence_motif" for node in ex.nodes)
    assert any(edge.type == "molecular_bond" and edge.features["bond_type"] == "double" for edge in ex.edges)
    assert "STRUCT_MOTIF:cath:1.10.8.10" in ex.target_tokens
    assert any(tok.startswith("STRUCT_DERIVED_SEQ_MOTIF:") for tok in ex.target_tokens)
    assert any(node.type == "coordinate_3d" for node in ex.nodes)
    assert "UGM:serializer:pdb" in ex.target_tokens
    assert "UGM:oracle:uma_feedback" in ex.target_tokens
    values, mask = extract_numeric_values(ex, 8)
    assert sum(mask) >= 3
    assert any(value != 0.0 for value in values)


def test_multimodal_dispatch_and_collator_build():
    rows = list(graphify_rows([_row()], "local_multimodal_graph_to_graph"))
    ex = graphify_multimodal(_row(), 0, "local_multimodal_graph_to_graph", molecular_input_policy=ALLOW_STRUCTURE)
    assert rows[0]["task"] == "multimodal_graph_to_graph"
    vocab = build_vocab([ex], extra_tokens=multimodal_reference_tokens())
    collator = RandomOrderCollator(vocab, max_source_tokens=128, max_target_tokens=96, max_numeric_targets=8)
    batch = collator([ex])
    assert batch["input_ids"].shape[0] == 1
    assert batch["numeric_mask"].sum().item() >= 3
    assert batch["source_numeric_features"].sum().item() > 0
    assert batch["coordinate_targets"].shape[-1] == 3
    assert batch["coordinate_mask"].sum().item() >= 3


def test_coordinate_targets_align_with_coord_tokens():
    ex = graphify_multimodal(_row(), 0, "local_multimodal_graph_to_graph", molecular_input_policy=ALLOW_STRUCTURE)
    targets, mask = coordinate_targets_by_index(ex)
    coord_idx = ex.target_tokens.index("COORD:f0:a1:x:pos_near")
    assert targets[coord_idx] == [1.25, 0.0, 0.0]
    assert mask[coord_idx] == [1.0, 0.0, 0.0]
    y_idx = ex.target_tokens.index("COORD:f0:a1:y:zero")
    assert targets[y_idx] == [1.25, 0.0, 0.0]
    assert mask[y_idx] == [0.0, 1.0, 0.0]


def test_coordinate_head_forward_backward_on_structure_batch():
    ex = graphify_multimodal(_row(), 0, "local_multimodal_graph_to_graph", molecular_input_policy=ALLOW_STRUCTURE)
    vocab = build_vocab([ex], extra_tokens=multimodal_reference_tokens())
    collator = RandomOrderCollator(
        vocab,
        max_source_tokens=24,
        max_target_tokens=192,
        max_seq_len=320,
        max_numeric_targets=8,
        order_mode="first",
    )
    batch = collator([ex])
    cfg = RandomOrderTokenGTConfig(
        vocab_size=len(vocab.token_to_id),
        hidden_dim=48,
        num_layers=1,
        num_heads=4,
        ffn_dim=96,
        max_seq_len=320,
        max_nodes=128,
        max_slots=64,
        endpoint_dim=16,
        identifier_dim=16,
        coordinate_head_enabled=True,
        coordinate_target_scale=10.0,
    )
    model = RandomOrderTokenGT(cfg)
    out = model(
        input_ids=batch["input_ids"],
        kind_ids=batch["kind_ids"],
        slot_ids=batch["slot_ids"],
        endpoint_ids=batch["endpoint_ids"],
        identifier_ids=batch["identifier_ids"],
        source_numeric_features=batch["source_numeric_features"],
        attention_mask=batch["attention_mask"],
        causal_mask=batch["causal_mask"],
        labels=batch["labels"],
        coordinate_targets=batch["coordinate_targets"],
        coordinate_mask=batch["coordinate_mask"],
    )
    assert out["coordinate_supervised_axes"].item() >= 3
    assert torch.isfinite(out["coordinate_loss"])
    total = out["loss"] + 0.1 * out["coordinate_loss"]
    total.backward()
    grad_norm = sum(
        param.grad.detach().abs().sum().item()
        for name, param in model.named_parameters()
        if name.startswith("coordinate_head") and param.grad is not None
    )
    assert grad_norm > 0.0


def test_uma_coordinate_queries_come_from_protein_sequence_without_structure_labels():
    ex = graphify_multimodal(
        {
            "task": "structure_dynamics_proxy",
            "protein_sequence": "MKT",
            "temperature": 340.0,
            "oracle": {"name": "uma"},
        },
        9,
        "local_multimodal_graph_to_graph",
    )
    assert not graph_structure_violations(ex)
    vocab = build_vocab([ex], extra_tokens=multimodal_reference_tokens())
    collator = RandomOrderCollator(
        vocab,
        max_source_tokens=48,
        max_target_tokens=32,
        max_seq_len=128,
        max_uma_coordinate_atoms=8,
        order_mode="first",
    )
    batch = collator([ex])
    assert batch["uma_coordinate_query_mask"].sum().item() == 8
    assert batch["uma_coordinate_symbols"][0][:4] == ["N", "C", "C", "O"]
    assert batch["coordinate_mask"].sum().item() == 0


def test_internal_coordinate_query_slots_come_from_sequence_without_structure_labels():
    ex = graphify_multimodal(
        {
            "task": "structure_dynamics_proxy",
            "protein_sequence": "MKT",
            "temperature": 340.0,
            "oracle": {"name": "uma"},
        },
        19,
        "local_multimodal_graph_to_graph",
    )
    assert not graph_structure_violations(ex)
    vocab = build_vocab([ex], extra_tokens=multimodal_reference_tokens())
    collator = RandomOrderCollator(
        vocab,
        max_source_tokens=48,
        max_target_tokens=32,
        max_seq_len=160,
        max_internal_coordinate_actions=12,
        order_mode="first",
    )
    batch = collator([ex])
    assert batch["internal_coordinate_query_mask"].sum().item() == 12
    assert set(batch["internal_coordinate_types"][0]) >= {"protein_phi", "protein_psi", "protein_omega"}
    assert batch["coordinate_mask"].sum().item() == 0


def test_uma_coordinate_head_proxy_loss_backprops_through_coordinate_readout():
    ex = graphify_multimodal(
        {
            "task": "structure_dynamics_proxy",
            "protein_sequence": "MKT",
            "temperature": 340.0,
            "oracle": {"name": "uma"},
        },
        10,
        "local_multimodal_graph_to_graph",
    )
    vocab = build_vocab([ex], extra_tokens=multimodal_reference_tokens())
    collator = RandomOrderCollator(
        vocab,
        max_source_tokens=48,
        max_target_tokens=32,
        max_seq_len=128,
        max_uma_coordinate_atoms=8,
        order_mode="first",
    )
    batch = collator([ex])
    cfg = RandomOrderTokenGTConfig(
        vocab_size=len(vocab.token_to_id),
        hidden_dim=48,
        num_layers=1,
        num_heads=4,
        ffn_dim=96,
        max_seq_len=128,
        max_nodes=128,
        max_slots=64,
        endpoint_dim=16,
        identifier_dim=16,
        coordinate_head_enabled=True,
        coordinate_target_scale=10.0,
    )
    model = RandomOrderTokenGT(cfg)
    out = model(
        input_ids=batch["input_ids"],
        kind_ids=batch["kind_ids"],
        slot_ids=batch["slot_ids"],
        endpoint_ids=batch["endpoint_ids"],
        identifier_ids=batch["identifier_ids"],
        source_numeric_features=batch["source_numeric_features"],
        attention_mask=batch["attention_mask"],
        causal_mask=batch["causal_mask"],
        labels=batch["labels"],
        coordinate_targets=batch["coordinate_targets"],
        coordinate_mask=batch["coordinate_mask"],
    )
    loss, metrics = uma_coordinate_head_oracle_loss(
        out["coordinate_mean"],
        batch["uma_coordinate_query_mask"],
        batch["uma_coordinate_symbols"],
        batch["examples"],
        backend="proxy",
        max_examples=1,
        max_atoms=8,
        dynamics_steps=2,
        force_step_size=0.05,
    )
    assert torch.isfinite(loss)
    assert metrics["uma_coordinate/oracle_examples"] == 1.0
    assert metrics["uma_coordinate/dynamics_steps"] == 2.0
    assert metrics["uma_coordinate/force_rms_ev_per_a"] > 0.0
    loss.backward()
    grad_norm = sum(
        param.grad.detach().abs().sum().item()
        for name, param in model.named_parameters()
        if name.startswith("coordinate_head") and param.grad is not None
    )
    assert grad_norm > 0.0


def test_uma_internal_coordinate_head_proxy_loss_backprops_through_action_readout():
    ex = graphify_multimodal(
        {
            "task": "structure_dynamics_proxy",
            "protein_sequence": "MKT",
            "temperature": 340.0,
            "oracle": {"name": "uma"},
        },
        20,
        "local_multimodal_graph_to_graph",
    )
    vocab = build_vocab([ex], extra_tokens=multimodal_reference_tokens())
    collator = RandomOrderCollator(
        vocab,
        max_source_tokens=48,
        max_target_tokens=32,
        max_seq_len=160,
        max_internal_coordinate_actions=12,
        order_mode="first",
    )
    batch = collator([ex])
    cfg = RandomOrderTokenGTConfig(
        vocab_size=len(vocab.token_to_id),
        hidden_dim=48,
        num_layers=1,
        num_heads=4,
        ffn_dim=96,
        max_seq_len=160,
        max_nodes=128,
        max_slots=64,
        endpoint_dim=16,
        identifier_dim=16,
        internal_coordinate_head_enabled=True,
    )
    model = RandomOrderTokenGT(cfg)
    out = model(
        input_ids=batch["input_ids"],
        kind_ids=batch["kind_ids"],
        slot_ids=batch["slot_ids"],
        endpoint_ids=batch["endpoint_ids"],
        identifier_ids=batch["identifier_ids"],
        source_numeric_features=batch["source_numeric_features"],
        attention_mask=batch["attention_mask"],
        causal_mask=batch["causal_mask"],
        labels=batch["labels"],
        coordinate_targets=batch["coordinate_targets"],
        coordinate_mask=batch["coordinate_mask"],
    )
    loss, metrics = uma_internal_coordinate_head_oracle_loss(
        out["internal_coordinate_mean"],
        batch["internal_coordinate_query_mask"],
        batch["internal_coordinate_type_ids"],
        batch["internal_coordinate_residue_indices"],
        batch["examples"],
        backend="proxy",
        max_examples=1,
        max_residues=3,
        max_atoms=8,
        dynamics_steps=2,
        force_step_size=0.05,
    )
    assert torch.isfinite(loss)
    assert metrics["uma_internal/oracle_examples"] == 1.0
    assert metrics["uma_internal/dynamics_steps"] == 2.0
    assert metrics["uma_internal/force_rms_ev_per_a"] > 0.0
    loss.backward()
    grad_norm = sum(
        param.grad.detach().abs().sum().item()
        for name, param in model.named_parameters()
        if name.startswith("internal_coordinate_head") and param.grad is not None
    )
    assert grad_norm > 0.0


def test_multimodal_random_orders_include_scientific_and_oracle_priorities():
    tokens = [
        "ANSWER:function",
        "COORD:xbin:0",
        "BOND:double",
        "UGM:graph_to_graph",
        "AA:W",
        "PDB:MODEL",
        "UGM:oracle:uma_feedback",
        "TEMP:300K",
        "ENERGY:bin:low",
        "ATTN_BIN:sequence_to_motion:b48",
        "TOKEN_MOTION:uma:refine:b48",
        "UMA_INFLUENCE:uma:trajectory_physics:b48",
        "UMA_TRAJ_BIN:refine:b48",
    ]
    scientific = scientific_graph_order(tokens)
    oracle = oracle_enabling_order(tokens)
    assert scientific.index(tokens.index("UGM:graph_to_graph")) < scientific.index(tokens.index("AA:W"))
    assert scientific.index(tokens.index("BOND:double")) < scientific.index(tokens.index("COORD:xbin:0"))
    assert oracle.index(tokens.index("BOND:double")) < oracle.index(tokens.index("ENERGY:bin:low"))
    assert oracle.index(tokens.index("ATTN_BIN:sequence_to_motion:b48")) < oracle.index(tokens.index("ENERGY:bin:low"))
    orders = build_orders(tokens, seed=3)
    assert scientific in orders
    assert oracle in orders


def test_multimodal_pdb_renderer_and_oracle_reward(monkeypatch, tmp_path):
    monkeypatch.setenv("UGM_UMA_BACKEND", "proxy")
    atoms = [{"element": "C", "name": "C1"}, {"element": "O", "name": "O1"}]
    frames = [[[0.0, 0.0, 0.0], [1.25, 0.0, 0.0]]]
    pdb = records_to_multimodel_pdb(atoms, frames, [{"src": 0, "dst": 1}])
    assert "MODEL" in pdb
    assert "ATOM" in pdb
    assert "CONECT" in pdb
    assert pdb.endswith("END\n")
    xyz = records_to_xyz_trajectory(atoms, frames)
    assert xyz.splitlines()[0] == "2"
    assert "frame=0" in xyz

    dcd_path = tmp_path / "ugm_test_structure_dynamics.dcd"
    written = write_mdtraj_trajectory(dcd_path, atoms, frames, [{"src": 0, "dst": 1}])
    assert written.exists()
    assert written.stat().st_size > 0

    ex = graphify_multimodal(_row(), 0, "local_multimodal_graph_to_graph", molecular_input_policy=ALLOW_STRUCTURE)
    predicted = [
        tok
        for tok in ex.target_tokens
        if tok.startswith(
            (
                "BOND:",
                "COORD:",
                "DIST:",
                "ENERGY:",
                "FORCE:",
                "PDB:",
                "TEMP:",
                "UGM:",
                "ATTN_BIN:",
                "ATTN_COARSE:",
                "TOKEN_COUPLING:",
                "UMA_INFLUENCE:",
                "TOKEN_MOTION:",
                "UMA_TRAJ_BIN:",
                "SEQ_STRUCT_DYN_PROXY:",
            )
        )
    ]
    assert multimodal_oracle_reward(ex, predicted) >= 0.45
    result = verify_example_tokens(ex, predicted)
    assert result.reward > 0.5


def test_predict_uma_coordinate_frame_for_structure_export():
    ex = graphify_multimodal(
        {
            "task": "structure_dynamics_proxy",
            "protein_sequence": "MKT",
            "temperature": 325.0,
            "oracle": {"name": "uma"},
        },
        19,
        "local_multimodal_graph_to_graph",
    )
    vocab = build_vocab([ex], extra_tokens=multimodal_reference_tokens())
    model = RandomOrderTokenGT(
        RandomOrderTokenGTConfig(
            vocab_size=len(vocab.token_to_id),
            hidden_dim=32,
            num_layers=1,
            num_heads=4,
            ffn_dim=64,
            max_seq_len=192,
            max_nodes=192,
            max_slots=64,
            coordinate_head_enabled=True,
        )
    )
    pred = predict_uma_coordinate_frame(
        model,
        vocab,
        ex,
        torch.device("cpu"),
        target_tokens=ex.target_tokens[:8],
        max_source_tokens=96,
        max_target_tokens=16,
        max_uma_coordinate_atoms=6,
    )
    assert len(pred["atoms"]) == 6
    assert len(pred["coordinates"]) == 6
    assert all(len(coord) == 3 for coord in pred["coordinates"])


def test_uma_proxy_reward_is_temperature_conditioned(monkeypatch):
    monkeypatch.setenv("UGM_UMA_BACKEND", "proxy")
    cool = graphify_multimodal({**_row(), "temperature": 300}, 0, "local_multimodal_graph_to_graph")
    hot = graphify_multimodal({**_row(), "temperature": 400}, 1, "local_multimodal_graph_to_graph")
    partial = ["UGM:graph_to_graph"]
    assert multimodal_oracle_reward(hot, partial) > multimodal_oracle_reward(cool, partial)

    diverse = [
        "UGM:graph_to_graph",
        "UGM:oracle:uma_feedback",
        "TEMP:400K",
        "SEQ_STRUCT_DYN_PROXY:uma_scored",
        "TOKEN_MOTION:uma:diversify:b54",
        "TOKEN_MOTION:uma:explore:b50",
        "TOKEN_MOTION:uma:expand:b50",
        "UMA_TRAJ_BIN:diversify:b54",
        "UMA_TRAJ_BIN:explore:b50",
        "UMA_TRAJ_BIN:expand:b50",
        "UMA_INFLUENCE:uma:diversity_pressure:b54",
    ]
    stable = [
        "UGM:graph_to_graph",
        "UGM:oracle:uma_feedback",
        "TEMP:300K",
        "SEQ_STRUCT_DYN_PROXY:uma_scored",
        "TOKEN_MOTION:uma:stabilize:b56",
        "TOKEN_MOTION:uma:refine:b52",
        "TOKEN_MOTION:uma:contract:b45",
        "UMA_TRAJ_BIN:stabilize:b56",
        "UMA_TRAJ_BIN:refine:b52",
        "UMA_TRAJ_BIN:contract:b45",
        "UMA_INFLUENCE:uma:score_sharpness:b60",
    ]
    assert multimodal_oracle_reward(hot, diverse) > multimodal_oracle_reward(hot, stable)
    assert multimodal_oracle_reward(cool, stable) > multimodal_oracle_reward(cool, diverse)


def test_motif_source_parsers_and_structure_derived_windows(tmp_path):
    prosite = tmp_path / "prosite.dat"
    prosite.write_text(
        "ID   ASN_GLYCOSYLATION; PATTERN.\n"
        "AC   PS00001;\n"
        "DE   N-glycosylation site.\n"
        "//\n",
        encoding="utf-8",
    )
    interpro = tmp_path / "interpro_entries.json"
    interpro.write_text(
        '{"results":[{"metadata":{"accession":"IPR000001","name":"Kringle","type":"Domain"}}]}',
        encoding="utf-8",
    )
    cath = tmp_path / "cath-names.txt"
    cath.write_text("1.10.8.10 Alpha hairpin description\n", encoding="utf-8")
    rfam = tmp_path / "rfam-family.txt"
    rfam.write_text("RF00001\t-\t-\t5S_rRNA\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t5S ribosomal RNA\n", encoding="utf-8")

    records = []
    records.extend(parse_prosite_dat(prosite))
    records.extend(parse_interpro_json(interpro))
    records.extend(parse_cath_names(cath))
    records.extend(parse_rfam_family(rfam))
    tokens, all_records = build_motif_vocabulary([prosite, interpro, cath, rfam])
    assert any(record.accession == "PS00001" for record in records)
    assert "SEQ_MOTIF:prosite:PS00001" in tokens
    assert "SEQ_MOTIF:interpro:IPR000001" in tokens
    assert "STRUCT_MOTIF:cath:1.10.8.10" in tokens
    assert "STRUCT_DERIVED_SEQ_MOTIF:cath:1.10.8.10" in tokens
    assert "SEQ_MOTIF_FROM_STRUCTURE:cath:1.10.8.10" in tokens
    assert "SEQ_MOTIF:rfam:RF00001" in tokens
    assert len(all_records) >= len(records)

    derived = derive_structure_sequence_motifs_from_atoms(_row())
    assert any(record.kind == "structure_derived_sequence" for record in derived)
