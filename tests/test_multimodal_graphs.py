from __future__ import annotations

from iska_reasoner.data.dataset import RandomOrderCollator, extract_numeric_values
from iska_reasoner.data.graphify import graphify_rows
from iska_reasoner.data.multimodal import (
    BOND_TYPES,
    graphify_multimodal,
    multimodal_reference_tokens,
    records_to_multimodel_pdb,
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
from iska_reasoner.tools import multimodal_metrics_for_example, multimodal_oracle_reward, verify_example_tokens


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
    collator = RandomOrderCollator(vocab, max_source_tokens=128, max_target_tokens=32, max_numeric_targets=8)
    batch = collator([ex])
    assert batch["input_ids"].shape[0] == 1
    assert batch["numeric_mask"].sum().item() >= 3
    assert batch["source_numeric_features"].sum().item() > 0


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


def test_multimodal_pdb_renderer_and_oracle_reward(monkeypatch):
    monkeypatch.setenv("UGM_UMA_BACKEND", "proxy")
    atoms = [{"element": "C", "name": "C1"}, {"element": "O", "name": "O1"}]
    frames = [[[0.0, 0.0, 0.0], [1.25, 0.0, 0.0]]]
    pdb = records_to_multimodel_pdb(atoms, frames, [{"src": 0, "dst": 1}])
    assert "MODEL" in pdb
    assert "ATOM" in pdb
    assert "CONECT" in pdb
    assert pdb.endswith("END\n")

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
