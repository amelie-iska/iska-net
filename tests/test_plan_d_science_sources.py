from __future__ import annotations

import json

from iska_reasoner.data.graphify import graphify_local_audio, graphify_unigenx
from iska_reasoner.data.phase_policy import ALLOW_STRUCTURE, graph_structure_violations
from iska_reasoner.tools import local_audio_metrics_for_example, science_metrics_for_example


def test_local_audio_graphification_keeps_metadata_and_taxonomy():
    row = {
        "file_name": "XC-example.flac",
        "metadata": json.dumps(
            {
                "phylum": "Chordata",
                "class": "Aves",
                "order": "Passeriformes",
                "family": "Sturnidae",
                "genus": "Onychognathus",
                "species": "Onychognathus tristramii",
                "duration": 12.0,
            }
        ),
        "source_dataset": "Xeno-canto",
        "id": "sample",
        "license": "CC BY-NC",
        "instruction_text": "What is the focal species?",
        "output": "Onychognathus tristramii",
        "task": "species-sci-detection-hard",
    }
    ex = graphify_local_audio(row, 0, "local_audio_reasoning")
    assert ex.task == "local_audio_reasoning"
    assert any(node.type == "audio_placeholder" for node in ex.nodes)
    assert any(node.type == "taxonomy_species" for node in ex.nodes)
    assert any(tok.startswith("AUDIO:task:") for tok in ex.target_tokens)
    metrics = local_audio_metrics_for_example(ex)
    assert metrics["audio/taxonomy_node_count_mean"] >= 6
    assert metrics["audio/noncommercial_license_rate"] == 1.0


def test_unigenx_qm9_graphification_defaults_to_sequence_only():
    row = {
        "smiles": "CO",
        "atomic_symbols": ["C", "O"],
        "pos": [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]],
        "gap": 0.123456,
        "mu": 1.2,
    }
    ex = graphify_unigenx(row, 0, "unigenx_qm9_train")
    assert ex.task == "unigenx_molecule_reasoning"
    assert not any(node.type == "coordinate_3d" for node in ex.nodes)
    assert not any(node.type == "molecule_property" for node in ex.nodes)
    assert not graph_structure_violations(ex)
    assert "UNIGENX:domain:molecule" in ex.target_tokens
    metrics = science_metrics_for_example(ex)
    assert metrics["science/molecule_smiles_present_rate"] == 1.0


def test_unigenx_qm9_structure_records_require_explicit_policy():
    row = {
        "smiles": "CO",
        "atomic_symbols": ["C", "O"],
        "pos": [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]],
        "gap": 0.123456,
        "mu": 1.2,
    }
    ex = graphify_unigenx(row, 0, "unigenx_qm9_train", molecular_input_policy=ALLOW_STRUCTURE)
    assert any(node.type == "coordinate_3d" for node in ex.nodes)
    assert any(node.type == "molecule_property" for node in ex.nodes)
    metrics = science_metrics_for_example(ex)
    assert metrics["science/coordinate_node_count_mean"] == 2.0


def test_unigenx_material_graphification_has_formula_nodes():
    row = {
        "prompt": "Classify crystal systems.",
        "completion": "['Cubic']",
        "formula": "['NaCl']",
        "mpid": "['mp-22862']",
    }
    ex = graphify_unigenx(row, 0, "unigenx_materials_crystal_system")
    assert ex.task == "unigenx_material_reasoning"
    assert any(node.type == "material_formula" for node in ex.nodes)
    assert "UNIGENX:domain:material" in ex.target_tokens
    metrics = science_metrics_for_example(ex)
    assert metrics["science/material_formula_count_mean"] == 1.0
