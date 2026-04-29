from __future__ import annotations

import torch

from iska_reasoner.data.graphify import (
    graphify_code,
    graphify_graph_reasoning,
    graphify_lean,
    graphify_molecule,
    graphify_nucleotide_sequence,
)
from iska_reasoner.graph.schema import GraphExample, Node
from iska_reasoner.models.random_order_tokengt import RandomOrderTokenGT, RandomOrderTokenGTConfig
from iska_reasoner.tools import (
    chem_metrics_for_example,
    code_metrics_for_example,
    lean_metrics_for_example,
    run_python_tests,
)


def test_code_graphification_and_python_tests():
    row = {
        "prompt": "Write add.",
        "canonical_solution": "def add(a, b):\n    return a + b\n",
        "entry_point": "add",
        "test": "from solution import add\n\ndef test_add():\n    assert add(2, 3) == 5\n",
    }
    ex = graphify_code(row, 0, "bigcodebench_smoke")
    assert any(node.type == "test" for node in ex.nodes)
    assert ex.metadata["entry_point"] == "add"
    result = run_python_tests(ex.metadata["canonical_solution"], ex.metadata["tests"])
    assert result.attempted and result.passed
    metrics = code_metrics_for_example(ex, [])
    assert metrics["code/has_tests_rate"] == 1.0
    assert metrics["code/pass_rate"] == 1.0


def test_lean_adapter_gracefully_handles_environment():
    ex = graphify_lean({"formal_statement": "theorem t : True := by trivial"}, 0, "lean_smoke")
    metrics = lean_metrics_for_example(ex)
    assert "lean/available" in metrics
    assert "lean/compile_attempt_rate" in metrics


def test_molecule_graphification_has_chem_metrics():
    ex = graphify_molecule({"smiles": "CCO", "label": "toy"}, 0, "chem_smoke")
    assert any(node.type in {"smiles", "selfies"} for node in ex.nodes)
    assert not any(node.type in {"atom", "atom_symbol", "coordinate"} for node in ex.nodes)
    metrics = chem_metrics_for_example(ex, [])
    assert "chem/rdkit_available" in metrics
    assert metrics["chem/smiles_present_rate"] == 1.0


def test_graphwalks_graphification_parses_edges_and_answer_nodes():
    row = {
        "prompt": (
            "Find the BFS frontier.\n"
            "Here is the graph to operate on:\n"
            "A -> B\nB -> C\nA -> D\n"
        ),
        "answer_nodes": ["B", "D"],
        "problem_type": "bfs_frontier",
    }
    ex = graphify_graph_reasoning(row, 0, "openai_graphwalks_train")
    assert ex.task == "graph_reasoning"
    assert any(edge.type == "directed_graph_edge" for edge in ex.edges)
    assert "ANSWER_NODE:B" in ex.target_tokens
    assert "ANSWER_NODE:D" in ex.target_tokens
    assert ex.metadata["parsed_edges"] == 3


def test_rfam_graphification_is_sequence_only_rna():
    ex = graphify_nucleotide_sequence(
        {"sequence": "AUGCUU", "family": "RF00001", "clan": "CL001", "description": "toy RNA"},
        0,
        "rfam_sequence_train",
    )
    assert ex.task == "rna_sequence_modeling"
    assert any(node.type == "rna_base" for node in ex.nodes)
    assert not any(node.type in {"atom", "coordinate", "coordinate_3d"} for node in ex.nodes)
    assert "RNA_FAMILY:RF00001" in ex.target_tokens


def test_dna_coding_graphification_keeps_regions_and_translation():
    ex = graphify_nucleotide_sequence(
        {
            "sequence": "ATGAAATGA",
            "accession": "toy",
            "exons": [{"start": 0, "end": 9, "gene": "x"}],
            "proteins": [{"sequence": "MK"}],
        },
        0,
        "dna_coding_regions_train",
    )
    assert ex.task == "dna_sequence_modeling"
    assert any(node.type == "dna_base" for node in ex.nodes)
    assert any(node.type == "dna_exon" for node in ex.nodes)
    assert any(node.type == "translated_protein_sequence" for node in ex.nodes)
    assert "DNA:translated_protein" in ex.target_tokens


def test_lora_and_checkpointing_model_forward():
    cfg = RandomOrderTokenGTConfig(
        vocab_size=32,
        hidden_dim=32,
        num_layers=1,
        num_heads=4,
        ffn_dim=64,
        max_seq_len=16,
        endpoint_dim=8,
        gradient_checkpointing=True,
        lora_rank=2,
        freeze_base_for_lora=True,
    )
    model = RandomOrderTokenGT(cfg)
    trainable = [name for name, param in model.named_parameters() if param.requires_grad]
    assert trainable and all("lora_" in name for name in trainable)
    batch = {
        "input_ids": torch.zeros(2, 8, dtype=torch.long),
        "kind_ids": torch.zeros(2, 8, dtype=torch.long),
        "slot_ids": torch.zeros(2, 8, dtype=torch.long),
        "endpoint_ids": torch.zeros(2, 8, 2, dtype=torch.long),
        "attention_mask": torch.ones(2, 8, dtype=torch.bool),
        "causal_mask": torch.zeros(8, 8, dtype=torch.bool),
        "labels": torch.full((2, 8), -100, dtype=torch.long),
    }
    batch["labels"][:, -1] = 1
    model.train()
    out = model(**batch)
    assert out["loss"].isfinite()
    out["loss"].backward()
    assert any(param.grad is not None for param in model.parameters() if param.requires_grad)
