from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

from tqdm.auto import tqdm

from iska_reasoner.data.audio import extract_audio_features
from iska_reasoner.data.hebrew import hebrew_text_graph
from iska_reasoner.data.motifs import normalize_fragment
from iska_reasoner.data.multimodal import graphify_multimodal
from iska_reasoner.data.phase_policy import ALLOW_STRUCTURE, SEQUENCE_ONLY, sanitize_row_for_phase, structure_fields_present
from iska_reasoner.data.synthetic import iter_synthetic_examples
from iska_reasoner.graph.orders import build_orders
from iska_reasoner.graph.schema import Edge, GraphExample, Node
from iska_reasoner.utils.io import read_jsonl, write_jsonl


def _stable_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def _word_nodes(text: str, max_words: int = 64) -> list[Node]:
    words = text.replace("\n", " ").split()[:max_words]
    return [Node(id=f"w{i}", type="token", value=word) for i, word in enumerate(words)]


def _chain_edges(nodes: list[Node]) -> list[Edge]:
    return [Edge(src=nodes[i].id, dst=nodes[i + 1].id, type="next_token") for i in range(len(nodes) - 1)]


def _text_token_nodes(parent: str, text: str, prefix: str, node_type: str = "token", max_words: int = 96) -> tuple[list[Node], list[Edge]]:
    words = re.findall(r"[A-Za-z0-9_./:+#=@%,'\\-]+", text.replace("\n", " "))[:max_words]
    nodes: list[Node] = []
    edges: list[Edge] = []
    prev: str | None = None
    for i, word in enumerate(words):
        node_id = f"{prefix}_tok{i}"
        nodes.append(Node(id=node_id, type=node_type, value=word, features={"index": i}))
        edges.append(Edge(src=parent, dst=node_id, type="contains_token"))
        if prev is not None:
            edges.append(Edge(src=prev, dst=node_id, type="next_token"))
        prev = node_id
    return nodes, edges


def _as_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)] if str(value).strip() else []


def _motif_items(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        return [item for item in re.split(r"[,;]\s*", value) if item]
    return [value]


def _motif_accession_source(value: Any, default_source: str) -> tuple[str, str, str]:
    if isinstance(value, dict):
        accession = value.get("accession") or value.get("id") or value.get("name") or value.get("motif") or "unknown"
        name = value.get("name") or value.get("description") or accession
        source = value.get("source") or default_source
    else:
        accession = value
        name = value
        source = default_source
    return normalize_fragment(accession), normalize_fragment(source), str(name)


def smiles_to_selfies(smiles: str) -> str:
    if not smiles:
        return ""
    try:
        import selfies as sf

        encoded = sf.encoder(smiles)
        return encoded or ""
    except Exception:
        return ""


def _selfies_sequence_nodes(selfies: str, prefix: str = "selfies", max_len: int = 160) -> tuple[list[Node], list[Edge], list[str]]:
    if not selfies:
        return [], [], []
    tokens = re.findall(r"\[[^\[\]]+\]", selfies)
    if not tokens:
        tokens = [ch for ch in selfies.strip() if not ch.isspace()]
    tokens = tokens[:max_len]
    nodes = [Node(id=prefix, type="selfies", value="".join(tokens), features={"length": len(tokens)})]
    edges: list[Edge] = []
    target_tokens = ["UGM:modality:selfies"]
    prev = None
    for i, token in enumerate(tokens):
        node_id = f"{prefix}_tok{i}"
        nodes.append(Node(id=node_id, type="selfies_token", value=token, features={"index": i}))
        edges.append(Edge(src=prefix, dst=node_id, type="contains_selfies_token"))
        if prev is not None:
            edges.append(Edge(src=prev, dst=node_id, type="sequence_next"))
        prev = node_id
        if i < 96:
            target_tokens.append(f"SELFIES:{token}")
    return nodes, edges, target_tokens


def _optional_molecular_descriptor_nodes(smiles: str, enabled: bool) -> tuple[list[Node], list[Edge], list[str]]:
    """Add non-coordinate 2D chemical descriptors when explicitly enabled."""
    if not enabled or not smiles:
        return [], [], []
    try:
        from rdkit import Chem
        from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
    except Exception:
        return [], [], []
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [], [], []
    descriptors = {
        "atom_count": float(mol.GetNumAtoms()),
        "bond_count": float(mol.GetNumBonds()),
        "ring_count": float(rdMolDescriptors.CalcNumRings(mol)),
        "mol_wt": float(Descriptors.MolWt(mol)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
        "logp": float(Crippen.MolLogP(mol)),
        "rotatable_bonds": float(Lipinski.NumRotatableBonds(mol)),
    }
    nodes: list[Node] = []
    edges: list[Edge] = []
    tokens: list[str] = []
    for i, (name, value) in enumerate(descriptors.items()):
        node_id = f"geom_feature_{i}"
        rounded = round(value, 4)
        nodes.append(Node(id=node_id, type="molecular_geometry_feature", value=f"{name}:{rounded}", features={"name": name, "value": rounded}))
        edges.append(Edge(src="mol", dst=node_id, type="has_geometric_feature"))
        tokens.append(f"GEOM_FEATURE:{name}:{round(value, 2)}")
    return nodes, edges, tokens


def graphify_math(row: dict[str, Any], idx: int, dataset_name: str) -> GraphExample:
    problem = str(row.get("question") or row.get("problem") or row.get("input") or "")
    solution = str(row.get("answer") or row.get("generated_solution") or row.get("solution") or "")
    expected = str(row.get("expected_answer") or row.get("answer") or "").strip()
    nodes = [Node(id="problem", type="prompt", value=problem[:2048]), Node(id="solution", type="solution", value=solution[:2048])]
    edges = [Edge(src="problem", dst="solution", type="has_solution")]
    problem_nodes, problem_edges = _text_token_nodes("problem", problem, "problem", "math_problem_token", max_words=128)
    solution_nodes, solution_edges = _text_token_nodes("solution", solution, "solution", "math_solution_token", max_words=160)
    nodes.extend(problem_nodes)
    nodes.extend(solution_nodes)
    edges.extend(problem_edges)
    edges.extend(solution_edges)
    if expected:
        nodes.append(Node(id="expected", type="answer", value=expected))
        edges.append(Edge(src="solution", dst="expected", type="supports"))
    target_tokens = ["CLAIM:math_problem", f"ANSWER:{expected or solution[:80]}"]
    ex = GraphExample(
        id=f"{dataset_name}_{idx}_{_stable_id(problem)}",
        task="math_reasoning",
        nodes=nodes,
        edges=edges,
        target_tokens=target_tokens,
        metadata={"source_dataset": dataset_name},
    )
    ex.decoder_orders = build_orders(ex.target_tokens, seed=idx)
    return ex


def _graph_tail(text: str) -> str:
    markers = ["Here is the graph to operate on:", "Q:", "The graph has the following edges:"]
    tail = text
    for marker in markers:
        if marker in tail:
            tail = tail.split(marker, 1)[-1]
            break
    return tail


def _parse_directed_edges(text: str, max_edges: int = 512) -> list[tuple[str, str]]:
    tail = _graph_tail(text)
    edges = [(src, dst) for src, dst in re.findall(r"\b([A-Za-z0-9_.:-]+)\s*->\s*([A-Za-z0-9_.:-]+)\b", tail)]
    return edges[:max_edges]


def _parse_tuple_edges(text: str, max_edges: int = 512) -> list[tuple[str, str]]:
    edges = [(src, dst) for src, dst in re.findall(r"\((\d+)\s*,\s*(\d+)\)", text)]
    return edges[:max_edges]


def graphify_graph_reasoning(row: dict[str, Any], idx: int, dataset_name: str) -> GraphExample:
    prompt = str(row.get("prompt") or row.get("query") or row.get("question") or row.get("input") or "")
    answer_value = row.get("answer_nodes") if row.get("answer_nodes") is not None else row.get("answer")
    task_name = str(row.get("problem_type") or row.get("task") or "graph_reasoning")
    directed_edges = _parse_directed_edges(prompt)
    tuple_edges = [] if directed_edges else _parse_tuple_edges(prompt)
    graph_edges = directed_edges or tuple_edges
    directed = bool(directed_edges)

    nodes: list[Node] = [
        Node(id="task", type="graph_reasoning_task", value=task_name),
        Node(id="prompt", type="prompt", value=prompt[:4096], features={"prompt_chars": len(prompt)}),
    ]
    edges: list[Edge] = [Edge(src="task", dst="prompt", type="has_prompt")]
    prompt_nodes, prompt_edges = _text_token_nodes("prompt", prompt, "graph_prompt", "graph_prompt_token", max_words=128)
    nodes.extend(prompt_nodes)
    edges.extend(prompt_edges)

    node_ids: dict[str, str] = {}
    for src, dst in graph_edges:
        for label in (src, dst):
            if label not in node_ids:
                node_id = f"g{len(node_ids)}"
                node_ids[label] = node_id
                nodes.append(Node(id=node_id, type="graph_node", value=label))
                edges.append(Edge(src="task", dst=node_id, type="contains_graph_node"))
        edges.append(
            Edge(
                src=node_ids[src],
                dst=node_ids[dst],
                type="directed_graph_edge" if directed else "undirected_graph_edge",
            )
        )
        if not directed:
            edges.append(Edge(src=node_ids[dst], dst=node_ids[src], type="undirected_graph_edge"))

    target_tokens = ["GRAPH:reasoning", f"GRAPH_TASK:{normalize_fragment(task_name)}"]
    if isinstance(answer_value, list):
        for node in answer_value[:64]:
            target_tokens.append(f"ANSWER_NODE:{normalize_fragment(node)}")
        if not answer_value:
            target_tokens.append("ANSWER_NODE:empty")
    else:
        answer_text = str(answer_value or "")
        normalized_answer = "yes" if "### Yes" in answer_text or answer_text.strip().lower().endswith("yes.") else None
        normalized_answer = "no" if "### No" in answer_text or answer_text.strip().lower().endswith("no.") else normalized_answer
        if normalized_answer:
            target_tokens.append(f"ANSWER:{normalized_answer}")
        elif answer_text:
            target_tokens.append(f"ANSWER:{answer_text[:96]}")
    target_tokens.append(f"GRAPH:edges:{len(graph_edges)}")

    ex = GraphExample(
        id=f"{dataset_name}_{idx}_{_stable_id(prompt + str(answer_value))}",
        task="graph_reasoning",
        nodes=nodes,
        edges=edges,
        target_tokens=target_tokens,
        metadata={
            "source_dataset": dataset_name,
            "problem_type": task_name,
            "parsed_edges": len(graph_edges),
            "directed": directed,
            "answer": answer_value,
        },
    )
    ex.decoder_orders = build_orders(ex.target_tokens, seed=idx)
    return ex


def graphify_code(row: dict[str, Any], idx: int, dataset_name: str) -> GraphExample:
    prompt = str(row.get("instruct_prompt") or row.get("complete_prompt") or row.get("description") or row.get("prompt") or "")
    canonical = str(row.get("canonical_solution") or row.get("solution") or row.get("code") or "")
    tests = (
        _as_text_list(row.get("test"))
        + _as_text_list(row.get("tests"))
        + _as_text_list(row.get("unit_tests"))
        + _as_text_list(row.get("private_tests"))
    )
    entry_point = str(row.get("entry_point") or row.get("function_name") or "")
    nodes = [Node(id="prompt", type="prompt", value=prompt[:1024])]
    edges: list[Edge] = []
    try:
        tree = ast.parse(canonical or "pass")
        funcs = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        imports = []
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                imports.extend(alias.name for alias in n.names)
            elif isinstance(n, ast.ImportFrom) and n.module:
                imports.append(n.module)
    except SyntaxError:
        funcs = []
        imports = []
    if canonical:
        nodes.append(Node(id="solution", type="canonical_solution", value=canonical[:2048]))
        edges.append(Edge(src="prompt", dst="solution", type="has_candidate_solution"))
    if entry_point:
        nodes.append(Node(id="entry", type="entry_point", value=entry_point))
        edges.append(Edge(src="prompt", dst="entry", type="requires_entry_point"))
    for i, fn in enumerate(funcs[:16]):
        nodes.append(Node(id=f"fn{i}", type="function", value=fn))
        edges.append(Edge(src="prompt", dst=f"fn{i}", type="requires"))
        if entry_point and fn == entry_point:
            edges.append(Edge(src="entry", dst=f"fn{i}", type="names_function"))
    for i, module in enumerate(sorted(set(imports))[:16]):
        node_id = f"import{i}"
        nodes.append(Node(id=node_id, type="import", value=module))
        edges.append(Edge(src="solution" if canonical else "prompt", dst=node_id, type="imports"))
    for i, test in enumerate(tests[:8]):
        node_id = f"test{i}"
        nodes.append(Node(id=node_id, type="test", value=test[:2048]))
        edges.append(Edge(src="prompt", dst=node_id, type="validated_by"))
    target_tokens = [f"CODE:function:{fn}" for fn in funcs[:4]] or ["CODE:solution"]
    if entry_point:
        target_tokens.append(f"CODE:entry:{entry_point}")
    target_tokens.append("CLAIM:tests_required")
    ex = GraphExample(
        id=f"{dataset_name}_{idx}_{_stable_id(prompt)}",
        task="code_reasoning",
        nodes=nodes,
        edges=edges,
        target_tokens=target_tokens,
        metadata={
            "source_dataset": dataset_name,
            "canonical_solution": canonical,
            "tests": tests[:16],
            "entry_point": entry_point,
            "imports": sorted(set(imports))[:32],
        },
    )
    ex.decoder_orders = build_orders(ex.target_tokens, seed=idx)
    return ex


def graphify_lean(row: dict[str, Any], idx: int, dataset_name: str) -> GraphExample:
    nl = str(row.get("natural_language_statement") or row.get("nl_statement") or row.get("statement") or row.get("informal_statement") or "")
    header = str(row.get("lean4_src_header") or row.get("src_header") or "")
    formal = str(row.get("formal_statement") or row.get("lean4_formalization") or row.get("formal") or row.get("theorem") or "")
    proof = str(row.get("formal_proof") or row.get("lean4_proof") or row.get("proof") or "")
    source = "\n".join(part for part in [header, formal, proof] if part.strip()).strip()
    imports = sorted(set(re.findall(r"(?m)^\s*import\s+([A-Za-z0-9_.'/-]+)", source)))
    nodes = [
        Node(id="nl", type="informal_statement", value=nl[:1024]),
        Node(id="formal", type="lean_statement", value=formal[:1024]),
    ]
    edges = [Edge(src="nl", dst="formal", type="formalizes")]
    if proof:
        nodes.append(Node(id="proof", type="lean_proof", value=proof[:1024]))
        edges.append(Edge(src="formal", dst="proof", type="proved_by"))
    for i, module in enumerate(imports[:16]):
        node_id = f"import{i}"
        nodes.append(Node(id=node_id, type="lean_import", value=module))
        edges.append(Edge(src="formal", dst=node_id, type="imports"))
    target_tokens = ["THEOREM:lean4", "CLAIM:formalization"] + (["ANSWER:has_proof"] if proof else ["ANSWER:statement_only"])
    ex = GraphExample(
        id=f"{dataset_name}_{idx}_{_stable_id(nl + formal)}",
        task="formal_proof",
        nodes=nodes,
        edges=edges,
        target_tokens=target_tokens,
        metadata={"source_dataset": dataset_name, "lean_source": source, "lean_imports": imports},
    )
    ex.decoder_orders = build_orders(ex.target_tokens, seed=idx)
    return ex


def graphify_molecule(
    row: dict[str, Any],
    idx: int,
    dataset_name: str,
    molecular_input_policy: str = SEQUENCE_ONLY,
    geometric_features: bool = False,
) -> GraphExample:
    original_row = dict(row)
    row = sanitize_row_for_phase(row, molecular_input_policy)
    smiles = str(row.get("SMILES") or row.get("smiles") or row.get("Smiles") or "")
    selfies = str(row.get("selfies") or row.get("SELFIES") or "") or smiles_to_selfies(smiles)
    target = str(row.get("label") or row.get("exp") or row.get("target") or row.get("Lipophilicity") or "")
    nodes = [Node(id="mol", type="molecule_sequence", value=selfies or smiles)]
    edges: list[Edge] = []
    rdkit_available = False
    rdkit_valid = False
    atom_count = 0
    bond_count = 0
    if smiles:
        nodes.append(Node(id="smiles", type="smiles", value=smiles))
        edges.append(Edge(src="mol", dst="smiles", type="has_smiles_sequence"))
    seq_nodes, seq_edges, seq_tokens = _selfies_sequence_nodes(selfies)
    nodes.extend(seq_nodes)
    edges.extend(seq_edges)
    if seq_nodes:
        edges.append(Edge(src="mol", dst=seq_nodes[0].id, type="has_selfies_sequence"))
    descriptor_nodes, descriptor_edges, descriptor_tokens = _optional_molecular_descriptor_nodes(smiles, geometric_features)
    nodes.extend(descriptor_nodes)
    edges.extend(descriptor_edges)
    target_tokens = seq_tokens or ["SMILES:sequence_candidate"]
    target_tokens.extend(descriptor_tokens)
    target_tokens.extend([f"CLAIM:property:{target}", "ANSWER:molecule_sequence"])
    ex = GraphExample(
        id=f"{dataset_name}_{idx}_{_stable_id(smiles)}",
        task="molecule_reasoning",
        nodes=nodes,
        edges=edges,
        target_tokens=target_tokens,
        metadata={
            "source_dataset": dataset_name,
            "smiles": smiles,
            "selfies": selfies,
            "target": target,
            "rdkit_available": rdkit_available,
            "rdkit_valid": rdkit_valid,
            "atom_count": atom_count,
            "bond_count": bond_count,
            "molecular_input_policy": molecular_input_policy,
            "geometric_features_enabled": geometric_features,
            "ignored_structure_fields": structure_fields_present(original_row),
            "license_warning": "check upstream metadata before scaling",
        },
    )
    ex.decoder_orders = build_orders(ex.target_tokens, seed=idx)
    return ex


def graphify_pubchem_smiles(row: dict[str, Any], idx: int, dataset_name: str) -> GraphExample:
    """Graphify high-volume PubChem CID-SMILES rows without per-row RDKit parsing."""
    smiles = str(row.get("smiles") or row.get("SMILES") or row.get("CanonicalSMILES") or "")
    selfies = str(row.get("selfies") or row.get("SELFIES") or "") or smiles_to_selfies(smiles)
    cid = str(row.get("cid") or row.get("CID") or row.get("pubchem_cid") or "").strip()
    title = str(row.get("Title") or row.get("title") or row.get("name") or "").strip()
    nodes = [Node(id="mol", type="molecule_sequence", value=selfies or smiles)]
    edges: list[Edge] = []
    if smiles:
        nodes.append(Node(id="smiles", type="smiles", value=smiles))
        edges.append(Edge(src="mol", dst="smiles", type="has_smiles_sequence"))
    seq_nodes, seq_edges, seq_tokens = _selfies_sequence_nodes(selfies)
    nodes.extend(seq_nodes)
    edges.extend(seq_edges)
    if seq_nodes:
        edges.append(Edge(src="mol", dst=seq_nodes[0].id, type="has_selfies_sequence"))
    if cid:
        nodes.append(Node(id="cid", type="pubchem_cid", value=cid))
        edges.append(Edge(src="cid", dst="mol", type="identifies"))
    if title:
        nodes.append(Node(id="title", type="compound_title", value=title[:256]))
        edges.append(Edge(src="title", dst="mol", type="names"))
    target_tokens = ["NATURELM:domain:pubchem", "MOLECULE:sequence_record"]
    target_tokens.extend(seq_tokens)
    if cid:
        target_tokens.append(f"PUBCHEM:CID:{cid}")
    if smiles:
        target_tokens.append(f"SMILES:{smiles[:120]}")
    ex = GraphExample(
        id=f"{dataset_name}_{idx}_{_stable_id(cid + smiles)}",
        task="naturelm_pubchem_reconstruction",
        nodes=nodes,
        edges=edges,
        target_tokens=target_tokens,
        metadata={
            "source_dataset": dataset_name,
            "pubchem_cid": cid,
            "smiles": smiles,
            "selfies": selfies,
            "title": title,
            "rdkit_skipped": True,
            "molecular_input_policy": SEQUENCE_ONLY,
        },
    )
    ex.decoder_orders = build_orders(ex.target_tokens, seed=idx)
    return ex


def graphify_local_audio(row: dict[str, Any], idx: int, dataset_name: str) -> GraphExample:
    instruction = str(row.get("instruction_text") or row.get("instruction") or "")
    output = str(row.get("output") or "")
    task = str(row.get("task") or "local_audio_task")
    file_name = str(row.get("file_name") or "")
    local_audio_path = str(row.get("local_audio_path") or row.get("audio_path") or "")
    source_dataset = str(row.get("source_dataset") or "")
    license_name = str(row.get("license") or "")
    metadata_raw = row.get("metadata") or "{}"
    try:
        metadata = json.loads(metadata_raw) if isinstance(metadata_raw, str) else dict(metadata_raw)
    except Exception:
        metadata = {}

    nodes = [
        Node(id="audio", type="audio_placeholder", value=file_name),
        Node(id="instruction", type="instruction", value=instruction[:1024]),
        Node(id="output", type="answer", value=output[:512]),
        Node(id="task", type="audio_task", value=task),
    ]
    edges = [
        Edge(src="audio", dst="instruction", type="queried_by"),
        Edge(src="instruction", dst="output", type="answered_by"),
        Edge(src="task", dst="instruction", type="specifies_prompt"),
    ]
    if source_dataset:
        nodes.append(Node(id="source", type="source_dataset", value=source_dataset))
        edges.append(Edge(src="source", dst="audio", type="provides"))
    if license_name:
        nodes.append(Node(id="license", type="license", value=license_name))
        edges.append(Edge(src="license", dst="audio", type="licenses"))

    taxonomy_keys = ["phylum", "class", "order", "family", "genus", "species", "subspecies", "common_name"]
    prev_taxon = None
    for key in taxonomy_keys:
        value = str(metadata.get(key) or "").strip()
        if not value:
            continue
        node_id = f"tax_{key}"
        nodes.append(Node(id=node_id, type=f"taxonomy_{key}", value=value))
        edges.append(Edge(src="audio", dst=node_id, type="has_taxonomy"))
        if prev_taxon is not None:
            edges.append(Edge(src=prev_taxon, dst=node_id, type="taxonomic_child"))
        prev_taxon = node_id
    if metadata.get("duration") is not None:
        nodes.append(Node(id="duration", type="audio_duration", value=str(metadata["duration"])))
        edges.append(Edge(src="audio", dst="duration", type="has_duration"))
    if local_audio_path:
        features = extract_audio_features(local_audio_path)
        nodes.append(Node(id="audio_features", type="audio_features", value=features.backend or "unavailable", features=features.to_features()))
        edges.append(Edge(src="audio", dst="audio_features", type="has_audio_features"))
        if features.available:
            nodes.append(Node(id="audio_duration_local", type="audio_duration", value=str(round(features.duration_s, 4)), features={"duration_s": features.duration_s}))
            edges.append(Edge(src="audio_features", dst="audio_duration_local", type="measures_duration"))

    target_tokens = [f"AUDIO:task:{task}", f"ANSWER:{output[:120]}"]
    if metadata.get("family"):
        target_tokens.append(f"TAXON:family:{metadata['family']}")
    if metadata.get("species"):
        target_tokens.append(f"TAXON:species:{metadata['species']}")
    ex = GraphExample(
        id=f"{dataset_name}_{idx}_{_stable_id(str(row.get('id') or file_name or instruction))}",
        task="local_audio_reasoning",
        nodes=nodes,
        edges=edges,
        target_tokens=target_tokens,
        metadata={
            "source_dataset": dataset_name,
            "upstream_source_dataset": source_dataset,
            "license": license_name,
            "task": task,
            "file_name": file_name,
            "local_audio_path": local_audio_path,
            "taxonomy": {key: metadata.get(key) for key in taxonomy_keys if metadata.get(key)},
            "audio_dropped": "audio" not in row,
        },
    )
    ex.decoder_orders = build_orders(ex.target_tokens, seed=idx)
    return ex


def _round_value(value: Any, digits: int = 4) -> str:
    try:
        return str(round(float(value), digits))
    except Exception:
        return str(value)


def _coord_text(coord: Any) -> str:
    if isinstance(coord, (list, tuple)) and len(coord) >= 3:
        return ",".join(_round_value(v, 4) for v in coord[:3])
    return str(coord)


def _sequence_nodes(sequence: str, prefix: str, node_type: str, max_len: int = 512) -> tuple[list[Node], list[Edge]]:
    nodes: list[Node] = []
    edges: list[Edge] = []
    prev = None
    for i, residue in enumerate(sequence[:max_len]):
        node_id = f"{prefix}{i}"
        nodes.append(Node(id=node_id, type=node_type, value=residue, features={"index": i}))
        if prev is not None:
            edges.append(Edge(src=prev, dst=node_id, type="sequence_next"))
        prev = node_id
    return nodes, edges


def graphify_nucleotide_sequence(row: dict[str, Any], idx: int, dataset_name: str, modality: str | None = None) -> GraphExample:
    lname = dataset_name.lower()
    sequence = str(row.get("rna_sequence") or row.get("dna_sequence") or row.get("sequence") or row.get("text") or "")
    clean = re.sub(r"\s+", "", sequence).upper()
    inferred = modality or ("rna" if "rna" in lname or "rfam" in lname or "rnacentral" in lname or ("U" in clean and "T" not in clean) else "dna")
    token_prefix = "RNA" if inferred == "rna" else "DNA"
    root_type = "rna_sequence" if inferred == "rna" else "dna_sequence"
    base_type = "rna_base" if inferred == "rna" else "dna_base"
    nodes: list[Node] = [
        Node(id="domain", type="science_domain", value=inferred),
        Node(id=inferred, type=root_type, value=clean[:512], features={"length": len(clean)}),
    ]
    edges: list[Edge] = [Edge(src="domain", dst=inferred, type="has_sequence")]
    base_nodes, base_edges = _sequence_nodes(clean, "base", base_type, max_len=128)
    nodes.extend(base_nodes)
    edges.extend(base_edges)
    for node in base_nodes:
        edges.append(Edge(src=inferred, dst=node.id, type="contains_base"))

    target_tokens = [f"UGM:modality:{inferred}", f"{token_prefix}:length:{len(clean)}"]
    target_tokens.extend(f"{token_prefix}:{char}" for char in clean[:64] if char)

    annotations = {
        "family": row.get("family"),
        "clan": row.get("clan"),
        "type": row.get("type"),
        "description": row.get("description"),
        "organism": row.get("organism"),
        "accession": row.get("accession") or row.get("id") or row.get("upi"),
    }
    for key, value in annotations.items():
        if value is None or value == "":
            continue
        node_id = f"ann_{key}"
        nodes.append(Node(id=node_id, type=f"{inferred}_{key}", value=str(value)[:512]))
        edges.append(Edge(src=inferred, dst=node_id, type=f"has_{key}"))
        if key in {"family", "clan", "type"}:
            target_tokens.append(f"{token_prefix}_{key.upper()}:{normalize_fragment(value)}")

    for region_key in ("exons", "introns"):
        for region_idx, region in enumerate(_motif_items(row.get(region_key))[:16]):
            if not isinstance(region, dict):
                continue
            node_id = f"{region_key}_{region_idx}"
            nodes.append(
                Node(
                    id=node_id,
                    type=f"{inferred}_{region_key[:-1]}",
                    value=str(region.get("gene") or f"{region_key}:{region_idx}")[:256],
                    features={"start": region.get("start"), "end": region.get("end")},
                )
            )
            edges.append(Edge(src=inferred, dst=node_id, type=f"has_{region_key[:-1]}"))
            target_tokens.append(f"{token_prefix}_REGION:{region_key[:-1]}")

    for protein_idx, protein in enumerate(_motif_items(row.get("proteins"))[:4]):
        if not isinstance(protein, dict):
            continue
        protein_seq = str(protein.get("sequence") or "")
        if not protein_seq:
            continue
        node_id = f"translated_protein_{protein_idx}"
        nodes.append(Node(id=node_id, type="translated_protein_sequence", value=protein_seq[:512], features={"length": len(protein_seq)}))
        edges.append(Edge(src=inferred, dst=node_id, type="translates_to"))
        target_tokens.append("DNA:translated_protein")

    ex = GraphExample(
        id=f"{dataset_name}_{idx}_{_stable_id(clean + str(annotations.get('accession') or ''))}",
        task=f"{inferred}_sequence_modeling",
        nodes=nodes,
        edges=edges,
        target_tokens=target_tokens,
        metadata={
            "source_dataset": dataset_name,
            "modality": inferred,
            "sequence_length": len(clean),
            "accession": annotations.get("accession"),
        },
    )
    ex.decoder_orders = build_orders(ex.target_tokens, seed=idx)
    return ex


def graphify_protein_ec(row: dict[str, Any], idx: int, dataset_name: str) -> GraphExample:
    sequence = str(row.get("protein_sequence") or row.get("sequence") or row.get("aa_sequence") or "")
    ec_number = str(row.get("ec_number") or row.get("EC") or row.get("ec") or "")
    organism = str(row.get("organism") or row.get("taxon") or "")
    function_text = str(
        row.get("function_description")
        or row.get("function")
        or row.get("protein_description")
        or row.get("protein_name")
        or row.get("description")
        or row.get("annotation")
        or row.get("text")
        or ""
    )
    nodes = [Node(id="domain", type="science_domain", value="protein"), Node(id="protein", type="protein_sequence", value=sequence[:2048], features={"length": len(sequence)})]
    edges: list[Edge] = [Edge(src="domain", dst="protein", type="has_sequence")]
    residue_nodes, residue_edges = _sequence_nodes(sequence, "res", "amino_acid", max_len=256)
    nodes.extend(residue_nodes)
    edges.extend(residue_edges)
    for node in residue_nodes:
        edges.append(Edge(src="protein", dst=node.id, type="contains_residue"))
    if ec_number:
        parts = ec_number.split(".")
        nodes.append(Node(id="ec", type="ec_number", value=ec_number))
        edges.append(Edge(src="ec", dst="protein", type="conditions_generation"))
        prev = "ec"
        for i, part in enumerate(parts):
            node_id = f"ec_part{i}"
            nodes.append(Node(id=node_id, type="ec_number_part", value=part, features={"level": i + 1}))
            edges.append(Edge(src=prev, dst=node_id, type="ec_hierarchy"))
            prev = node_id
    if organism:
        nodes.append(Node(id="organism", type="organism", value=organism))
        edges.append(Edge(src="organism", dst="protein", type="source_organism"))
    if function_text:
        nodes.append(Node(id="function", type="function_description", value=function_text[:1024]))
        edges.append(Edge(src="protein", dst="function", type="has_function_description"))
        edges.append(Edge(src="domain", dst="function", type="requests_text_output"))
    motif_count = 0
    motif_specs = [
        ("sequence_motifs", "sequence_motif", "SEQ_MOTIF", "has_sequence_motif"),
        ("sequence_motif", "sequence_motif", "SEQ_MOTIF", "has_sequence_motif"),
        ("protein_motifs", "sequence_motif", "SEQ_MOTIF", "has_sequence_motif"),
        ("prosite", "sequence_motif", "SEQ_MOTIF", "has_sequence_motif"),
        ("interpro", "sequence_motif", "SEQ_MOTIF", "has_sequence_motif"),
        ("rfam", "sequence_motif", "SEQ_MOTIF", "has_sequence_motif"),
        ("sequence_motifs_from_structure", "sequence_motif_from_structure_vocab", "SEQ_MOTIF_FROM_STRUCTURE", "has_sequence_motif_from_structure_vocab"),
        ("sequence_motif_from_structure", "sequence_motif_from_structure_vocab", "SEQ_MOTIF_FROM_STRUCTURE", "has_sequence_motif_from_structure_vocab"),
        ("structure_vocab_sequence_motifs", "sequence_motif_from_structure_vocab", "SEQ_MOTIF_FROM_STRUCTURE", "has_sequence_motif_from_structure_vocab"),
    ]
    motif_tokens: list[str] = []
    for key, node_type, token_prefix, edge_type in motif_specs:
        for value in _motif_items(row.get(key)):
            accession, source, name = _motif_accession_source(value, key)
            node_id = f"motif{motif_count}"
            nodes.append(Node(id=node_id, type=node_type, value=name[:256], features={"accession": accession, "source": source}))
            edges.append(Edge(src="protein", dst=node_id, type=edge_type))
            if function_text:
                edges.append(Edge(src=node_id, dst="function", type="supports_function_description"))
            motif_tokens.append(f"{token_prefix}:{source}:{accession}")
            motif_count += 1
            if motif_count >= 128:
                break
        if motif_count >= 128:
            break
    target_tokens = ["UNIGENX:domain:protein", f"PROTEIN:length:{len(sequence)}"]
    if ec_number:
        target_tokens.append(f"EC:{ec_number}")
    if function_text:
        target_tokens.extend(["UGM:task:function_description", "UGM:serializer:text", f"ANSWER:{function_text[:120]}"])
    target_tokens.extend(motif_tokens)
    ex = GraphExample(
        id=f"{dataset_name}_{idx}_{_stable_id(sequence + ec_number)}",
        task="unigenx_ec_protein_generation",
        nodes=nodes,
        edges=edges,
        target_tokens=target_tokens,
        metadata={
            "source_dataset": dataset_name,
            "sequence_length": len(sequence),
            "ec_number": ec_number,
            "organism": organism,
            "function_description_present": bool(function_text),
        },
    )
    ex.decoder_orders = build_orders(ex.target_tokens, seed=idx)
    return ex


def _coord_nodes(coords: Any, prefix: str, node_type: str, parent: str, edge_type: str, max_len: int = 256) -> tuple[list[Node], list[Edge]]:
    nodes: list[Node] = []
    edges: list[Edge] = []
    if not isinstance(coords, list):
        return nodes, edges
    for i, coord in enumerate(coords[:max_len]):
        node_id = f"{prefix}{i}"
        nodes.append(Node(id=node_id, type=node_type, value=_coord_text(coord), features={"index": i}))
        edges.append(Edge(src=parent, dst=node_id, type=edge_type))
    return nodes, edges


def graphify_protein_ligand_docking(row: dict[str, Any], idx: int, dataset_name: str) -> GraphExample:
    ligand_smiles = str(row.get("ligand_smiles") or row.get("smiles") or row.get("SMILES") or "")
    protein_sequence = str(row.get("protein_sequence") or row.get("sequence") or "")
    pocket_atoms = row.get("pocket_atoms") or row.get("pocket_atom_symbols") or []
    pocket_coords = row.get("pocket_coords") or row.get("pocket_coordinates") or row.get("apo_coords") or []
    ligand_coords = row.get("ligand_coords") or row.get("ligand_coordinates") or []
    affinity = str(row.get("affinity") or row.get("binding_affinity") or row.get("Kd") or row.get("Ki") or row.get("IC50") or "")
    nodes = [
        Node(id="domain", type="science_domain", value="protein_ligand_docking"),
        Node(id="ligand", type="smiles", value=ligand_smiles),
        Node(id="protein", type="protein_sequence", value=protein_sequence[:2048], features={"length": len(protein_sequence)}),
    ]
    edges = [Edge(src="domain", dst="ligand", type="has_ligand"), Edge(src="domain", dst="protein", type="has_protein")]
    for i, atom in enumerate(list(pocket_atoms)[:256]):
        node_id = f"pocket_atom{i}"
        nodes.append(Node(id=node_id, type="pocket_atom", value=str(atom), features={"index": i}))
        edges.append(Edge(src="protein", dst=node_id, type="has_pocket_atom"))
    coord_nodes, coord_edges = _coord_nodes(pocket_coords, "pocket_coord", "protein_coordinate", "protein", "has_pocket_coordinate")
    nodes.extend(coord_nodes)
    edges.extend(coord_edges)
    coord_nodes, coord_edges = _coord_nodes(ligand_coords, "ligand_coord", "ligand_coordinate", "ligand", "has_ligand_coordinate")
    nodes.extend(coord_nodes)
    edges.extend(coord_edges)
    if affinity:
        nodes.append(Node(id="affinity", type="binding_affinity", value=affinity))
        edges.append(Edge(src="ligand", dst="affinity", type="measured_by"))
        edges.append(Edge(src="protein", dst="affinity", type="measured_by"))
    target_tokens = ["UNIGENX:domain:docking"]
    if ligand_smiles:
        target_tokens.append(f"SMILES:{ligand_smiles[:120]}")
    if affinity:
        target_tokens.append(f"AFFINITY:{affinity[:80]}")
    else:
        target_tokens.append("ANSWER:docking_graph")
    ex = GraphExample(
        id=f"{dataset_name}_{idx}_{_stable_id(ligand_smiles + protein_sequence[:64] + affinity)}",
        task="unigenx_protein_ligand_docking",
        nodes=nodes,
        edges=edges,
        target_tokens=target_tokens,
        metadata={
            "source_dataset": dataset_name,
            "ligand_smiles": ligand_smiles,
            "protein_length": len(protein_sequence),
            "pocket_atom_count": len(pocket_atoms) if hasattr(pocket_atoms, "__len__") else 0,
            "ligand_coordinate_count": len(ligand_coords) if hasattr(ligand_coords, "__len__") else 0,
            "affinity": affinity,
        },
    )
    ex.decoder_orders = build_orders(ex.target_tokens, seed=idx)
    return ex


def graphify_bioactivity(row: dict[str, Any], idx: int, dataset_name: str) -> GraphExample:
    smiles = str(row.get("smiles") or row.get("SMILES") or row.get("canonical_smiles") or row.get("Ligand SMILES") or "")
    protein = str(row.get("protein_sequence") or row.get("target_sequence") or row.get("Target Sequence") or row.get("sequence") or "")
    target_name = str(row.get("target_name") or row.get("Target Name") or row.get("protein_name") or "")
    assay_type = str(row.get("assay_type") or row.get("standard_type") or row.get("type") or "")
    value = str(row.get("standard_value") or row.get("affinity") or row.get("Ki") or row.get("Kd") or row.get("IC50") or "")
    units = str(row.get("standard_units") or row.get("units") or "")
    nodes = [Node(id="ligand", type="smiles", value=smiles), Node(id="target", type="protein_sequence", value=protein[:2048], features={"length": len(protein)})]
    edges = [Edge(src="ligand", dst="target", type="tested_against")]
    if target_name:
        nodes.append(Node(id="target_name", type="target_name", value=target_name))
        edges.append(Edge(src="target_name", dst="target", type="names_target"))
    if assay_type:
        nodes.append(Node(id="assay", type="assay_type", value=assay_type))
        edges.append(Edge(src="assay", dst="ligand", type="measures"))
    if value:
        nodes.append(Node(id="assay_value", type="assay_value", value=value, features={"units": units}))
        edges.append(Edge(src="assay_value", dst="assay" if assay_type else "ligand", type="has_value"))
    target_tokens = ["BIOMED:bioactivity"]
    if smiles:
        target_tokens.append(f"SMILES:{smiles[:120]}")
    if value:
        target_tokens.append(f"ASSAY:{assay_type or 'affinity'}:{value}{units}")
    ex = GraphExample(
        id=f"{dataset_name}_{idx}_{_stable_id(smiles + protein[:80] + value)}",
        task="biomed_bioactivity",
        nodes=nodes,
        edges=edges,
        target_tokens=target_tokens,
        metadata={"source_dataset": dataset_name, "smiles": smiles, "target_name": target_name, "assay_type": assay_type, "assay_value": value, "units": units},
    )
    ex.decoder_orders = build_orders(ex.target_tokens, seed=idx)
    return ex


def graphify_unigenx(
    row: dict[str, Any],
    idx: int,
    dataset_name: str,
    molecular_input_policy: str = SEQUENCE_ONLY,
    geometric_features: bool = False,
) -> GraphExample:
    original_row = dict(row)
    if molecular_input_policy == SEQUENCE_ONLY:
        row = sanitize_row_for_phase(row, molecular_input_policy)
    prompt = str(row.get("prompt") or "")
    completion = str(row.get("completion") or row.get("output") or "")
    formula = row.get("formula")
    mpid = row.get("mpid")
    smiles = str(row.get("smiles") or row.get("SMILES") or row.get("canonical_smiles") or "")
    selfies = str(row.get("selfies") or row.get("SELFIES") or "") or smiles_to_selfies(smiles)
    atom_symbols = row.get("atomic_symbols") or row.get("atoms") or row.get("atom_symbols") or []
    coords = row.get("pos") or row.get("positions") or row.get("coords") or []

    if row.get("ligand_smiles") or row.get("pocket_atoms") or row.get("pocket_coords") or row.get("ligand_coords"):
        return graphify_protein_ligand_docking(row, idx, dataset_name)
    if row.get("protein_sequence") or row.get("sequence") or row.get("ec_number") or row.get("EC"):
        return graphify_protein_ec(row, idx, dataset_name)

    if formula is not None or "material" in dataset_name.lower() or "crystal" in dataset_name.lower():
        nodes = [
            Node(id="domain", type="science_domain", value="material"),
            Node(id="prompt", type="instruction", value=prompt[:1024]),
            Node(id="completion", type="material_answer", value=completion[:512]),
        ]
        edges = [Edge(src="domain", dst="prompt", type="conditions"), Edge(src="prompt", dst="completion", type="answered_by")]
        formula_values = _as_text_list(formula)
        mpid_values = _as_text_list(mpid)
        for i, value in enumerate(formula_values[:32]):
            node_id = f"formula{i}"
            nodes.append(Node(id=node_id, type="material_formula", value=value))
            edges.append(Edge(src="prompt", dst=node_id, type="mentions_formula"))
        for i, value in enumerate(mpid_values[:32]):
            node_id = f"mpid{i}"
            nodes.append(Node(id=node_id, type="material_project_id", value=value))
            edges.append(Edge(src="prompt", dst=node_id, type="mentions_material_id"))
        target_tokens = ["UNIGENX:domain:material", f"ANSWER:{completion[:120]}"]
        if formula_values:
            target_tokens.append(f"MATERIAL:formula:{formula_values[0][:80]}")
        ex = GraphExample(
            id=f"{dataset_name}_{idx}_{_stable_id(prompt + completion)}",
            task="unigenx_material_reasoning",
            nodes=nodes,
            edges=edges,
            target_tokens=target_tokens,
            metadata={"source_dataset": dataset_name, "formula": formula, "mpid": mpid, "completion": completion},
        )
        ex.decoder_orders = build_orders(ex.target_tokens, seed=idx)
        return ex

    nodes = [Node(id="domain", type="science_domain", value="molecule")]
    edges: list[Edge] = []
    if smiles:
        nodes.append(Node(id="smiles", type="smiles", value=smiles))
        edges.append(Edge(src="domain", dst="smiles", type="has_smiles_sequence"))
    seq_nodes, seq_edges, seq_tokens = _selfies_sequence_nodes(selfies)
    nodes.extend(seq_nodes)
    edges.extend(seq_edges)
    if seq_nodes:
        edges.append(Edge(src="domain", dst=seq_nodes[0].id, type="has_selfies_sequence"))
    target_tokens = ["UNIGENX:domain:molecule"]
    target_tokens.extend(seq_tokens)
    if molecular_input_policy == ALLOW_STRUCTURE:
        for i, atom in enumerate(list(atom_symbols)[:128]):
            node_id = f"atom{i}"
            nodes.append(Node(id=node_id, type="atom_symbol", value=str(atom)))
            edges.append(Edge(src="domain", dst=node_id, type="contains_atom"))
            if i > 0:
                edges.append(Edge(src=f"atom{i-1}", dst=node_id, type="sequence_next"))
        for i, coord in enumerate(list(coords)[:128]):
            node_id = f"coord{i}"
            nodes.append(Node(id=node_id, type="coordinate_3d", value=_coord_text(coord), features={"index": i}))
            atom_id = f"atom{i}"
            if any(node.id == atom_id for node in nodes):
                edges.append(Edge(src=atom_id, dst=node_id, type="has_coordinate"))
            else:
                edges.append(Edge(src="domain", dst=node_id, type="has_coordinate"))
        property_keys = ["A", "B", "C", "mu", "alpha", "homo", "lumo", "gap", "r2", "zpve", "u0", "u", "h", "g", "cv"]
        for key in property_keys:
            if row.get(key) is None:
                continue
            node_id = f"prop_{key}"
            nodes.append(Node(id=node_id, type="molecule_property", value=f"{key}:{_round_value(row[key])}", features={"property": key}))
            edges.append(Edge(src="domain", dst=node_id, type="has_property"))
    else:
        descriptor_nodes, descriptor_edges, descriptor_tokens = _optional_molecular_descriptor_nodes(smiles, geometric_features)
        nodes.extend(descriptor_nodes)
        edges.extend(descriptor_edges)
        target_tokens.extend(descriptor_tokens)
    if smiles:
        target_tokens.append(f"SMILES:{smiles[:120]}")
    if molecular_input_policy == ALLOW_STRUCTURE and row.get("gap") is not None:
        target_tokens.append(f"PROPERTY:gap:{_round_value(row['gap'])}")
    elif molecular_input_policy == ALLOW_STRUCTURE and row.get("mu") is not None:
        target_tokens.append(f"PROPERTY:mu:{_round_value(row['mu'])}")
    else:
        target_tokens.append("ANSWER:molecule_sequence")
    ex = GraphExample(
        id=f"{dataset_name}_{idx}_{_stable_id(smiles + str(idx))}",
        task="unigenx_molecule_reasoning",
        nodes=nodes,
        edges=edges,
        target_tokens=target_tokens,
        metadata={
            "source_dataset": dataset_name,
            "smiles": smiles,
            "selfies": selfies,
            "atom_count": len(atom_symbols) if molecular_input_policy == ALLOW_STRUCTURE and hasattr(atom_symbols, "__len__") else 0,
            "coordinate_count": len(coords) if molecular_input_policy == ALLOW_STRUCTURE and hasattr(coords, "__len__") else 0,
            "property_keys": [key for key in ["A", "B", "C", "mu", "alpha", "homo", "lumo", "gap", "r2", "zpve", "u0", "u", "h", "g", "cv"] if molecular_input_policy == ALLOW_STRUCTURE and row.get(key) is not None],
            "molecular_input_policy": molecular_input_policy,
            "geometric_features_enabled": geometric_features,
            "ignored_structure_fields": structure_fields_present(original_row) if molecular_input_policy == SEQUENCE_ONLY else [],
        },
    )
    ex.decoder_orders = build_orders(ex.target_tokens, seed=idx)
    return ex


def graphify_generic(row: dict[str, Any], idx: int, dataset_name: str) -> GraphExample:
    text = " ".join(str(v) for v in row.values() if isinstance(v, (str, int, float)))[:2048]
    nodes = _word_nodes(text)
    target_tokens = ["CLAIM:generic_record", f"ANSWER:{text[:80]}"]
    ex = GraphExample(
        id=f"{dataset_name}_{idx}_{_stable_id(text)}",
        task="generic_graph",
        nodes=nodes,
        edges=_chain_edges(nodes),
        target_tokens=target_tokens,
        metadata={"source_dataset": dataset_name},
    )
    ex.decoder_orders = build_orders(ex.target_tokens, seed=idx)
    return ex


def graphify_hebrew_row(row: dict[str, Any], idx: int, dataset_name: str) -> GraphExample:
    lname = dataset_name.lower()
    instruction = str(row.get("instruction") or "")
    input_text = str(row.get("input") or "")
    output = str(row.get("output") or "")
    if instruction or output:
        prompt = "\n".join(part for part in [instruction, input_text] if part.strip())
        extra_nodes = [
            Node(id="instruction", type="hebrew_instruction", value=instruction[:1024]),
            Node(id="answer", type="hebrew_answer", value=output[:1024]),
        ]
        extra_edges = [Edge(src="instruction", dst="answer", type="answered_by")]
        return hebrew_text_graph(
            prompt + "\n" + output,
            idx,
            dataset_name,
            task="hebrew_instruction_sft",
            extra_nodes=extra_nodes,
            extra_edges=extra_edges,
            metadata={"instruction": instruction, "input": input_text, "output": output},
            answer=output,
        )

    text = str(row.get("text") or row.get("content") or row.get("paragraph") or "")
    title = str(row.get("title") or row.get("id") or "")
    task = "hebrew_pretraining"
    metadata = {"title": title, "license": row.get("license"), "set_id": row.get("set_id"), "record_id": row.get("record_id")}
    extra_nodes: list[Node] = []
    extra_edges: list[Edge] = []
    if title:
        extra_nodes.append(Node(id="title", type="hebrew_title", value=title[:512]))
        extra_edges.append(Edge(src="title", dst="text", type="titles"))
    if "wikianswers" in lname:
        task = "hebrew_question_similarity"
        if row.get("set_id") is not None:
            extra_nodes.append(Node(id="set", type="question_set", value=str(row.get("set_id"))))
            extra_edges.append(Edge(src="set", dst="text", type="groups_question"))
    if "medical" in lname:
        task = "hebrew_medical_pretraining"
    if "talmud" in lname or "sefaria" in lname:
        task = "hebrew_classical_text"
    return hebrew_text_graph(text or title, idx, dataset_name, task=task, extra_nodes=extra_nodes, extra_edges=extra_edges, metadata=metadata)


def graphify_rows(rows: Iterable[dict[str, Any]], dataset_name: str, start_idx: int = 0) -> Iterable[dict[str, Any]]:
    for local_idx, row in enumerate(rows):
        idx = start_idx + local_idx
        lname = dataset_name.lower()
        molecular_input_policy = ALLOW_STRUCTURE if "structure_dynamics" in lname else SEQUENCE_ONLY
        if "graphwalk" in lname or "graphinstruct" in lname or ("graphwiz" in lname and (row.get("query") or row.get("answer"))):
            ex = graphify_graph_reasoning(row, idx, dataset_name)
        elif "local_audio" in lname or row.get("local_audio_path") or row.get("audio_path"):
            ex = graphify_local_audio(row, idx, dataset_name)
        elif (
            "multimodal" in lname
            or "graph_to_graph" in lname
            or row.get("selfies")
            or row.get("SELFIES")
            or row.get("dna_sequence")
            or row.get("rna_sequence")
            or row.get("frames")
            or row.get("trajectory")
        ):
            ex = graphify_multimodal(row, idx, dataset_name, molecular_input_policy=molecular_input_policy)
        elif "pubchem" in lname:
            ex = graphify_pubchem_smiles(row, idx, dataset_name)
        elif "unigenx" in lname or "qm9" in lname or ("material" in lname and "crystal" in lname):
            ex = graphify_unigenx(row, idx, dataset_name, molecular_input_policy=molecular_input_policy)
        elif "pdbbind" in lname or "docking" in lname or row.get("ligand_smiles") or row.get("pocket_atoms"):
            ex = graphify_protein_ligand_docking(row, idx, dataset_name)
        elif "binding" in lname or "chembl" in lname or "bioactivity" in lname:
            ex = graphify_bioactivity(row, idx, dataset_name)
        elif (
            "protrek" in lname
            or "uniprot" in lname
            or "protein_function" in lname
            or ("protein" in lname and ("ec" in lname or row.get("ec_number") or row.get("EC") or row.get("function_description") or row.get("function")))
            or row.get("protein_sequence")
        ):
            ex = graphify_protein_ec(row, idx, dataset_name)
        elif (
            "rfam" in lname
            or "rnacentral" in lname
            or "dna" in lname
            or "genomic" in lname
            or row.get("rna_sequence")
            or row.get("dna_sequence")
            or (row.get("sequence") and ("family" in row or "clan" in row or "exons" in row or "introns" in row))
        ):
            ex = graphify_nucleotide_sequence(row, idx, dataset_name)
        elif "hebrew" in lname or "talmud" in lname or "sefaria" in lname:
            ex = graphify_hebrew_row(row, idx, dataset_name)
        elif "gsm" in lname or "math" in lname or "numina" in lname:
            ex = graphify_math(row, idx, dataset_name)
        elif "bigcode" in lname or "code" in lname:
            ex = graphify_code(row, idx, dataset_name)
        elif "lean" in lname or "proof" in lname:
            ex = graphify_lean(row, idx, dataset_name)
        elif "molecule" in lname or "zinc" in lname or "chem" in lname:
            ex = graphify_molecule(row, idx, dataset_name)
        else:
            ex = graphify_generic(row, idx, dataset_name)
        yield ex.to_dict()


def graphify_path(input_path: str | Path, output_path: str | Path, dataset_name: str) -> int:
    rows = list(read_jsonl(input_path))
    return write_jsonl(output_path, tqdm(graphify_rows(rows, dataset_name), total=len(rows), desc=f"graphify/{dataset_name}"))


def write_synthetic(output_path: str | Path, count: int, seed: int = 13) -> int:
    return write_jsonl(output_path, (ex.to_dict() for ex in iter_synthetic_examples(count=count, seed=seed)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert raw records to graph JSONL.")
    parser.add_argument("--input")
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-name", default="synthetic")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    if args.synthetic:
        count = write_synthetic(args.output, args.count, args.seed)
    else:
        if not args.input:
            raise SystemExit("--input is required unless --synthetic is set")
        count = graphify_path(args.input, args.output, args.dataset_name)
    print(f"Wrote {count} graph examples to {args.output}")


if __name__ == "__main__":
    main()
