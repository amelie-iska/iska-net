from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

from tqdm.auto import tqdm

from iska_reasoner.data.bioselfies import add_bioselfies_graph, bioselfies_from_modalities
from iska_reasoner.data.audio import extract_audio_features
from iska_reasoner.data.hebrew import hebrew_text_graph
from iska_reasoner.data.motifs import normalize_fragment
from iska_reasoner.data.multimodal import _add_oracle_attention_motion_priors, graphify_multimodal, parse_temperature_kelvin
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


def _as_sequence_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, (list, tuple)):
                    return [str(item) for item in parsed if str(item).strip()]
            except Exception:
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, list):
                        return [str(item) for item in parsed if str(item).strip()]
                except Exception:
                    pass
        return [text]
    return [str(value)] if str(value).strip() else []


def _as_index_list(value: Any) -> list[int]:
    out: list[int] = []
    for item in _as_sequence_list(value):
        try:
            out.append(int(item))
        except Exception:
            for piece in re.findall(r"-?\d+", item):
                try:
                    out.append(int(piece))
                except Exception:
                    pass
    return out


def _as_bool_list(value: Any) -> list[bool]:
    out: list[bool] = []
    for item in _as_sequence_list(value):
        text = str(item).strip().lower()
        out.append(text in {"1", "true", "t", "+", "plus", "forward"})
    return out


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


def _first_text(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _record_items(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, dict):
        return [value]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith(("[", "{")):
            try:
                parsed = json.loads(text)
                return _record_items(parsed)
            except Exception:
                try:
                    parsed = ast.literal_eval(text)
                    return _record_items(parsed)
                except Exception:
                    pass
        return [item for item in re.split(r"[;\n]\s*", text) if item]
    return [value]


def _record_text(value: Any, keys: tuple[str, ...] = ("description", "text", "value", "name", "id", "accession")) -> str:
    if isinstance(value, dict):
        return _first_text(value, keys)
    return str(value).strip()


def _normalize_measure(value: str, fallback: str = "affinity") -> str:
    text = normalize_fragment(value or fallback)
    aliases = {
        "kd": "Kd",
        "ki": "Ki",
        "ic50": "IC50",
        "ec50": "EC50",
        "kon": "kon",
        "koff": "koff",
        "delta_g": "dG",
        "dg": "dG",
    }
    return aliases.get(text.lower(), text or fallback)


def _affinity_value(row: dict[str, Any]) -> tuple[str, str, str]:
    measure = _first_text(row, ("affinity_type", "measure", "measurement", "standard_type", "type")) or "affinity"
    for key in ("affinity", "binding_affinity", "Kd", "KD", "Ki", "KI", "IC50", "EC50", "kon", "koff", "delta_g", "dG", "standard_value"):
        value = row.get(key)
        if value is not None and str(value).strip():
            if measure == "affinity" and key not in {"affinity", "binding_affinity", "standard_value"}:
                measure = key
            units = _first_text(row, ("affinity_units", "units", "standard_units", "Standard Units"))
            return _normalize_measure(measure), str(value).strip(), units
    return _normalize_measure(measure), "", _first_text(row, ("affinity_units", "units", "standard_units", "Standard Units"))


def _numeric_affinity_strength(measure: str, value: str, units: str) -> tuple[float | None, str]:
    try:
        numeric = float(re.sub(r"[^0-9.eE+-]", "", str(value)))
    except Exception:
        return None, "unknown"
    measure_norm = _normalize_measure(measure).lower()
    units_norm = normalize_fragment(units).lower()
    molar = numeric
    if units_norm in {"pm", "picomolar"}:
        molar *= 1e-12
    elif units_norm in {"nm", "nanomolar"}:
        molar *= 1e-9
    elif units_norm in {"um", "micromolar", "µm"}:
        molar *= 1e-6
    elif units_norm in {"mm", "millimolar"}:
        molar *= 1e-3
    elif units_norm in {"m", "molar"}:
        molar *= 1.0
    elif measure_norm == "dg":
        if numeric <= -10:
            return 0.95, "very_strong"
        if numeric <= -7:
            return 0.75, "strong"
        if numeric <= -5:
            return 0.5, "moderate"
        return 0.25, "weak"
    if measure_norm in {"kd", "ki", "ic50", "ec50", "affinity"}:
        if molar <= 1e-10:
            return 0.95, "very_strong"
        if molar <= 1e-8:
            return 0.80, "strong"
        if molar <= 1e-6:
            return 0.55, "moderate"
        if molar <= 1e-4:
            return 0.30, "weak"
        return 0.15, "very_weak"
    if measure_norm == "kon":
        if numeric >= 1e6:
            return 0.80, "strong"
        if numeric >= 1e4:
            return 0.55, "moderate"
        return 0.25, "weak"
    if measure_norm == "koff":
        if numeric <= 1e-4:
            return 0.85, "strong"
        if numeric <= 1e-2:
            return 0.55, "moderate"
        return 0.25, "weak"
    return None, "unknown"


def _add_uniprot_annotations(
    nodes: list[Node],
    edges: list[Edge],
    row: dict[str, Any],
    *,
    protein_node: str = "protein",
    function_node: str | None = None,
) -> list[str]:
    tokens: list[str] = []
    accession = _first_text(row, ("accession", "Entry", "entry", "uniprot_id", "UniProtKB", "primaryAccession"))
    if accession:
        nodes.append(Node(id="uniprot_accession", type="uniprot_accession", value=accession))
        edges.append(Edge(src="uniprot_accession", dst=protein_node, type="identifies_uniprot_entry"))
        tokens.append(f"UNIPROT:accession:{normalize_fragment(accession)}")
    scalar_specs = [
        (("protein_name", "Protein names", "recommended_name"), "protein_name", "names_protein", "UNIPROT:protein_name"),
        (("gene_names", "Gene Names", "gene"), "gene_name", "encoded_by_gene", "UNIPROT:gene"),
        (("organism", "Organism", "source_organism"), "organism", "source_organism", "UNIPROT:organism"),
        (("taxonomy_id", "Organism ID", "Organism (ID)", "taxon", "taxid"), "taxonomy_id", "has_taxonomy", "UNIPROT:taxon"),
        (("reviewed", "Reviewed"), "uniprot_review_status", "has_review_status", "UNIPROT:reviewed"),
        (("subcellular_location", "Subcellular location [CC]", "location"), "subcellular_location", "localizes_to", "UNIPROT:location"),
        (("pathway", "Pathway", "pathways"), "pathway_annotation", "has_pathway", "UNIPROT:pathway"),
        (("cofactor", "Cofactor", "cofactors"), "cofactor_annotation", "has_cofactor", "UNIPROT:cofactor"),
        (("catalytic_activity", "Catalytic activity", "catalytic"), "catalytic_activity", "has_catalytic_activity", "UNIPROT:catalytic_activity"),
        (("subunit", "Subunit structure", "subunit_structure"), "subunit_annotation", "has_subunit", "UNIPROT:subunit"),
    ]
    scalar_idx = 0
    for keys, node_type, edge_type, token_prefix in scalar_specs:
        for item in _record_items(_first_text(row, keys)):
            text = _record_text(item)
            if not text:
                continue
            node_id = f"uniprot_scalar_{scalar_idx}"
            nodes.append(Node(id=node_id, type=node_type, value=text[:512]))
            edges.append(Edge(src=protein_node, dst=node_id, type=edge_type))
            if function_node:
                edges.append(Edge(src=node_id, dst=function_node, type="supports_function_description"))
            tokens.append(f"{token_prefix}:{normalize_fragment(text)[:80]}")
            scalar_idx += 1
            if scalar_idx >= 64:
                break
    list_specs = [
        (("keywords", "Keywords"), "uniprot_keyword", "has_keyword", "UNIPROT:keyword"),
        (("go_terms", "go", "Gene Ontology IDs", "Gene Ontology (GO)"), "go_annotation", "has_go_annotation", "UNIPROT:go"),
        (("domains", "domain", "domain_extents"), "uniprot_domain", "has_domain_annotation", "UNIPROT:domain"),
        (("families", "protein_families", "family"), "protein_family", "has_family_annotation", "UNIPROT:family"),
        (("ptm", "post_translational_modification", "modified_residue"), "ptm_annotation", "has_ptm_annotation", "UNIPROT:ptm"),
        (("variants", "variant", "natural_variant"), "variant_annotation", "has_variant_annotation", "UNIPROT:variant"),
    ]
    list_idx = 0
    for keys, node_type, edge_type, token_prefix in list_specs:
        raw_value = None
        for key in keys:
            if row.get(key) is not None:
                raw_value = row.get(key)
                break
        for item in _record_items(raw_value):
            text = _record_text(item)
            if not text:
                continue
            features = {}
            if isinstance(item, dict):
                for name in ("begin", "end", "position", "evidence", "id", "type"):
                    if item.get(name) is not None:
                        features[name] = item.get(name)
            node_id = f"uniprot_list_{list_idx}"
            nodes.append(Node(id=node_id, type=node_type, value=text[:512], features=features))
            edges.append(Edge(src=protein_node, dst=node_id, type=edge_type))
            if function_node:
                edges.append(Edge(src=node_id, dst=function_node, type="supports_function_description"))
            tokens.append(f"{token_prefix}:{normalize_fragment(text)[:80]}")
            list_idx += 1
            if list_idx >= 128:
                break
    feature_idx = 0
    feature_sources = (
        "features",
        "Features",
        "uniprot_features",
        "Binding site",
        "Active site",
        "Metal binding",
        "DNA binding",
        "binding_sites",
        "binding_site",
        "active_sites",
        "active_site",
        "metal_binding",
        "dna_binding",
        "calcium_binding",
        "site",
    )
    for key in feature_sources:
        for item in _record_items(row.get(key)):
            if isinstance(item, dict):
                feature_type = _first_text(item, ("type", "category", "feature_type", "key")) or key
                description = _record_text(item, ("description", "ligand", "note", "text", "value", "type"))
                begin = item.get("begin") or item.get("start") or item.get("position")
                end = item.get("end") or item.get("stop") or begin
            else:
                feature_type = key
                description = str(item)
                begin = None
                end = None
            feature_norm = normalize_fragment(feature_type)
            node_id = f"uniprot_feature_{feature_idx}"
            nodes.append(
                Node(
                    id=node_id,
                    type="uniprot_feature",
                    value=(description or feature_type)[:512],
                    features={"feature_type": feature_norm, "begin": begin, "end": end},
                )
            )
            edges.append(Edge(src=protein_node, dst=node_id, type="has_uniprot_feature"))
            if "binding" in feature_norm or key in {"binding_sites", "binding_site"}:
                edges.append(Edge(src=node_id, dst=protein_node, type="marks_binding_site"))
                tokens.append("UNIPROT:feature:binding_site")
            tokens.append(f"UNIPROT:feature:{feature_norm}")
            if begin is not None:
                tokens.append(f"UNIPROT:feature_position:{normalize_fragment(str(begin))}")
            feature_idx += 1
            if feature_idx >= 192:
                break
    return sorted(dict.fromkeys(tokens))


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


def _bioselfies_from_components(row: dict[str, Any], components: Iterable[tuple[str, str, str]] = ()) -> str:
    explicit = str(row.get("bioselfies") or row.get("bio_selfies") or row.get("BioSELFIES") or "").strip()
    if explicit:
        return explicit
    tokens: list[str] = []

    def add_break() -> None:
        if tokens and tokens[-1] != "[CHAIN:break]":
            tokens.append("[CHAIN:break]")

    component_list = list(components)
    if not component_list:
        local = dict(row)
        smiles = str(local.get("smiles") or local.get("SMILES") or local.get("canonical_smiles") or local.get("ligand_smiles") or "")
        selfies = str(local.get("selfies") or local.get("SELFIES") or local.get("ligand_selfies") or "") or smiles_to_selfies(smiles)
        if selfies:
            local["selfies"] = selfies
        return bioselfies_from_modalities(local)

    for kind, value, _name in component_list:
        kind_norm = normalize_fragment(kind)
        clean = re.sub(r"\s+", "", str(value))
        if not clean:
            continue
        add_break()
        if kind_norm == "protein":
            tokens.extend(f"[AA:{char if char in 'ACDEFGHIKLMNPQRSTVWYBJOUXZ' else 'X'}]" for char in clean.upper() if char.isalpha())
        elif kind_norm == "rna":
            tokens.extend(f"[RNA:{char if char in 'ACGURYSWKMBDHVN' else 'N'}]" for char in clean.upper() if char.isalpha())
        elif kind_norm == "dna":
            tokens.extend(f"[DNA:{char if char in 'ACGTRYSWKMBDHVN' else 'N'}]" for char in clean.upper() if char.isalpha())
        elif kind_norm in {"ligand", "smiles"}:
            tokens.extend(re.findall(r"\[[^\[\]]+\]", smiles_to_selfies(clean)))
        elif kind_norm in {"ligand_selfies", "selfies"}:
            tokens.extend(re.findall(r"\[[^\[\]]+\]", clean))
    return "".join(tokens[:8192])


def _add_bioselfies_and_structure_dynamics(
    nodes: list[Node],
    edges: list[Edge],
    target_tokens: list[str],
    row: dict[str, Any],
    *,
    components: Iterable[tuple[str, str, str]] = (),
    function_text: str = "",
) -> None:
    """Add BioSELFIES and UMA/all-atom Cartesian candidate records.

    These are symbolic candidate targets for the oracle-guided path. They do
    not add coordinate, energy, force, PDB, SDF, mmCIF, conformer, or trajectory
    labels to the row.
    """
    node_ids = {node.id for node in nodes}
    if "task" not in node_ids:
        nodes.append(Node(id="task", type="structure_dynamics_task", value="structure_dynamics_proxy"))
        for parent in ("domain", "complex", "protein", "target"):
            if parent in node_ids:
                edges.append(Edge(src=parent, dst="task", type="requests_structure_dynamics_proxy"))
                break

    bioselfies_text = _bioselfies_from_components(row, components)
    modalities: set[str] = set()
    if bioselfies_text:
        root_id = "bioselfies" if "bioselfies" not in node_ids else "bioselfies_bio"
        result = add_bioselfies_graph(nodes, edges, bioselfies_text, root_id=root_id)
        target_tokens.extend(result.target_tokens)
        modalities.update(item for item in result.modalities if item in {"protein", "dna", "rna", "selfies"})
    for kind, _value, _name in components:
        kind_norm = normalize_fragment(kind)
        if kind_norm in {"protein", "dna", "rna"}:
            modalities.add(kind_norm)
        elif kind_norm in {"ligand", "ligand_selfies", "selfies", "smiles"}:
            modalities.add("selfies")
    if row.get("protein_sequence") or row.get("sequence") or row.get("Sequence"):
        modalities.add("protein")
    if row.get("rna_sequence"):
        modalities.add("rna")
    if row.get("dna_sequence"):
        modalities.add("dna")
    if row.get("smiles") or row.get("SMILES") or row.get("ligand_smiles") or row.get("selfies") or row.get("SELFIES"):
        modalities.add("selfies")
    if not modalities:
        return
    stage_row = dict(row)
    stage_row["structure_dynamics_proxy"] = True
    stage_row.setdefault("oracle", {"name": "uma"})
    temp_k = parse_temperature_kelvin(stage_row.get("temperature") or stage_row.get("temp") or stage_row.get("T"))
    target_tokens.extend(
        _add_oracle_attention_motion_priors(
            nodes,
            edges,
            stage_row,
            sorted(modalities),
            temp_k,
            function_text,
            "structure_dynamics_proxy",
        )
    )


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
    bio_row = dict(row)
    bio_row["rna_sequence" if inferred == "rna" else "dna_sequence"] = clean
    _add_bioselfies_and_structure_dynamics(
        nodes,
        edges,
        target_tokens,
        bio_row,
        function_text=str(row.get("description") or row.get("function_description") or ""),
    )

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
    sequence = str(row.get("protein_sequence") or row.get("sequence") or row.get("Sequence") or row.get("aa_sequence") or "")
    ec_number = str(row.get("ec_number") or row.get("EC") or row.get("ec") or row.get("EC number") or "")
    organism = str(row.get("organism") or row.get("Organism") or row.get("taxon") or "")
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
    uniprot_tokens = _add_uniprot_annotations(
        nodes,
        edges,
        row,
        protein_node="protein",
        function_node="function" if function_text else None,
    )
    target_tokens = ["UNIGENX:domain:protein", f"PROTEIN:length:{len(sequence)}"]
    if ec_number:
        target_tokens.append(f"EC:{ec_number}")
    if function_text:
        target_tokens.extend(["UGM:task:function_description", "UGM:serializer:text", f"ANSWER:{function_text[:120]}"])
    target_tokens.extend(motif_tokens)
    target_tokens.extend(uniprot_tokens)
    bio_row = dict(row)
    bio_row["protein_sequence"] = sequence
    _add_bioselfies_and_structure_dynamics(nodes, edges, target_tokens, bio_row, function_text=function_text)
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


def _complex_components(row: dict[str, Any]) -> list[tuple[str, str, str]]:
    components: list[tuple[str, str, str]] = []
    for item in _record_items(row.get("components") or row.get("complex_components")):
        if not isinstance(item, dict):
            continue
        kind = normalize_fragment(_first_text(item, ("type", "kind", "modality")) or "biomolecule")
        sequence = _first_text(item, ("sequence", "protein_sequence", "rna_sequence", "dna_sequence", "smiles", "selfies", "value"))
        name = _first_text(item, ("name", "id", "accession")) or f"{kind}_{len(components)}"
        if sequence:
            components.append((kind, sequence, name))
    keyed_specs = [
        ("protein", ("protein_sequence_a", "protein_a_sequence", "sequence_a", "chain_a_sequence", "protein_sequence_1", "protein1_sequence")),
        ("protein", ("protein_sequence_b", "protein_b_sequence", "sequence_b", "chain_b_sequence", "protein_sequence_2", "protein2_sequence")),
        ("protein", ("protein_sequence", "target_sequence")),
        ("rna", ("rna_sequence", "rna_sequence_a", "rna_sequence_1")),
        ("rna", ("rna_sequence_b", "rna_sequence_2")),
        ("dna", ("dna_sequence", "dna_sequence_a", "dna_sequence_1")),
        ("dna", ("dna_sequence_b", "dna_sequence_2")),
        ("ligand", ("ligand_smiles", "smiles", "SMILES")),
        ("ligand_selfies", ("selfies", "SELFIES", "ligand_selfies")),
    ]
    seen = {(kind, value) for kind, value, _name in components}
    for kind, keys in keyed_specs:
        value = _first_text(row, keys)
        if value and (kind, value) not in seen:
            components.append((kind, value, keys[0]))
            seen.add((kind, value))
    return components


def _component_bioselfies_token(kind: str, value: str) -> str:
    kind_norm = normalize_fragment(kind)
    local: dict[str, Any] = {}
    if kind_norm == "protein":
        local["protein_sequence"] = value
        return bioselfies_from_modalities(local)
    if kind_norm == "rna":
        local["rna_sequence"] = value
        return bioselfies_from_modalities(local)
    if kind_norm == "dna":
        local["dna_sequence"] = value
        return bioselfies_from_modalities(local)
    if kind_norm in {"ligand", "smiles"}:
        return smiles_to_selfies(value)
    if kind_norm in {"ligand_selfies", "selfies"}:
        return value
    return ""


def _has_complex_affinity(row: dict[str, Any]) -> bool:
    _measure, value, _units = _affinity_value(row)
    return bool(value) and len(_complex_components(row)) >= 2


def _add_affinity_contact_priors(
    nodes: list[Node],
    edges: list[Edge],
    target_tokens: list[str],
    components: list[tuple[str, str, str]],
    *,
    measure: str,
    affinity: str,
    units: str,
    interaction_type: str,
) -> None:
    if not affinity or len(components) < 2:
        return
    strength, label = _numeric_affinity_strength(measure, affinity, units)
    strength_features = {"measure": measure, "value": affinity, "units": units, "strength_label": label}
    if strength is not None:
        strength_features["strength"] = round(strength, 6)
    nodes.append(Node(id="affinity_contact_prior", type="affinity_contact_prior", value=label, features=strength_features))
    edges.append(Edge(src="affinity", dst="affinity_contact_prior", type="weights_interface_prior"))
    target_tokens.extend(
        [
            "CONTACT_PATCH:affinity_weighted_interface",
            f"AFFINITY_CONTACT:{_normalize_measure(measure)}:{label}",
            "JACOBIAN_CONTACT:affinity_weighted_interface",
        ]
    )
    protein_indices = [idx for idx, (kind, _value, _name) in enumerate(components[:8]) if kind == "protein"]
    for pair_idx, src_idx in enumerate(range(min(len(components), 8))):
        for dst_idx in range(src_idx + 1, min(len(components), 8)):
            src_kind, src_seq, _src_name = components[src_idx]
            dst_kind, dst_seq, _dst_name = components[dst_idx]
            node_id = f"interface_prior_{pair_idx}_{src_idx}_{dst_idx}"
            pair_idx += 1
            pair_type = f"{src_kind}_{dst_kind}"
            features = {
                **strength_features,
                "interaction_type": interaction_type,
                "component_i": src_idx,
                "component_j": dst_idx,
                "pair_type": pair_type,
            }
            nodes.append(Node(id=node_id, type="affinity_weighted_interface_prior", value=pair_type, features=features))
            edges.append(Edge(src=f"component{src_idx}", dst=node_id, type="has_interface_prior"))
            edges.append(Edge(src=f"component{dst_idx}", dst=node_id, type="has_interface_prior"))
            edges.append(Edge(src="affinity_contact_prior", dst=node_id, type="weights_pair_interface"))
            target_tokens.append(f"COMPLEX_CONTACT:{normalize_fragment(pair_type)}:{label}")
            if src_kind == "protein" and dst_kind == "protein":
                target_tokens.append("PPI_CONTACT:affinity_weighted")
                for rank, frac in enumerate((0.25, 0.5, 0.75)):
                    src_res = max(1, min(len(src_seq), int(len(src_seq) * frac) or 1))
                    dst_res = max(1, min(len(dst_seq), int(len(dst_seq) * (1.0 - frac)) or 1))
                    contact_id = f"ppi_affinity_contact_{src_idx}_{dst_idx}_{rank}"
                    nodes.append(
                        Node(
                            id=contact_id,
                            type="ppi_affinity_contact_candidate",
                            value=f"{src_res}:{dst_res}:{label}",
                            features={
                                "component_i": src_idx,
                                "component_j": dst_idx,
                                "residue_i": src_res,
                                "residue_j": dst_res,
                                "strength_label": label,
                                "strength": round(strength, 6) if strength is not None else None,
                            },
                        )
                    )
                    edges.append(Edge(src=node_id, dst=contact_id, type="proposes_affinity_weighted_contact"))
    if len(protein_indices) >= 2:
        target_tokens.append("CONTACT_PATCH:protein_protein_interface")


def graphify_biomolecular_complex_affinity(row: dict[str, Any], idx: int, dataset_name: str) -> GraphExample:
    components = _complex_components(row)
    measure, affinity, units = _affinity_value(row)
    interaction_type = _first_text(row, ("interaction_type", "complex_type", "assay_type", "standard_type")) or "biomolecular_complex"
    nodes = [Node(id="domain", type="science_domain", value="biomolecular_complex_affinity"), Node(id="complex", type="biomolecular_complex", value=interaction_type)]
    edges = [Edge(src="domain", dst="complex", type="has_complex")]
    target_tokens = ["BIOMED:complex_affinity", f"COMPLEX:interaction:{normalize_fragment(interaction_type)}"]
    for comp_idx, (kind, value, name) in enumerate(components[:8]):
        node_id = f"component{comp_idx}"
        node_type = {
            "protein": "protein_sequence",
            "rna": "rna_sequence",
            "dna": "dna_sequence",
            "ligand": "smiles",
            "ligand_selfies": "selfies",
        }.get(kind, "biomolecule_component")
        nodes.append(Node(id=node_id, type=node_type, value=value[:2048], features={"component_index": comp_idx, "component_name": name, "component_kind": kind, "length": len(value)}))
        edges.append(Edge(src="complex", dst=node_id, type="has_component"))
        target_tokens.append(f"COMPLEX:component:{kind}")
        component_selfies = _component_bioselfies_token(kind, value)
        if component_selfies:
            selfies_node_id = f"component{comp_idx}_selfies"
            nodes.append(Node(id=selfies_node_id, type="component_selfies", value=component_selfies[:8192], features={"component_index": comp_idx, "component_kind": kind}))
            edges.append(Edge(src=node_id, dst=selfies_node_id, type="serialized_as_selfies"))
            target_tokens.append(f"COMPONENT_SELFIES:{kind}")
        if kind == "protein":
            residue_nodes, residue_edges = _sequence_nodes(value, f"component{comp_idx}_res", "amino_acid", max_len=128)
            nodes.extend(residue_nodes)
            edges.extend(residue_edges)
            for residue in residue_nodes:
                edges.append(Edge(src=node_id, dst=residue.id, type="contains_residue"))
        elif kind in {"rna", "dna"}:
            base_type = "rna_base" if kind == "rna" else "dna_base"
            base_nodes, base_edges = _sequence_nodes(value, f"component{comp_idx}_base", base_type, max_len=128)
            nodes.extend(base_nodes)
            edges.extend(base_edges)
            for base in base_nodes:
                edges.append(Edge(src=node_id, dst=base.id, type="contains_base"))
    for src_idx in range(min(len(components), 8)):
        for dst_idx in range(src_idx + 1, min(len(components), 8)):
            edges.append(Edge(src=f"component{src_idx}", dst=f"component{dst_idx}", type="forms_complex_with", features={"interaction_type": interaction_type}))
    if affinity:
        nodes.append(Node(id="affinity", type="binding_affinity", value=affinity, features={"measure": measure, "units": units}))
        edges.append(Edge(src="complex", dst="affinity", type="has_affinity_measurement"))
        target_tokens.append(f"AFFINITY:{measure}:{affinity[:80]}{units[:24]}")
        _add_affinity_contact_priors(
            nodes,
            edges,
            target_tokens,
            components,
            measure=measure,
            affinity=affinity,
            units=units,
            interaction_type=interaction_type,
        )
    for condition_key, node_type, token_prefix in [
        ("temperature", "temperature", "TEMP"),
        ("pH", "ph", "PH"),
        ("buffer", "assay_buffer", "BUFFER"),
        ("assay", "assay_type", "ASSAY"),
    ]:
        value = row.get(condition_key)
        if value is None or not str(value).strip():
            continue
        node_id = f"condition_{normalize_fragment(condition_key)}"
        nodes.append(Node(id=node_id, type=node_type, value=str(value)[:256]))
        edges.append(Edge(src=node_id, dst="affinity", type="conditions_affinity" if affinity else "conditions_complex"))
        target_tokens.append(f"{token_prefix}:{normalize_fragment(str(value))[:80]}")
    _add_bioselfies_and_structure_dynamics(
        nodes,
        edges,
        target_tokens,
        row,
        components=components,
        function_text=str(row.get("function_description") or row.get("description") or ""),
    )
    ex = GraphExample(
        id=f"{dataset_name}_{idx}_{_stable_id('|'.join(value[:64] for _kind, value, _name in components) + affinity)}",
        task="biomolecular_complex_affinity",
        nodes=nodes,
        edges=edges,
        target_tokens=target_tokens,
        metadata={
            "source_dataset": dataset_name,
            "component_count": len(components),
            "component_kinds": [kind for kind, _value, _name in components],
            "affinity": affinity,
            "affinity_measure": measure,
            "affinity_units": units,
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
    bio_components: list[tuple[str, str, str]] = []
    if protein:
        bio_components.append(("protein", protein, target_name or "target"))
    if smiles:
        bio_components.append(("ligand", smiles, "ligand"))
    _add_bioselfies_and_structure_dynamics(
        nodes,
        edges,
        target_tokens,
        row,
        components=bio_components,
        function_text=str(row.get("function_description") or row.get("description") or target_name),
    )
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


def _omg_contact_records(row: dict[str, Any]) -> list[dict[str, Any]]:
    raw = row.get("categorical_jacobian_contacts") or row.get("jacobian_contacts") or row.get("glm2_contacts") or []
    if isinstance(raw, dict):
        raw = raw.get("contacts") or raw.get("pairs") or []
    records: list[dict[str, Any]] = []
    for item in _as_sequence_list(raw) if isinstance(raw, str) else list(raw or []):
        try:
            if isinstance(item, dict):
                src = int(item.get("src", item.get("i", item.get("residue_i", 0))))
                dst = int(item.get("dst", item.get("j", item.get("residue_j", 0))))
                score = float(item.get("score", item.get("probability", item.get("jacobian", 0.0))))
                kind = str(item.get("kind") or item.get("type") or "categorical_jacobian")
            elif isinstance(item, (list, tuple)) and len(item) >= 3:
                src, dst, score = int(item[0]), int(item[1]), float(item[2])
                kind = "categorical_jacobian"
            else:
                continue
        except Exception:
            continue
        if src > 0 and dst > 0 and src != dst:
            records.append({"src": src, "dst": dst, "score": score, "kind": kind})
    return records


def graphify_omg_mixed_contig(row: dict[str, Any], idx: int, dataset_name: str) -> GraphExample:
    cds_seqs = _as_sequence_list(row.get("CDS_seqs") or row.get("cds_seqs") or row.get("protein_sequences"))
    igs_seqs = _as_sequence_list(row.get("IGS_seqs") or row.get("igs_seqs") or row.get("intergenic_sequences"))
    cds_ids = _as_sequence_list(row.get("CDS_ids") or row.get("cds_ids"))
    igs_ids = _as_sequence_list(row.get("IGS_ids") or row.get("igs_ids"))
    cds_positions = _as_index_list(row.get("CDS_position_ids") or row.get("cds_position_ids")) or list(range(1, 2 * len(cds_seqs) + 1, 2))
    igs_positions = _as_index_list(row.get("IGS_position_ids") or row.get("igs_position_ids")) or list(range(0, 2 * len(igs_seqs), 2))
    orientations = _as_bool_list(row.get("CDS_orientations") or row.get("cds_orientations"))
    nodes = [
        Node(id="domain", type="science_domain", value="open_metagenome"),
        Node(id="contig", type="omg_mixed_contig", value=str(row.get("contig_id") or row.get("sample_id") or row.get("id") or idx), features={"cds_count": len(cds_seqs), "igs_count": len(igs_seqs)}),
    ]
    edges = [Edge(src="domain", dst="contig", type="has_omg_contig")]
    target_tokens = [
        "OMG:mixed_modality_contig",
        "OMG:cds_igs_interleaved",
        "UGM:tokenizer:bioselfies",
        "UGM:modality:bioselfies",
        f"OMG:cds_count:{min(len(cds_seqs), 64)}",
        f"OMG:igs_count:{min(len(igs_seqs), 64)}",
    ]
    if igs_seqs:
        target_tokens.extend(["OMG:has_intergenic_sequence", "GLM2_CONTEXT:intergenic_regulatory_syntax"])
    if len(cds_seqs) >= 2:
        target_tokens.extend(["CONTACT_PATCH:categorical_jacobian", "JACOBIAN_CONTACT:inter_cds_candidate"])
    elements: list[tuple[int, str, int, str, str, bool | None]] = []
    for i, seq in enumerate(cds_seqs):
        elements.append((cds_positions[i] if i < len(cds_positions) else 2 * i + 1, "cds", i, seq, cds_ids[i] if i < len(cds_ids) else f"CDS_{i}", orientations[i] if i < len(orientations) else None))
    for i, seq in enumerate(igs_seqs):
        elements.append((igs_positions[i] if i < len(igs_positions) else 2 * i, "igs", i, seq, igs_ids[i] if i < len(igs_ids) else f"IGS_{i}", None))
    elements.sort(key=lambda item: item[0])
    prev_node = "contig"
    for element_order, (position, kind, local_idx, seq, source_id, orientation) in enumerate(elements[:1000]):
        node_id = f"{kind}{local_idx}"
        clean = re.sub(r"\s+", "", seq)
        features: dict[str, Any] = {"position_id": position, "source_id": source_id[:256], "length": len(clean), "element_order": element_order}
        if orientation is not None:
            features["orientation"] = "+" if orientation else "-"
        if kind == "cds":
            clean = clean.upper()
            cds_bioselfies = bioselfies_from_modalities({"protein_sequence": clean})
            features["bioselfies"] = cds_bioselfies[:8192]
            nodes.append(Node(id=node_id, type="omg_cds", value=cds_bioselfies[:8192], features={**features, "native_sequence": clean[:2048]}))
            edges.append(Edge(src="contig", dst=node_id, type="contains_cds"))
            orient_token = "PLUS" if orientation is not False else "MINUS"
            target_tokens.append(f"OMG_CDS_ORIENT:{orient_token}")
            target_tokens.append("COMPONENT_SELFIES:protein")
            for aa_idx, aa in enumerate(clean[:96]):
                res_id = f"{node_id}_aa{aa_idx}"
                nodes.append(Node(id=res_id, type="omg_cds_amino_acid", value=aa, features={"index": aa_idx, "cds_index": local_idx}))
                edges.append(Edge(src=node_id, dst=res_id, type="contains_residue"))
                if aa_idx:
                    edges.append(Edge(src=f"{node_id}_aa{aa_idx-1}", dst=res_id, type="sequence_next"))
                target_tokens.append(f"OMG_AA:{aa}")
        else:
            clean = clean.lower()
            igs_bioselfies = bioselfies_from_modalities({"dna_sequence": clean})
            features["bioselfies"] = igs_bioselfies[:8192]
            nodes.append(Node(id=node_id, type="omg_igs", value=igs_bioselfies[:8192], features={**features, "native_sequence": clean[:2048]}))
            edges.append(Edge(src="contig", dst=node_id, type="contains_intergenic_sequence"))
            target_tokens.append("COMPONENT_SELFIES:dna")
            for nt_idx, nt in enumerate(clean[:128]):
                nt_id = f"{node_id}_nt{nt_idx}"
                nodes.append(Node(id=nt_id, type="omg_igs_nucleotide", value=nt, features={"index": nt_idx, "igs_index": local_idx}))
                edges.append(Edge(src=node_id, dst=nt_id, type="contains_nucleotide"))
                if nt_idx:
                    edges.append(Edge(src=f"{node_id}_nt{nt_idx-1}", dst=nt_id, type="sequence_next"))
                target_tokens.append(f"OMG_IGS:{nt.upper()}")
        if prev_node != "contig":
            edges.append(Edge(src=prev_node, dst=node_id, type="genomic_element_next"))
        prev_node = node_id
    for contact_idx, contact in enumerate(_omg_contact_records(row)[:256]):
        node_id = f"jacobian_contact{contact_idx}"
        score = float(contact["score"])
        nodes.append(Node(id=node_id, type="categorical_jacobian_contact", value=f"{contact['src']}:{contact['dst']}:{score:.4f}", features=contact))
        edges.append(Edge(src="contig", dst=node_id, type="has_categorical_jacobian_contact"))
        target_tokens.append("CONTACT_PATCH:categorical_jacobian")
        target_tokens.append(f"JACOBIAN_CONTACT:{normalize_fragment(contact['kind'])}")
    ex = GraphExample(
        id=f"{dataset_name}_{idx}_{_stable_id(str(row.get('id') or '') + ''.join(cds_seqs[:2]) + ''.join(igs_seqs[:2]))}",
        task="omg_mixed_metagenomic_context",
        nodes=nodes,
        edges=edges,
        target_tokens=list(dict.fromkeys(target_tokens)),
        metadata={
            "source_dataset": dataset_name,
            "cds_count": len(cds_seqs),
            "igs_count": len(igs_seqs),
            "has_intergenic_sequence": bool(igs_seqs),
            "license_warning": "OMG is CC-BY-SA-4.0; preserve attribution/share-alike requirements before scaling.",
        },
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
        elif "omg" in lname or "open_metagenome" in lname or row.get("CDS_seqs") or row.get("IGS_seqs"):
            ex = graphify_omg_mixed_contig(row, idx, dataset_name)
        elif "local_audio" in lname or row.get("local_audio_path") or row.get("audio_path"):
            ex = graphify_local_audio(row, idx, dataset_name)
        elif "complex_affinity" in lname or "biomolecular_affinity" in lname or "ppi_affinity" in lname or _has_complex_affinity(row):
            ex = graphify_biomolecular_complex_affinity(row, idx, dataset_name)
        elif (
            "multimodal" in lname
            or "graph_to_graph" in lname
            or row.get("selfies")
            or row.get("SELFIES")
            or row.get("bioselfies")
            or row.get("bio_selfies")
            or row.get("BioSELFIES")
            or str(row.get("input_representation") or row.get("tokenizer") or "").lower() in {"bioselfies", "bio_selfies", "bio-selfies"}
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
