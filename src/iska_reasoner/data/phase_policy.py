from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from iska_reasoner.graph.orders import build_orders
from iska_reasoner.graph.schema import Edge, GraphExample, Node


SEQUENCE_ONLY = "sequence_only"
ALLOW_STRUCTURE = "allow_structure"

STRUCTURE_INPUT_FIELDS = {
    "atoms",
    "atom_symbols",
    "atomic_symbols",
    "bonds",
    "bond_records",
    "coordinates",
    "coords",
    "distances",
    "distogram_targets",
    "energy",
    "force_vectors",
    "forces",
    "frames",
    "ligand_coordinates",
    "ligand_coords",
    "pdb",
    "pdb_id",
    "pocket_atoms",
    "pocket_atom_symbols",
    "pocket_coordinates",
    "pocket_coords",
    "pos",
    "positions",
    "structure",
    "structure_derived_sequence_motif",
    "structure_derived_sequence_motifs",
    "structure_motif",
    "structure_motifs",
    "trajectory",
}

STRUCTURE_PROPERTY_FIELDS = {
    "A",
    "B",
    "C",
    "alpha",
    "cv",
    "g",
    "gap",
    "h",
    "homo",
    "lumo",
    "mu",
    "r2",
    "u",
    "u0",
    "zpve",
}

STRUCTURE_NODE_TYPES = {
    "all_atom_structure",
    "coordinate_3d",
    "distance_record",
    "energy_record",
    "force_record",
    "ligand_coordinate",
    "pocket_atom",
    "protein_coordinate",
    "structure_derived_sequence_motif",
    "structure_motif",
    "trajectory_frame",
}

STRUCTURE_EDGE_TYPES = {
    "contains_coordinate",
    "has_coordinate",
    "has_derived_structure_motif",
    "has_distance",
    "has_energy",
    "has_force",
    "has_frame",
    "has_ligand_coordinate",
    "has_pocket_atom",
    "has_pocket_coordinate",
    "has_structure_motif",
    "requests_output_graph",
}

STRUCTURE_TARGET_PREFIXES = (
    "COORD:",
    "DIST:",
    "ENERGY:",
    "FORCE:",
    "PDB:",
    "STRUCT_DERIVED_SEQ_MOTIF:",
    "STRUCT_MOTIF:",
)

MOLECULAR_GRAPH_NODE_TYPES = {"atom", "atom_symbol"}
MOLECULAR_GRAPH_EDGE_TYPES = {"bond", "contains_atom", "molecular_bond"}
MOLECULAR_GRAPH_TARGET_PREFIXES = ("ATOM:", "BOND:")
PHYSICS_PROPERTY_NODE_TYPES = {"molecule_property"}
PHYSICS_TARGET_PREFIXES = ("PROPERTY:",)

STRUCTURE_TARGET_TOKENS = {
    "ANSWER:coordinate_graph",
    "UGM:modality:all_atom",
    "UGM:modality:trajectory",
    "UGM:serializer:pdb",
}

STRUCTURE_TASK_TERMS = ("structure", "conformer", "trajectory", "dynamics", "fold")
ACTUAL_STRUCTURE_FILE_SUFFIXES = (".pdb", ".ent", ".cif", ".mmcif", ".bcif", ".sdf", ".mol2", ".xtc", ".trr", ".dcd")


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if value == "":
        return False
    if isinstance(value, (list, tuple, set, dict)) and not value:
        return False
    return True


def structure_fields_present(row: Mapping[str, Any]) -> list[str]:
    """Return structure-derived fields with non-empty values in a raw row."""
    blocked = STRUCTURE_INPUT_FIELDS | STRUCTURE_PROPERTY_FIELDS
    return sorted(key for key, value in row.items() if key in blocked and _has_value(value))


def sanitize_row_for_phase(row: Mapping[str, Any], policy: str = SEQUENCE_ONLY) -> dict[str, Any]:
    """Remove structure-derived inputs for the early SELFIES/sequence-only phase.

    The early molecular curriculum should learn from string/sequence records:
    SELFIES, SMILES, proteins, DNA, RNA, names, prompts, and optional textual
    annotations. String-derived molecular graph records, such as RDKit atoms
    and bonds parsed from SMILES, are allowed because they contain connectivity
    but no coordinate or trajectory supervision. Coordinates, distance labels,
    force/energy labels, PDB records, and row-local structure-derived motifs
    are deferred to the later structure/dynamics phase. Sequence motif tokens
    imported from a frozen structure-derived motif vocabulary should use fields
    such as `sequence_motifs_from_structure`; those remain sequence-only
    metadata.
    """
    if policy == ALLOW_STRUCTURE:
        return dict(row)
    if policy != SEQUENCE_ONLY:
        raise ValueError(f"Unknown molecular input policy: {policy}")

    out = dict(row)
    ignored = structure_fields_present(out)
    for key in ignored:
        out.pop(key, None)

    task = str(out.get("task") or out.get("target_task") or "")
    if any(term in task.lower() for term in STRUCTURE_TASK_TERMS):
        out["task"] = "sequence_to_structure_dynamics_proxy"
        out["_original_task"] = task
    if ignored:
        out["_ignored_structure_fields"] = ignored
        out["_phase_policy"] = SEQUENCE_ONLY
    return out


def _has_string_molecule_anchor(example: GraphExample) -> bool:
    metadata = example.metadata or {}
    if metadata.get("smiles") or metadata.get("selfies"):
        return True
    return any(node.type in {"smiles", "selfies", "bioselfies", "molecule_sequence"} for node in example.nodes)


def graph_structure_violations(example: GraphExample) -> list[str]:
    """Return records that violate the first-run sequence/string curriculum.

    The policy forbids actual structure and dynamics supervision, not molecular
    connectivity inferred from a SMILES/SELFIES string. Older processed
    MoleculeNet rows may contain atom and bond nodes parsed from SMILES; those
    are accepted when the graph has a string molecule anchor and no coordinate,
    energy, force, PDB, all-atom, or trajectory records.
    """
    violations: list[str] = []
    molecular_graph_records: list[str] = []
    for node in example.nodes:
        if node.type in STRUCTURE_NODE_TYPES:
            violations.append(f"node:{node.type}:{node.id}")
        elif node.type in PHYSICS_PROPERTY_NODE_TYPES:
            violations.append(f"node:{node.type}:{node.id}")
        elif node.type in MOLECULAR_GRAPH_NODE_TYPES:
            molecular_graph_records.append(f"node:{node.type}:{node.id}")
    for edge in example.edges:
        if edge.type in STRUCTURE_EDGE_TYPES:
            violations.append(f"edge:{edge.type}:{edge.src}->{edge.dst}")
        elif edge.type in MOLECULAR_GRAPH_EDGE_TYPES:
            molecular_graph_records.append(f"edge:{edge.type}:{edge.src}->{edge.dst}")
    for token in example.target_tokens:
        if token in STRUCTURE_TARGET_TOKENS or token.startswith(STRUCTURE_TARGET_PREFIXES):
            violations.append(f"target:{token}")
        elif token.startswith(PHYSICS_TARGET_PREFIXES):
            violations.append(f"target:{token}")
        elif token.startswith(MOLECULAR_GRAPH_TARGET_PREFIXES):
            molecular_graph_records.append(f"target:{token}")
    metadata = example.metadata or {}
    modalities = set(metadata.get("modalities") or [])
    for modality in sorted(modalities.intersection({"all_atom", "trajectory"})):
        violations.append(f"metadata:modality:{modality}")
    for key in ["frame_count", "coordinate_count", "ligand_coordinate_count"]:
        try:
            value = int(metadata.get(key) or 0)
        except Exception:
            value = 0
        if value:
            violations.append(f"metadata:{key}:{value}")
    for key in ["atom_count", "bond_count", "bond_type_count"]:
        try:
            value = int(metadata.get(key) or 0)
        except Exception:
            value = 0
        if value:
            molecular_graph_records.append(f"metadata:{key}:{value}")
    if molecular_graph_records and (violations or not _has_string_molecule_anchor(example)):
        violations.extend(molecular_graph_records)
    return violations


def sanitize_graph_example_for_sequence_only(example: GraphExample) -> GraphExample:
    """Strip prohibited structure/dynamics supervision from an old graph row.

    This is intentionally conservative and used only when a training config
    enables the first-run sequence-only policy. It preserves text, SMILES,
    SELFIES, protein, DNA/RNA, temperature, verifier, and oracle-context
    records, while dropping coordinates, direct physics properties, all-atom
    structure records, trajectories, PDB renderers, and dependent atom records
    from legacy graphified corpora.
    """

    hard_node_ids: set[str] = set()
    for node in example.nodes:
        if node.type in STRUCTURE_NODE_TYPES or node.type in PHYSICS_PROPERTY_NODE_TYPES:
            hard_node_ids.add(node.id)

    metadata = dict(example.metadata or {})
    hard_metadata = False
    for key in ["frame_count", "coordinate_count", "ligand_coordinate_count"]:
        try:
            hard_metadata = hard_metadata or int(metadata.get(key) or 0) != 0
        except Exception:
            hard_metadata = True
    modalities = set(metadata.get("modalities") or [])
    hard_metadata = hard_metadata or bool(modalities.intersection({"all_atom", "trajectory"}))

    hard_target = any(
        token in STRUCTURE_TARGET_TOKENS
        or token.startswith(STRUCTURE_TARGET_PREFIXES)
        or token.startswith(PHYSICS_TARGET_PREFIXES)
        for token in example.target_tokens
    )
    hard_edges = [
        edge
        for edge in example.edges
        if edge.type in STRUCTURE_EDGE_TYPES or edge.src in hard_node_ids or edge.dst in hard_node_ids
    ]
    has_hard_structure = bool(hard_node_ids or hard_edges or hard_target or hard_metadata)

    remove_node_ids = set(hard_node_ids)
    if has_hard_structure:
        remove_node_ids.update(node.id for node in example.nodes if node.type in MOLECULAR_GRAPH_NODE_TYPES)

    nodes = [Node(id=node.id, type=node.type, value=node.value, features=dict(node.features)) for node in example.nodes if node.id not in remove_node_ids]
    node_ids = {node.id for node in nodes}
    edges = [
        Edge(src=edge.src, dst=edge.dst, type=edge.type, features=dict(edge.features))
        for edge in example.edges
        if edge.src in node_ids
        and edge.dst in node_ids
        and edge.type not in STRUCTURE_EDGE_TYPES
        and not (has_hard_structure and edge.type in MOLECULAR_GRAPH_EDGE_TYPES)
    ]
    target_tokens = [
        token
        for token in example.target_tokens
        if token not in STRUCTURE_TARGET_TOKENS
        and not token.startswith(STRUCTURE_TARGET_PREFIXES)
        and not token.startswith(PHYSICS_TARGET_PREFIXES)
        and not (has_hard_structure and token.startswith(MOLECULAR_GRAPH_TARGET_PREFIXES))
    ]

    if has_hard_structure and _has_string_molecule_anchor(GraphExample(example.id, example.task, nodes, edges, target_tokens, metadata, [])):
        if not any(token.startswith(("SMILES:", "SELFIES:")) for token in target_tokens):
            smiles_node = next((node for node in nodes if node.type == "smiles" and node.value), None)
            selfies_node = next((node for node in nodes if node.type == "selfies" and node.value), None)
            if selfies_node is not None:
                target_tokens.append(f"SELFIES:{selfies_node.value[:120]}")
            elif smiles_node is not None:
                target_tokens.append(f"SMILES:{smiles_node.value[:120]}")
        if not any(token.startswith("ANSWER:") for token in target_tokens):
            target_tokens.append("ANSWER:molecule_sequence")

    if has_hard_structure:
        for key in ["atom_count", "bond_count", "bond_type_count", "coordinate_count", "frame_count", "ligand_coordinate_count"]:
            metadata[key] = 0
        metadata["property_keys"] = []
        metadata["modalities"] = sorted(modalities.difference({"all_atom", "trajectory"}))
        ignored = list(metadata.get("ignored_structure_fields") or [])
        if "legacy_graph_structure_records" not in ignored:
            ignored.append("legacy_graph_structure_records")
        metadata["ignored_structure_fields"] = ignored
        metadata["sequence_only_sanitized"] = True
        metadata["removed_structure_record_count"] = len(remove_node_ids) + len(example.edges) - len(edges)

    sanitized = GraphExample(
        id=example.id,
        task=example.task,
        nodes=nodes,
        edges=edges,
        target_tokens=list(dict.fromkeys(target_tokens)),
        metadata=metadata,
    )
    sanitized.decoder_orders = build_orders(sanitized.target_tokens, seed=0)
    return sanitized


def actual_structure_file_source(example: GraphExample) -> str | None:
    metadata = example.metadata or {}
    for key in ["source_path", "structure_path", "pdb_path", "mmcif_path", "trajectory_path"]:
        value = str(metadata.get(key) or "")
        lower = value.lower()
        if lower.endswith(ACTUAL_STRUCTURE_FILE_SUFFIXES):
            return value
    return None
