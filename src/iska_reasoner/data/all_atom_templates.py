from __future__ import annotations

from typing import Any, Iterable

from iska_reasoner.graph.schema import Edge, GraphExample, Node


ALL_ATOM_CONTACT_TOKENS = (
    "ALL_ATOM_CONTACT:template_graph",
    "ALL_ATOM_CONTACT:atom_nodes",
    "ALL_ATOM_CONTACT:bond_edge_tokens",
    "ALL_ATOM_CONTACT:attention_map_ready",
    "ALL_ATOM_CONTACT:source_bioselfies",
    "ALL_ATOM_CONTACT:budgeted_8192_source",
)

ALL_ATOM_BOND_TYPES = (
    "single",
    "double",
    "triple",
    "aromatic",
    "covalent",
    "peptide",
    "phosphodiester",
)

ALL_ATOM_ELEMENTS = ("H", "B", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I")


def _clean_sequence(value: Any, alphabet: str, *, max_len: int) -> str:
    text = "".join(ch for ch in str(value or "").upper() if ch.isalpha())
    allowed = set(alphabet)
    return "".join(ch if ch in allowed else "X" for ch in text[:max_len])


def _first_text(row: dict[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _component_sequences(
    row: dict[str, Any],
    components: Iterable[tuple[str, str, str]] = (),
    *,
    max_residues: int,
) -> tuple[str, str, str, str, str]:
    proteins: list[str] = []
    dnas: list[str] = []
    rnas: list[str] = []
    selfies = _first_text(row, ("selfies", "SELFIES", "ligand_selfies"))
    smiles = _first_text(row, ("smiles", "SMILES", "canonical_smiles", "ligand_smiles"))
    for kind, value, _name in components:
        kind_norm = str(kind).strip().lower().replace("-", "_")
        text = str(value or "").strip()
        if not text:
            continue
        if kind_norm == "protein":
            proteins.append(text)
        elif kind_norm == "dna":
            dnas.append(text)
        elif kind_norm == "rna":
            rnas.append(text)
        elif kind_norm in {"ligand_selfies", "selfies"} and not selfies:
            selfies = text
        elif kind_norm in {"ligand", "smiles"} and not smiles:
            smiles = text
    protein = _first_text(row, ("protein_sequence", "sequence", "Sequence", "aa_sequence")) or "".join(proteins)
    dna = _first_text(row, ("dna_sequence", "dna")) or "".join(dnas)
    rna = _first_text(row, ("rna_sequence", "rna")) or "".join(rnas)
    return (
        _clean_sequence(protein, "ACDEFGHIKLMNPQRSTVWYBJOUXZ", max_len=max_residues),
        _clean_sequence(dna, "ACGTRYSWKMBDHVN", max_len=max_residues),
        _clean_sequence(rna, "ACGURYSWKMBDHVN", max_len=max_residues),
        selfies,
        smiles,
    )


def _int_from_row(row: dict[str, Any], keys: Iterable[str], default: int) -> int:
    for key in keys:
        try:
            value = int(row.get(key))
            if value > 0:
                return value
        except Exception:
            continue
    return default


def _normal_bond_type(value: Any) -> str:
    text = str(value or "single").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "singlebond": "single",
        "doublebond": "double",
        "triplebond": "triple",
        "bondtype.single": "single",
        "bondtype.double": "double",
        "bondtype.triple": "triple",
        "bondtype.aromatic": "aromatic",
        "amide": "covalent",
    }
    text = aliases.get(text, text)
    return text if text in ALL_ATOM_BOND_TYPES else "single"


def _normal_element(value: Any) -> str:
    text = str(value or "C").strip()
    if not text:
        return "C"
    if text[:2].capitalize() in {"Cl", "Br"}:
        return text[:2].capitalize()
    element = text[:1].upper()
    return element if element in set(ALL_ATOM_ELEMENTS) else "C"


def build_all_atom_contact_template_graph(
    row: dict[str, Any],
    *,
    components: Iterable[tuple[str, str, str]] = (),
    source_root_id: str = "structure_dynamics_proxy",
    root_id: str = "all_atom_contact_template",
) -> tuple[list[Node], list[Edge], list[str], dict[str, Any]]:
    """Create source-side all-atom atom nodes and bond edge tokens.

    The template is initialized from sequence/SELFIES strings only. It does not
    read supervised coordinates, structure files, energy labels, force labels,
    or trajectories. The number of atom nodes is budgeted because TokenGT uses
    one source token per node and one source token per edge.
    """
    max_residues = _int_from_row(row, ("all_atom_template_max_residues", "max_residues"), 500)
    source_budget = _int_from_row(row, ("all_atom_template_source_tokens", "max_source_tokens"), 8192)
    requested_atoms = _int_from_row(row, ("all_atom_template_max_atoms", "max_uma_coordinate_atoms", "trajectory_max_atoms"), 8192)
    budgeted_atoms = max(1, (source_budget - 512) // 2)
    max_atoms = max(1, min(requested_atoms, budgeted_atoms))
    protein, dna, rna, selfies, smiles = _component_sequences(row, components, max_residues=max_residues)
    if not any((protein, dna, rna, selfies, smiles)):
        return [], [], [], {"all_atom_template_atoms": 0, "all_atom_template_bonds": 0}

    metadata: dict[str, Any] = {
        "protein_sequence": protein,
        "dna_sequence": dna,
        "rna_sequence": rna,
        "selfies": selfies,
        "smiles": smiles,
    }
    seed_nodes: list[Node] = []
    if protein:
        seed_nodes.append(Node(id="template_protein", type="protein_sequence", value=protein, features={"length": len(protein)}))
    if dna:
        seed_nodes.append(Node(id="template_dna", type="dna_sequence", value=dna, features={"length": len(dna)}))
    if rna:
        seed_nodes.append(Node(id="template_rna", type="rna_sequence", value=rna, features={"length": len(rna)}))
    if selfies:
        seed_nodes.append(Node(id="template_selfies", type="selfies", value=selfies[:8192], features={"length": len(selfies)}))
    if smiles:
        seed_nodes.append(Node(id="template_smiles", type="smiles", value=smiles[:2048], features={"length": len(smiles)}))
    seed = GraphExample(id="all_atom_contact_template_seed", task="structure_dynamics_proxy", nodes=seed_nodes, edges=[], target_tokens=[], metadata=metadata)

    try:
        from iska_reasoner.inference.structure_dynamics import derive_full_cartesian_geometry

        geometry = derive_full_cartesian_geometry(seed, max_atoms=max_atoms, max_residues=max_residues)
    except Exception:
        return [], [], [], {"all_atom_template_atoms": 0, "all_atom_template_bonds": 0, "all_atom_template_error": True}

    atoms = list(geometry.get("atoms") or [])[:max_atoms]
    bonds = [dict(bond) for bond in list(geometry.get("bonds") or [])]
    if not atoms:
        return [], [], [], {"all_atom_template_atoms": 0, "all_atom_template_bonds": 0}

    nodes: list[Node] = [
        Node(
            id=root_id,
            type="all_atom_contact_template",
            value="sequence_initialized_all_atom_graph",
            features={
                "atom_count": len(atoms),
                "max_atoms": max_atoms,
                "source_token_budget": source_budget,
                "coordinate_source": "sequence_selfies_unfolded_template",
            },
        )
    ]
    edges: list[Edge] = [Edge(src=source_root_id, dst=root_id, type="has_all_atom_contact_template")]
    tokens: list[str] = list(ALL_ATOM_CONTACT_TOKENS)
    component_counts: dict[str, int] = {}
    for atom_idx, atom in enumerate(atoms):
        element = _normal_element(atom.get("element") or atom.get("name"))
        component = str(atom.get("component") or "unknown")
        component_counts[component] = component_counts.get(component, 0) + 1
        node_id = f"{root_id}_atom_{atom_idx}"
        nodes.append(
            Node(
                id=node_id,
                type="all_atom_template_atom",
                value=f"{component}:{atom.get('residue', '')}:{atom.get('name', element)}",
                features={
                    "index": atom_idx,
                    "element": element,
                    "atom_name": str(atom.get("name") or element),
                    "residue": str(atom.get("residue") or ""),
                    "residue_index": int(atom.get("residue_index") or 0),
                    "component": component,
                    "component_index": int(atom.get("component_index") or 0),
                    "chain": str(atom.get("chain") or ""),
                    "all_atom_contact_template": True,
                },
            )
        )
        edges.append(Edge(src=root_id, dst=node_id, type="contains_atom", features={"all_atom_contact_template": True, "element": element}))
        tokens.append(f"ALL_ATOM_ELEMENT:{element}")

    atom_count = len(atoms)
    max_bonds = max(0, source_budget - atom_count - 512)
    kept_bonds = 0
    for bond in bonds:
        if kept_bonds >= max_bonds:
            break
        try:
            src_idx = int(bond.get("src", bond.get("i", 0)))
            dst_idx = int(bond.get("dst", bond.get("j", 0)))
        except Exception:
            continue
        if src_idx < 0 or dst_idx < 0 or src_idx >= atom_count or dst_idx >= atom_count or src_idx == dst_idx:
            continue
        bond_type = _normal_bond_type(bond.get("bond_type") or bond.get("type") or bond.get("order"))
        edges.append(
            Edge(
                src=f"{root_id}_atom_{src_idx}",
                dst=f"{root_id}_atom_{dst_idx}",
                type="molecular_bond",
                features={"bond_type": bond_type, "all_atom_contact_template": True},
            )
        )
        tokens.append(f"BOND:{bond_type}")
        tokens.append(f"ALL_ATOM_BOND:{bond_type}")
        kept_bonds += 1

    for component in sorted(component_counts):
        tokens.append(f"ALL_ATOM_COMPONENT:{component}")

    return (
        nodes,
        edges,
        list(dict.fromkeys(tokens)),
        {
            "all_atom_template_atoms": atom_count,
            "all_atom_template_bonds": kept_bonds,
            "all_atom_template_source_token_budget": source_budget,
            "all_atom_template_max_atoms": max_atoms,
            "all_atom_template_components": component_counts,
            "all_atom_template_truncated": int(len(geometry.get("atoms") or [])) >= max_atoms,
        },
    )
