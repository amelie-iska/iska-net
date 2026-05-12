from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from iska_reasoner.graph.schema import Edge, Node


AA_CODES = set("ACDEFGHIKLMNPQRSTVWYBJOUXZ")
DNA_CODES = set("ACGTRYSWKMBDHVN")
RNA_CODES = set("ACGURYSWKMBDHVN")
SELFIES_ATOMS = {
    "[C]",
    "[N]",
    "[O]",
    "[S]",
    "[P]",
    "[F]",
    "[Cl]",
    "[Br]",
    "[I]",
    "[=C]",
    "[=N]",
    "[=O]",
    "[#C]",
}
BIOSELFIES_SPECIAL_TOKENS = [
    "[CHAIN:break]",
    "[BRANCH:start]",
    "[BRANCH:end]",
    "[LINK:peptide]",
    "[LINK:phosphodiester]",
    "[LINK:glycosidic]",
    "[LINK:single]",
    "[LINK:double]",
    "[LINK:aromatic]",
    "[LINK:hydrogen]",
    "[LINK:base_pair]",
    "[LINK:stacking]",
    "[PATCH:open]",
    "[PATCH:close]",
    "[HBOND:donor]",
    "[HBOND:acceptor]",
    "[HBOND:candidate]",
    "[TORSION:backbone]",
    "[TORSION:sidechain]",
    "[TORSION:nucleic_acid]",
    "[THOUGHT:start]",
    "[THOUGHT:end]",
]
HYBRID_TOKENIZATION_TOKENS = [
    "coarse_residue",
    "coarse_base",
    "atom_patch",
    "ligand_selfies",
    "hbond_candidate",
    "interaction_edge",
    "backbone_torsion",
    "sidechain_torsion",
    "nucleic_acid_torsion",
    "open_patch",
    "close_patch",
]
HBOND_TOKENS = ["donor", "acceptor", "candidate", "virtual_hydrogen", "validity_filter"]
TORSION_TOKENS = [
    "protein_phi",
    "protein_psi",
    "protein_omega",
    "sidechain_chi",
    "rna_alpha",
    "rna_beta",
    "rna_gamma",
    "rna_delta",
    "rna_epsilon",
    "rna_zeta",
    "glycosidic_chi",
    "sugar_pucker",
]


@dataclass(slots=True)
class BioSelfiesDecodeResult:
    nodes: list[Node]
    edges: list[Edge]
    target_tokens: list[str]
    modalities: list[str]
    warnings: list[str] = field(default_factory=list)
    component_count: int = 0
    atom_count: int = 0
    residue_count: int = 0
    base_count: int = 0


def tokenize_bioselfies(text: str) -> list[str]:
    """Return bracket tokens for a BioSELFIES-style string.

    The decoder is deliberately total: malformed text is converted to escaped
    unknown tokens instead of raising. This keeps graphification robust while
    preserving a warning in metadata.
    """
    raw = str(text or "").strip()
    if not raw:
        return []
    tokens = re.findall(r"\[[^\[\]]+\]", raw)
    if tokens:
        return tokens
    return [f"[UNK:{piece}]" for piece in re.findall(r"\S+", raw)]


BIOSELFIES_MAX_INPUT_TOKENS = 8192


def bioselfies_from_modalities(row: dict[str, Any], max_len: int = BIOSELFIES_MAX_INPUT_TOKENS) -> str:
    """Serialize non-structural sequence/string fields into BioSELFIES tokens."""
    tokens: list[str] = []
    protein = str(row.get("protein_sequence") or row.get("sequence") or row.get("aa_sequence") or "")
    dna = str(row.get("dna_sequence") or row.get("dna") or "")
    rna = str(row.get("rna_sequence") or row.get("rna") or "")
    selfies = str(row.get("selfies") or row.get("SELFIES") or row.get("molecule_selfies") or "")
    if not selfies:
        smiles = str(row.get("smiles") or row.get("SMILES") or row.get("canonical_smiles") or row.get("ligand_smiles") or "")
        selfies = smiles_to_selfies_text(smiles)
    if protein:
        for char in re.sub(r"\s+", "", protein).upper():
            if char.isalpha():
                tokens.append(f"[AA:{char if char in AA_CODES else 'X'}]")
    if dna:
        if tokens:
            tokens.append("[CHAIN:break]")
        for char in re.sub(r"\s+", "", dna).upper():
            if char.isalpha():
                tokens.append(f"[DNA:{char if char in DNA_CODES else 'N'}]")
    if rna:
        if tokens:
            tokens.append("[CHAIN:break]")
        for char in re.sub(r"\s+", "", rna).upper():
            if char.isalpha():
                tokens.append(f"[RNA:{char if char in RNA_CODES else 'N'}]")
    if selfies:
        if tokens:
            tokens.append("[CHAIN:break]")
        tokens.extend(tokenize_bioselfies(selfies))
    return "".join(tokens[:max_len])


def modality_bioselfies_fields(row: dict[str, Any], max_len: int = BIOSELFIES_MAX_INPUT_TOKENS) -> dict[str, str]:
    """Return SELFIES/BioSELFIES strings for protein, DNA, RNA, and molecule fields."""
    protein = str(row.get("protein_sequence") or row.get("sequence") or row.get("aa_sequence") or "")
    dna = str(row.get("dna_sequence") or row.get("dna") or "")
    rna = str(row.get("rna_sequence") or row.get("rna") or "")
    selfies = str(row.get("selfies") or row.get("SELFIES") or row.get("molecule_selfies") or "")
    if not selfies:
        smiles = str(row.get("smiles") or row.get("SMILES") or row.get("canonical_smiles") or row.get("ligand_smiles") or "")
        selfies = smiles_to_selfies_text(smiles)
    fields: dict[str, str] = {}
    if protein:
        fields["protein_bioselfies"] = "".join(f"[AA:{char if char in AA_CODES else 'X'}]" for char in re.sub(r"\s+", "", protein).upper() if char.isalpha())[:max_len]
    if dna:
        fields["dna_bioselfies"] = "".join(f"[DNA:{char if char in DNA_CODES else 'N'}]" for char in re.sub(r"\s+", "", dna).upper() if char.isalpha())[:max_len]
    if rna:
        fields["rna_bioselfies"] = "".join(f"[RNA:{char if char in RNA_CODES else 'N'}]" for char in re.sub(r"\s+", "", rna).upper() if char.isalpha())[:max_len]
    if selfies:
        fields["molecule_selfies"] = "".join(tokenize_bioselfies(selfies))[:max_len]
    combined = bioselfies_from_modalities(row, max_len=max_len)
    if combined:
        fields["bioselfies"] = combined
    return fields


def smiles_to_selfies_text(smiles: str) -> str:
    """Best-effort SMILES-to-SELFIES conversion without making SELFIES required."""
    text = str(smiles or "").strip()
    if not text:
        return ""
    try:
        import selfies as sf  # type: ignore

        return str(sf.encoder(text))
    except Exception:
        return ""


def reference_bioselfies_tokens() -> list[str]:
    tokens: list[str] = ["UGM:tokenizer:bioselfies", "UGM:tokenizer:hybrid_multiresolution", "UGM:modality:bioselfies"]
    tokens.extend(f"BIOSELFIES:[AA:{aa}]" for aa in sorted(AA_CODES))
    tokens.extend(f"BIOSELFIES:[DNA:{base}]" for base in sorted(DNA_CODES))
    tokens.extend(f"BIOSELFIES:[RNA:{base}]" for base in sorted(RNA_CODES))
    tokens.extend(f"BIOSELFIES:{token}" for token in sorted(SELFIES_ATOMS))
    tokens.extend(f"BIOSELFIES:{token}" for token in BIOSELFIES_SPECIAL_TOKENS)
    tokens.extend(["BIOSELFIES:UNKNOWN", "BIOSELFIES:REPAIR"])
    tokens.extend(f"HYBRID:{token}" for token in HYBRID_TOKENIZATION_TOKENS)
    tokens.extend(f"HBOND:{token}" for token in HBOND_TOKENS)
    tokens.extend(f"TORSION:{token}" for token in TORSION_TOKENS)
    return sorted(dict.fromkeys(tokens))


def add_bioselfies_graph(
    nodes: list[Node],
    edges: list[Edge],
    text: str,
    *,
    root_id: str = "bioselfies",
    max_tokens: int = BIOSELFIES_MAX_INPUT_TOKENS,
) -> BioSelfiesDecodeResult:
    """Decode a supported BioSELFIES subset into graph tokens.

    Supported tokens are intentionally explicit: amino acids, DNA/RNA bases,
    ordinary SELFIES atom tokens, ``[ATOM:X]``, ``[LINK:kind]``, modification,
    patch, torsion, hydrogen-bond, branch, chain-break, and thought-control
    records. Unknown tokens become typed unknown nodes. The resulting object is
    always a valid graph fragment and never contains coordinate, distance,
    force, energy, PDB, or trajectory supervision.
    """
    raw_tokens = tokenize_bioselfies(text)[:max_tokens]
    result_nodes: list[Node] = []
    result_edges: list[Edge] = []
    target_tokens: list[str] = ["UGM:modality:bioselfies", "UGM:tokenizer:bioselfies"]
    modalities: set[str] = {"bioselfies"}
    warnings: list[str] = []
    root = Node(root_id, "bioselfies", "".join(raw_tokens), {"length": len(raw_tokens), "total_decoder": True})
    result_nodes.append(root)
    previous_component: str | None = None
    pending_link = "sequence_next"
    branch_stack: list[str | None] = []
    atom_count = 0
    residue_count = 0
    base_count = 0
    component_count = 0

    def add_component(node: Node, sequence_edge: str = "sequence_next") -> None:
        nonlocal previous_component, pending_link, component_count
        result_nodes.append(node)
        result_edges.append(Edge(root_id, node.id, "contains_bioselfies_component"))
        if previous_component is not None:
            result_edges.append(
                Edge(
                    previous_component,
                    node.id,
                    "bioselfies_link",
                    {"bond_type": pending_link if pending_link != "sequence_next" else sequence_edge},
                )
            )
        previous_component = node.id
        pending_link = "sequence_next"
        component_count += 1

    for idx, token in enumerate(raw_tokens):
        target_tokens.append(f"BIOSELFIES:{token}" if token in BIOSELFIES_SPECIAL_TOKENS or token in SELFIES_ATOMS else f"BIOSELFIES:{token}")
        inner = token[1:-1] if token.startswith("[") and token.endswith("]") else token
        if inner.startswith("AA:"):
            aa = inner.split(":", 1)[1].upper()[:1] or "X"
            if aa not in AA_CODES:
                warnings.append(f"unknown amino acid {aa!r} at token {idx}; repaired to X")
                aa = "X"
                target_tokens.append("BIOSELFIES:REPAIR")
            node = Node(f"{root_id}_aa{residue_count}", "amino_acid", aa, {"index": residue_count, "bioselfies_index": idx})
            add_component(node, sequence_edge="peptide")
            target_tokens.append(f"AA:{aa}")
            modalities.add("protein")
            residue_count += 1
        elif inner.startswith("DNA:"):
            base = inner.split(":", 1)[1].upper()[:1] or "N"
            if base not in DNA_CODES:
                warnings.append(f"unknown DNA base {base!r} at token {idx}; repaired to N")
                base = "N"
                target_tokens.append("BIOSELFIES:REPAIR")
            node = Node(f"{root_id}_dna{base_count}", "dna_base", base, {"index": base_count, "bioselfies_index": idx})
            add_component(node, sequence_edge="phosphodiester")
            target_tokens.append(f"DNA:{base}")
            modalities.add("dna")
            base_count += 1
        elif inner.startswith("RNA:"):
            base = inner.split(":", 1)[1].upper()[:1] or "N"
            if base not in RNA_CODES:
                warnings.append(f"unknown RNA base {base!r} at token {idx}; repaired to N")
                base = "N"
                target_tokens.append("BIOSELFIES:REPAIR")
            node = Node(f"{root_id}_rna{base_count}", "rna_base", base, {"index": base_count, "bioselfies_index": idx})
            add_component(node, sequence_edge="phosphodiester")
            target_tokens.append(f"RNA:{base}")
            modalities.add("rna")
            base_count += 1
        elif inner.startswith("ATOM:"):
            element = inner.split(":", 1)[1].strip() or "C"
            node = Node(f"{root_id}_atom{atom_count}", "atom", element, {"index": atom_count, "element": element, "bioselfies_index": idx})
            add_component(node, sequence_edge="single")
            target_tokens.append(f"ATOM:{element}")
            modalities.add("selfies")
            atom_count += 1
        elif token in SELFIES_ATOMS or inner.startswith(("=", "#")) or inner in {"C", "N", "O", "S", "P", "F", "Cl", "Br", "I"}:
            element = inner.lstrip("=#") or "C"
            node = Node(f"{root_id}_selfies{atom_count}", "selfies_token", token, {"index": atom_count, "element_hint": element, "bioselfies_index": idx})
            add_component(node, sequence_edge="selfies_next")
            target_tokens.append(f"SELFIES:{token}")
            modalities.add("selfies")
            atom_count += 1
        elif inner.startswith("LINK:"):
            pending_link = inner.split(":", 1)[1].strip().lower().replace("-", "_") or "single"
            link_id = f"{root_id}_link{idx}"
            result_nodes.append(Node(link_id, "bioselfies_link_token", pending_link, {"bioselfies_index": idx, "bond_type": pending_link}))
            result_edges.append(Edge(root_id, link_id, "contains_bioselfies_control"))
            target_tokens.append(f"BOND:{pending_link}")
        elif inner.startswith("MOD:"):
            mod = inner.split(":", 1)[1].strip() or "unknown"
            mod_id = f"{root_id}_mod{idx}"
            result_nodes.append(Node(mod_id, "biomolecular_modification", mod, {"bioselfies_index": idx}))
            result_edges.append(Edge(root_id, mod_id, "contains_bioselfies_control"))
            if previous_component is not None:
                result_edges.append(Edge(previous_component, mod_id, "has_biomolecular_modification"))
        elif inner == "CHAIN:break":
            break_id = f"{root_id}_chain_break{idx}"
            result_nodes.append(Node(break_id, "chain_break", "break", {"bioselfies_index": idx}))
            result_edges.append(Edge(root_id, break_id, "contains_bioselfies_control"))
            previous_component = None
            pending_link = "sequence_next"
        elif inner == "BRANCH:start":
            branch_stack.append(previous_component)
            result_nodes.append(Node(f"{root_id}_branch_start{idx}", "branch_control", "start", {"bioselfies_index": idx}))
        elif inner == "BRANCH:end":
            previous_component = branch_stack.pop() if branch_stack else previous_component
            result_nodes.append(Node(f"{root_id}_branch_end{idx}", "branch_control", "end", {"bioselfies_index": idx}))
        elif inner.startswith("PATCH:"):
            action = inner.split(":", 1)[1].strip().lower() or "open"
            patch_id = f"{root_id}_patch{idx}"
            result_nodes.append(Node(patch_id, "adaptive_patch_control", action, {"bioselfies_index": idx}))
            result_edges.append(Edge(root_id, patch_id, "controls_hybrid_resolution"))
            target_tokens.append(f"HYBRID:{'open_patch' if action == 'open' else 'close_patch'}")
        elif inner.startswith("HBOND:"):
            role = inner.split(":", 1)[1].strip().lower() or "candidate"
            node_id = f"{root_id}_hbond{idx}"
            result_nodes.append(Node(node_id, "hydrogen_bond_control", role, {"bioselfies_index": idx}))
            result_edges.append(Edge(root_id, node_id, "contains_hbond_hypothesis"))
            target_tokens.append(f"HBOND:{role if role in HBOND_TOKENS else 'candidate'}")
        elif inner.startswith("TORSION:"):
            family = inner.split(":", 1)[1].strip().lower() or "backbone"
            node_id = f"{root_id}_torsion{idx}"
            result_nodes.append(Node(node_id, "torsion_control", family, {"bioselfies_index": idx}))
            result_edges.append(Edge(root_id, node_id, "contains_torsion_hypothesis"))
            mapped = {"backbone": "protein_phi", "sidechain": "sidechain_chi", "nucleic_acid": "rna_alpha"}.get(family, "protein_phi")
            target_tokens.append(f"TORSION:{mapped}")
        elif inner.startswith("THOUGHT:"):
            thought = inner.split(":", 1)[1].strip().lower() or "state"
            result_nodes.append(Node(f"{root_id}_thought{idx}", "latent_thought_control", thought, {"bioselfies_index": idx}))
            target_tokens.append(f"REASON:{thought}" if thought in {"merge", "branch", "verify", "repair"} else "REASON:thought")
        else:
            warnings.append(f"unsupported BioSELFIES token {token!r} at index {idx}; kept as unknown")
            node_id = f"{root_id}_unknown{idx}"
            result_nodes.append(Node(node_id, "bioselfies_unknown", token, {"bioselfies_index": idx}))
            result_edges.append(Edge(root_id, node_id, "contains_bioselfies_unknown"))
            target_tokens.append("BIOSELFIES:UNKNOWN")

    nodes.extend(result_nodes)
    edges.extend(result_edges)
    return BioSelfiesDecodeResult(
        nodes=result_nodes,
        edges=result_edges,
        target_tokens=list(dict.fromkeys(target_tokens)),
        modalities=sorted(modalities),
        warnings=warnings,
        component_count=component_count,
        atom_count=atom_count,
        residue_count=residue_count,
        base_count=base_count,
    )
