from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from iska_reasoner.graph.schema import GraphExample, Node


EXPLICIT_SPLIT_KEYS = (
    "split_key",
    "entity_split_key",
    "sequence_cluster",
    "uniref_cluster",
    "mmseqs_cluster",
    "structure_cluster",
    "foldseek_cluster",
    "cath_domain",
    "ecod_domain",
    "rfam_family",
    "rna_family",
    "scaffold",
    "bemis_murcko_scaffold",
    "inchi_key_block",
    "target_family",
    "disease_family",
    "hebrew_root",
    "hebrew_template",
    "graph_generator_seed",
    "canonical_graph_hash",
    "document_id",
    "doi",
    "arxiv_id",
    "pdb_release_date",
    "source_release",
)


@dataclass(slots=True)
class SplitReport:
    policy: str
    counts: dict[str, int] = field(default_factory=dict)
    group_counts: dict[str, int] = field(default_factory=dict)
    key_family_counts: dict[str, int] = field(default_factory=dict)

    def add(self, split: str, group_key: str) -> None:
        self.counts[split] = self.counts.get(split, 0) + 1
        self.group_counts[group_key] = self.group_counts.get(group_key, 0) + 1
        family = group_key.split(":", 1)[0] if ":" in group_key else "unknown"
        self.key_family_counts[family] = self.key_family_counts.get(family, 0) + 1

    def to_dict(self) -> dict[str, Any]:
        collisions = sum(1 for count in self.group_counts.values() if count > 1)
        return {
            "policy": self.policy,
            "counts": dict(sorted(self.counts.items())),
            "groups": len(self.group_counts),
            "multi_example_groups": collisions,
            "largest_group": max(self.group_counts.values(), default=0),
            "key_family_counts": dict(sorted(self.key_family_counts.items())),
        }


def stable_hash(text: str, digest_size: int = 12) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:digest_size]


def split_name_from_key(group_key: str, val_ratio: float, test_ratio: float) -> str:
    bucket = int(hashlib.sha1(group_key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    if bucket < test_ratio:
        return "test"
    if bucket < test_ratio + val_ratio:
        return "val"
    return "train"


def row_hash_split_key(example: GraphExample) -> str:
    return f"row_hash:{_graph_hash(example)}"


def scientific_split_key(example: GraphExample) -> str:
    explicit = _explicit_metadata_key(example.metadata)
    if explicit:
        return explicit

    protein_sequence = _metadata_first(example.metadata, "protein_sequence", "sequence", "aa_sequence") or _first_node_value(
        example.nodes, {"protein_sequence"}
    )
    if protein_sequence:
        return f"protein_seq:{_sequence_key(protein_sequence, alphabet='protein')}"

    rna_sequence = _metadata_first(example.metadata, "rna_sequence", "rna") or _first_node_value(
        example.nodes, {"rna_sequence", "rna"}
    )
    if rna_sequence:
        family = _metadata_first(example.metadata, "rfam_family", "rna_family", "family")
        if family:
            return f"rna_family:{_clean(family)}"
        return f"rna_seq:{_sequence_key(rna_sequence, alphabet='rna')}"

    dna_sequence = _metadata_first(example.metadata, "dna_sequence", "dna") or _first_node_value(
        example.nodes, {"dna_sequence", "dna"}
    )
    if dna_sequence:
        return f"dna_seq:{_sequence_key(dna_sequence, alphabet='dna')}"

    smiles = _metadata_first(example.metadata, "smiles", "ligand_smiles", "selfies") or _first_node_value(
        example.nodes, {"smiles", "selfies"}
    )
    if smiles:
        return _molecule_key(smiles, example.metadata)

    hebrew_root = _metadata_first(example.metadata, "root", "hebrew_root") or _first_node_value(
        example.nodes, {"hebrew_root", "root"}
    )
    if hebrew_root:
        template = _metadata_first(example.metadata, "binyan", "template", "hebrew_template") or _first_node_value(
            example.nodes, {"hebrew_template", "binyan", "template"}
        )
        return f"hebrew_root:{_clean(hebrew_root)}:{_clean(template or 'any')}"

    graph_seed = _metadata_first(example.metadata, "graph_generator_seed", "generator_seed")
    if graph_seed:
        return f"graph_seed:{_clean(graph_seed)}"

    doc_key = _document_key(example)
    if doc_key:
        return doc_key

    return row_hash_split_key(example)


def split_key_for_policy(example: GraphExample, policy: str) -> str:
    if policy == "row_hash":
        return row_hash_split_key(example)
    if policy == "entity":
        return scientific_split_key(example)
    raise ValueError(f"Unknown split policy {policy!r}; expected row_hash or entity")


def assign_split_for_policy(example: GraphExample, policy: str, val_ratio: float, test_ratio: float) -> tuple[str, str]:
    group_key = split_key_for_policy(example, policy)
    return split_name_from_key(group_key, val_ratio, test_ratio), group_key


def _explicit_metadata_key(metadata: dict[str, Any]) -> str:
    for key in EXPLICIT_SPLIT_KEYS:
        value = metadata.get(key)
        if value is not None and str(value).strip():
            return f"{key}:{_clean(value)}"
    return ""


def _graph_hash(example: GraphExample) -> str:
    payload = {
        "task": example.task,
        "nodes": sorted((node.type, node.value) for node in example.nodes),
        "edges": sorted((edge.src, edge.dst, edge.type) for edge in example.edges),
        "target_tokens": sorted(example.target_tokens),
    }
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _metadata_first(metadata: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = metadata.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _first_node_value(nodes: Iterable[Node], node_types: set[str]) -> str:
    for node in nodes:
        if node.type in node_types and node.value:
            return str(node.value).strip()
    return ""


def _clean(value: Any, max_len: int = 80) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9_.:+-]+", "", text)
    return text[:max_len] or "unknown"


def _sequence_key(sequence: str, alphabet: str) -> str:
    seq = re.sub(r"[^A-Za-z]", "", sequence).upper()
    if not seq:
        return "empty"
    if len(seq) <= 32:
        return f"exact:{stable_hash(seq)}"
    k = 5 if alphabet == "protein" else 7
    kmers = sorted({seq[i : i + k] for i in range(max(1, len(seq) - k + 1))})
    if not kmers:
        return f"len{len(seq)}:{stable_hash(seq)}"
    sketch = _minhash(kmers, salts=12)
    length_bucket = 2 ** max(0, (len(seq) - 1).bit_length())
    return f"len{length_bucket}:mh:{'-'.join(sketch[:4])}"


def _minhash(values: list[str], salts: int = 12) -> list[str]:
    signature: list[str] = []
    for i in range(salts):
        salt = f"split{i}:"
        best = min(hashlib.blake2b((salt + value).encode("utf-8"), digest_size=8).hexdigest() for value in values)
        signature.append(best[:8])
    return signature


def _molecule_key(smiles_or_selfies: str, metadata: dict[str, Any]) -> str:
    inchi = _metadata_first(metadata, "inchi_key", "inchikey", "standard_inchi_key")
    if inchi:
        return f"inchi_key_block:{_clean(inchi.split('-', 1)[0])}"
    scaffold = _metadata_first(metadata, "scaffold", "bemis_murcko_scaffold")
    if scaffold:
        return f"scaffold:{_clean(scaffold)}"
    rdkit_scaffold = _rdkit_scaffold(smiles_or_selfies)
    if rdkit_scaffold:
        return f"scaffold:{stable_hash(rdkit_scaffold)}"
    return f"molecule:{stable_hash(smiles_or_selfies)}"


def _rdkit_scaffold(smiles: str) -> str:
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except Exception:
        return ""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except Exception:
        return ""
    return scaffold or ""


def _document_key(example: GraphExample) -> str:
    metadata = example.metadata
    for key in ("doi", "arxiv_id", "paper_id", "document_id", "source_document", "title", "record_id", "set_id"):
        value = metadata.get(key)
        if value is not None and str(value).strip():
            return f"document:{_clean(value)}"
    for node in example.nodes:
        if node.type in {"doi", "arxiv_id", "paper_id", "document_id", "title", "hebrew_title"} and node.value:
            return f"document:{_clean(node.value)}"
    return ""
