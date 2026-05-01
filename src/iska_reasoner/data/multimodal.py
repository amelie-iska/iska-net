from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Iterable

from iska_reasoner.data.motifs import (
    DEFAULT_STRUCTURE_MOTIFS,
    build_motif_vocabulary,
    derive_structure_sequence_motifs_from_atoms,
    normalize_fragment,
)
from iska_reasoner.data.phase_policy import ALLOW_STRUCTURE, SEQUENCE_ONLY, sanitize_row_for_phase, structure_fields_present
from iska_reasoner.graph.orders import build_orders
from iska_reasoner.graph.schema import Edge, GraphExample, Node


PROTEIN_AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
PROTEIN_EXTRA_RESIDUES = ["B", "J", "O", "U", "X", "Z"]
DNA_BASES = list("ACGT")
RNA_BASES = list("ACGU")
NUCLEIC_AMBIGUITY_CODES = list("RYSWKMBDHVN")

BOND_TYPES = [
    "none",
    "single",
    "double",
    "triple",
    "aromatic",
    "amide",
    "peptide",
    "phosphodiester",
    "glycosidic",
    "disulfide",
    "dative",
    "coordinate",
    "ionic",
    "hydrogen",
    "base_pair",
    "stacking",
    "watson_crick",
    "wobble",
    "hoogsteen",
    "salt_bridge",
    "metal_coordinate",
]

SELFIES_ATOM_TOKENS = [
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
    "[Ring1]",
    "[Branch1]",
    "[Branch2]",
]

STRUCTURE_MOTIF_TOKENS = list(DEFAULT_STRUCTURE_MOTIFS)

ATOM_SLOTS = [
    "N",
    "CA",
    "C",
    "O",
    "CB",
    "CG",
    "CD",
    "CE",
    "NZ",
    "OD1",
    "OD2",
    "OE1",
    "OE2",
    "SG",
    "SD",
    "NE",
    "NH1",
    "NH2",
    "OG",
    "OG1",
    "P",
    "OP1",
    "OP2",
    "O5P",
    "C5P",
    "C4P",
    "C3P",
    "O3P",
    "C2P",
    "C1P",
]

TEMPERATURE_MIN_K = 300.0
TEMPERATURE_MAX_K = 400.0
TEMPERATURES_K = [400, 375, 350, 325, 300]
TEMPERATURE_BIN_EDGES_K = list(range(300, 401, 10))
DISTANCE_BINS_A = [
    "0_2",
    "2_4",
    "4_6",
    "6_8",
    "8_10",
    "10_12",
    "12_16",
    "16_20",
    "20_plus",
]
COORDINATE_BINS = [
    "neg_far",
    "neg_mid",
    "neg_near",
    "zero",
    "pos_near",
    "pos_mid",
    "pos_far",
]
ENERGY_BINS = ["very_low", "low", "medium", "high", "very_high"]
FORCE_MAGNITUDE_BINS = ["zero", "tiny", "small", "medium", "large"]
FORCE_DIRECTIONS = ["px", "nx", "py", "ny", "pz", "nz", "mixed"]
PDB_RECORD_TYPES = ["MODEL", "ATOM", "HETATM", "CONECT", "REMARK", "ENDMDL", "END"]
TOOL_TOKENS = ["lean", "python", "rdkit", "openmm", "uma", "retriever", "pdb_parser"]
REASONING_TOKENS = [
    "thought",
    "claim",
    "evidence",
    "depends_on",
    "supports",
    "contradicts",
    "tool_call",
    "tool_result",
    "repair",
    "verify",
    "merge",
    "branch",
]
AF_STYLE_BIN_COUNT = 64
AF_STYLE_BIN_LABELS = [f"b{i:02d}" for i in range(AF_STYLE_BIN_COUNT)]
ATTENTION_BIN_LEVELS = AF_STYLE_BIN_LABELS
ATTENTION_COARSE_LEVELS = ["low", "medium", "high", "critical"]
ATTENTION_ROUTE_FAMILIES = [
    "vertex_edge",
    "sequence_to_motif",
    "motif_to_oracle",
    "temperature_to_oracle",
    "oracle_to_sequence",
    "sequence_to_motion",
    "function_to_reason",
    "thought_to_tool",
]
UMA_COUPLING_CHANNELS = [
    "sequence_oracle",
    "motif_oracle",
    "temperature_oracle",
    "function_oracle",
    "reasoning_oracle",
    "trajectory_oracle",
]
UMA_INFLUENCE_CHANNELS = [
    "score_sharpness",
    "temperature_scaling",
    "trajectory_physics",
    "candidate_acceptance",
    "diversity_pressure",
]
TOKEN_MOTION_ACTIONS = [
    "temperature_scaled",
    "stabilize",
    "refine",
    "diversify",
    "explore",
    "contract",
    "expand",
    "trajectory_follow",
    "oracle_reject",
    "oracle_accept",
]
STRUCTURE_DYNAMICS_PROXY_TOKENS = [
    "sequence_only_input",
    "temperature_conditioned",
    "uma_scored",
    "function_grounded",
    "no_structure_file",
    "candidate_motion_graph",
    "uma_trajectory_proxy",
    "physics_scored",
]


def _stable_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _as_str(value: Any) -> str:
    return "" if value is None else str(value)


def _round_float(value: Any, digits: int = 4) -> float | None:
    try:
        return round(float(value), digits)
    except Exception:
        return None


def _coord_text(coord: Any) -> str:
    if isinstance(coord, dict):
        values = [coord.get("x"), coord.get("y"), coord.get("z")]
    else:
        values = list(coord[:3]) if isinstance(coord, (list, tuple)) and len(coord) >= 3 else []
    rounded = [_round_float(value) for value in values]
    if len(rounded) == 3 and all(value is not None for value in rounded):
        return ",".join(str(value) for value in rounded)
    return _as_str(coord)


def _flatten_text(text: str, max_words: int = 48) -> list[str]:
    return re.findall(r"[A-Za-z0-9_./:+#=@%-]+", text.replace("\n", " "))[:max_words]


def tokenize_selfies(selfies: str) -> list[str]:
    tokens = re.findall(r"\[[^\[\]]+\]", selfies)
    if tokens:
        return tokens
    return [ch for ch in selfies.strip() if not ch.isspace()]


def normalize_bond_type(value: Any) -> str:
    text = _as_str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "base-pair": "base_pair",
        "watson-crick": "watson_crick",
        "watson_crick_pair": "watson_crick",
        "salt-bridge": "salt_bridge",
        "metal": "metal_coordinate",
        "coordination": "coordinate",
    }
    text = aliases.get(text, text)
    return text if text in BOND_TYPES else "single"


def parse_temperature_kelvin(value: Any) -> float | None:
    if value is None or value == "":
        return None
    match = re.search(r"[-+]?\d+(?:\.\d+)?", str(value))
    if not match:
        return None
    return float(match.group(0))


def normalize_temperature(value: Any) -> str | None:
    temp = parse_temperature_kelvin(value)
    if temp is None:
        return None
    nearest = min(TEMPERATURES_K, key=lambda item: abs(item - temp))
    return f"{nearest}K"


def _temperature_bin_token(temp_k: float) -> str:
    clamped = max(TEMPERATURE_MIN_K, min(TEMPERATURE_MAX_K, temp_k))
    for lo, hi in zip(TEMPERATURE_BIN_EDGES_K[:-1], TEMPERATURE_BIN_EDGES_K[1:]):
        if lo <= clamped <= hi:
            return f"TEMP_BIN:{lo}_{hi}K"
    return "TEMP_BIN:390_400K"


def _temperature_feature_dict(temp_k: float, anchor: str) -> dict[str, float | int | str]:
    clamped = max(TEMPERATURE_MIN_K, min(TEMPERATURE_MAX_K, temp_k))
    norm = (clamped - TEMPERATURE_MIN_K) / (TEMPERATURE_MAX_K - TEMPERATURE_MIN_K)
    return {
        "kelvin": round(temp_k, 4),
        "kelvin_clamped": round(clamped, 4),
        "kelvin_norm": round(norm, 6),
        "kelvin_centered": round(2.0 * norm - 1.0, 6),
        "inverse_temperature_300_over_T": round(300.0 / max(clamped, 1e-6), 6),
        "anchor": anchor,
    }


def _temperature_norm(temp_k: float | None) -> float:
    if temp_k is None:
        return 0.5
    clamped = max(TEMPERATURE_MIN_K, min(TEMPERATURE_MAX_K, temp_k))
    return (clamped - TEMPERATURE_MIN_K) / (TEMPERATURE_MAX_K - TEMPERATURE_MIN_K)


def _fine_bin(score: float) -> int:
    score = max(0.0, min(1.0, float(score)))
    return int(round(score * (AF_STYLE_BIN_COUNT - 1)))


def _fine_bin_label(score_or_bin: float | int) -> str:
    if isinstance(score_or_bin, int):
        idx = max(0, min(AF_STYLE_BIN_COUNT - 1, score_or_bin))
    else:
        idx = _fine_bin(score_or_bin)
    return AF_STYLE_BIN_LABELS[idx]


def _bin_center(bin_idx: int) -> float:
    return round((bin_idx + 0.5) / AF_STYLE_BIN_COUNT, 6)


def _coarse_level_from_bin(bin_idx: int) -> str:
    if bin_idx >= 48:
        return "critical"
    if bin_idx >= 32:
        return "high"
    if bin_idx >= 16:
        return "medium"
    return "low"


def _temperature_attention_bin(temp_k: float | None, route: str, complexity: float = 0.5) -> int:
    """Map continuous temperature into AF-style 64-way routing bins.

    The bin is an ordinal coupling target, analogous in spirit to distogram
    bins: it does not change the TokenGT attention operator, but it gives the
    model fine-grained records that can be rewarded by UMA-scored graph-state
    trajectories at a specified temperature.
    """

    t = _temperature_norm(temp_k)
    c = max(0.0, min(1.0, complexity))
    if route in {"temperature_to_oracle", "oracle_to_sequence"}:
        score = 0.96 - 0.28 * t + 0.05 * c
    elif route == "sequence_to_motion":
        score = 0.91 - 0.22 * t + 0.06 * c
    elif route in {"motif_to_oracle", "sequence_to_motif"}:
        score = 0.72 - 0.14 * t + 0.10 * c
    elif route == "function_to_reason":
        score = 0.64 + 0.08 * c
    elif route == "thought_to_tool":
        score = 0.58 + 0.14 * t
    elif route == "vertex_edge":
        score = 0.67 + 0.05 * c
    else:
        score = 0.50 + 0.10 * t
    return _fine_bin(score)


def _uma_coupling_bin(temp_k: float | None, channel: str, complexity: float = 0.5) -> int:
    t = _temperature_norm(temp_k)
    c = max(0.0, min(1.0, complexity))
    if channel == "temperature_oracle":
        score = 0.96 - 0.18 * abs(t - 0.25)
    elif channel == "trajectory_oracle":
        score = 0.82 - 0.20 * t + 0.08 * c
    elif channel == "sequence_oracle":
        score = 0.76 - 0.08 * t + 0.08 * c
    elif channel == "motif_oracle":
        score = 0.70 - 0.06 * t + 0.12 * c
    elif channel == "function_oracle":
        score = 0.66 + 0.10 * c
    else:
        score = 0.58 + 0.06 * t
    return _fine_bin(score)


def _uma_influence_bin(temp_k: float | None, channel: str, complexity: float = 0.5) -> int:
    t = _temperature_norm(temp_k)
    c = max(0.0, min(1.0, complexity))
    if channel == "score_sharpness":
        score = 0.98 - 0.42 * t
    elif channel == "temperature_scaling":
        score = 0.68 + 0.26 * abs(0.5 - t)
    elif channel == "trajectory_physics":
        score = 0.80 - 0.16 * t + 0.08 * c
    elif channel == "candidate_acceptance":
        score = 0.52 + 0.30 * t
    elif channel == "diversity_pressure":
        score = 0.42 + 0.45 * t
    else:
        score = 0.55
    return _fine_bin(score)


def _temperature_motion_specs(temp_k: float | None, complexity: float = 0.5) -> list[tuple[str, int]]:
    t = _temperature_norm(temp_k)
    c = max(0.0, min(1.0, complexity))
    scores = {
        "temperature_scaled": 0.94,
        "trajectory_follow": 0.82 - 0.18 * t + 0.06 * c,
        "oracle_accept": 0.74 - 0.16 * t,
        "oracle_reject": 0.62 + 0.20 * (1.0 - t),
        "stabilize": 0.90 - 0.52 * t,
        "refine": 0.84 - 0.30 * t + 0.04 * c,
        "contract": 0.72 - 0.34 * t,
        "explore": 0.38 + 0.42 * t,
        "diversify": 0.28 + 0.58 * t,
        "expand": 0.24 + 0.56 * t,
    }
    selected = ["temperature_scaled", "trajectory_follow", "oracle_accept", "oracle_reject"]
    if t <= 0.25:
        selected.extend(["stabilize", "refine", "contract"])
    elif t <= 0.60:
        selected.extend(["refine", "contract", "explore"])
    else:
        selected.extend(["explore", "diversify", "expand"])
    deduped = list(dict.fromkeys(selected))
    return [(action, _fine_bin(scores[action])) for action in deduped]


def _bin_coord(value: float | None) -> str:
    if value is None:
        return "zero"
    if value < -10:
        return "neg_far"
    if value < -3:
        return "neg_mid"
    if value < -0.25:
        return "neg_near"
    if value <= 0.25:
        return "zero"
    if value <= 3:
        return "pos_near"
    if value <= 10:
        return "pos_mid"
    return "pos_far"


def _coord_record_token(frame_idx: int, atom_idx: int, axis: str, value: float | None) -> str:
    return f"COORD:f{frame_idx}:a{atom_idx}:{axis}:{_bin_coord(value)}"


def _bin_energy(value: float | None) -> str:
    if value is None:
        return "medium"
    if value < -50:
        return "very_low"
    if value < -5:
        return "low"
    if value <= 5:
        return "medium"
    if value <= 50:
        return "high"
    return "very_high"


def _bin_distance(value: float | None) -> str:
    if value is None:
        return "20_plus"
    if value < 2:
        return "0_2"
    if value < 4:
        return "2_4"
    if value < 6:
        return "4_6"
    if value < 8:
        return "6_8"
    if value < 10:
        return "8_10"
    if value < 12:
        return "10_12"
    if value < 16:
        return "12_16"
    if value < 20:
        return "16_20"
    return "20_plus"


def multimodal_reference_tokens(extra_motif_paths: Iterable[str | Path] = ()) -> list[str]:
    tokens: list[str] = [
        "UGM:graph_to_graph",
        "UGM:task:structure_generation",
        "UGM:task:function_description",
        "UGM:task:conformer_trajectory",
        "UGM:task:sequence_to_structure_dynamics_proxy",
        "UGM:task:structure_dynamics_proxy",
        "UGM:oracle:uma_feedback",
        "UGM:decoder:random_order_ar",
        "UGM:decoder:gflownet",
        "UGM:serializer:pdb",
        "UGM:serializer:text",
        "UGM:serializer:selfies",
        "UGM:serializer:lean",
    ]
    tokens.extend(f"UGM:modality:{name}" for name in ["text", "protein", "selfies", "dna", "rna", "all_atom", "trajectory"])
    tokens.extend(f"AA:{aa}" for aa in PROTEIN_AMINO_ACIDS + PROTEIN_EXTRA_RESIDUES)
    tokens.extend(f"DNA:{base}" for base in DNA_BASES + NUCLEIC_AMBIGUITY_CODES)
    tokens.extend(f"RNA:{base}" for base in RNA_BASES + NUCLEIC_AMBIGUITY_CODES)
    tokens.extend(f"SELFIES:{tok}" for tok in SELFIES_ATOM_TOKENS)
    tokens.extend(f"BOND:{bond_type}" for bond_type in BOND_TYPES)
    tokens.extend(f"MOTIF:{motif}" for motif in STRUCTURE_MOTIF_TOKENS)
    motif_tokens, _ = build_motif_vocabulary(extra_motif_paths)
    tokens.extend(motif_tokens)
    tokens.extend(f"ATOM_SLOT:{slot}" for slot in ATOM_SLOTS)
    tokens.append("TEMP:CONTINUOUS")
    tokens.extend(f"TEMP:{temp}K" for temp in TEMPERATURES_K)
    tokens.extend(f"TEMP_ANCHOR:{temp}K" for temp in TEMPERATURES_K)
    tokens.extend(f"TEMP_BIN:{lo}_{hi}K" for lo, hi in zip(TEMPERATURE_BIN_EDGES_K[:-1], TEMPERATURE_BIN_EDGES_K[1:]))
    tokens.extend(f"DIST:{bin_name}" for bin_name in DISTANCE_BINS_A)
    tokens.extend(
        f"COORD:f{frame}:a{atom}:{axis}:{bin_name}"
        for frame in range(4)
        for atom in range(48)
        for axis in ["x", "y", "z"]
        for bin_name in COORDINATE_BINS
    )
    tokens.extend(f"ENERGY:{bin_name}" for bin_name in ENERGY_BINS)
    tokens.extend(f"FORCE:mag:{bin_name}" for bin_name in FORCE_MAGNITUDE_BINS)
    tokens.extend(f"FORCE:dir:{direction}" for direction in FORCE_DIRECTIONS)
    tokens.extend(f"PDB:{record}" for record in PDB_RECORD_TYPES)
    tokens.extend(f"TOOL:{tool}" for tool in TOOL_TOKENS)
    tokens.extend(f"REASON:{token}" for token in REASONING_TOKENS)
    tokens.extend(f"ATTN_BIN:{route}:{level}" for route in ATTENTION_ROUTE_FAMILIES for level in ATTENTION_BIN_LEVELS)
    tokens.extend(f"ATTN_COARSE:{route}:{level}" for route in ATTENTION_ROUTE_FAMILIES for level in ATTENTION_COARSE_LEVELS)
    tokens.extend(
        f"TOKEN_COUPLING:uma:{channel}:{level}"
        for channel in UMA_COUPLING_CHANNELS
        for level in ATTENTION_BIN_LEVELS
    )
    tokens.extend(
        f"UMA_INFLUENCE:uma:{channel}:{level}"
        for channel in UMA_INFLUENCE_CHANNELS
        for level in ATTENTION_BIN_LEVELS
    )
    tokens.extend(f"TOKEN_MOTION:uma:{action}:{level}" for action in TOKEN_MOTION_ACTIONS for level in ATTENTION_BIN_LEVELS)
    tokens.extend(f"UMA_TRAJ_BIN:{action}:{level}" for action in TOKEN_MOTION_ACTIONS for level in ATTENTION_BIN_LEVELS)
    tokens.extend(f"SEQ_STRUCT_DYN_PROXY:{token}" for token in STRUCTURE_DYNAMICS_PROXY_TOKENS)
    tokens.extend(f"SEQ_STRUCT_DYN_PROXY:input:{modality}" for modality in ["selfies", "protein", "dna", "rna"])
    return sorted(dict.fromkeys(tokens))


def write_multimodal_reference_tokens(path: str | Path, extra_motif_paths: Iterable[str | Path] = ()) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tokens = multimodal_reference_tokens(extra_motif_paths)
    path.write_text("\n".join(tokens) + "\n", encoding="utf-8")
    return len(tokens)


def _add_prompt(nodes: list[Node], edges: list[Edge], prompt: str) -> None:
    nodes.append(Node(id="prompt", type="instruction", value=prompt[:1024]))
    prev = "prompt"
    for i, word in enumerate(_flatten_text(prompt, 48)):
        node_id = f"prompt_tok{i}"
        nodes.append(Node(id=node_id, type="prompt_token", value=word, features={"index": i}))
        edges.append(Edge(src="prompt", dst=node_id, type="contains_token"))
        if i > 0:
            edges.append(Edge(src=prev, dst=node_id, type="next_token"))
        prev = node_id


def _add_sequence(
    nodes: list[Node],
    edges: list[Edge],
    sequence: str,
    root_id: str,
    root_type: str,
    item_type: str,
    edge_type: str,
    token_prefix: str,
    max_len: int,
) -> list[str]:
    if not sequence:
        return []
    clean = re.sub(r"\s+", "", sequence).upper()[:max_len]
    nodes.append(Node(id=root_id, type=root_type, value=clean, features={"length": len(clean)}))
    edges.append(Edge(src="task", dst=root_id, type="has_modality"))
    target_tokens: list[str] = []
    prev = None
    for i, char in enumerate(clean):
        node_id = f"{root_id}_{i}"
        nodes.append(Node(id=node_id, type=item_type, value=char, features={"index": i}))
        edges.append(Edge(src=root_id, dst=node_id, type=edge_type))
        if prev is not None:
            edges.append(Edge(src=prev, dst=node_id, type="sequence_next"))
        prev = node_id
        if i < 96:
            target_tokens.append(f"{token_prefix}:{char}")
    return target_tokens


def _add_selfies(nodes: list[Node], edges: list[Edge], selfies: str, max_len: int = 160) -> list[str]:
    if not selfies:
        return []
    tokens = tokenize_selfies(selfies)[:max_len]
    nodes.append(Node(id="selfies", type="selfies", value="".join(tokens), features={"length": len(tokens)}))
    edges.append(Edge(src="task", dst="selfies", type="has_modality"))
    target_tokens: list[str] = ["UGM:modality:selfies"]
    prev = None
    for i, token in enumerate(tokens):
        node_id = f"selfies_tok{i}"
        nodes.append(Node(id=node_id, type="selfies_token", value=token, features={"index": i}))
        edges.append(Edge(src="selfies", dst=node_id, type="contains_selfies_token"))
        if prev is not None:
            edges.append(Edge(src=prev, dst=node_id, type="sequence_next"))
        prev = node_id
        if i < 96:
            target_tokens.append(f"SELFIES:{token}")
    return target_tokens


def _optional_molecular_descriptor_nodes(nodes: list[Node], edges: list[Edge], smiles: str, enabled: bool) -> list[str]:
    if not enabled or not smiles:
        return []
    try:
        from rdkit import Chem
        from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
    except Exception:
        return []
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    descriptors = {
        "atom_count": float(mol.GetNumAtoms()),
        "bond_count": float(mol.GetNumBonds()),
        "ring_count": float(rdMolDescriptors.CalcNumRings(mol)),
        "mol_wt": float(Descriptors.MolWt(mol)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
        "logp": float(Crippen.MolLogP(mol)),
        "rotatable_bonds": float(Lipinski.NumRotatableBonds(mol)),
    }
    parent = "selfies" if any(node.id == "selfies" for node in nodes) else "task"
    tokens: list[str] = []
    for i, (name, value) in enumerate(descriptors.items()):
        node_id = f"geom_feature_{i}"
        rounded = round(value, 4)
        nodes.append(Node(id=node_id, type="molecular_geometry_feature", value=f"{name}:{rounded}", features={"name": name, "value": rounded}))
        edges.append(Edge(src=parent, dst=node_id, type="has_geometric_feature"))
        tokens.append(f"GEOM_FEATURE:{name}:{round(value, 2)}")
    return tokens


def _atom_element(atom: Any) -> str:
    if isinstance(atom, dict):
        return _as_str(atom.get("element") or atom.get("symbol") or atom.get("atom") or atom.get("name") or "C")
    return _as_str(atom or "C")


def _add_atoms_and_bonds(nodes: list[Node], edges: list[Edge], row: dict[str, Any], max_atoms: int = 128) -> tuple[list[str], list[str]]:
    atoms = _as_list(row.get("atoms") or row.get("atomic_symbols") or row.get("atom_symbols"))
    bonds = _as_list(row.get("bonds") or row.get("bond_records"))
    target_tokens: list[str] = []
    atom_ids: list[str] = []
    if atoms or bonds or row.get("frames") or row.get("coordinates") or row.get("coords"):
        nodes.append(Node(id="structure", type="all_atom_structure", value="decoded_structure"))
        edges.append(Edge(src="task", dst="structure", type="requests_output_graph"))
        target_tokens.append("UGM:modality:all_atom")
    for i, atom in enumerate(atoms[:max_atoms]):
        element = _atom_element(atom)
        node_id = f"atom{i}"
        atom_ids.append(node_id)
        features = {"index": i, "element": element}
        if isinstance(atom, dict):
            features.update({str(k): v for k, v in atom.items() if isinstance(v, (int, float, str, bool))})
        nodes.append(Node(id=node_id, type="atom", value=element, features=features))
        edges.append(Edge(src="structure", dst=node_id, type="contains_atom"))
        if i < 96:
            target_tokens.append(f"ATOM:{element}")
    for i, bond in enumerate(bonds[: max_atoms * 2]):
        if isinstance(bond, dict):
            src_idx = int(bond.get("src", bond.get("begin", bond.get("i", 0))))
            dst_idx = int(bond.get("dst", bond.get("end", bond.get("j", 0))))
            bond_type = normalize_bond_type(bond.get("bond_type") or bond.get("type") or bond.get("order"))
        elif isinstance(bond, (list, tuple)) and len(bond) >= 2:
            src_idx = int(bond[0])
            dst_idx = int(bond[1])
            bond_type = normalize_bond_type(bond[2] if len(bond) >= 3 else "single")
        else:
            continue
        src = f"atom{src_idx}"
        dst = f"atom{dst_idx}"
        if src not in atom_ids or dst not in atom_ids:
            continue
        edges.append(Edge(src=src, dst=dst, type="molecular_bond", features={"bond_type": bond_type}))
        target_tokens.append(f"BOND:{bond_type}")
    return target_tokens, atom_ids


def _frame_list(row: dict[str, Any]) -> list[Any]:
    frames = row.get("frames") or row.get("trajectory") or row.get("coordinates") or row.get("coords") or []
    frames = _as_list(frames)
    if frames and isinstance(frames[0], (list, tuple)) and len(frames[0]) >= 3 and all(isinstance(v, (int, float)) for v in frames[0][:3]):
        return [frames]
    return frames


def _add_frames(
    nodes: list[Node],
    edges: list[Edge],
    row: dict[str, Any],
    atom_ids: list[str],
    max_frames: int = 8,
    max_atoms: int = 128,
) -> list[str]:
    target_tokens: list[str] = []
    frames = _frame_list(row)
    if not frames:
        return target_tokens
    if not any(node.id == "structure" for node in nodes):
        nodes.append(Node(id="structure", type="all_atom_structure", value="decoded_structure"))
        edges.append(Edge(src="task", dst="structure", type="requests_output_graph"))
    target_tokens.extend(["UGM:modality:trajectory", "PDB:MODEL", "PDB:ATOM", "PDB:ENDMDL"])
    for frame_idx, frame in enumerate(frames[:max_frames]):
        frame_id = f"frame{frame_idx}"
        nodes.append(Node(id=frame_id, type="trajectory_frame", value=str(frame_idx), features={"frame": frame_idx}))
        edges.append(Edge(src="structure", dst=frame_id, type="has_frame"))
        coords = _as_list(frame.get("coordinates") if isinstance(frame, dict) else frame)
        for atom_idx, coord in enumerate(coords[:max_atoms]):
            coord_id = f"coord{frame_idx}_{atom_idx}"
            xyz = coord if isinstance(coord, dict) else list(coord[:3]) if isinstance(coord, (list, tuple)) else []
            x = _round_float(xyz.get("x") if isinstance(xyz, dict) else xyz[0] if len(xyz) > 0 else None)
            y = _round_float(xyz.get("y") if isinstance(xyz, dict) else xyz[1] if len(xyz) > 1 else None)
            z = _round_float(xyz.get("z") if isinstance(xyz, dict) else xyz[2] if len(xyz) > 2 else None)
            nodes.append(
                Node(
                    id=coord_id,
                    type="coordinate_3d",
                    value=_coord_text(coord),
                    features={"index": atom_idx, "frame": frame_idx, "x": x or 0.0, "y": y or 0.0, "z": z or 0.0},
                )
            )
            parent = atom_ids[atom_idx] if atom_idx < len(atom_ids) else frame_id
            edges.append(Edge(src=parent, dst=coord_id, type="has_coordinate"))
            edges.append(Edge(src=frame_id, dst=coord_id, type="contains_coordinate"))
            if atom_idx < 48 and frame_idx < 4:
                target_tokens.extend(
                    [
                        _coord_record_token(frame_idx, atom_idx, "x", x),
                        _coord_record_token(frame_idx, atom_idx, "y", y),
                        _coord_record_token(frame_idx, atom_idx, "z", z),
                    ]
                )
        energy = frame.get("energy") if isinstance(frame, dict) else None
        energy = row.get("energy") if energy is None else energy
        energy_value = _round_float(energy)
        if energy_value is not None:
            energy_id = f"energy{frame_idx}"
            nodes.append(Node(id=energy_id, type="energy_record", value=str(energy_value), features={"energy": energy_value, "frame": frame_idx}))
            edges.append(Edge(src=frame_id, dst=energy_id, type="has_energy"))
            target_tokens.append(f"ENERGY:{_bin_energy(energy_value)}")
    return target_tokens


def _add_forces(nodes: list[Node], edges: list[Edge], row: dict[str, Any], atom_ids: list[str], max_forces: int = 64) -> list[str]:
    forces = _as_list(row.get("forces") or row.get("force_vectors"))
    target_tokens: list[str] = []
    for i, force in enumerate(forces[:max_forces]):
        if isinstance(force, dict):
            values = [force.get("x"), force.get("y"), force.get("z")]
        elif isinstance(force, (list, tuple)) and len(force) >= 3:
            values = list(force[:3])
        else:
            continue
        xyz = [_round_float(v) or 0.0 for v in values]
        mag = sum(v * v for v in xyz) ** 0.5
        direction = FORCE_DIRECTIONS[max(range(6), key=lambda j: [xyz[0], -xyz[0], xyz[1], -xyz[1], xyz[2], -xyz[2]][j])]
        if mag < 1e-8:
            direction = "mixed"
        node_id = f"force{i}"
        nodes.append(Node(id=node_id, type="force_record", value=",".join(str(v) for v in xyz), features={"index": i, "fx": xyz[0], "fy": xyz[1], "fz": xyz[2], "magnitude": mag}))
        parent = atom_ids[i] if i < len(atom_ids) else "structure"
        edges.append(Edge(src=parent, dst=node_id, type="has_force"))
        target_tokens.extend([f"FORCE:mag:{_force_bin(mag)}", f"FORCE:dir:{direction}"])
    return target_tokens


def _force_bin(mag: float) -> str:
    if mag < 1e-6:
        return "zero"
    if mag < 0.05:
        return "tiny"
    if mag < 0.5:
        return "small"
    if mag < 2.0:
        return "medium"
    return "large"


def _add_distances(nodes: list[Node], edges: list[Edge], row: dict[str, Any], atom_ids: list[str], max_distances: int = 96) -> list[str]:
    distances = _as_list(row.get("distances") or row.get("distogram_targets"))
    target_tokens: list[str] = []
    for i, dist in enumerate(distances[:max_distances]):
        if isinstance(dist, dict):
            src_idx = int(dist.get("i", dist.get("src", 0)))
            dst_idx = int(dist.get("j", dist.get("dst", 0)))
            value = _round_float(dist.get("distance") or dist.get("value") or dist.get("angstrom"))
        elif isinstance(dist, (list, tuple)) and len(dist) >= 3:
            src_idx = int(dist[0])
            dst_idx = int(dist[1])
            value = _round_float(dist[2])
        else:
            continue
        src = atom_ids[src_idx] if src_idx < len(atom_ids) else "structure"
        dst = atom_ids[dst_idx] if dst_idx < len(atom_ids) else "structure"
        node_id = f"dist{i}"
        nodes.append(Node(id=node_id, type="distance_record", value=str(value), features={"distance_a": value or 0.0, "src_index": src_idx, "dst_index": dst_idx}))
        edges.append(Edge(src=src, dst=node_id, type="has_distance"))
        edges.append(Edge(src=node_id, dst=dst, type="has_distance"))
        target_tokens.append(f"DIST:{_bin_distance(value)}")
    return target_tokens


def _motif_values(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        return [item for item in re.split(r"[,;]\s*", value) if item]
    return [value]


def _motif_accession(value: Any) -> tuple[str, str]:
    if isinstance(value, dict):
        accession = value.get("accession") or value.get("id") or value.get("name") or value.get("motif") or "unknown"
        name = value.get("name") or value.get("description") or accession
        return str(accession), str(name)
    return str(value), str(value)


def _add_motifs(nodes: list[Node], edges: list[Edge], row: dict[str, Any], max_motifs: int = 256) -> list[str]:
    specs = [
        ("sequence_motifs", "sequence_motif", "SEQ_MOTIF", "has_sequence_motif"),
        ("sequence_motif", "sequence_motif", "SEQ_MOTIF", "has_sequence_motif"),
        ("protein_motifs", "sequence_motif", "SEQ_MOTIF", "has_sequence_motif"),
        ("prosite", "sequence_motif", "SEQ_MOTIF", "has_sequence_motif"),
        ("interpro", "sequence_motif", "SEQ_MOTIF", "has_sequence_motif"),
        ("rfam", "sequence_motif", "SEQ_MOTIF", "has_sequence_motif"),
        (
            "sequence_motifs_from_structure",
            "sequence_motif_from_structure_vocab",
            "SEQ_MOTIF_FROM_STRUCTURE",
            "has_sequence_motif_from_structure_vocab",
        ),
        (
            "sequence_motif_from_structure",
            "sequence_motif_from_structure_vocab",
            "SEQ_MOTIF_FROM_STRUCTURE",
            "has_sequence_motif_from_structure_vocab",
        ),
        (
            "structure_vocab_sequence_motifs",
            "sequence_motif_from_structure_vocab",
            "SEQ_MOTIF_FROM_STRUCTURE",
            "has_sequence_motif_from_structure_vocab",
        ),
        ("structure_motifs", "structure_motif", "STRUCT_MOTIF", "has_structure_motif"),
        ("structure_motif", "structure_motif", "STRUCT_MOTIF", "has_structure_motif"),
        ("cath", "structure_motif", "STRUCT_MOTIF", "has_structure_motif"),
        (
            "structure_derived_sequence_motifs",
            "structure_derived_sequence_motif",
            "STRUCT_DERIVED_SEQ_MOTIF",
            "has_structure_derived_sequence_motif",
        ),
        (
            "structure_derived_sequence_motif",
            "structure_derived_sequence_motif",
            "STRUCT_DERIVED_SEQ_MOTIF",
            "has_structure_derived_sequence_motif",
        ),
    ]
    target_tokens: list[str] = []
    count = 0
    for key, node_type, prefix, edge_type in specs:
        for value in _motif_values(row.get(key)):
            if count >= max_motifs:
                return target_tokens
            accession, name = _motif_accession(value)
            source = str(value.get("source") if isinstance(value, dict) and value.get("source") else key)
            accession_norm = normalize_fragment(accession)
            source_norm = normalize_fragment(source)
            node_id = f"motif{count}"
            nodes.append(
                Node(
                    id=node_id,
                    type=node_type,
                    value=name[:256],
                    features={"accession": accession_norm, "source": source_norm, "prefix": prefix},
                )
            )
            edges.append(Edge(src="task", dst=node_id, type=edge_type))
            target_tokens.append(f"{prefix}:{source_norm}:{accession_norm}")
            count += 1
    for value in _motif_values(row.get("motifs")):
        if count >= max_motifs:
            break
        accession, name = _motif_accession(value)
        accession_norm = normalize_fragment(accession)
        node_id = f"motif{count}"
        nodes.append(Node(id=node_id, type="motif", value=name[:256], features={"accession": accession_norm, "source": "generic"}))
        edges.append(Edge(src="task", dst=node_id, type="has_motif"))
        target_tokens.append(f"MOTIF:{accession_norm}")
        count += 1

    for record in derive_structure_sequence_motifs_from_atoms(row):
        if count >= max_motifs:
            break
        node_id = f"motif{count}"
        nodes.append(
            Node(
                id=node_id,
                type=f"{record.kind}_motif",
                value=record.name[:256],
                features={"accession": normalize_fragment(record.accession), "source": normalize_fragment(record.source), "parent": normalize_fragment(record.parent)},
            )
        )
        edges.append(Edge(src="task", dst=node_id, type="has_derived_structure_motif"))
        target_tokens.append(record.token())
        count += 1
    return target_tokens


def _add_oracle_attention_motion_priors(
    nodes: list[Node],
    edges: list[Edge],
    row: dict[str, Any],
    modalities: Iterable[str],
    temp_k: float | None,
    function_text: str,
    task_value: str,
) -> list[str]:
    modality_set = {item for item in modalities if item in {"selfies", "protein", "dna", "rna"}}
    if not modality_set:
        return []
    task_text = f"{task_value} {row.get('_original_task') or ''} {row.get('prompt') or ''} {row.get('instruction') or ''}".lower()
    stage_flags = (
        "uma_stage",
        "uma_binning",
        "oracle_binning",
        "enable_uma_binning",
        "structure_dynamics_proxy",
        "structure_dynamics_stage",
    )
    explicit_stage_flag = any(bool(row.get(flag)) for flag in stage_flags)
    oracle_requested = bool(row.get("oracle")) or explicit_stage_flag or any(
        term in task_text for term in ["structure", "dynamics", "conformer", "fold", "oracle", "uma"]
    )
    if not oracle_requested:
        return []

    tokens: list[str] = ["UGM:oracle:uma_feedback"]
    if not any(node.id == "uma_oracle" for node in nodes):
        features: dict[str, Any] = {"oracle": "UMA", "sequence_only": True}
        if temp_k is not None:
            features.update({"temperature_k": round(temp_k, 4), "temperature_conditioned": True})
        nodes.append(Node(id="uma_oracle", type="uma_oracle_feedback", value="sequence_only_temperature_conditioned", features=features))
        edges.append(Edge(src="task", dst="uma_oracle", type="uses_oracle_feedback"))

    proxy_features = {
        "sequence_only_input": True,
        "no_structure_files": True,
        "modalities": ",".join(sorted(modality_set)),
    }
    if temp_k is not None:
        proxy_features["temperature_k"] = round(temp_k, 4)
    nodes.append(Node(id="structure_dynamics_proxy", type="sequence_structure_dynamics_proxy", value="uma_scored_candidate_motion_graph", features=proxy_features))
    edges.append(Edge(src="task", dst="structure_dynamics_proxy", type="requests_sequence_structure_dynamics_proxy"))
    edges.append(Edge(src="uma_oracle", dst="structure_dynamics_proxy", type="scores_proxy_motion_graph"))
    if any(node.id == "temperature" for node in nodes):
        edges.append(Edge(src="temperature", dst="uma_oracle", type="conditions_oracle_scoring"))
        edges.append(Edge(src="temperature", dst="structure_dynamics_proxy", type="conditions_proxy_motion"))
    for modality in sorted(modality_set):
        root_ids = ["selfies", "smiles"] if modality == "selfies" else [modality]
        for root_id in root_ids:
            if any(node.id == root_id for node in nodes):
                edges.append(Edge(src=root_id, dst="structure_dynamics_proxy", type="sequence_drives_proxy_motion"))
        tokens.append(f"SEQ_STRUCT_DYN_PROXY:input:{modality}")
    tokens.extend(
        [
            "UGM:task:structure_dynamics_proxy",
            "SEQ_STRUCT_DYN_PROXY:sequence_only_input",
            "SEQ_STRUCT_DYN_PROXY:uma_scored",
            "SEQ_STRUCT_DYN_PROXY:no_structure_file",
            "SEQ_STRUCT_DYN_PROXY:candidate_motion_graph",
        ]
    )
    if temp_k is not None:
        tokens.append("SEQ_STRUCT_DYN_PROXY:temperature_conditioned")
    if function_text:
        tokens.append("SEQ_STRUCT_DYN_PROXY:function_grounded")
    tokens.extend(["SEQ_STRUCT_DYN_PROXY:uma_trajectory_proxy", "SEQ_STRUCT_DYN_PROXY:physics_scored"])

    sequence_nodes = [
        node
        for node in nodes
        if node.type
        in {
            "amino_acid",
            "dna_base",
            "rna_base",
            "selfies_token",
            "smiles_char",
            "sequence_motif",
            "sequence_motif_from_structure_vocab",
        }
    ]
    complexity = min(1.0, (len(sequence_nodes) / 256.0) + (0.15 if function_text else 0.0) + (0.10 * len(modality_set)))

    routes = ["vertex_edge", "sequence_to_motif", "motif_to_oracle", "oracle_to_sequence", "sequence_to_motion"]
    if temp_k is not None:
        routes.append("temperature_to_oracle")
    if function_text:
        routes.append("function_to_reason")
    for i, route in enumerate(routes):
        bin_idx = _temperature_attention_bin(temp_k, route, complexity)
        level = _fine_bin_label(bin_idx)
        coarse = _coarse_level_from_bin(bin_idx)
        node_id = f"attention_bin_{i}"
        nodes.append(
            Node(
                id=node_id,
                type="attention_coupling_bin",
                value=f"{route}:{level}",
                features={
                    "route": route,
                    "bin": level,
                    "bin_index": bin_idx,
                    "bin_count": AF_STYLE_BIN_COUNT,
                    "bin_center": _bin_center(bin_idx),
                    "coarse_level": coarse,
                    "temperature_k": round(temp_k, 4) if temp_k is not None else None,
                },
            )
        )
        edges.append(Edge(src="task", dst=node_id, type="has_attention_coupling_prior"))
        edges.append(Edge(src=node_id, dst="uma_oracle", type="routes_oracle_coupling"))
        tokens.append(f"ATTN_BIN:{route}:{level}")
        tokens.append(f"ATTN_COARSE:{route}:{coarse}")

    for channel in UMA_COUPLING_CHANNELS:
        if channel == "temperature_oracle" and temp_k is None:
            continue
        if channel == "function_oracle" and not function_text:
            continue
        bin_idx = _uma_coupling_bin(temp_k, channel, complexity)
        level = _fine_bin_label(bin_idx)
        tokens.append(f"TOKEN_COUPLING:uma:{channel}:{level}")
        node_id = f"uma_coupling_{channel}"
        nodes.append(
            Node(
                id=node_id,
                type="uma_coupling_strength_bin",
                value=f"{channel}:{level}",
                features={
                    "channel": channel,
                    "bin": level,
                    "bin_index": bin_idx,
                    "bin_count": AF_STYLE_BIN_COUNT,
                    "strength_center": _bin_center(bin_idx),
                    "temperature_k": round(temp_k, 4) if temp_k is not None else None,
                },
            )
        )
        edges.append(Edge(src="uma_oracle", dst=node_id, type="sets_token_coupling_strength"))
        edges.append(Edge(src=node_id, dst="structure_dynamics_proxy", type="conditions_proxy_motion"))

    for channel in UMA_INFLUENCE_CHANNELS:
        bin_idx = _uma_influence_bin(temp_k, channel, complexity)
        level = _fine_bin_label(bin_idx)
        tokens.append(f"UMA_INFLUENCE:uma:{channel}:{level}")
        node_id = f"uma_influence_{channel}"
        nodes.append(
            Node(
                id=node_id,
                type="uma_influence_bin",
                value=f"{channel}:{level}",
                features={
                    "channel": channel,
                    "bin": level,
                    "bin_index": bin_idx,
                    "bin_count": AF_STYLE_BIN_COUNT,
                    "influence_center": _bin_center(bin_idx),
                    "temperature_k": round(temp_k, 4) if temp_k is not None else None,
                },
            )
        )
        edges.append(Edge(src="uma_oracle", dst=node_id, type="sets_uma_influence"))
        edges.append(Edge(src=node_id, dst="structure_dynamics_proxy", type="scores_proxy_trajectory"))

    for i, (action, bin_idx) in enumerate(_temperature_motion_specs(temp_k, complexity)):
        level = _fine_bin_label(bin_idx)
        node_id = f"token_motion_{i}"
        nodes.append(
            Node(
                id=node_id,
                type="token_motion_prior",
                value=f"{action}:{level}",
                features={
                    "action": action,
                    "bin": level,
                    "bin_index": bin_idx,
                    "bin_count": AF_STYLE_BIN_COUNT,
                    "motion_center": _bin_center(bin_idx),
                    "oracle": "UMA",
                    "temperature_conditioned": temp_k is not None,
                    "temperature_k": round(temp_k, 4) if temp_k is not None else None,
                },
            )
        )
        edges.append(Edge(src="uma_oracle", dst=node_id, type="uma_modulates_token_motion"))
        edges.append(Edge(src=node_id, dst="structure_dynamics_proxy", type="updates_proxy_motion_state"))
        tokens.append(f"TOKEN_MOTION:uma:{action}:{level}")
        tokens.append(f"UMA_TRAJ_BIN:{action}:{level}")
    return tokens


def graphify_multimodal(
    row: dict[str, Any],
    idx: int,
    dataset_name: str,
    molecular_input_policy: str = SEQUENCE_ONLY,
    geometric_features: bool = False,
) -> GraphExample:
    original_row = dict(row)
    row = sanitize_row_for_phase(row, molecular_input_policy)
    prompt = _as_str(row.get("prompt") or row.get("instruction") or row.get("text") or "Generate a graph-structured scientific output.")
    task_value = _as_str(row.get("task") or row.get("target_task") or "sequence_or_selfies_reconstruction")
    raw_temp = row.get("temperature") or row.get("temp") or row.get("T")
    temp_k = parse_temperature_kelvin(raw_temp)
    temp_anchor = normalize_temperature(raw_temp)

    nodes: list[Node] = [
        Node(id="task", type="multimodal_task", value=task_value),
    ]
    edges: list[Edge] = []
    _add_prompt(nodes, edges, prompt)
    edges.append(Edge(src="task", dst="prompt", type="specified_by"))
    target_tokens = ["UGM:graph_to_graph", f"UGM:task:{task_value}"]
    sequence_modalities: list[str] = []

    protein_sequence = _as_str(row.get("protein_sequence") or row.get("sequence") or row.get("aa_sequence"))
    if protein_sequence:
        sequence_modalities.append("protein")
        target_tokens.append("UGM:modality:protein")
        target_tokens.extend(_add_sequence(nodes, edges, protein_sequence, "protein", "protein_sequence", "amino_acid", "contains_residue", "AA", 512))

    selfies = _as_str(row.get("selfies") or row.get("SELFIES"))
    if selfies:
        sequence_modalities.append("selfies")
        target_tokens.extend(_add_selfies(nodes, edges, selfies))

    dna_sequence = _as_str(row.get("dna_sequence") or row.get("dna"))
    if dna_sequence:
        sequence_modalities.append("dna")
        target_tokens.append("UGM:modality:dna")
        target_tokens.extend(_add_sequence(nodes, edges, dna_sequence, "dna", "dna_sequence", "dna_base", "contains_base", "DNA", 512))

    rna_sequence = _as_str(row.get("rna_sequence") or row.get("rna"))
    if rna_sequence:
        sequence_modalities.append("rna")
        target_tokens.append("UGM:modality:rna")
        target_tokens.extend(_add_sequence(nodes, edges, rna_sequence, "rna", "rna_sequence", "rna_base", "contains_base", "RNA", 512))

    smiles = _as_str(row.get("smiles") or row.get("SMILES") or row.get("canonical_smiles"))
    if smiles and not selfies:
        sequence_modalities.append("selfies")
        nodes.append(Node(id="smiles", type="smiles", value=smiles))
        edges.append(Edge(src="task", dst="smiles", type="has_modality"))
        target_tokens.extend(["UGM:modality:selfies", f"SMILES:{smiles[:120]}"])
    target_tokens.extend(_optional_molecular_descriptor_nodes(nodes, edges, smiles, geometric_features))

    if temp_k is not None and temp_anchor:
        temp_text = f"{temp_k:.4f}".rstrip("0").rstrip(".")
        temp_value = f"{temp_text}K"
        nodes.append(Node(id="temperature", type="temperature", value=temp_value, features=_temperature_feature_dict(temp_k, temp_anchor)))
        edges.append(Edge(src="task", dst="temperature", type="conditions_generation"))
        target_tokens.extend(["TEMP:CONTINUOUS", f"TEMP:{temp_anchor}", f"TEMP_ANCHOR:{temp_anchor}", _temperature_bin_token(temp_k)])

    target_tokens.extend(_add_motifs(nodes, edges, row))
    function_text = _as_str(row.get("function") or row.get("function_description") or row.get("target_text") or row.get("answer"))
    target_tokens.extend(_add_oracle_attention_motion_priors(nodes, edges, row, sequence_modalities, temp_k, function_text, task_value))
    atom_ids: list[str] = []
    if molecular_input_policy == ALLOW_STRUCTURE:
        structure_tokens, atom_ids = _add_atoms_and_bonds(nodes, edges, row)
        target_tokens.extend(structure_tokens)
        target_tokens.extend(_add_distances(nodes, edges, row, atom_ids))
        target_tokens.extend(_add_frames(nodes, edges, row, atom_ids))
        target_tokens.extend(_add_forces(nodes, edges, row, atom_ids))

    if function_text:
        nodes.append(Node(id="function", type="function_description", value=function_text[:1024]))
        edges.append(Edge(src="task", dst="function", type="requests_text_output"))
        target_tokens.extend(["UGM:serializer:text", f"ANSWER:{function_text[:120]}"])

    if any(tok.startswith("PDB:") for tok in target_tokens):
        target_tokens.append("UGM:serializer:pdb")
    if row.get("oracle") or row.get("energy") is not None or row.get("forces"):
        target_tokens.append("UGM:oracle:uma_feedback")
    target_tokens = list(dict.fromkeys(target_tokens))

    ex = GraphExample(
        id=f"{dataset_name}_{idx}_{_stable_id(prompt + protein_sequence[:32] + selfies + dna_sequence[:32] + rna_sequence[:32] + smiles)}",
        task="multimodal_graph_to_graph",
        nodes=nodes,
        edges=edges,
        target_tokens=target_tokens,
        metadata={
            "source_dataset": dataset_name,
            "task": task_value,
            "temperature": round(temp_k, 4) if temp_k is not None else None,
            "temperature_anchor": temp_anchor,
            "temperature_range_k": [TEMPERATURE_MIN_K, TEMPERATURE_MAX_K],
            "modalities": _modalities_from_tokens(target_tokens),
            "atom_count": len(atom_ids),
            "bond_type_count": sum(1 for tok in target_tokens if tok.startswith("BOND:")),
            "frame_count": len(_frame_list(row)),
            "molecular_input_policy": molecular_input_policy,
            "geometric_features_enabled": geometric_features,
            "ignored_structure_fields": structure_fields_present(original_row) if molecular_input_policy == SEQUENCE_ONLY else [],
            "license_warning": "check upstream molecular/science licenses before scaling",
        },
    )
    ex.decoder_orders = build_orders(ex.target_tokens, seed=idx)
    return ex


def _modalities_from_tokens(tokens: Iterable[str]) -> list[str]:
    modalities = []
    for token in tokens:
        if token.startswith("UGM:modality:"):
            modalities.append(token.rsplit(":", 1)[1])
    return sorted(set(modalities))


def records_to_multimodel_pdb(atoms: list[dict[str, Any]], frames: list[list[list[float]]], bonds: list[dict[str, Any]] | None = None) -> str:
    rows: list[str] = []
    for frame_idx, coords in enumerate(frames, start=1):
        rows.append(f"MODEL     {frame_idx:4d}")
        for atom_idx, atom in enumerate(atoms, start=1):
            element = _atom_element(atom).strip()[:2].rjust(2)
            name = _as_str(atom.get("name") if isinstance(atom, dict) else element).strip()[:4].rjust(4)
            residue = _as_str(atom.get("residue") if isinstance(atom, dict) else "MOL").strip()[:3].rjust(3)
            chain = _as_str(atom.get("chain") if isinstance(atom, dict) else "A").strip()[:1] or "A"
            resseq = int(atom.get("residue_index", 1)) if isinstance(atom, dict) else 1
            coord = coords[atom_idx - 1] if atom_idx - 1 < len(coords) else [0.0, 0.0, 0.0]
            x, y, z = [float(v) for v in coord[:3]]
            rows.append(
                f"ATOM  {atom_idx:5d} {name} {residue} {chain}{resseq:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element}"
            )
        for bond in bonds or []:
            try:
                src = int(bond.get("src", bond.get("i", 0))) + 1
                dst = int(bond.get("dst", bond.get("j", 0))) + 1
            except Exception:
                continue
            rows.append(f"CONECT{src:5d}{dst:5d}")
        rows.append("ENDMDL")
    rows.append("END")
    return "\n".join(rows) + "\n"


def iter_synthetic_multimodal_examples(count: int = 16, seed: int = 17) -> Iterable[dict[str, Any]]:
    residues = "ACDEFGHIKLMNPQRSTVWY"
    selfies_samples = ["[C][O]", "[C][=O][O]", "[N][C][C][O]", "[C][C][Branch1][C][O]"]
    dna_samples = ["ATGCGTAC", "GGATCCGA", "TTAACCGG"]
    rna_samples = ["AUGCGUAC", "GGACUUGA", "UUCGGAAC"]
    for i in range(count):
        length = 6 + (i % 7)
        protein = "".join(residues[(i + j) % len(residues)] for j in range(length))
        atoms = [
            {"element": "N", "name": "N", "residue": "GLY", "residue_index": 1},
            {"element": "C", "name": "CA", "residue": "GLY", "residue_index": 1},
            {"element": "C", "name": "C", "residue": "GLY", "residue_index": 1},
            {"element": "O", "name": "O", "residue": "GLY", "residue_index": 1},
        ]
        base = i * 0.05
        frames = [
            [[0.0 + base, 0.0, 0.0], [1.45 + base, 0.1, 0.0], [2.2 + base, 1.1, 0.0], [3.2 + base, 1.0, 0.1]],
            [[0.1 + base, 0.0, 0.0], [1.50 + base, 0.2, 0.0], [2.25 + base, 1.0, 0.1], [3.25 + base, 1.1, 0.0]],
        ]
        yield {
            "prompt": "Generate an all-atom graph and function summary for a mixed biomolecular input.",
            "task": "conformer_trajectory" if i % 2 else "structure_generation",
            "protein_sequence": protein,
            "selfies": selfies_samples[i % len(selfies_samples)],
            "dna_sequence": dna_samples[i % len(dna_samples)] if i % 3 == 0 else "",
            "rna_sequence": rna_samples[i % len(rna_samples)] if i % 3 == 1 else "",
            "temperature": TEMPERATURE_MIN_K + ((i * 17.5) % (TEMPERATURE_MAX_K - TEMPERATURE_MIN_K)),
            "atoms": atoms,
            "bonds": [{"src": 0, "dst": 1, "bond_type": "peptide"}, {"src": 1, "dst": 2, "bond_type": "single"}, {"src": 2, "dst": 3, "bond_type": "double"}],
            "distances": [[0, 1, 1.45], [1, 2, 1.52], [0, 3, 3.2]],
            "frames": frames,
            "energy": -5.0 + i,
            "forces": [[0.01, 0.0, 0.0], [-0.1, 0.2, 0.0], [0.0, -0.3, 0.1], [0.0, 0.0, -0.2]],
            "function_description": "Toy mixed-modal example for graph-to-graph structure and oracle feedback training.",
        }
