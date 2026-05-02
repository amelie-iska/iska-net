from __future__ import annotations

import json
import os
import random
import re
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from iska_reasoner.data.vocab import GraphVocab
from iska_reasoner.graph.schema import GraphExample, graph_source_numeric_features, graph_source_tokens
from iska_reasoner.topology import topology_feature_tensor


KIND_TO_ID = {
    "pad": 0,
    "special": 1,
    "node": 2,
    "edge": 3,
    "position": 4,
    "target": 5,
}


NUMERIC_NODE_TYPES = {
    "coordinate_3d",
    "molecule_property",
    "material_property",
    "audio_duration",
    "binding_affinity",
    "assay_value",
    "protein_coordinate",
    "ligand_coordinate",
    "temperature",
}

COORD_TOKEN_RE = re.compile(r"^COORD:f(?P<frame>\d+):a(?P<atom>\d+):(?P<axis>[xyz]):")
AXIS_TO_INDEX = {"x": 0, "y": 1, "z": 2}
COORDINATE_NODE_TYPES = {"coordinate_3d", "protein_coordinate", "ligand_coordinate"}
COMMON_ATOMIC_NUMBERS = {
    "H": 1,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Br": 35,
    "I": 53,
}
PROTEIN_BACKBONE_SYMBOLS = ("N", "C", "C", "O")
INTERNAL_COORD_TYPES = [
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
    "ligand_torsion",
]
INTERNAL_COORD_TYPE_TO_ID = {name: idx + 1 for idx, name in enumerate(INTERNAL_COORD_TYPES)}
PROTEIN_INTERNAL_COORD_TYPES = ("protein_phi", "protein_psi", "protein_omega", "sidechain_chi")
NUCLEIC_INTERNAL_COORD_TYPES = ("rna_alpha", "rna_beta", "rna_gamma", "rna_delta", "rna_epsilon", "rna_zeta", "glycosidic_chi", "sugar_pucker")


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _first_int(features: dict[str, Any], names: tuple[str, ...], default: int | None = None) -> int | None:
    for name in names:
        if name in features:
            parsed = _int_or_none(features.get(name))
            if parsed is not None:
                return parsed
    return default


def _xyz_from_node(node: Any) -> list[float] | None:
    xyz: list[float] = []
    for axis in ("x", "y", "z"):
        value = _float_or_none(node.features.get(axis))
        if value is None:
            return None
        xyz.append(value)
    return xyz


def _xyz_from_text(text: str) -> list[float] | None:
    values = [_float_or_none(match) for match in re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", text)]
    values = [value for value in values if value is not None]
    if len(values) < 3:
        return None
    return [float(values[0]), float(values[1]), float(values[2])]


def _coordinate_node_lookup(example: GraphExample) -> dict[tuple[int, int], list[float]]:
    lookup: dict[tuple[int, int], list[float]] = {}
    for node in example.nodes:
        if node.type not in COORDINATE_NODE_TYPES:
            continue
        atom_index = _first_int(
            node.features,
            ("index", "atom_index", "atom", "slot", "slot_index", "residue_index"),
        )
        if atom_index is None:
            continue
        frame = _first_int(node.features, ("frame", "frame_index", "model"), default=0)
        if frame is None:
            frame = 0
        xyz = _xyz_from_node(node) or _xyz_from_text(str(node.value))
        if xyz is None:
            continue
        lookup[(frame, atom_index)] = xyz
    return lookup


def coordinate_targets_by_index(example: GraphExample) -> tuple[list[list[float]], list[list[float]]]:
    """Return continuous coordinate side-channel targets aligned to target tokens.

    The symbolic graph-token objective remains primary: each coordinate record
    is still emitted as a normal target token such as ``COORD:f0:a3:x:pos_near``.
    This helper supplies an optional continuous target for the same `<POS>` slot
    when the graph contains explicit coordinate records and the model config has
    enabled the coordinate head.
    """
    targets = [[0.0, 0.0, 0.0] for _ in example.target_tokens]
    masks = [[0.0, 0.0, 0.0] for _ in example.target_tokens]
    lookup = _coordinate_node_lookup(example)

    for target_idx, token in enumerate(example.target_tokens):
        match = COORD_TOKEN_RE.match(token)
        if match is None:
            continue
        frame = int(match.group("frame"))
        atom = int(match.group("atom"))
        axis = match.group("axis")
        xyz = lookup.get((frame, atom))
        if xyz is None:
            xyz = lookup.get((0, atom))
        if xyz is None:
            continue
        axis_idx = AXIS_TO_INDEX[axis]
        targets[target_idx] = [float(xyz[0]), float(xyz[1]), float(xyz[2])]
        masks[target_idx][axis_idx] = 1.0

    for item in example.coordinate_targets:
        target_indices: list[int] = []
        if "target_index" in item:
            parsed = _int_or_none(item.get("target_index"))
            if parsed is not None and 0 <= parsed < len(example.target_tokens):
                target_indices.append(parsed)
        elif "token" in item:
            target_indices.extend(
                idx for idx, token in enumerate(example.target_tokens) if token == str(item["token"])
            )
        if not target_indices:
            continue

        coord = item.get("coord", item.get("xyz", item.get("value")))
        item_mask = item.get("mask")
        parsed_coord: list[float] | None = None
        if isinstance(coord, dict):
            parsed_coord = [_float_or_none(coord.get(axis)) or 0.0 for axis in ("x", "y", "z")]
        elif isinstance(coord, (list, tuple)) and len(coord) >= 3:
            values = [_float_or_none(coord[idx]) for idx in range(3)]
            parsed_coord = [float(value) if value is not None else 0.0 for value in values]

        axis = str(item.get("axis", "")).lower()
        axis_idx = AXIS_TO_INDEX.get(axis)
        scalar_value = _float_or_none(item.get("axis_value", item.get("scalar")))
        if scalar_value is None and axis_idx is not None:
            scalar_value = _float_or_none(item.get("value"))

        for idx in target_indices:
            if parsed_coord is not None:
                targets[idx] = parsed_coord
                if isinstance(item_mask, (list, tuple)) and len(item_mask) >= 3:
                    masks[idx] = [1.0 if float(item_mask[j]) else 0.0 for j in range(3)]
                else:
                    masks[idx] = [1.0, 1.0, 1.0]
            if axis_idx is not None and scalar_value is not None:
                targets[idx][axis_idx] = float(scalar_value)
                masks[idx][axis_idx] = 1.0
    return targets, masks


def _candidate_smiles(example: GraphExample) -> str:
    metadata = example.metadata or {}
    for key in ("smiles", "SMILES", "canonical_smiles"):
        value = metadata.get(key)
        if value:
            return str(value)
    for token in example.target_tokens:
        if token.startswith("SMILES:"):
            return token.split(":", 1)[1]
    for node in example.nodes:
        if node.type == "smiles" and node.value:
            return str(node.value)
    return ""


def _protein_backbone_symbols(example: GraphExample, max_atoms: int) -> list[str]:
    """Return a coarse protein backbone atom list from sequence records only."""
    if max_atoms <= 0:
        return []
    sequence = ""
    residues: list[str] = []
    metadata = example.metadata or {}
    for key in ("protein_sequence", "sequence", "aa_sequence"):
        value = metadata.get(key)
        if value:
            sequence = str(value)
            break
    for node in example.nodes:
        if node.type == "amino_acid" and node.value:
            residues.append(str(node.value).strip()[:1])
        elif node.type in {"protein_sequence", "translated_protein_sequence"} and node.value and not sequence:
            sequence = str(node.value)
    residue_count = len(residues) if residues else sum(1 for char in sequence if char.isalpha())
    symbols: list[str] = []
    for _ in range(residue_count):
        symbols.extend(PROTEIN_BACKBONE_SYMBOLS)
        if len(symbols) >= max_atoms:
            return symbols[:max_atoms]
    return symbols[:max_atoms]


def uma_coordinate_symbols(example: GraphExample, max_atoms: int) -> list[str]:
    """Derive UMA coordinate-query atom symbols without reading structure labels."""
    if max_atoms <= 0:
        return []
    symbols: list[str] = []
    for node in example.nodes:
        if node.type not in {"atom", "atom_symbol"}:
            continue
        raw = node.features.get("element") or node.features.get("symbol") or node.value
        symbol = str(raw).strip()
        if symbol:
            symbols.append(symbol)
        if len(symbols) >= max_atoms:
            return symbols[:max_atoms]

    protein_symbols = _protein_backbone_symbols(example, max_atoms)
    if protein_symbols:
        return protein_symbols[:max_atoms]

    smiles = _candidate_smiles(example)
    if not smiles:
        return []
    try:
        from rdkit import Chem
        from rdkit import RDLogger

        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []
        return [atom.GetSymbol() for atom in mol.GetAtoms()][:max_atoms]
    except Exception:
        return []


def internal_coordinate_actions(example: GraphExample, max_actions: int) -> list[dict[str, Any]]:
    """Derive internal-coordinate action slots from symbolic graph records only."""
    if max_actions <= 0:
        return []
    actions: list[dict[str, Any]] = []
    residues = [node for node in example.nodes if node.type == "amino_acid"]
    bases = [node for node in example.nodes if node.type in {"rna_base", "dna_base"}]
    ligand_tokens = [node for node in example.nodes if node.type in {"selfies_token", "atom", "smiles_char"}]

    for residue_idx, _node in enumerate(residues):
        for coord_type in PROTEIN_INTERNAL_COORD_TYPES:
            actions.append({"type": coord_type, "residue_index": residue_idx, "component": "protein"})
            if len(actions) >= max_actions:
                return actions
    for base_idx, node in enumerate(bases):
        component = "rna" if node.type == "rna_base" else "dna"
        for coord_type in NUCLEIC_INTERNAL_COORD_TYPES:
            actions.append({"type": coord_type, "residue_index": base_idx, "component": component})
            if len(actions) >= max_actions:
                return actions
    torsion_count = max(0, min(len(ligand_tokens) - 1, max_actions - len(actions)))
    for torsion_idx in range(torsion_count):
        actions.append({"type": "ligand_torsion", "residue_index": torsion_idx, "component": "ligand"})
    return actions[:max_actions]


def extract_numeric_values(example: GraphExample, max_values: int) -> tuple[list[float], list[float]]:
    values: list[float] = []
    if max_values <= 0:
        return [], []
    for node in example.nodes:
        if node.type not in NUMERIC_NODE_TYPES and not any(isinstance(v, (int, float)) for v in node.features.values()):
            continue
        for value in node.features.values():
            if isinstance(value, (int, float)) and len(values) < max_values:
                values.append(float(value))
        if len(values) >= max_values:
            break
        for match in re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", node.value):
            values.append(float(match))
            if len(values) >= max_values:
                break
        if len(values) >= max_values:
            break
    mask = [1.0] * len(values)
    if len(values) < max_values:
        values.extend([0.0] * (max_values - len(values)))
        mask.extend([0.0] * (max_values - len(mask)))
    return values[:max_values], mask[:max_values]


@dataclass
class EncodedExample:
    token_ids: list[int]
    kind_ids: list[int]
    endpoint_ids: list[tuple[int, int]]
    identifier_ids: list[int]
    numeric_features: list[list[float]]
    coordinate_targets: list[list[float]]
    coordinate_mask: list[list[float]]
    uma_coordinate_query_mask: list[float]
    uma_coordinate_symbols: list[str]
    internal_coordinate_query_mask: list[float]
    internal_coordinate_type_ids: list[int]
    internal_coordinate_residue_indices: list[int]
    internal_coordinate_types: list[str]
    slot_ids: list[int]
    labels: list[int]
    task: str
    example_id: str


class GraphJsonlDataset(Dataset[GraphExample]):
    def __init__(self, path: str | Path, preload: bool = False, transform: Callable[[GraphExample], GraphExample] | None = None):
        self.path = Path(path)
        self.preload = preload
        self.transform = transform
        self._handle = None
        self._handle_pid: int | None = None
        self.examples: list[GraphExample] | None = None
        self.offsets: array[int] = array("Q")
        if preload:
            from iska_reasoner.utils.io import read_jsonl

            self.examples = [GraphExample.from_dict(row) for row in read_jsonl(self.path)]
        else:
            self._build_offsets()
        if len(self) == 0:
            raise ValueError(f"No examples in {self.path}")

    def _offset_cache_paths(self) -> tuple[Path, Path]:
        return (
            self.path.with_name(f"{self.path.name}.offsets.u64"),
            self.path.with_name(f"{self.path.name}.offsets.meta.json"),
        )

    def _load_offsets_cache(self, file_size: int, mtime_ns: int) -> bool:
        if os.environ.get("UGM_DISABLE_JSONL_OFFSET_CACHE") == "1":
            return False
        offsets_path, meta_path = self._offset_cache_paths()
        if not offsets_path.exists() or not meta_path.exists():
            return False
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if (
                meta.get("version") != 1
                or int(meta.get("file_size", -1)) != file_size
                or int(meta.get("mtime_ns", -1)) != mtime_ns
            ):
                return False
            offsets = array("Q")
            with offsets_path.open("rb") as handle:
                offsets.fromfile(handle, int(meta.get("rows", 0)))
        except (OSError, ValueError, json.JSONDecodeError, EOFError):
            return False
        if len(offsets) != int(meta.get("rows", -1)):
            return False
        self.offsets = offsets
        return True

    def _save_offsets_cache(self, file_size: int, mtime_ns: int) -> None:
        if os.environ.get("UGM_DISABLE_JSONL_OFFSET_CACHE") == "1":
            return
        offsets_path, meta_path = self._offset_cache_paths()
        offsets_tmp = offsets_path.with_suffix(offsets_path.suffix + ".tmp")
        meta_tmp = meta_path.with_suffix(meta_path.suffix + ".tmp")
        meta = {
            "version": 1,
            "source": self.path.name,
            "file_size": file_size,
            "mtime_ns": mtime_ns,
            "rows": len(self.offsets),
        }
        try:
            with offsets_tmp.open("wb") as handle:
                self.offsets.tofile(handle)
            meta_tmp.write_text(json.dumps(meta, sort_keys=True), encoding="utf-8")
            offsets_tmp.replace(offsets_path)
            meta_tmp.replace(meta_path)
        except OSError:
            for tmp in (offsets_tmp, meta_tmp):
                try:
                    tmp.unlink()
                except OSError:
                    pass

    def _build_offsets(self) -> None:
        stat = self.path.stat()
        file_size = stat.st_size
        mtime_ns = stat.st_mtime_ns
        if self._load_offsets_cache(file_size=file_size, mtime_ns=mtime_ns):
            return
        show_progress = file_size >= 64 * 1024 * 1024
        with self.path.open("rb") as handle:
            with tqdm(
                total=file_size,
                desc=f"index/{self.path.name}",
                unit="B",
                unit_scale=True,
                disable=not show_progress,
                leave=False,
            ) as pbar:
                while True:
                    offset = handle.tell()
                    line = handle.readline()
                    if not line:
                        break
                    if line.strip():
                        self.offsets.append(offset)
                    pbar.update(len(line))
        self._save_offsets_cache(file_size=file_size, mtime_ns=mtime_ns)

    def _row_at(self, index: int) -> dict[str, Any]:
        pid = os.getpid()
        if self._handle is None or self._handle.closed or self._handle_pid != pid:
            if self._handle is not None and not self._handle.closed:
                self._handle.close()
            self._handle = self.path.open("rb")
            self._handle_pid = pid
        self._handle.seek(int(self.offsets[index]))
        line = self._handle.readline()
        return json.loads(line.decode("utf-8"))

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_handle"] = None
        state["_handle_pid"] = None
        return state

    def __del__(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle is not None and not handle.closed:
            handle.close()

    def __len__(self) -> int:
        return len(self.examples) if self.examples is not None else len(self.offsets)

    def __getitem__(self, index: int) -> GraphExample:
        if self.examples is not None:
            example = self.examples[index]
        else:
            example = GraphExample.from_dict(self._row_at(index))
        return self.transform(example) if self.transform is not None else example


def select_order(example: GraphExample, rng: random.Random, mode: str = "sample") -> list[int]:
    n = len(example.target_tokens)
    if n == 0:
        return []
    if not example.decoder_orders:
        return list(range(n))
    if mode == "first":
        return example.decoder_orders[0]
    if mode == "reverse":
        return list(reversed(example.decoder_orders[0]))
    return rng.choice(example.decoder_orders)


def encode_example(
    example: GraphExample,
    vocab: GraphVocab,
    order: list[int],
    max_source_tokens: int,
    max_target_tokens: int,
    max_seq_len: int | None = None,
    max_uma_coordinate_atoms: int = 0,
    max_internal_coordinate_actions: int = 0,
) -> EncodedExample:
    source_tokens, source_kinds, endpoints, identifiers = graph_source_tokens(example)
    source_numeric_features = graph_source_numeric_features(example)
    query_symbols = uma_coordinate_symbols(example, max_uma_coordinate_atoms)
    internal_actions = internal_coordinate_actions(example, max_internal_coordinate_actions)
    if max_seq_len is not None:
        target_count = min(len(order), max_target_tokens)
        reserved = 1 + 2 * target_count
        query_budget = max(0, max_seq_len - reserved - 1)
        internal_budget = min(len(internal_actions), query_budget)
        coord_budget = min(len(query_symbols), max(0, query_budget - internal_budget))
        source_budget = max(1, min(max_source_tokens, max_seq_len - reserved - internal_budget - coord_budget))
        internal_actions = internal_actions[:internal_budget]
        query_symbols = query_symbols[:coord_budget]
    else:
        source_budget = max_source_tokens
    source_tokens = source_tokens[:source_budget]
    source_kinds = source_kinds[:source_budget]
    endpoints = endpoints[:source_budget]
    identifiers = identifiers[:source_budget]
    source_numeric_features = source_numeric_features[:source_budget]
    per_target_coords, per_target_coord_mask = coordinate_targets_by_index(example)

    token_ids = [vocab.encode(tok) for tok in source_tokens]
    kind_ids = [KIND_TO_ID.get(kind, KIND_TO_ID["special"]) for kind in source_kinds]
    endpoint_ids = list(endpoints)
    identifier_ids = list(identifiers)
    numeric_features = list(source_numeric_features)
    coordinate_targets = [[0.0, 0.0, 0.0] for _ in token_ids]
    coordinate_mask = [[0.0, 0.0, 0.0] for _ in token_ids]
    uma_coordinate_query_mask = [0.0 for _ in token_ids]
    internal_coordinate_query_mask = [0.0 for _ in token_ids]
    internal_coordinate_type_ids = [0 for _ in token_ids]
    internal_coordinate_residue_indices = [-1 for _ in token_ids]
    slot_ids = [0 for _ in token_ids]
    labels = [-100 for _ in token_ids]

    internal_count = max(1, len(internal_actions))
    for action_idx, action in enumerate(internal_actions):
        coord_type = str(action["type"])
        type_id = INTERNAL_COORD_TYPE_TO_ID.get(coord_type, 0)
        residue_index = int(action.get("residue_index", -1))
        token_ids.append(vocab.encode(f"INTERNAL_COORD_QUERY:{coord_type}"))
        kind_ids.append(KIND_TO_ID["special"])
        endpoint_ids.append((0, 0))
        identifier_ids.append(0)
        numeric_features.append([2.0, action_idx / internal_count, type_id / 32.0, max(0, residue_index) / 512.0])
        coordinate_targets.append([0.0, 0.0, 0.0])
        coordinate_mask.append([0.0, 0.0, 0.0])
        uma_coordinate_query_mask.append(0.0)
        internal_coordinate_query_mask.append(1.0)
        internal_coordinate_type_ids.append(type_id)
        internal_coordinate_residue_indices.append(residue_index)
        slot_ids.append(min(action_idx + 1, max_target_tokens))
        labels.append(-100)

    query_count = max(1, len(query_symbols))
    for atom_idx, symbol in enumerate(query_symbols):
        token_ids.append(vocab.encode(f"UMA_COORD_QUERY:{symbol}"))
        kind_ids.append(KIND_TO_ID["special"])
        endpoint_ids.append((0, 0))
        identifier_ids.append(0)
        atomic_num = COMMON_ATOMIC_NUMBERS.get(symbol, 0)
        numeric_features.append([1.0, atom_idx / query_count, atomic_num / 100.0, query_count / 128.0])
        coordinate_targets.append([0.0, 0.0, 0.0])
        coordinate_mask.append([0.0, 0.0, 0.0])
        uma_coordinate_query_mask.append(1.0)
        internal_coordinate_query_mask.append(0.0)
        internal_coordinate_type_ids.append(0)
        internal_coordinate_residue_indices.append(-1)
        slot_ids.append(min(atom_idx + 1, max_target_tokens))
        labels.append(-100)

    token_ids.append(vocab.encode("<SEP>"))
    kind_ids.append(KIND_TO_ID["special"])
    endpoint_ids.append((0, 0))
    identifier_ids.append(0)
    numeric_features.append([0.0] * 4)
    coordinate_targets.append([0.0, 0.0, 0.0])
    coordinate_mask.append([0.0, 0.0, 0.0])
    uma_coordinate_query_mask.append(0.0)
    internal_coordinate_query_mask.append(0.0)
    internal_coordinate_type_ids.append(0)
    internal_coordinate_residue_indices.append(-1)
    slot_ids.append(0)
    labels.append(-100)

    for reveal_rank, target_idx in enumerate(order[:max_target_tokens], start=1):
        token = example.target_tokens[target_idx]
        token_ids.append(vocab.encode("<POS>"))
        kind_ids.append(KIND_TO_ID["position"])
        endpoint_ids.append((0, 0))
        identifier_ids.append(0)
        numeric_features.append([0.0] * 4)
        coordinate_targets.append(per_target_coords[target_idx])
        coordinate_mask.append(per_target_coord_mask[target_idx])
        uma_coordinate_query_mask.append(0.0)
        internal_coordinate_query_mask.append(0.0)
        internal_coordinate_type_ids.append(0)
        internal_coordinate_residue_indices.append(-1)
        slot_ids.append(min(target_idx + 1, max_target_tokens))
        labels.append(vocab.encode(token))

        token_ids.append(vocab.encode(token))
        kind_ids.append(KIND_TO_ID["target"])
        endpoint_ids.append((0, 0))
        identifier_ids.append(0)
        numeric_features.append([0.0] * 4)
        coordinate_targets.append([0.0, 0.0, 0.0])
        coordinate_mask.append([0.0, 0.0, 0.0])
        uma_coordinate_query_mask.append(0.0)
        internal_coordinate_query_mask.append(0.0)
        internal_coordinate_type_ids.append(0)
        internal_coordinate_residue_indices.append(-1)
        slot_ids.append(min(target_idx + 1, max_target_tokens))
        labels.append(-100)

    return EncodedExample(
        token_ids=token_ids,
        kind_ids=kind_ids,
        endpoint_ids=endpoint_ids,
        identifier_ids=identifier_ids,
        numeric_features=numeric_features,
        coordinate_targets=coordinate_targets,
        coordinate_mask=coordinate_mask,
        uma_coordinate_query_mask=uma_coordinate_query_mask,
        uma_coordinate_symbols=query_symbols,
        internal_coordinate_query_mask=internal_coordinate_query_mask,
        internal_coordinate_type_ids=internal_coordinate_type_ids,
        internal_coordinate_residue_indices=internal_coordinate_residue_indices,
        internal_coordinate_types=[str(action["type"]) for action in internal_actions],
        slot_ids=slot_ids,
        labels=labels,
        task=example.task,
        example_id=example.id,
    )


class RandomOrderCollator:
    def __init__(
        self,
        vocab: GraphVocab,
        max_source_tokens: int = 128,
        max_target_tokens: int = 64,
        max_seq_len: int = 256,
        max_numeric_targets: int = 0,
        max_uma_coordinate_atoms: int = 0,
        max_internal_coordinate_actions: int = 0,
        order_mode: str = "sample",
        seed: int = 17,
    ):
        self.vocab = vocab
        self.max_source_tokens = max_source_tokens
        self.max_target_tokens = max_target_tokens
        self.max_seq_len = max_seq_len
        self.max_numeric_targets = max_numeric_targets
        self.max_uma_coordinate_atoms = max_uma_coordinate_atoms
        self.max_internal_coordinate_actions = max_internal_coordinate_actions
        self.order_mode = order_mode
        self.rng = random.Random(seed)

    def __call__(self, examples: list[GraphExample]) -> dict[str, Any]:
        encoded = [
            encode_example(
                ex,
                self.vocab,
                select_order(ex, self.rng, self.order_mode),
                self.max_source_tokens,
                self.max_target_tokens,
                self.max_seq_len,
                self.max_uma_coordinate_atoms,
                self.max_internal_coordinate_actions,
            )
            for ex in examples
        ]
        seq_len = min(max(len(ex.token_ids) for ex in encoded), self.max_seq_len)
        batch = len(encoded)

        input_ids = torch.full((batch, seq_len), self.vocab.pad_id, dtype=torch.long)
        kind_ids = torch.zeros((batch, seq_len), dtype=torch.long)
        slot_ids = torch.zeros((batch, seq_len), dtype=torch.long)
        endpoint_ids = torch.zeros((batch, seq_len, 2), dtype=torch.long)
        identifier_ids = torch.zeros((batch, seq_len), dtype=torch.long)
        source_numeric_features = torch.zeros((batch, seq_len, 4), dtype=torch.float32)
        coordinate_targets = torch.zeros((batch, seq_len, 3), dtype=torch.float32)
        coordinate_mask = torch.zeros((batch, seq_len, 3), dtype=torch.float32)
        uma_coordinate_query_mask = torch.zeros((batch, seq_len), dtype=torch.float32)
        internal_coordinate_query_mask = torch.zeros((batch, seq_len), dtype=torch.float32)
        internal_coordinate_type_ids = torch.zeros((batch, seq_len), dtype=torch.long)
        internal_coordinate_residue_indices = torch.full((batch, seq_len), -1, dtype=torch.long)
        labels = torch.full((batch, seq_len), -100, dtype=torch.long)
        attention_mask = torch.zeros((batch, seq_len), dtype=torch.bool)
        topology_features = topology_feature_tensor(examples)
        numeric_targets = torch.zeros((batch, self.max_numeric_targets), dtype=torch.float32)
        numeric_mask = torch.zeros((batch, self.max_numeric_targets), dtype=torch.float32)

        for row, ex in enumerate(encoded):
            n = min(len(ex.token_ids), seq_len)
            input_ids[row, :n] = torch.tensor(ex.token_ids[:n], dtype=torch.long)
            kind_ids[row, :n] = torch.tensor(ex.kind_ids[:n], dtype=torch.long)
            slot_ids[row, :n] = torch.tensor(ex.slot_ids[:n], dtype=torch.long)
            endpoint_ids[row, :n] = torch.tensor(ex.endpoint_ids[:n], dtype=torch.long)
            identifier_ids[row, :n] = torch.tensor(ex.identifier_ids[:n], dtype=torch.long)
            source_numeric_features[row, :n] = torch.tensor(ex.numeric_features[:n], dtype=torch.float32)
            coordinate_targets[row, :n] = torch.tensor(ex.coordinate_targets[:n], dtype=torch.float32)
            coordinate_mask[row, :n] = torch.tensor(ex.coordinate_mask[:n], dtype=torch.float32)
            uma_coordinate_query_mask[row, :n] = torch.tensor(ex.uma_coordinate_query_mask[:n], dtype=torch.float32)
            internal_coordinate_query_mask[row, :n] = torch.tensor(ex.internal_coordinate_query_mask[:n], dtype=torch.float32)
            internal_coordinate_type_ids[row, :n] = torch.tensor(ex.internal_coordinate_type_ids[:n], dtype=torch.long)
            internal_coordinate_residue_indices[row, :n] = torch.tensor(ex.internal_coordinate_residue_indices[:n], dtype=torch.long)
            labels[row, :n] = torch.tensor(ex.labels[:n], dtype=torch.long)
            attention_mask[row, :n] = True
            values, value_mask = extract_numeric_values(examples[row], self.max_numeric_targets)
            if values:
                numeric_targets[row] = torch.tensor(values, dtype=torch.float32)
                numeric_mask[row] = torch.tensor(value_mask, dtype=torch.float32)

        # Causal mask: True means disallowed for nn.Transformer.
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        return {
            "input_ids": input_ids,
            "kind_ids": kind_ids,
            "slot_ids": slot_ids,
            "endpoint_ids": endpoint_ids,
            "identifier_ids": identifier_ids,
            "source_numeric_features": source_numeric_features,
            "coordinate_targets": coordinate_targets,
            "coordinate_mask": coordinate_mask,
            "uma_coordinate_query_mask": uma_coordinate_query_mask,
            "internal_coordinate_query_mask": internal_coordinate_query_mask,
            "internal_coordinate_type_ids": internal_coordinate_type_ids,
            "internal_coordinate_residue_indices": internal_coordinate_residue_indices,
            "labels": labels,
            "attention_mask": attention_mask,
            "causal_mask": causal_mask,
            "topology_features": topology_features,
            "numeric_targets": numeric_targets,
            "numeric_mask": numeric_mask,
            "tasks": [ex.task for ex in examples],
            "example_ids": [ex.id for ex in examples],
            "uma_coordinate_symbols": [ex.uma_coordinate_symbols for ex in encoded],
            "internal_coordinate_types": [ex.internal_coordinate_types for ex in encoded],
            "examples": examples,
        }
