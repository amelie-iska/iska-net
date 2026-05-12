from __future__ import annotations

import math
from typing import Any

from iska_reasoner.graph.schema import GraphExample


AA3 = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "E": "GLU",
    "Q": "GLN",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
}

AA_HEAVY_ATOM_NAMES = {
    "A": ("N", "CA", "C", "O", "CB"),
    "R": ("N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"),
    "N": ("N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"),
    "D": ("N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"),
    "C": ("N", "CA", "C", "O", "CB", "SG"),
    "E": ("N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"),
    "Q": ("N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"),
    "G": ("N", "CA", "C", "O"),
    "H": ("N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"),
    "I": ("N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"),
    "L": ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"),
    "K": ("N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"),
    "M": ("N", "CA", "C", "O", "CB", "CG", "SD", "CE"),
    "F": ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    "P": ("N", "CA", "C", "O", "CB", "CG", "CD"),
    "S": ("N", "CA", "C", "O", "CB", "OG"),
    "T": ("N", "CA", "C", "O", "CB", "OG1", "CG2"),
    "W": ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"),
    "Y": ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"),
    "V": ("N", "CA", "C", "O", "CB", "CG1", "CG2"),
}
AA1_FROM_AA3 = {value: key for key, value in AA3.items()}

NUCLEIC_BACKBONE_NAMES_RNA = ("P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'")
NUCLEIC_BACKBONE_NAMES_DNA = ("P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'")
NUCLEIC_BASE_NAMES = {
    "A": ("N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4"),
    "C": ("N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"),
    "G": ("N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4"),
    "T": ("N1", "C2", "O2", "N3", "C4", "O4", "C5", "C7", "C6"),
    "U": ("N1", "C2", "O2", "N3", "C4", "O4", "C5", "C6"),
}


def _element_from_atom_name(name: str, residue: str | None = None) -> str:
    clean = "".join(ch for ch in str(name).strip() if ch.isalpha())
    if not clean:
        return "C"
    upper = clean.upper()
    if upper.startswith("CL"):
        return "Cl"
    if upper.startswith("BR"):
        return "Br"
    first = upper[0]
    if first in {"H", "B", "C", "N", "O", "F", "P", "S", "I"}:
        return first
    return "C"


def _sequence_from_nodes(example: GraphExample, root_type: str, item_type: str, metadata_keys: tuple[str, ...]) -> str:
    metadata = example.metadata or {}
    for key in metadata_keys:
        value = metadata.get(key)
        if value:
            return "".join(ch for ch in str(value).upper() if ch.isalpha())
    residues: list[str] = []
    for node in example.nodes:
        if node.type == item_type and node.value:
            residues.append(str(node.value).strip().upper()[:1])
        elif node.type == root_type and node.value and not residues:
            return "".join(ch for ch in str(node.value).upper() if ch.isalpha())
    return "".join(residues)


def _selfies_symbols(example: GraphExample) -> list[str]:
    symbols: list[str] = []
    for node in example.nodes:
        if node.type != "selfies_token":
            continue
        raw = str(node.value or "").strip().strip("[]").lstrip("=#")
        if raw:
            symbols.append(raw[:2] if raw[:2] in {"Cl", "Br"} else raw[:1])
    return symbols


def _smiles_symbols(example: GraphExample) -> list[str]:
    smiles = ""
    for node in example.nodes:
        if node.type == "smiles" and node.value:
            smiles = str(node.value)
            break
    if not smiles:
        return []
    try:
        from rdkit import Chem
        from rdkit import RDLogger

        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []
        mol = Chem.AddHs(mol)
        return [atom.GetSymbol() for atom in mol.GetAtoms()]
    except Exception:
        return []


def derive_full_atom_records(example: GraphExample, max_atoms: int = 5000, max_residues: int = 500) -> list[dict[str, Any]]:
    """Build full-size all-heavy atom records from sequence/string inputs only.

    These records are for generated structure-dynamics artifacts. They do not
    read supervised coordinates, contact maps, energies, forces, PDB/mmCIF/SDF,
    or trajectory labels.
    """
    atoms: list[dict[str, Any]] = []

    protein = _sequence_from_nodes(example, "protein_sequence", "amino_acid", ("protein_sequence", "sequence", "aa_sequence"))[:max_residues]
    for residue_index, aa in enumerate(protein, start=1):
        residue = AA3.get(aa, "UNK")
        for atom_name in AA_HEAVY_ATOM_NAMES.get(aa, ("N", "CA", "C", "O", "CB")):
            atoms.append(
                {
                    "element": _element_from_atom_name(atom_name, residue),
                    "name": atom_name,
                    "residue": residue,
                    "residue_index": residue_index,
                    "chain": "A",
                    "component": "protein",
                    "component_index": residue_index - 1,
                }
            )
            if len(atoms) >= max_atoms:
                return atoms

    dna = _sequence_from_nodes(example, "dna_sequence", "dna_base", ("dna_sequence", "dna"))[:max_residues]
    for base_index, base in enumerate(dna, start=1):
        residue = f"D{base if base in {'A', 'C', 'G', 'T'} else 'N'}"
        for atom_name in NUCLEIC_BACKBONE_NAMES_DNA + NUCLEIC_BASE_NAMES.get(base, NUCLEIC_BASE_NAMES["A"]):
            atoms.append(
                {
                    "element": _element_from_atom_name(atom_name, residue),
                    "name": atom_name,
                    "residue": residue,
                    "residue_index": base_index,
                    "chain": "D",
                    "component": "dna",
                    "component_index": base_index - 1,
                }
            )
            if len(atoms) >= max_atoms:
                return atoms

    rna = _sequence_from_nodes(example, "rna_sequence", "rna_base", ("rna_sequence", "rna"))[:max_residues]
    for base_index, base in enumerate(rna, start=1):
        residue = base if base in {"A", "C", "G", "U"} else "N"
        for atom_name in NUCLEIC_BACKBONE_NAMES_RNA + NUCLEIC_BASE_NAMES.get(base, NUCLEIC_BASE_NAMES["A"]):
            atoms.append(
                {
                    "element": _element_from_atom_name(atom_name, residue),
                    "name": atom_name,
                    "residue": residue,
                    "residue_index": base_index,
                    "chain": "R",
                    "component": "rna",
                    "component_index": base_index - 1,
                }
            )
            if len(atoms) >= max_atoms:
                return atoms

    ligand_symbols = _smiles_symbols(example) or _selfies_symbols(example)
    for atom_index, symbol in enumerate(ligand_symbols, start=1):
        atoms.append(
            {
                "element": symbol,
                "name": f"{symbol[:2]}{atom_index}",
                "residue": "LIG",
                "residue_index": 1,
                "chain": "L",
                "component": "ligand",
                "component_index": atom_index - 1,
            }
        )
        if len(atoms) >= max_atoms:
            return atoms

    return atoms


def _local_offset(atom: dict[str, Any]) -> tuple[float, float, float]:
    name = str(atom.get("name", "C")).upper()
    component = str(atom.get("component", "protein"))
    seed = sum((idx + 1) * ord(ch) for idx, ch in enumerate(name))
    if component == "ligand":
        idx = int(atom.get("component_index", 0))
        return (1.35 * (idx % 6), 1.15 * ((idx // 6) % 3), 0.85 * (idx // 18))
    if component in {"dna", "rna"}:
        if name in {"P", "OP1", "OP2", "O5'"}:
            return (-1.4 + 0.38 * (seed % 7), -0.75 + 0.08 * (seed % 5), 0.12 * ((seed // 5) % 5))
        if "'" in name:
            return (0.15 + 0.22 * (seed % 9), 0.1 + 0.12 * ((seed // 3) % 7), 0.45 + 0.16 * ((seed // 11) % 6))
        phase = (seed % 360) * math.pi / 180.0
        radius = 0.8 + 0.08 * (seed % 11)
        return (0.8 + radius * math.cos(phase), 1.25 + radius * math.sin(phase), 0.18 * ((seed // 17) % 7))
    offsets = {
        "N": (-1.15, 0.05, 0.0),
        "CA": (0.0, 0.0, 0.0),
        "C": (1.25, 0.2, 0.0),
        "O": (1.85, -0.6, 0.0),
        "CB": (-0.25, 1.45, 0.2),
    }
    if name in offsets:
        return offsets[name]
    residue = str(atom.get("residue", "")).upper()[:3]
    aa = AA1_FROM_AA3.get(residue)
    if aa and name in AA_HEAVY_ATOM_NAMES.get(aa, ()):
        side_idx = max(0, AA_HEAVY_ATOM_NAMES[aa].index(name) - 4)
        branch = (seed % 5) - 2
        phase = 0.73 * side_idx + 0.31 * branch
        return (
            -0.25 + 0.34 * math.sin(phase),
            1.45 + 1.18 * (side_idx + 1),
            0.20 + 0.42 * math.cos(phase) + 0.12 * branch,
        )
    ring = 0.85 + 0.09 * (seed % 9)
    phase = (seed % 360) * math.pi / 180.0
    layer = ((seed // 13) % 7) - 3
    return (-0.2 + ring * math.cos(phase), 1.65 + ring * math.sin(phase), 0.22 * layer)


def generated_initial_coordinates(atoms: list[dict[str, Any]], anchor_coordinates: list[list[float]] | None = None) -> list[list[float]]:
    coords: list[list[float]] = []
    for atom_index, atom in enumerate(atoms):
        component = str(atom.get("component", "protein"))
        residue_index = int(atom.get("residue_index", 1))
        comp_idx = int(atom.get("component_index", atom_index))
        if component == "protein":
            theta = residue_index * 1.75
            center = (2.15 * math.cos(theta), 2.15 * math.sin(theta), 1.52 * residue_index)
        elif component in {"dna", "rna"}:
            theta = residue_index * 0.62 + (0.3 if component == "rna" else 0.0)
            radius = 8.5 if component == "dna" else 6.5
            center = (radius * math.cos(theta), radius * math.sin(theta), 3.25 * residue_index)
        else:
            center = (0.0, 0.0, -3.0 + 0.35 * comp_idx)
        dx, dy, dz = _local_offset(atom)
        coords.append([center[0] + dx, center[1] + dy, center[2] + dz])

    if anchor_coordinates:
        usable = min(len(coords), len(anchor_coordinates))
        if usable:
            ax = sum(float(xyz[0]) for xyz in anchor_coordinates[:usable]) / usable
            ay = sum(float(xyz[1]) for xyz in anchor_coordinates[:usable]) / usable
            az = sum(float(xyz[2]) for xyz in anchor_coordinates[:usable]) / usable
            cx = sum(xyz[0] for xyz in coords[:usable]) / usable
            cy = sum(xyz[1] for xyz in coords[:usable]) / usable
            cz = sum(xyz[2] for xyz in coords[:usable]) / usable
            shift = (ax - cx, ay - cy, az - cz)
            coords = [[xyz[0] + shift[0], xyz[1] + shift[1], xyz[2] + shift[2]] for xyz in coords]
    return coords


def smooth_trajectory_frames(
    atoms: list[dict[str, Any]],
    initial_coords: list[list[float]],
    frame_count: int,
    temperature_k: float | None = None,
) -> list[list[list[float]]]:
    frame_count = max(1, int(frame_count))
    temp_scale = 0.75 + max(0.0, min(200.0, float(temperature_k or 300.0) - 280.0)) / 220.0
    amplitude = 0.18 * temp_scale
    frames: list[list[list[float]]] = []
    for frame_idx in range(frame_count):
        phase = 2.0 * math.pi * frame_idx / max(8, frame_count)
        frame: list[list[float]] = []
        for atom_idx, xyz in enumerate(initial_coords):
            atom = atoms[atom_idx]
            residue_index = int(atom.get("residue_index", atom_idx + 1))
            component_phase = {"protein": 0.0, "dna": 0.8, "rna": 1.2, "ligand": 1.7}.get(str(atom.get("component")), 0.4)
            slow = phase + residue_index * 0.071 + component_phase
            fast = phase * 1.7 + atom_idx * 0.113
            frame.append(
                [
                    float(xyz[0]) + amplitude * math.sin(slow) + 0.04 * math.sin(fast),
                    float(xyz[1]) + amplitude * math.cos(slow * 0.9) + 0.04 * math.cos(fast * 0.7),
                    float(xyz[2]) + 0.55 * amplitude * math.sin(slow * 0.73 + fast * 0.07),
                ]
            )
        frames.append(frame)
    return frames


def _sampled_pair_distances(coords: list[list[float]], max_points: int = 1200) -> tuple[float, int, int]:
    n = len(coords)
    if n < 2:
        return 999.0, 0, 0
    stride = max(1, math.ceil(n / max_points))
    sample = coords[::stride]
    min_dist2 = float("inf")
    clashes = 0
    pairs = 0
    for i in range(len(sample)):
        xi, yi, zi = sample[i]
        for j in range(i + 1, len(sample)):
            xj, yj, zj = sample[j]
            dist2 = (xi - xj) ** 2 + (yi - yj) ** 2 + (zi - zj) ** 2
            if dist2 < min_dist2:
                min_dist2 = dist2
            if dist2 < 0.75**2:
                clashes += 1
            pairs += 1
    return math.sqrt(min_dist2), clashes, pairs


def high_quality_trajectory_score(
    atoms: list[dict[str, Any]],
    frames: list[list[list[float]]],
    *,
    target_frames: int = 64,
    expected_residues: int | None = None,
) -> dict[str, Any]:
    if not atoms or not frames:
        return {"long_hq_score": 0.0, "score_profile": "long_high_quality_simulation_proxy"}
    first = frames[0]
    min_dist, clashes, sampled_pairs = _sampled_pair_distances(first)
    clash_rate = clashes / max(1, sampled_pairs)

    step_rmsds: list[float] = []
    for prev, cur in zip(frames, frames[1:], strict=False):
        if not prev or not cur:
            continue
        count = min(len(prev), len(cur))
        msd = sum(
            (prev[i][0] - cur[i][0]) ** 2 + (prev[i][1] - cur[i][1]) ** 2 + (prev[i][2] - cur[i][2]) ** 2
            for i in range(count)
        ) / max(1, count)
        step_rmsds.append(math.sqrt(msd))
    mean_step = sum(step_rmsds) / max(1, len(step_rmsds))
    max_step = max(step_rmsds) if step_rmsds else 0.0

    def rg(coords: list[list[float]]) -> float:
        n = max(1, len(coords))
        cx = sum(xyz[0] for xyz in coords) / n
        cy = sum(xyz[1] for xyz in coords) / n
        cz = sum(xyz[2] for xyz in coords) / n
        return math.sqrt(sum((xyz[0] - cx) ** 2 + (xyz[1] - cy) ** 2 + (xyz[2] - cz) ** 2 for xyz in coords) / n)

    rgs = [rg(frame) for frame in frames]
    rg_mean = sum(rgs) / max(1, len(rgs))
    rg_std = math.sqrt(sum((value - rg_mean) ** 2 for value in rgs) / max(1, len(rgs)))
    residue_count = len({(atom.get("chain"), atom.get("residue_index")) for atom in atoms if atom.get("component") in {"protein", "dna", "rna"}})
    frame_score = min(1.0, len(frames) / max(1, target_frames))
    size_score = 1.0 if expected_residues is None else min(1.0, residue_count / max(1, expected_residues))
    clash_score = max(0.0, 1.0 - 60.0 * clash_rate)
    min_distance_score = max(0.0, min(1.0, (min_dist - 0.70) / 0.40))
    smoothness_score = max(0.0, 1.0 - abs(mean_step - 0.12) / 0.35)
    max_step_score = max(0.0, 1.0 - max(0.0, max_step - 0.5) / 1.5)
    rg_stability_score = max(0.0, 1.0 - (rg_std / max(1e-6, rg_mean)) * 25.0)
    long_hq_score = (
        0.18 * frame_score
        + 0.16 * size_score
        + 0.20 * clash_score
        + 0.16 * min_distance_score
        + 0.14 * smoothness_score
        + 0.06 * max_step_score
        + 0.10 * rg_stability_score
    )
    return {
        "score_profile": "long_high_quality_simulation_proxy",
        "long_hq_score": round(float(long_hq_score), 6),
        "atom_count": len(atoms),
        "residue_count": residue_count,
        "frame_count": len(frames),
        "target_frames": target_frames,
        "expected_residues": expected_residues,
        "min_sampled_distance_a": round(float(min_dist), 6),
        "sampled_clash_rate": round(float(clash_rate), 8),
        "mean_step_rmsd_a": round(float(mean_step), 6),
        "max_step_rmsd_a": round(float(max_step), 6),
        "radius_gyration_mean_a": round(float(rg_mean), 6),
        "radius_gyration_std_a": round(float(rg_std), 6),
        "frame_score": round(float(frame_score), 6),
        "size_score": round(float(size_score), 6),
        "clash_score": round(float(clash_score), 6),
        "min_distance_score": round(float(min_distance_score), 6),
        "smoothness_score": round(float(smoothness_score), 6),
        "max_step_score": round(float(max_step_score), 6),
        "rg_stability_score": round(float(rg_stability_score), 6),
        "note": "Proxy scores use long-run simulation-style criteria; strict FairChem/OpenMM rescoring is still required for physical claims.",
    }
