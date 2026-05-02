from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from iska_reasoner.data.dataset import INTERNAL_COORD_TYPE_TO_ID
from iska_reasoner.oracles import score_uma_coordinate_candidate


PROTEIN_PHI_ID = INTERNAL_COORD_TYPE_TO_ID["protein_phi"]
PROTEIN_PSI_ID = INTERNAL_COORD_TYPE_TO_ID["protein_psi"]
PROTEIN_OMEGA_ID = INTERNAL_COORD_TYPE_TO_ID["protein_omega"]
PROTEIN_BACKBONE_SYMBOLS = ("N", "C", "C", "O")


def _temperature_kelvin(example: Any) -> float | None:
    for node in getattr(example, "nodes", []) or []:
        if getattr(node, "type", "") != "temperature":
            continue
        features = getattr(node, "features", {}) or {}
        for key in ("kelvin", "kelvin_clamped", "temperature_k"):
            try:
                return float(features[key])
            except Exception:
                pass
        try:
            return float(str(getattr(node, "value", "")).rstrip("Kk"))
        except Exception:
            return None
    try:
        return float((getattr(example, "metadata", {}) or {}).get("temperature"))
    except Exception:
        return None


def _coordinate_regularizer(
    positions: torch.Tensor,
    *,
    center_weight: float,
    repulsion_weight: float,
    min_distance: float,
) -> torch.Tensor:
    if positions.numel() == 0:
        return positions.sum() * 0.0
    loss = positions.sum() * 0.0
    if center_weight > 0:
        loss = loss + float(center_weight) * positions.mean(dim=0).pow(2).mean()
    if repulsion_weight > 0 and positions.size(0) > 1:
        distances = torch.pdist(positions.float(), p=2)
        loss = loss + float(repulsion_weight) * F.relu(float(min_distance) - distances).pow(2).mean()
    return loss


def uma_coordinate_head_oracle_loss(
    coordinate_mean: torch.Tensor | None,
    query_mask: torch.Tensor | None,
    symbols_batch: list[list[str]] | None,
    examples: list[Any],
    *,
    backend: str = "fairchem",
    repo_path: str = "data/external_repos/fairchem",
    model_name: str = "uma-s-1p2",
    task_name: str = "omol",
    device_name: str = "cuda",
    strict: bool = False,
    max_examples: int = 1,
    max_atoms: int = 16,
    force_clip: float = 10.0,
    center_weight: float = 0.001,
    repulsion_weight: float = 0.01,
    min_distance: float = 0.75,
    dynamics_steps: int = 1,
    force_step_size: float = 0.02,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Train coordinate proposals from UMA force feedback, not coordinates.

    UMA is an external oracle, so its force vector is detached. The surrogate
    loss ``-F_UMA(x).detach() dot x`` has gradient ``-F_UMA(x)`` with respect to
    the model coordinates, so gradient descent moves coordinates along the UMA
    force direction. With ``dynamics_steps > 1`` the same candidate is rolled
    forward by detached UMA forces and rescored, making reasoning iterations act
    like a small learned time discretization. No PDB/SDF/mmCIF coordinates or
    supervised force labels are read by this function.
    """
    if coordinate_mean is None or query_mask is None or not symbols_batch:
        zero_base = coordinate_mean if coordinate_mean is not None else query_mask
        zero = zero_base.sum() * 0.0 if torch.is_tensor(zero_base) else torch.tensor(0.0)
        return zero, {
            "uma_coordinate/oracle_examples": 0.0,
            "uma_coordinate/available_rate": 0.0,
            "uma_coordinate/atom_count": 0.0,
            "uma_coordinate/reward": 0.0,
            "uma_coordinate/energy_per_atom_ev": 0.0,
            "uma_coordinate/energy_per_atom_end_ev": 0.0,
            "uma_coordinate/force_rms_ev_per_a": 0.0,
            "uma_coordinate/displacement_a": 0.0,
            "uma_coordinate/dynamics_steps": 0.0,
            "uma_coordinate/surrogate_loss": 0.0,
        }

    coords = coordinate_mean.float()
    mask = query_mask.to(device=coords.device, dtype=torch.bool)
    losses: list[torch.Tensor] = []
    available = 0
    rewards: list[float] = []
    energies: list[float] = []
    end_energies: list[float] = []
    force_rms_values: list[float] = []
    displacements: list[float] = []
    atom_counts: list[int] = []
    checked = 0
    rollout_steps = max(1, int(dynamics_steps))
    step_size = float(force_step_size)
    for row, symbols in enumerate(symbols_batch[: coords.size(0)]):
        if checked >= max(1, int(max_examples)):
            break
        positions_idx = torch.where(mask[row])[0]
        n = min(len(symbols), int(max_atoms), int(positions_idx.numel()))
        if n <= 0:
            continue
        positions_idx = positions_idx[:n]
        symbols = symbols[:n]
        checked += 1
        atom_counts.append(n)
        positions0 = coords[row, positions_idx, :]
        positions_t = positions0
        sample_losses: list[torch.Tensor] = []
        sample_available = False
        start_energy: float | None = None
        last_energy: float | None = None
        temp_k = _temperature_kelvin(examples[row]) if row < len(examples) else None
        for rollout_idx in range(rollout_steps):
            result = score_uma_coordinate_candidate(
                symbols,
                positions_t.detach().cpu().tolist(),
                temperature_k=temp_k,
                backend=backend,
                strict=strict,
                repo_path=repo_path,
                model_name=model_name,
                task_name=task_name,
                device=device_name,
            )
            rewards.append(float(result.reward))
            if result.energy_per_atom_ev is not None:
                energy_value = float(result.energy_per_atom_ev)
                if rollout_idx == 0:
                    start_energy = energy_value
                last_energy = energy_value
            if result.force_rms_ev_per_a is not None:
                force_rms_values.append(float(result.force_rms_ev_per_a))
            if not result.available or not result.forces_ev_per_a:
                break
            force = torch.tensor(result.forces_ev_per_a[:n], device=coords.device, dtype=coords.dtype)
            if force.size(0) != n:
                break
            if force_clip > 0:
                force = force.clamp(min=-float(force_clip), max=float(force_clip))
            sample_losses.append(-(force.detach() * positions_t).sum(dim=-1).mean())
            positions_t = positions_t + step_size * force.detach()
            sample_available = True
        if start_energy is not None:
            energies.append(start_energy)
        if last_energy is not None:
            end_energies.append(last_energy)
        if sample_available and sample_losses:
            regularized = torch.stack(sample_losses).mean()
            regularized = regularized + _coordinate_regularizer(
                positions0,
                center_weight=center_weight,
                repulsion_weight=repulsion_weight,
                min_distance=min_distance,
            )
            losses.append(regularized)
            available += 1
            displacements.append(float((positions_t.detach() - positions0.detach()).norm(dim=-1).mean().cpu().item()))

    if not losses:
        zero = coords.sum() * 0.0
    else:
        zero = torch.stack(losses).mean()
    return zero, {
        "uma_coordinate/oracle_examples": float(checked),
        "uma_coordinate/available_rate": float(available / max(1, checked)),
        "uma_coordinate/atom_count": float(sum(atom_counts) / max(1, len(atom_counts))) if atom_counts else 0.0,
        "uma_coordinate/reward": float(sum(rewards) / max(1, len(rewards))) if rewards else 0.0,
        "uma_coordinate/energy_per_atom_ev": float(sum(energies) / max(1, len(energies))) if energies else 0.0,
        "uma_coordinate/energy_per_atom_end_ev": float(sum(end_energies) / max(1, len(end_energies))) if end_energies else 0.0,
        "uma_coordinate/force_rms_ev_per_a": float(sum(force_rms_values) / max(1, len(force_rms_values))) if force_rms_values else 0.0,
        "uma_coordinate/displacement_a": float(sum(displacements) / max(1, len(displacements))) if displacements else 0.0,
        "uma_coordinate/dynamics_steps": float(rollout_steps if checked else 0),
        "uma_coordinate/surrogate_loss": float(zero.detach().float().cpu().item()) if zero.numel() == 1 else 0.0,
    }


def _zero_internal_coordinate_loss(base: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, float]]:
    zero = base.sum() * 0.0 if torch.is_tensor(base) else torch.tensor(0.0)
    return zero, {
        "uma_internal/oracle_examples": 0.0,
        "uma_internal/available_rate": 0.0,
        "uma_internal/residue_count": 0.0,
        "uma_internal/atom_count": 0.0,
        "uma_internal/reward": 0.0,
        "uma_internal/energy_per_atom_ev": 0.0,
        "uma_internal/energy_per_atom_end_ev": 0.0,
        "uma_internal/force_rms_ev_per_a": 0.0,
        "uma_internal/displacement_a": 0.0,
        "uma_internal/dynamics_steps": 0.0,
        "uma_internal/surrogate_loss": 0.0,
    }


def _protein_internal_angle_table(
    internal_mean_row: torch.Tensor,
    mask_row: torch.Tensor,
    type_ids_row: torch.Tensor,
    residue_indices_row: torch.Tensor,
    *,
    max_residues: int,
) -> torch.Tensor | None:
    active = torch.where(mask_row.bool() & residue_indices_row.ge(0))[0]
    if active.numel() == 0:
        return None
    protein_active = active[
        (type_ids_row[active] == PROTEIN_PHI_ID)
        | (type_ids_row[active] == PROTEIN_PSI_ID)
        | (type_ids_row[active] == PROTEIN_OMEGA_ID)
    ]
    if protein_active.numel() == 0:
        return None
    residue_count = int(residue_indices_row[protein_active].max().item()) + 1
    residue_count = max(1, min(int(max_residues), residue_count))
    angles = internal_mean_row.new_zeros((residue_count, 3))
    counts = internal_mean_row.new_zeros((residue_count, 3))
    for idx in protein_active.tolist():
        residue_idx = int(residue_indices_row[idx].item())
        if residue_idx < 0 or residue_idx >= residue_count:
            continue
        type_id = int(type_ids_row[idx].item())
        if type_id == PROTEIN_PHI_ID:
            channel = 0
        elif type_id == PROTEIN_PSI_ID:
            channel = 1
        elif type_id == PROTEIN_OMEGA_ID:
            channel = 2
        else:
            continue
        angles[residue_idx, channel] = angles[residue_idx, channel] + internal_mean_row[idx]
        counts[residue_idx, channel] = counts[residue_idx, channel] + 1.0
    angles = angles / counts.clamp_min(1.0)
    if counts.sum() <= 0:
        return None
    return angles


def _coarse_backbone_from_internal_angles(angles: torch.Tensor, *, max_atoms: int) -> tuple[list[str], torch.Tensor]:
    """Map torsion-like actions into a differentiable coarse backbone scaffold.

    This is not a supervised structure decoder. It is a compact coordinate
    carrier for oracle feedback: internal angle actions define a smooth chain,
    UMA scores that generated chain, and the detached force field trains the
    internal-coordinate head.
    """
    if angles.numel() == 0 or max_atoms <= 0:
        return [], angles.new_zeros((0, 3))
    phi = angles[:, 0]
    psi = angles[:, 1]
    omega = angles[:, 2]
    turn = 0.55 * torch.sin(phi) + 0.35 * torch.sin(psi) + 0.15 * torch.sin(omega)
    theta = torch.cumsum(turn, dim=0)
    step = torch.stack(
        [
            1.45 * torch.cos(theta),
            1.45 * torch.sin(theta),
            1.05 + 0.20 * torch.cos(omega),
        ],
        dim=-1,
    )
    centers = torch.cumsum(step, dim=0)
    tangent = F.normalize(step, dim=-1)
    normal = F.normalize(
        torch.stack(
            [
                -torch.sin(theta),
                torch.cos(theta),
                torch.zeros_like(theta),
            ],
            dim=-1,
        ),
        dim=-1,
    )
    binormal = torch.zeros_like(normal)
    binormal[:, 2] = 1.0
    n_pos = centers - 0.55 * tangent + 0.12 * normal
    ca_pos = centers
    c_pos = centers + 0.55 * tangent
    o_pos = c_pos + 0.22 * normal + 0.10 * binormal
    positions = torch.stack([n_pos, ca_pos, c_pos, o_pos], dim=1).reshape(-1, 3)
    symbols = [symbol for _ in range(angles.size(0)) for symbol in PROTEIN_BACKBONE_SYMBOLS]
    n = min(int(max_atoms), len(symbols), int(positions.size(0)))
    return symbols[:n], positions[:n]


def uma_internal_coordinate_head_oracle_loss(
    internal_mean: torch.Tensor | None,
    query_mask: torch.Tensor | None,
    type_ids: torch.Tensor | None,
    residue_indices: torch.Tensor | None,
    examples: list[Any],
    *,
    backend: str = "fairchem",
    repo_path: str = "data/external_repos/fairchem",
    model_name: str = "uma-s-1p2",
    task_name: str = "omol",
    device_name: str = "cuda",
    strict: bool = False,
    max_examples: int = 1,
    max_residues: int = 4,
    max_atoms: int = 16,
    force_clip: float = 10.0,
    center_weight: float = 0.001,
    repulsion_weight: float = 0.01,
    min_distance: float = 0.75,
    dynamics_steps: int = 1,
    force_step_size: float = 0.02,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Train symbolic internal-coordinate actions from UMA forces.

    The model emits torsion-like action means in radians. This helper converts
    protein phi/psi/omega action slots into a differentiable coarse backbone,
    queries UMA on the generated coordinates, and applies the same detached
    force surrogate used by the raw coordinate head. It is intentionally
    sequence-derived: the coordinate scaffold is generated from the model's
    actions, not loaded from PDB/AFDB/SDF/MD frames.
    """
    if internal_mean is None or query_mask is None or type_ids is None or residue_indices is None:
        return _zero_internal_coordinate_loss(internal_mean if internal_mean is not None else query_mask)

    means = internal_mean.float()
    mask = query_mask.to(device=means.device, dtype=torch.bool)
    type_ids_t = type_ids.to(device=means.device)
    residue_indices_t = residue_indices.to(device=means.device)
    losses: list[torch.Tensor] = []
    available = 0
    checked = 0
    rewards: list[float] = []
    energies: list[float] = []
    end_energies: list[float] = []
    force_rms_values: list[float] = []
    displacements: list[float] = []
    residue_counts: list[int] = []
    atom_counts: list[int] = []
    rollout_steps = max(1, int(dynamics_steps))
    step_size = float(force_step_size)

    for row in range(means.size(0)):
        if checked >= max(1, int(max_examples)):
            break
        angles = _protein_internal_angle_table(
            means[row],
            mask[row],
            type_ids_t[row],
            residue_indices_t[row],
            max_residues=max_residues,
        )
        if angles is None:
            continue
        symbols, positions0 = _coarse_backbone_from_internal_angles(angles, max_atoms=max_atoms)
        n = min(len(symbols), int(positions0.size(0)))
        if n <= 0:
            continue
        symbols = symbols[:n]
        positions0 = positions0[:n]
        residue_counts.append(int(angles.size(0)))
        atom_counts.append(n)
        checked += 1
        sample_losses: list[torch.Tensor] = []
        positions_t = positions0
        sample_available = False
        start_energy: float | None = None
        last_energy: float | None = None
        temp_k = _temperature_kelvin(examples[row]) if row < len(examples) else None
        for rollout_idx in range(rollout_steps):
            result = score_uma_coordinate_candidate(
                symbols,
                positions_t.detach().cpu().tolist(),
                temperature_k=temp_k,
                backend=backend,
                strict=strict,
                repo_path=repo_path,
                model_name=model_name,
                task_name=task_name,
                device=device_name,
            )
            rewards.append(float(result.reward))
            if result.energy_per_atom_ev is not None:
                energy_value = float(result.energy_per_atom_ev)
                if rollout_idx == 0:
                    start_energy = energy_value
                last_energy = energy_value
            if result.force_rms_ev_per_a is not None:
                force_rms_values.append(float(result.force_rms_ev_per_a))
            if not result.available or not result.forces_ev_per_a:
                break
            force = torch.tensor(result.forces_ev_per_a[:n], device=means.device, dtype=means.dtype)
            if force.size(0) != n:
                break
            if force_clip > 0:
                force = force.clamp(min=-float(force_clip), max=float(force_clip))
            sample_losses.append(-(force.detach() * positions_t).sum(dim=-1).mean())
            positions_t = positions_t + step_size * force.detach()
            sample_available = True
        if start_energy is not None:
            energies.append(start_energy)
        if last_energy is not None:
            end_energies.append(last_energy)
        if sample_available and sample_losses:
            regularized = torch.stack(sample_losses).mean()
            regularized = regularized + _coordinate_regularizer(
                positions0,
                center_weight=center_weight,
                repulsion_weight=repulsion_weight,
                min_distance=min_distance,
            )
            losses.append(regularized)
            available += 1
            displacements.append(float((positions_t.detach() - positions0.detach()).norm(dim=-1).mean().cpu().item()))

    if not losses:
        loss = means.sum() * 0.0
    else:
        loss = torch.stack(losses).mean()
    return loss, {
        "uma_internal/oracle_examples": float(checked),
        "uma_internal/available_rate": float(available / max(1, checked)),
        "uma_internal/residue_count": float(sum(residue_counts) / max(1, len(residue_counts))) if residue_counts else 0.0,
        "uma_internal/atom_count": float(sum(atom_counts) / max(1, len(atom_counts))) if atom_counts else 0.0,
        "uma_internal/reward": float(sum(rewards) / max(1, len(rewards))) if rewards else 0.0,
        "uma_internal/energy_per_atom_ev": float(sum(energies) / max(1, len(energies))) if energies else 0.0,
        "uma_internal/energy_per_atom_end_ev": float(sum(end_energies) / max(1, len(end_energies))) if end_energies else 0.0,
        "uma_internal/force_rms_ev_per_a": float(sum(force_rms_values) / max(1, len(force_rms_values))) if force_rms_values else 0.0,
        "uma_internal/displacement_a": float(sum(displacements) / max(1, len(displacements))) if displacements else 0.0,
        "uma_internal/dynamics_steps": float(rollout_steps if checked else 0),
        "uma_internal/surrogate_loss": float(loss.detach().float().cpu().item()) if loss.numel() == 1 else 0.0,
    }
