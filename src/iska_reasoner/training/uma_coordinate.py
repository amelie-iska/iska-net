from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from iska_reasoner.oracles import score_uma_coordinate_candidate


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
