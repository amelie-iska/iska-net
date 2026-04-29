from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def hidden_state_topology_metrics(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    max_points: int = 64,
    bins: int = 8,
    prefix: str = "hidden_topology/",
) -> dict[str, float]:
    """Cheap topology/distogram monitor over hidden graph-token states.

    This is intentionally bounded for training-time use. It subsamples visible
    tokens, computes pairwise hidden-state distances, reports a distogram
    entropy, and approximates H0 persistence with an MST total over distances.
    It is a monitoring layer, not a full differentiable persistent-homology
    backend.
    """

    if hidden_states.numel() == 0:
        return _zero_metrics(prefix)
    batch_metrics = []
    for hidden, mask in zip(hidden_states.detach(), attention_mask.detach()):
        active = hidden[mask.bool()]
        if active.size(0) == 0:
            continue
        if active.size(0) > max_points:
            # Deterministic stride subsampling keeps runs reproducible.
            index = torch.linspace(0, active.size(0) - 1, steps=max_points, device=active.device).long()
            active = active.index_select(0, index)
        active_float = active.float()
        distances = torch.cdist(active_float, active_float, p=2)
        js_distances = _pairwise_js_distance(F.softmax(active_float, dim=-1))
        upper = distances[torch.triu(torch.ones_like(distances, dtype=torch.bool), diagonal=1)]
        js_upper = js_distances[torch.triu(torch.ones_like(js_distances, dtype=torch.bool), diagonal=1)]
        if upper.numel() == 0:
            batch_metrics.append(
                {
                    "point_count": float(active.size(0)),
                    "pair_distance_mean": 0.0,
                    "pair_distance_std": 0.0,
                    "distogram_entropy": 0.0,
                    "js_distance_mean": 0.0,
                    "js_distance_std": 0.0,
                    "js_distogram_entropy": 0.0,
                    "geometry_js_correlation": 0.0,
                    "h0_total_persistence": 0.0,
                }
            )
            continue
        hist = torch.histc(upper.detach().cpu(), bins=bins, min=float(upper.min().item()), max=float(upper.max().item() + 1e-6))
        probs = hist.float() / hist.sum().clamp_min(1.0)
        entropy = -(probs * probs.clamp_min(1e-8).log()).sum().item()
        js_hist = torch.histc(js_upper.detach().cpu(), bins=bins, min=0.0, max=1.0)
        js_probs = js_hist.float() / js_hist.sum().clamp_min(1.0)
        js_entropy = -(js_probs * js_probs.clamp_min(1e-8).log()).sum().item()
        batch_metrics.append(
            {
                "point_count": float(active.size(0)),
                "pair_distance_mean": float(upper.mean().item()),
                "pair_distance_std": float(upper.std(unbiased=False).item()),
                "distogram_entropy": float(entropy),
                "js_distance_mean": float(js_upper.mean().item()),
                "js_distance_std": float(js_upper.std(unbiased=False).item()),
                "js_distogram_entropy": float(js_entropy),
                "geometry_js_correlation": _corrcoef(upper, js_upper),
                "h0_total_persistence": _mst_total(distances),
            }
        )
    if not batch_metrics:
        return _zero_metrics(prefix)
    out: dict[str, float] = {}
    keys = batch_metrics[0].keys()
    for key in keys:
        out[f"{prefix}{key}_mean"] = float(sum(row[key] for row in batch_metrics) / len(batch_metrics))
    return out


def hidden_topology_collapse_loss(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    margin: float = 0.5,
    max_points: int = 64,
) -> torch.Tensor:
    """Differentiable proxy regularizer that discourages hidden-state collapse."""

    losses: list[torch.Tensor] = []
    for hidden, mask in zip(hidden_states, attention_mask):
        active = hidden[mask.bool()]
        if active.size(0) <= 1:
            continue
        if active.size(0) > max_points:
            index = torch.linspace(0, active.size(0) - 1, steps=max_points, device=active.device).long()
            active = active.index_select(0, index)
        distances = torch.cdist(active.float(), active.float(), p=2)
        upper = distances[torch.triu(torch.ones_like(distances, dtype=torch.bool), diagonal=1)]
        if upper.numel():
            losses.append(F.relu(float(margin) - upper.mean()).pow(2))
    if not losses:
        return hidden_states.sum() * 0.0
    return torch.stack(losses).mean()


def hidden_js_geometry_loss(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    margin: float = 0.05,
    max_points: int = 64,
) -> torch.Tensor:
    """Differentiable proxy that discourages collapse after softmax geometry.

    Hidden vectors are turned into probability distributions with softmax, then
    compared by Jensen-Shannon distance. This gives a bounded distributional
    geometry signal complementary to Euclidean embedding distances.
    """

    losses: list[torch.Tensor] = []
    for hidden, mask in zip(hidden_states, attention_mask):
        active = hidden[mask.bool()]
        if active.size(0) <= 1:
            continue
        if active.size(0) > max_points:
            index = torch.linspace(0, active.size(0) - 1, steps=max_points, device=active.device).long()
            active = active.index_select(0, index)
        js = _pairwise_js_distance(F.softmax(active.float(), dim=-1))
        upper = js[torch.triu(torch.ones_like(js, dtype=torch.bool), diagonal=1)]
        if upper.numel():
            losses.append(F.relu(float(margin) - upper.mean()).pow(2))
    if not losses:
        return hidden_states.sum() * 0.0
    return torch.stack(losses).mean()


def _zero_metrics(prefix: str) -> dict[str, float]:
    return {
        f"{prefix}point_count_mean": 0.0,
        f"{prefix}pair_distance_mean_mean": 0.0,
        f"{prefix}pair_distance_std_mean": 0.0,
        f"{prefix}distogram_entropy_mean": 0.0,
        f"{prefix}js_distance_mean_mean": 0.0,
        f"{prefix}js_distance_std_mean": 0.0,
        f"{prefix}js_distogram_entropy_mean": 0.0,
        f"{prefix}geometry_js_correlation_mean": 0.0,
        f"{prefix}h0_total_persistence_mean": 0.0,
    }


def _pairwise_js_distance(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = probs.clamp_min(eps)
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(eps)
    q = p.unsqueeze(0)
    p2 = p.unsqueeze(1)
    m = 0.5 * (p2 + q)
    js = 0.5 * (p2 * (p2.clamp_min(eps).log() - m.clamp_min(eps).log())).sum(dim=-1)
    js = js + 0.5 * (q * (q.clamp_min(eps).log() - m.clamp_min(eps).log())).sum(dim=-1)
    return js.clamp_min(0.0).sqrt()


def _corrcoef(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.numel() <= 1 or y.numel() <= 1:
        return 0.0
    x0 = x.float() - x.float().mean()
    y0 = y.float() - y.float().mean()
    denom = x0.norm() * y0.norm()
    if float(denom.item()) <= 1e-12:
        return 0.0
    return float((x0 * y0).sum().div(denom).item())


def _mst_total(distances: torch.Tensor) -> float:
    n = distances.size(0)
    if n <= 1:
        return 0.0
    edges: list[tuple[float, int, int]] = []
    d_cpu = distances.detach().cpu()
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((float(d_cpu[i, j]), i, j))
    edges.sort(key=lambda item: item[0])
    parent = list(range(n))
    rank = [0] * n

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    total = 0.0
    merges = 0
    for dist, i, j in edges:
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        if rank[ri] < rank[rj]:
            ri, rj = rj, ri
        parent[rj] = ri
        if rank[ri] == rank[rj]:
            rank[ri] += 1
        total += dist
        merges += 1
        if merges == n - 1:
            break
    return float(total)
