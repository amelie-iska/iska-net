from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import torch
import torch.nn as nn


class TropicalAttention(nn.Module):
    """Max-plus style attention for algorithmic-selection experiments.

    In soft mode this is ordinary temperature-scaled dot-product attention.
    In hard mode the output is the value vector at the maximum score. This is
    intentionally small and standalone so it can be used in ablations without
    replacing the main PyTorch transformer stack.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 1, hard: bool = False, temperature: float = 1.0):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.hard = hard
        self.temperature = temperature
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / max(self.head_dim**0.5, 1.0)
        scores = scores / max(float(self.temperature), 1e-6)
        if mask is not None:
            scores = scores.masked_fill(~mask[:, None, None, :].bool(), -1e9)
        if self.hard:
            index = scores.argmax(dim=-1)
            weights = torch.zeros_like(scores).scatter_(-1, index.unsqueeze(-1), 1.0)
        else:
            weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(weights, v).transpose(1, 2).reshape(batch, seq_len, self.hidden_dim)
        return self.out_proj(out), weights


@dataclass(slots=True)
class TropicalCellSignature:
    active_rate: float
    transition_rate: float
    unique_cells: float

    def metric_dict(self, prefix: str = "tropical/cell_") -> dict[str, float]:
        return {
            f"{prefix}active_rate": self.active_rate,
            f"{prefix}transition_rate": self.transition_rate,
            f"{prefix}unique_cells": self.unique_cells,
        }


@torch.no_grad()
def activation_cell_signature(hidden_states: torch.Tensor, threshold: float = 0.0) -> TropicalCellSignature:
    """Approximate tropical cell transitions by hidden-coordinate signs."""

    if hidden_states.numel() == 0:
        return TropicalCellSignature(0.0, 0.0, 0.0)
    signs = hidden_states.float().gt(threshold)
    active_rate = signs.float().mean().item()
    if signs.size(1) <= 1:
        transition_rate = 0.0
    else:
        transition_rate = signs[:, 1:].ne(signs[:, :-1]).float().mean().item()
    flat = signs.reshape(signs.size(0), signs.size(1), -1)
    # A compact signature: pack up to 32 sign bits for approximate cell counts.
    bits = flat[..., : min(32, flat.size(-1))].to(torch.int64)
    powers = (2 ** torch.arange(bits.size(-1), device=bits.device, dtype=torch.int64)).view(1, 1, -1)
    packed = (bits * powers).sum(dim=-1)
    unique_cells = float(torch.unique(packed).numel())
    return TropicalCellSignature(float(active_rate), float(transition_rate), unique_cells)


def tropical_max_spanning_arborescence(nodes: list[str], edge_scores: dict[tuple[str, str], float], root: str | None = None) -> list[tuple[str, str, float]]:
    """Select a maximum-score directed dependency tree.

    NetworkX supplies Chu-Liu/Edmonds through maximum_spanning_arborescence.
    The optional root is enforced by adding a high-scoring synthetic root edge
    and then removing it from the returned tree.
    """

    if not nodes:
        return []
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    synthetic_root = "__ROOT__"
    if root is not None:
        graph.add_node(synthetic_root)
        graph.add_edge(synthetic_root, root, weight=max(edge_scores.values(), default=0.0) + 1.0)
    for (src, dst), score in edge_scores.items():
        if src != dst and src in nodes and dst in nodes:
            graph.add_edge(src, dst, weight=float(score))
    if graph.number_of_edges() == 0:
        return []
    try:
        tree = nx.maximum_spanning_arborescence(graph, attr="weight")
    except Exception:
        return []
    out = []
    for src, dst, data in tree.edges(data=True):
        if src == synthetic_root or dst == synthetic_root:
            continue
        out.append((str(src), str(dst), float(data.get("weight", 0.0))))
    return out

