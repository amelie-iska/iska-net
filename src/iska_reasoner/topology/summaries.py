from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable

import networkx as nx
import numpy as np
import torch

from iska_reasoner.graph.schema import GraphExample


TOPOLOGY_FEATURE_NAMES = [
    "node_count",
    "edge_count",
    "component_count",
    "cycle_rank",
    "edge_type_entropy",
    "h0_total_persistence",
    "laplacian_lambda2",
]

ADVANCED_TOPOLOGY_FEATURE_NAMES = [
    "ph_backend_available",
    "ph_h0_total_persistence",
    "ph_h1_total_persistence",
    "ph_h1_feature_count",
    "persistent_laplacian_trace_mean",
    "persistent_laplacian_lambda2_mean",
    "persistent_laplacian_scales",
]


@dataclass(slots=True)
class TopologySummary:
    node_count: float
    edge_count: float
    component_count: float
    cycle_rank: float
    edge_type_entropy: float
    h0_total_persistence: float
    laplacian_lambda2: float
    degree_mean: float
    degree_max: float
    density: float

    def feature_vector(self) -> list[float]:
        data = asdict(self)
        return [float(data[name]) for name in TOPOLOGY_FEATURE_NAMES]

    def metric_dict(self, prefix: str = "topology/") -> dict[str, float]:
        return {f"{prefix}{key}": float(value) for key, value in asdict(self).items()}


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


def graph_to_networkx(example: GraphExample) -> nx.Graph:
    g = nx.Graph()
    for node in example.nodes:
        g.add_node(node.id, type=node.type, value=node.value)
    for edge in example.edges:
        g.add_edge(edge.src, edge.dst, type=edge.type)
    return g


def edge_type_entropy(example: GraphExample) -> float:
    counts: dict[str, int] = {}
    for edge in example.edges:
        counts[edge.type] = counts.get(edge.type, 0) + 1
    total = sum(counts.values())
    if total == 0:
        return 0.0
    probs = [count / total for count in counts.values()]
    return float(-sum(p * math.log(p + 1e-12) for p in probs))


def h0_total_persistence_from_graph(g: nx.Graph) -> float:
    n = g.number_of_nodes()
    if n <= 1:
        return 0.0
    nodes = list(g.nodes)
    index = {node: i for i, node in enumerate(nodes)}
    lengths = dict(nx.all_pairs_shortest_path_length(g))
    weighted_edges: list[tuple[float, int, int]] = []
    for i, src in enumerate(nodes):
        src_lengths = lengths.get(src, {})
        for dst in nodes[i + 1 :]:
            if dst in src_lengths:
                weighted_edges.append((float(src_lengths[dst]), index[src], index[dst]))
    weighted_edges.sort(key=lambda item: item[0])
    uf = _UnionFind(n)
    total = 0.0
    merges = 0
    for dist, i, j in weighted_edges:
        if uf.union(i, j):
            total += dist
            merges += 1
            if merges == n - 1:
                break
    return float(total)


def laplacian_lambda2(g: nx.Graph) -> float:
    n = g.number_of_nodes()
    if n <= 1:
        return 0.0
    try:
        nodes = list(g.nodes)
        index = {node: i for i, node in enumerate(nodes)}
        adj = np.zeros((n, n), dtype=float)
        for src, dst in g.edges:
            i, j = index[src], index[dst]
            adj[i, j] = 1.0
            adj[j, i] = 1.0
        deg = np.diag(adj.sum(axis=1))
        lap = deg - adj
        eigvals = np.linalg.eigvalsh(lap)
    except Exception:
        return 0.0
    eigvals = np.sort(np.real(eigvals))
    return float(eigvals[1]) if len(eigvals) > 1 else 0.0


def graph_distance_matrix(g: nx.Graph) -> np.ndarray:
    """Return all-pairs shortest-path distances with finite disconnected fill.

    Optional PH libraries expect a dense distance matrix. Disconnected graph
    components get a conservative fill distance larger than the largest finite
    graph distance so they do not create artificial early merges.
    """

    n = g.number_of_nodes()
    if n == 0:
        return np.zeros((0, 0), dtype=float)
    nodes = list(g.nodes)
    index = {node: i for i, node in enumerate(nodes)}
    dist = np.full((n, n), np.inf, dtype=float)
    np.fill_diagonal(dist, 0.0)
    for src, lengths in nx.all_pairs_shortest_path_length(g):
        i = index[src]
        for dst, value in lengths.items():
            dist[i, index[dst]] = float(value)
    finite = dist[np.isfinite(dist)]
    fill = float(finite.max() + 1.0) if finite.size else 1.0
    dist[~np.isfinite(dist)] = fill
    return dist


def persistence_diagram_summary(distance_matrix: np.ndarray, maxdim: int = 1) -> dict[str, float]:
    """Compute PH summaries with ripser/gudhi if present, else use fallback.

    The fallback keeps the feature shape stable in environments without TDA
    dependencies. It reports H0 total persistence from an MST-like union-find
    and approximates H1 feature count with graph cycle rank elsewhere.
    """

    if distance_matrix.size == 0:
        return {
            "ph_backend_available": 0.0,
            "ph_h0_total_persistence": 0.0,
            "ph_h1_total_persistence": 0.0,
            "ph_h1_feature_count": 0.0,
        }
    try:
        from ripser import ripser  # type: ignore

        diagrams = ripser(distance_matrix, distance_matrix=True, maxdim=maxdim).get("dgms", [])
        return _diagram_metrics(diagrams, backend_available=True)
    except Exception:
        pass
    try:
        import gudhi as gd  # type: ignore

        rips = gd.RipsComplex(distance_matrix=distance_matrix)
        simplex_tree = rips.create_simplex_tree(max_dimension=maxdim + 1)
        simplex_tree.persistence()
        diagrams: list[np.ndarray] = []
        for dim in range(maxdim + 1):
            intervals = simplex_tree.persistence_intervals_in_dimension(dim)
            diagrams.append(np.asarray(intervals, dtype=float))
        return _diagram_metrics(diagrams, backend_available=True)
    except Exception:
        return {
            "ph_backend_available": 0.0,
            "ph_h0_total_persistence": _mst_total_from_distance(distance_matrix),
            "ph_h1_total_persistence": 0.0,
            "ph_h1_feature_count": 0.0,
        }


def _diagram_metrics(diagrams: list[np.ndarray], backend_available: bool) -> dict[str, float]:
    def total_persistence(dim: int) -> float:
        if dim >= len(diagrams) or diagrams[dim].size == 0:
            return 0.0
        arr = np.asarray(diagrams[dim], dtype=float)
        births = arr[:, 0]
        deaths = arr[:, 1]
        finite = np.isfinite(deaths)
        if not finite.any():
            return 0.0
        return float(np.maximum(0.0, deaths[finite] - births[finite]).sum())

    h1_count = 0.0
    if len(diagrams) > 1 and diagrams[1].size:
        deaths = np.asarray(diagrams[1], dtype=float)[:, 1]
        h1_count = float(np.isfinite(deaths).sum())
    return {
        "ph_backend_available": 1.0 if backend_available else 0.0,
        "ph_h0_total_persistence": total_persistence(0),
        "ph_h1_total_persistence": total_persistence(1),
        "ph_h1_feature_count": h1_count,
    }


def _mst_total_from_distance(distance_matrix: np.ndarray) -> float:
    n = distance_matrix.shape[0]
    if n <= 1:
        return 0.0
    weighted_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            weighted_edges.append((float(distance_matrix[i, j]), i, j))
    weighted_edges.sort(key=lambda row: row[0])
    uf = _UnionFind(n)
    total = 0.0
    merges = 0
    for dist, i, j in weighted_edges:
        if uf.union(i, j):
            total += dist
            merges += 1
            if merges == n - 1:
                break
    return float(total)


def persistent_laplacian_summary(g: nx.Graph, max_scales: int = 4) -> dict[str, float]:
    """Spectral summary over thresholded shortest-path graph filtrations.

    This is a lightweight persistent-Laplacian proxy: for several distance
    thresholds, build the graph containing edges within that threshold and
    summarize the ordinary graph Laplacian spectra.
    """

    n = g.number_of_nodes()
    if n <= 1:
        return {
            "persistent_laplacian_trace_mean": 0.0,
            "persistent_laplacian_lambda2_mean": 0.0,
            "persistent_laplacian_scales": 0.0,
        }
    dist = graph_distance_matrix(g)
    finite = sorted({float(v) for v in dist.reshape(-1) if np.isfinite(v) and v > 0})
    if not finite:
        return {
            "persistent_laplacian_trace_mean": 0.0,
            "persistent_laplacian_lambda2_mean": 0.0,
            "persistent_laplacian_scales": 0.0,
        }
    scales = finite[:max_scales]
    traces = []
    lambdas = []
    nodes = list(g.nodes)
    for scale in scales:
        fg = nx.Graph()
        fg.add_nodes_from(nodes)
        for i, src in enumerate(nodes):
            for j in range(i + 1, len(nodes)):
                if dist[i, j] <= scale:
                    fg.add_edge(src, nodes[j])
        traces.append(float(2 * fg.number_of_edges()))
        lambdas.append(laplacian_lambda2(fg))
    return {
        "persistent_laplacian_trace_mean": float(np.mean(traces)) if traces else 0.0,
        "persistent_laplacian_lambda2_mean": float(np.mean(lambdas)) if lambdas else 0.0,
        "persistent_laplacian_scales": float(len(scales)),
    }


def summarize_graph(example: GraphExample) -> TopologySummary:
    g = graph_to_networkx(example)
    node_count = g.number_of_nodes()
    edge_count = g.number_of_edges()
    component_count = nx.number_connected_components(g) if node_count else 0
    cycle_rank = edge_count - node_count + component_count
    degrees = [degree for _, degree in g.degree()]
    possible_edges = node_count * (node_count - 1) / 2
    return TopologySummary(
        node_count=float(node_count),
        edge_count=float(edge_count),
        component_count=float(component_count),
        cycle_rank=float(max(0, cycle_rank)),
        edge_type_entropy=edge_type_entropy(example),
        h0_total_persistence=h0_total_persistence_from_graph(g),
        laplacian_lambda2=laplacian_lambda2(g),
        degree_mean=float(np.mean(degrees)) if degrees else 0.0,
        degree_max=float(max(degrees)) if degrees else 0.0,
        density=float(edge_count / possible_edges) if possible_edges > 0 else 0.0,
    )


def summarize_graph_advanced(example: GraphExample) -> dict[str, float]:
    g = graph_to_networkx(example)
    dist = graph_distance_matrix(g)
    ph = persistence_diagram_summary(dist, maxdim=1)
    if not ph.get("ph_backend_available"):
        ph["ph_h1_feature_count"] = float(max(0, g.number_of_edges() - g.number_of_nodes() + nx.number_connected_components(g))) if g.number_of_nodes() else 0.0
    ph.update(persistent_laplacian_summary(g))
    return {name: float(ph.get(name, 0.0)) for name in ADVANCED_TOPOLOGY_FEATURE_NAMES}


def topology_feature_tensor(examples: Iterable[GraphExample]) -> torch.Tensor:
    return torch.tensor([summarize_graph(ex).feature_vector() for ex in examples], dtype=torch.float32)
