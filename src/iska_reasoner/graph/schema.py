from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class Node:
    id: str
    type: str
    value: str
    features: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Edge:
    src: str
    dst: str
    type: str
    features: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GraphExample:
    id: str
    task: str
    nodes: list[Node]
    edges: list[Edge]
    target_tokens: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)
    decoder_orders: list[list[int]] = field(default_factory=list)

    def validate(self) -> None:
        node_ids = {node.id for node in self.nodes}
        if len(node_ids) != len(self.nodes):
            raise ValueError(f"Example {self.id} has duplicate node IDs")
        for edge in self.edges:
            if edge.src not in node_ids or edge.dst not in node_ids:
                raise ValueError(f"Example {self.id} has edge with missing endpoint: {edge}")
        n = len(self.target_tokens)
        for order in self.decoder_orders:
            if sorted(order) != list(range(n)):
                raise ValueError(f"Example {self.id} has invalid decoder order {order}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "task": self.task,
            "nodes": [asdict(node) for node in self.nodes],
            "edges": [asdict(edge) for edge in self.edges],
            "target_tokens": self.target_tokens,
            "decoder_orders": self.decoder_orders,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "GraphExample":
        ex = cls(
            id=str(row["id"]),
            task=str(row.get("task", "unknown")),
            nodes=[Node(**node) for node in row.get("nodes", [])],
            edges=[Edge(**edge) for edge in row.get("edges", [])],
            target_tokens=[str(tok) for tok in row.get("target_tokens", [])],
            decoder_orders=[[int(i) for i in order] for order in row.get("decoder_orders", [])],
            metadata=dict(row.get("metadata", {})),
        )
        ex.validate()
        return ex


def node_token(node: Node) -> str:
    value = node.value.replace("\n", "\\n")
    return f"N|{node.type}|{value}"


def edge_token(edge: Edge) -> str:
    return f"E|{edge.type}|{edge.src}|{edge.dst}"


def graph_source_tokens(example: GraphExample) -> tuple[list[str], list[str], list[tuple[int, int]], list[int]]:
    node_index = {node.id: i + 1 for i, node in enumerate(example.nodes)}
    edge_id_offset = len(example.nodes)
    tokens = ["<GRAPH>"]
    kinds = ["special"]
    endpoints = [(0, 0)]
    identifiers = [0]

    for node in example.nodes:
        idx = node_index[node.id]
        tokens.append(node_token(node))
        kinds.append("node")
        endpoints.append((idx, idx))
        identifiers.append(idx)

    for edge_idx, edge in enumerate(example.edges, start=1):
        tokens.append(edge_token(edge))
        kinds.append("edge")
        endpoints.append((node_index[edge.src], node_index[edge.dst]))
        identifiers.append(edge_id_offset + edge_idx)

    return tokens, kinds, endpoints, identifiers


def _float_feature(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def graph_source_numeric_features(example: GraphExample, dim: int = 4) -> list[list[float]]:
    """Return dense numeric conditioning features aligned to ``graph_source_tokens``.

    TokenGT still receives ordinary discrete graph tokens. This side channel is
    intentionally small and currently used for continuous conditioning records
    such as Kelvin temperature. Rounded temperature tokens remain useful anchors
    for vocabulary stability, while these features let the model see that
    312.5K and 313.0K are nearby conditions.
    """
    if dim <= 0:
        return []
    features: list[list[float]] = [[0.0] * dim]
    for node in example.nodes:
        row = [0.0] * dim
        if node.type == "temperature":
            kelvin = _float_feature(node.features.get("kelvin"))
            if kelvin is None:
                text = node.value.rstrip("Kk")
                kelvin = _float_feature(text)
            if kelvin is not None:
                clamped = max(300.0, min(400.0, kelvin))
                norm = (clamped - 300.0) / 100.0
                row[0] = 1.0
                if dim > 1:
                    row[1] = norm
                if dim > 2:
                    row[2] = 2.0 * norm - 1.0
                if dim > 3:
                    row[3] = 300.0 / max(clamped, 1e-6)
        features.append(row)
    for _edge in example.edges:
        features.append([0.0] * dim)
    return features
