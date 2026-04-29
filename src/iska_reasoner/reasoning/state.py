from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from iska_reasoner.graph.schema import Edge, GraphExample, Node


@dataclass(slots=True)
class GraphStateAction:
    op: str
    node: Node | None = None
    edge: Edge | None = None
    token: str | None = None
    observation: str | None = None


@dataclass(slots=True)
class ReasoningState:
    """Graph-state evolution container for non-chain reasoning.

    A state contains the current graph, hidden-thought handles, verifier/tool
    observations, and action history. Natural-language chain-of-thought is only
    one possible rendering of this state; the primary object is the evolving
    graph and its latent thought records.
    """

    graph: GraphExample
    hidden_thought_ids: list[str] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)
    action_history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph": self.graph.to_dict(),
            "hidden_thought_ids": list(self.hidden_thought_ids),
            "observations": list(self.observations),
            "action_history": list(self.action_history),
        }


def state_from_example(example: GraphExample) -> ReasoningState:
    thought_ids = [node.id for node in example.nodes if node.type in {"thought", "latent_thought", "soft_thought"}]
    return ReasoningState(graph=example, hidden_thought_ids=thought_ids)


def apply_graph_state_action(state: ReasoningState, action: GraphStateAction) -> ReasoningState:
    graph = GraphExample(
        id=state.graph.id,
        task=state.graph.task,
        nodes=list(state.graph.nodes),
        edges=list(state.graph.edges),
        target_tokens=list(state.graph.target_tokens),
        metadata=dict(state.graph.metadata),
        decoder_orders=[list(order) for order in state.graph.decoder_orders],
    )
    hidden_thought_ids = list(state.hidden_thought_ids)
    observations = list(state.observations)
    if action.op == "add_node" and action.node is not None:
        graph.nodes.append(action.node)
        if action.node.type in {"thought", "latent_thought", "soft_thought"}:
            hidden_thought_ids.append(action.node.id)
    elif action.op == "add_edge" and action.edge is not None:
        graph.edges.append(action.edge)
    elif action.op == "add_token" and action.token is not None:
        graph.target_tokens.append(action.token)
    elif action.op == "observe" and action.observation is not None:
        observations.append(action.observation)
    else:
        raise ValueError(f"Unsupported graph-state action: {action}")
    graph.validate()
    history = list(state.action_history)
    history.append({"op": action.op, "token": action.token, "observation": action.observation})
    return ReasoningState(graph=graph, hidden_thought_ids=hidden_thought_ids, observations=observations, action_history=history)
