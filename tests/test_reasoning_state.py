from iska_reasoner.graph.schema import Edge, GraphExample, Node
from iska_reasoner.reasoning import GraphStateAction, apply_graph_state_action, state_from_example


def test_reasoning_state_evolves_graph_not_plain_chain():
    ex = GraphExample(id="r", task="reason", nodes=[Node(id="q", type="question", value="x")], edges=[], target_tokens=[])
    state = state_from_example(ex)
    state = apply_graph_state_action(state, GraphStateAction(op="add_node", node=Node(id="t0", type="latent_thought", value="")))
    state = apply_graph_state_action(state, GraphStateAction(op="add_edge", edge=Edge(src="q", dst="t0", type="supports")))
    state = apply_graph_state_action(state, GraphStateAction(op="observe", observation="verifier:pending"))
    assert state.hidden_thought_ids == ["t0"]
    assert state.graph.edges[0].type == "supports"
    assert state.observations == ["verifier:pending"]
