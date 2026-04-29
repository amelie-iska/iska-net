from iska_reasoner.data.synthetic import iter_synthetic_examples
from iska_reasoner.graph.schema import GraphExample, graph_source_tokens


def test_synthetic_examples_validate_round_trip():
    ex = next(iter_synthetic_examples(1))
    row = ex.to_dict()
    restored = GraphExample.from_dict(row)
    assert restored.id == ex.id
    assert restored.target_tokens
    tokens, kinds, endpoints, identifiers = graph_source_tokens(restored)
    assert tokens[0] == "<GRAPH>"
    assert len(tokens) == len(kinds) == len(endpoints) == len(identifiers)
    node_identifiers = [identifier for kind, identifier in zip(kinds, identifiers) if kind == "node"]
    edge_identifiers = [identifier for kind, identifier in zip(kinds, identifiers) if kind == "edge"]
    assert node_identifiers
    assert edge_identifiers
    assert set(node_identifiers).isdisjoint(edge_identifiers)
