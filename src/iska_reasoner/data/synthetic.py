from __future__ import annotations

import random
from typing import Iterator

from iska_reasoner.graph.orders import build_orders
from iska_reasoner.graph.schema import Edge, GraphExample, Node


def make_path_example(idx: int, rng: random.Random, min_len: int = 3, max_len: int = 7) -> GraphExample:
    length = rng.randint(min_len, max_len)
    nodes = [Node(id=f"n{i}", type="entity", value=f"v{i}") for i in range(length)]
    edges = [Edge(src=f"n{i}", dst=f"n{i+1}", type="next") for i in range(length - 1)]
    # Add a distractor edge that should not change the answer.
    if length > 4:
        edges.append(Edge(src="n0", dst=f"n{length-1}", type="distractor"))
    target_tokens = [
        f"CLAIM:path_length:{length - 1}",
        f"CLAIM:start:{nodes[0].value}",
        f"CLAIM:end:{nodes[-1].value}",
        f"ANSWER:{length - 1}",
    ]
    ex = GraphExample(
        id=f"synthetic_path_{idx}",
        task="synthetic_path",
        nodes=nodes,
        edges=edges,
        target_tokens=target_tokens,
        metadata={"generator": "make_path_example", "answer": length - 1},
    )
    ex.decoder_orders = build_orders(ex.target_tokens, seed=idx)
    return ex


def make_tool_repair_example(idx: int, rng: random.Random) -> GraphExample:
    a = rng.randint(2, 9)
    b = rng.randint(2, 9)
    wrong = a + b + rng.choice([-1, 1])
    right = a + b
    nodes = [
        Node(id="problem", type="prompt", value=f"Compute {a}+{b}"),
        Node(id="tool0", type="tool_call", value=f"python: print({a}+{b})"),
        Node(id="err0", type="verifier", value=f"expected {right}, got {wrong}"),
        Node(id="repair0", type="repair", value=f"replace {wrong} with {right}"),
    ]
    edges = [
        Edge(src="problem", dst="tool0", type="calls"),
        Edge(src="tool0", dst="err0", type="checked_by"),
        Edge(src="err0", dst="repair0", type="repairs"),
    ]
    target_tokens = [
        f"CODE:print({a}+{b})",
        f"UNIT:integer",
        f"CLAIM:tool_result:{right}",
        f"ANSWER:{right}",
    ]
    ex = GraphExample(
        id=f"synthetic_tool_{idx}",
        task="synthetic_tool_repair",
        nodes=nodes,
        edges=edges,
        target_tokens=target_tokens,
        metadata={"generator": "make_tool_repair_example", "answer": right},
    )
    ex.decoder_orders = build_orders(ex.target_tokens, seed=10_000 + idx)
    return ex


def iter_synthetic_examples(count: int, seed: int = 13) -> Iterator[GraphExample]:
    rng = random.Random(seed)
    for idx in range(count):
        if idx % 2 == 0:
            yield make_path_example(idx, rng)
        else:
            yield make_tool_repair_example(idx, rng)

