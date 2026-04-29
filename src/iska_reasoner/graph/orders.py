from __future__ import annotations

import random


def uniform_random_order(n: int, rng: random.Random) -> list[int]:
    order = list(range(n))
    rng.shuffle(order)
    return order


def dependency_order(tokens: list[str]) -> list[int]:
    priority = {"NODE": 0, "EDGE": 1, "CLAIM": 2, "ANSWER": 3}
    return sorted(range(len(tokens)), key=lambda i: (priority.get(tokens[i].split(":", 1)[0], 10), i))


def verifier_enabling_order(tokens: list[str]) -> list[int]:
    priority = {"UNIT": 0, "CODE": 0, "THEOREM": 0, "SMILES": 0, "CLAIM": 1, "ANSWER": 2}
    return sorted(range(len(tokens)), key=lambda i: (priority.get(tokens[i].split(":", 1)[0], 5), i))


def scientific_graph_order(tokens: list[str]) -> list[int]:
    """Coarse-to-fine order for scientific graph-record targets.

    The order keeps the model-type/task/modality records first, then sequence
    and molecular identity records, then physical constraints, then renderers
    and text. It is intentionally prefix-based so it works for unseen motif
    names without changing the code.
    """

    def priority(token: str) -> tuple[int, int]:
        if token == "UGM:graph_to_graph" or token.startswith("UGM:task:"):
            return (0, 0)
        if token.startswith("UGM:modality:"):
            return (1, 0)
        if token.startswith(("AA:", "DNA:", "RNA:", "SELFIES:", "SMILES:")):
            return (2, 0)
        if token.startswith(("ATOM:", "SLOT:", "ELEMENT:", "MOTIF:", "SEQ_MOTIF:", "SEQ_MOTIF_NAME:", "SEQ_MOTIF_FROM_STRUCTURE:", "STRUCT_MOTIF:", "STRUCT_MOTIF_NAME:", "STRUCT_DERIVED_SEQ_MOTIF:", "STRUCT_DERIVED_SEQ_MOTIF_NAME:")):
            return (3, 0)
        if token.startswith(("ATTN_BIN:", "ATTN_COARSE:", "TOKEN_COUPLING:", "UMA_INFLUENCE:")):
            return (4, 0)
        if token.startswith(("SEQ_STRUCT_DYN_PROXY:", "TOKEN_MOTION:", "UMA_TRAJ_BIN:")):
            return (5, 0)
        if token.startswith("BOND:"):
            return (6, 0)
        if token.startswith(("TEMP:", "DIST:", "TORSION:")):
            return (7, 0)
        if token.startswith("COORD:"):
            return (8, 0)
        if token.startswith(("ENERGY:", "FORCE:")):
            return (9, 0)
        if token.startswith("PDB:"):
            return (10, 0)
        if token.startswith("UGM:oracle:"):
            return (11, 0)
        if token.startswith("UGM:serializer:"):
            return (12, 0)
        if token.startswith("ANSWER:"):
            return (13, 0)
        return (20, 0)

    return sorted(range(len(tokens)), key=lambda i: (*priority(tokens[i]), i))


def oracle_enabling_order(tokens: list[str]) -> list[int]:
    """Order records that enable chemistry/structure validation before prose."""

    def priority(token: str) -> tuple[int, int]:
        if token == "UGM:graph_to_graph" or token.startswith("UGM:task:"):
            return (0, 0)
        if token.startswith(("ATTN_BIN:", "ATTN_COARSE:", "TOKEN_COUPLING:", "UMA_INFLUENCE:", "TOKEN_MOTION:", "UMA_TRAJ_BIN:", "SEQ_STRUCT_DYN_PROXY:")):
            return (1, 0)
        if token.startswith(("ATOM:", "BOND:", "TEMP:")):
            return (2, 0)
        if token.startswith(("DIST:", "COORD:")):
            return (3, 0)
        if token.startswith(("ENERGY:", "FORCE:", "UGM:oracle:")):
            return (4, 0)
        if token.startswith("PDB:"):
            return (5, 0)
        if token.startswith("UGM:serializer:"):
            return (6, 0)
        if token.startswith("ANSWER:"):
            return (7, 0)
        return (10, 0)

    return sorted(range(len(tokens)), key=lambda i: (*priority(tokens[i]), i))


def build_orders(tokens: list[str], seed: int, n_random: int = 2) -> list[list[int]]:
    rng = random.Random(seed)
    n = len(tokens)
    if n == 0:
        return []
    orders = [list(range(n)), dependency_order(tokens), verifier_enabling_order(tokens)]
    if any(tok.startswith(("UGM:", "AA:", "DNA:", "RNA:", "SELFIES:", "ATOM:", "BOND:", "COORD:", "DIST:", "ENERGY:", "FORCE:", "PDB:", "SEQ_MOTIF:", "SEQ_MOTIF_FROM_STRUCTURE:", "STRUCT_MOTIF:", "STRUCT_DERIVED_SEQ_MOTIF:", "ATTN_BIN:", "ATTN_COARSE:", "TOKEN_COUPLING:", "UMA_INFLUENCE:", "TOKEN_MOTION:", "UMA_TRAJ_BIN:", "SEQ_STRUCT_DYN_PROXY:")) for tok in tokens):
        orders.extend([scientific_graph_order(tokens), oracle_enabling_order(tokens)])
    for _ in range(n_random):
        orders.append(uniform_random_order(n, rng))
    deduped: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for order in orders:
        key = tuple(order)
        if key not in seen:
            deduped.append(order)
            seen.add(key)
    return deduped
