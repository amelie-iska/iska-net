from iska_reasoner.data.splits import assign_split_for_policy, scientific_split_key
from iska_reasoner.graph.schema import Edge, GraphExample, Node


def test_entity_split_groups_related_protein_sequences_by_minhash_key():
    seq = "MKWVTFISLLLLFSSAYSRGVFRRDTHKSEIAHRFKDLGE"
    a = GraphExample(
        id="a",
        task="protein",
        nodes=[Node(id="protein", type="protein_sequence", value=seq)],
        edges=[],
        target_tokens=["A"],
    )
    b = GraphExample(
        id="b",
        task="protein",
        nodes=[Node(id="protein", type="protein_sequence", value=seq)],
        edges=[],
        target_tokens=["B"],
    )
    assert scientific_split_key(a) == scientific_split_key(b)
    assert assign_split_for_policy(a, "entity", 0.1, 0.1)[0] == assign_split_for_policy(b, "entity", 0.1, 0.1)[0]


def test_entity_split_prefers_explicit_scaffold_and_inchikey_blocks():
    a = GraphExample(
        id="a",
        task="molecule",
        nodes=[Node(id="mol", type="smiles", value="CCO")],
        edges=[],
        target_tokens=["A"],
        metadata={"inchi_key": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"},
    )
    b = GraphExample(
        id="b",
        task="molecule",
        nodes=[Node(id="mol", type="smiles", value="CCCO")],
        edges=[],
        target_tokens=["B"],
        metadata={"inchi_key": "LFQSCWFLJHTTHZ-OTHER"},
    )
    assert scientific_split_key(a) == "inchi_key_block:lfqscwfljhtthz"
    assert scientific_split_key(a) == scientific_split_key(b)


def test_row_hash_policy_keeps_row_level_behavior():
    ex = GraphExample(
        id="x",
        task="tiny",
        nodes=[Node(id="n", type="text", value="hello")],
        edges=[Edge(src="n", dst="n", type="self")],
        target_tokens=["HELLO"],
    )
    split, key = assign_split_for_policy(ex, "row_hash", 0.1, 0.1)
    assert split in {"train", "val", "test"}
    assert key.startswith("row_hash:")
