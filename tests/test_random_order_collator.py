import torch

from iska_reasoner.data.dataset import KIND_TO_ID, RandomOrderCollator
from iska_reasoner.data.synthetic import iter_synthetic_examples
from iska_reasoner.data.vocab import build_vocab


def test_position_labels_predict_content_before_content_visible():
    examples = list(iter_synthetic_examples(4))
    vocab = build_vocab(examples)
    collator = RandomOrderCollator(vocab=vocab, order_mode="first", max_seq_len=128)
    batch = collator(examples[:2])
    assert batch["input_ids"].shape == batch["labels"].shape
    assert batch["identifier_ids"].shape == batch["input_ids"].shape
    node_ids = batch["identifier_ids"][0][batch["kind_ids"][0].eq(KIND_TO_ID["node"])]
    edge_ids = batch["identifier_ids"][0][batch["kind_ids"][0].eq(KIND_TO_ID["edge"])]
    assert node_ids.numel() > 0
    assert edge_ids.numel() > 0
    assert int(node_ids.min()) > 0
    assert int(edge_ids.min()) > int(node_ids.max())
    pos_mask = batch["kind_ids"].eq(KIND_TO_ID["position"])
    assert pos_mask.any()
    assert batch["labels"][pos_mask].ne(-100).all()
    target_mask = batch["kind_ids"].eq(KIND_TO_ID["target"])
    assert batch["labels"][target_mask].eq(-100).all()
    assert torch.equal(batch["causal_mask"], torch.triu(batch["causal_mask"], diagonal=1))
