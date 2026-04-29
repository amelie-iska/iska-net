import torch

from iska_reasoner.data.dataset import RandomOrderCollator
from iska_reasoner.data.synthetic import iter_synthetic_examples
from iska_reasoner.data.vocab import build_vocab
from iska_reasoner.models.random_order_tokengt import RandomOrderTokenGT, RandomOrderTokenGTConfig


def test_tiny_model_forward_backward():
    examples = list(iter_synthetic_examples(4))
    vocab = build_vocab(examples)
    batch = RandomOrderCollator(vocab=vocab, max_seq_len=96)(examples)
    model = RandomOrderTokenGT(
        RandomOrderTokenGTConfig(
            vocab_size=len(vocab.token_to_id),
            hidden_dim=32,
            num_layers=1,
            num_heads=4,
            ffn_dim=64,
            max_seq_len=96,
            endpoint_dim=8,
        )
    )
    out = model(**{k: batch[k] for k in ["input_ids", "kind_ids", "slot_ids", "endpoint_ids", "attention_mask", "causal_mask", "labels"]})
    assert out["loss"].isfinite()
    out["loss"].backward()
    assert any(p.grad is not None for p in model.parameters())

