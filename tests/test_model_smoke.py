import torch

from iska_reasoner.data.dataset import RandomOrderCollator
from iska_reasoner.data.synthetic import iter_synthetic_examples
from iska_reasoner.data.vocab import build_vocab
from iska_reasoner.models.random_order_tokengt import RandomOrderTokenGT, RandomOrderTokenGTConfig
from iska_reasoner.tropical import MultiHeadTropicalAttention


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


def test_multi_head_tropical_attention_respects_masks():
    attn = MultiHeadTropicalAttention(
        hidden_dim=16,
        num_heads=4,
        use_projection=True,
        use_norm_shift=True,
        projective_normalize=True,
        context_clamp=8.0,
    )
    x = torch.randn(2, 5, 16)
    causal_mask = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)
    key_padding_mask = torch.tensor([[False, False, False, False, False], [False, False, False, True, True]])
    out, scores = attn(x, attn_mask=causal_mask, key_padding_mask=key_padding_mask)
    assert out.shape == (2, 5, 16)
    assert scores.shape == (2, 4, 5, 5)
    assert torch.isfinite(out).all()
    assert torch.isfinite(scores).all()
    assert scores[0, :, 0, 1:].max().item() <= attn.score_floor / 2
    assert scores[1, :, :, 3:].max().item() <= attn.score_floor / 2
    assert "tropical_attention/top1_margin" in attn.last_metrics


def test_tiny_model_tropical_attention_forward_backward():
    examples = list(iter_synthetic_examples(4))
    vocab = build_vocab(examples)
    batch = RandomOrderCollator(vocab=vocab, max_seq_len=64)(examples)
    model = RandomOrderTokenGT(
        RandomOrderTokenGTConfig(
            vocab_size=len(vocab.token_to_id),
            hidden_dim=32,
            num_layers=1,
            num_heads=4,
            ffn_dim=64,
            max_seq_len=64,
            endpoint_dim=8,
            attention_backend="tropical",
            tropical_context_clamp=8.0,
        )
    )
    out = model(**{k: batch[k] for k in ["input_ids", "kind_ids", "slot_ids", "endpoint_ids", "attention_mask", "causal_mask", "labels"]})
    assert out["loss"].isfinite()
    assert "attention_metrics" in out
    assert "tropical_attention/top1_margin" in out["attention_metrics"]
    out["loss"].backward()
    assert any("query_trop" in name and p.grad is not None for name, p in model.named_parameters())
