import copy

import torch

from iska_reasoner.data.dataset import RandomOrderCollator
from iska_reasoner.data.synthetic import iter_synthetic_examples
from iska_reasoner.data.vocab import build_vocab
from iska_reasoner.models.random_order_tokengt import RandomOrderTokenGT, RandomOrderTokenGTConfig
from iska_reasoner.tropical import FlashSDPAAttention, MultiHeadTropicalAttention


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
    attn.emit_contact_map = True
    x = torch.randn(2, 5, 16)
    causal_mask = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)
    key_padding_mask = torch.tensor([[False, False, False, False, False], [False, False, False, True, True]])
    out, scores = attn(x, attn_mask=causal_mask, key_padding_mask=key_padding_mask)
    assert out.shape == (2, 5, 16)
    assert scores.shape == (2, 4, 5, 5)
    assert torch.isfinite(out).all()
    assert torch.isfinite(scores).all()
    assert attn.last_contact_map is not None
    assert attn.last_contact_map.shape == (2, 4, 5, 5)
    assert torch.allclose(attn.last_contact_map[1, :, 3:, :], torch.zeros_like(attn.last_contact_map[1, :, 3:, :]))
    assert attn.last_contact_map[0, :, 0, 1:].max().item() == 0.0
    assert attn.last_contact_map[1, :, :, 3:].max().item() == 0.0
    assert scores[0, :, 0, 1:].max().item() <= attn.score_floor / 2
    assert scores[1, :, :, 3:].max().item() <= attn.score_floor / 2
    assert "tropical_attention/top1_margin" in attn.last_metrics


def test_multi_head_tropical_attention_query_chunk_matches_full():
    torch.manual_seed(7)
    full = MultiHeadTropicalAttention(
        hidden_dim=16,
        num_heads=4,
        use_projection=True,
        use_norm_shift=True,
        projective_normalize=True,
        context_clamp=8.0,
        query_chunk_size=0,
    )
    chunked = copy.deepcopy(full)
    chunked.query_chunk_size = 2
    full.emit_contact_map = True
    chunked.emit_contact_map = True
    full.detach_contact_map = False
    chunked.detach_contact_map = False
    x = torch.randn(2, 5, 16)
    causal_mask = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)
    key_padding_mask = torch.tensor([[False, False, False, False, False], [False, False, False, True, True]])
    out_full, scores_full = full(x, attn_mask=causal_mask, key_padding_mask=key_padding_mask)
    out_chunked, scores_chunked = chunked(x, attn_mask=causal_mask, key_padding_mask=key_padding_mask)
    assert torch.allclose(out_full, out_chunked, atol=1e-5, rtol=1e-5)
    assert torch.allclose(scores_full, scores_chunked, atol=1e-5, rtol=1e-5)
    assert full.last_contact_map is not None
    assert chunked.last_contact_map is not None
    assert torch.allclose(full.last_contact_map, chunked.last_contact_map, atol=1e-5, rtol=1e-5)


def test_flash_sdpa_attention_respects_masks():
    attn = FlashSDPAAttention(hidden_dim=16, num_heads=4)
    x = torch.randn(2, 5, 16)
    causal_mask = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)
    key_padding_mask = torch.tensor([[False, False, False, False, False], [False, False, False, True, True]])
    out = attn(x, attn_mask=causal_mask, key_padding_mask=key_padding_mask)
    assert out.shape == (2, 5, 16)
    assert torch.isfinite(out).all()
    assert attn.last_metrics["flash_attention/enabled"].item() == 1.0


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


def test_tiny_model_hybrid_flash_tropical_forward_backward():
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
            attention_backend="hybrid_flash_tropical",
            tropical_context_clamp=8.0,
            emit_attention_contact_maps=True,
        )
    )
    out = model(**{k: batch[k] for k in ["input_ids", "kind_ids", "slot_ids", "endpoint_ids", "attention_mask", "causal_mask", "labels"]})
    assert out["loss"].isfinite()
    assert "attention_metrics" in out
    assert "flash_attention/enabled" in out["attention_metrics"]
    assert "tropical_attention/top1_margin" in out["attention_metrics"]
    assert "hybrid_attention/enabled" in out["attention_metrics"]
    assert "attention_contact_maps" in out
    assert out["attention_contact_maps"].shape[:3] == (4, 1, 4)
    seq_len = batch["input_ids"].shape[1]
    assert out["attention_contact_maps"].shape[-2:] == (seq_len, seq_len)
    out["loss"].backward()
    assert any("softmax_attn.q_proj" in name and p.grad is not None for name, p in model.named_parameters())
    assert any("tropical_attn.query_trop" in name and p.grad is not None for name, p in model.named_parameters())


def test_sparse_hybrid_runs_mhta_only_on_configured_layers():
    examples = list(iter_synthetic_examples(4))
    vocab = build_vocab(examples)
    batch = RandomOrderCollator(vocab=vocab, max_seq_len=64)(examples)
    model = RandomOrderTokenGT(
        RandomOrderTokenGTConfig(
            vocab_size=len(vocab.token_to_id),
            hidden_dim=32,
            num_layers=2,
            num_heads=4,
            ffn_dim=64,
            max_seq_len=64,
            endpoint_dim=8,
            attention_backend="hybrid_flash_tropical",
            hybrid_tropical_layers=[-1],
            tropical_context_clamp=8.0,
            emit_attention_contact_maps=True,
            gradient_checkpointing=True,
        )
    )
    assert model.hybrid_tropical_layers == [1]
    out = model(**{k: batch[k] for k in ["input_ids", "kind_ids", "slot_ids", "endpoint_ids", "attention_mask", "causal_mask", "labels"]})
    assert out["loss"].isfinite()
    metrics = out["attention_metrics"]
    assert metrics["hybrid_attention/tropical_active"].item() == 0.5
    assert "flash_attention/enabled" in metrics
    assert "tropical_attention/top1_margin" in metrics
    assert out["attention_contact_maps"].shape[:3] == (4, 1, 4)
    out["loss"].backward()
    assert model.encoder.layers[0].softmax_attn.q_proj.weight.grad is not None
    assert model.encoder.layers[0].tropical_attn.query_trop.weight.grad is None
    assert model.encoder.layers[1].tropical_attn.query_trop.weight.grad is not None
