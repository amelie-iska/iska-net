from pathlib import Path

from iska_reasoner.data.reference_repos import combined_reference_tokens, naturelm_tokens_from_sfm, unigenx_tokens_from_repo, write_tokens
from iska_reasoner.data.synthetic import iter_synthetic_examples
from iska_reasoner.data.vocab import build_vocab, read_extra_tokens


def test_sfm_and_unigenx_reference_tokens_exist():
    sfm = Path("data/external_repos/sfm")
    unigenx = Path("data/external_repos/unigenx")
    if not sfm.exists() or not unigenx.exists():
        return
    sfm_tokens = naturelm_tokens_from_sfm(sfm)
    unigenx_tokens = unigenx_tokens_from_repo(unigenx)
    assert "<protein>" in sfm_tokens
    assert "<material>" in sfm_tokens
    assert "<coord>" in unigenx_tokens
    assert "UNIGENX:TOK:<molecule>" in unigenx_tokens


def test_extra_vocab_tokens_are_loaded(tmp_path: Path):
    token_path = tmp_path / "extra.txt"
    count = write_tokens(["<protein>", "<coord>", "UNIGENX:TOK:C"], token_path)
    assert count == 3
    extra = read_extra_tokens([token_path])
    vocab = build_vocab(iter_synthetic_examples(2), max_size=5, extra_tokens=extra)
    assert "<protein>" in vocab.token_to_id
    assert "<coord>" in vocab.token_to_id
    assert "UNIGENX:TOK:C" in vocab.token_to_id
