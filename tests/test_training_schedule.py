from __future__ import annotations

from iska_reasoner.training.stage_runner import _optional_positive_int, _resolve_total_steps


def test_full_epoch_steps_use_effective_batch():
    cfg = {
        "train": {
            "batch_size": 1,
            "gradient_accumulation_steps": 32,
            "max_steps": "full_epoch",
            "full_epochs": 1.0,
        }
    }

    assert _resolve_total_steps(cfg, train_examples=7_181_690) == 224_428


def test_fractional_full_epochs_round_up():
    cfg = {
        "train": {
            "batch_size": 2,
            "gradient_accumulation_steps": 4,
            "max_steps": "full_epoch",
            "full_epochs": 1.5,
        }
    }

    assert _resolve_total_steps(cfg, train_examples=100) == 20


def test_explicit_max_steps_still_supported():
    cfg = {
        "train": {
            "batch_size": 1,
            "gradient_accumulation_steps": 32,
            "max_steps": 2000,
        }
    }

    assert _resolve_total_steps(cfg, train_examples=7_181_690) == 2000


def test_optional_positive_int_accepts_full_sentinel():
    assert _optional_positive_int("full") is None
    assert _optional_positive_int("0") is None
    assert _optional_positive_int(512) == 512
