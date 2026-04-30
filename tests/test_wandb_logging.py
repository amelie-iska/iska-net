from __future__ import annotations

import sys
import types

from iska_reasoner.utils.logging import WandbLogger


def test_wandb_logger_coerces_tags_to_strings(monkeypatch):
    calls = {}

    fake = types.SimpleNamespace()

    def init(**kwargs):
        calls["kwargs"] = kwargs
        return object()

    fake.init = init
    fake.log = lambda *args, **kwargs: None
    fake.finish = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "wandb", fake)
    monkeypatch.setenv("WANDB_TAGS", "shell,4090")

    logger = WandbLogger(
        {
            "enabled": True,
            "project": "test",
            "mode": "disabled",
            "tags": ["random-order", 4090, None],
        },
        run_name="tag-test",
    )

    assert logger.enabled
    assert calls["kwargs"]["tags"] == ["random-order", "4090", "shell", "4090"]
