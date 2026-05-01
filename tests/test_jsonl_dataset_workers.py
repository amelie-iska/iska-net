from __future__ import annotations

import json

from torch.utils.data import DataLoader

import iska_reasoner.data.dataset as dataset_module
from iska_reasoner.data.dataset import GraphJsonlDataset


def _write_jsonl(path, rows: int = 128) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for idx in range(rows):
            row = {
                "id": f"row-{idx}",
                "task": "reader_test",
                "nodes": [
                    {
                        "id": "n0",
                        "type": "text",
                        "value": "x" * 12_000,
                        "features": {"idx": idx},
                    }
                ],
                "edges": [],
                "target_tokens": [f"target-{idx}"],
                "decoder_orders": [[0]],
                "metadata": {},
            }
            handle.write(json.dumps(row) + "\n")


def test_dataset_reopens_inherited_file_handle_for_new_process(monkeypatch, tmp_path):
    path = tmp_path / "rows.jsonl"
    _write_jsonl(path, rows=2)
    dataset = GraphJsonlDataset(path)

    assert dataset[0].id == "row-0"
    parent_handle = dataset._handle
    parent_pid = dataset._handle_pid
    assert parent_handle is not None
    assert parent_pid is not None

    monkeypatch.setattr(dataset_module.os, "getpid", lambda: parent_pid + 1)

    row = dataset._row_at(1)

    assert row["id"] == "row-1"
    assert parent_handle.closed
    assert dataset._handle is not parent_handle
    assert dataset._handle_pid == parent_pid + 1


def test_dataset_reads_with_multiple_dataloader_workers_after_parent_read(tmp_path):
    path = tmp_path / "rows.jsonl"
    _write_jsonl(path)
    dataset = GraphJsonlDataset(path)

    assert dataset[0].id == "row-0"
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        collate_fn=lambda rows: rows[0].id,
    )

    seen = set()
    for idx, row_id in enumerate(loader):
        seen.add(row_id)
        if idx >= 63:
            break

    assert len(seen) == 64


def test_dataset_reuses_offset_cache(monkeypatch, tmp_path):
    path = tmp_path / "rows.jsonl"
    _write_jsonl(path, rows=3)

    first = GraphJsonlDataset(path)
    assert len(first) == 3
    offsets_path = path.with_name(f"{path.name}.offsets.u64")
    meta_path = path.with_name(f"{path.name}.offsets.meta.json")
    assert offsets_path.exists()
    assert meta_path.exists()

    def fail_rebuild(*args, **kwargs):
        raise AssertionError("offset cache was not reused")

    monkeypatch.setattr(dataset_module, "tqdm", fail_rebuild)
    cached = GraphJsonlDataset(path)
    assert len(cached) == 3
    assert cached[2].id == "row-2"
