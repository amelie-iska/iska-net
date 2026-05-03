import json
from pathlib import Path

from scripts.repair_jsonl_concatenation import repair_jsonl


def test_repair_jsonl_concatenation_splits_appended_objects_and_clears_offsets(tmp_path: Path):
    path = tmp_path / "train.jsonl"
    rows = [{"a": 1}, {"b": 2}, {"c": 3}]
    path.write_text(json.dumps(rows[0]) + json.dumps(rows[1]) + "\n" + json.dumps(rows[2]) + "\n", encoding="utf-8")
    (tmp_path / "train.jsonl.offsets.u64").write_bytes(b"stale")
    (tmp_path / "train.jsonl.offsets.meta.json").write_text("{}", encoding="utf-8")

    summary = repair_jsonl(path)

    assert summary["input_lines"] == 2
    assert summary["output_lines"] == 3
    assert summary["repaired_lines"] == 1
    assert summary["line_delta"] == 1
    assert [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] == rows
    assert not (tmp_path / "train.jsonl.offsets.u64").exists()
    assert not (tmp_path / "train.jsonl.offsets.meta.json").exists()
