from pathlib import Path

from iska_reasoner.data.curate import curate_files
from iska_reasoner.data.synthetic import iter_synthetic_examples
from iska_reasoner.utils.io import write_jsonl


def test_curate_dedup_and_split(tmp_path: Path):
    examples = [ex.to_dict() for ex in iter_synthetic_examples(6)]
    rows = examples + [examples[0]]
    input_path = tmp_path / "input.jsonl"
    out_dir = tmp_path / "curated"
    write_jsonl(input_path, rows)
    summary = curate_files([input_path], out_dir, val_ratio=0.2, test_ratio=0.2)
    assert summary["input_rows"] == 7
    assert summary["duplicates_removed"] == 1
    assert summary["kept_rows"] == 6
    assert (out_dir / "train.jsonl").exists()
    assert (out_dir / "summary.json").exists()

