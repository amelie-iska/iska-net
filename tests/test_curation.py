from pathlib import Path
import hashlib
import json

from iska_reasoner.data.curate import _curate_resume_config, _curate_resume_path, curate_files
from iska_reasoner.data.splits import split_name_from_key
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


def test_fast_curate_resume_keeps_existing_temp_rows(tmp_path: Path):
    rows = [ex.to_dict() for ex in iter_synthetic_examples(5)]
    input_path = tmp_path / "input.jsonl"
    out_dir = tmp_path / "curated"
    out_dir.mkdir()
    lines = [json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows]
    input_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    val_ratio = 0.2
    test_ratio = 0.2
    config = _curate_resume_config([input_path], val_ratio, test_ratio, "row_hash", "row_hash", "none", True)
    for split in ("train", "val", "test"):
        (out_dir / f".{split}.jsonl.tmp").write_text("", encoding="utf-8")
    for line in lines[:2]:
        h = hashlib.sha1(line.encode("utf-8")).hexdigest()
        split = split_name_from_key(f"row_hash:{h}", val_ratio, test_ratio)
        with (out_dir / f".{split}.jsonl.tmp").open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    _curate_resume_path(out_dir).write_text(
        json.dumps(
            {
                "version": 1,
                "config": config,
                "processed_rows": {str(input_path): 2},
                "counters": {
                    "input_rows": 2,
                    "invalid_rows": 0,
                    "duplicates_removed": 0,
                    "near_duplicates_removed": 0,
                    "low_quality_removed": 0,
                    "license_blocked": 0,
                    "contamination_removed": 0,
                },
            }
        ),
        encoding="utf-8",
    )

    summary = curate_files(
        [input_path],
        out_dir,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        split_policy="row_hash",
        dedup_key="row_hash",
        quality_mode="none",
        fast_copy=True,
        resume=True,
        resume_state_every=1,
    )

    assert summary["input_rows"] == 5
    assert summary["kept_rows"] == 5
    assert not _curate_resume_path(out_dir).exists()
    assert sum(1 for split in ("train", "val", "test") for _ in (out_dir / f"{split}.jsonl").open()) == 5


def test_fast_curate_resume_keeps_orphan_temp_rows_without_state(tmp_path: Path):
    rows = [ex.to_dict() for ex in iter_synthetic_examples(4)]
    input_path = tmp_path / "input.jsonl"
    out_dir = tmp_path / "curated"
    out_dir.mkdir()
    lines = [json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows]
    input_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    val_ratio = 0.2
    test_ratio = 0.2
    for split in ("train", "val", "test"):
        (out_dir / f".{split}.jsonl.tmp").write_text("", encoding="utf-8")
    h = hashlib.sha1(lines[0].encode("utf-8")).hexdigest()
    split = split_name_from_key(f"row_hash:{h}", val_ratio, test_ratio)
    (out_dir / f".{split}.jsonl.tmp").write_text(lines[0] + "\n", encoding="utf-8")

    summary = curate_files(
        [input_path],
        out_dir,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        split_policy="row_hash",
        dedup_key="row_hash",
        quality_mode="none",
        fast_copy=True,
        resume=True,
        resume_state_every=1,
    )

    assert summary["kept_rows"] == 4
    assert summary["duplicates_removed"] == 1
    assert sum(1 for split in ("train", "val", "test") for _ in (out_dir / f"{split}.jsonl").open()) == 4
