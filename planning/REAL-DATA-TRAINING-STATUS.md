# Real-Data Training Status

Date: 2026-04-29

## Suboptimal Or Incomplete Items Found

- The full public selected-split downloads exist as parquet snapshots and the manifest-aware streaming graphification path is implemented.
- Current local `data/processed/real_full_selected_mix/` is not integrity-clean: `summary.json` expects 7,328,008 examples, while the split files currently contain 3,817,202 examples. This indicates an interrupted graphification run. The full training script now blocks on `scripts/check_dataset_integrity.py` until graphification is rerun to completion.
- The default small 4090 real-data run could not be launched while another local GPU training process was holding about 19 GB of VRAM.
- The first tiny fallback run duplicated the large in-memory dataset in DataLoader workers and entered expensive verifier validation during training.
- Validation repeatedly probed `lean --version`, causing a multi-second subprocess check per validation batch.
- Real-data validation and test configs were missing capped, repeatable functionality checks.
- README and runbook did not document the real-data graphification, busy-GPU fallback, explicit validation/test, or inference smoke paths.
- The extended paper and docs needed replacement of narrow sequence-only protein terminology with Universal Graph Model (UGM).

## Implementation Plan Applied

1. Add capacity-aware dataset audit and full selected-split parquet download tooling.
2. Add streaming graphification from downloaded parquet snapshots into deterministic train/validation/test JSONL graph splits.
3. Add 4090 real-data configs and a busy-GPU-safe tiny fallback using the same expanded graph mix.
4. Disable training-time validation in the busy-GPU fallback and run validation/test explicitly after checkpoint creation.
5. Cache Lean availability/version probing in the verifier stack.
6. Add validation `max_batches` support for fast train/test/infer functionality checks.
7. Remove legacy model-type terminology and keep UGM as the model type name.
8. Update README, dataset docs, and the 4090 runbook with real commands and current local status.
9. Add an integrity gate so stale or partial graph JSONL splits cannot be mistaken for complete pretraining data.

## Implemented Artifacts

- `scripts/audit_dataset_capacity.py`
- `scripts/download_hf_selected_splits.py`
- `scripts/graphify_full_parquet_manifest.py`
- `scripts/check_dataset_integrity.py`
- `config/data/real_4090_mix.yaml`
- `config/data/real_data_tiny_mix.yaml`
- `config/train/real_4090_local.yaml`
- `config/train/real_data_tiny_local.yaml`
- `config/train/overrides/real_data_tiny_resume_250.yaml`
- `config/validate/real_4090_validation.yaml`
- `config/validate/real_data_tiny_validation.yaml`
- `config/validate/real_data_tiny_test.yaml`
- `config/inference/real_4090_inference.yaml`
- `config/inference/real_data_tiny_inference.yaml`

## Local Data Status

- `data/raw/`: bounded manifest samples and provenance.
- `data/raw_hf_full/`: 28 parquet files, about 2.8 GB.
- `data/processed/real_full_selected_mix/`: currently fails integrity check because split JSONL line counts do not match `summary.json`.
- Current actual `real_full_selected_mix` counts: 3,740,874 train, 38,107 validation, 38,221 test.
- Expected completed `real_full_selected_mix` counts after rerunning graphification: 7,181,690 train, 73,044 validation, 73,274 test.
- `data/processed/real_4090_mix/train.jsonl`: 226,293 examples.
- `data/processed/real_4090_mix/val.jsonl`: 2,293 examples.
- `data/processed/real_4090_mix/test.jsonl`: 2,271 examples.
- The real-data configs include `data/processed/reference_tokens/naturelm_unigenx_tokens.txt` and `data/processed/reference_tokens/multimodal_graph_tokens.txt`, so NatureLM/SFM, UniGenX, and the full public motif vocabulary are included in the vocabulary.

## Training And Evaluation Run

- Busy-GPU fallback model: `config/model/tiny_tokengt.yaml`.
- Dataset/config: `config/data/real_data_tiny_mix.yaml`.
- Training config: `config/train/real_data_tiny_local.yaml`.
- Final checkpoint: `outputs/real_data_tiny_local/checkpoint_final.pt`.
- Step checkpoint: `outputs/real_data_tiny_local/checkpoint_step_500.pt`.
- Validation metrics: `outputs/real_data_tiny_local/validation_metrics.json`.
- Test metrics: `outputs/real_data_tiny_local/test_metrics.json`.
- Inference smoke: `outputs/real_data_tiny_local/infer_smoke.json`.
- Quality assessment artifact: `outputs/real_data_tiny_local/quality_assess.json`.
- Readiness artifact: `outputs/real_data_tiny_local/readiness.json`.

Validation smoke over 32 batches:

- loss: `5.690464416737086`
- token accuracy: `0.759765625`
- perplexity: `296.0310705048447`

Test smoke over 32 batches:

- loss: `5.739218676681048`
- token accuracy: `0.763671875`
- perplexity: `310.8214640258172`

Inference emitted `UNIGENX:domain:molecule`, which confirms checkpoint loading and generation plumbing. It does not indicate scientific usefulness.

## Verification

- `conda run -n tokengt pytest -q`: 40 passed, 3 warnings.
- `conda run -n tokengt python scripts/quality_assess.py`: `ready_to_roll: true`.
- `conda run -n tokengt python scripts/check_readiness.py --json`: CUDA, Lean, RDKit, topology packages, SFM/NatureLM references, UniGenX references, and reference-token files are present.
- Terminology scan outside ignored external artifacts: no legacy architecture-name term, no narrow acronym term, and no sequence-only protein-model phrase.

## Ready-To-Roll Status

The codebase is ready for the next real training run after the full selected corpus passes `scripts/check_dataset_integrity.py`. The max 4090 config is ready to launch once the GPU is free and split counts match `summary.json`. The completed tiny fallback checkpoint validates the local pipeline; it does not replace the full integrity-checked pretraining run.

This is not a scientifically useful model yet. It is a real-data functionality checkpoint on a constrained fallback model.
