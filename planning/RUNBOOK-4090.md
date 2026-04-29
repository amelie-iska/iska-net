# 4090 Training, Validation, and Inference Runbook

Date: 2026-04-29

This runbook assumes one RTX 4090-class GPU, the `tokengt` conda environment, and the corrected SFM/NatureLM plus UniGenX integration.

## 1. Readiness Check

```bash
conda run -n tokengt python scripts/check_readiness.py
```

Expected: CUDA available, SFM/UniGenX reference repos present, reference-token file present, and optional Python packages importable. Lean may be unavailable; Lean metrics degrade gracefully unless you install a Lean 4 toolchain.

## 2. W&B

Online run:

```bash
wandb login
```

Use either a train config that already enables W&B, such as `config/train/science_sft_4090.yaml`, or overlay W&B onto any smoke config:

```bash
conda run -n tokengt python scripts/train_stage.py \
  --config config/model/tiny_lora_checkpointed.yaml \
  --config config/data/science_mix.yaml \
  --config config/train/science_sft_tiny.yaml \
  --config config/train/overrides/wandb_online.yaml
```

Offline debug run:

```bash
conda run -n tokengt python scripts/train_stage.py \
  --config config/model/tiny_lora_checkpointed.yaml \
  --config config/data/science_mix.yaml \
  --config config/train/science_sft_tiny.yaml \
  --config config/train/overrides/wandb_offline.yaml
```

## 3. Scale Data Deliberately

Before increasing limits, read `planning/LICENSE-REVIEW.md`.

Audit, download, and graphify the full real-data selected-split corpus:

```bash
conda run -n tokengt python scripts/audit_dataset_capacity.py

conda run -n tokengt python scripts/download_hf_selected_splits.py \
  --manifest data/manifests/datasets.yaml \
  --out-dir data/raw_hf_full \
  --max-total-gib 32

conda run -n tokengt python scripts/graphify_full_parquet_manifest.py \
  --manifest data/manifests/datasets.yaml \
  --raw-full-dir data/raw_hf_full \
  --output-dir data/processed/real_full_selected_mix \
  --val-ratio 0.01 \
  --test-ratio 0.01 \
  --batch-size 8192 \
  --progress-every 10000
```

The April 29, 2026 full selected public corpus contains 7,181,690 train examples, 73,044 validation examples, and 73,274 test examples. It includes NatureLM/SFM and UniGenX reference vocabulary through `data/processed/reference_tokens/naturelm_unigenx_tokens.txt`.

Science smoke-to-small scale:

```bash
conda run -n tokengt python scripts/acquire_datasets.py --dataset unigenx_qm9_train --limit 256
conda run -n tokengt python scripts/acquire_datasets.py --dataset unigenx_materials_crystal_system --limit 256

conda run -n tokengt python scripts/graphify_data.py \
  --input data/raw/unigenx_qm9_train/train.jsonl \
  --output data/processed/unigenx_qm9_train/train.jsonl \
  --dataset-name unigenx_qm9_train

conda run -n tokengt python scripts/graphify_data.py \
  --input data/raw/unigenx_materials_crystal_system/train.jsonl \
  --output data/processed/unigenx_materials_crystal_system/train.jsonl \
  --dataset-name unigenx_materials_crystal_system

conda run -n tokengt python scripts/merge_jsonl.py \
  --input data/processed/unigenx_qm9_train/train.jsonl \
  --input data/processed/unigenx_materials_crystal_system/train.jsonl \
  --output data/processed/science_mix/all.jsonl

conda run -n tokengt python scripts/curate_data.py \
  --input data/processed/science_mix/all.jsonl \
  --output-dir data/processed/science_mix \
  --val-ratio 0.1 \
  --test-ratio 0.05
```

## 4. Train

Full selected-corpus 4090 run:

```bash
conda run -n tokengt python scripts/train_stage.py \
  --config config/model/small_4090_tokengt.yaml \
  --config config/data/real_full_selected_mix.yaml \
  --config config/train/real_full_selected_local.yaml
```

Run validation explicitly after the full checkpoint exists.

Science 4090 run:

```bash
conda run -n tokengt python scripts/train_stage.py \
  --config config/model/small_4090_tokengt.yaml \
  --config config/data/science_mix.yaml \
  --config config/train/science_sft_4090.yaml
```

Hebrew 4090 run:

```bash
conda run -n tokengt python scripts/train_stage.py \
  --config config/model/small_4090_tokengt.yaml \
  --config config/data/hebrew_mix.yaml \
  --config config/train/hebrew_sft_4090.yaml
```

General graph-of-thought GFlowNet run:

```bash
conda run -n tokengt python scripts/train_stage.py \
  --config config/data/synthetic_graphs.yaml \
  --config config/train/gflownet_got_4090.yaml
```

Hebrew root GFlowNet run:

```bash
conda run -n tokengt python scripts/train_stage.py \
  --config config/data/hebrew_roots.yaml \
  --config config/train/hebrew_root_gflownet_4090.yaml
```

Multimodal phase-2 4090 run:

```bash
conda run -n tokengt python scripts/build_multimodal_vocab.py \
  --download-public-motifs \
  --output data/processed/reference_tokens/multimodal_graph_tokens.txt

conda run -n tokengt python scripts/prepare_multimodal_sources.py \
  --input-dir data/local/multimodal \
  --synthetic-if-empty \
  --output data/processed/multimodal_graphs/all.jsonl

conda run -n tokengt python scripts/curate_data.py \
  --input data/processed/multimodal_graphs/all.jsonl \
  --output-dir data/processed/multimodal_graphs \
  --val-ratio 0.2 \
  --test-ratio 0.1

conda run -n tokengt python scripts/train_stage.py \
  --config config/model/max_4090_tokengt.yaml \
  --config config/data/multimodal_graphs_4090.yaml \
  --config config/train/multimodal_phase2_4090.yaml
```

The vocabulary command builds both `motif_graph_tokens.txt` and `multimodal_graph_tokens.txt`. The current full public motif build contains 74,789 motif records and 148,669 motif tokens from core defaults, PROSITE, InterPro, CATH, and Rfam; use `SKIP_INTERPRO_MOTIF_DOWNLOAD=1` in `scripts/run_full_training_sequence.sh` only for a deliberate fast smoke run.

Oracle-feedback GFlowNet 4090 run:

```bash
conda run -n tokengt python scripts/train_stage.py \
  --config config/data/multimodal_graphs_4090.yaml \
  --config config/train/multimodal_oracle_gflownet_4090.yaml
```

Structure/dynamics evaluation-only gate:

```bash
conda run -n tokengt python scripts/prepare_structure_dynamics_sources.py \
  --input-dir data/local/structure_dynamics \
  --purpose eval \
  --synthetic-if-empty \
  --output data/processed/structure_dynamics_graphs/all.jsonl

conda run -n tokengt python scripts/curate_data.py \
  --input data/processed/structure_dynamics_graphs/all.jsonl \
  --output-dir data/processed/structure_dynamics_graphs \
  --val-ratio 0.2 \
  --test-ratio 0.1
```

Do not train `structure_dynamics_4090` or `structure_dynamics_oracle_gflownet_4090` in the first run. The active structure-dynamics proxy training is the sequence-only multimodal/GFlowNet path with continuous temperature, UMA oracle reward, attention/coupling bins, token-motion priors, and function-description grounding.

## 5. Validate

The current max 4090 profile is `576,767,128` trainable parameters at vocab size `262144`, `max_seq_len: 1024`, `hidden_dim: 1024`, `num_layers: 24`, `num_heads: 16`, and `ffn_dim: 4096`.

Full selected-corpus validation:

```bash
conda run -n tokengt python scripts/validate_stage.py \
  --config config/validate/real_full_selected_validation.yaml \
  --device cpu
```

Full selected-corpus test:

```bash
conda run -n tokengt python scripts/validate_stage.py \
  --config config/validate/real_full_selected_test.yaml \
  --device cpu
```

```bash
conda run -n tokengt python scripts/validate_stage.py \
  --config config/validate/science_validation.yaml \
  --checkpoint outputs/science_sft_4090/checkpoint_final.pt \
  --vocab outputs/science_sft_4090/vocab.jsonl \
  --device cuda
```

For GFlowNet:

```bash
conda run -n tokengt python scripts/validate_gflownet.py \
  --config config/data/hebrew_roots.yaml \
  --config config/train/hebrew_root_gflownet_4090.yaml \
  --checkpoint outputs/hebrew_root_gflownet_4090/gflownet_final.pt \
  --data data/processed/hebrew_root_synthetic/train.jsonl \
  --device cuda \
  --output outputs/hebrew_root_gflownet_4090/validation.json
```

For multimodal phase-2:

```bash
conda run -n tokengt python scripts/validate_stage.py \
  --config config/validate/multimodal_4090_validation.yaml \
  --device cuda

conda run -n tokengt python scripts/validate_stage.py \
  --config config/validate/multimodal_4090_test.yaml \
  --device cuda
```

For multimodal oracle-feedback GFlowNet:

```bash
conda run -n tokengt python scripts/validate_gflownet.py \
  --config config/data/multimodal_graphs_4090.yaml \
  --config config/train/multimodal_oracle_gflownet_4090.yaml \
  --config config/validate/multimodal_4090_gflownet_validation.yaml \
  --device cuda
```

For structure/dynamics evaluation-only audits:

```bash
conda run -n tokengt python scripts/validate_stage.py \
  --config config/validate/structure_dynamics_validation.yaml \
  --device cuda

conda run -n tokengt python scripts/validate_stage.py \
  --config config/validate/structure_dynamics_test.yaml \
  --device cuda

conda run -n tokengt python scripts/validate_gflownet.py \
  --config config/data/structure_dynamics_graphs.yaml \
  --config config/train/structure_dynamics_oracle_gflownet_4090.yaml \
  --config config/validate/structure_dynamics_gflownet_validation.yaml \
  --device cuda
```

## 6. Infer

Full selected-corpus inference:

```bash
conda run -n tokengt python scripts/infer.py \
  --config config/inference/real_full_selected_inference.yaml \
  --text "Create a graph reasoning sketch for a protein ligand binding question." \
  --max-steps 8 \
  --device cpu \
  --output outputs/real_full_selected_local/infer_smoke.json
```

Run this after `outputs/real_full_selected_local/checkpoint_final.pt` exists.

Update `config/inference/tiny_inference.yaml` or pass checkpoint/vocab arguments if supported by the current CLI. Minimal smoke inference:

```bash
conda run -n tokengt python scripts/infer.py \
  --config config/inference/tiny_inference.yaml \
  --text "Create a graph reasoning sketch for a protein-to-molecule design task." \
  --device cuda
```

For file-based inference:

```bash
conda run -n tokengt python scripts/infer.py \
  --config config/inference/multimodal_4090_inference.yaml \
  --multimodal-json-file data/processed/multimodal_graphs/example_infer_row.json \
  --output outputs/multimodal_phase2_4090/infer.json \
  --device cuda
```

Multimodal UGM inference:

```bash
conda run -n tokengt python scripts/infer.py \
  --config config/inference/multimodal_4090_inference.yaml \
  --prompt "Generate graph records for a mixed protein and ligand input." \
  --protein-sequence "MKTW" \
  --selfies "[C][=O][O]" \
  --dna-sequence "ATGC" \
  --temperature-k 300 \
  --max-steps 8 \
  --device cuda
```

Ready-to-roll assessment:

```bash
conda run -n tokengt python scripts/quality_assess.py
```

## 7. Watchpoints

- If W&B online fails, use `config/train/overrides/wandb_offline.yaml`.
- If a run OOMs, reduce `batch_size` first, then `max_seq_len`, then model depth.
- Keep `reuse_vocab: false` for science configs when reference tokens change.
- Do not scale sparse-license datasets before `planning/LICENSE-REVIEW.md` is updated.
