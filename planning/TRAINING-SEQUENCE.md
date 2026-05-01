# End-to-End Training Sequence

This is the command sequence for running the project from readiness checks through complete selected public graph pretraining and all implemented follow-on stages.

The complete selected public corpus target is `data/processed/real_full_selected_mix/`. The previous completed baseline had 7,328,008 graph examples and 1,107,708,497 untruncated model-sequence graph tokens. The expanded default now adds ranked graph-reasoning, scientific sequence/function, math verifier, and DCLM-slice sources under `MAX_GRAPH_TOKENS=5000000000`; final expanded counts are produced by the next full run. The full corpus does not include gated or user-provided local exports; see `planning/FULL-PRETRAINING-DATASET.md` for the completeness boundary.

Before starting GPU training, verify that the 4090 is free:

```bash
nvidia-smi
```

To run every stage in order with per-stage logs, a master log, progress events, runner-level W&B events, and live tqdm streaming through `conda run --no-capture-output`, use:

```bash
scripts/run_full_training_sequence.sh
```

For the full phase 1 plus phase 2 curriculum with the correct budget defaults, W&B online logging, and training-first behavior:

```bash
scripts/run_full_phase1_phase2_training.sh
```

Logs are written under `logs/full_training_sequence/<RUN_ID>/`, with `logs/full_training_sequence/latest` pointing at the newest run. The script uses the full selected public graph corpus, honors manifest-local row caps, and enforces the 5B graph-token guard. Standard softmax training writes or uses the 2x context config at `config/generated/real_full_selected_context_2x.yaml`; `ENABLE_TROPICAL_ATTENTION=1` writes or uses the compact exact-coverage context config at `config/generated/real_full_selected_context_compact.yaml` to avoid wasting quadratic MHTA memory on unused context. The phase wrapper now defaults to `TRAINING_FIRST=1`, `SKIP_REFERENCE_REFRESH_IF_READY=1`, `SKIP_INTERPRO_MOTIF_DOWNLOAD=1`, and `WANDB_ENABLED=1`, so an already-prepared workspace proceeds to training quickly and reports both shell-stage progress and training metrics to W&B. Useful controls:

```bash
DRY_RUN=1 scripts/run_full_training_sequence.sh
START_AT=03 scripts/run_full_training_sequence.sh
STOP_AFTER=02 scripts/run_full_training_sequence.sh
VALIDATION_DEVICE=cpu scripts/run_full_training_sequence.sh
MAX_GRAPH_TOKENS=4500000000 scripts/run_full_phase1_phase2_training.sh
WANDB_MODE=offline scripts/run_full_phase1_phase2_training.sh
WANDB_ENABLED=0 scripts/run_full_phase1_phase2_training.sh
TRAINING_FIRST=0 SKIP_INTERPRO_MOTIF_DOWNLOAD=0 scripts/run_full_phase1_phase2_training.sh
ENABLE_TROPICAL_ATTENTION=1 FULL_TRAIN_BATCH_SIZE=2 FULL_TRAIN_GRAD_ACCUM=18 scripts/run_full_phase1_phase2_training_250m.sh
ENABLE_TROPICAL_ATTENTION=1 scripts/train_full_selected_250m_direct.sh
```

`ENABLE_TROPICAL_ATTENTION=1` defaults to the hybrid Flash-eligible SDPA plus layer-sparse MHTA backend in `config/model/overrides/hybrid_flash_mhta_backend.yaml`. That override runs FlashAttention/SDPA in every layer and MHTA only on `hybrid_tropical_layers: [-1]` by default. Use `TROPICAL_ATTENTION_CONFIG=config/model/overrides/tropical_attention_backend.yaml` only when running a pure-MHTA ablation. The direct 250M wrapper skips the slow readiness, integrity, policy, and vocabulary pregame when `OUTPUT_DIR/vocab.jsonl` already exists.

## 0. Readiness

```bash
conda run -n tokengt python scripts/check_readiness.py
conda run -n tokengt python scripts/quality_assess.py
```

## 1. Reference Repositories And Vocabularies

```bash
conda run -n tokengt python scripts/acquire_model_files.py --repo-name sfm
conda run -n tokengt python scripts/acquire_model_files.py --repo-name unigenx
conda run -n tokengt python scripts/acquire_model_files.py --repo-name tropical_attention

conda run -n tokengt python scripts/extract_reference_tokens.py \
  --sfm-dir data/external_repos/sfm \
  --unigenx-dir data/external_repos/unigenx \
  --output data/processed/reference_tokens/naturelm_unigenx_tokens.txt

conda run -n tokengt python scripts/build_multimodal_vocab.py \
  --download-public-motifs \
  --output data/processed/reference_tokens/multimodal_graph_tokens.txt
```

The vocabulary step also writes `data/processed/reference_tokens/motif_graph_tokens.txt` and `data/processed/reference_tokens/motif_graph_tokens.summary.json`. The current full public motif build has 74,789 motif records and 148,669 motif tokens from core defaults, PROSITE, InterPro, CATH, and Rfam.

## 2. Full Public Selected-Split Corpus

Audit capacity:

```bash
conda run -n tokengt python scripts/audit_dataset_capacity.py
```

Download every public selected split exposed as Hugging Face parquet:

```bash
conda run -n tokengt python scripts/download_hf_selected_splits.py \
  --manifest data/manifests/datasets.yaml \
  --out-dir data/raw_hf_full \
  --max-total-gib 32
```

Graphify the selected corpus. This command honors manifest-local row caps and has global, dataset-level, and parquet-file tqdm progress bars plus live split counters:

```bash
conda run -n tokengt python scripts/graphify_full_parquet_manifest.py \
  --manifest data/manifests/datasets.yaml \
  --raw-full-dir data/raw_hf_full \
  --output-dir data/processed/real_full_selected_mix \
  --val-ratio 0.01 \
  --test-ratio 0.01 \
  --batch-size 8192 \
  --progress-every 10000
```

Count graph tokens with split-aware tqdm totals and enforce the default 5B untruncated graph-token cap:

```bash
conda run -n tokengt python scripts/count_graph_tokens.py \
  --data-dir data/processed/real_full_selected_mix \
  --output data/processed/real_full_selected_mix/token_counts.json \
  --progress-every 100000 \
  --max-model-sequence-tokens-total 5000000000
```

Generate the context-window audit and config at roughly twice the largest row:

```bash
conda run -n tokengt python scripts/inspect_context_requirements.py \
  --data-dir data/processed/real_full_selected_mix \
  --output data/processed/real_full_selected_mix/context_requirements.json \
  --context-multiplier 2.0 \
  --write-context-config config/generated/real_full_selected_context_2x.yaml
```

Verify the split files match `summary.json` before training:

```bash
conda run -n tokengt python scripts/check_dataset_integrity.py \
  --data-dir data/processed/real_full_selected_mix \
  --output data/processed/real_full_selected_mix/integrity.json
```

## 3. Complete Public Graph Pretraining

```bash
conda run -n tokengt python scripts/train_stage.py \
  --config config/model/max_4090_tokengt.yaml \
  --config config/data/real_full_selected_mix.yaml \
  --config config/generated/real_full_selected_context_2x.yaml \
  --config config/train/real_full_selected_local.yaml
```

Validation and test:

```bash
conda run -n tokengt python scripts/validate_stage.py \
  --config config/validate/real_full_selected_validation.yaml

conda run -n tokengt python scripts/validate_stage.py \
  --config config/validate/real_full_selected_test.yaml
```

Inference after the full checkpoint exists:

```bash
conda run -n tokengt python scripts/infer.py \
  --config config/inference/real_full_selected_inference.yaml \
  --text "Create a graph reasoning sketch for a protein-ligand binding task." \
  --output outputs/real_full_selected_local/infer_smoke.json
```

## 4. Science SFT Stage

```bash
conda run -n tokengt python scripts/train_stage.py \
  --config config/model/max_4090_tokengt.yaml \
  --config config/data/science_mix.yaml \
  --config config/train/science_sft_4090.yaml

conda run -n tokengt python scripts/validate_stage.py \
  --config config/validate/science_validation.yaml
```

## 5. Hebrew SFT Stage

```bash
conda run -n tokengt python scripts/train_stage.py \
  --config config/model/max_4090_tokengt.yaml \
  --config config/data/hebrew_mix.yaml \
  --config config/train/hebrew_sft_4090.yaml

conda run -n tokengt python scripts/validate_stage.py \
  --config config/validate/hebrew_validation.yaml
```

## 6. General Graph-Of-Thought GFlowNet Stage

```bash
conda run -n tokengt python scripts/train_stage.py \
  --config config/data/synthetic_graphs.yaml \
  --config config/train/gflownet_got_4090.yaml

conda run -n tokengt python scripts/validate_gflownet.py \
  --config config/train/gflownet_got_4090.yaml \
  --config config/validate/gflownet_validation.yaml \
  --checkpoint outputs/gflownet_got_4090/gflownet_final.pt \
  --data data/processed/synthetic_graphs/train.jsonl \
  --device cuda \
  --output outputs/gflownet_got_4090/validation.json
```

## 7. Hebrew Root GFlowNet Stage

```bash
conda run -n tokengt python scripts/train_stage.py \
  --config config/data/hebrew_roots.yaml \
  --config config/train/hebrew_root_gflownet_4090.yaml

conda run -n tokengt python scripts/validate_gflownet.py \
  --config config/data/hebrew_roots.yaml \
  --config config/train/hebrew_root_gflownet_4090.yaml \
  --checkpoint outputs/hebrew_root_gflownet_4090/gflownet_final.pt \
  --data data/processed/hebrew_root_synthetic/train.jsonl \
  --device cuda \
  --output outputs/hebrew_root_gflownet_4090/validation.json
```

## 8. Multimodal Phase-2 Graph-To-Graph Stage

Prepare the multimodal graph set from local reviewed sources when present. If `data/local/multimodal/` is empty, this falls back to synthetic smoke data so the code path remains testable:

```bash
conda run -n tokengt python scripts/prepare_multimodal_sources.py \
  --input-dir data/local/multimodal \
  --synthetic-if-empty \
  --output data/processed/multimodal_graphs/all.jsonl

conda run -n tokengt python scripts/curate_data.py \
  --input data/processed/multimodal_graphs/all.jsonl \
  --output-dir data/processed/multimodal_graphs \
  --val-ratio 0.2 \
  --test-ratio 0.1
```

Train and validate:

```bash
conda run -n tokengt python scripts/train_stage.py \
  --config config/model/max_4090_tokengt.yaml \
  --config config/data/multimodal_graphs_4090.yaml \
  --config config/train/multimodal_phase2_4090.yaml

conda run -n tokengt python scripts/validate_stage.py \
  --config config/validate/multimodal_4090_validation.yaml

conda run -n tokengt python scripts/validate_stage.py \
  --config config/validate/multimodal_4090_test.yaml

conda run -n tokengt python scripts/infer.py \
  --config config/inference/multimodal_4090_inference.yaml \
  --multimodal-json-file path/to/example_multimodal_row.json \
  --render-input-pdb \
  --output outputs/multimodal_phase2_4090/infer_multimodal_sequence.json
```

## 9. Multimodal Oracle-Feedback GFlowNet Stage

```bash
conda run -n tokengt python scripts/train_stage.py \
  --config config/data/multimodal_graphs_4090.yaml \
  --config config/train/multimodal_oracle_gflownet_4090.yaml

conda run -n tokengt python scripts/validate_gflownet.py \
  --config config/data/multimodal_graphs_4090.yaml \
  --config config/train/multimodal_oracle_gflownet_4090.yaml \
  --config config/validate/multimodal_4090_gflownet_validation.yaml \
  --device cuda
```

## 10. Structure Prediction And Dynamics Evaluation Gate

The current first-run curriculum does not train on actual structure or dynamics files. Prepare local structure/dynamics graph records only for validation, test, leakage audits, parser checks, or future-phase dry runs. Complete structure-file training remains gated by an explicit later approval. Generated candidate atom/bond/coordinate/frame records remain valid graph-state outputs when produced from sequence inputs and scored by UMA/verifier feedback; generated-token PDB rendering is not required in this pass.

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

conda run -n tokengt python scripts/validate_stage.py \
  --config config/validate/structure_dynamics_validation.yaml

conda run -n tokengt python scripts/validate_stage.py \
  --config config/validate/structure_dynamics_test.yaml

conda run -n tokengt python scripts/infer.py \
  --config config/inference/structure_dynamics_inference.yaml \
  --multimodal-json-file path/to/example_structure_dynamics_row.json \
  --render-input-pdb \
  --output outputs/structure_dynamics_4090/infer_structure_dynamics_sequence.json
```

## 11. Structure/Dynamics Oracle-Feedback Future Gate

This stage is intentionally disabled for first-run training. UMA-style feedback for the active curriculum is handled through sequence-only multimodal graph states: `ATTN_BIN:*`, `TOKEN_COUPLING:uma:*`, `UMA_INFLUENCE:uma:*`, `TOKEN_MOTION:uma:*`, `UMA_TRAJ_BIN:*`, `SEQ_STRUCT_DYN_PROXY:*`, continuous temperature features, and GFlowNet trajectory-balance over graph-state records. Those fine-bin targets are emitted only in the UMA structure-dynamics-proxy stage; ordinary sequence and function-description rows keep temperature/function nodes without the 64-bin UMA proxy targets.

```bash
conda run -n tokengt python scripts/validate_gflownet.py \
  --config config/data/structure_dynamics_graphs.yaml \
  --config config/train/structure_dynamics_oracle_gflownet_4090.yaml \
  --config config/validate/structure_dynamics_gflownet_validation.yaml \
  --device cuda
```

## 12. Final QA

```bash
conda run -n tokengt pytest -q
conda run -n tokengt python scripts/quality_assess.py
conda run -n tokengt python scripts/check_readiness.py
```
