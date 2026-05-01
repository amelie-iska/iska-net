#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export MODEL_CONFIG="${MODEL_CONFIG:-config/model/ugm_250m_tokengt.yaml}"
export FULL_SELECTED_DATA_CONFIG="${FULL_SELECTED_DATA_CONFIG:-config/data/real_full_selected_mix_250m.yaml}"
export FULL_SELECTED_TRAIN_CONFIG="${FULL_SELECTED_TRAIN_CONFIG:-config/train/real_full_selected_250m_local.yaml}"
export FULL_SELECTED_VALIDATION_CONFIG="${FULL_SELECTED_VALIDATION_CONFIG:-config/validate/real_full_selected_250m_validation.yaml}"
export FULL_SELECTED_TEST_CONFIG="${FULL_SELECTED_TEST_CONFIG:-config/validate/real_full_selected_250m_test.yaml}"
export FULL_SELECTED_INFERENCE_CONFIG="${FULL_SELECTED_INFERENCE_CONFIG:-config/inference/real_full_selected_250m_inference.yaml}"
export FULL_SELECTED_INFERENCE_OUTPUT="${FULL_SELECTED_INFERENCE_OUTPUT:-outputs/real_full_selected_250m_local/infer_full_sequence.json}"
export ENABLE_TROPICAL_ATTENTION="${ENABLE_TROPICAL_ATTENTION:-0}"
export TROPICAL_ATTENTION_CONFIG="${TROPICAL_ATTENTION_CONFIG:-config/model/overrides/hybrid_flash_mhta_backend.yaml}"
if [[ "$ENABLE_TROPICAL_ATTENTION" == "1" ]]; then
  export FULL_TRAIN_BATCH_SIZE="${FULL_TRAIN_BATCH_SIZE:-2}"
  export FULL_TRAIN_EVAL_BATCH_SIZE="${FULL_TRAIN_EVAL_BATCH_SIZE:-2}"
  export FULL_TRAIN_GRAD_ACCUM="${FULL_TRAIN_GRAD_ACCUM:-18}"
  export WANDB_TAGS="${WANDB_TAGS:-full-run,phase1,phase2,tokengt,4090,250m,hybrid-flash-mhta,tropical-attention,layer-sparse-mhta,mhta}"
else
  export FULL_TRAIN_BATCH_SIZE="${FULL_TRAIN_BATCH_SIZE:-6}"
  export FULL_TRAIN_EVAL_BATCH_SIZE="${FULL_TRAIN_EVAL_BATCH_SIZE:-6}"
  export FULL_TRAIN_GRAD_ACCUM="${FULL_TRAIN_GRAD_ACCUM:-6}"
  export WANDB_TAGS="${WANDB_TAGS:-full-run,phase1,phase2,tokengt,4090,250m}"
fi
export WANDB_GROUP="${WANDB_GROUP:-${RUN_ID:-full-phase1-phase2-250m}}"

exec "$ROOT/scripts/run_full_phase1_phase2_training.sh" "$@"
