#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Cohesive full curriculum runner:
# - Phase 1: selected public graph corpus, capped by a 5B untruncated graph-token guard.
# - Phase 2: sequence-first multimodal/function/oracle-feedback stages.
# Structure-file training remains disabled by default.
export MAX_GRAPH_TOKENS="${MAX_GRAPH_TOKENS:-5000000000}"
export MAX_TOTAL_GIB="${MAX_TOTAL_GIB:-64}"
export ENABLE_STRUCTURE_TRAINING="${ENABLE_STRUCTURE_TRAINING:-0}"
export SKIP_INTERPRO_MOTIF_DOWNLOAD="${SKIP_INTERPRO_MOTIF_DOWNLOAD:-1}"
export SKIP_REFERENCE_REFRESH_IF_READY="${SKIP_REFERENCE_REFRESH_IF_READY:-1}"
export TRAINING_FIRST="${TRAINING_FIRST:-1}"
export FULL_TRAIN_EPOCHS="${FULL_TRAIN_EPOCHS:-1.0}"
export FULL_TRAIN_EVAL_EVERY="${FULL_TRAIN_EVAL_EVERY:-10000}"
export FULL_TRAIN_EVAL_MAX_BATCHES="${FULL_TRAIN_EVAL_MAX_BATCHES:-512}"
export FULL_TRAIN_CHECKPOINT_EVERY="${FULL_TRAIN_CHECKPOINT_EVERY:-5000}"
export WANDB_ENABLED="${WANDB_ENABLED:-1}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-iska-ugm}"
export WANDB_GROUP="${WANDB_GROUP:-${RUN_ID:-full-phase1-phase2}}"
export WANDB_TAGS="${WANDB_TAGS:-full-run,phase1,phase2,tokengt,4090}"
export WANDB_LOG_COMMANDS="${WANDB_LOG_COMMANDS:-1}"
export REQUIRE_UMA_WEIGHTS="${REQUIRE_UMA_WEIGHTS:-1}"
export UMA_MODEL_NAME="${UMA_MODEL_NAME:-uma-s-1p2}"
export UMA_TASK_NAME="${UMA_TASK_NAME:-omol}"
export UMA_DEVICE="${UMA_DEVICE:-cuda}"
export UMA_SCORE_SMOKE="${UMA_SCORE_SMOKE:-0}"

exec "$ROOT/scripts/run_full_training_sequence.sh" "$@"
