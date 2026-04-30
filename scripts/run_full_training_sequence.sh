#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
CONDA_ENV="${CONDA_ENV:-tokengt}"
LOG_ROOT="${LOG_ROOT:-logs/full_training_sequence}"
RUN_DIR="${RUN_DIR:-$LOG_ROOT/$RUN_ID}"
STAGE_DIR="$RUN_DIR/stages"
COMMAND_DIR="$RUN_DIR/commands"
MASTER_LOG="$RUN_DIR/master.log"
STATUS_TSV="$RUN_DIR/status.tsv"
PROGRESS_JSONL="$RUN_DIR/progress.jsonl"
SUMMARY_MD="$RUN_DIR/summary.md"

START_AT="${START_AT:-00}"
STOP_AFTER="${STOP_AFTER:-99}"
DRY_RUN="${DRY_RUN:-0}"
DEVICE="${DEVICE:-cuda}"
VALIDATION_DEVICE="${VALIDATION_DEVICE:-$DEVICE}"
MODEL_CONFIG="${MODEL_CONFIG:-config/model/max_4090_tokengt.yaml}"
FULL_SELECTED_DATA_CONFIG="${FULL_SELECTED_DATA_CONFIG:-config/data/real_full_selected_mix.yaml}"
FULL_SELECTED_TRAIN_CONFIG="${FULL_SELECTED_TRAIN_CONFIG:-config/train/real_full_selected_local.yaml}"
FULL_SELECTED_VALIDATION_CONFIG="${FULL_SELECTED_VALIDATION_CONFIG:-config/validate/real_full_selected_validation.yaml}"
FULL_SELECTED_TEST_CONFIG="${FULL_SELECTED_TEST_CONFIG:-config/validate/real_full_selected_test.yaml}"
FULL_SELECTED_INFERENCE_CONFIG="${FULL_SELECTED_INFERENCE_CONFIG:-config/inference/real_full_selected_inference.yaml}"
FULL_SELECTED_INFERENCE_OUTPUT="${FULL_SELECTED_INFERENCE_OUTPUT:-outputs/real_full_selected_local/infer_full_sequence.json}"
MAX_TOTAL_GIB="${MAX_TOTAL_GIB:-32}"
MAX_GRAPH_TOKENS="${MAX_GRAPH_TOKENS:-5000000000}"
GRAPHIFY_BATCH_SIZE="${GRAPHIFY_BATCH_SIZE:-8192}"
GRAPHIFY_PROGRESS_EVERY="${GRAPHIFY_PROGRESS_EVERY:-10000}"
COUNT_PROGRESS_EVERY="${COUNT_PROGRESS_EVERY:-100000}"
MULTIMODAL_COUNT="${MULTIMODAL_COUNT:-32}"
MULTIMODAL_INPUT_DIR="${MULTIMODAL_INPUT_DIR:-data/local/multimodal}"
STRUCTURE_DYNAMICS_INPUT_DIR="${STRUCTURE_DYNAMICS_INPUT_DIR:-data/local/structure_dynamics}"
STRUCTURE_DYNAMICS_COUNT="${STRUCTURE_DYNAMICS_COUNT:-32}"
ENABLE_STRUCTURE_TRAINING="${ENABLE_STRUCTURE_TRAINING:-0}"
SKIP_INTERPRO_MOTIF_DOWNLOAD="${SKIP_INTERPRO_MOTIF_DOWNLOAD:-0}"
SKIP_REFERENCE_REFRESH_IF_READY="${SKIP_REFERENCE_REFRESH_IF_READY:-1}"
TRAINING_FIRST="${TRAINING_FIRST:-0}"
FULL_TRAIN_EPOCHS="${FULL_TRAIN_EPOCHS:-1.0}"
FULL_TRAIN_EVAL_EVERY="${FULL_TRAIN_EVAL_EVERY:-10000}"
FULL_TRAIN_EVAL_MAX_BATCHES="${FULL_TRAIN_EVAL_MAX_BATCHES:-512}"
FULL_TRAIN_CHECKPOINT_EVERY="${FULL_TRAIN_CHECKPOINT_EVERY:-5000}"
FULL_TRAIN_NUM_WORKERS="${FULL_TRAIN_NUM_WORKERS:-8}"
FULL_TRAIN_EVAL_NUM_WORKERS="${FULL_TRAIN_EVAL_NUM_WORKERS:-2}"
FULL_TRAIN_PREFETCH_FACTOR="${FULL_TRAIN_PREFETCH_FACTOR:-4}"
WANDB_ENABLED="${WANDB_ENABLED:-1}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-iska-ugm}"
WANDB_GROUP="${WANDB_GROUP:-$RUN_ID}"
WANDB_TAGS="${WANDB_TAGS:-full-runner,phase1,phase2,tokengt,4090}"
WANDB_LOG_COMMANDS="${WANDB_LOG_COMMANDS:-1}"
INFERENCE_TEXT="${INFERENCE_TEXT:-Create a graph reasoning sketch for a protein-ligand binding task.}"
REQUIRE_UMA_WEIGHTS="${REQUIRE_UMA_WEIGHTS:-1}"
UMA_MODEL_NAME="${UMA_MODEL_NAME:-uma-s-1p2}"
UMA_TASK_NAME="${UMA_TASK_NAME:-omol}"
UMA_DEVICE="${UMA_DEVICE:-$DEVICE}"
UMA_SCORE_SMOKE="${UMA_SCORE_SMOKE:-0}"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TQDM_DYNAMIC_NCOLS="${TQDM_DYNAMIC_NCOLS:-1}"
export TQDM_MININTERVAL="${TQDM_MININTERVAL:-0.5}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export ROOT RUN_ID CONDA_ENV LOG_ROOT RUN_DIR DEVICE VALIDATION_DEVICE MODEL_CONFIG
export FULL_SELECTED_DATA_CONFIG FULL_SELECTED_TRAIN_CONFIG FULL_SELECTED_VALIDATION_CONFIG FULL_SELECTED_TEST_CONFIG FULL_SELECTED_INFERENCE_CONFIG FULL_SELECTED_INFERENCE_OUTPUT
export MAX_TOTAL_GIB MAX_GRAPH_TOKENS GRAPHIFY_BATCH_SIZE GRAPHIFY_PROGRESS_EVERY COUNT_PROGRESS_EVERY
export MULTIMODAL_COUNT MULTIMODAL_INPUT_DIR STRUCTURE_DYNAMICS_INPUT_DIR STRUCTURE_DYNAMICS_COUNT ENABLE_STRUCTURE_TRAINING SKIP_INTERPRO_MOTIF_DOWNLOAD INFERENCE_TEXT
export SKIP_REFERENCE_REFRESH_IF_READY TRAINING_FIRST WANDB_ENABLED WANDB_MODE WANDB_PROJECT WANDB_GROUP WANDB_TAGS WANDB_LOG_COMMANDS
export FULL_TRAIN_EPOCHS FULL_TRAIN_EVAL_EVERY FULL_TRAIN_EVAL_MAX_BATCHES FULL_TRAIN_CHECKPOINT_EVERY
export FULL_TRAIN_NUM_WORKERS FULL_TRAIN_EVAL_NUM_WORKERS FULL_TRAIN_PREFETCH_FACTOR
export REQUIRE_UMA_WEIGHTS UMA_MODEL_NAME UMA_TASK_NAME UMA_DEVICE UMA_SCORE_SMOKE

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*" | tee -a "$MASTER_LOG"
}

json_event() {
  local stage_id="$1"
  local stage_name="$2"
  local event="$3"
  local status="${4:-}"
  local seconds="${5:-}"
  printf '{"time":"%s","stage_id":"%s","stage_name":"%s","event":"%s","status":"%s","seconds":"%s"}\n' \
    "$(timestamp)" "$stage_id" "$stage_name" "$event" "$status" "$seconds" >> "$PROGRESS_JSONL"
}

wandb_event() {
  local stage_id="$1"
  local stage_name="$2"
  local event="$3"
  local status="${4:-}"
  local seconds="${5:-}"
  if [[ "$WANDB_ENABLED" != "1" ]]; then
    return 0
  fi
  conda run --no-capture-output -n "$CONDA_ENV" python scripts/wandb_stage_event.py \
    --stage-id "$stage_id" \
    --stage-name "$stage_name" \
    --event "$event" \
    --status "$status" \
    --seconds "$seconds" \
    --run-dir "$RUN_DIR" >> "$MASTER_LOG" 2>&1 || true
}

usage() {
  cat <<'EOF'
Run the full UGM training sequence from readiness through final QA.

Default behavior is full selected public graphification and training with
manifest-local row caps and a global graph-token guard. Progress bars stream
live through conda using --no-capture-output.

Environment overrides:
  CONDA_ENV=tokengt
  RUN_ID=YYYYMMDDTHHMMSSZ
  LOG_ROOT=logs/full_training_sequence
  START_AT=00
  STOP_AFTER=99
  DRY_RUN=1
  DEVICE=cuda
  VALIDATION_DEVICE=cuda
  MODEL_CONFIG=config/model/max_4090_tokengt.yaml
  FULL_SELECTED_DATA_CONFIG=config/data/real_full_selected_mix.yaml
  FULL_SELECTED_TRAIN_CONFIG=config/train/real_full_selected_local.yaml
  FULL_SELECTED_VALIDATION_CONFIG=config/validate/real_full_selected_validation.yaml
  FULL_SELECTED_TEST_CONFIG=config/validate/real_full_selected_test.yaml
  FULL_SELECTED_INFERENCE_CONFIG=config/inference/real_full_selected_inference.yaml
  FULL_SELECTED_INFERENCE_OUTPUT=outputs/real_full_selected_local/infer_full_sequence.json
  MAX_TOTAL_GIB=32
  MAX_GRAPH_TOKENS=5000000000
  GRAPHIFY_BATCH_SIZE=8192
  GRAPHIFY_PROGRESS_EVERY=10000
  COUNT_PROGRESS_EVERY=100000
  MULTIMODAL_COUNT=32
  MULTIMODAL_INPUT_DIR=data/local/multimodal
  STRUCTURE_DYNAMICS_INPUT_DIR=data/local/structure_dynamics
  STRUCTURE_DYNAMICS_COUNT=32
  ENABLE_STRUCTURE_TRAINING=0
  SKIP_INTERPRO_MOTIF_DOWNLOAD=0
  SKIP_REFERENCE_REFRESH_IF_READY=1
  TRAINING_FIRST=0   # set to 1 to use existing data/vocab artifacts and start training quickly
  FULL_TRAIN_EPOCHS=1.0
  FULL_TRAIN_EVAL_EVERY=10000
  FULL_TRAIN_EVAL_MAX_BATCHES=512  # set to full/0 for full validation during training
  FULL_TRAIN_CHECKPOINT_EVERY=5000
  FULL_TRAIN_NUM_WORKERS=8
  FULL_TRAIN_EVAL_NUM_WORKERS=2
  FULL_TRAIN_PREFETCH_FACTOR=4
  REQUIRE_UMA_WEIGHTS=1
  UMA_MODEL_NAME=uma-s-1p2
  UMA_TASK_NAME=omol
  UMA_DEVICE=cuda
  UMA_SCORE_SMOKE=0   # set to 1 to run a strict CCO scoring call before training
  WANDB_ENABLED=1
  WANDB_MODE=online
  WANDB_PROJECT=iska-ugm
  WANDB_GROUP=<RUN_ID>
  WANDB_TAGS=full-runner,phase1,phase2,tokengt,4090
  WANDB_LOG_COMMANDS=1

Examples:
  scripts/run_full_training_sequence.sh
  START_AT=03 scripts/run_full_training_sequence.sh
  DRY_RUN=1 scripts/run_full_training_sequence.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

mkdir -p "$STAGE_DIR" "$COMMAND_DIR"
ln -sfn "$RUN_DIR" "$LOG_ROOT/latest"

stage_should_run() {
  local stage_id="$1"
  [[ "$stage_id" < "$START_AT" ]] && return 1
  [[ "$STOP_AFTER" < "$stage_id" ]] && return 1
  return 0
}

run_stage() {
  local stage_id="$1"
  local slug="$2"
  local stage_name="$3"
  shift 3

  if ! stage_should_run "$stage_id"; then
    log "SKIP stage $stage_id $stage_name (START_AT=$START_AT STOP_AFTER=$STOP_AFTER)"
    return 0
  fi

  local command_script="$COMMAND_DIR/${stage_id}_${slug}.sh"
  local stage_log="$STAGE_DIR/${stage_id}_${slug}.log"
  cat > "$command_script" <<'BASH'
#!/usr/bin/env bash
set -Eeuo pipefail
cd "$ROOT"

cx() {
  conda run --no-capture-output -n "$CONDA_ENV" "$@"
}

command_string() {
  local out=""
  printf -v out '%q ' "$@"
  printf '%s' "$out"
}

wandb_command_event() {
  local event="$1"
  local command="$2"
  local status="${3:-}"
  local seconds="${4:-}"
  if [[ "${WANDB_ENABLED:-0}" != "1" || "${WANDB_LOG_COMMANDS:-0}" != "1" ]]; then
    return 0
  fi
  conda run --no-capture-output -n "$CONDA_ENV" python scripts/wandb_stage_event.py \
    --stage-id "${STAGE_ID:-unknown}" \
    --stage-name "${STAGE_NAME:-unknown}" \
    --event "$event" \
    --status "$status" \
    --seconds "$seconds" \
    --run-dir "$RUN_DIR" \
    --command-index "${COMMAND_INDEX:-0}" \
    --command "$command" >> "$RUN_DIR/wandb_command_events.log" 2>&1 || true
}

run() {
  COMMAND_INDEX=$((COMMAND_INDEX + 1))
  local cmd_str
  cmd_str="$(command_string "$@")"
  printf '\n[%s] $' "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  printf ' %q' "$@"
  printf '\n'
  wandb_command_event "command_start" "$cmd_str"
  local start_epoch
  start_epoch="$(date +%s)"
  set +e
  "$@"
  local status="$?"
  set -e
  local seconds=$(( $(date +%s) - start_epoch ))
  wandb_command_event "command_end" "$cmd_str" "$status" "$seconds"
  return "$status"
}

run_cx() {
  COMMAND_INDEX=$((COMMAND_INDEX + 1))
  local cmd_str
  cmd_str="conda run --no-capture-output -n ${CONDA_ENV} $(command_string "$@")"
  printf '\n[%s] $ conda run --no-capture-output -n %q' "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" "$CONDA_ENV"
  printf ' %q' "$@"
  printf '\n'
  wandb_command_event "command_start" "$cmd_str"
  local start_epoch
  start_epoch="$(date +%s)"
  set +e
  cx "$@"
  local status="$?"
  set -e
  local seconds=$(( $(date +%s) - start_epoch ))
  wandb_command_event "command_end" "$cmd_str" "$status" "$seconds"
  return "$status"
}

run_train_stage() {
  local args=(python scripts/train_stage.py "$@")
  if [[ "${WANDB_ENABLED:-0}" == "1" ]]; then
    args+=(--config config/train/overrides/wandb_online.yaml)
  fi
  run_cx "${args[@]}"
}

COMMAND_INDEX=0
BASH
  cat >> "$command_script"
  chmod +x "$command_script"

  if [[ ! -s "$STATUS_TSV" ]]; then
    printf 'stage_id\tstage_name\tstatus\tstart_utc\tend_utc\tseconds\tlog\n' > "$STATUS_TSV"
  fi

  log "START stage $stage_id $stage_name"
  json_event "$stage_id" "$stage_name" "start"
  wandb_event "$stage_id" "$stage_name" "stage_start"
  local start_epoch
  local start_utc
  start_epoch="$(date +%s)"
  start_utc="$(timestamp)"

  if [[ "$DRY_RUN" == "1" ]]; then
    log "DRY_RUN stage $stage_id $stage_name command file: $command_script"
    sed 's/^/[dry-run] /' "$command_script" | tee -a "$MASTER_LOG"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$stage_id" "$stage_name" "dry_run" "$start_utc" "$(timestamp)" "0" "$stage_log" >> "$STATUS_TSV"
    json_event "$stage_id" "$stage_name" "end" "dry_run" "0"
    wandb_event "$stage_id" "$stage_name" "stage_end" "dry_run" "0"
    return 0
  fi

  set +e
  STAGE_ID="$stage_id" STAGE_NAME="$stage_name" STAGE_SLUG="$slug" bash "$command_script" 2>&1 | tee -a "$stage_log" "$MASTER_LOG"
  local status="${PIPESTATUS[0]}"
  set -e

  local end_epoch
  end_epoch="$(date +%s)"
  local seconds=$((end_epoch - start_epoch))
  local end_utc
  end_utc="$(timestamp)"

  if [[ "$status" -eq 0 ]]; then
    log "DONE stage $stage_id $stage_name (${seconds}s)"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$stage_id" "$stage_name" "ok" "$start_utc" "$end_utc" "$seconds" "$stage_log" >> "$STATUS_TSV"
    json_event "$stage_id" "$stage_name" "end" "ok" "$seconds"
    wandb_event "$stage_id" "$stage_name" "stage_end" "ok" "$seconds"
  else
    log "FAIL stage $stage_id $stage_name status=$status (${seconds}s)"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$stage_id" "$stage_name" "fail:$status" "$start_utc" "$end_utc" "$seconds" "$stage_log" >> "$STATUS_TSV"
    json_event "$stage_id" "$stage_name" "end" "fail:$status" "$seconds"
    wandb_event "$stage_id" "$stage_name" "stage_end" "fail:$status" "$seconds"
    exit "$status"
  fi
}

cat > "$SUMMARY_MD" <<EOF
# Full Training Sequence Run

- Run ID: \`$RUN_ID\`
- Started: \`$(timestamp)\`
- Root: \`$ROOT\`
- Conda env: \`$CONDA_ENV\`
- Device: \`$DEVICE\`
- Validation device: \`$VALIDATION_DEVICE\`
- Model config: \`$MODEL_CONFIG\`
- Full selected configs: \`$FULL_SELECTED_DATA_CONFIG / $FULL_SELECTED_TRAIN_CONFIG\`
- Full graph output: \`data/processed/real_full_selected_mix\`
- Max selected graph-token budget: \`$MAX_GRAPH_TOKENS\`
- Master log: \`$MASTER_LOG\`
- Stage status: \`$STATUS_TSV\`
- Progress events: \`$PROGRESS_JSONL\`
- W&B enabled: \`$WANDB_ENABLED\`
- W&B project/group/mode: \`$WANDB_PROJECT / $WANDB_GROUP / $WANDB_MODE\`
- Full phase-1 training epochs: \`$FULL_TRAIN_EPOCHS\`
- In-training validation: every \`$FULL_TRAIN_EVAL_EVERY\` steps, max batches \`$FULL_TRAIN_EVAL_MAX_BATCHES\`
- Full phase-1 DataLoader: workers \`$FULL_TRAIN_NUM_WORKERS\`, eval workers \`$FULL_TRAIN_EVAL_NUM_WORKERS\`, prefetch factor \`$FULL_TRAIN_PREFETCH_FACTOR\`

EOF

log "Run directory: $RUN_DIR"
log "Full selected-corpus mode: manifest per-dataset caps are honored; total graph-token guard is MAX_GRAPH_TOKENS=$MAX_GRAPH_TOKENS."
log "Model config: $MODEL_CONFIG"
log "Full selected configs: data=$FULL_SELECTED_DATA_CONFIG train=$FULL_SELECTED_TRAIN_CONFIG val=$FULL_SELECTED_VALIDATION_CONFIG test=$FULL_SELECTED_TEST_CONFIG infer=$FULL_SELECTED_INFERENCE_CONFIG"
log "TQDM_DYNAMIC_NCOLS=$TQDM_DYNAMIC_NCOLS TQDM_MININTERVAL=$TQDM_MININTERVAL PYTHONUNBUFFERED=$PYTHONUNBUFFERED"
log "W&B: enabled=$WANDB_ENABLED project=$WANDB_PROJECT group=$WANDB_GROUP mode=$WANDB_MODE command_events=$WANDB_LOG_COMMANDS"
log "TRAINING_FIRST=$TRAINING_FIRST SKIP_REFERENCE_REFRESH_IF_READY=$SKIP_REFERENCE_REFRESH_IF_READY"
log "FULL_TRAIN_EPOCHS=$FULL_TRAIN_EPOCHS FULL_TRAIN_EVAL_EVERY=$FULL_TRAIN_EVAL_EVERY FULL_TRAIN_EVAL_MAX_BATCHES=$FULL_TRAIN_EVAL_MAX_BATCHES FULL_TRAIN_CHECKPOINT_EVERY=$FULL_TRAIN_CHECKPOINT_EVERY"
log "FULL_TRAIN_NUM_WORKERS=$FULL_TRAIN_NUM_WORKERS FULL_TRAIN_EVAL_NUM_WORKERS=$FULL_TRAIN_EVAL_NUM_WORKERS FULL_TRAIN_PREFETCH_FACTOR=$FULL_TRAIN_PREFETCH_FACTOR"
log "UMA: require_weights=$REQUIRE_UMA_WEIGHTS model=$UMA_MODEL_NAME task=$UMA_TASK_NAME device=$UMA_DEVICE score_smoke=$UMA_SCORE_SMOKE"

run_stage "00" "readiness" "Readiness and local capacity" <<'BASH'
run nvidia-smi
run df -h .
run free -h
run_cx python scripts/check_readiness.py
run_cx python scripts/quality_assess.py
BASH

run_stage "01" "reference_vocab" "Reference repositories and vocabularies" <<'BASH'
if [[ "$SKIP_REFERENCE_REFRESH_IF_READY" == "1" \
  && -s data/processed/reference_tokens/naturelm_unigenx_tokens.txt \
  && -s data/processed/reference_tokens/multimodal_graph_tokens.txt \
  && -s data/processed/reference_tokens/motif_graph_tokens.txt \
  && -d data/external_repos/fairchem/.git ]]; then
  printf 'Reference vocab artifacts already exist; skipping slow refresh because SKIP_REFERENCE_REFRESH_IF_READY=1.\n'
else
  run_cx python scripts/acquire_model_files.py --repo-name sfm
  run_cx python scripts/acquire_model_files.py --repo-name unigenx
  run_cx python scripts/acquire_model_files.py --repo-name fairchem
  run_cx python scripts/extract_reference_tokens.py \
    --sfm-dir data/external_repos/sfm \
    --unigenx-dir data/external_repos/unigenx \
    --output data/processed/reference_tokens/naturelm_unigenx_tokens.txt
  MOTIF_ARGS=(--download-public-motifs --output data/processed/reference_tokens/multimodal_graph_tokens.txt)
  if [[ "$SKIP_INTERPRO_MOTIF_DOWNLOAD" == "1" ]]; then
    MOTIF_ARGS+=(--skip-interpro-download)
  fi
  run_cx python scripts/build_multimodal_vocab.py "${MOTIF_ARGS[@]}"
fi
if [[ "$REQUIRE_UMA_WEIGHTS" == "1" ]]; then
  UMA_ARGS=(--repo data/external_repos/fairchem --model-name "$UMA_MODEL_NAME" --task-name "$UMA_TASK_NAME" --device "$UMA_DEVICE")
  if [[ "$UMA_SCORE_SMOKE" == "1" ]]; then
    UMA_ARGS+=(--score-smoke)
  fi
  run_cx python scripts/download_uma_weights.py "${UMA_ARGS[@]}"
else
  printf 'Skipping UMA weight preflight because REQUIRE_UMA_WEIGHTS=%s.\n' "$REQUIRE_UMA_WEIGHTS"
fi
BASH

run_stage "02" "full_data" "Full public selected-split data download, graphify, and token counts" <<'BASH'
if [[ "$TRAINING_FIRST" == "1" \
  && -s data/processed/real_full_selected_mix/train.jsonl \
  && -s data/processed/real_full_selected_mix/val.jsonl \
  && -s data/processed/real_full_selected_mix/test.jsonl \
  && -s data/processed/real_full_selected_mix/token_counts.json \
  && -s data/processed/real_full_selected_mix/integrity.json ]]; then
  printf 'Existing full selected graph corpus found; skipping download/graphify because TRAINING_FIRST=1.\n'
  if [[ ! -s config/generated/real_full_selected_context_2x.yaml ]]; then
    if [[ -s data/processed/real_full_selected_mix/context_requirements.json ]]; then
      run_cx python scripts/write_context_config_from_summary.py \
        --summary data/processed/real_full_selected_mix/context_requirements.json \
        --output config/generated/real_full_selected_context_2x.yaml \
        --context-multiplier 2.0
    else
      run_cx python scripts/inspect_context_requirements.py \
        --data-dir data/processed/real_full_selected_mix \
        --output data/processed/real_full_selected_mix/context_requirements.json \
        --context-multiplier 2.0 \
        --write-context-config config/generated/real_full_selected_context_2x.yaml
    fi
  fi
  exit 0
fi
run_cx python scripts/audit_dataset_capacity.py
run_cx python scripts/download_hf_selected_splits.py \
  --manifest data/manifests/datasets.yaml \
  --out-dir data/raw_hf_full \
  --max-total-gib "$MAX_TOTAL_GIB"
run_cx python scripts/graphify_full_parquet_manifest.py \
  --manifest data/manifests/datasets.yaml \
  --raw-full-dir data/raw_hf_full \
  --output-dir data/processed/real_full_selected_mix \
  --val-ratio 0.01 \
  --test-ratio 0.01 \
  --batch-size "$GRAPHIFY_BATCH_SIZE" \
  --progress-every "$GRAPHIFY_PROGRESS_EVERY"
run_cx python scripts/count_graph_tokens.py \
  --data-dir data/processed/real_full_selected_mix \
  --output data/processed/real_full_selected_mix/token_counts.json \
  --progress-every "$COUNT_PROGRESS_EVERY" \
  --max-model-sequence-tokens-total "$MAX_GRAPH_TOKENS"
run_cx python scripts/inspect_context_requirements.py \
  --data-dir data/processed/real_full_selected_mix \
  --output data/processed/real_full_selected_mix/context_requirements.json \
  --context-multiplier 2.0 \
  --write-context-config config/generated/real_full_selected_context_2x.yaml
run_cx python scripts/check_dataset_integrity.py \
  --data-dir data/processed/real_full_selected_mix \
  --output data/processed/real_full_selected_mix/integrity.json
BASH

run_stage "03" "full_pretraining" "Complete public graph pretraining, validation, test, and inference" <<'BASH'
if [[ ! -s config/generated/real_full_selected_context_2x.yaml ]]; then
  if [[ -s data/processed/real_full_selected_mix/context_requirements.json ]]; then
    run_cx python scripts/write_context_config_from_summary.py \
      --summary data/processed/real_full_selected_mix/context_requirements.json \
      --output config/generated/real_full_selected_context_2x.yaml \
      --context-multiplier 2.0
  else
    run_cx python scripts/inspect_context_requirements.py \
      --data-dir data/processed/real_full_selected_mix \
      --output data/processed/real_full_selected_mix/context_requirements.json \
      --context-multiplier 2.0 \
      --write-context-config config/generated/real_full_selected_context_2x.yaml
  fi
fi
run_cx python scripts/check_dataset_integrity.py \
  --data-dir data/processed/real_full_selected_mix \
  --output data/processed/real_full_selected_mix/integrity.json
FULL_TRAIN_OVERRIDE="$RUN_DIR/full_selected_training_override.yaml"
cat > "$FULL_TRAIN_OVERRIDE" <<YAML
train:
  max_steps: full_epoch
  full_epochs: $FULL_TRAIN_EPOCHS
  eval_every: $FULL_TRAIN_EVAL_EVERY
  eval_max_batches: $FULL_TRAIN_EVAL_MAX_BATCHES
  checkpoint_every: $FULL_TRAIN_CHECKPOINT_EVERY
  num_workers: $FULL_TRAIN_NUM_WORKERS
  eval_num_workers: $FULL_TRAIN_EVAL_NUM_WORKERS
  pin_memory: true
  persistent_workers: true
  prefetch_factor: $FULL_TRAIN_PREFETCH_FACTOR
YAML
printf 'Full phase-1 training override written to %s\n' "$FULL_TRAIN_OVERRIDE"
cat "$FULL_TRAIN_OVERRIDE"
run_train_stage \
  --config "$MODEL_CONFIG" \
  --config "$FULL_SELECTED_DATA_CONFIG" \
  --config config/generated/real_full_selected_context_2x.yaml \
  --config "$FULL_SELECTED_TRAIN_CONFIG" \
  --config "$FULL_TRAIN_OVERRIDE"
run_cx python scripts/validate_stage.py \
  --config "$FULL_SELECTED_VALIDATION_CONFIG" \
  --device "$VALIDATION_DEVICE"
run_cx python scripts/validate_stage.py \
  --config "$FULL_SELECTED_TEST_CONFIG" \
  --device "$VALIDATION_DEVICE"
run_cx python scripts/infer.py \
  --config "$FULL_SELECTED_INFERENCE_CONFIG" \
  --text "$INFERENCE_TEXT" \
  --device "$VALIDATION_DEVICE" \
  --output "$FULL_SELECTED_INFERENCE_OUTPUT"
BASH

run_stage "04" "science_sft" "Science SFT stage" <<'BASH'
run_train_stage \
  --config "$MODEL_CONFIG" \
  --config config/data/science_mix.yaml \
  --config config/train/science_sft_4090.yaml
run_cx python scripts/validate_stage.py \
  --config config/validate/science_validation.yaml \
  --device "$VALIDATION_DEVICE"
BASH

run_stage "05" "hebrew_sft" "Hebrew SFT stage" <<'BASH'
run_train_stage \
  --config "$MODEL_CONFIG" \
  --config config/data/hebrew_mix.yaml \
  --config config/train/hebrew_sft_4090.yaml
run_cx python scripts/validate_stage.py \
  --config config/validate/hebrew_validation.yaml \
  --device "$VALIDATION_DEVICE"
BASH

run_stage "06" "got_gflownet" "General graph-of-thought GFlowNet stage" <<'BASH'
run_train_stage \
  --config config/data/synthetic_graphs.yaml \
  --config config/train/gflownet_got_4090.yaml
run_cx python scripts/validate_gflownet.py \
  --config config/train/gflownet_got_4090.yaml \
  --config config/validate/gflownet_validation.yaml \
  --checkpoint outputs/gflownet_got_4090/gflownet_final.pt \
  --data data/processed/synthetic_graphs/train.jsonl \
  --device "$VALIDATION_DEVICE" \
  --output outputs/gflownet_got_4090/validation.json
BASH

run_stage "07" "hebrew_root_gflownet" "Hebrew root GFlowNet stage" <<'BASH'
run_train_stage \
  --config config/data/hebrew_roots.yaml \
  --config config/train/hebrew_root_gflownet_4090.yaml
run_cx python scripts/validate_gflownet.py \
  --config config/data/hebrew_roots.yaml \
  --config config/train/hebrew_root_gflownet_4090.yaml \
  --checkpoint outputs/hebrew_root_gflownet_4090/gflownet_final.pt \
  --data data/processed/hebrew_root_synthetic/train.jsonl \
  --device "$VALIDATION_DEVICE" \
  --output outputs/hebrew_root_gflownet_4090/validation.json
BASH

run_stage "08" "multimodal_phase2" "Multimodal phase-2 graph-to-graph stage" <<'BASH'
MM_ARGS=(--output data/processed/multimodal_graphs/all.jsonl --dataset-name local_multimodal_graph_to_graph --synthetic-if-empty --count "$MULTIMODAL_COUNT")
if [[ -d "$MULTIMODAL_INPUT_DIR" ]]; then
  MM_ARGS+=(--input-dir "$MULTIMODAL_INPUT_DIR")
fi
run_cx python scripts/prepare_multimodal_sources.py "${MM_ARGS[@]}"
run_cx python scripts/curate_data.py \
  --input data/processed/multimodal_graphs/all.jsonl \
  --output-dir data/processed/multimodal_graphs \
  --val-ratio 0.2 \
  --test-ratio 0.1
run_train_stage \
  --config "$MODEL_CONFIG" \
  --config config/data/multimodal_graphs_4090.yaml \
  --config config/train/multimodal_phase2_4090.yaml
run_cx python scripts/validate_stage.py \
  --config config/validate/multimodal_4090_validation.yaml \
  --device "$VALIDATION_DEVICE"
run_cx python scripts/validate_stage.py \
  --config config/validate/multimodal_4090_test.yaml \
  --device "$VALIDATION_DEVICE"
run_cx python scripts/infer.py \
  --config config/inference/multimodal_4090_inference.yaml \
  --multimodal-json '{"prompt":"Generate sequence-first graph records for a temperature-conditioned biomolecular candidate.","protein_sequence":"ACDE","selfies":"[C][=O][O]","temperature":315.5,"task":"sequence_or_selfies_reconstruction","oracle":{"name":"uma","mode":"score_candidate"}}' \
  --device "$VALIDATION_DEVICE" \
  --output outputs/multimodal_phase2_4090/infer_multimodal_sequence.json
BASH

run_stage "09" "multimodal_oracle_gflownet" "Multimodal oracle-feedback GFlowNet stage" <<'BASH'
if [[ "$REQUIRE_UMA_WEIGHTS" == "1" ]]; then
  run_cx python scripts/download_uma_weights.py \
    --repo data/external_repos/fairchem \
    --model-name "$UMA_MODEL_NAME" \
    --task-name "$UMA_TASK_NAME" \
    --device "$UMA_DEVICE"
fi
run_train_stage \
  --config config/data/multimodal_graphs_4090.yaml \
  --config config/train/multimodal_oracle_gflownet_4090.yaml
run_cx python scripts/validate_gflownet.py \
  --config config/data/multimodal_graphs_4090.yaml \
  --config config/train/multimodal_oracle_gflownet_4090.yaml \
  --config config/validate/multimodal_4090_gflownet_validation.yaml \
  --device "$VALIDATION_DEVICE"
BASH

run_stage "10" "structure_dynamics_eval_only" "Structure-side parser and validation-only smoke stage" <<'BASH'
if [[ "$ENABLE_STRUCTURE_TRAINING" != "1" ]]; then
  printf 'Structure/dynamics training is disabled by default. Preparing/evaluating structure-side rows is allowed only as validation/test/eval, not train.\n'
fi
SD_ARGS=(--output data/processed/structure_dynamics_graphs/all.jsonl --dataset-name local_structure_dynamics_graph_to_graph --synthetic-if-empty --synthetic-count "$STRUCTURE_DYNAMICS_COUNT")
if [[ -d "$STRUCTURE_DYNAMICS_INPUT_DIR" ]]; then
  SD_ARGS+=(--input-dir "$STRUCTURE_DYNAMICS_INPUT_DIR")
fi
run_cx python scripts/prepare_structure_dynamics_sources.py "${SD_ARGS[@]}" --purpose eval
run_cx python scripts/curate_data.py \
  --input data/processed/structure_dynamics_graphs/all.jsonl \
  --output-dir data/processed/structure_dynamics_graphs \
  --val-ratio 0.2 \
  --test-ratio 0.1
if [[ "$ENABLE_STRUCTURE_TRAINING" != "1" ]]; then
  printf 'Skipping structure/dynamics train_stage because ENABLE_STRUCTURE_TRAINING=%s.\n' "$ENABLE_STRUCTURE_TRAINING"
  exit 0
fi
run_train_stage \
  --config "$MODEL_CONFIG" \
  --config config/data/structure_dynamics_graphs.yaml \
  --config config/train/structure_dynamics_4090.yaml
run_cx python scripts/validate_stage.py \
  --config config/validate/structure_dynamics_validation.yaml \
  --device "$VALIDATION_DEVICE"
run_cx python scripts/validate_stage.py \
  --config config/validate/structure_dynamics_test.yaml \
  --device "$VALIDATION_DEVICE"
run_cx python scripts/infer.py \
  --config config/inference/structure_dynamics_inference.yaml \
  --multimodal-json '{"prompt":"Generate temperature-conditioned sequence-first candidate records for validation-only oracle scoring.","protein_sequence":"ACDE","selfies":"[C][=O][O]","temperature":337.25,"task":"sequence_or_selfies_reconstruction","oracle":{"name":"uma","mode":"score_candidate"}}' \
  --device "$VALIDATION_DEVICE" \
  --output outputs/structure_dynamics_4090/infer_structure_dynamics_sequence.json
BASH

run_stage "11" "structure_dynamics_oracle_gflownet" "Structure/dynamics oracle-feedback GFlowNet stage" <<'BASH'
if [[ "$ENABLE_STRUCTURE_TRAINING" != "1" ]]; then
  printf 'Skipping structure/dynamics oracle GFlowNet because ENABLE_STRUCTURE_TRAINING=%s.\n' "$ENABLE_STRUCTURE_TRAINING"
  exit 0
fi
run_train_stage \
  --config config/data/structure_dynamics_graphs.yaml \
  --config config/train/structure_dynamics_oracle_gflownet_4090.yaml
run_cx python scripts/validate_gflownet.py \
  --config config/data/structure_dynamics_graphs.yaml \
  --config config/train/structure_dynamics_oracle_gflownet_4090.yaml \
  --config config/validate/structure_dynamics_gflownet_validation.yaml \
  --device "$VALIDATION_DEVICE"
BASH

run_stage "12" "final_qa" "Final QA" <<'BASH'
run_cx pytest -q
run_cx python scripts/quality_assess.py
run_cx python scripts/check_readiness.py
run nvidia-smi
BASH

{
  printf '\n## Completed\n\n'
  printf -- '- Finished: `%s`\n' "$(timestamp)"
  printf -- '- Status file: `%s`\n' "$STATUS_TSV"
  printf -- '- Master log: `%s`\n' "$MASTER_LOG"
} >> "$SUMMARY_MD"

log "All requested stages completed. Summary: $SUMMARY_MD"
