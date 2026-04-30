#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_ID="${RUN_ID:-$(date -u +"%Y%m%dT%H%M%SZ")}"
CONDA_ENV="${CONDA_ENV:-tokengt}"
LOG_ROOT="${LOG_ROOT:-logs/direct_training}"
RUN_DIR="$LOG_ROOT/$RUN_ID"
mkdir -p "$RUN_DIR"

MODEL_CONFIG="${MODEL_CONFIG:-config/model/ugm_250m_tokengt.yaml}"
DATA_CONFIG="${DATA_CONFIG:-config/data/real_full_selected_mix_250m.yaml}"
CONTEXT_CONFIG="${CONTEXT_CONFIG:-config/generated/real_full_selected_context_2x.yaml}"
TRAIN_CONFIG="${TRAIN_CONFIG:-config/train/real_full_selected_250m_local.yaml}"
WANDB_CONFIG="${WANDB_CONFIG:-config/train/overrides/wandb_online.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/real_full_selected_250m_local}"
VOCAB_PATH="${VOCAB_PATH:-$OUTPUT_DIR/vocab.jsonl}"

FULL_TRAIN_BATCH_SIZE="${FULL_TRAIN_BATCH_SIZE:-6}"
FULL_TRAIN_EVAL_BATCH_SIZE="${FULL_TRAIN_EVAL_BATCH_SIZE:-$FULL_TRAIN_BATCH_SIZE}"
FULL_TRAIN_GRAD_ACCUM="${FULL_TRAIN_GRAD_ACCUM:-6}"
FULL_TRAIN_MAX_STEPS="${FULL_TRAIN_MAX_STEPS:-full_epoch}"
FULL_TRAIN_EPOCHS="${FULL_TRAIN_EPOCHS:-1.0}"
FULL_TRAIN_EVAL_EVERY="${FULL_TRAIN_EVAL_EVERY:-10000}"
FULL_TRAIN_EVAL_MAX_BATCHES="${FULL_TRAIN_EVAL_MAX_BATCHES:-512}"
FULL_TRAIN_CHECKPOINT_EVERY="${FULL_TRAIN_CHECKPOINT_EVERY:-5000}"
FULL_TRAIN_NUM_WORKERS="${FULL_TRAIN_NUM_WORKERS:-8}"
FULL_TRAIN_EVAL_NUM_WORKERS="${FULL_TRAIN_EVAL_NUM_WORKERS:-2}"
FULL_TRAIN_PREFETCH_FACTOR="${FULL_TRAIN_PREFETCH_FACTOR:-4}"
SKIP_POLICY_CHECK="${SKIP_POLICY_CHECK:-1}"

export RUN_ID
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TQDM_DYNAMIC_NCOLS="${TQDM_DYNAMIC_NCOLS:-1}"
export TQDM_MININTERVAL="${TQDM_MININTERVAL:-0.5}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_PROJECT="${WANDB_PROJECT:-iska-ugm}"
export WANDB_GROUP="${WANDB_GROUP:-direct-250m}"
export WANDB_TAGS="${WANDB_TAGS:-direct,phase1,tokengt,4090,250m}"

for path in "$MODEL_CONFIG" "$DATA_CONFIG" "$CONTEXT_CONFIG" "$TRAIN_CONFIG" "$WANDB_CONFIG"; do
  if [[ ! -f "$path" ]]; then
    printf 'Required config not found: %s\n' "$path" >&2
    exit 1
  fi
done
if [[ ! -s "$VOCAB_PATH" ]]; then
  printf 'Expected existing vocab at %s. Run the full wrapper once, or set VOCAB_PATH to a built vocab.\n' "$VOCAB_PATH" >&2
  exit 1
fi

OVERRIDE="$RUN_DIR/direct_250m_training_override.yaml"
cat > "$OVERRIDE" <<YAML
run:
  output_dir: $OUTPUT_DIR
data:
  vocab_path: $VOCAB_PATH
  reuse_vocab: true
  skip_policy_check: $SKIP_POLICY_CHECK
train:
  max_steps: $FULL_TRAIN_MAX_STEPS
YAML
case "$(printf '%s' "$FULL_TRAIN_MAX_STEPS" | tr '[:upper:]' '[:lower:]')" in
  auto|epoch|full_epoch|full-dataset|full_dataset)
    printf '  full_epochs: %s\n' "$FULL_TRAIN_EPOCHS" >> "$OVERRIDE"
    ;;
  *)
    printf '  full_epochs: null\n' >> "$OVERRIDE"
    ;;
esac
cat >> "$OVERRIDE" <<YAML
  eval_every: $FULL_TRAIN_EVAL_EVERY
  eval_max_batches: $FULL_TRAIN_EVAL_MAX_BATCHES
  checkpoint_every: $FULL_TRAIN_CHECKPOINT_EVERY
  num_workers: $FULL_TRAIN_NUM_WORKERS
  eval_num_workers: $FULL_TRAIN_EVAL_NUM_WORKERS
  pin_memory: true
  persistent_workers: true
  prefetch_factor: $FULL_TRAIN_PREFETCH_FACTOR
  batch_size: $FULL_TRAIN_BATCH_SIZE
  eval_batch_size: $FULL_TRAIN_EVAL_BATCH_SIZE
  gradient_accumulation_steps: $FULL_TRAIN_GRAD_ACCUM
YAML

printf '[%s] Direct 250M phase-1 training\n' "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
printf 'Run directory: %s\n' "$RUN_DIR"
printf 'Override:\n'
cat "$OVERRIDE"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'DRY_RUN=1; not launching train_stage.py\n'
  exit 0
fi

exec conda run --no-capture-output -n "$CONDA_ENV" python scripts/train_stage.py \
  --config "$MODEL_CONFIG" \
  --config "$DATA_CONFIG" \
  --config "$CONTEXT_CONFIG" \
  --config "$TRAIN_CONFIG" \
  --config "$OVERRIDE" \
  --config "$WANDB_CONFIG"
