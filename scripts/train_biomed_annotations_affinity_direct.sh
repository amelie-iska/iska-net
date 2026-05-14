#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_ID="${RUN_ID:-$(date -u +"%Y%m%dT%H%M%SZ")}"
CONDA_ENV="${CONDA_ENV:-tokengt}"
LOG_ROOT="${LOG_ROOT:-logs/biomed_direct_training}"
RUN_DIR="$LOG_ROOT/$RUN_ID"
mkdir -p "$RUN_DIR"

MODEL_CONFIG="${MODEL_CONFIG:-config/model/ugm_250m_tokengt.yaml}"
DATA_CONFIG="${DATA_CONFIG:-config/data/biomed_annotations_affinity_250m.yaml}"
TRAIN_CONFIG="${TRAIN_CONFIG:-config/train/biomed_annotations_affinity_250m.yaml}"
WANDB_CONFIG="${WANDB_CONFIG:-config/train/overrides/wandb_online.yaml}"

INCLUDE_ORIGINAL_FULL_SELECTED="${INCLUDE_ORIGINAL_FULL_SELECTED:-0}"
case "$(printf '%s' "$INCLUDE_ORIGINAL_FULL_SELECTED" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|on) INCLUDE_ORIGINAL_FULL_SELECTED=1 ;;
  *) INCLUDE_ORIGINAL_FULL_SELECTED=0 ;;
esac
DATA_DIR_WAS_SET="${DATA_DIR+x}"
DATA_DIR="${DATA_DIR:-data/processed/biomed_annotations_affinity}"
if [[ "$INCLUDE_ORIGINAL_FULL_SELECTED" == "1" && -z "$DATA_DIR_WAS_SET" ]]; then
  DATA_DIR="data/processed/biomed_annotations_affinity_plus_original_full_selected"
fi
ORIGINAL_FULL_SELECTED_DIR="${ORIGINAL_FULL_SELECTED_DIR:-data/processed/real_full_selected_mix}"
ORIGINAL_FULL_SELECTED_SPLITS="${ORIGINAL_FULL_SELECTED_SPLITS:-train val test}"
UNIPROT_GRAPH_JSONL="${UNIPROT_GRAPH_JSONL:-data/processed/uniprot_features_local_export/all.jsonl}"
AFFINITY_GRAPH_JSONL="${AFFINITY_GRAPH_JSONL:-data/processed/biomolecular_complex_affinity_local/all.jsonl}"
DEFAULT_UNIPROT_FEATURES_INPUT="$ROOT/data/local/uniprot_features.tsv"
DEFAULT_AFFINITY_INPUT="$ROOT/data/local/complex_affinity.tsv"
UNIPROT_FEATURES_INPUTS="${UNIPROT_FEATURES_INPUTS:-$DEFAULT_UNIPROT_FEATURES_INPUT}"
AFFINITY_INPUTS="${AFFINITY_INPUTS:-$DEFAULT_AFFINITY_INPUT}"
AFFINITY_KIND="${AFFINITY_KIND:-biomolecular_affinity}"
PREPARE_FULL_BIOMED_SOURCES="${PREPARE_FULL_BIOMED_SOURCES:-auto}"
FULL_BIOMED_SOURCE_SUMMARY="${FULL_BIOMED_SOURCE_SUMMARY:-data/local/biomed_training_sources.summary.json}"
PREPARE_UNIPROT="${PREPARE_UNIPROT:-auto}"
PREPARE_AFFINITY="${PREPARE_AFFINITY:-auto}"
CURATE_DATA="${CURATE_DATA:-auto}"
FAST_CURATE="${FAST_CURATE:-1}"
RESUME_CURATE="${RESUME_CURATE:-1}"
CURATE_INDEX_ONLY="${CURATE_INDEX_ONLY:-auto}"
CURATE_RESUME_STATE_EVERY="${CURATE_RESUME_STATE_EVERY:-10000}"
VAL_RATIO="${VAL_RATIO:-0.05}"
TEST_RATIO="${TEST_RATIO:-0.05}"
SPLIT_POLICY="${SPLIT_POLICY:-entity}"
LIMIT_UNIPROT="${LIMIT_UNIPROT:-}"
LIMIT_AFFINITY="${LIMIT_AFFINITY:-}"
EXTRA_INPUT_GRAPHS="${EXTRA_INPUT_GRAPHS:-}"
BUILD_BIO_PHASE_SUBSETS="${BUILD_BIO_PHASE_SUBSETS:-1}"
STATIC_STRUCTURE_TARGET_ROWS="${STATIC_STRUCTURE_TARGET_ROWS:-25000}"
STRUCTURE_DYNAMICS_TARGET_ROWS="${STRUCTURE_DYNAMICS_TARGET_ROWS:-2500}"
BIO_PHASE_SUBSET_SOURCE_INCLUDE="${BIO_PHASE_SUBSET_SOURCE_INCLUDE:-auto}"

OUTPUT_DIR="${OUTPUT_DIR:-outputs/biomed_annotations_affinity_250m}"
if [[ "$INCLUDE_ORIGINAL_FULL_SELECTED" == "1" && "$OUTPUT_DIR" == "outputs/biomed_annotations_affinity_250m" ]]; then
  OUTPUT_DIR="outputs/biomed_annotations_affinity_plus_original_250m"
fi
VOCAB_PATH="${VOCAB_PATH:-$OUTPUT_DIR/vocab.jsonl}"
REUSE_VOCAB="${REUSE_VOCAB:-auto}"
SKIP_POLICY_CHECK="${SKIP_POLICY_CHECK:-0}"
TRAIN_DEVICE="${TRAIN_DEVICE:-cuda}"

ENABLE_TROPICAL_ATTENTION="${ENABLE_TROPICAL_ATTENTION:-1}"
TROPICAL_ATTENTION_CONFIG="${TROPICAL_ATTENTION_CONFIG:-config/model/overrides/hybrid_flash_mhta_backend.yaml}"
ENABLE_UMA_COORDINATE_HEAD="${ENABLE_UMA_COORDINATE_HEAD:-1}"
UMA_COORDINATE_HEAD_CONFIG="${UMA_COORDINATE_HEAD_CONFIG:-config/train/overrides/uma_coordinate_head.yaml}"
ENABLE_UMA_INTERNAL_COORDINATES="${ENABLE_UMA_INTERNAL_COORDINATES:-1}"
UMA_INTERNAL_COORDINATES_CONFIG="${UMA_INTERNAL_COORDINATES_CONFIG:-config/train/overrides/uma_internal_coordinates.yaml}"
ENABLE_LONG_ALL_ATOM_CARTESIAN_HEAD="${ENABLE_LONG_ALL_ATOM_CARTESIAN_HEAD:-0}"
UMA_ALL_ATOM_CARTESIAN_HEAD_CONFIG="${UMA_ALL_ATOM_CARTESIAN_HEAD_CONFIG:-config/train/overrides/uma_all_atom_cartesian_head_8192.yaml}"
ENABLE_UMA_CONTACT_GEOMETRY="${ENABLE_UMA_CONTACT_GEOMETRY:-0}"
UMA_CONTACT_GEOMETRY_CONFIG="${UMA_CONTACT_GEOMETRY_CONFIG:-config/train/overrides/uma_contact_geometry_loss.yaml}"
EXTRA_TRAIN_CONFIGS="${EXTRA_TRAIN_CONFIGS:-}"

case "$(printf '%s' "$CURATE_INDEX_ONLY" | tr '[:upper:]' '[:lower:]')" in
  auto)
    if [[ "$ENABLE_LONG_ALL_ATOM_CARTESIAN_HEAD" == "1" || "$DATA_DIR" == *all_atom_contact* ]]; then
      CURATE_INDEX_ONLY=1
    else
      CURATE_INDEX_ONLY=0
    fi
    ;;
  1|true|yes|on) CURATE_INDEX_ONLY=1 ;;
  *) CURATE_INDEX_ONLY=0 ;;
esac
if [[ "$BIO_PHASE_SUBSET_SOURCE_INCLUDE" == "auto" ]]; then
  if [[ "$CURATE_INDEX_ONLY" == "1" && ( "$ENABLE_LONG_ALL_ATOM_CARTESIAN_HEAD" == "1" || "$DATA_DIR" == *all_atom_contact* ) ]]; then
    BIO_PHASE_SUBSET_SOURCE_INCLUDE="all_atom_contact"
  else
    BIO_PHASE_SUBSET_SOURCE_INCLUDE=""
  fi
fi

TRAIN_PHASES="${TRAIN_PHASES:-sft}"
GFLOWNET_SFT_CONFIG="${GFLOWNET_SFT_CONFIG:-config/train/biomed_annotations_affinity_gflownet_sft_4090.yaml}"
GFLOWNET_SFT_VALIDATION_CONFIG="${GFLOWNET_SFT_VALIDATION_CONFIG:-config/validate/biomed_annotations_affinity_gflownet_sft_validation.yaml}"
STRUCTURE_DYNAMICS_GFLOWNET_CONFIG="${STRUCTURE_DYNAMICS_GFLOWNET_CONFIG:-config/train/biomed_annotations_affinity_structure_dynamics_gflownet_4090.yaml}"
STRUCTURE_DYNAMICS_GFLOWNET_VALIDATION_CONFIG="${STRUCTURE_DYNAMICS_GFLOWNET_VALIDATION_CONFIG:-config/validate/biomed_annotations_affinity_structure_dynamics_gflownet_validation.yaml}"
GFLOWNET_SFT_OUTPUT_DIR="${GFLOWNET_SFT_OUTPUT_DIR:-outputs/biomed_annotations_affinity_gflownet_sft_4090}"
STRUCTURE_DYNAMICS_GFLOWNET_OUTPUT_DIR="${STRUCTURE_DYNAMICS_GFLOWNET_OUTPUT_DIR:-outputs/biomed_annotations_affinity_structure_dynamics_gflownet_4090}"
GFLOWNET_SFT_STAGE_NAME="${GFLOWNET_SFT_STAGE_NAME:-gflownet_sft}"
STRUCTURE_DYNAMICS_GFLOWNET_STAGE_NAME="${STRUCTURE_DYNAMICS_GFLOWNET_STAGE_NAME:-structure_dynamics_gflownet}"
STATIC_STRUCTURE_DATA_DIR="${STATIC_STRUCTURE_DATA_DIR:-${DATA_DIR}_static_structure}"
STRUCTURE_DYNAMICS_GFLOWNET_DATA_DIR="${STRUCTURE_DYNAMICS_GFLOWNET_DATA_DIR:-${DATA_DIR}_structure_dynamics_2500}"
GFLOWNET_SFT_DATA_DIR="${GFLOWNET_SFT_DATA_DIR:-$DATA_DIR}"
if [[ "$INCLUDE_ORIGINAL_FULL_SELECTED" == "1" ]]; then
  if [[ "$GFLOWNET_SFT_OUTPUT_DIR" == "outputs/biomed_annotations_affinity_gflownet_sft_4090" ]]; then
    GFLOWNET_SFT_OUTPUT_DIR="outputs/biomed_annotations_affinity_plus_original_gflownet_sft_4090"
  fi
  if [[ "$STRUCTURE_DYNAMICS_GFLOWNET_OUTPUT_DIR" == "outputs/biomed_annotations_affinity_structure_dynamics_gflownet_4090" ]]; then
    STRUCTURE_DYNAMICS_GFLOWNET_OUTPUT_DIR="outputs/biomed_annotations_affinity_plus_original_structure_dynamics_gflownet_4090"
  fi
  if [[ "$GFLOWNET_SFT_STAGE_NAME" == "gflownet_sft" ]]; then
    GFLOWNET_SFT_STAGE_NAME="gflownet_sft_plus_original"
  fi
  if [[ "$STRUCTURE_DYNAMICS_GFLOWNET_STAGE_NAME" == "structure_dynamics_gflownet" ]]; then
    STRUCTURE_DYNAMICS_GFLOWNET_STAGE_NAME="structure_dynamics_gflownet_plus_original"
  fi
fi
VALIDATION_DEVICE="${VALIDATION_DEVICE:-cuda}"
VALIDATE_GFLOWNET="${VALIDATE_GFLOWNET:-0}"

TRAIN_MAX_STEPS="${TRAIN_MAX_STEPS:-full_epoch}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-1.0}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
TRAIN_EVAL_BATCH_SIZE="${TRAIN_EVAL_BATCH_SIZE:-$TRAIN_BATCH_SIZE}"
TRAIN_GRAD_ACCUM="${TRAIN_GRAD_ACCUM:-18}"
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-8}"
TRAIN_EVAL_NUM_WORKERS="${TRAIN_EVAL_NUM_WORKERS:-2}"
TRAIN_PREFETCH_FACTOR="${TRAIN_PREFETCH_FACTOR:-4}"
TRAIN_EVAL_EVERY="${TRAIN_EVAL_EVERY:-1000}"
TRAIN_EVAL_MAX_BATCHES="${TRAIN_EVAL_MAX_BATCHES:-256}"
TRAIN_CHECKPOINT_EVERY="${TRAIN_CHECKPOINT_EVERY:-1000}"
TRAIN_STAGE_NAME="${TRAIN_STAGE_NAME:-}"
if [[ -z "$TRAIN_STAGE_NAME" ]]; then
  if [[ "$INCLUDE_ORIGINAL_FULL_SELECTED" == "1" ]]; then
    TRAIN_STAGE_NAME="biomed_annotations_affinity_plus_original_sft"
  else
    TRAIN_STAGE_NAME="biomed_annotations_affinity_sft"
  fi
fi

export RUN_ID
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TQDM_DYNAMIC_NCOLS="${TQDM_DYNAMIC_NCOLS:-1}"
export TQDM_MININTERVAL="${TQDM_MININTERVAL:-0.5}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_PROJECT="${WANDB_PROJECT:-iska-ugm}"
export WANDB_GROUP="${WANDB_GROUP:-biomed-annotations-affinity-direct}"
export WANDB_TAGS="${WANDB_TAGS:-direct,biomed,uniprot,complex-affinity,250m,hybrid-flash-mhta,uma-coordinate-head,uma-internal-coordinates}"
if [[ "$INCLUDE_ORIGINAL_FULL_SELECTED" == "1" && "$WANDB_TAGS" != *"original-full-selected"* ]]; then
  export WANDB_TAGS="$WANDB_TAGS,original-full-selected,combined-full"
fi

run() {
  printf '\n[%s] $' "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  printf ' %q' "$@"
  printf '\n'
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    return 0
  fi
  "$@"
}

run_cx() {
  run conda run --no-capture-output -n "$CONDA_ENV" "$@"
}

append_config() {
  local config="$1"
  if [[ ! -f "$config" ]]; then
    printf 'Required config not found: %s\n' "$config" >&2
    exit 1
  fi
}

prepare_science() {
  local kind="$1"
  local dataset_name="$2"
  local output="$3"
  local limit="$4"
  shift 4
  local args=(python scripts/prepare_science_sources.py --kind "$kind" --dataset-name "$dataset_name" --output "$output")
  if [[ -n "$limit" ]]; then
    args+=(--limit "$limit")
  fi
  for input in "$@"; do
    args+=(--input "$input")
  done
  run_cx "${args[@]}"
}

should_prepare() {
  local mode="$1"
  local output="$2"
  shift 2
  case "$mode" in
    1|true|yes|force) return 0 ;;
    0|false|no) return 1 ;;
    auto)
      if [[ "$#" -eq 0 ]]; then
        return 1
      fi
      if [[ ! -s "$output" ]]; then
        return 0
      fi
      for input in "$@"; do
        if [[ -f "$input" && "$input" -nt "$output" ]]; then
          return 0
        fi
      done
      return 1
      ;;
    *)
      printf 'Invalid prepare mode: %s\n' "$mode" >&2
      exit 1
      ;;
  esac
}

should_prepare_full_sources() {
  local mode="$1"
  case "$mode" in
    1|true|yes|force) return 0 ;;
    0|false|no) return 1 ;;
    auto)
      [[ "$UNIPROT_FEATURES_INPUTS" == "$DEFAULT_UNIPROT_FEATURES_INPUT" || "$AFFINITY_INPUTS" == "$DEFAULT_AFFINITY_INPUT" ]]
      return $?
      ;;
    *)
      printf 'Invalid full-source prepare mode: %s\n' "$mode" >&2
      exit 1
      ;;
  esac
}

append_if_exists() {
  local path="$1"
  local array_name="$2"
  if [[ -s "$path" ]]; then
    eval "$array_name+=(\"\$path\")"
  fi
}

append_required_or_dry_run() {
  local path="$1"
  local array_name="$2"
  if [[ -s "$path" || "${DRY_RUN:-0}" == "1" ]]; then
    eval "$array_name+=(\"\$path\")"
    return 0
  fi
  printf 'Required graph JSONL is missing or empty: %s\n' "$path" >&2
  return 1
}

is_placeholder_path() {
  local path="$1"
  [[ "$path" == /path/to/* || "$path" == "/absolute/path/to/"* || "$path" == *"REPLACE_ME"* || "$path" == *"replace_me"* ]]
}

validate_source_inputs() {
  local label="$1"
  shift
  local missing=0
  for input in "$@"; do
    if is_placeholder_path "$input"; then
      printf '%s contains a placeholder path: %s\n' "$label" "$input" >&2
      printf 'Replace it with a real local TSV/CSV/JSON/JSONL/FASTA path, or unset %s to use an existing graph JSONL.\n' "$label" >&2
      missing=1
    elif [[ ! -f "$input" ]]; then
      printf '%s input file does not exist: %s\n' "$label" "$input" >&2
      missing=1
    fi
  done
  if [[ "$missing" != "0" ]]; then
    return 1
  fi
  return 0
}

for config in "$MODEL_CONFIG" "$DATA_CONFIG" "$TRAIN_CONFIG" "$WANDB_CONFIG"; do
  append_config "$config"
done
if [[ "$ENABLE_TROPICAL_ATTENTION" == "1" ]]; then
  append_config "$TROPICAL_ATTENTION_CONFIG"
fi
if [[ "$ENABLE_LONG_ALL_ATOM_CARTESIAN_HEAD" == "1" ]]; then
  append_config "$UMA_ALL_ATOM_CARTESIAN_HEAD_CONFIG"
  EXTRA_TRAIN_CONFIGS="${EXTRA_TRAIN_CONFIGS:+$EXTRA_TRAIN_CONFIGS }$UMA_ALL_ATOM_CARTESIAN_HEAD_CONFIG"
else
  if [[ "$ENABLE_UMA_COORDINATE_HEAD" == "1" ]]; then
    append_config "$UMA_COORDINATE_HEAD_CONFIG"
    EXTRA_TRAIN_CONFIGS="${EXTRA_TRAIN_CONFIGS:+$EXTRA_TRAIN_CONFIGS }$UMA_COORDINATE_HEAD_CONFIG"
  fi
  if [[ "$ENABLE_UMA_INTERNAL_COORDINATES" == "1" ]]; then
    append_config "$UMA_INTERNAL_COORDINATES_CONFIG"
    EXTRA_TRAIN_CONFIGS="${EXTRA_TRAIN_CONFIGS:+$EXTRA_TRAIN_CONFIGS }$UMA_INTERNAL_COORDINATES_CONFIG"
  fi
fi
if [[ "$ENABLE_UMA_CONTACT_GEOMETRY" == "1" ]]; then
  append_config "$UMA_CONTACT_GEOMETRY_CONFIG"
  EXTRA_TRAIN_CONFIGS="${EXTRA_TRAIN_CONFIGS:+$EXTRA_TRAIN_CONFIGS }$UMA_CONTACT_GEOMETRY_CONFIG"
fi
if [[ -n "$EXTRA_TRAIN_CONFIGS" ]]; then
  for config in $EXTRA_TRAIN_CONFIGS; do
    append_config "$config"
  done
fi

if should_prepare_full_sources "$PREPARE_FULL_BIOMED_SOURCES"; then
  source_args=(python scripts/prepare_biomed_training_sources.py --summary "$FULL_BIOMED_SOURCE_SUMMARY")
  if [[ "$UNIPROT_FEATURES_INPUTS" == "$DEFAULT_UNIPROT_FEATURES_INPUT" ]]; then
    source_args+=(--uniprot-output "$DEFAULT_UNIPROT_FEATURES_INPUT")
  else
    source_args+=(--skip-uniprot)
  fi
  if [[ "$AFFINITY_INPUTS" == "$DEFAULT_AFFINITY_INPUT" ]]; then
    source_args+=(--affinity-output "$DEFAULT_AFFINITY_INPUT")
  else
    source_args+=(--skip-affinity)
  fi
  if [[ "$PREPARE_FULL_BIOMED_SOURCES" == "force" || "$PREPARE_FULL_BIOMED_SOURCES" == "1" || "$PREPARE_FULL_BIOMED_SOURCES" == "true" || "$PREPARE_FULL_BIOMED_SOURCES" == "yes" ]]; then
    source_args+=(--force)
  fi
  run_cx "${source_args[@]}"
fi

read -r -a UNIPROT_INPUT_ARRAY <<< "$UNIPROT_FEATURES_INPUTS"
read -r -a AFFINITY_INPUT_ARRAY <<< "$AFFINITY_INPUTS"
validate_source_inputs UNIPROT_FEATURES_INPUTS "${UNIPROT_INPUT_ARRAY[@]}"
validate_source_inputs AFFINITY_INPUTS "${AFFINITY_INPUT_ARRAY[@]}"

if should_prepare "$PREPARE_UNIPROT" "$UNIPROT_GRAPH_JSONL" "${UNIPROT_INPUT_ARRAY[@]}"; then
  prepare_science uniprot_features uniprot_features_local_export "$UNIPROT_GRAPH_JSONL" "$LIMIT_UNIPROT" "${UNIPROT_INPUT_ARRAY[@]}"
fi
if should_prepare "$PREPARE_AFFINITY" "$AFFINITY_GRAPH_JSONL" "${AFFINITY_INPUT_ARRAY[@]}"; then
  prepare_science "$AFFINITY_KIND" biomolecular_complex_affinity_local "$AFFINITY_GRAPH_JSONL" "$LIMIT_AFFINITY" "${AFFINITY_INPUT_ARRAY[@]}"
fi

INPUT_GRAPHS=()
append_if_exists "$UNIPROT_GRAPH_JSONL" INPUT_GRAPHS
append_if_exists "$AFFINITY_GRAPH_JSONL" INPUT_GRAPHS
if [[ "$INCLUDE_ORIGINAL_FULL_SELECTED" == "1" ]]; then
  for split in $ORIGINAL_FULL_SELECTED_SPLITS; do
    append_required_or_dry_run "$ORIGINAL_FULL_SELECTED_DIR/$split.jsonl" INPUT_GRAPHS
  done
fi
read -r -a EXTRA_INPUT_ARRAY <<< "$EXTRA_INPUT_GRAPHS"
for graph in "${EXTRA_INPUT_ARRAY[@]}"; do
  [[ -n "$graph" ]] || continue
  append_required_or_dry_run "$graph" INPUT_GRAPHS
done
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  if [[ "${#UNIPROT_INPUT_ARRAY[@]}" -gt 0 && ! " ${INPUT_GRAPHS[*]} " =~ " $UNIPROT_GRAPH_JSONL " ]]; then
    INPUT_GRAPHS+=("$UNIPROT_GRAPH_JSONL")
  fi
  if [[ "${#AFFINITY_INPUT_ARRAY[@]}" -gt 0 && ! " ${INPUT_GRAPHS[*]} " =~ " $AFFINITY_GRAPH_JSONL " ]]; then
    INPUT_GRAPHS+=("$AFFINITY_GRAPH_JSONL")
  fi
fi

if [[ "${#INPUT_GRAPHS[@]}" -eq 0 ]]; then
  printf 'No prepared UniProt, affinity, or original full-selected graph JSONL found.\n' >&2
  printf 'Set UNIPROT_FEATURES_INPUTS and/or AFFINITY_INPUTS, or create:\n' >&2
  printf '  %s\n  %s\n' "$UNIPROT_GRAPH_JSONL" "$AFFINITY_GRAPH_JSONL" >&2
  if [[ "$INCLUDE_ORIGINAL_FULL_SELECTED" == "1" ]]; then
    printf 'Also expected original selected splits under: %s\n' "$ORIGINAL_FULL_SELECTED_DIR" >&2
  fi
  exit 1
fi

needs_curation=0
if [[ "$CURATE_DATA" == "1" || "$CURATE_DATA" == "true" || "$CURATE_DATA" == "force" ]]; then
  needs_curation=1
elif [[ "$CURATE_DATA" == "auto" ]]; then
  if [[ ! -s "$DATA_DIR/train.jsonl" ]]; then
    needs_curation=1
  else
    for graph in "${INPUT_GRAPHS[@]}"; do
      if [[ "$graph" -nt "$DATA_DIR/train.jsonl" ]]; then
        needs_curation=1
        break
      fi
    done
  fi
fi

if [[ "$needs_curation" == "1" ]]; then
  curate_args=(python scripts/curate_data.py --output-dir "$DATA_DIR" --val-ratio "$VAL_RATIO" --test-ratio "$TEST_RATIO" --split-policy "$SPLIT_POLICY")
  if [[ "$FAST_CURATE" == "1" || "$FAST_CURATE" == "true" || "$FAST_CURATE" == "yes" ]]; then
    curate_args+=(--dedup-key row_hash --quality-mode none --fast-copy)
    if [[ "$CURATE_INDEX_ONLY" == "1" ]]; then
      curate_args+=(--index-only)
    fi
    if [[ "$RESUME_CURATE" == "1" || "$RESUME_CURATE" == "true" || "$RESUME_CURATE" == "yes" ]]; then
      curate_args+=(--resume --resume-state-every "$CURATE_RESUME_STATE_EVERY")
    fi
  fi
  for graph in "${INPUT_GRAPHS[@]}"; do
    curate_args+=(--input "$graph")
  done
  run_cx "${curate_args[@]}"
fi

run_cx python scripts/check_dataset_integrity.py --data-dir "$DATA_DIR" --output "$DATA_DIR/integrity.json"

if [[ "$BUILD_BIO_PHASE_SUBSETS" == "1" || "$BUILD_BIO_PHASE_SUBSETS" == "true" || "$BUILD_BIO_PHASE_SUBSETS" == "yes" || "$BUILD_BIO_PHASE_SUBSETS" == "force" ]]; then
  subset_args=(
    python scripts/build_bio_phase_subsets.py
    --input-dir "$DATA_DIR"
    --static-output-dir "$STATIC_STRUCTURE_DATA_DIR"
    --dynamics-output-dir "$STRUCTURE_DYNAMICS_GFLOWNET_DATA_DIR"
    --static-target-rows "$STATIC_STRUCTURE_TARGET_ROWS"
    --dynamics-target-rows "$STRUCTURE_DYNAMICS_TARGET_ROWS"
    --val-ratio "$VAL_RATIO"
    --test-ratio "$TEST_RATIO"
    --summary "$DATA_DIR/bio_phase_subsets.json"
  )
  if [[ -n "$BIO_PHASE_SUBSET_SOURCE_INCLUDE" ]]; then
    for pattern in $BIO_PHASE_SUBSET_SOURCE_INCLUDE; do
      subset_args+=(--source-path-include "$pattern")
    done
  fi
  run_cx "${subset_args[@]}"
fi

if [[ "${DRY_RUN:-0}" != "1" && ! -s "$DATA_DIR/train.jsonl" ]]; then
  printf 'Training split is missing or empty: %s/train.jsonl\n' "$DATA_DIR" >&2
  exit 1
fi
VAL_PATH="$DATA_DIR/val.jsonl"
if [[ "${DRY_RUN:-0}" != "1" && ! -s "$VAL_PATH" ]]; then
  VAL_PATH=""
fi
GFN_VALIDATION_DATA="${VAL_PATH:-$DATA_DIR/train.jsonl}"
GFLOWNET_SFT_VAL_PATH="$GFLOWNET_SFT_DATA_DIR/val.jsonl"
if [[ "${DRY_RUN:-0}" != "1" && ! -s "$GFLOWNET_SFT_VAL_PATH" ]]; then
  GFLOWNET_SFT_VAL_PATH=""
fi
STRUCTURE_DYNAMICS_GFLOWNET_VAL_PATH="$STRUCTURE_DYNAMICS_GFLOWNET_DATA_DIR/val.jsonl"
if [[ "${DRY_RUN:-0}" != "1" && ! -s "$STRUCTURE_DYNAMICS_GFLOWNET_VAL_PATH" ]]; then
  STRUCTURE_DYNAMICS_GFLOWNET_VAL_PATH=""
fi

if [[ "$REUSE_VOCAB" == "auto" ]]; then
  if [[ -s "$VOCAB_PATH" ]]; then
    REUSE_VOCAB=true
  else
    REUSE_VOCAB=false
  fi
fi

OVERRIDE="$RUN_DIR/biomed_annotations_affinity_override.yaml"
GFN_DATA_OVERRIDE="$RUN_DIR/biomed_annotations_affinity_data_override.yaml"
GFN_SFT_DATA_OVERRIDE="$RUN_DIR/biomed_annotations_affinity_gflownet_sft_data_override.yaml"
STRUCTURE_DYNAMICS_GFN_DATA_OVERRIDE="$RUN_DIR/biomed_annotations_affinity_structure_dynamics_gflownet_data_override.yaml"
GFN_SFT_OVERRIDE="$RUN_DIR/biomed_annotations_affinity_gflownet_sft_override.yaml"
STRUCTURE_DYNAMICS_GFN_OVERRIDE="$RUN_DIR/biomed_annotations_affinity_structure_dynamics_gflownet_override.yaml"
cat > "$OVERRIDE" <<YAML
stage:
  name: $TRAIN_STAGE_NAME
run:
  output_dir: $OUTPUT_DIR
  device: $TRAIN_DEVICE
data:
  train_path: $DATA_DIR/train.jsonl
  val_path: $VAL_PATH
  vocab_path: $VOCAB_PATH
  reuse_vocab: $REUSE_VOCAB
  skip_policy_check: $SKIP_POLICY_CHECK
train:
  max_steps: $TRAIN_MAX_STEPS
YAML
case "$(printf '%s' "$TRAIN_MAX_STEPS" | tr '[:upper:]' '[:lower:]')" in
  auto|epoch|full_epoch|full-dataset|full_dataset)
    printf '  full_epochs: %s\n' "$TRAIN_EPOCHS" >> "$OVERRIDE"
    ;;
  *)
    printf '  full_epochs: null\n' >> "$OVERRIDE"
    ;;
esac
cat >> "$OVERRIDE" <<YAML
  eval_every: $TRAIN_EVAL_EVERY
  eval_max_batches: $TRAIN_EVAL_MAX_BATCHES
  checkpoint_every: $TRAIN_CHECKPOINT_EVERY
  num_workers: $TRAIN_NUM_WORKERS
  eval_num_workers: $TRAIN_EVAL_NUM_WORKERS
  pin_memory: true
  persistent_workers: true
  prefetch_factor: $TRAIN_PREFETCH_FACTOR
  batch_size: $TRAIN_BATCH_SIZE
  eval_batch_size: $TRAIN_EVAL_BATCH_SIZE
  gradient_accumulation_steps: $TRAIN_GRAD_ACCUM
YAML
cat > "$GFN_DATA_OVERRIDE" <<YAML
data:
  train_path: $DATA_DIR/train.jsonl
  val_path: $VAL_PATH
YAML
cat > "$GFN_SFT_DATA_OVERRIDE" <<YAML
data:
  train_path: $GFLOWNET_SFT_DATA_DIR/train.jsonl
  val_path: $GFLOWNET_SFT_VAL_PATH
YAML
cat > "$STRUCTURE_DYNAMICS_GFN_DATA_OVERRIDE" <<YAML
data:
  train_path: $STRUCTURE_DYNAMICS_GFLOWNET_DATA_DIR/train.jsonl
  val_path: $STRUCTURE_DYNAMICS_GFLOWNET_VAL_PATH
YAML
cat > "$GFN_SFT_OVERRIDE" <<YAML
stage:
  name: $GFLOWNET_SFT_STAGE_NAME
run:
  output_dir: $GFLOWNET_SFT_OUTPUT_DIR
YAML
cat > "$STRUCTURE_DYNAMICS_GFN_OVERRIDE" <<YAML
stage:
  name: $STRUCTURE_DYNAMICS_GFLOWNET_STAGE_NAME
run:
  output_dir: $STRUCTURE_DYNAMICS_GFLOWNET_OUTPUT_DIR
YAML

printf '[%s] Direct UniProt + affinity training\n' "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
printf 'Run directory: %s\n' "$RUN_DIR"
printf 'Data dir: %s\n' "$DATA_DIR"
printf 'Include original full selected corpus: %s (%s: %s)\n' "$INCLUDE_ORIGINAL_FULL_SELECTED" "$ORIGINAL_FULL_SELECTED_DIR" "$ORIGINAL_FULL_SELECTED_SPLITS"
printf 'Input graph files: %s\n' "${INPUT_GRAPHS[*]}"
printf 'Extra input graph files: %s\n' "${EXTRA_INPUT_GRAPHS:-<none>}"
printf 'Train phases: %s\n' "$TRAIN_PHASES"
printf 'Curation: fast=%s resume=%s index_only=%s output=%s\n' "$FAST_CURATE" "$RESUME_CURATE" "$CURATE_INDEX_ONLY" "$DATA_DIR"
printf 'Phase subsets: build=%s static=%s rows -> %s structure_dynamics=%s rows -> %s\n' \
  "$BUILD_BIO_PHASE_SUBSETS" "$STATIC_STRUCTURE_TARGET_ROWS" "$STATIC_STRUCTURE_DATA_DIR" "$STRUCTURE_DYNAMICS_TARGET_ROWS" "$STRUCTURE_DYNAMICS_GFLOWNET_DATA_DIR"
printf 'Phase subset source-path include: %s\n' "${BIO_PHASE_SUBSET_SOURCE_INCLUDE:-<none>}"
printf 'Model-stage coordinate head: uma=%s internal=%s long_all_atom_8192=%s configs=%s\n' \
  "$ENABLE_UMA_COORDINATE_HEAD" "$ENABLE_UMA_INTERNAL_COORDINATES" "$ENABLE_LONG_ALL_ATOM_CARTESIAN_HEAD" "${EXTRA_TRAIN_CONFIGS:-<none>}"
printf 'Note: coordinate/internal-coordinate heads are model-stage overrides; standalone GFlowNet phases train token-set policies over the generated structure-dynamics candidate vocabulary.\n'
printf 'Override:\n'
cat "$OVERRIDE"

run_train_stage() {
  local args=(python scripts/train_stage.py "$@")
  if [[ "$ENABLE_TROPICAL_ATTENTION" == "1" ]]; then
    args+=(--config "$TROPICAL_ATTENTION_CONFIG")
  fi
  if [[ -n "$EXTRA_TRAIN_CONFIGS" ]]; then
    for config in $EXTRA_TRAIN_CONFIGS; do
      args+=(--config "$config")
    done
  fi
  args+=(--config "$WANDB_CONFIG")
  run_cx "${args[@]}"
}

run_gflownet_stage() {
  local config="$1"
  local stage_override="$2"
  local data_override="$3"
  run_cx python scripts/train_stage.py --config "$DATA_CONFIG" --config "$data_override" --config "$config" --config "$stage_override" --config "$WANDB_CONFIG"
}

validate_gflownet_stage() {
  local config="$1"
  local validation_config="$2"
  local stage_override="$3"
  local data_override="$4"
  local validation_data="$5"
  if [[ "$VALIDATE_GFLOWNET" != "1" ]]; then
    return 0
  fi
  append_config "$validation_config"
  run_cx python scripts/validate_gflownet.py --config "$DATA_CONFIG" --config "$data_override" --config "$config" --config "$stage_override" --config "$validation_config" --data "$validation_data" --device "$VALIDATION_DEVICE"
}

IFS=',' read -r -a PHASE_ARRAY <<< "$TRAIN_PHASES"
for phase in "${PHASE_ARRAY[@]}"; do
  phase="$(printf '%s' "$phase" | tr '[:upper:]' '[:lower:]' | tr '-' '_')"
  case "$phase" in
    sft)
      run_train_stage --config "$MODEL_CONFIG" --config "$DATA_CONFIG" --config "$TRAIN_CONFIG" --config "$OVERRIDE"
      ;;
    gflownet_sft)
      append_config "$GFLOWNET_SFT_CONFIG"
      run_gflownet_stage "$GFLOWNET_SFT_CONFIG" "$GFN_SFT_OVERRIDE" "$GFN_SFT_DATA_OVERRIDE"
      validate_gflownet_stage "$GFLOWNET_SFT_CONFIG" "$GFLOWNET_SFT_VALIDATION_CONFIG" "$GFN_SFT_OVERRIDE" "$GFN_SFT_DATA_OVERRIDE" "${GFLOWNET_SFT_VAL_PATH:-$GFLOWNET_SFT_DATA_DIR/train.jsonl}"
      ;;
    structure_dynamics_gflownet)
      append_config "$STRUCTURE_DYNAMICS_GFLOWNET_CONFIG"
      run_gflownet_stage "$STRUCTURE_DYNAMICS_GFLOWNET_CONFIG" "$STRUCTURE_DYNAMICS_GFN_OVERRIDE" "$STRUCTURE_DYNAMICS_GFN_DATA_OVERRIDE"
      validate_gflownet_stage "$STRUCTURE_DYNAMICS_GFLOWNET_CONFIG" "$STRUCTURE_DYNAMICS_GFLOWNET_VALIDATION_CONFIG" "$STRUCTURE_DYNAMICS_GFN_OVERRIDE" "$STRUCTURE_DYNAMICS_GFN_DATA_OVERRIDE" "${STRUCTURE_DYNAMICS_GFLOWNET_VAL_PATH:-$STRUCTURE_DYNAMICS_GFLOWNET_DATA_DIR/train.jsonl}"
      ;;
    all)
      run_train_stage --config "$MODEL_CONFIG" --config "$DATA_CONFIG" --config "$TRAIN_CONFIG" --config "$OVERRIDE"
      append_config "$GFLOWNET_SFT_CONFIG"
      append_config "$STRUCTURE_DYNAMICS_GFLOWNET_CONFIG"
      run_gflownet_stage "$GFLOWNET_SFT_CONFIG" "$GFN_SFT_OVERRIDE" "$GFN_SFT_DATA_OVERRIDE"
      validate_gflownet_stage "$GFLOWNET_SFT_CONFIG" "$GFLOWNET_SFT_VALIDATION_CONFIG" "$GFN_SFT_OVERRIDE" "$GFN_SFT_DATA_OVERRIDE" "${GFLOWNET_SFT_VAL_PATH:-$GFLOWNET_SFT_DATA_DIR/train.jsonl}"
      run_gflownet_stage "$STRUCTURE_DYNAMICS_GFLOWNET_CONFIG" "$STRUCTURE_DYNAMICS_GFN_OVERRIDE" "$STRUCTURE_DYNAMICS_GFN_DATA_OVERRIDE"
      validate_gflownet_stage "$STRUCTURE_DYNAMICS_GFLOWNET_CONFIG" "$STRUCTURE_DYNAMICS_GFLOWNET_VALIDATION_CONFIG" "$STRUCTURE_DYNAMICS_GFN_OVERRIDE" "$STRUCTURE_DYNAMICS_GFN_DATA_OVERRIDE" "${STRUCTURE_DYNAMICS_GFLOWNET_VAL_PATH:-$STRUCTURE_DYNAMICS_GFLOWNET_DATA_DIR/train.jsonl}"
      ;;
    none)
      printf 'TRAIN_PHASES=none; dataset preparation and checks completed without training.\n'
      ;;
    *)
      printf 'Unknown TRAIN_PHASES entry: %s\n' "$phase" >&2
      exit 1
      ;;
  esac
done
