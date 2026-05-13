#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_ID="${RUN_ID:-$(date -u +"%Y%m%dT%H%M%SZ")-all-atom-contact}"
CONDA_ENV="${CONDA_ENV:-tokengt}"

UNIPROT_GRAPH_JSONL="${UNIPROT_GRAPH_JSONL:-data/processed/uniprot_features_local_export_all_atom_contact/all.jsonl}"
AFFINITY_GRAPH_JSONL="${AFFINITY_GRAPH_JSONL:-data/processed/biomolecular_complex_affinity_all_atom_contact/all.jsonl}"
UNIPROT_FEATURES_INPUTS="${UNIPROT_FEATURES_INPUTS:-data/local/uniprot_features.tsv}"
AFFINITY_INPUTS="${AFFINITY_INPUTS:-data/local/complex_affinity.tsv}"
PREPARE_UNIPROT="${PREPARE_UNIPROT:-auto}"
PREPARE_AFFINITY="${PREPARE_AFFINITY:-auto}"

should_prepare() {
  local mode="$1"
  local output="$2"
  shift 2
  case "$mode" in
    1|true|yes|force) return 0 ;;
    0|false|no) return 1 ;;
    auto)
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

run() {
  printf '\n[%s] $' "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  printf ' %q' "$@"
  printf '\n'
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    return 0
  fi
  "$@"
}

mkdir -p "$(dirname "$UNIPROT_GRAPH_JSONL")" "$(dirname "$AFFINITY_GRAPH_JSONL")"

if should_prepare "$PREPARE_UNIPROT" "$UNIPROT_GRAPH_JSONL" "$UNIPROT_FEATURES_INPUTS"; then
  run conda run --no-capture-output -n "$CONDA_ENV" python scripts/prepare_science_sources.py \
    --kind uniprot_features \
    --dataset-name uniprot_features_local_export_all_atom_contact \
    --output "$UNIPROT_GRAPH_JSONL" \
    --input "$UNIPROT_FEATURES_INPUTS"
fi

if should_prepare "$PREPARE_AFFINITY" "$AFFINITY_GRAPH_JSONL" "$AFFINITY_INPUTS"; then
  run conda run --no-capture-output -n "$CONDA_ENV" python scripts/prepare_science_sources.py \
    --kind biomolecular_affinity \
    --dataset-name biomolecular_complex_affinity_all_atom_contact \
    --output "$AFFINITY_GRAPH_JSONL" \
    --input "$AFFINITY_INPUTS"
fi

export RUN_ID
export UNIPROT_GRAPH_JSONL
export AFFINITY_GRAPH_JSONL
export DATA_DIR="${DATA_DIR:-data/processed/biomed_annotations_affinity_plus_original_full_selected_all_atom_contact}"
export ENABLE_LONG_ALL_ATOM_CARTESIAN_HEAD="${ENABLE_LONG_ALL_ATOM_CARTESIAN_HEAD:-1}"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/biomed_annotations_affinity_plus_original_250m_all_atom_contact}"
export VOCAB_PATH="${VOCAB_PATH:-$OUTPUT_DIR/vocab.jsonl}"
export REUSE_VOCAB="${REUSE_VOCAB:-false}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
export TRAIN_EVAL_BATCH_SIZE="${TRAIN_EVAL_BATCH_SIZE:-$TRAIN_BATCH_SIZE}"
export TRAIN_GRAD_ACCUM="${TRAIN_GRAD_ACCUM:-36}"
export PREPARE_FULL_BIOMED_SOURCES=0
export PREPARE_UNIPROT=0
export PREPARE_AFFINITY=0
export CURATE_DATA="${CURATE_DATA:-force}"
export FAST_CURATE="${FAST_CURATE:-1}"
export RESUME_CURATE="${RESUME_CURATE:-1}"
export CURATE_INDEX_ONLY="${CURATE_INDEX_ONLY:-1}"
export INCLUDE_ORIGINAL_FULL_SELECTED="${INCLUDE_ORIGINAL_FULL_SELECTED:-1}"
export TRAIN_PHASES="${TRAIN_PHASES:-all}"

run ./scripts/train_biomed_annotations_affinity_direct.sh
