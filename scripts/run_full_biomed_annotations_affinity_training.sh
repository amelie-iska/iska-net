#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LOCK_DIR="${LOCK_DIR:-logs/biomed_direct_training/.full_biomed_training.lock}"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  printf 'Another full biomed annotations/affinity prep or training run appears to be active: %s\n' "$LOCK_DIR" >&2
  printf 'If this is stale, remove it after confirming no run is active.\n' >&2
  exit 1
fi
cleanup() {
  rmdir "$LOCK_DIR" 2>/dev/null || true
}
trap cleanup EXIT INT TERM HUP

export RUN_ID="${RUN_ID:-$(date -u +"%Y%m%dT%H%M%SZ")}"
export PREPARE_FULL_BIOMED_SOURCES="${PREPARE_FULL_BIOMED_SOURCES:-auto}"
export PREPARE_UNIPROT="${PREPARE_UNIPROT:-auto}"
export PREPARE_AFFINITY="${PREPARE_AFFINITY:-auto}"
export CURATE_DATA="${CURATE_DATA:-auto}"
export FAST_CURATE="${FAST_CURATE:-1}"
export RESUME_CURATE="${RESUME_CURATE:-1}"
export TRAIN_PHASES="${TRAIN_PHASES:-all}"
export INCLUDE_ORIGINAL_FULL_SELECTED="${INCLUDE_ORIGINAL_FULL_SELECTED:-0}"
export ORIGINAL_FULL_SELECTED_DIR="${ORIGINAL_FULL_SELECTED_DIR:-data/processed/real_full_selected_mix}"
export ORIGINAL_FULL_SELECTED_SPLITS="${ORIGINAL_FULL_SELECTED_SPLITS:-train val test}"

export UNIPROT_FEATURES_INPUTS="${UNIPROT_FEATURES_INPUTS:-$ROOT/data/local/uniprot_features.tsv}"
export AFFINITY_INPUTS="${AFFINITY_INPUTS:-$ROOT/data/local/complex_affinity.tsv}"

export ENABLE_TROPICAL_ATTENTION="${ENABLE_TROPICAL_ATTENTION:-1}"
export ENABLE_UMA_COORDINATE_HEAD="${ENABLE_UMA_COORDINATE_HEAD:-1}"
export ENABLE_UMA_INTERNAL_COORDINATES="${ENABLE_UMA_INTERNAL_COORDINATES:-1}"
export ENABLE_LONG_ALL_ATOM_CARTESIAN_HEAD="${ENABLE_LONG_ALL_ATOM_CARTESIAN_HEAD:-0}"
export ENABLE_UMA_CONTACT_GEOMETRY="${ENABLE_UMA_CONTACT_GEOMETRY:-0}"

export MODEL_CONFIG="${MODEL_CONFIG:-config/model/ugm_250m_tokengt.yaml}"
export DATA_CONFIG="${DATA_CONFIG:-config/data/biomed_annotations_affinity_250m.yaml}"
export TRAIN_CONFIG="${TRAIN_CONFIG:-config/train/biomed_annotations_affinity_250m.yaml}"
export GFLOWNET_SFT_CONFIG="${GFLOWNET_SFT_CONFIG:-config/train/biomed_annotations_affinity_gflownet_sft_4090.yaml}"
export STRUCTURE_DYNAMICS_GFLOWNET_CONFIG="${STRUCTURE_DYNAMICS_GFLOWNET_CONFIG:-config/train/biomed_annotations_affinity_structure_dynamics_gflownet_4090.yaml}"

printf '[%s] Full biomed annotations/affinity prep + training\n' "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
printf 'RUN_ID=%s\n' "$RUN_ID"
printf 'PREPARE_FULL_BIOMED_SOURCES=%s PREPARE_UNIPROT=%s PREPARE_AFFINITY=%s CURATE_DATA=%s FAST_CURATE=%s RESUME_CURATE=%s TRAIN_PHASES=%s\n' \
  "$PREPARE_FULL_BIOMED_SOURCES" "$PREPARE_UNIPROT" "$PREPARE_AFFINITY" "$CURATE_DATA" "$FAST_CURATE" "$RESUME_CURATE" "$TRAIN_PHASES"
printf 'INCLUDE_ORIGINAL_FULL_SELECTED=%s ORIGINAL_FULL_SELECTED_DIR=%s ORIGINAL_FULL_SELECTED_SPLITS=%s\n' \
  "$INCLUDE_ORIGINAL_FULL_SELECTED" "$ORIGINAL_FULL_SELECTED_DIR" "$ORIGINAL_FULL_SELECTED_SPLITS"
printf 'UNIPROT_FEATURES_INPUTS=%s\n' "$UNIPROT_FEATURES_INPUTS"
printf 'AFFINITY_INPUTS=%s\n' "$AFFINITY_INPUTS"

./scripts/train_biomed_annotations_affinity_direct.sh
