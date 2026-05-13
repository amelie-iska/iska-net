#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_ID="${RUN_ID:-$(date -u +"%Y%m%dT%H%M%SZ")-bio-scale-all-atom-contact}"
CONDA_ENV="${CONDA_ENV:-tokengt}"

BIO_SEQUENCE_TARGET_ROWS_PER_MODALITY="${BIO_SEQUENCE_TARGET_ROWS_PER_MODALITY:-3000000}"
PROTEIN_SEQUENCE_TARGET_ROWS="${PROTEIN_SEQUENCE_TARGET_ROWS:-3000000}"
STRUCTURE_DYNAMICS_TARGET_ROWS="${STRUCTURE_DYNAMICS_TARGET_ROWS:-2500}"
STATIC_STRUCTURE_TARGET_ROWS="${STATIC_STRUCTURE_TARGET_ROWS:-25000}"
BIO_SEQUENCE_RAW_DIR="${BIO_SEQUENCE_RAW_DIR:-data/raw_hf_bio_scale}"
BIO_SEQUENCE_PROCESSED_DIR="${BIO_SEQUENCE_PROCESSED_DIR:-data/processed/bio_sequence_scale_mix}"
PROTEIN_SCALE_TSV="${PROTEIN_SCALE_TSV:-data/local/uniprot_features_scale.tsv}"
PROTEIN_SCALE_GRAPH_JSONL="${PROTEIN_SCALE_GRAPH_JSONL:-data/processed/uniprot_features_scale/all.jsonl}"
PROTEIN_SCALE_SUMMARY="${PROTEIN_SCALE_SUMMARY:-data/local/biomed_training_sources.protein_scale.summary.json}"
BIO_SCALE_TARGET_SUMMARY="${BIO_SCALE_TARGET_SUMMARY:-data/processed/bio_sequence_scale_mix/bio_scale_target_summary.json}"
BIO_SEQUENCE_MAX_DOWNLOAD_GIB="${BIO_SEQUENCE_MAX_DOWNLOAD_GIB:-64}"
BIO_SCALE_ALLOW_SOURCE_LIMITED="${BIO_SCALE_ALLOW_SOURCE_LIMITED:-dna,protein_function_text}"
BIO_SCALE_COMPACT="${BIO_SCALE_COMPACT:-1}"
BIO_SCALE_MAX_SEQUENCE_CHARS="${BIO_SCALE_MAX_SEQUENCE_CHARS:-8192}"
PREPARE_PROTEIN_SCALE_REST="${PREPARE_PROTEIN_SCALE_REST:-0}"
UNIPROT_SCALE_QUERY="${UNIPROT_SCALE_QUERY:-*}"
UNIPROT_FEATURES_INPUTS="${UNIPROT_FEATURES_INPUTS:-data/local/uniprot_features.tsv}"
AFFINITY_INPUTS="${AFFINITY_INPUTS:-data/local/complex_affinity.tsv}"

BIO_SEQUENCE_DATASETS=(
  uniprot_uniref50_sequence_train
  pubchem10m_selfies_train
  uniprot_function_text_train
  rfam_sequence_train
  rnacentral_8192_sequence_train
  dna_coding_regions_train
)

run() {
  printf '\n[%s] $' "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  printf ' %q' "$@"
  printf '\n'
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    return 0
  fi
  "$@"
}

printf '[%s] Bio-scale all-atom-contact training\n' "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
printf 'RUN_ID=%s protein_target=%s per_modality_target=%s structure_dynamics_target=%s static_structure_target=%s\n' \
  "$RUN_ID" "$PROTEIN_SEQUENCE_TARGET_ROWS" "$BIO_SEQUENCE_TARGET_ROWS_PER_MODALITY" "$STRUCTURE_DYNAMICS_TARGET_ROWS" "$STATIC_STRUCTURE_TARGET_ROWS"

if [[ "$PREPARE_PROTEIN_SCALE_REST" == "1" || "$PREPARE_PROTEIN_SCALE_REST" == "true" || "$PREPARE_PROTEIN_SCALE_REST" == "yes" ]]; then
  run conda run --no-capture-output -n "$CONDA_ENV" python scripts/prepare_biomed_training_sources.py \
    --uniprot-output "$PROTEIN_SCALE_TSV" \
    --skip-affinity \
    --uniprot-query "$UNIPROT_SCALE_QUERY" \
    --limit-uniprot "$PROTEIN_SEQUENCE_TARGET_ROWS" \
    --min-uniprot-rows "$PROTEIN_SEQUENCE_TARGET_ROWS" \
    --summary "$PROTEIN_SCALE_SUMMARY"

  run conda run --no-capture-output -n "$CONDA_ENV" python scripts/prepare_science_sources.py \
    --kind uniprot_features \
    --dataset-name uniprot_features_scale \
    --output "$PROTEIN_SCALE_GRAPH_JSONL" \
    --input "$PROTEIN_SCALE_TSV"
fi

for dataset in "${BIO_SEQUENCE_DATASETS[@]}"; do
  run conda run --no-capture-output -n "$CONDA_ENV" python scripts/download_hf_selected_splits.py \
    --manifest data/manifests/datasets.yaml \
    --out-dir "$BIO_SEQUENCE_RAW_DIR" \
    --dataset "$dataset" \
    --max-total-gib "$BIO_SEQUENCE_MAX_DOWNLOAD_GIB"
done

graphify_args=(
  python scripts/graphify_full_parquet_manifest.py
  --manifest data/manifests/datasets.yaml
  --raw-full-dir "$BIO_SEQUENCE_RAW_DIR"
  --output-dir "$BIO_SEQUENCE_PROCESSED_DIR"
  --val-ratio 0.01
  --test-ratio 0.01
  --batch-size 8192
  --progress-every 10000
  --no-nested-progress
  --max-rows-per-dataset "$BIO_SEQUENCE_TARGET_ROWS_PER_MODALITY"
)
if [[ "$BIO_SCALE_COMPACT" == "1" || "$BIO_SCALE_COMPACT" == "true" || "$BIO_SCALE_COMPACT" == "yes" ]]; then
  graphify_args+=(--bio-scale-compact --bio-scale-max-sequence-chars "$BIO_SCALE_MAX_SEQUENCE_CHARS")
fi
for dataset in "${BIO_SEQUENCE_DATASETS[@]}"; do
  graphify_args+=(--dataset "$dataset")
done
run conda run --no-capture-output -n "$CONDA_ENV" "${graphify_args[@]}"

run conda run --no-capture-output -n "$CONDA_ENV" python scripts/check_dataset_integrity.py \
  --data-dir "$BIO_SEQUENCE_PROCESSED_DIR" \
  --output "$BIO_SEQUENCE_PROCESSED_DIR/integrity.json"

run conda run --no-capture-output -n "$CONDA_ENV" python scripts/check_bio_scale_targets.py \
  --protein-summary "$PROTEIN_SCALE_SUMMARY" \
  --bio-sequence-summary "$BIO_SEQUENCE_PROCESSED_DIR/summary.json" \
  --target-rows "$BIO_SEQUENCE_TARGET_ROWS_PER_MODALITY" \
  --allow-source-limited "$BIO_SCALE_ALLOW_SOURCE_LIMITED" \
  --output "$BIO_SCALE_TARGET_SUMMARY"

EXTRA_GRAPHS=()
if [[ -s "$PROTEIN_SCALE_GRAPH_JSONL" ]]; then
  EXTRA_GRAPHS+=("$PROTEIN_SCALE_GRAPH_JSONL")
fi
EXTRA_GRAPHS+=(
  "$BIO_SEQUENCE_PROCESSED_DIR/train.jsonl"
  "$BIO_SEQUENCE_PROCESSED_DIR/val.jsonl"
  "$BIO_SEQUENCE_PROCESSED_DIR/test.jsonl"
)

run env \
  EXTRA_INPUT_GRAPHS="${EXTRA_GRAPHS[*]}" \
  RUN_ID="$RUN_ID" \
  UNIPROT_FEATURES_INPUTS="$UNIPROT_FEATURES_INPUTS" \
  AFFINITY_INPUTS="$AFFINITY_INPUTS" \
  STRUCTURE_DYNAMICS_TARGET_ROWS="$STRUCTURE_DYNAMICS_TARGET_ROWS" \
  STATIC_STRUCTURE_TARGET_ROWS="$STATIC_STRUCTURE_TARGET_ROWS" \
  PREPARE_UNIPROT="${PREPARE_UNIPROT:-auto}" \
  PREPARE_AFFINITY="${PREPARE_AFFINITY:-auto}" \
  CURATE_DATA="${CURATE_DATA:-force}" \
  TRAIN_PHASES="${TRAIN_PHASES:-all}" \
  ./scripts/run_all_atom_contact_biomed_retrain.sh
