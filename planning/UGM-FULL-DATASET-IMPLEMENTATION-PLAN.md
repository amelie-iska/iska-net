# UGM Full-Dataset Implementation Plan

Date: 2026-04-29

This plan implements `planning/UGM-FULL-DATASET-DIFF-AUDIT.md`. The target state is a training-ready, validation-ready, test-ready, and inference-ready repo whose default full run uses the best-ranked additional datasets up to a 5B untruncated graph-token budget and includes OpenAI GraphWalks by default.

## Phase 1: Dataset Selection And Manifest

Status: completed.

Required changes:

1. Add ranked active manifest entries for:
   - OpenAI GraphWalks.
   - GraphWiz GraphInstruct-RFT-72K.
   - PubChem10M SELFIES, capped at 8,000,000 rows.
   - UniProt function text descriptions.
   - Rfam sequence, capped at 1,000,000 rows.
   - RNAcentral 8192 sequence, capped at 500,000 rows.
   - DNA coding regions, capped at 500,000 rows.
   - OpenMathReasoning TIR, capped at 1,300,000 rows.
   - OpenMathReasoning GenSelect, capped at 300,000 rows.
   - DCLM baseline 1B.
2. Add `full_training_quality_rank` for auditability.
3. Add manifest row caps to PubChem10M SELFIES, Rfam, RNAcentral, DNA coding regions, OpenMathReasoning TIR, and OpenMathReasoning GenSelect.
4. Keep structure-file datasets excluded from default phase-1 training.

Validation:

```bash
python - <<'PY'
from iska_reasoner.utils.config import load_yaml
m = load_yaml("data/manifests/datasets.yaml")
names = [d["name"] for d in m["datasets"]]
for name in [
    "openai_graphwalks_train",
    "graphwiz_graphinstruct_rft_72k_train",
    "pubchem10m_selfies_train",
    "uniprot_function_text_train",
    "rfam_sequence_train",
    "rnacentral_8192_sequence_train",
    "dna_coding_regions_train",
    "openmathreasoning_tir_train",
    "openmathreasoning_genselect_train",
    "dclm_baseline_1b_train",
]:
    assert name in names, name
PY
```

## Phase 2: Graphification

Status: completed.

Required changes:

1. Implement `graphify_graph_reasoning` for GraphWalks and GraphInstruct rows.
   - Parse explicit directed edges such as `A -> B`.
   - Parse tuple-style edges such as `(0, 11)`.
   - Emit graph-node, graph-edge, task, prompt, answer-node, and prompt-token records.
2. Implement `graphify_nucleotide_sequence` for RNA/DNA rows.
   - Infer RNA for Rfam/RNAcentral or U-without-T sequences.
   - Infer DNA for DNA coding-region data.
   - Emit base, family, clan/type, accession, exon/intron, and translated-protein records.
   - Do not emit atom, coordinate, energy, force, PDB, mmCIF, SDF, or trajectory records.
3. Expand protein/function graphification for UniProt function text rows.
4. Keep SELFIES-only PubChem rows on the molecule sequence path.

Validation:

```bash
pytest -q tests/test_domain_slices.py
```

## Phase 3: Full-Dataset Download And Graphify Controls

Status: completed.

Required changes:

1. `scripts/download_hf_selected_splits.py`
   - Skip rows with `full_training_enabled: false`.
   - Support optional `full_training_max_parquet_files`.
   - Support optional `full_training_max_parquet_bytes`.
2. `scripts/graphify_full_parquet_manifest.py`
   - Honor `full_training_max_rows` from the manifest.
   - Combine manifest caps with global `--max-rows-per-dataset` by taking the lower cap.
   - Keep `--row-budget` as a global stop condition.
   - Write `per_dataset_limits` to `summary.json`.

Validation:

```bash
pytest -q tests/test_full_dataset_progress.py
```

## Phase 4: 5B Token Guard And 2x Context

Status: completed.

Required changes:

1. Add `--max-model-sequence-tokens-total` to `scripts/count_graph_tokens.py`.
2. Write the token-count JSON before failing when the cap is exceeded.
3. Add `MAX_GRAPH_TOKENS=5000000000` to the full runner.
4. Run `scripts/inspect_context_requirements.py` after graphification with `--context-multiplier 2.0`.
5. Write `config/generated/real_full_selected_context_2x.yaml`.
6. Include the generated context config in the full pretraining stage.

Validation:

```bash
python -m py_compile scripts/count_graph_tokens.py scripts/run_full_training_sequence.sh
bash -n scripts/run_full_training_sequence.sh
pytest -q tests/test_full_dataset_progress.py
```

## Phase 5: Full Phase 1 And 2 Runner

Status: completed.

Required changes:

1. Add `scripts/run_full_phase1_phase2_training.sh`.
2. Set defaults:
   - `MAX_GRAPH_TOKENS=5000000000`.
   - `MAX_TOTAL_GIB=64`.
   - `ENABLE_STRUCTURE_TRAINING=0`.
3. Delegate to `scripts/run_full_training_sequence.sh`.
4. Preserve tqdm/progress/logging/W&B paths already used by the underlying runner.

Validation:

```bash
bash -n scripts/run_full_phase1_phase2_training.sh
DRY_RUN=1 scripts/run_full_phase1_phase2_training.sh
```

## Phase 6: Documentation And Paper

Status: completed for this pass.

Required changes:

1. Update `README.md`:
   - Add ranked default dataset selection.
   - Add 5B token guard.
   - Add OpenAI GraphWalks default.
   - Add full phase 1+2 runner.
2. Update `planning/FULL-PRETRAINING-DATASET.md`:
   - Preserve old completed corpus as historical.
   - Add expanded selected-corpus plan and commands.
3. Update paper:
   - Add the selected default sources and ranking rationale.
   - Keep no-structure-file-supervision wording.
   - Keep PDB rendering optional.

Validation:

```bash
rg -n "OpenAI GraphWalks|5B|full_phase1_phase2|PubChem10M|OpenMathReasoning|DCLM" README.md planning assets/human_learning_transformer_learning_review_dataset_expanded.tex
```

## Phase 7: End-To-End Verification

Status: completed for static and unit-level checks. Full data retrieval/training is intentionally launched by the run script because it can be long-running.

Commands:

```bash
python -m py_compile \
  src/iska_reasoner/data/graphify.py \
  scripts/download_hf_selected_splits.py \
  scripts/graphify_full_parquet_manifest.py \
  scripts/count_graph_tokens.py

bash -n scripts/run_full_training_sequence.sh
bash -n scripts/run_full_phase1_phase2_training.sh

pytest -q tests/test_domain_slices.py tests/test_full_dataset_progress.py
pytest -q
```

Full execution command:

```bash
scripts/run_full_phase1_phase2_training.sh
```

Resume controls:

```bash
START_AT=02 scripts/run_full_phase1_phase2_training.sh
STOP_AFTER=03 scripts/run_full_phase1_phase2_training.sh
DRY_RUN=1 scripts/run_full_phase1_phase2_training.sh
```

## Fallback If 5B Guard Fails

Reduce selected data from lowest marginal priority upward:

1. Reduce or disable `dclm_baseline_1b_train`.
2. Reduce `openmathreasoning_genselect_train`.
3. Reduce `openmathreasoning_tir_train`.
4. Reduce DNA/RNA row caps.
5. Reduce PubChem10M SELFIES.
6. Keep GraphWalks and GraphInstruct unless impossible; they are the highest-value graph-state reasoning data.

## Readiness Criteria

The repo is ready for a full run when:

- Dataset manifest parses.
- Download script dry-runs or downloads selected splits under disk budget.
- Graphify honors manifest caps.
- Token count JSON reports `within_model_sequence_token_budget: true`.
- Context audit writes `config/generated/real_full_selected_context_2x.yaml`.
- Integrity check passes.
- Phase 1 train config loads.
- Phase 2 multimodal/oracle configs load.
- Tests pass.
