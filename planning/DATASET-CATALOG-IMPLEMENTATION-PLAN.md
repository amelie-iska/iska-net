# Dataset Catalog Implementation Plan

Created: 2026-04-29

This plan operationalizes `planning/DATASET-CATALOG-SPLITS.md`. The goal is not only to document datasets, but to make the catalog enforceable: every dataset entry must be classified, every active public corpus must have integrity and token-count checks, every reference vocabulary must be verified, and every restricted/local source must be an explicit deferred blocker rather than an ambiguous missing dataset.

## Success Criteria

- [x] The full selected public Hugging Face parquet corpus is integrity-checked before training.
- [x] Full corpus token counts are present and tied to split counts.
- [x] SFM/NatureLM and UniGenX reference tokens are checked.
- [x] Public motif and multimodal vocabularies are checked.
- [x] Every manifest entry is classified as included, available, generated, deferred, or erroneous.
- [x] Manifest-only, restricted, and local-user-provided sources are explicit deferred entries.
- [x] A machine-readable catalog status report is generated.
- [x] A Markdown catalog status report is generated.
- [x] Unit tests cover catalog classification and report generation.
- [x] README and planning docs include the new validation command.

## Implementation Phases

### Phase 1: Catalog State Model

**Objective:** Convert the prose catalog into an executable inspection model.

**Implemented files:**

- `src/iska_reasoner/data/catalog.py`
- `scripts/validate_dataset_catalog.py`
- `tests/test_dataset_catalog.py`

**State model requirements:**

- [x] Load `data/manifests/datasets.yaml`.
- [x] Inspect public Hugging Face selected parquet snapshots under `data/raw_hf_full/`.
- [x] Inspect full graph corpus summary, integrity, and token-count files under `data/processed/real_full_selected_mix/`.
- [x] Inspect `data/manifests/dataset_capacity_audit.json` when present for remote selected-split sizes.
- [x] Inspect reference token files:
  - `data/processed/reference_tokens/naturelm_unigenx_tokens.txt`
  - `data/processed/reference_tokens/motif_graph_tokens.txt`
  - `data/processed/reference_tokens/motif_graph_tokens.summary.json`
  - `data/processed/reference_tokens/multimodal_graph_tokens.txt`
- [x] Inspect processed corpora:
  - `data/processed/curated_graphs/`
  - `data/processed/hebrew_mix/`
  - `data/processed/science_mix/`
  - `data/processed/multimodal_graphs/`
  - `data/processed/real_4090_mix/`
  - `data/processed/real_full_selected_mix/`

**Classification states:**

- [x] `included_full_public_corpus`: public HF split exists in the full graph corpus and full-corpus integrity is clean.
- [x] `raw_parquet_available_not_graphified`: public HF parquet is present but not included in an integrity-clean full graph corpus.
- [x] `missing_public_hf_parquet`: public HF entry is active but raw parquet is missing.
- [x] `deferred_manifest_only_or_restricted`: large/gated/manifest-only source is intentionally outside the public parquet corpus.
- [x] `git_source_available`: git source has been cloned under `data/raw/<name>/repo`.
- [x] `git_source_missing`: git source has not yet been cloned.
- [x] `generated_source_available`: project-generated source or processed synthetic data exists.
- [x] `generated_source_missing`: local-generated source has not been created.
- [x] `local_user_export_available`: local-file source has usable local files.
- [x] `deferred_local_user_export_required`: local-file source requires user-provided export.
- [x] `unsupported_manifest_method`: manifest method is unknown to the validator.

### Phase 2: Readiness Rules

**Objective:** Make readiness strict for active public data while allowing expected local/restricted deferrals.

**Implemented rules:**

- [x] Full public corpus must have `integrity_ok: true`.
- [x] Full public corpus must have `token_counts.json`.
- [x] NatureLM/UniGenX reference token file must exist and contain tokens.
- [x] Motif vocabulary must have at least 100,000 tokens and at least 70,000 motif records.
- [x] Multimodal vocabulary must have at least 100,000 tokens.
- [x] All active, non-manifest-only HF entries must be included in the full public corpus.
- [x] Deferred local/restricted entries are listed but do not make the public-corpus readiness check fail.

**Current status:**

- [x] `scripts/validate_dataset_catalog.py --no-progress` reports ready.
- [x] Current report has 35 manifest entries, 19 public full entries, 19 included public entries, 12 deferred entries, and 0 errors.

### Phase 3: Reporting

**Objective:** Produce both human-readable and machine-readable reports.

**Implemented outputs:**

- [x] `data/manifests/dataset_catalog_status.json`
- [x] `planning/DATASET-CATALOG-STATUS.md`

**Markdown report sections:**

- [x] Readiness summary.
- [x] Full selected public corpus status.
- [x] Reference vocabulary status.
- [x] Manifest entry status table.
- [x] Processed corpus table.
- [x] Deferred entries list with links.

**Machine-readable report sections:**

- [x] `ready`
- [x] `errors`
- [x] `summary`
- [x] `full_corpus`
- [x] `references`
- [x] `manifest_entries`
- [x] `deferred_entries`
- [x] `processed_corpora`

### Phase 4: Tests

**Objective:** Prevent regressions in catalog classification and report generation.

**Implemented tests:**

- [x] Byte formatting utility.
- [x] Source link construction for Hugging Face, git, and local named sources.
- [x] Ready public corpus plus deferred local export classification.
- [x] Markdown report generation.

**Validation commands:**

```bash
conda run -n tokengt pytest -q tests/test_dataset_catalog.py
conda run -n tokengt python scripts/validate_dataset_catalog.py --no-progress
```

### Phase 5: Integration With Readiness and Documentation

**Objective:** Make catalog validation discoverable and part of the normal run path.

**Implemented updates:**

- [x] `README.md` documents `scripts/validate_dataset_catalog.py`.
- [x] `planning/DATASET-CATALOG-SPLITS.md` links to the implementation status report.
- [x] `scripts/quality_assess.py` requires the catalog validator and reports.
- [x] `scripts/check_readiness.py` includes the catalog validator and status files in path checks.

## Operating Procedure

Run these commands after any dataset manifest, acquisition, graphification, vocabulary, or curation change:

```bash
conda run -n tokengt python scripts/check_dataset_integrity.py \
  --data-dir data/processed/real_full_selected_mix \
  --output data/processed/real_full_selected_mix/integrity.json

conda run -n tokengt python scripts/count_graph_tokens.py \
  --data-dir data/processed/real_full_selected_mix \
  --output data/processed/real_full_selected_mix/token_counts.json \
  --progress-every 100000

conda run -n tokengt python scripts/validate_dataset_catalog.py --no-progress

conda run -n tokengt python scripts/quality_assess.py
```

## Deferred Source Handling

The following entries are intentionally not marked complete unless actual source data is present:

- `the_stack_v2`
- `zinc20`
- `hebrew_verb_complements_lexicon`
- `chembl_local_export`
- `bindingdb_local_export`
- `naturelm_pubchem_local`
- `ugm_multimodal_local`
- `naturelm_uniprot_local`
- `naturelm_refseq_local`
- `naturelm_materials_project_local`
- `pdbbind_docking_local`
- `ec_protein_generation_local`

Each deferred source needs a separate provenance review, local export or authorized acquisition path, preparation command, curation split, token-count run, and catalog validation rerun.

## Final Implementation Status

- [x] Plan written.
- [x] Catalog status code implemented.
- [x] CLI implemented.
- [x] Tests added.
- [x] Reports generated.
- [x] Readiness/docs integration completed.
- [x] Full public selected corpus remains ready.
- [x] Deferred sources are explicit and not silently treated as complete.
