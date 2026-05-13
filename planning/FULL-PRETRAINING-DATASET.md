# Full Selected Public Pretraining Dataset

Status: historical baseline implemented; expanded selected-corpus plan added on 2026-04-29. A completed run must pass `scripts/check_dataset_integrity.py`; interrupted graphification leaves stale split counts and is not training-ready.

This file describes the selected graph pretraining corpus at:

- Raw parquet cache: `data/raw_hf_full/`
- Graph JSONL corpus: `data/processed/real_full_selected_mix/`
- Token count summary: `data/processed/real_full_selected_mix/token_counts.json`

## Completeness Boundary

The baseline completed corpus includes all 19 public Hugging Face selected splits that exposed downloadable parquet files through the Dataset Viewer API at the time of the baseline run. The expanded default corpus adds a ranked set of targeted quality sources under a `MAX_GRAPH_TOKENS=5000000000` guard. Final expanded counts are produced by the next complete run and must report `within_model_sequence_token_budget: true` in `data/processed/real_full_selected_mix/token_counts.json`.

The expanded default additions are:

| Rank | Dataset | Purpose | Control |
|---:|---|---|---|
| 1 | `openai_graphwalks_train` | Long-context graph-state reasoning and answer-node prediction. | Include fully. |
| 2 | `graphwiz_graphinstruct_rft_72k_train` | Breadth over graph QA/instruction templates. | Include fully. |
| 3 | `pubchem10m_selfies_train` | Molecule SELFIES sequence pretraining without structure-file inputs. | Manifest row cap: 8,000,000. |
| 4 | `uniprot_function_text_train` | Protein sequence-to-function description grounding. | Include fully. |
| 5 | `uniprot_uniref50_sequence_train` | Million-row protein sequence coverage from UniRef50 representative sequences. | Manifest row cap: 3,000,000 for the bio-scale run. |
| 6 | `rfam_sequence_train` | RNA family sequence grounding. | Manifest row cap: 3,000,000 for the bio-scale run. |
| 7 | `rnacentral_8192_sequence_train` | Broad RNA sequence diversity. | Manifest row cap: 3,000,000 for the bio-scale run. |
| 8 | `dna_coding_regions_train` | DNA coding-region, exon/intron, and translated-protein annotations. | Manifest row cap requests 3,000,000, but the current public split is source-limited below that. |
| 9 | `openmathreasoning_tir_train` | Tool-integrated math reasoning. | Manifest row cap: 1,300,000. |
| 10 | `openmathreasoning_genselect_train` | Verifier-style solution selection. | Manifest row cap: 300,000. |
| 11 | `dclm_baseline_1b_train` | Curated general-language quality/fluency. | Include fully unless the 5B guard fails. |

Dataset Viewer sample rows estimate the capped expanded default at roughly 4.9B untruncated graph-sequence tokens after adding the old 1.108B-token baseline. The hard `count_graph_tokens.py` guard remains authoritative.

For the separate bio-scale all-atom contact run, `scripts/run_bio_scale_all_atom_contact_training.sh` adds a 3M-row UniRef50 protein sequence slice plus public molecule/RNA/DNA sequence graphification before curation. The broad sequence sources are compact BioSELFIES graph records by default (`BIO_SCALE_COMPACT=1`, `BIO_SCALE_MAX_SEQUENCE_CHARS=8192`), while the static contact and structure-dynamics subsets keep the all-atom Cartesian/contact/bond/affinity records. The runner checks the resulting modality counts with `scripts/check_bio_scale_targets.py`; protein, molecule SELFIES, and RNA must meet the requested target, while DNA is allowed to be source-limited and is reported as such. A slower 3M-row UniProtKB REST feature stream remains available with `PREPARE_PROTEIN_SCALE_REST=1`, but the default protein scale source is UniRef50 parquet for throughput.

The corpus still does not include manifest-only sources that require credentials, unavailable parquet export, or user-provided local exports:

- `the_stack_v2`: Hugging Face Dataset Viewer returned HTTP 401 unauthorized.
- `zinc20`: split names are visible, but Dataset Viewer exposes no parquet files or row-count metadata.
- `chembl_local_export`, `bindingdb_local_export`, `naturelm_pubchem_local`, `naturelm_uniprot_local`, `naturelm_refseq_local`, `naturelm_materials_project_local`, `pdbbind_docking_local`, `ec_protein_generation_local`, `ugm_multimodal_local`, and `hebrew_verb_complements_lexicon`: local-file schemas are implemented, but complete data requires reviewed user-provided exports.

Reference repositories and vocabulary sources are integrated separately:

- SFM/NatureLM reference repo: `data/external_repos/sfm`
- UniGenX reference repo: `data/external_repos/unigenx`
- Combined reference tokens: `data/processed/reference_tokens/naturelm_unigenx_tokens.txt`
- Motif graph tokens: `data/processed/reference_tokens/motif_graph_tokens.txt`
- Motif summary: `data/processed/reference_tokens/motif_graph_tokens.summary.json`
- Multimodal graph tokens: `data/processed/reference_tokens/multimodal_graph_tokens.txt`

The current full public motif vocabulary build is complete for the implemented public metadata sources:

| Source/family | Records |
|---|---:|
| Core defaults | 41 |
| InterPro | 51,489 |
| PROSITE | 2,730 |
| Rfam | 4,227 |
| CATH | 16,302 |
| Total motif records | 74,789 |
| Total motif tokens | 148,669 |

Those records break down into 55,644 sequence records, 10,987 structure records, and 8,158 structure-derived sequence records. Structure-derived sequence motifs from actual atom/frame rows are also generated during multimodal graphification.

## Baseline Corpus Totals

| Metric | Count |
|---|---:|
| Examples | 7,328,008 |
| Source graph tokens | 1,002,568,675 |
| Target graph tokens | 48,905,907 |
| Supervised prediction tokens | 48,905,907 |
| Untruncated model-sequence tokens | 1,107,708,497 |

The untruncated model-sequence token count follows the project training format:

```text
source graph tokens + <SEP> + (<POS>, target token) for each target token
```

Training configs still cap each example at `max_source_tokens` and `max_target_tokens`, so the effective tokens consumed by a particular run depend on the active data config.

## Baseline Split Totals

| Split | Examples | Source graph tokens | Target graph tokens | Untruncated model-sequence tokens |
|---|---:|---:|---:|---:|
| Train | 7,181,690 | 982,364,563 | 47,927,976 | 1,085,402,205 |
| Validation | 73,044 | 10,082,466 | 487,280 | 11,130,070 |
| Test | 73,274 | 10,121,646 | 490,651 | 11,176,222 |

## Baseline Dataset Totals

| Dataset | Examples | Source graph tokens | Target graph tokens | Untruncated model-sequence tokens |
|---|---:|---:|---:|---:|
| `gsm8k_main_train` | 7,473 | 44,838 | 14,946 | 82,203 |
| `gsm8k_main_test` | 1,319 | 7,914 | 2,638 | 14,509 |
| `openmathinstruct2_train_1m` | 1,000,000 | 5,998,534 | 2,000,000 | 10,998,534 |
| `numinamath_cot_train` | 859,494 | 3,437,976 | 1,718,988 | 7,735,446 |
| `numinamath_tir_train` | 72,441 | 289,764 | 144,882 | 651,969 |
| `bigcodebench_v014` | 1,140 | 9,120 | 3,420 | 17,100 |
| `lean_workbook_train` | 25,214 | 100,856 | 75,642 | 277,354 |
| `proofnetsharp_valid` | 185 | 1,110 | 555 | 2,405 |
| `moleculenet_lipophilicity` | 4,200 | 359,435 | 12,600 | 388,835 |
| `unigenx_qm9_train` | 133,885 | 16,456,970 | 401,655 | 17,394,165 |
| `unigenx_materials_crystal_system` | 2,692 | 26,920 | 8,076 | 45,764 |
| `hebrew_sefaria_train` | 1,955,969 | 835,060,636 | 24,261,934 | 885,540,473 |
| `hebrew_synthetic_medical_train` | 4,811 | 4,008,073 | 62,356 | 4,137,596 |
| `hebrew_wikianswers_lists` | 1,214,714 | 111,993,121 | 13,210,081 | 139,627,997 |
| `hebrew_wikianswers_queries` | 49,998 | 4,360,789 | 541,002 | 5,492,791 |
| `hebrew_alpaca_train` | 9,000 | 4,492,795 | 125,103 | 4,752,001 |
| `talmud_hebrew_train` | 37 | 32,417 | 481 | 33,416 |
| `hebrew_wikipedia_train` | 148,707 | 4,867,033 | 811,361 | 6,638,462 |
| `binding_affinity_public` | 1,836,729 | 11,020,374 | 5,510,187 | 23,877,477 |

## Commands

Build the selected graph corpus:

```bash
conda run -n tokengt python scripts/graphify_full_parquet_manifest.py \
  --manifest data/manifests/datasets.yaml \
  --raw-full-dir data/raw_hf_full \
  --output-dir data/processed/real_full_selected_mix \
  --val-ratio 0.01 \
  --test-ratio 0.01 \
  --batch-size 8192 \
  --progress-every 10000
```

Count graph tokens. This command uses split totals from `summary.json` when present, reports live source, target, sequence-token, dataset, and error counters, and enforces the 5B token cap:

```bash
conda run -n tokengt python scripts/count_graph_tokens.py \
  --data-dir data/processed/real_full_selected_mix \
  --output data/processed/real_full_selected_mix/token_counts.json \
  --progress-every 100000 \
  --max-model-sequence-tokens-total 5000000000
```

Generate the 2x largest-row context config:

```bash
conda run -n tokengt python scripts/inspect_context_requirements.py \
  --data-dir data/processed/real_full_selected_mix \
  --output data/processed/real_full_selected_mix/context_requirements.json \
  --context-multiplier 2.0 \
  --write-context-config config/generated/real_full_selected_context_2x.yaml
```

Verify split-file integrity before training:

```bash
conda run -n tokengt python scripts/check_dataset_integrity.py \
  --data-dir data/processed/real_full_selected_mix \
  --output data/processed/real_full_selected_mix/integrity.json
```

If this command reports a mismatch, rerun graphification to completion. The training sequence script runs the same check before full pretraining.

Train on the full selected corpus when the GPU is free:

```bash
conda run -n tokengt python scripts/train_stage.py \
  --config config/model/max_4090_tokengt.yaml \
  --config config/data/real_full_selected_mix.yaml \
  --config config/generated/real_full_selected_context_2x.yaml \
  --config config/train/real_full_selected_local.yaml
```

For the full phase 1 plus phase 2 curriculum with the correct defaults:

```bash
scripts/run_full_phase1_phase2_training.sh
```
