# Dataset Acquisition and Curation

## Policy

The paper names very large sources. This repo does not bulk-download them by default. Instead:

1. Store dataset provenance and license notes in `data/manifests/datasets.yaml`.
2. Pull small public samples into `data/raw/` with `scripts/acquire_datasets.py`.
3. Convert raw rows into graph JSONL with `scripts/graphify_data.py`.
4. Merge curated graph JSONL files with `scripts/merge_jsonl.py`.
5. Scale limits intentionally when storage, licensing, contamination checks, and training budget are ready.

## Current Local Samples

Small samples have been downloaded and graphified for:

- `openai/gsm8k` -> `data/raw/gsm8k_main_train/train.jsonl`
- `bigcode/bigcodebench` -> `data/raw/bigcodebench_v014/v0.1.4.jsonl`
- `PAug/ProofNetSharp` -> `data/raw/proofnetsharp_valid/valid.jsonl`
- `scikit-fingerprints/MoleculeNet_Lipophilicity` -> `data/raw/moleculenet_lipophilicity/train.jsonl`
- `yairschiff/qm9` -> `data/raw/unigenx_qm9_train/train.jsonl`
- `vinven7/materials-crystal-system-classification` -> `data/raw/unigenx_materials_crystal_system/train.jsonl`
- `sivan22/sefaria-hebrew` -> `data/raw/hebrew_sefaria_train/train.jsonl`
- `cp500/synthetic_hebrew_medical_text` -> `data/raw/hebrew_synthetic_medical_train/train.jsonl`
- `imvladikon/wikianswers_hebrew` -> `data/raw/hebrew_wikianswers_{lists,queries}/train.jsonl`
- `ashercn97/hebrew_alpaca` -> `data/raw/hebrew_alpaca_train/train.jsonl`
- `guyhadad01/Talmud-Hebrew` -> `data/raw/talmud_hebrew_train/train.jsonl`
- `YanFren/Hebrew_wikipedia` -> `data/raw/hebrew_wikipedia_train/train.jsonl`
- synthetic graph/tool traces -> `data/processed/synthetic_graphs/train.jsonl`
- UGM multimodal synthetic source rows -> `data/raw/ugm_multimodal_synthetic/train.jsonl`

Merged graphified sample:

- `data/processed/mixed_graphs/train.jsonl`
- curated deterministic splits under `data/processed/curated_graphs/`
- PLAN-D tiny science mix under `data/processed/science_mix/`
- PLAN-E Hebrew mix under `data/processed/hebrew_mix/`
- PLAN-H synthetic/local multimodal graph-to-graph rows under `data/processed/multimodal_graphs/`

Reference implementation clones:

- `data/external_repos/torchgfn`
- `data/external_repos/sfm`
- `data/external_repos/unigenx`
- `data/raw/hebrew_ud_htb/repo`
- `data/raw/hebrew_qa_nnlp/repo`
- `data/raw/hebrew_nakdimon/repo`

Model/repo metadata:

- SFM/NatureLM GitHub metadata under `data/external_models/sfm/`
- UniGenX GitHub metadata under `data/external_models/unigenx/`
- optional NatureLM Hugging Face checkpoint metadata under `data/external_models/naturelm_8x7b_hf_checkpoint/` when explicitly acquired

Full selected Hugging Face split snapshots:

- `data/raw_hf_full/` contains the full public non-manifest-only HF selected splits from `data/manifests/datasets.yaml`.
- Current footprint after the April 29, 2026 audit/download: 28 parquet files, about 2.8 GB.
- Manifest-only datasets such as `the_stack_v2` and `zinc20`, plus local-file reconstruction sources such as ChEMBL, BindingDB, PubChem, UniProt, RefSeq, Materials Project, PDBbind, and EC protein rows, are still provenance placeholders until reviewed local files or explicit gated access are provided.

Full selected public graph corpus target after integrity-checked graphification:

- `data/processed/real_full_selected_mix/train.jsonl`: 7,181,690 examples.
- `data/processed/real_full_selected_mix/val.jsonl`: 73,044 examples.
- `data/processed/real_full_selected_mix/test.jsonl`: 73,274 examples.
- The corpus is built from all public selected-split parquet rows currently available in `data/raw_hf_full/`, with no per-dataset cap.
- Run `scripts/check_dataset_integrity.py --data-dir data/processed/real_full_selected_mix` before training; interrupted graphification runs leave stale summary counts.
- The train configs include `data/processed/reference_tokens/naturelm_unigenx_tokens.txt`, so NatureLM/SFM and UniGenX reference tokens are part of the real-data vocabulary extension.

## Commands

Acquire a small sample:

```bash
conda run -n tokengt python scripts/acquire_datasets.py --dataset gsm8k_main_train --limit 64
```

Acquire every feasible manifest entry: bounded HF samples, GitHub repos, project-generated synthetic rows, and provenance records for skipped local/manifest-only entries:

```bash
conda run -n tokengt python scripts/acquire_datasets.py \
  --manifest data/manifests/datasets.yaml \
  --out-dir data/raw
```

Audit local capacity and current remote selected-split sizes:

```bash
conda run -n tokengt python scripts/audit_dataset_capacity.py
```

Download full public HF selected splits to parquet snapshots:

```bash
conda run -n tokengt python scripts/download_hf_selected_splits.py \
  --manifest data/manifests/datasets.yaml \
  --out-dir data/raw_hf_full \
  --max-total-gib 32
```

Graphify the downloaded parquet snapshots into the full selected public graph corpus:

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

Graphify raw data:

```bash
conda run -n tokengt python scripts/graphify_data.py \
  --input data/raw/gsm8k_main_train/train.jsonl \
  --output data/processed/gsm8k_main_train/train.jsonl \
  --dataset-name gsm8k_main_train
```

Generate synthetic graph data:

```bash
conda run -n tokengt python scripts/graphify_data.py \
  --synthetic --count 512 \
  --output data/processed/synthetic_graphs/train.jsonl
```

Merge graphified datasets:

```bash
conda run -n tokengt python scripts/merge_jsonl.py \
  --input data/processed/synthetic_graphs/train.jsonl \
  --input data/processed/gsm8k_main_train/train.jsonl \
  --output data/processed/mixed_graphs/train.jsonl
```

Curate graphified examples:

```bash
conda run -n tokengt python scripts/curate_data.py \
  --input data/processed/mixed_graphs/train.jsonl \
  --output-dir data/processed/curated_graphs \
  --val-ratio 0.1 \
  --test-ratio 0.1
```

The curation step validates schema, removes exact graph duplicates, computes a graph quality score, writes deterministic splits, and writes `summary.json`.

## PLAN-D/PLAN-G Science Sources

The authoritative NatureLM integration for this project is the SFM repository at `https://github.com/amelie-iska/SFM`, which contains `NatureLM/README.md`, checkpoint links, and domain-tagged examples for the sequence-based science model described in arXiv `2502.07527`. The full NatureLM training corpus is not exposed as one public dataset, so this repo uses SFM/NatureLM as a reference source for science-domain tags, checkpoint provenance, and local reconstruction schemas.

UniGenX is the model described by arXiv `2503.06687v2`: heterogeneous molecule/material/protein/docking data are sequentialized with domain special tokens, symbolic tokens, numeric fields, and a numeric generation objective. The authoritative implementation source for this project is `https://github.com/amelie-iska/UniGenX`; this repo extracts its dictionary and tokenizer tokens into the TokenGT vocabulary, while active UGM training uses graph-token autoregression for coordinate/property records rather than a separate numeric diffusion loss.

Clone reference repos and extract extra vocabulary:

```bash
conda run -n tokengt python scripts/acquire_model_files.py --repo-name sfm
conda run -n tokengt python scripts/acquire_model_files.py --repo-name unigenx
conda run -n tokengt python scripts/extract_reference_tokens.py \
  --sfm-dir data/external_repos/sfm \
  --unigenx-dir data/external_repos/unigenx \
  --output data/processed/reference_tokens/naturelm_unigenx_tokens.txt
```

Acquire the UniGenX-style smoke samples:

```bash
conda run -n tokengt python scripts/acquire_datasets.py --dataset unigenx_qm9_train --limit 4
conda run -n tokengt python scripts/acquire_datasets.py --dataset unigenx_materials_crystal_system --limit 4
```

Graphify them:

```bash
conda run -n tokengt python scripts/graphify_data.py \
  --input data/raw/unigenx_qm9_train/train.jsonl \
  --output data/processed/unigenx_qm9_train/train.jsonl \
  --dataset-name unigenx_qm9_train

conda run -n tokengt python scripts/graphify_data.py \
  --input data/raw/unigenx_materials_crystal_system/train.jsonl \
  --output data/processed/unigenx_materials_crystal_system/train.jsonl \
  --dataset-name unigenx_materials_crystal_system
```

Merge and curate:

```bash
conda run -n tokengt python scripts/merge_jsonl.py \
  --input data/processed/unigenx_qm9_train/train.jsonl \
  --input data/processed/unigenx_materials_crystal_system/train.jsonl \
  --output data/processed/science_mix/all.jsonl

conda run -n tokengt python scripts/curate_data.py \
  --input data/processed/science_mix/all.jsonl \
  --output-dir data/processed/science_mix \
  --val-ratio 0.2 \
  --test-ratio 0.0
```

Refresh SFM and UniGenX GitHub metadata:

```bash
conda run -n tokengt python scripts/acquire_model_files.py --repo-name sfm
conda run -n tokengt python scripts/acquire_model_files.py --repo-name unigenx
```

Use the optional NatureLM Hugging Face checkpoint entries only when you intentionally want checkpoint metadata or carefully bounded file downloads. Large checkpoint shards are intentionally skipped unless explicitly allowed.

## PLAN-F Deferred Science and Bioactivity Sources

PLAN-F adds manifest entries and local preparation paths for sources that are too large, gated, or provenance-sensitive for default bulk download:

- `binding_affinity_public`: small opt-in Hugging Face sample for protein/ligand affinity graphification.
- `chembl_local_export`: local ChEMBL CSV/TSV exports.
- `bindingdb_local_export`: local BindingDB CSV/TSV exports.
- `naturelm_pubchem_local`, `naturelm_uniprot_local`, `naturelm_refseq_local`, `naturelm_materials_project_local`: local reconstruction pieces for an SFM science mix.
- `pdbbind_docking_local`: local protein-ligand docking rows.
- `ec_protein_generation_local`: local EC-number protein generation rows.

Small public bioactivity sample:

```bash
conda run -n tokengt python scripts/acquire_datasets.py \
  --dataset binding_affinity_public --limit 16

conda run -n tokengt python scripts/graphify_data.py \
  --input data/raw/binding_affinity_public/train.jsonl \
  --output data/processed/binding_affinity_public/train.jsonl \
  --dataset-name binding_affinity_public
```

Local source preparation:

```bash
conda run -n tokengt python scripts/prepare_science_sources.py \
  --kind chembl \
  --input /path/to/chembl_export.tsv \
  --output data/processed/local_chembl/train.jsonl \
  --limit 1000

conda run -n tokengt python scripts/prepare_science_sources.py \
  --kind ec \
  --input /path/to/proteins.fasta \
  --output data/processed/local_ec/train.jsonl
```

`scripts/prepare_science_sources.py` accepts CSV, TSV, JSON, JSONL, and FASTA inputs. It normalizes common column names into graph rows for molecules, proteins, EC-number conditioning, materials, bioactivity assays, and docking coordinates.

## PLAN-H Multimodal Graph-to-Graph Sources

The first multimodal phase extends the science mix with graph records for natural-language prompts, proteins, SELFIES/SMILES, DNA/RNA, sequence motifs, safe structure-derived sequence-motif vocabulary tokens, function-description records, continuous temperature, and tool/oracle records. AF-style 64-bin attention/coupling/UMA-influence/motion/trajectory-proxy records are stage-gated to the UMA structure-dynamics-proxy curriculum stage and are not emitted for ordinary sequence/function rows. NatureLM/SFM and UniGenX reference tokens stay in the vocabulary. Generated coordinate/frame candidate records are allowed as model outputs, but actual structure-file records such as atoms, coordinates, frames, energy, force, PDB/mmCIF/SDF, and row-local structure-derived motifs remain excluded as training data until a later explicit phase.

Build the full public motif vocabulary and neutral multimodal reference vocabulary:

```bash
conda run -n tokengt python scripts/build_multimodal_vocab.py \
  --download-public-motifs \
  --output data/processed/reference_tokens/multimodal_graph_tokens.txt
```

This writes:

- `data/raw_motifs/public/prosite.dat`
- `data/raw_motifs/public/interpro_entries.json`
- `data/raw_motifs/public/cath-names.txt`
- `data/raw_motifs/public/rfam-family.txt.gz`
- `data/processed/reference_tokens/motif_graph_tokens.txt`
- `data/processed/reference_tokens/motif_graph_tokens.summary.json`

The current complete local public motif build contains 74,789 motif records and 148,669 motif tokens: 55,644 sequence records, 10,987 structure records, and 8,158 structure-derived sequence records. The merged multimodal reference vocabulary contains 148,909 tokens.

Create a synthetic smoke dataset:

```bash
conda run -n tokengt python scripts/prepare_multimodal_sources.py \
  --synthetic \
  --count 32 \
  --output data/processed/multimodal_graphs/all.jsonl

conda run -n tokengt python scripts/curate_data.py \
  --input data/processed/multimodal_graphs/all.jsonl \
  --output-dir data/processed/multimodal_graphs \
  --val-ratio 0.2 \
  --test-ratio 0.1
```

Prepare local JSON/JSONL/CSV/TSV/FASTA rows:

```bash
conda run -n tokengt python scripts/prepare_multimodal_sources.py \
  --input /path/to/mixed_rows.jsonl \
  --dataset-name local_multimodal_graph_to_graph \
  --output data/processed/local_multimodal/train.jsonl \
  --limit 1000
```

Recognized first-run row fields include `prompt`, `task`, `protein_sequence`, `selfies`, `smiles`, `dna_sequence`, `rna_sequence`, `temperature`, `oracle`, `sequence_motifs`, `sequence_motifs_from_structure`, `protein_motifs`, `prosite`, `interpro`, `rfam`, `motifs`, and `function_description`. Structure-derived sequence motif vocabulary tokens use `SEQ_MOTIF_FROM_STRUCTURE:*` and are legal only when they are frozen sequence annotations rather than row-local coordinate/contact labels. Scaled use should draw from reviewed sources such as UniProt/InterPro/PROSITE/CATH, PubChem/ChEMBL/ZINC, RNAcentral/Rfam, NatureLM/SFM-style sequence science rows, sanitized UniGenX sequence/function metadata, and ProTrek-style protein sequence/function pairs. Evaluation-only sources may include PDB/RCSB/NAKB/RNA 3D Hub, GEOM/SPICE/OMol25, ATLAS, and mdCATH.

## PLAN-E Hebrew Sources

The Hebrew slice combines broad text, instruction, QA, morphosyntax, diacritization, and root-extension data.

HF samples:

```bash
for ds in \
  hebrew_sefaria_train \
  hebrew_synthetic_medical_train \
  hebrew_wikianswers_lists \
  hebrew_wikianswers_queries \
  hebrew_alpaca_train \
  talmud_hebrew_train \
  hebrew_wikipedia_train
do
  conda run -n tokengt python scripts/acquire_datasets.py --dataset "$ds" --limit 8
done
```

GitHub resources:

```bash
conda run -n tokengt python scripts/acquire_datasets.py --dataset hebrew_ud_htb
conda run -n tokengt python scripts/acquire_datasets.py --dataset hebrew_qa_nnlp
conda run -n tokengt python scripts/acquire_datasets.py --dataset hebrew_nakdimon
```

The UD Hebrew HTB repository supplies CoNLL-U morphosyntax and Hebrew-specific features such as `HebBinyan`. The HeQ repository supplies extractive Hebrew QA. Nakdimon supplies diacritized Hebrew text through its submodule. The Verb Complements Lexicon is supported as a local TSV/CSV schema but is manifest-only until an official source file is available.

Prepare structured Hebrew sources:

```bash
conda run -n tokengt python scripts/prepare_hebrew_sources.py \
  --ud-limit 16 \
  --qa-limit 16 \
  --nakdimon-limit 8 \
  --root-count 32 \
  --verb-complements-limit 16
```

Graphify HF rows with `scripts/graphify_data.py`, using the dataset name as listed in the manifest. Example:

```bash
conda run -n tokengt python scripts/graphify_data.py \
  --input data/raw/hebrew_alpaca_train/train.jsonl \
  --output data/processed/hebrew_alpaca_train/train.jsonl \
  --dataset-name hebrew_alpaca_train
```

Merge and curate the Hebrew mix:

```bash
conda run -n tokengt python scripts/merge_jsonl.py \
  --input data/processed/hebrew_sefaria_train/train.jsonl \
  --input data/processed/hebrew_synthetic_medical_train/train.jsonl \
  --input data/processed/hebrew_wikianswers_lists/train.jsonl \
  --input data/processed/hebrew_wikianswers_queries/train.jsonl \
  --input data/processed/hebrew_alpaca_train/train.jsonl \
  --input data/processed/talmud_hebrew_train/train.jsonl \
  --input data/processed/hebrew_wikipedia_train/train.jsonl \
  --input data/processed/hebrew_ud_htb/train.jsonl \
  --input data/processed/hebrew_qa_nnlp/train.jsonl \
  --input data/processed/hebrew_nakdimon/train.jsonl \
  --input data/processed/hebrew_root_synthetic/train.jsonl \
  --output data/processed/hebrew_mix/all.jsonl

conda run -n tokengt python scripts/curate_data.py \
  --input data/processed/hebrew_mix/all.jsonl \
  --output-dir data/processed/hebrew_mix \
  --val-ratio 0.15 \
  --test-ratio 0.0
```

## Vertical-Slice Configs

- `config/data/code_graphs.yaml`: BigCodeBench-style graph data with canonical solution, tests, imports, and entry-point metadata.
- `config/data/lean_graphs.yaml`: Lean/proof graph data with statement, proof, import, and `lean_source` metadata.
- `config/data/chem_graphs.yaml`: molecule graph data with SMILES, target/property, and RDKit atom/bond metadata when available.
- `config/data/sfm_naturelm.yaml`: SFM/NatureLM reference-token science mix.
- `config/data/unigenx_qm9.yaml`: UniGenX-style molecule, coordinate, and property graph data from QM9 rows.
- `config/data/unigenx_materials.yaml`: UniGenX-style material formula and crystal-system instruction data.
- `config/data/science_mix.yaml`: merged PLAN-D science graph data.
- `config/data/hebrew_mix.yaml`: mixed Hebrew graph data.
- `config/data/hebrew_roots.yaml`: synthetic root-extension graph data for GFlowNet training.
- `config/data/curated_graphs.yaml`: mixed curated splits for aggregate validation.

Each can be paired with `config/model/tiny_lora_checkpointed.yaml` and its matching `config/train/*_sft_tiny.yaml` smoke config.

## Graph Schema

Every graphified row has:

- `id`
- `task`
- `nodes`: `id`, `type`, `value`, `features`
- `edges`: `src`, `dst`, `type`, `features`
- `target_tokens`
- `decoder_orders`
- `metadata`

## Scaling Notes

- `nvidia/OpenMathInstruct-2`, `AI-MO/NuminaMath-*`, `internlm/Lean-Workbook`, and `bigcode/bigcodebench` can be scaled by increasing `--limit`.
- `bigcode/the-stack-v2` is gated and huge; keep as manifest-only until access and storage are planned.
- MoleculeNet Lipophilicity has unknown license in HF metadata; use only for smoke tests until provenance is reviewed.
- ChEMBL and BindingDB now have local CSV/TSV preparation support through `scripts/prepare_science_sources.py`; official bulk download automation still requires source-specific access, storage, and license review.
