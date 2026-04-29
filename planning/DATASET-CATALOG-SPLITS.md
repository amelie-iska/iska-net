# Dataset Catalog and Split Plan

Created: 2026-04-29

This document lists the datasets and reference sources currently represented by the repository manifests, processed corpora, and vocabulary builders. It includes direct source links, short descriptions, known sizes, and split handling. The full selected public Phase 1 graph corpus is currently integrity-clean at `data/processed/real_full_selected_mix/`.

Executable implementation plan and live status:

- Implementation plan: `planning/DATASET-CATALOG-IMPLEMENTATION-PLAN.md`
- Generated status report: `planning/DATASET-CATALOG-STATUS.md`
- Machine-readable status: `data/manifests/dataset_catalog_status.json`
- Validator command: `conda run -n tokengt python scripts/validate_dataset_catalog.py --no-progress`

## Split Policies

| Corpus type | Split policy |
|---|---|
| Full selected public Hugging Face parquet corpus | Rows are graphified from downloadable parquet files. Split is deterministic by SHA1 bucket of the example id: test if bucket is below `test_ratio`, validation if below `test_ratio + val_ratio`, otherwise train. Current full run uses `val_ratio=0.01` and `test_ratio=0.01`. |
| 4090 bounded public corpus | Same deterministic SHA1 split policy as the full corpus, with a per-dataset row cap in `real_4090_mix`. Current processed split is train `226,293`, validation `2,293`, test `2,271`. |
| Curated local/smoke graph corpora | Rows are deduplicated and quality-filtered, then split by graph hash using the curation ratios in the active config. Small corpora may have zero test examples because of deterministic hashing and limited row count. |
| Manifest-only local sources | No automatic full split exists. User-provided exports must be prepared with the relevant preparation script, curated, counted, and integrity-checked before training. |
| Reference repositories and vocabularies | Not train/validation/test datasets by themselves. They provide reference tokens, schemas, model-method provenance, or motif vocabularies that are included in training configs. |

## Current Full Selected Public Corpus Totals

| Metric | Count |
|---|---:|
| Examples | 7,328,008 |
| Source graph tokens | 1,002,568,675 |
| Target graph tokens | 48,905,907 |
| Supervised prediction tokens | 48,905,907 |
| Untruncated model-sequence graph tokens | 1,107,708,497 |
| Train examples | 7,181,690 |
| Validation examples | 73,044 |
| Test examples | 73,274 |

The integrity file at `data/processed/real_full_selected_mix/integrity.json` currently reports `ok: true` with actual split counts matching the summary counts.

## Reference Repositories and Vocabulary Sources

| Source | Link | Description | Size in repo | Split handling |
|---|---|---|---:|---|
| SFM / NatureLM reference repo | <https://github.com/amelie-iska/SFM> | Reference implementation and README/checkpoint provenance for NatureLM-style science modeling across small molecules, materials, proteins, DNA, and RNA. | Reference repo checkout plus extracted tokens; not counted as graph examples. | No data split; contributes reference vocabulary and methodology. |
| NatureLM paper | <https://arxiv.org/abs/2502.07527> | Paper reference for the NatureLM scientific foundation model framing. | Documentation reference only. | No split. |
| UniGenX reference repo | <https://github.com/amelie-iska/UniGenX> | Reference implementation for unified sequence/structure generation with autoregressive and diffusion-style ideas. | Reference repo checkout plus extracted dictionary/tokenizer tokens. | No data split; contributes vocabulary and methodology. |
| UniGenX paper | <https://arxiv.org/abs/2503.06687> | Paper reference for UniGenX-style symbolic plus numeric generation. | Documentation reference only. | No split. |
| NatureLM + UniGenX reference token file | `data/processed/reference_tokens/naturelm_unigenx_tokens.txt` | Extracted reference tokens used to extend the graph vocabulary. | 1,164 tokens. | No split; vocabulary input. |
| PROSITE motif metadata | <https://ftp.expasy.org/databases/prosite/prosite.dat> | Protein sequence motifs and domains used for sequence-motif tokens. | 2,730 motif records in current build. | No split; vocabulary source. |
| InterPro motif metadata | <https://www.ebi.ac.uk/interpro/> | Protein family, domain, site, and functional annotation metadata. | 51,489 motif records in current build. | No split; vocabulary source. |
| CATH classification metadata | <https://www.cathdb.info/> | Structure classification names used for structure-motif tokens. | 16,302 motif records in current build. | No split; vocabulary source. |
| Rfam family metadata | <https://rfam.org/> | RNA family metadata used for RNA sequence and structure motif tokens. | 4,227 motif records in current build. | No split; vocabulary source. |
| Combined public motif vocabulary | `data/processed/reference_tokens/motif_graph_tokens.txt` | Sequence, structure, and structure-derived sequence motif tokens from core defaults, PROSITE, InterPro, CATH, and Rfam. | 74,789 motif records; 148,669 motif tokens. | No split; vocabulary input. |
| Multimodal graph token vocabulary | `data/processed/reference_tokens/multimodal_graph_tokens.txt` | UGM text, graph, reasoning, tool-use, molecular, bond, residue, nucleic-acid, geometry, oracle, and motif vocabulary. | 148,909 reference tokens. | No split; vocabulary input. |

## Full Selected Public Hugging Face Corpus

These are the 19 public Hugging Face selected splits that expose downloadable parquet files and are included in the completed full graph corpus.

| Dataset | Link | Description | Upstream split | Remote selected split size | Graph examples | Local split handling |
|---|---|---|---|---:|---:|---|
| `gsm8k_main_train` | <https://huggingface.co/datasets/openai/gsm8k> | Grade-school math word problems with natural-language solutions. | `main/train` | 2.2 MB | 7,473 | SHA1 98/1/1 train/val/test. |
| `gsm8k_main_test` | <https://huggingface.co/datasets/openai/gsm8k> | GSM8K held-out math test split included as validation-style reasoning data. | `main/test` | 409.3 KB | 1,319 | SHA1 98/1/1 train/val/test inside graph corpus. |
| `openmathinstruct2_train_1m` | <https://huggingface.co/datasets/nvidia/OpenMathInstruct-2> | One-million-row synthetic math instruction corpus. | `default/train_1M` | 609.4 MB | 1,000,000 | SHA1 98/1/1 train/val/test. |
| `numinamath_cot_train` | <https://huggingface.co/datasets/AI-MO/NuminaMath-CoT> | Chain-of-thought math reasoning examples. | `default/train` | 1.1 GB | 859,494 | SHA1 98/1/1 train/val/test. |
| `numinamath_tir_train` | <https://huggingface.co/datasets/AI-MO/NuminaMath-TIR> | Tool-integrated mathematical reasoning examples. | `default/train` | 140.5 MB | 72,441 | SHA1 98/1/1 train/val/test. |
| `bigcodebench_v014` | <https://huggingface.co/datasets/bigcode/bigcodebench> | Code-generation benchmark rows with diverse function-call and programming tasks. | `default/v0.1.4` | 2.3 MB | 1,140 | SHA1 98/1/1 train/val/test. |
| `lean_workbook_train` | <https://huggingface.co/datasets/internlm/Lean-Workbook> | Lean formalization and theorem-proving examples. | `default/train` | 4.4 MB | 25,214 | SHA1 98/1/1 train/val/test. |
| `proofnetsharp_valid` | <https://huggingface.co/datasets/PAug/ProofNetSharp> | Lean/ProofNet-style formal proof validation examples. | `default/valid` | 96.6 KB | 185 | SHA1 98/1/1 train/val/test. |
| `moleculenet_lipophilicity` | <https://huggingface.co/datasets/scikit-fingerprints/MoleculeNet_Lipophilicity> | MoleculeNet lipophilicity rows for molecule reasoning. | `default/train` | 123.3 KB | 4,200 | SHA1 98/1/1 train/val/test. |
| `unigenx_qm9_train` | <https://huggingface.co/datasets/yairschiff/qm9> | QM9-style source rows; first-run graphification keeps SMILES/SELFIES-style strings and allowed non-structure metadata while dropping atom coordinates, energies, and forces. | `default/train` | 172.0 MB | 133,885 | SHA1 98/1/1 train/val/test. |
| `unigenx_materials_crystal_system` | <https://huggingface.co/datasets/vinven7/materials-crystal-system-classification> | Materials Project style instruction rows mapping formulas to crystal-system labels. | `default/train` | 760.9 KB | 2,692 | SHA1 98/1/1 train/val/test. |
| `hebrew_sefaria_train` | <https://huggingface.co/datasets/sivan22/sefaria-hebrew> | Sefaria-derived Hebrew text rows, graphified with Hebrew text and heuristic morphology records. | `default/train` | 519.0 MB | 1,955,969 | SHA1 98/1/1 train/val/test. |
| `hebrew_synthetic_medical_train` | <https://huggingface.co/datasets/cp500/synthetic_hebrew_medical_text> | Synthetic Hebrew medical text rows. | `default/train` | 5.7 MB | 4,811 | SHA1 98/1/1 train/val/test. |
| `hebrew_wikianswers_lists` | <https://huggingface.co/datasets/imvladikon/wikianswers_hebrew> | Hebrew WikiAnswers grouped-list side for question similarity. | `lists/train` | 37.5 MB | 1,214,714 | SHA1 98/1/1 train/val/test. |
| `hebrew_wikianswers_queries` | <https://huggingface.co/datasets/imvladikon/wikianswers_hebrew> | Hebrew WikiAnswers query side for question similarity. | `queries/train` | 2.1 MB | 49,998 | SHA1 98/1/1 train/val/test. |
| `hebrew_alpaca_train` | <https://huggingface.co/datasets/ashercn97/hebrew_alpaca> | Hebrew instruction/input/output examples. | `default/train` | 2.2 MB | 9,000 | SHA1 98/1/1 train/val/test. |
| `talmud_hebrew_train` | <https://huggingface.co/datasets/guyhadad01/Talmud-Hebrew> | Talmud Hebrew text rows. | `default/train` | 6.7 MB | 37 | SHA1 98/1/1 train/val/test. |
| `hebrew_wikipedia_train` | <https://huggingface.co/datasets/YanFren/Hebrew_wikipedia> | Hebrew Wikipedia text rows. | `default/train` | 5.9 MB | 148,707 | SHA1 98/1/1 train/val/test. |
| `binding_affinity_public` | <https://huggingface.co/datasets/jglaser/binding_affinity> | Protein-sequence, ligand-SMILES, and binding-affinity rows from public bioactivity sources. | `default/train` | 134.0 MB | 1,836,729 | SHA1 98/1/1 train/val/test. |

## Full Corpus Token Counts by Dataset

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

## Manifest-Only, Restricted, Git, and Local Sources

These entries are part of the repo manifest and implementation plan, but they are not included in the completed public parquet corpus unless explicitly prepared or authorized.

| Dataset | Link | Description | Known size | Split handling |
|---|---|---|---:|---|
| `the_stack_v2` | <https://huggingface.co/datasets/bigcode/the-stack-v2> | Large code pretraining corpus. | Unknown locally; Dataset Viewer access returned unauthorized in the capacity audit. | Manifest-only; requires access review and explicit download. |
| `zinc20` | <https://huggingface.co/datasets/zpn/zinc20> | ZINC20 molecule corpus. | Unknown locally; no parquet metadata exposed in audit. | Manifest-only; requires separate acquisition path. |
| `hebrew_ud_htb` | <https://github.com/UniversalDependencies/UD_Hebrew-HTB> | Universal Dependencies Hebrew HTB CoNLL-U treebank for morphology and syntax. | Local raw footprint about 13.4 MB when cloned. | Git clone/reuse, then curation split by graph hash. |
| `hebrew_qa_nnlp` | <https://github.com/NNLP-IL/Hebrew-Question-Answering-Dataset> | Hebrew question answering dataset repository. | Local raw footprint about 20.9 MB when cloned. | Git clone/reuse, then curation split by graph hash. |
| `hebrew_nakdimon` | <https://github.com/elazarg/nakdimon> | Hebrew diacritization repository. | Local raw footprint about 244.8 MB when cloned with submodules. | Git clone/reuse, then curation split by graph hash. |
| `hebrew_verb_complements_lexicon` | Local/user-provided | Hebrew verb complement lexicon with root, binyan, lexical item, and complement fields. | Unknown until provided. | Local TSV/CSV prepared and curated by graph hash. |
| `chembl_local_export` | <https://www.ebi.ac.uk/chembl/> | ChEMBL assay, target, molecule, and bioactivity exports for science/medicine reasoning. | Unknown until user export is provided. | `prepare_science_sources.py --kind chembl`, then curation split. |
| `bindingdb_local_export` | <https://www.bindingdb.org/> | BindingDB target-ligand binding measurements. | Unknown until user export is provided. | `prepare_science_sources.py --kind bindingdb`, then curation split. |
| `naturelm_pubchem_local` | <https://pubchem.ncbi.nlm.nih.gov/> | PubChem-derived molecule rows for NatureLM-style science reconstruction. | Unknown until local export is provided. | `prepare_science_sources.py --kind pubchem`, then curation split. |
| `naturelm_uniprot_local` | <https://www.uniprot.org/> | UniProt protein sequence/function rows for NatureLM-style reconstruction. | Unknown until local export is provided. | `prepare_science_sources.py --kind uniprot`, then curation split. |
| `naturelm_refseq_local` | <https://www.ncbi.nlm.nih.gov/refseq/> | RefSeq/NCBI sequence rows for DNA/RNA/protein science reconstruction. | Unknown until local export is provided. | `prepare_science_sources.py --kind refseq`, then curation split. |
| `naturelm_materials_project_local` | <https://materialsproject.org/> | Materials Project rows for materials formula/property reasoning. | Unknown until local export is provided. | `prepare_science_sources.py --kind materials_project`, then curation split. |
| `pdbbind_docking_local` | <http://www.pdbbind.org.cn/> | Protein-ligand docking rows with ligand, pocket, and affinity records. | Unknown until local export is provided. | `prepare_science_sources.py --kind pdbbind`, then curation split. |
| `ec_protein_generation_local` | <https://enzyme.expasy.org/> | EC-number protein generation and enzyme annotation rows. | Unknown until local export is provided. | `prepare_science_sources.py --kind ec`, then curation split. |
| `ugm_multimodal_synthetic` | Project-generated | Synthetic phase-2 graph-to-graph examples for text, protein, SELFIES, DNA/RNA, atom/bond, temperature, coordinate, energy, and force records. | 32 generated rows in current processed corpus. | Generated, then curated split currently train 20, validation 8, test 4. |
| `ugm_multimodal_local` | Local/user-provided | Reviewed local exports from UniProt, InterPro, PDB, AlphaFoldDB, GEOM, SPICE, OMol25, PubChem, ChEMBL, RNAcentral, Rfam, and RNA3DHub-style sources. | Unknown until exports are provided. | `prepare_multimodal_sources.py`, then curation split. |

## Current Processed Derived Corpora

| Processed corpus | Path | Description | Kept examples | Train | Validation | Test | Notes |
|---|---|---|---:|---:|---:|---:|---|
| Curated graph smoke corpus | `data/processed/curated_graphs/` | Small mixed synthetic, math, code, proof, and molecule graph corpus. | 54 | 46 | 2 | 6 | Includes synthetic tool repair and path tasks. |
| Hebrew mix | `data/processed/hebrew_mix/` | Small Hebrew text, QA, morphosyntax, diacritization, instruction, and root-extension graph corpus. | 112 | 98 | 14 | 0 | Test is zero because of small deterministic split. |
| Science mix | `data/processed/science_mix/` | Small UniGenX molecule/material science-reasoning corpus. | 8 | 7 | 1 | 0 | Smoke data only. |
| Multimodal graph smoke corpus | `data/processed/multimodal_graphs/` | Synthetic graph-to-graph multimodal records for second-phase smoke testing. | 32 | 20 | 8 | 4 | Includes atom/bond/coordinate/temperature/oracle-style records. |
| 4090 bounded public graph corpus | `data/processed/real_4090_mix/` | Public manifest corpus capped at 20,000 rows per dataset for local GPU experiments. | 230,857 | 226,293 | 2,293 | 2,271 | SHA1 split with 1% validation and 1% test. |
| Full selected public graph corpus | `data/processed/real_full_selected_mix/` | Completed public non-gated Hugging Face selected parquet corpus. | 7,328,008 | 7,181,690 | 73,044 | 73,274 | Integrity check currently passes. |

## Full Corpus Token Counts by Split

| Split | Examples | Source graph tokens | Target graph tokens | Untruncated model-sequence tokens |
|---|---:|---:|---:|---:|
| Train | 7,181,690 | 982,364,563 | 47,927,976 | 1,085,402,205 |
| Validation | 73,044 | 10,082,466 | 487,280 | 11,130,070 |
| Test | 73,274 | 10,121,646 | 490,651 | 11,176,222 |

## Completeness Notes

- The completed full corpus includes the public, non-gated Hugging Face parquet splits listed above. It does not include manifest-only local exports or restricted datasets.
- SFM/NatureLM and UniGenX are included in two ways: reference vocabulary/methodology from their repositories, and selected UniGenX-style public datasets (`unigenx_qm9_train`, `unigenx_materials_crystal_system`) in the full public graph corpus.
- The public motif vocabulary is complete for the implemented public metadata sources: core defaults, PROSITE, InterPro, CATH, and Rfam.
- Structure-derived sequence motifs from actual atom/frame rows are generated during multimodal graphification when such rows are present; they are in addition to the original base text/graph/reasoning/molecular vocabulary.
- Any local or restricted dataset must be provenance-reviewed, prepared, curated, counted, and integrity-checked before being treated as training-ready.
