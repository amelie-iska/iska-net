# Dataset Catalog Implementation Status

Generated from `scripts/validate_dataset_catalog.py`.

## Readiness

- Ready: `true`
- Errors: 0
- Deferred local/restricted entries: 8

## Full Selected Public Corpus

- Path: `data/processed/real_full_selected_mix`
- Integrity OK: `true`
- Summary exists: `true`
- Token counts exist: `true`
- Examples: 7,328,008
- Train/validation/test: 7,181,690 / 73,044 / 73,274
- Source graph tokens: 1,002,568,675
- Target graph tokens: 48,905,907
- Untruncated model-sequence graph tokens: 1,107,708,497

## Reference Vocabularies

| Source | Exists | Size |
|---|---:|---:|
| NatureLM + UniGenX tokens | `true` | 1,164 tokens |
| Motif tokens | `true` | 148,669 tokens |
| Motif records | `true` | 74,789 records |
| Multimodal tokens | `true` | 148,909 tokens |

## Manifest Entry Status

| Dataset | Method | Status | Link | Full graph examples | Raw/parquet size | Split |
|---|---|---|---|---:|---:|---|
| `gsm8k_main_train` | hf_rows | `included_full_public_corpus` | <https://huggingface.co/datasets/openai/gsm8k> | 7,473 | 2.2 MB | main/train |
| `gsm8k_main_test` | hf_rows | `included_full_public_corpus` | <https://huggingface.co/datasets/openai/gsm8k> | 1,319 | 409.3 KB | main/test |
| `openmathinstruct2_train_1m` | hf_rows | `included_full_public_corpus` | <https://huggingface.co/datasets/nvidia/OpenMathInstruct-2> | 1,000,000 | 609.4 MB | default/train_1M |
| `numinamath_cot_train` | hf_rows | `included_full_public_corpus` | <https://huggingface.co/datasets/AI-MO/NuminaMath-CoT> | 859,494 | 1.1 GB | default/train |
| `numinamath_tir_train` | hf_rows | `included_full_public_corpus` | <https://huggingface.co/datasets/AI-MO/NuminaMath-TIR> | 72,441 | 140.5 MB | default/train |
| `bigcodebench_v014` | hf_rows | `included_full_public_corpus` | <https://huggingface.co/datasets/bigcode/bigcodebench> | 1,140 | 2.3 MB | default/v0.1.4 |
| `lean_workbook_train` | hf_rows | `included_full_public_corpus` | <https://huggingface.co/datasets/internlm/Lean-Workbook> | 25,214 | 4.4 MB | default/train |
| `proofnetsharp_valid` | hf_rows | `included_full_public_corpus` | <https://huggingface.co/datasets/PAug/ProofNetSharp> | 185 | 96.6 KB | default/valid |
| `moleculenet_lipophilicity` | hf_rows | `included_full_public_corpus` | <https://huggingface.co/datasets/scikit-fingerprints/MoleculeNet_Lipophilicity> | 4,200 | 123.3 KB | default/train |
| `the_stack_v2` | hf_rows | `deferred_manifest_only_or_restricted` | <https://huggingface.co/datasets/bigcode/the-stack-v2> | 0 | 266.0 B | default/train |
| `zinc20` | hf_rows | `deferred_manifest_only_or_restricted` | <https://huggingface.co/datasets/zpn/zinc20> | 0 | 242.0 B | default/train |
| `unigenx_qm9_train` | hf_rows | `included_full_public_corpus` | <https://huggingface.co/datasets/yairschiff/qm9> | 133,885 | 172.0 MB | default/train |
| `unigenx_materials_crystal_system` | hf_rows | `included_full_public_corpus` | <https://huggingface.co/datasets/vinven7/materials-crystal-system-classification> | 2,692 | 760.9 KB | default/train |
| `hebrew_sefaria_train` | hf_rows | `included_full_public_corpus` | <https://huggingface.co/datasets/sivan22/sefaria-hebrew> | 1,955,969 | 519.0 MB | default/train |
| `hebrew_synthetic_medical_train` | hf_rows | `included_full_public_corpus` | <https://huggingface.co/datasets/cp500/synthetic_hebrew_medical_text> | 4,811 | 5.7 MB | default/train |
| `hebrew_wikianswers_lists` | hf_rows | `included_full_public_corpus` | <https://huggingface.co/datasets/imvladikon/wikianswers_hebrew> | 1,214,714 | 37.5 MB | lists/train |
| `hebrew_wikianswers_queries` | hf_rows | `included_full_public_corpus` | <https://huggingface.co/datasets/imvladikon/wikianswers_hebrew> | 49,998 | 2.1 MB | queries/train |
| `hebrew_alpaca_train` | hf_rows | `included_full_public_corpus` | <https://huggingface.co/datasets/ashercn97/hebrew_alpaca> | 9,000 | 2.2 MB | default/train |
| `talmud_hebrew_train` | hf_rows | `included_full_public_corpus` | <https://huggingface.co/datasets/guyhadad01/Talmud-Hebrew> | 37 | 6.7 MB | default/train |
| `hebrew_wikipedia_train` | hf_rows | `included_full_public_corpus` | <https://huggingface.co/datasets/YanFren/Hebrew_wikipedia> | 148,707 | 5.9 MB | default/train |
| `hebrew_ud_htb` | git_clone | `git_source_available` | <https://github.com/UniversalDependencies/UD_Hebrew-HTB> | 0 | 13.4 MB | train |
| `hebrew_qa_nnlp` | git_clone | `git_source_available` | <https://github.com/NNLP-IL/Hebrew-Question-Answering-Dataset> | 0 | 20.9 MB | train |
| `hebrew_nakdimon` | git_clone | `git_source_available` | <https://github.com/elazarg/nakdimon> | 0 | 244.8 MB | train |
| `hebrew_verb_complements_lexicon` | local_file | `deferred_local_user_export_required` | local/user-provided | 0 | 283.0 B | train |
| `binding_affinity_public` | hf_rows | `included_full_public_corpus` | <https://huggingface.co/datasets/jglaser/binding_affinity> | 1,836,729 | 134.0 MB | default/train |
| `chembl_local_export` | local_file | `deferred_local_user_export_required` | <https://www.ebi.ac.uk/chembl/> | 0 | 271.0 B | train |
| `bindingdb_local_export` | local_file | `deferred_local_user_export_required` | <https://www.bindingdb.org/> | 0 | 274.0 B | train |
| `naturelm_pubchem_local` | local_file | `local_user_export_available` | <https://pubchem.ncbi.nlm.nih.gov/> | 0 | 1.4 GB | train |
| `ugm_multimodal_synthetic` | local_generated | `generated_source_available` | local/user-provided | 0 | 35.7 KB | train |
| `ugm_multimodal_local` | local_file | `deferred_local_user_export_required` | local/user-provided | 0 | 273.0 B | train |
| `naturelm_uniprot_local` | local_file | `local_user_export_available` | <https://www.uniprot.org/> | 0 | 97.3 MB | train |
| `naturelm_refseq_local` | local_file | `local_user_export_available` | <https://www.ncbi.nlm.nih.gov/refseq/> | 0 | 135.8 MB | train |
| `naturelm_materials_project_local` | local_file | `local_user_export_available` | <https://materialsproject.org/> | 0 | 1.3 KB | train |
| `pdbbind_docking_local` | local_file | `deferred_eval_or_future_phase` | <http://www.pdbbind.org.cn/> | 0 | 273.0 B | validation/test/future |
| `ec_protein_generation_local` | local_file | `deferred_local_user_export_required` | <https://enzyme.expasy.org/> | 0 | 279.0 B | train |

## Processed Corpora

| Corpus | Exists | Examples | Split sizes | Integrity |
|---|---:|---:|---|---:|
| `curated_graphs` | `true` | 54 | test=6, train=46, val=2 | `n/a` |
| `hebrew_mix` | `true` | 112 | test=0, train=98, val=14 | `n/a` |
| `science_mix` | `true` | 8 | test=0, train=7, val=1 | `n/a` |
| `multimodal_graphs` | `true` | 32 | test=4, train=23, val=5 | `n/a` |
| `real_4090_mix` | `true` | 230,857 | test=2,271, train=226,293, val=2,293 | `n/a` |
| `real_full_selected_mix` | `true` | 7,328,008 | test=73,274, train=7,181,690, val=73,044 | `true` |

## Deferred Entries

Deferred entries are expected blockers, not silent failures. They require credentials, upstream review, local user-provided exports, or a deliberate acquisition path before they can be counted as complete training data.

- `the_stack_v2`: `deferred_manifest_only_or_restricted`; link: https://huggingface.co/datasets/bigcode/the-stack-v2
- `zinc20`: `deferred_manifest_only_or_restricted`; link: https://huggingface.co/datasets/zpn/zinc20
- `hebrew_verb_complements_lexicon`: `deferred_local_user_export_required`; link: local/user-provided
- `chembl_local_export`: `deferred_local_user_export_required`; link: https://www.ebi.ac.uk/chembl/
- `bindingdb_local_export`: `deferred_local_user_export_required`; link: https://www.bindingdb.org/
- `ugm_multimodal_local`: `deferred_local_user_export_required`; link: local/user-provided
- `pdbbind_docking_local`: `deferred_local_user_export_required`; link: http://www.pdbbind.org.cn/
- `ec_protein_generation_local`: `deferred_local_user_export_required`; link: https://enzyme.expasy.org/
