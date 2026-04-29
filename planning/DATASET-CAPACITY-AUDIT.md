# Dataset Capacity Audit

Created: `2026-04-29T09:04:11.427480Z`

## Local Capacity

- Disk free: 2.5 TB / 3.6 TB.
- Reserved free-space floor: 512.0 GB.
- RAM available: 56.5 GB / 60.2 GB.
- Swap free: 59.5 GB / 64.0 GB.
- GPU: NVIDIA GeForce RTX 4090 (24.0 GB total).

## Summary

- Manifest entries: 35.
- Current `data/raw` footprint covered by the manifest: 287.9 MB.
- Current full selected HF parquet footprint: 2.8 GB across 28 files.
- Known HF selected-split size total: 2.8 GB across 19 entries.
- Full-source downloads are not automatic; manifest-only and local-file entries remain provenance or user-provided paths.

## Entry Status

| Dataset | Method | Limit | Local | Remote selected split | Action |
|---|---:|---:|---:|---:|---|
| `gsm8k_main_train` | hf_rows | 64 | 37.1 KB | 2.2 MB | `download_manifest_sample_full_split_disk_feasible` |
| `gsm8k_main_test` | hf_rows | 64 | 34.8 KB | 409.3 KB | `download_manifest_sample_full_split_disk_feasible` |
| `openmathinstruct2_train_1m` | hf_rows | 64 | 98.1 KB | 609.4 MB | `download_manifest_sample_full_split_disk_feasible` |
| `numinamath_cot_train` | hf_rows | 64 | 188.5 KB | 1.1 GB | `download_manifest_sample_full_split_disk_feasible` |
| `numinamath_tir_train` | hf_rows | 64 | 309.7 KB | 140.5 MB | `download_manifest_sample_full_split_disk_feasible` |
| `bigcodebench_v014` | hf_rows | 64 | 375.4 KB | 2.3 MB | `download_manifest_sample_full_split_disk_feasible` |
| `lean_workbook_train` | hf_rows | 32 | 23.1 KB | 4.4 MB | `download_manifest_sample_full_split_disk_feasible` |
| `proofnetsharp_valid` | hf_rows | 64 | 72.3 KB | 96.6 KB | `download_manifest_sample_full_split_disk_feasible` |
| `moleculenet_lipophilicity` | hf_rows | 64 | 5.0 KB | 123.3 KB | `download_manifest_sample_full_split_disk_feasible` |
| `the_stack_v2` | hf_rows | 0 | 266.0 B | unknown | `skip_manifest_only_large_or_restricted` |
| `zinc20` | hf_rows | 0 | 242.0 B | unknown | `skip_manifest_only_large_or_restricted` |
| `unigenx_qm9_train` | hf_rows | 16 | 18.9 KB | 172.0 MB | `download_manifest_sample_full_split_disk_feasible` |
| `unigenx_materials_crystal_system` | hf_rows | 16 | 15.3 KB | 760.9 KB | `download_manifest_sample_full_split_disk_feasible` |
| `hebrew_sefaria_train` | hf_rows | 32 | 29.1 KB | 519.0 MB | `download_manifest_sample_full_split_disk_feasible` |
| `hebrew_synthetic_medical_train` | hf_rows | 32 | 60.1 KB | 5.7 MB | `download_manifest_sample_full_split_disk_feasible` |
| `hebrew_wikianswers_lists` | hf_rows | 32 | 3.7 KB | 37.5 MB | `download_manifest_sample_full_split_disk_feasible` |
| `hebrew_wikianswers_queries` | hf_rows | 32 | 3.1 KB | 2.1 MB | `download_manifest_sample_full_split_disk_feasible` |
| `hebrew_alpaca_train` | hf_rows | 32 | 20.7 KB | 2.2 MB | `download_manifest_sample_full_split_disk_feasible` |
| `talmud_hebrew_train` | hf_rows | 16 | 7.6 MB | 6.7 MB | `download_manifest_sample_full_split_disk_feasible` |
| `hebrew_wikipedia_train` | hf_rows | 32 | 3.8 KB | 5.9 MB | `download_manifest_sample_full_split_disk_feasible` |
| `hebrew_ud_htb` | git_clone | 0 | 13.4 MB | unknown | `clone_or_reuse_git_repo` |
| `hebrew_qa_nnlp` | git_clone | 0 | 20.9 MB | unknown | `clone_or_reuse_git_repo` |
| `hebrew_nakdimon` | git_clone | 0 | 244.8 MB | unknown | `clone_or_reuse_git_repo` |
| `hebrew_verb_complements_lexicon` | local_file | 0 | 283.0 B | unknown | `skip_local_user_provided` |
| `binding_affinity_public` | hf_rows | 16 | 9.6 KB | 134.0 MB | `download_manifest_sample_full_split_disk_feasible` |
| `chembl_local_export` | local_file | 0 | 271.0 B | unknown | `skip_local_user_provided` |
| `bindingdb_local_export` | local_file | 0 | 274.0 B | unknown | `skip_local_user_provided` |
| `naturelm_pubchem_local` | local_file | 0 | 274.0 B | unknown | `skip_local_user_provided` |
| `ugm_multimodal_synthetic` | local_generated | 32 | 35.7 KB | unknown | `generate_project_synthetic_rows` |
| `ugm_multimodal_local` | local_file | 0 | 273.0 B | unknown | `skip_local_user_provided` |
| `naturelm_uniprot_local` | local_file | 0 | 274.0 B | unknown | `skip_local_user_provided` |
| `naturelm_refseq_local` | local_file | 0 | 273.0 B | unknown | `skip_local_user_provided` |
| `naturelm_materials_project_local` | local_file | 0 | 284.0 B | unknown | `skip_local_user_provided` |
| `pdbbind_docking_local` | local_file | 0 | 273.0 B | unknown | `skip_local_user_provided` |
| `ec_protein_generation_local` | local_file | 0 | 279.0 B | unknown | `skip_local_user_provided` |
