# PLAN-G: Correct SFM NatureLM and UniGenX Repo Integration

Status: implemented and smoke-tested  
Date: 2026-04-29

## 1. Correction

The NatureLM reference for this project is the Science Foundation Model repo:

- `https://github.com/amelie-iska/SFM`

The UniGenX reference is:

- `https://github.com/amelie-iska/UniGenX`

Earlier PLAN-D docs mixed NatureLM and UniGenX sources incorrectly. Those notes have been superseded; this file is the authoritative source for the NatureLM/UniGenX integration.

## 2. Paper Research

Primary papers and source links:

- NatureLM: `Nature Language Model: Deciphering the Language of Nature for Scientific Discovery`, arXiv:2502.07527.
  - Key implementation facts: scientific entities are represented as sequences; domains include small molecules, materials, proteins, DNA, and RNA; examples use explicit domain tags such as `<protein>`, `<material>`, `<dna>`, `<rna>`, `<antibody>`, and material space-group markers.
- UniGenX: `UniGenX: a unified generative foundation model that couples sequence, structure and function to accelerate scientific design across proteins, molecules and materials`, arXiv:2503.06687.
  - Key implementation facts: heterogeneous symbolic/numeric data is represented as a mixed stream; a decoder-only autoregressive transformer supplies global context; a conditional diffusion head generates numeric fields; domains include materials, molecules, proteins, and protein-ligand docking.

Local repo inspection:

- `data/external_repos/sfm` at commit `bcb1351bc6490d4a98cae74e7eead30a028fae00`.
  - Contains `NatureLM/README.md`, checkpoint links, and examples of domain-tagged sequence instructions/responses.
- `data/external_repos/unigenx` at commit `43d7d49c3e243b982696d4659622ef80c0d1e8e2`.
  - Contains `unigenx/data/dict*.txt`, `unigenx/data/tokenizer.py`, `unigenx/data/dataset.py`, `unigenx/model/diffloss.py`, and `unigenx/model/unigenx.py`.
  - Dictionaries include material elements, molecule/SMILES tokens, protein amino acids, unified domain tags, order tokens, and coordinate tokens.

## 3. Implementation Plan

1. Add/retarget reference repo acquisition so `sfm` and `unigenx` clone into `data/external_repos/`.
2. Add token extraction from:
   - SFM/NatureLM README domain examples;
   - UniGenX dictionary files and tokenizer special tokens.
3. Extend `GraphVocab` build paths to accept extra vocabulary files from these extracted tokens.
4. Add configs that include extracted SFM/UniGenX token vocab for science training.
5. Update docs so SFM/NatureLM and UniGenX source provenance is unambiguous.
6. Update model-repo metadata acquisition to use the GitHub repos by default.
7. Test reference-token extraction, vocab extension, and existing training/validation smoke paths.

## 4. Definition of Done

- `scripts/acquire_model_files.py --repo-name sfm` and `--repo-name unigenx` work against GitHub repos.
- `scripts/extract_reference_tokens.py` writes an extra-vocab text file from SFM and UniGenX.
- Science data configs consume that extra vocab.
- README and planning docs cite the right repos and papers.
- `conda run -n tokengt pytest -q` passes.

## 5. Implementation Log

- Cloned/refreshed SFM at `data/external_repos/sfm`, commit `bcb1351bc6490d4a98cae74e7eead30a028fae00`.
- Cloned/refreshed UniGenX at `data/external_repos/unigenx`, commit `43d7d49c3e243b982696d4659622ef80c0d1e8e2`.
- Added `src/iska_reasoner/data/reference_repos.py` and `scripts/extract_reference_tokens.py`.
- Extracted `1164` SFM/NatureLM and UniGenX reference tokens into `data/processed/reference_tokens/naturelm_unigenx_tokens.txt`.
- Updated `GraphVocab` so `data.extra_vocab_paths` tokens are protected even when a future `max_vocab_size` is configured.
- Retargeted `data/manifests/model_repos.yaml` to GitHub reference repos by default, while keeping NatureLM Hugging Face checkpoints as explicit optional entries.
- Removed the old audio-dataset integration from the active science pipeline and configs.
- Rebuilt the tiny science mix from UniGenX molecule/material rows plus SFM/NatureLM reference vocabulary.

## 6. Verification Log

- `conda run -n tokengt python scripts/acquire_model_files.py --repo-name sfm`: wrote `data/external_models/sfm/FILES.json`.
- `conda run -n tokengt python scripts/acquire_model_files.py --repo-name unigenx`: wrote `data/external_models/unigenx/FILES.json`.
- `conda run -n tokengt python scripts/acquire_model_files.py --repo-name naturelm_8x7b_hf_checkpoint`: wrote optional checkpoint metadata to `data/external_models/naturelm_8x7b_hf_checkpoint/FILES.json` without downloading shards.
- `conda run -n tokengt python scripts/extract_reference_tokens.py`: wrote `1164` tokens.
- `conda run -n tokengt pytest -q`: `29 passed`, with only PyTorch nested-tensor warnings.
- `conda run -n tokengt python scripts/train_stage.py --config config/model/tiny_lora_checkpointed.yaml --config config/data/science_mix.yaml --config config/train/science_sft_tiny.yaml`: completed 20 science SFT steps after the source cleanup and rebuilt `outputs/science_sft_tiny/vocab.jsonl` with `1316` tokens.
- Verified the rebuilt science vocab contains `<protein>`, `<material>`, `<coord>`, and `UNIGENX:TOK:<molecule>`.
- `conda run -n tokengt python scripts/validate_stage.py --config config/validate/science_validation.yaml --device cpu`: completed and reported science, topology, tropical, verifier, and numeric-diffusion metrics.
- Added `planning/LICENSE-REVIEW.md` and `planning/RUNBOOK-4090.md`.
- Added W&B online/offline overlays and 4090 train configs.
- Installed/verified optional runtime tools in the `tokengt` env: `datasets`, `transformers`, `rdkit`, `soundfile`, `torchaudio`, `ripser`, `gudhi`, and Lean 4 via `elan`.
- `conda run -n tokengt python scripts/check_readiness.py --json`: reports CUDA RTX 4090, Lean, SFM/UniGenX repos, reference vocabulary, W&B, RDKit, audio libraries, and topology libraries available.
- W&B offline overlay smoke training completed and wrote a local offline W&B run.
- `conda run -n tokengt python scripts/infer.py --config config/inference/science_tiny_inference.yaml --text "Create a graph reasoning sketch for a protein-to-molecule design task." --max-steps 4 --device cpu`: completed inference smoke. The checkpoint is still a smoke checkpoint, so output quality is not meaningful yet.
