# PLAN-E: Hebrew Morphology, Shoresh, and Root-Extension Reasoning

Status: implementation plan and verification log  
Date: 2026-04-29

## 1. Goal

This slice integrates Hebrew pretraining and intermediate-training data into the random-order TokenGT reasoning stack. The focus is Hebrew text, morphology, lemmatization, shoresh/root structure, and graph-of-thought root-extension training.

Implementation goals:

1. add dataset manifest entries for the requested Hebrew Hugging Face datasets;
2. clone public GitHub resources for Hebrew QA, UD Hebrew HTB, and Nakdimon/diacritization where licenses allow local research use;
3. add Hebrew graphification that turns text into token, lemma/root, radical, template, dependency, QA, and instruction-answer graphs;
4. add root-extension graph data for GFlowNet trajectory-balance training;
5. add configs for Hebrew SFT, Hebrew root GFlowNet, and validation;
6. add Hebrew-specific metrics, tests, and documentation.

## 2. Background Research

### 2.1 Downloadable Hugging Face Text Datasets

Requested HF datasets and observed schemas:

- `sivan22/sefaria-hebrew`: Hebrew Jewish texts from Sefaria; rows include `language`, `title`, `versionSource`, `versionTitle`, `license`, and `text`.
- `cp500/synthetic_hebrew_medical_text`: synthetic Hebrew medical notes; rows include `text`. The dataset card has sparse provenance, so this remains smoke-test only until reviewed.
- `imvladikon/wikianswers_hebrew`: Hebrew WikiAnswers sentence-similarity style data; configs include `lists` and `queries`, each with `record_id`, `set_id`, and `text`.
- `ashercn97/hebrew_alpaca`: Hebrew instruction-following rows with `instruction`, `input`, and `output`.
- `guyhadad01/Talmud-Hebrew`: small Talmud text rows with `id` and `content`.
- `YanFren/Hebrew_wikipedia`: MIT-tagged Hebrew Wikipedia text rows.

Decision:

- Pull small samples by default using the existing Dataset Viewer acquisition path.
- Convert each row into Hebrew text/instruction/QA graphs and add heuristic root-template nodes for roots inferred from the Hebrew surface forms.
- Keep license/provenance warnings in metadata and docs.

### 2.2 Hebrew Treebank / UD Hebrew HTB

The reliable public replacement for "Hebrew Treebank 2.0" in this repo is `UniversalDependencies/UD_Hebrew-HTB`. Its README states that it is a Universal Dependencies corpus for Hebrew, converted from the Hebrew Constituency Treebank v2, with 6,216 sentences in CoNLL-U format. The UD page reports CC BY-NC-SA 4.0, Hebrew morphological features including `HebBinyan`, and dependency labels including `compound:affix`, `compound:smixut`, `nsubj`, `obj`, and `root`.

Decision:

- Clone the UD repo under `data/raw/hebrew_ud_htb/repo`.
- Parse CoNLL-U locally without a new dependency.
- Graphify tokens, lemmas, UPOS/XPOS, morphological features, dependency edges, `HebBinyan`, and inferred shoresh/root nodes.
- Use UD data as the main structured morphology backbone.

### 2.3 Verb Complements Lexicon

The requested Verb Complements Lexicon schema appears in public snippets with fields such as `verb_complement`, `verb_LexiconItem`, `verb_dottedLexiconItem`, `verb_transliteratedLexiconItem`, `verb_binyan`, `verb_root`, and complement statistics. I did not find an official downloadable repository or canonical source URL during research.

Decision:

- Add manifest-only documentation and implement TSV/CSV ingestion support for this schema.
- Do not fabricate or scrape unlicensed data. If a local TSV/CSV is later placed under `data/raw/hebrew_verb_complements_lexicon/`, the graphifier can ingest it and add root-complement graphs.

### 2.4 Dicta/Nakdimon and Diacritized Hebrew

Dicta/Nakdan is a professional Hebrew diacritizer. Nakdimon is an MIT-licensed open-source Hebrew diacritizer; its README points to `hebrew_diacritized` as the training set and reports training/evaluation workflows. MenakBERT's model card also points to the Nakdimon dataset and describes it as 274,436 dotted Hebrew tokens across 413 documents.

Decision:

- Clone `elazarg/nakdimon` with submodules when possible.
- Use available text/test files as diacritized-vs-undotted graph pairs.
- Treat these as vocalization/morphology support data rather than root-gold data.

### 2.5 Semitic Root Encoding

Semitic Root Encoding (SRE) represents Semitic words with root, template-stem, and BPE tokens. The available paper is a method paper rather than a public dataset dump. It motivates this repo's graphifier: root nodes, radical nodes, template nodes, and surface-form nodes are all first-class graph tokens.

Decision:

- Implement SRE-style graphification with heuristic roots and templates.
- Add synthetic root-extension graphs for controlled GFlowNet training.
- Keep the method separate from claims of gold morphological analysis unless a gold root field exists.

## 3. Slice E1: Acquisition

Required implementation:

- Add HF manifest entries for all requested Hebrew datasets.
- Add `git_clone` support to `scripts/acquire_datasets.py` via `src/iska_reasoner/data/acquire.py`.
- Add clone manifest entries for:
  - `UniversalDependencies/UD_Hebrew-HTB`;
  - `NNLP-IL/Hebrew-Question-Answering-Dataset`;
  - `elazarg/nakdimon`;
  - manifest-only Verb Complements Lexicon.

Definition of done:

- HF samples can be pulled into `data/raw/<name>/`.
- GitHub repos can be cloned into `data/raw/<name>/repo`.

## 4. Slice E2: Hebrew Graphification

Required implementation:

- Add Hebrew normalization and diacritic stripping utilities.
- Add heuristic shoresh/root extraction and template signatures.
- Add graphifiers for:
  - generic Hebrew text rows;
  - instruction-answer rows;
  - Hebrew QA rows;
  - UD CoNLL-U sentences;
  - Nakdimon dotted/undotted pairs;
  - optional Verb Complements TSV rows.
- Add root-extension synthetic graphs.

Definition of done:

- Graph schema validation passes for all new graph types.
- Target tokens include `HEBREW:root:*`, `HEBREW:template:*`, `HEBREW:lemma:*`, `HEBREW:binyan:*`, and `HEBREW:derived:*` where applicable.

## 5. Slice E3: Training and GFlowNet Adaptation

Required implementation:

- Add `config/data/hebrew_mix.yaml`, `config/data/hebrew_roots.yaml`, `config/train/hebrew_sft_tiny.yaml`, `config/train/hebrew_root_gflownet_tiny.yaml`, and `config/validate/hebrew_validation.yaml`.
- Route Hebrew root-extension examples through the existing GFlowNet trajectory-balance stage.
- Add Hebrew metrics to validation and GFlowNet logs.

Definition of done:

- Hebrew SFT smoke training can run.
- Hebrew root GFlowNet smoke training can run.
- Validation reports Hebrew root/template/QA/diacritic metrics.

## 6. Slice E4: Tests and Docs

Required implementation:

- Add tests for root inference, Hebrew text graphification, UD CoNLL-U graphification, Verb Complements TSV support, root-extension generation, and Hebrew metrics.
- Update README and planning docs with acquisition, graphification, training, validation, and licensing notes.

## 7. Deferred Items

- Production-grade Hebrew morphological analyzer integration.
- Dicta/Nakdan API calls.
- Gold shoresh extraction beyond UD lemma/binyan features and optional local Verb Complements TSV files.
- Full-scale pretraining on all requested Hebrew corpora.
- Human validation of heuristic root/template outputs.

## 8. Verification Log

- Added `git_clone` acquisition support in `src/iska_reasoner/data/acquire.py`.
- Added manifest entries for requested HF datasets, UD Hebrew HTB, NNLP-IL HeQ, Nakdimon, and manifest-only Verb Complements Lexicon support.
- Pulled tiny HF samples into:
  - `data/raw/hebrew_sefaria_train/train.jsonl`
  - `data/raw/hebrew_synthetic_medical_train/train.jsonl`
  - `data/raw/hebrew_wikianswers_lists/train.jsonl`
  - `data/raw/hebrew_wikianswers_queries/train.jsonl`
  - `data/raw/hebrew_alpaca_train/train.jsonl`
  - `data/raw/talmud_hebrew_train/train.jsonl`
  - `data/raw/hebrew_wikipedia_train/train.jsonl`
- Cloned public repositories into:
  - `data/raw/hebrew_ud_htb/repo`
  - `data/raw/hebrew_qa_nnlp/repo`
  - `data/raw/hebrew_nakdimon/repo`
- Added Hebrew morphology utilities and graphification support in `src/iska_reasoner/data/hebrew.py`.
- Added `scripts/prepare_hebrew_sources.py` for UD CoNLL-U, HeQ, Nakdimon, root-extension synthetic data, and optional Verb Complements TSV/CSV ingestion.
- Graphified and curated `data/processed/hebrew_mix/all.jsonl`.
  - Local curation summary: 128 input rows, 112 kept rows, 16 duplicate synthetic root rows removed, 0 invalid rows.
  - Task families present: classical text, diacritization, instruction SFT, medical pretraining, morphosyntax, general pretraining, QA, question similarity, and root extension.
- Added Hebrew configs:
  - `config/data/hebrew_mix.yaml`
  - `config/data/hebrew_roots.yaml`
  - `config/train/hebrew_sft_tiny.yaml`
  - `config/train/hebrew_root_gflownet_tiny.yaml`
  - `config/validate/hebrew_validation.yaml`
- Added Hebrew tests in `tests/test_hebrew_slices.py`.
- Fixed random-order collator source truncation so long source graphs reserve target `<POS>` slots and do not produce all-ignore-label batches.
- Ran Hebrew SFT smoke training:
  - `conda run -n tokengt python scripts/train_stage.py --config config/model/tiny_lora_checkpointed.yaml --config config/data/hebrew_mix.yaml --config config/train/hebrew_sft_tiny.yaml`
  - output checkpoint: `outputs/hebrew_sft_tiny/checkpoint_final.pt`
- Ran Hebrew root-extension GFlowNet training:
  - `conda run -n tokengt python scripts/train_stage.py --config config/data/hebrew_roots.yaml --config config/train/hebrew_root_gflownet_tiny.yaml`
  - output checkpoint: `outputs/hebrew_root_gflownet_tiny/gflownet_final.pt`
- Ran validation:
  - `conda run -n tokengt python scripts/validate_stage.py --config config/validate/hebrew_validation.yaml --device cpu`
  - reports `validation/hebrew/*`, topology, tropical, and verifier metrics.
  - `conda run -n tokengt python scripts/validate_gflownet.py --config config/data/hebrew_roots.yaml --config config/train/hebrew_root_gflownet_tiny.yaml --checkpoint outputs/hebrew_root_gflownet_tiny/gflownet_final.pt --data data/processed/hebrew_root_synthetic/train.jsonl --device cpu --output outputs/hebrew_root_gflownet_tiny/validation.json`
  - reports `gflownet_val/hebrew/*` metrics.
- Ran full tests:
  - `conda run -n tokengt pytest -q`
  - status: `21 passed, 2 warnings`.
