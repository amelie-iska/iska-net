# PLAN-F: Deferred Component Closure

Status: implemented with smoke verification  
Date: 2026-04-29

## 1. Goal

PLAN-F closes the previously deferred implementation surface from PLAN-B through PLAN-E without turning local smoke workflows into uncontrolled multi-terabyte downloads. The implementation rule is:

- code paths, schemas, configs, tests, and docs are implemented now;
- huge/gated/unsafe datasets and checkpoints are exposed through explicit opt-in acquisition or local-file preparation paths;
- no large checkpoint, audio corpus, or full external science corpus is downloaded by default.

## 2. Deferred Items Covered

### 2.1 Topology

- Add optional `ripser`/`gudhi` integration when installed.
- Add fallback persistence summaries beyond H0 union-find.
- Add persistent Laplacian-style spectral summaries over graph filtrations.

### 2.2 Tropical Reasoning

- Add a tropical attention module for max-plus/hard-selection experiments.
- Add MLP/hidden-state cell transition signatures.
- Add a tropical dependency parser based on maximum spanning arborescence.

### 2.3 Verifiers, Sandboxing, and Curation

- Add stricter subprocess resource limits for Python verifier execution.
- Add MinHash/near-dedup, contamination scanning, and license-policy filtering to curation.
- Keep true microVM sandboxing documented as an external deployment boundary; local code gets OS resource limits and isolated tempdirs.

### 2.4 Autoregressive Numeric Records and UniGenX-Style Science

- Keep numeric coordinate/property supervision inside the random-order graph-token stream rather than a separate diffusion head.
- Encode generated structure positions as identity-bearing coordinate tokens, for example `COORD:f0:a17:x:pos_near`.
- Extend graphification for proteins, EC-number tasks, protein-ligand docking, and local SFM science-source reconstruction.
- Keep large UniGenX checkpoint download explicit through model acquisition flags.

### 2.5 Local Audio Features

- Add optional audio feature extraction for local audio paths using `soundfile`, `torchaudio`, or stdlib WAV fallback.
- Keep external audio corpora outside the active science pipeline unless a separate reviewed source is added.

### 2.6 Biomedical Data

- Add ChEMBL/BindingDB-style local-file ingestion and graphification.
- Add manifest entries for public binding-affinity substitutes and full-source local manifests.
- Add assay normalization, simple safety/medicinal filters, and metrics.

### 2.7 GFlowNet

- Add optional learned backward policy.
- Add context-coupled policy inputs from graph features.
- Add subtrajectory-balance auxiliary loss.
- Add graph edit action-space utilities beyond simple add-token actions.

### 2.8 External LLM / Upstream TokenGT Scale-Up

- Add opt-in QLoRA/PEFT training script skeleton with hard dependency checks.
- Add upstream TokenGT/Fairseq wrapper probe and docs.
- Keep full upstream Fairseq training out of the smoke path unless dependencies are installed.

## 3. Definition of Done

- New modules and scripts have tests.
- Existing smoke commands keep working.
- Configs expose the new paths without changing the default small runs.
- Docs identify which components are opt-in due to size, license, or external dependencies.
- `conda run -n tokengt pytest -q` passes after implementation.

## 4. Verification Log

Implemented on 2026-04-29:

- Added advanced topology summaries:
  - optional `ripser`/`gudhi` persistence;
  - fallback H0/H1 graph persistence;
  - persistent-Laplacian-style shortest-path filtration spectra.
- Added tropical utilities:
  - `TropicalAttention`;
  - hidden activation cell signatures;
  - maximum-spanning-arborescence parser.
- Replaced the old UniGenX-style numeric diffusion experiment with autoregressive graph-token numeric records:
  - generated coordinate targets use frame/atom/axis/bin tokens;
  - active training and validation no longer log `numeric_diffusion_loss`;
  - source numeric features such as temperature remain available as conditioning features.
- Added PLAN-F science graphification:
  - protein/EC rows;
  - protein-ligand docking rows;
  - ChEMBL/BindingDB-style bioactivity rows;
  - local `scripts/prepare_science_sources.py` for CSV/TSV/JSON/JSONL/FASTA sources.
- Added local audio feature extraction for rows with `local_audio_path` or `audio_path`.
- Added stronger curation:
  - MinHash near-dedup;
  - blocked license pattern filtering;
  - contamination-file filtering.
- Added safer verifier execution through temporary directories plus POSIX memory, CPU, and file-size limits.
- Added medicinal-chemistry triage metrics and bioactivity metrics.
- Added GFlowNet upgrades:
  - topology context conditioning;
  - learned backward policy;
  - subtrajectory-balance auxiliary loss;
  - graph edit action-space labels.
- Added opt-in scale-up/probe scripts:
  - `scripts/train_qlora_external.py`;
  - `scripts/probe_upstream_tokengt.py`.
- Added manifest entries for public binding-affinity smoke data and local ChEMBL, BindingDB, PubChem, UniProt, RefSeq, Materials Project, PDBbind/docking, and EC protein-generation sources.

Commands run:

- `conda run -n tokengt pytest -q`
  - result: `29 passed, 3 warnings`.
- `conda run -n tokengt python scripts/train_stage.py --config config/model/tiny_lora_checkpointed.yaml --config config/data/science_mix.yaml --config config/train/science_sft_tiny.yaml`
  - result: completed 20 science SFT steps.
- `conda run -n tokengt python scripts/validate_stage.py --config config/validate/science_validation.yaml --device cpu`
  - result: passed and reported `validation/science/*` and chemistry triage metrics.
- `conda run -n tokengt python scripts/train_stage.py --config config/data/synthetic_graphs.yaml --config config/train/gflownet_got_tiny.yaml`
  - result: completed 20 context/backward/subtrajectory GFlowNet steps.
- `conda run -n tokengt python scripts/validate_gflownet.py --config config/train/gflownet_got_tiny.yaml --config config/validate/gflownet_validation.yaml --device cpu`
  - result: passed and reported action coverage and context-enabled rollout metrics.
- `conda run -n tokengt python scripts/train_stage.py --config config/data/hebrew_roots.yaml --config config/train/hebrew_root_gflownet_tiny.yaml`
  - result: completed 20 context/backward/subtrajectory Hebrew root GFlowNet steps.
- `conda run -n tokengt python scripts/validate_gflownet.py --config config/data/hebrew_roots.yaml --config config/train/hebrew_root_gflownet_tiny.yaml --checkpoint outputs/hebrew_root_gflownet_tiny/gflownet_final.pt --data data/processed/hebrew_root_synthetic/train.jsonl --device cpu --output outputs/hebrew_root_gflownet_tiny/validation.json`
  - result: passed and reported `gflownet_val/hebrew/*` metrics.
- `conda run -n tokengt python scripts/acquire_datasets.py --dataset binding_affinity_public --limit 2`
  - result: acquired two public binding-affinity rows.
- `conda run -n tokengt python scripts/graphify_data.py --input data/raw/binding_affinity_public/train.jsonl --output data/processed/binding_affinity_public/train.jsonl --dataset-name binding_affinity_public`
  - result: produced two `biomed_bioactivity` graph rows.
