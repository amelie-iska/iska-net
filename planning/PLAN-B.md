# PLAN-B: Completing the Random-Order TokenGT Agentic Reasoning Stack

Status: implementation plan for closing PLAN-A scaffold gaps  
Date: 2026-04-29  

## 1. Goal

PLAN-A produced a working scaffold. PLAN-B turns it into a more complete research system by implementing the missing subsystems that were identified after the first pass:

- topology and distogram diagnostics;
- tropical/annealing diagnostics;
- verifier/tool adapters;
- curation, deduplication, splits, quality scoring, and provenance tracking;
- richer validation metrics;
- more faithful graph-of-thought GFlowNet actions and rewards;
- better stage-specific metrics and W&B organization;
- inference-time graph-of-thought rollout hooks;
- documentation and tests.

This is still a local research implementation, not a frontier-scale training stack. Full 15B training, dense 256K context, production LeanDojo/mathlib extraction, full ChEMBL/BindingDB ingestion, and gated corpus ingestion remain explicit scale-up tasks.

## 2. Implementation Work Items

### 2.1 Repository Hygiene

- Add root `.gitignore` for outputs, caches, checkpoints, raw data, and env artifacts.
- Keep small curated JSONL samples in place for smoke tests.
- Do not modify `./tokengt` internals unless a wrapper requires it.

### 2.2 Topology and Distograms

Implement package `src/iska_reasoner/topology/`.

Required:

- graph shortest-path distograms;
- graph component count;
- graph cycle rank;
- degree statistics;
- edge-type entropy;
- lightweight H0 persistence by union-find over pairwise distances;
- Laplacian spectral summary using NumPy;
- topology metric prefix for W&B:
  - `topology/components_mean`
  - `topology/cycle_rank_mean`
  - `topology/edge_type_entropy_mean`
  - `topology/h0_total_persistence_mean`
  - `topology/laplacian_lambda2_mean`

Deferred scale-up:

- differentiable persistent homology;
- persistent Laplacians over filtrations;
- GPU TDA;
- `ripser`/`gudhi` optional integration.

### 2.3 Tropical Diagnostics

Implement package `src/iska_reasoner/tropical/`.

Required:

- temperature schedule;
- entropy and top-1/top-2 margin of logits;
- max-plus hard-selection diagnostics;
- optional stage config keys:
  - `tropical.enabled`
  - `tropical.temperature`
  - `tropical.temperature_min`
  - `tropical.anneal_steps`

Metrics:

- `tropical/logit_entropy`
- `tropical/top1_margin`
- `tropical/temperature`

Deferred scale-up:

- custom tropical attention kernel;
- MLP cell transition signatures from gate patterns;
- tropical dynamic programming parsers.

### 2.4 Verifier and Tool Adapters

Implement package `src/iska_reasoner/tools/`.

Required:

- Python code execution with timeout for controlled snippets;
- simple numeric answer checker;
- Lean syntax/tool availability adapter with graceful fallback;
- RDKit SMILES validity adapter with graceful fallback;
- graph example reward adapter used by validation and GFlowNet.

Metrics:

- `verifier/pass_rate`
- `verifier/python_pass_rate`
- `verifier/lean_available`
- `verifier/rdkit_available`
- `verifier/reward_mean`

Deferred scale-up:

- LeanDojo state tracing;
- full mathlib proof compilation pipeline;
- secure sandbox/microVM for untrusted code;
- real chemistry property filters and assay consistency checks.

### 2.5 Dataset Curation

Implement `src/iska_reasoner/data/curate.py` and `scripts/curate_data.py`.

Required:

- schema validation;
- exact duplicate removal by normalized graph hash;
- deterministic train/val/test split;
- quality score based on graph density, target count, verifier/tool/evidence metadata, and task type;
- provenance summary JSON;
- tqdm progress bars.

Metrics/docs:

- counts before/after dedup;
- split sizes;
- quality score summary;
- task distribution.

Deferred scale-up:

- MinHash/near-dedup;
- benchmark contamination checks;
- license policy enforcement;
- semantic leakage scans.

### 2.6 Model and Training Enhancements

Required:

- Add tropical diagnostics to normal training steps.
- Add topology batch summaries to data loader/training logs.
- Add stage-specific metric namespaces:
  - `graph_pretrain/*`
  - `tool_sft/*`
  - `proof_code_sft/*`
  - `topology_aux/*`
  - `gflownet/*`
- Add optional topology auxiliary loss placeholder that is real but lightweight: regress graph-level topology summary from graph representation.
- Add checkpoint config compatibility for the new topology head.

Deferred scale-up:

- actual upstream TokenGT Fairseq/Performer/Laplacian encoder swap;
- LoRA/QLoRA;
- gradient checkpointing;
- FlashAttention kernels.

### 2.7 GFlowNet Enhancements

Required:

- Replace pure set-only toy reward with verifier-aware reward.
- Track action coverage, exact match, token recall, extra-token penalty.
- Save rollout samples.
- Log trajectory diversity.

Deferred scale-up:

- learned backward policy;
- graph edit actions beyond target-token add actions;
- coupling directly to the TokenGT hidden state;
- subtrajectory balance.

### 2.8 Validation and Inference

Validation required:

- existing loss/perplexity/token accuracy;
- graph-level topology summaries;
- verifier reward/pass rate;
- random-order exact-set match proxy;
- GFlowNet reward summaries when checkpoint type is GFlowNet.

Inference required:

- graph-of-thought rollout mode that can call verifier reward adapter after token generation;
- output generated tokens plus reward/pass diagnostics.

### 2.9 Tests

Add tests for:

- topology summaries;
- tropical diagnostics;
- verifier adapters;
- curation dedup/splits;
- enhanced validation metrics;
- GFlowNet verifier reward path.

## 3. Definition of Done

- [x] `PLAN-B.md` exists.
- [x] Topology, tropical, tools, and curation modules exist.
- [x] CLI scripts cover curation and richer validation/inference.
- [x] Training logs include topology/tropical/verifier metrics when enabled.
- [x] GFlowNet stage uses verifier-aware rewards.
- [x] README and planning docs are updated.
- [x] Tests pass.
- [x] A smoke run verifies training, validation, inference, curation, and GFlowNet after the changes.

## 4. Verification Log

Completed on 2026-04-29 in the existing `tokengt` conda environment:

- `pytest -q`: 8 passed.
- Curation CLI: deduplicated `data/processed/mixed_graphs/train.jsonl` into `data/processed/curated_graphs`.
- Graph pretrain tiny run: completed 20 steps with topology/tropical metrics.
- Validation CLI: emitted loss, perplexity, token accuracy, topology, tropical, and verifier metrics.
- Inference CLI: emitted generated graph tokens and verifier diagnostics.
- GFlowNet tiny run: completed 20 verifier-aware trajectory-balance steps.
- Topology auxiliary tiny run: completed 20 steps with topology-head loss.
