# PLAN-A: Random-Order TokenGT Agentic Reasoning Model

Status: implemented scaffold with smoke-test verification  
Date: 2026-04-28  
Repository root: `/home/iska/Documents/amelie/bio/iska-net`

## 1. Objective

Build a training-ready research codebase for a sub-15B, single-RTX-4090-friendly random-order autoregressive agentic reasoning model based on TokenGT-style graph tokens. The system should implement the core methods described in `assets/human_learning_transformer_learning_review_extended.tex`:

- typed graphification of language, code, proof, tool, and biomedical examples;
- TokenGT node/edge/hyperedge style sequence encoding;
- random-order autoregressive graph decoding instead of strict right-to-left or left-to-right sequence decoding;
- staged training for graph pretraining, supervised tool/proof/code reasoning, verifier repair traces, and GFlowNet-style graph-of-thought trajectory learning;
- validation and inference entry points for each stage;
- step-based metrics, progress bars, structured logging, and W&B reporting;
- 4090-feasible defaults using small model configs, gradient accumulation, mixed precision, and small curated samples.

The implementation should be organized so a new colleague can understand the project from the README, planning docs, configs, and source tree.

## 2. Constraints and Assumptions

- Current repository has an existing `./tokengt` checkout. We will treat it as the architectural reference and avoid invasive edits inside it unless necessary.
- The root repository itself is not currently a Git repository; do not assume root-level Git metadata.
- Full pretraining of a 15B dense model on one RTX 4090 is not feasible. The code will support prototype-scale training and adapter/distillation pathways, with configs scaled down by default.
- Large datasets from the paper are mostly too large to download blindly. We will implement acquisition scripts/manifests and download small smoke-test subsets locally into `./data` where possible.
- The codebase should not depend on private APIs or unpublished datasets. “Codex 5.3/5.4/5.5 traces” will be represented as a teacher-synthetic trace pipeline with verifier gates, not as a claimed public dataset.

## 3. Target Directory Layout

```text
.
├── assets/
├── config/
│   ├── data/
│   ├── model/
│   ├── train/
│   ├── validate/
│   └── inference/
├── data/
│   ├── raw/
│   ├── external_repos/
│   ├── samples/
│   ├── processed/
│   ├── manifests/
│   └── README.md
├── planning/
│   ├── PLAN-A.md
│   ├── BACKGROUND-RESEARCH.md
│   ├── DATASETS.md
│   ├── METRICS.md
│   └── ARCHITECTURE.md
├── scripts/
│   ├── setup_conda_env.sh
│   ├── acquire_datasets.py
│   ├── graphify_data.py
│   ├── train_stage.py
│   ├── validate_stage.py
│   └── infer.py
├── src/
│   └── iska_reasoner/
│       ├── data/
│       ├── graph/
│       ├── models/
│       ├── training/
│       ├── gflownet/
│       ├── validation/
│       ├── inference/
│       └── utils/
└── tokengt/
```

## 4. Implementation Phases

### Phase 0: Repository Orientation

- Inspect existing `tokengt` interfaces, especially tokenization, encoder layers, collators, and attention modules.
- Identify the minimum reusable pieces from TokenGT: graph-token structural embeddings, graph encoder, attention modules, and ORF/laplacian encodings.
- Preserve the existing TokenGT checkout as a reference dependency; wrap it from project code rather than rewriting it wholesale.

### Phase 1: Planning and Research

- Write this plan before implementation.
- Perform background research on:
  - current TokenGT repo/API;
  - current GFlowNet reference implementations suitable for language/trajectory balance;
  - current public datasets and small downloadable subsets for proof, math, code, graph-of-thought, molecules, and tool-use;
  - current Hugging Face dataset availability and download methods.
- Update this plan after research with concrete repo URLs, dataset manifests, and implementation decisions.

### Phase 2: Environment and Project Scaffolding

- Create `environment.yml` and `scripts/setup_conda_env.sh` for a conda environment.
- Include core dependencies:
  - `torch`, `pyyaml`, `tqdm`, `wandb`, `numpy`, `networkx`, `datasets`, `huggingface_hub`, `transformers`, `tokenizers`, `scikit-learn`;
  - optional/soft dependencies: `rdkit`, `lean-dojo`, `gudhi` or `ripser`, `pytest`, `ruff`.
- Create source package under `src/iska_reasoner`.
- Add robust config loading, logging, seed control, JSONL helpers, W&B helpers, checkpoint helpers, and device detection.

### Phase 3: Dataset Acquisition and Curation

- Create `data/manifests/datasets.yaml` listing datasets, expected size, license/provenance notes, default smoke-test limits, and acquisition method.
- Implement `scripts/acquire_datasets.py`:
  - supports `--manifest`, `--dataset`, `--limit`, `--dry-run`, `--force`;
  - downloads small JSONL samples to `data/raw/<dataset_name>/`;
  - records provenance and timestamps.
- Initial smoke-test datasets:
  - synthetic graph reasoning generated locally;
  - small math reasoning examples from a public HF dataset if accessible;
  - small code instruction examples from a permissive public HF dataset if accessible;
  - small molecule examples from a public HF dataset or local generated SMILES if needed;
  - local mini proof/tool traces generated by scripts for deterministic tests.
- Add clear documentation that large datasets such as FineWeb, Dolma, The Stack v2, ChEMBL, mathlib, and LeanDojo are manifest entries and scripted acquisition targets, not downloaded in full by default.

### Phase 4: Graphification and Random-Order Data

- Define a stable graph JSON schema:
  - nodes: `id`, `type`, `value`, `features`;
  - edges: `src`, `dst`, `type`, `features`;
  - optional hyperedges/simplices;
  - target graph delta;
  - decoder order(s);
  - verifier/tool trace;
  - provenance.
- Implement graph builders:
  - synthetic typed graph tasks;
  - text-to-claim/tool graph fallback;
  - code dependency graph fallback using Python AST;
  - molecule graph fallback for SMILES when RDKit is installed, otherwise simple parser stubs;
  - proof/tool trace graph format.
- Implement random-order target generation:
  - uniform random order;
  - dependency/topological order;
  - verifier-enabling order;
  - uncertainty/anchor placeholder order.
- Implement collators for masked visible-set autoregression over graph token IDs.

### Phase 5: Model

- Implement a compact `RandomOrderTokenGT` wrapper:
  - graph node/edge/type embeddings;
  - structural position embeddings;
  - TokenGT-compatible transformer encoder;
  - decoder heads for next graph token, node type, edge type, action type, and verifier value.
- Prefer wrapping local TokenGT modules when straightforward; otherwise implement a small compatible PyTorch transformer that preserves the TokenGT graph-token input contract and can later be swapped for the upstream TokenGT encoder.
- Implement causal/visible-set masking for random-order autoregression.
- Add optional topology summary features and tropical temperature controls.

### Phase 6: Training Stages

Implement a unified stage runner with stage-specific losses and metrics.

1. `graph_pretrain`: graph token reconstruction and random-order graph autoregression.
2. `tool_sft`: supervised tool call/action prediction and repair trace learning.
3. `proof_code_sft`: proof/code graph supervised training.
4. `topology_aux`: optional distogram/topology auxiliary prediction on graph distances.
5. `gflownet_got`: trajectory-balance training over graph-of-thought actions with verifier rewards.

All stages must support:

- tqdm progress bars;
- step-based W&B metrics;
- local JSONL metrics log;
- checkpoint save/load;
- validation every N steps;
- gradient accumulation, mixed precision, clipping;
- deterministic smoke-test mode.

### Phase 7: GFlowNet Implementation

- Clone or reference a GFlowNet repository into `data/external_repos/` for provenance and comparison.
- Implement local rigorous trajectory-balance training rather than only importing a demo:
  - forward policy over graph actions;
  - backward policy over reverse actions;
  - learnable or configured log partition `logZ`;
  - terminal reward from verifier proxy;
  - trajectory balance loss and metrics.
- Include metrics:
  - `gflownet/tb_loss`;
  - `gflownet/logZ`;
  - `gflownet/reward_mean`;
  - `gflownet/trajectory_len`;
  - `gflownet/terminal_valid_rate`;
  - `gflownet/action_entropy`.

### Phase 8: Validation and Inference

- Implement `scripts/validate_stage.py`:
  - random-order NLL;
  - graph edit accuracy;
  - node/edge/action F1 where labels exist;
  - verifier pass rate for synthetic/tool tasks;
  - topology distance proxy;
  - GFlowNet terminal reward and diversity metrics.
- Implement `scripts/infer.py`:
  - converts raw input or graph JSON to model input;
  - supports greedy, sampled, and random-order completion;
  - supports graph-of-thought rollout with optional verifier proxy;
  - writes graph outputs and decoded summaries.

### Phase 9: Documentation and Tests

- Write `README.md` with:
  - project purpose;
  - environment setup;
  - dataset acquisition commands;
  - graphification commands;
  - all training stages;
  - validation and inference commands;
  - W&B metrics documentation;
  - 4090-friendly defaults and scaling notes.
- Write planning docs:
  - `BACKGROUND-RESEARCH.md`;
  - `DATASETS.md`;
  - `METRICS.md`;
  - `ARCHITECTURE.md`.
- Add tests:
  - graph schema validation;
  - random-order masking;
  - tiny model forward/backward;
  - one-step training;
  - GFlowNet trajectory-balance smoke test;
  - inference smoke test.

## 5. Background Research Results and Decisions

Research pass completed on 2026-04-28. The implementation decisions below supersede the initial open questions.

### 5.1 TokenGT

- Existing checkout: `./tokengt`, remote `https://github.com/amelie-iska/tokengt.git`, commit `42d6e91`.
- Upstream public TokenGT repo: `https://github.com/jw9730/tokengt`.
- The usable conceptual interface is `GraphFeatureTokenizer` plus `TokenGTGraphEncoder`.
- The upstream large-scale model imports Fairseq and has an uninitialized `large-scale-regression/fairseq` submodule in this checkout, so the project will implement a local compact PyTorch model that preserves the TokenGT graph-token contract. This avoids making Fairseq a hard dependency for the new agent while still using `./tokengt` as the base/reference implementation.

### 5.2 GFlowNet

- Reference repo to clone: `https://github.com/GFNOrg/torchgfn.git`.
- Rationale: it is the current modular PyTorch GFlowNet library with trajectory-balance support and documentation. It is appropriate as a reference dependency, but the project will implement its own small graph-of-thought trajectory-balance trainer in `src/iska_reasoner/gflownet/` so the language/graph action semantics are explicit and auditable.

### 5.3 Dataset Acquisition Defaults

Default acquisition will use Hugging Face Dataset Viewer or `datasets` streaming to write small JSONL samples into `data/raw/<dataset>/`. Large datasets are represented in manifests and are not fully downloaded unless explicitly requested.

Initial concrete datasets:

- Math:
  - `openai/gsm8k`, config `main`, splits `train`, `test`, MIT license.
  - `nvidia/OpenMathInstruct-2`, config `default`, splits `train`, `train_1M`, `train_2M`, `train_5M`, CC-BY-4.0, large; sample only by default.
  - `AI-MO/NuminaMath-CoT`, config `default`, splits `train`, `test`, Apache-2.0.
  - `AI-MO/NuminaMath-TIR`, config `default`, splits `train`, `test`, Apache-2.0.
- Code/tool:
  - `bigcode/bigcodebench`, config `default`, latest visible split `v0.1.4`, Apache-2.0.
  - `deepmind/code_contests`, CC-BY-4.0, sample only.
  - `bigcode/the-stack-v2` is gated and very large; manifest only by default.
- Formal proof:
  - `internlm/Lean-Workbook`, config `default`, split `train`, Apache-2.0.
  - `PAug/ProofNetSharp`, config `default`, splits `valid`, `test`, MIT.
  - LeanDojo is retained as a documented target, but no public HF dataset ID was found in the research pass; use package/repo workflows later.
- Chemistry/biomed:
  - `scikit-fingerprints/MoleculeNet_Lipophilicity`, config `default`, split `train`, small and suitable for smoke tests.
  - `zpn/zinc20`, MIT but huge; manifest only by default.
  - ChEMBL/BindingDB should be added via dedicated downloaders later, not bulk-downloaded blindly.
- Synthetic:
  - local generated graph reasoning, graph-of-thought, proof/tool repair, Python AST, and simple molecule-like examples. These are required for deterministic tests.

### 5.4 Random-Order Autoregression

- The implementation will follow the paper's graph-set formulation rather than adopting a vision-specific random-order architecture wholesale.
- Relevant ideas:
  - XLNet-style permutation factorization validates random factorization masks.
  - sigma-GPT and RandAR motivate explicit order/position instruction tokens.
  - graph generation papers emphasize that graph autoregressive order is an important modeling choice.
- Local implementation decision: every graph token has a structural identity and a reveal-order identity; the model receives visible-set masks and predicts the next revealed graph token in the sampled order.

### 5.5 Environment

- Conda exists and there is already a `tokengt` env with Python 3.11 and PyTorch 2.8.0+cu128.
- Missing packages include `datasets` and `transformers`; the setup script/environment file will install them.
- Default commands will use `conda run -n iska-ugm ...` after creating the new env, but code should also run in the existing `tokengt` env once dependencies are installed.

## 6. Background Research Questions

1. Which GFlowNet repo is best to clone for reference and provenance?
2. Which public HF datasets are accessible and small enough for smoke-test acquisition now?
3. Which TokenGT module path is most stable for reuse from the local checkout?
4. What recent random-order autoregressive work should be referenced in the docs?
5. Which components need to be implemented locally to be faithful to the paper rather than only stitched together?

Resolved by Section 5; retain the questions as audit markers for future revisions.

## 7. Initial Risk Register

- TokenGT upstream code may depend on old Fairseq assumptions; mitigation: wrap concepts and keep a local small PyTorch implementation compatible with its graph-token contract.
- Full dataset acquisition can be too large; mitigation: manifest plus small samples and dry-run flags.
- GFlowNet language reasoning is under-specified; mitigation: implement a graph-action synthetic environment with verifier rewards first, then allow real verifier adapters.
- Topological losses can be expensive; mitigation: start with graph-distance/distogram proxies and optional `gudhi`/`ripser` integration.
- Lean/RDKit dependencies may be heavy; mitigation: soft imports and deterministic fallback validators for tests.

## 8. Definition of Done

- [x] `planning/PLAN-A.md` is revised after research.
- [x] `README.md`, `planning/DATASETS.md`, `planning/METRICS.md`, and `planning/ARCHITECTURE.md` exist and are accurate.
- [x] `config/` contains meaningful YAML files for model, data, training, validation, and inference.
- [x] `scripts/setup_conda_env.sh` and `environment.yml` exist.
- [x] dataset acquisition and graphification scripts run in smoke-test mode.
- [x] training, validation, inference, and GFlowNet training scripts run on a tiny sample.
- [x] tests pass in the existing `tokengt` conda env.
- [x] W&B metrics are implemented with offline/disabled fallback.
- [x] No large unintended downloads are required to run the smoke tests.

## 9. Verification Log

Completed on 2026-04-28 in the existing `tokengt` conda environment:

- `pytest -q`: 4 passed.
- Synthetic graph generation: wrote 64 rows to `data/processed/synthetic_graphs/train.jsonl`.
- `graph_pretrain` tiny run: completed 20 steps and wrote `outputs/synthetic_graph_pretrain/checkpoint_final.pt`.
- Validation CLI: produced `validation/loss`, `validation/perplexity`, and `validation/token_accuracy`.
- Inference CLI: generated graph tokens from a text prompt.
- `gflownet_got` tiny run: completed 20 trajectory-balance steps and wrote `outputs/gflownet_got_tiny/gflownet_final.pt`.
- Dataset acquisition: small samples downloaded for GSM8K, BigCodeBench, ProofNetSharp, and MoleculeNet Lipophilicity.
- Dataset graphification: raw samples converted and merged into `data/processed/mixed_graphs/train.jsonl`.
