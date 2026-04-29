# PLAN-C: Sequential Vertical-Slice Completion Plan

Status: implementation plan for deeper vertical slices  
Date: 2026-04-29

## 1. Goal

PLAN-B made the scaffold broader and added lightweight topology, tropical diagnostics, curation, and verifier-aware GFlowNet training. PLAN-C deepens the system into sequential vertical slices so each domain has a concrete train/validate/infer path:

1. engineering and config hardening;
2. model/training quality improvements;
3. code and unit-test tool-use slice;
4. Lean/formal-proof slice;
5. chemistry/molecule slice;
6. GFlowNet graph-of-thought slice;
7. validation, inference, and documentation.

The method is deliberately sequential. Each slice must have data schema support, graphification, training metrics, validation metrics, tests, docs, and a smoke command before moving to the next.

## 2. Non-Negotiable Constraints

- Keep all current smoke tests passing.
- Keep large/gated corpora opt-in. Do not bulk-download gated or multi-GB data by default.
- Keep tool execution conservative: Python tool execution is for local controlled snippets only and is not a production security sandbox.
- Prefer explicit fallbacks when Lean, RDKit, `datasets`, or external tools are unavailable.
- W&B metric namespaces must stay organized by stage/domain.

## 3. Slice A: Engineering and Config Hardening

Required implementation:

- root Git initialization if absent;
- config-driven inference CLI;
- memory/profile CLI for 4090 sizing;
- richer YAML configs for code, Lean, chemistry, GFlowNet, and validation;
- packaged import path still works without manually setting `PYTHONPATH`;
- docs updated with exact commands.

Metrics/logging:

- `profile/parameter_count`
- `profile/trainable_parameter_count`
- `profile/estimated_param_memory_mb`
- `profile/device`

Definition of done:

- `scripts/profile_model.py` works.
- `scripts/infer.py --config ...` works.

## 4. Slice B: Model and Training Quality

Required implementation:

- optional gradient checkpointing in local model;
- simple LoRA adapters for local linear layers;
- learning-rate scheduler;
- resume-from-checkpoint config/CLI;
- train/validation split files instead of only random split when provided;
- W&B grouped metrics with `stage/domain/metric` pattern.

Deferred:

- true QLoRA on external LLMs;
- FlashAttention kernel replacement;
- full upstream Fairseq TokenGT training.

Definition of done:

- tiny config can enable LoRA and checkpointing.
- resume path loads checkpoints.

## 5. Slice C: Code and Unit-Test Tool-Use

Required implementation:

- graphification of BigCodeBench tests, entry points, imports, and canonical solution metadata;
- Python unit-test runner for generated/canonical code with timeout;
- code validation metrics:
  - `code/pass_rate`
  - `code/test_count_mean`
  - `code/has_tests_rate`
  - `code/python_error_rate`
- code-specific config.

Definition of done:

- code validator can run on sampled BigCodeBench rows or graph examples with embedded tests.
- validation reports code metrics.

## 6. Slice D: Lean/Formal-Proof

Required implementation:

- Lean availability/version probe;
- Lean file compilation adapter when `lean` is installed;
- graphification of Lean theorem/proof fields into theorem, proof, imports, and informal statement nodes;
- Lean validation metrics:
  - `lean/available`
  - `lean/compile_attempt_rate`
  - `lean/compile_success_rate`
  - `lean/error_rate`
- Lean-specific config.

Deferred:

- LeanDojo proof-state tracing;
- mathlib environment management;
- tactic-level repair loop.

Definition of done:

- adapter gracefully reports unavailable Lean and compiles a temporary file when Lean exists.

## 7. Slice E: Chemistry/Molecule

Required implementation:

- RDKit atom/bond graphification when RDKit is available;
- fallback SMILES graphifier when unavailable;
- molecule metrics:
  - `chem/rdkit_available`
  - `chem/smiles_valid_rate`
  - `chem/atom_count_mean`
  - `chem/bond_count_mean`
- chemistry-specific config.

Deferred:

- ChEMBL/BindingDB dedicated ingest;
- assay normalization at scale;
- toxicity/synthesis filters.

Definition of done:

- molecule graphifier produces atom/bond nodes for RDKit-enabled environments.

## 8. Slice F: GFlowNet Graph-of-Thought

Required implementation:

- verifier-aware rewards already exist; extend with domain metric aggregation;
- rollout validation script for saved GFlowNet checkpoints;
- save per-step rollout samples with domain/task labels;
- action diversity metrics.

Deferred:

- learned backward policy;
- true graph edit actions;
- hidden-state-coupled GFlowNet.

Definition of done:

- GFlowNet rollouts can be validated after training.

## 9. Slice G: Validation and Inference

Required implementation:

- validation aggregates topology, tropical, verifier, code, Lean, and chemistry metrics;
- inference supports config files;
- inference supports a simple verifier-guided graph-of-thought retry loop;
- docs explain training/validation/inference per vertical slice.

Definition of done:

- `pytest` passes.
- smoke commands pass:
  - curation;
  - graph pretrain;
  - topology aux;
  - code validation;
  - Lean adapter validation;
  - chemistry validation;
  - GFlowNet training and validation;
  - config-driven inference.

## 10. Verification Log

Completed on 2026-04-29.

Commands run:

- `conda run -n tokengt pytest -q`
  - result: 12 passed, 2 PyTorch nested-tensor warnings.
- `conda run -n tokengt python scripts/profile_model.py --config config/model/tiny_lora_checkpointed.yaml --vocab-size 256`
  - result: reports parameter count, trainable count, rough parameter memory, and device.
- `conda run -n tokengt python scripts/validate_stage.py --config config/validate/domain_validation.yaml --device cpu`
  - result: reports validation loss/perplexity/token accuracy plus topology, tropical, verifier, code, Lean, and chemistry metrics.
- `conda run -n tokengt python scripts/train_stage.py --config config/model/tiny_lora_checkpointed.yaml --config config/data/code_graphs.yaml --config config/train/code_sft_tiny.yaml`
  - result: code vertical-slice smoke training completes 20 steps with validation.
- `conda run -n tokengt python scripts/train_stage.py --config config/model/tiny_lora_checkpointed.yaml --config config/data/lean_graphs.yaml --config config/train/lean_sft_tiny.yaml`
  - result: Lean vertical-slice smoke training completes 20 steps with validation.
- `conda run -n tokengt python scripts/train_stage.py --config config/model/tiny_lora_checkpointed.yaml --config config/data/chem_graphs.yaml --config config/train/chem_sft_tiny.yaml`
  - result: chemistry vertical-slice smoke training completes 20 steps with validation.
- `conda run -n tokengt python scripts/train_stage.py --config config/data/synthetic_graphs.yaml --config config/train/gflownet_got_tiny.yaml`
  - result: verifier-aware GFlowNet trajectory-balance smoke training completes 20 steps.
- `conda run -n tokengt python scripts/validate_gflownet.py --config config/train/gflownet_got_tiny.yaml --config config/validate/gflownet_validation.yaml --device cpu`
  - result: reports rollout reward, diversity, trajectory length, verifier metrics, and terminal validity.
- `conda run -n tokengt python scripts/infer.py --config config/inference/tiny_inference.yaml --text "Compute the path length from v0 to v4 in a next-edge graph." --max-steps 4 --retries 2 --device cpu`
  - result: config-driven inference returns token JSON and verifier diagnostics.

Implementation notes:

- LoRA is implemented for compatible local linear layers. PyTorch transformer fused fast paths are disabled when LoRA is active so adapter `forward()` paths are honored.
- Generic `CODE:function:*` graph labels are not treated as executable Python. Only explicit `CODE:python:*` and `CODE:raw:*` tokens are executed.
- Lean and RDKit are optional. Metrics report availability and degrade gracefully when those tools are not installed.
- Large/gated corpora remain opt-in through the manifest; local smoke data is intentionally small.
