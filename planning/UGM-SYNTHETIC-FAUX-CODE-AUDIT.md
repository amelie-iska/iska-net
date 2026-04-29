# UGM Synthetic, Proxy, And Faux-Code Audit

This audit covers first-party implementation, tests, configs, README, math notes, and planning docs. It excludes generated logs, generated model outputs, large processed corpora, and vendored external repositories under `data/external_repos/`.

## Complete First-Party Marker Inventory

The following files contain one or more of: `synthetic`, `faux`, `fake`, `stub`, `placeholder`, `proxy`, `toy`, `dummy`, `mock`, `not implemented`, `manifest_only`, `local_generated`, `fallback`, `TODO`, or `FIXME`.

| File | Classification | Required action |
|---|---|---|
| `src/iska_reasoner/tools/verifiers.py` | Former active UMA token-completeness reward proxy. | Replaced active UMA reward path with FairChem/UMA adapter; explicit `UGM_UMA_BACKEND=proxy` remains test/smoke only. |
| `src/iska_reasoner/oracles/uma.py` | Real external oracle adapter with explicit proxy smoke backend. | Keep production configs on `backend: fairchem`, `strict: true`; proxy backend is not production. |
| `src/iska_reasoner/data/multimodal.py` | Contains synthetic multimodal generator and legacy `SEQ_STRUCT_DYN_PROXY:*` token family names. | Synthetic generator remains smoke-only. Rename legacy `PROXY` token family to candidate/scored-candidate in a migration if backward compatibility with old checkpoints is not needed. |
| `src/iska_reasoner/data/synthetic.py` | Deterministic synthetic graph and tool-repair examples. | Test and bootstrap only; not a production training source. |
| `src/iska_reasoner/data/graphify.py` | Synthetic graph CLI and `audio_placeholder` node for local audio metadata. | Synthetic CLI is smoke-only. Replace `audio_placeholder` with real extracted audio-feature rows before audio training. |
| `src/iska_reasoner/data/acquire.py` | `local_generated` and `synthetic` acquisition methods. | Keep only for smoke/test manifests; production manifests must use real HF/local curated sources. |
| `src/iska_reasoner/data/catalog.py` | `manifest_only` and `local_generated` catalog statuses. | Correctly marks deferred/restricted sources; not a faux implementation by itself. |
| `src/iska_reasoner/data/phase_policy.py` | Mentions later structure/dynamics phase and proxy task naming. | Policy is real; update names if `SEQ_STRUCT_DYN_PROXY` migration is performed. |
| `src/iska_reasoner/topology/summaries.py` | Fallback persistent-homology and persistent-Laplacian approximations. | Acceptable fallback for missing `ripser`/`gudhi`; production readiness requires dependencies installed, which `check_readiness.py` verifies. |
| `src/iska_reasoner/topology/folding.py` | Sequence-only contact-field estimator formerly described as a proxy. | Use as an internal contact estimate, not as physical validation. FairChem/UMA supplies external scoring. |
| `src/iska_reasoner/topology/hidden.py` | Differentiable topology proxy losses. | Analysis/regularization only; not an external physics oracle. |
| `src/iska_reasoner/tropical/attention.py` | Tropical diagnostics/fallback language. | Diagnostic implementation, not faux external behavior. |
| `scripts/prepare_multimodal_sources.py` | `--synthetic` and `--synthetic-if-empty`. | Smoke-only flags. Production should omit these flags so missing data fails. |
| `scripts/prepare_structure_dynamics_sources.py` | `--synthetic-if-empty` and future-phase guard. | Smoke/eval only; train remains blocked unless explicitly approved. |
| `scripts/prepare_hebrew_sources.py` | Synthetic root-extension rows. | Test/ablation only unless replaced with licensed real Hebrew morphology sources. |
| `scripts/run_graph_state_ablation.py` | Ablation/scaffold command surface. | Research ablation only. |
| `scripts/audit_dataset_capacity.py` | Capacity estimates and unknown-size fallbacks. | Real audit utility; no replacement required. |
| `scripts/download_hf_selected_splits.py` | Download fallback/error handling. | Real utility; no replacement required. |
| `scripts/graphify_full_parquet_manifest.py` | Error fallback paths. | Real utility; no replacement required. |
| `config/data/synthetic_graphs.yaml` | Synthetic graph dataset config. | Smoke/tiny only. |
| `config/train/graph_pretrain_tiny.yaml` | Synthetic graph tiny training config. | Smoke/tiny only. |
| `config/data/hebrew_roots.yaml` | Synthetic Hebrew-root graph data path. | Replace with real Hebrew morphology data before any production Hebrew claims. |
| `config/inference/tiny_inference.yaml` | Tiny/smoke inference. | Smoke only. |
| `config/validate/domain_validation.yaml` | Validation harness references smoke data. | Use real validation configs for production. |
| `config/validate/gflownet_validation.yaml` | GFlowNet smoke validation. | Use oracle-specific validation configs for production. |
| `config/curriculum/ugm_sequence_first_curriculum.yaml` | Mentions UMA/proxy/curriculum staging. | Update terminology after any `PROXY` token migration. |
| `tests/*` files matching the marker scan | Test fixtures, monkeypatches, and smoke assertions. | Valid as tests; production code must not depend on test proxy defaults. |
| `README.md`, `MATH.md`, `planning/*.md`, `assets/*.tex` | Documentation of smoke paths, future phases, and earlier proxy wording. | Updated core README/MATH/architecture wording for FairChem/UMA. Some planning files still document historical proxy status; treat them as dated plans, not implementation truth. |

## Replacement Plan

1. **Oracle acquisition:** Track the Amelie Schreiber FairChem fork in `data/manifests/model_repos.yaml` and clone it to `data/external_repos/fairchem` during reference-repo acquisition.
2. **Real oracle adapter:** Provide a single first-party adapter that imports FairChem from the local clone, builds an ASE `Atoms` candidate from SMILES with RDKit conformer generation, calls `pretrained_mlip.get_predict_unit("uma-s-1p2")`, attaches `FAIRChemCalculator(task_name="omol")`, and converts energy/force outputs into a bounded reward.
3. **Strict production behavior:** Production oracle configs set `oracle.backend: fairchem` and `strict: true`. Missing clone, missing UMA model access, invalid candidate strings, or FairChem runtime failure must stop the oracle stage instead of silently using a proxy.
4. **Explicit smoke behavior:** Deterministic proxy reward remains only behind `UGM_UMA_BACKEND=proxy` for unit tests and smoke runs where downloading gated UMA weights is inappropriate.
5. **Readiness checks:** Readiness and QA must report the FairChem clone and importability. `scripts/download_uma_weights.py` resolves or downloads the gated UMA-S-1.2 checkpoint and reference YAMLs through FairChem's registry; `scripts/check_uma_oracle.py` verifies clone/import state and can run an actual strict UMA scoring call when credentials are available.
6. **Production data discipline:** `--synthetic` and `--synthetic-if-empty` remain available for tests but should be absent from production commands. Production data prep should fail when reviewed local or public data is missing.
7. **Naming cleanup:** The remaining `SEQ_STRUCT_DYN_PROXY:*` tokens are legacy names for sequence-conditioned candidate records. A later migration should rename them to `SEQ_STRUCT_DYN_CANDIDATE:*` if old checkpoints and vocab artifacts can be invalidated.

## Implemented In This Pass

- Added `src/iska_reasoner/oracles/uma.py` and `src/iska_reasoner/oracles/__init__.py`.
- Added FairChem to `data/manifests/model_repos.yaml`.
- Cloned `https://github.com/amelie-iska/fairchem.git` to `data/external_repos/fairchem`.
- Added `scripts/check_uma_oracle.py`.
- Added `scripts/download_uma_weights.py`.
- Changed `multimodal_oracle_reward` to use FairChem by default and proxy only when explicitly requested.
- Added `oracle.backend: fairchem`, `strict: true`, model, task, device, and repo settings to oracle GFlowNet train/validation configs.
- Updated `scripts/run_full_training_sequence.sh` to acquire FairChem in stage 01 and verify/download UMA weights before oracle stages.
- Updated readiness and QA checks to include FairChem.
- Updated README, MATH, and architecture notes to describe real FairChem/UMA behavior.

## Remaining Non-Production Artifacts

The repo still contains synthetic data generators and smoke configs by design. They are test harnesses, not production training paths. The only remaining active naming issue is the legacy `PROXY` token family; it no longer implies proxy reward scoring, but the name should be migrated before publication if checkpoint compatibility is not a concern.
