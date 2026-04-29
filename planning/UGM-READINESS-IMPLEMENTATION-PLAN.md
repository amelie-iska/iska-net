# UGM Readiness Implementation Plan

Date: 2026-04-29

This plan turns `planning/UGM-READINESS-DIFF-AUDIT.md` into concrete changes. The target state is a sequence-first UGM repo that is ready to train, validate, test, and run inference while preserving the policy boundary: no actual structure-file supervision in the first run, but generated structure-dynamics candidate graphs remain active outputs. Per the latest implementation direction, generated-token PDB rendering is optional and not part of this pass.

Implementation status: completed for the current pass. Remaining work is listed in Phase 6 as non-blocking future work.

## Phase 1: Paper Corrections

Status: completed.

Files:

- `assets/human_learning_transformer_learning_review_dataset_expanded.tex`

Required edits:

1. In the graph-to-graph convention, state that generated structure-dynamics candidates are graph records first, while PDB/mmCIF/SDF/MD datasets remain evaluation/future-phase sources.
2. In the GFlowNet objective, state that candidate \(Y_G\) may contain generated atom, bond, coordinate, frame, and optional serializer records.
3. In the training objective section, replace the proxy-only target set with a generated candidate graph target set that includes stage-gated UMA/proxy records plus generated structure-dynamics records.
4. In the sampling section, explicitly say generated coordinate/frame records can be sampled and oracle-scored.
5. In the renderer corollary, keep PDB as an optional downstream serializer rather than a required current implementation item.
6. In risks, report the correct caveat: first-run outputs are UMA-oracle-guided sequence-only predictions, not supervised structure-file-trained or experimentally validated physical simulations.
7. In the conclusion, name generated structure-dynamics candidate graphs.

Validation:

- Run a stale-phrase scan over the paper, README, MATH notes, source, configs, and non-audit planning docs. The scan should return no active contradictions about generated coordinate/frame outputs versus future-gated structure-file supervision.

## Phase 2: Planning Corrections

Status: completed.

Files:

- `planning/PLAN-H.md`
- `planning/ARCHITECTURE.md`
- `planning/DATASETS.md`
- `planning/STRUCTURE-DYNAMICS-TRAINING.md`
- `planning/BACKGROUND-RESEARCH.md`
- `planning/TRAINING-SEQUENCE.md`

Required edits:

1. Replace ambiguous "PDB output is later-phase only" wording with "structure-file-derived training is later-phase only; generated coordinate/frame candidate graphs are active, and PDB rendering is optional/deferred."
2. Record that FairChem/UMA is wired through the production oracle adapter, while deterministic proxy scoring is retained only for tests and smoke runs.
3. Record that full attention-map extraction is not implemented yet; hidden geometry, JS geometry, and binned coupling records are implemented now.
4. Record the context audit result:
   - max untruncated sequence: 776
   - recommended max sequence length: 1,552
   - generated config: `config/generated/naturelm_public_sources_context_2x.yaml`
5. Record the train/val/test readiness results from the entity split and policy audits.

Validation:

- Run a targeted `rg` scan over `planning/` for stale wording.

## Phase 3: Inference Output Boundary

Status: completed. No generated-token PDB renderer was added.

Generated-token PDB rendering is not necessary for this pass. Inference remains graph-token output plus verifier/domain metrics. Input-row PDB rendering for explicitly provided evaluation coordinates remains available behind the existing `--render-input-pdb` flag, but no new generated-token PDB renderer will be added now.

Validation:

- Keep existing multimodal renderer tests unchanged.
- Ensure no new code path requires generated PDB rendering for train, validation, test, or inference readiness.

## Phase 4: Folding-Contact Metrics in Train/Val

Status: completed.

Files:

- `src/iska_reasoner/training/stage_runner.py`
- `src/iska_reasoner/validation/evaluate.py`
- `config/train/graph_state_ablation_topology_tiny.yaml`
- `config/train/graph_state_ablation_topo_tropical_tiny.yaml`
- `config/train/graph_state_ablation_tropical_tiny.yaml`

Implementation:

1. Import `folding_contact_field` and `folding_contact_metrics`.
2. Add `hidden_topology.folding_contact_enabled`.
3. During train logging, when enabled, compute a contact field from hidden states and the attention mask.
4. During validation, when enabled, report the same metrics.
5. Log metrics through the existing `metrics.jsonl` and W&B paths. Expected metric keys include:
   - `folding_contact/mean`
   - `folding_contact/std`
   - `folding_contact/entropy`
   - `folding_contact/density_05`
   - `folding_contact/density_08`
   - `folding_contact/top_contact_mean`
   - `folding_contact/effective_contact_count`
6. Enable this flag in the topology and topology+tropical graph-state ablation configs; leave it optional elsewhere.

Validation:

- Existing folding-contact unit tests should continue to pass.
- Full pytest should pass.

## Phase 5: Verification and Readiness

Status: completed.

Commands:

```bash
PYTHONPATH=. conda run -n tokengt pytest -q tests/test_hidden_topology.py tests/test_multimodal_graphs.py
PYTHONPATH=. conda run -n tokengt pytest -q
rg -n -i 'later[- ]phase.*validation|not a coordinate.*PDB|reserved for validation-only|not be reported as structure prediction' assets/human_learning_transformer_learning_review_dataset_expanded.tex MATH.md README.md planning/PLAN-H.md planning/ARCHITECTURE.md planning/DATASETS.md planning/STRUCTURE-DYNAMICS-TRAINING.md planning/BACKGROUND-RESEARCH.md planning/TRAINING-SEQUENCE.md
```

Expected results:

- Focused tests pass.
- Full tests pass.
- Stale phrase scan returns no active contradictions, except lines that explicitly and correctly refer to actual structure-file training being future-gated.

## Phase 6: Readiness Statement

After implementation and tests:

- Training readiness: entity-split sequence corpus, stage-gated UMA candidate records, topology/tropical ablation configs, W&B/offline metrics, and no structure-file training leakage.
- Validation readiness: val/test splits exist; policy, identifier, token, context, hidden topology, JS geometry, folding-contact, verifier, and domain metrics are available.
- Inference readiness: text/multimodal inputs produce graph tokens, verifier metrics, and domain metrics; generated-token PDB rendering is intentionally not required in this pass.
- Not yet complete: direct attention-map extraction, full chemical repair, and high-fidelity coordinate decoding remain explicit future work. Live UMA scoring is implemented through the FairChem adapter but still requires local access to the gated `facebook/UMA` weights at runtime.
