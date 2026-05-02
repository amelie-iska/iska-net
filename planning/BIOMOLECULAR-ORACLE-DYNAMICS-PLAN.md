# UGM Biomolecular Oracle-Dynamics Plan

This plan maps the addendum in `assets/main.tex` onto the current UGM implementation. The implementation target is narrow and testable: train the graph-token policy without supervised structure files, let embedding-space graph-of-thought state emit coordinate/contact hypotheses, and use UMA/FairChem energy and force feedback to shape those hypotheses.

## Implemented Contract

The codebase now supports a concrete BioSELFIES-style symbolic layer:

- `src/iska_reasoner/data/bioselfies.py` decodes bracketed BioSELFIES records into typed graph fragments.
- Supported tokens include amino acids, DNA/RNA bases, ordinary SELFIES atom tokens, explicit atom tokens, link tokens, modification tokens, chain breaks, branches, adaptive patch controls, hydrogen-bond controls, torsion controls, and latent-thought controls.
- The decoder is total: unsupported tokens become `bioselfies_unknown` nodes and `BIOSELFIES:UNKNOWN` target records rather than parser crashes.
- The graph fragment is symbolic only. It does not introduce coordinate, distance, force, energy, PDB, mmCIF, SDF, conformer, or trajectory supervision.
- `graphify_multimodal` accepts `bioselfies`, `bio_selfies`, or `input_representation: bioselfies`. With `bioselfies_only: true`, the row is represented through the BioSELFIES graph rather than separate FASTA/RNA/DNA/SELFIES source channels.
- The sequence-only dataset policy treats `bioselfies` as a string molecule anchor, so atom/bond records decoded from BioSELFIES are allowed when no structure/dynamics labels are present.

The current geometry path remains the existing UGM path:

- source-side `UMA_COORD_QUERY:*` slots are derived from symbolic sequence/graph records;
- the coordinate head reads out coordinates from the same hidden graph-of-thought embeddings used by random-order graph decoding;
- `loss.uma_coordinate_oracle_weight` trains generated coordinates with detached UMA force feedback;
- `uma_coordinate_dynamics_steps` rolls generated coordinates forward with detached UMA forces, making repeated reasoning iterations behave like short physical-time updates;
- optional MHTA/Flash hybrid attention can emit contact maps, and `uma_contact_geometry_loss` can align contact support and embedding geometry to UMA-scored rows.

## Training Interpretation

The strict training claim is:

> UGM policy training uses sequence, string, symbolic graph, temperature, verifier, and oracle feedback records. It does not copy PDB/AFDB/mmCIF/SDF coordinates, MD frames, conformer libraries, structure-token labels, or direct energy/force labels as supervised targets.

UMA is the physics source in this setup. The model is not discovering biomolecular statistical mechanics from sequence alone. It is amortizing an oracle-defined distribution from sequence-only or symbolic graph inputs.

## Practical Phases

1. **BioSELFIES symbolic pretraining**
   - Graphify rows with native modalities and with `bioselfies_only: true`.
   - Compare parser validity, graph fidelity, token count, and training loss.
   - Keep structure-file policy enabled.

2. **Sequence-only UMA coordinate training**
   - Use `ENABLE_UMA_COORDINATE_HEAD=1`.
   - Keep `loss.coordinate_loss_weight: 0.0`.
   - Train on rows with protein/SELFIES/DNA/RNA or BioSELFIES inputs and temperature nodes.

3. **Contact-map coupling**
   - Enable `ENABLE_TROPICAL_ATTENTION=1` and append `config/train/overrides/uma_contact_geometry_loss.yaml`.
   - Treat emitted attention maps as contact hypotheses, not labels.
   - Watch W&B metrics under `folding_contact/*`, `uma_contact/*`, `tropical_attention/*`, and `hybrid_attention/*`.

4. **Oracle-feedback GFlowNet**
   - Use the multimodal oracle GFlowNet configs after phase-1 graph-token training.
   - Reward token sets that preserve valid sequence/string chemistry, oracle records, temperature-conditioned motion records, and verifier consistency.
   - Keep independent oracle checks for reward exploitation.

5. **Geometry-feature ablation only**
   - Geometry features may be introduced as a labeled ablation, never mixed into the strict no-structure-input result.
   - Provenance must be logged as oracle-generated, physics-initialized, or database-derived.

## Configuration Hooks

Direct 250M training with the strict current path:

```bash
./scripts/train_full_selected_250m_oracle_dynamics_direct.sh
```

Full preflight plus training/validation/test/inference:

```bash
./scripts/run_full_phase1_phase2_training_250m_oracle_dynamics.sh
```

Diagnostics-only contact maps:

```bash
ENABLE_TROPICAL_ATTENTION=1 \
EXTRA_TRAIN_CONFIGS="config/model/overrides/attention_contact_maps.yaml config/train/overrides/folding_contact_diagnostics.yaml" \
./scripts/run_full_phase1_phase2_training_250m.sh
```

Backpropagating contact-map and embedding-geometry alignment:

```bash
ENABLE_TROPICAL_ATTENTION=1 \
ENABLE_UMA_COORDINATE_HEAD=1 \
EXTRA_TRAIN_CONFIGS="config/train/overrides/uma_contact_geometry_loss.yaml" \
./scripts/run_full_phase1_phase2_training_250m.sh
```

The function-level readiness checklist and manual equivalent config stack are maintained in `planning/ORACLE-DYNAMICS-TRAINING-RUNBOOK.md`.

## Required Metrics

Training must report:

- token loss and accuracy;
- UMA coordinate reward, energy per atom, force RMS, displacement, and rollout steps;
- contact entropy and contact strength when contact coupling is enabled;
- MHTA/Flash backend activity and margins when tropical attention is enabled;
- GFlowNet trajectory-balance loss, reward mean/max, unique terminal states, action entropy, and temperature-diversity bonuses.

Evaluation must separate:

- strict sequence/string/BioSELFIES-only inputs;
- enriched non-structural graph inputs;
- geometry-feature ablations;
- structure/dynamics evaluation-only datasets.

## Failure Criteria

Scale-up should stop if:

- the strict policy check fails on the current corpus;
- BioSELFIES graphification creates invalid graph endpoints;
- UMA coordinate loss cannot backpropagate through the coordinate head;
- high-temperature runs do not increase contact/coordinate diversity;
- independent oracle checks disagree sharply with UMA rewards;
- path outputs are only disconnected low-energy endpoints rather than plausible force-aligned transitions.
