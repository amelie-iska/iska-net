# Structure And Dynamics Evaluation And Future-Phase Plan

This file documents the structure/dynamics code path under the current sequence-first UGM curriculum. It is **not** an active first-run training plan.

The first molecular run trains from SELFIES/SMILES strings and protein, RNA, and DNA sequence records only. Actual PDB/mmCIF/SDF files, coordinate frames, distance maps, MD trajectories, supervised energy labels, and supervised force labels are excluded from training. UMA-style scoring is used as an external, temperature-conditioned oracle for graph-state rewards and validation, not as a direct energy/force-label dataset. Generated atom, bond, coordinate, and frame graph records remain active candidate outputs when the model samples a structure-dynamics graph state.

## Current Role

- `scripts/prepare_structure_dynamics_sources.py` exists for evaluation-only row preparation, parser smoke tests, leakage audits, and future-phase experiments.
- `config/train/structure_dynamics_4090.yaml` and `config/train/structure_dynamics_oracle_gflownet_4090.yaml` are intentionally disabled unless a later explicit structure phase is approved.
- `scripts/run_full_training_sequence.sh` skips structure/dynamics training unless `ENABLE_STRUCTURE_TRAINING=1` is set deliberately.
- PDB rendering remains an optional inference/evaluation utility for explicitly provided rows. PDB text is a renderer output, not the primitive learned target for the first run, and generated-token PDB rendering is not required in this pass.

## First-Run Data Contract

Allowed first-run molecular fields:

```text
prompt, task, protein_sequence, sequence, selfies, smiles,
dna_sequence, rna_sequence, temperature, oracle,
sequence_motifs, sequence_motifs_from_structure,
function_description, function, annotation, assay metadata
```

Disallowed first-run training fields:

```text
atoms, bonds, frames, coordinates, distances, contact_maps,
energy, forces, force_bins, pdb, mmcif, sdf, trajectory,
structure_motifs, structure_derived_sequence_motifs
```

The safe exception is a frozen vocabulary of sequence motifs derived from structure-motif catalogs, represented as `SEQ_MOTIF_FROM_STRUCTURE:*`, when the row itself contains no coordinate, contact, or structure-file supervision.

## Evaluation-Only Commands

Prepare local structure-side rows only for validation, test, contamination audits, or future-phase dry runs:

```bash
conda run -n tokengt python scripts/prepare_structure_dynamics_sources.py \
  --input-dir data/local/structure_dynamics \
  --purpose eval \
  --synthetic-if-empty \
  --output data/processed/structure_dynamics_graphs/all.jsonl

conda run -n tokengt python scripts/curate_data.py \
  --input data/processed/structure_dynamics_graphs/all.jsonl \
  --output-dir data/processed/structure_dynamics_graphs \
  --val-ratio 0.2 \
  --test-ratio 0.1
```

Run evaluation only:

```bash
conda run -n tokengt python scripts/validate_stage.py \
  --config config/validate/structure_dynamics_validation.yaml

conda run -n tokengt python scripts/validate_stage.py \
  --config config/validate/structure_dynamics_test.yaml
```

## Future-Phase Gate

Before enabling any actual structure-file training, the repo must add and pass:

1. A license-reviewed structure/dynamics source manifest.
2. Temporal, scaffold, sequence-identity, family, and structure-similarity split audits.
3. A contamination report against all first-run sequence/function corpora.
4. Explicit approval to set `ENABLE_STRUCTURE_TRAINING=1`.
5. Separate W&B projects or tags that distinguish structure-file training from the sequence-first oracle curriculum.

Until those gates are satisfied, structure/dynamics files are validation-only.
