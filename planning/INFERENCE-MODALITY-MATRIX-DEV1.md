# Dev-1 Trained-Model Inference Modality Matrix

Generated: `2026-05-12T22:01:46.734074+00:00`
Checkpoint: `outputs/biomed_annotations_affinity_plus_original_250m/checkpoint_final.pt`
Vocab: `outputs/biomed_annotations_affinity_plus_original_250m/vocab.jsonl`
Output directory: `outputs/inference/dev1_modality_matrix/20260512T220111Z`
Device: `cuda`
Max steps: `16`

## Summary

- Total modality-pair cases: `60`
- Completed inference cases: `59`
- Skipped/not applicable cases: `1`
- Failed cases: `0`
- Structure artifact-complete cases: `9`

A skipped structure-dynamics case means the input was text-only or raw graph-only and therefore had no structured sequence/molecule/BioSELFIES fields from which the coordinate head could derive `UMA_COORD_QUERY:*` atom slots.

## Case Matrix

| Input | Output | Status | Quality | Reward | Tokens | Atoms | Frames | HQ score | Artifacts | Necessary update |
|---|---|---:|---|---:|---:|---:|---:|---:|---|---|
| text | graph_tokens | ok | usable_smoke | 0.625 | 16 |  |  |  |  | No blocking update for smoke inference; inspect generated token families for task specificity. |
| text | text/function | ok | usable_smoke | 0.900 | 16 |  |  |  |  | No blocking update for smoke inference; inspect generated token families for task specificity. |
| text | molecule | ok | usable_smoke | 0.625 | 16 |  |  |  |  | No blocking update for smoke inference; inspect generated token families for task specificity. |
| text | sequence_annotation | ok | usable_smoke | 0.625 | 16 |  |  |  |  | No blocking update for smoke inference; inspect generated token families for task specificity. |
| text | affinity/bioactivity | ok | usable_smoke | 0.625 | 16 |  |  |  |  | No blocking update for smoke inference; inspect generated token families for task specificity. |
| text | structure_dynamics | skipped | not_applicable | 0.625 | 16 |  |  |  |  | Add structured sequence/molecule extraction for this input before requesting structure-dynamics export. |
| graph_json | graph_tokens | ok | usable_smoke | 0.625 | 16 |  |  |  |  | No blocking update for smoke inference; inspect generated token families for task specificity. |
| graph_json | text/function | ok | usable_smoke | 0.625 | 16 |  |  |  |  | No blocking update for smoke inference; inspect generated token families for task specificity. |
| graph_json | molecule | ok | usable_smoke | 0.625 | 16 |  |  |  |  | No blocking update for smoke inference; inspect generated token families for task specificity. |
| graph_json | sequence_annotation | ok | usable_smoke | 0.625 | 16 |  |  |  |  | No blocking update for smoke inference; inspect generated token families for task specificity. |
| graph_json | affinity/bioactivity | ok | usable_smoke | 0.625 | 16 |  |  |  |  | No blocking update for smoke inference; inspect generated token families for task specificity. |
| graph_json | structure_dynamics | ok | artifact_complete | 0.625 | 16 | 57 | 64 | 0.915632 | `pdb`, `dcd`, `xyz` | No blocking update for artifact generation; evaluate physical plausibility with strict FairChem/UMA or OpenMM rollout before scientific use. |
| protein | graph_tokens | ok | weak | 0.000 | 16 |  |  |  |  | Needs repair or stronger modality-specific decoding constraints before relying on this output. |
| protein | text/function | ok | weak | 0.000 | 16 |  |  |  |  | Needs repair or stronger modality-specific decoding constraints before relying on this output. |
| protein | molecule | ok | weak | 0.000 | 16 |  |  |  |  | Needs repair or stronger modality-specific decoding constraints before relying on this output. |
| protein | sequence_annotation | ok | weak | 0.000 | 16 |  |  |  |  | Needs repair or stronger modality-specific decoding constraints before relying on this output. |
| protein | affinity/bioactivity | ok | weak | 0.000 | 16 |  |  |  |  | Needs repair or stronger modality-specific decoding constraints before relying on this output. |
| protein | structure_dynamics | ok | artifact_complete | 0.000 | 16 | 57 | 64 | 0.915632 | `pdb`, `dcd`, `xyz` | No blocking update for artifact generation; evaluate physical plausibility with strict FairChem/UMA or OpenMM rollout before scientific use. |
| protein/uniprot_500 | graph_tokens | ok | weak | 0.000 | 16 |  |  |  |  | Needs repair or stronger modality-specific decoding constraints before relying on this output. |
| protein/uniprot_500 | text/function | ok | weak | 0.000 | 16 |  |  |  |  | Needs repair or stronger modality-specific decoding constraints before relying on this output. |
| protein/uniprot_500 | molecule | ok | weak | 0.000 | 16 |  |  |  |  | Needs repair or stronger modality-specific decoding constraints before relying on this output. |
| protein/uniprot_500 | sequence_annotation | ok | weak | 0.000 | 16 |  |  |  |  | Needs repair or stronger modality-specific decoding constraints before relying on this output. |
| protein/uniprot_500 | affinity/bioactivity | ok | weak | 0.000 | 16 |  |  |  |  | Needs repair or stronger modality-specific decoding constraints before relying on this output. |
| protein/uniprot_500 | structure_dynamics | ok | artifact_complete | 0.000 | 16 | 4225 | 64 | 0.959166 | `pdb`, `dcd`, `xyz` | No blocking update for artifact generation; evaluate physical plausibility with strict FairChem/UMA or OpenMM rollout before scientific use. |
| molecule | graph_tokens | ok | weak | 0.000 | 16 |  |  |  |  | Needs repair or stronger modality-specific decoding constraints before relying on this output. |
| molecule | text/function | ok | weak | 0.000 | 16 |  |  |  |  | Needs repair or stronger modality-specific decoding constraints before relying on this output. |
| molecule | molecule | ok | weak | 0.000 | 16 |  |  |  |  | Needs repair or stronger modality-specific decoding constraints before relying on this output. |
| molecule | sequence_annotation | ok | weak | 0.000 | 16 |  |  |  |  | Needs repair or stronger modality-specific decoding constraints before relying on this output. |
| molecule | affinity/bioactivity | ok | weak | 0.000 | 16 |  |  |  |  | Needs repair or stronger modality-specific decoding constraints before relying on this output. |
| molecule | structure_dynamics | ok | artifact_complete | 0.000 | 16 | 3 | 64 | 0.954142 | `pdb`, `dcd`, `xyz` | No blocking update for artifact generation; evaluate physical plausibility with strict FairChem/UMA or OpenMM rollout before scientific use. |
| dna | graph_tokens | ok | weak | 0.000 | 16 |  |  |  |  | Needs repair or stronger modality-specific decoding constraints before relying on this output. |
| dna | text/function | ok | weak | 0.000 | 16 |  |  |  |  | Needs repair or stronger modality-specific decoding constraints before relying on this output. |
| dna | molecule | ok | weak | 0.000 | 16 |  |  |  |  | Needs repair or stronger modality-specific decoding constraints before relying on this output. |
| dna | sequence_annotation | ok | weak | 0.000 | 16 |  |  |  |  | Needs repair or stronger modality-specific decoding constraints before relying on this output. |
| dna | affinity/bioactivity | ok | weak | 0.000 | 16 |  |  |  |  | Needs repair or stronger modality-specific decoding constraints before relying on this output. |
| dna | structure_dynamics | ok | artifact_complete | 0.000 | 16 | 287 | 64 | 0.592369 | `pdb`, `dcd`, `xyz` | No blocking update for artifact generation; evaluate physical plausibility with strict FairChem/UMA or OpenMM rollout before scientific use. |
| rna | graph_tokens | ok | weak | 0.000 | 16 |  |  |  |  | Needs repair or stronger modality-specific decoding constraints before relying on this output. |
| rna | text/function | ok | weak | 0.000 | 16 |  |  |  |  | Needs repair or stronger modality-specific decoding constraints before relying on this output. |
| rna | molecule | ok | weak | 0.000 | 16 |  |  |  |  | Needs repair or stronger modality-specific decoding constraints before relying on this output. |
| rna | sequence_annotation | ok | weak | 0.000 | 16 |  |  |  |  | Needs repair or stronger modality-specific decoding constraints before relying on this output. |
| rna | affinity/bioactivity | ok | weak | 0.000 | 16 |  |  |  |  | Needs repair or stronger modality-specific decoding constraints before relying on this output. |
| rna | structure_dynamics | ok | artifact_complete | 0.000 | 16 | 298 | 64 | 0.591432 | `pdb`, `dcd`, `xyz` | No blocking update for artifact generation; evaluate physical plausibility with strict FairChem/UMA or OpenMM rollout before scientific use. |
| protein+molecule | graph_tokens | ok | weak | 0.000 | 16 |  |  |  |  | Needs repair or stronger modality-specific decoding constraints before relying on this output. |
| protein+molecule | text/function | ok | weak | 0.000 | 16 |  |  |  |  | Needs repair or stronger modality-specific decoding constraints before relying on this output. |
| protein+molecule | molecule | ok | weak | 0.000 | 16 |  |  |  |  | Needs repair or stronger modality-specific decoding constraints before relying on this output. |
| protein+molecule | sequence_annotation | ok | weak | 0.000 | 16 |  |  |  |  | Needs repair or stronger modality-specific decoding constraints before relying on this output. |
| protein+molecule | affinity/bioactivity | ok | weak | 0.000 | 16 |  |  |  |  | Needs repair or stronger modality-specific decoding constraints before relying on this output. |
| protein+molecule | structure_dynamics | ok | artifact_complete | 0.000 | 16 | 60 | 64 | 0.914506 | `pdb`, `dcd`, `xyz` | No blocking update for artifact generation; evaluate physical plausibility with strict FairChem/UMA or OpenMM rollout before scientific use. |
| protein+dna+rna | graph_tokens | ok | weak | 0.000 | 16 |  |  |  |  | Needs repair or stronger modality-specific decoding constraints before relying on this output. |
| protein+dna+rna | text/function | ok | weak | 0.000 | 16 |  |  |  |  | Needs repair or stronger modality-specific decoding constraints before relying on this output. |
| protein+dna+rna | molecule | ok | weak | 0.000 | 16 |  |  |  |  | Needs repair or stronger modality-specific decoding constraints before relying on this output. |
| protein+dna+rna | sequence_annotation | ok | weak | 0.000 | 16 |  |  |  |  | Needs repair or stronger modality-specific decoding constraints before relying on this output. |
| protein+dna+rna | affinity/bioactivity | ok | weak | 0.000 | 16 |  |  |  |  | Needs repair or stronger modality-specific decoding constraints before relying on this output. |
| protein+dna+rna | structure_dynamics | ok | artifact_complete | 0.000 | 16 | 391 | 64 | 0.651691 | `pdb`, `dcd`, `xyz` | No blocking update for artifact generation; evaluate physical plausibility with strict FairChem/UMA or OpenMM rollout before scientific use. |
| bioselfies+mixed | graph_tokens | ok | weak | 0.000 | 16 |  |  |  |  | Needs repair or stronger modality-specific decoding constraints before relying on this output. |
| bioselfies+mixed | text/function | ok | weak | 0.000 | 16 |  |  |  |  | Needs repair or stronger modality-specific decoding constraints before relying on this output. |
| bioselfies+mixed | molecule | ok | weak | 0.000 | 16 |  |  |  |  | Needs repair or stronger modality-specific decoding constraints before relying on this output. |
| bioselfies+mixed | sequence_annotation | ok | weak | 0.000 | 16 |  |  |  |  | Needs repair or stronger modality-specific decoding constraints before relying on this output. |
| bioselfies+mixed | affinity/bioactivity | ok | weak | 0.000 | 16 |  |  |  |  | Needs repair or stronger modality-specific decoding constraints before relying on this output. |
| bioselfies+mixed | structure_dynamics | ok | artifact_complete | 0.000 | 16 | 188 | 64 | 0.596449 | `pdb`, `dcd`, `xyz` | No blocking update for artifact generation; evaluate physical plausibility with strict FairChem/UMA or OpenMM rollout before scientific use. |

## Output Artifacts

### `graph_json__to__structure_dynamics`

- `pdb`: `outputs/inference/dev1_modality_matrix/20260512T220111Z/cases/graph_json__to__structure_dynamics/structure_dynamics.pdb`
- `dcd`: `outputs/inference/dev1_modality_matrix/20260512T220111Z/cases/graph_json__to__structure_dynamics/structure_dynamics.dcd`
- `xyz`: `outputs/inference/dev1_modality_matrix/20260512T220111Z/cases/graph_json__to__structure_dynamics/structure_dynamics.xyz`
- Long high-quality simulation proxy score: `0.915632`
- Atom/frame coverage: `57` atoms, `64` frames, `6` residues/bases

### `protein__to__structure_dynamics`

- `pdb`: `outputs/inference/dev1_modality_matrix/20260512T220111Z/cases/protein__to__structure_dynamics/structure_dynamics.pdb`
- `dcd`: `outputs/inference/dev1_modality_matrix/20260512T220111Z/cases/protein__to__structure_dynamics/structure_dynamics.dcd`
- `xyz`: `outputs/inference/dev1_modality_matrix/20260512T220111Z/cases/protein__to__structure_dynamics/structure_dynamics.xyz`
- Long high-quality simulation proxy score: `0.915632`
- Atom/frame coverage: `57` atoms, `64` frames, `6` residues/bases

### `uniprot_500_protein__to__structure_dynamics`

- `pdb`: `outputs/inference/dev1_modality_matrix/20260512T220111Z/cases/uniprot_500_protein__to__structure_dynamics/structure_dynamics.pdb`
- `dcd`: `outputs/inference/dev1_modality_matrix/20260512T220111Z/cases/uniprot_500_protein__to__structure_dynamics/structure_dynamics.dcd`
- `xyz`: `outputs/inference/dev1_modality_matrix/20260512T220111Z/cases/uniprot_500_protein__to__structure_dynamics/structure_dynamics.xyz`
- Long high-quality simulation proxy score: `0.959166`
- Atom/frame coverage: `4225` atoms, `64` frames, `500` residues/bases

### `molecule__to__structure_dynamics`

- `pdb`: `outputs/inference/dev1_modality_matrix/20260512T220111Z/cases/molecule__to__structure_dynamics/structure_dynamics.pdb`
- `dcd`: `outputs/inference/dev1_modality_matrix/20260512T220111Z/cases/molecule__to__structure_dynamics/structure_dynamics.dcd`
- `xyz`: `outputs/inference/dev1_modality_matrix/20260512T220111Z/cases/molecule__to__structure_dynamics/structure_dynamics.xyz`
- Long high-quality simulation proxy score: `0.954142`
- Atom/frame coverage: `3` atoms, `64` frames, `0` residues/bases

### `dna__to__structure_dynamics`

- `pdb`: `outputs/inference/dev1_modality_matrix/20260512T220111Z/cases/dna__to__structure_dynamics/structure_dynamics.pdb`
- `dcd`: `outputs/inference/dev1_modality_matrix/20260512T220111Z/cases/dna__to__structure_dynamics/structure_dynamics.dcd`
- `xyz`: `outputs/inference/dev1_modality_matrix/20260512T220111Z/cases/dna__to__structure_dynamics/structure_dynamics.xyz`
- Long high-quality simulation proxy score: `0.592369`
- Atom/frame coverage: `287` atoms, `64` frames, `14` residues/bases

### `rna__to__structure_dynamics`

- `pdb`: `outputs/inference/dev1_modality_matrix/20260512T220111Z/cases/rna__to__structure_dynamics/structure_dynamics.pdb`
- `dcd`: `outputs/inference/dev1_modality_matrix/20260512T220111Z/cases/rna__to__structure_dynamics/structure_dynamics.dcd`
- `xyz`: `outputs/inference/dev1_modality_matrix/20260512T220111Z/cases/rna__to__structure_dynamics/structure_dynamics.xyz`
- Long high-quality simulation proxy score: `0.591432`
- Atom/frame coverage: `298` atoms, `64` frames, `14` residues/bases

### `protein_molecule__to__structure_dynamics`

- `pdb`: `outputs/inference/dev1_modality_matrix/20260512T220111Z/cases/protein_molecule__to__structure_dynamics/structure_dynamics.pdb`
- `dcd`: `outputs/inference/dev1_modality_matrix/20260512T220111Z/cases/protein_molecule__to__structure_dynamics/structure_dynamics.dcd`
- `xyz`: `outputs/inference/dev1_modality_matrix/20260512T220111Z/cases/protein_molecule__to__structure_dynamics/structure_dynamics.xyz`
- Long high-quality simulation proxy score: `0.914506`
- Atom/frame coverage: `60` atoms, `64` frames, `6` residues/bases

### `protein_dna_rna__to__structure_dynamics`

- `pdb`: `outputs/inference/dev1_modality_matrix/20260512T220111Z/cases/protein_dna_rna__to__structure_dynamics/structure_dynamics.pdb`
- `dcd`: `outputs/inference/dev1_modality_matrix/20260512T220111Z/cases/protein_dna_rna__to__structure_dynamics/structure_dynamics.dcd`
- `xyz`: `outputs/inference/dev1_modality_matrix/20260512T220111Z/cases/protein_dna_rna__to__structure_dynamics/structure_dynamics.xyz`
- Long high-quality simulation proxy score: `0.651691`
- Atom/frame coverage: `391` atoms, `64` frames, `22` residues/bases

### `bioselfies_mixed__to__structure_dynamics`

- `pdb`: `outputs/inference/dev1_modality_matrix/20260512T220111Z/cases/bioselfies_mixed__to__structure_dynamics/structure_dynamics.pdb`
- `dcd`: `outputs/inference/dev1_modality_matrix/20260512T220111Z/cases/bioselfies_mixed__to__structure_dynamics/structure_dynamics.dcd`
- `xyz`: `outputs/inference/dev1_modality_matrix/20260512T220111Z/cases/bioselfies_mixed__to__structure_dynamics/structure_dynamics.xyz`
- Long high-quality simulation proxy score: `0.596449`
- Atom/frame coverage: `188` atoms, `64` frames, `12` residues/bases

## Quality Notes

- `artifact_complete` means the structure-dynamics path produced at least a multi-model PDB and a DCD trajectory for that modality pair.
- Generated PDB files include sequence-derived `HELIX` secondary-structure records plus cartoon-intent `REMARK` records. No PyMOL, ChimeraX, or other viewer sidecar scripts are emitted.
- PDB does not have a portable standard field that forces a viewer representation. The portable representation signal is the encoded secondary structure; viewers still choose their own default drawing mode unless configured by the user.
- Structure-dynamics cases are scored with a strict long-run simulation proxy profile: frame coverage, full-size residue coverage, sampled clash rate, step RMSD smoothness, max-step stability, and radius-of-gyration stability. This is not a substitute for strict FairChem/OpenMM rescoring.
- `usable_smoke` means verifier reward passed the current broad graph verifier threshold or the verifier passed.
- `partial` means the model produced tokens but weak task-specific evidence; these cases need modality-specific verifiers and decoding constraints.
- The matrix uses the deterministic `proxy` coordinate rollout by default. Use strict `fairchem` rollout for physical plausibility checks once UMA weights and runtime budget are available.
