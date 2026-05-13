# UGM Oracle-Dynamics Training Runbook

This runbook is the executable overview for the addendum path in `assets/main.tex`: BioSELFIES or native sequence/string inputs, optional hybrid FlashAttention/MHTA contact maps, UMA/FairChem coordinate-force feedback, and optional contact-map/embedding-geometry alignment. The strict training boundary remains unchanged: the policy does not train from PDB/mmCIF/SDF coordinates, conformer libraries, MD frames, structure-token labels, or direct supervised energy/force labels.

## Fast Commands

Use the full preflight wrapper when you want readiness checks, UMA-weight verification, integrity checks, policy scan, training, validation, test, inference, and phase-2 stages:

```bash
./scripts/run_full_phase1_phase2_training_250m_oracle_dynamics.sh
```

Use the direct wrapper when the corpus, 250M vocab, context config, FairChem repo, and UMA weights are already present:

```bash
./scripts/train_full_selected_250m_oracle_dynamics_direct.sh
```

The direct wrapper is the fastest route to training. It defaults to:

- `ENABLE_TROPICAL_ATTENTION=1`
- `ENABLE_UMA_COORDINATE_HEAD=1`
- `ENABLE_UMA_INTERNAL_COORDINATES=1`
- `EXTRA_TRAIN_CONFIGS+=config/train/overrides/uma_contact_geometry_loss.yaml`
- `EXTRA_TRAIN_CONFIGS+=config/train/overrides/uma_internal_coordinates.yaml`
- `SKIP_POLICY_CHECK=1`
- `FULL_TRAIN_BATCH_SIZE=1`
- `FULL_TRAIN_EVAL_BATCH_SIZE=1`
- `FULL_TRAIN_GRAD_ACCUM=36`

The micro-batch default is conservative for a 24GB RTX 4090 because it combines the hybrid Flash/MHTA backend, continuous coordinate readout, UMA-force surrogate, and contact-map geometry loss. If memory is stable, try:

```bash
FULL_TRAIN_BATCH_SIZE=2 FULL_TRAIN_EVAL_BATCH_SIZE=2 FULL_TRAIN_GRAD_ACCUM=18 \
./scripts/train_full_selected_250m_oracle_dynamics_direct.sh
```

For a 20-step smoke run:

```bash
FULL_TRAIN_MAX_STEPS=20 ./scripts/train_full_selected_250m_oracle_dynamics_direct.sh
```

For a dry run that prints the resolved override without launching training:

```bash
DRY_RUN=1 ./scripts/train_full_selected_250m_oracle_dynamics_direct.sh
```

## Manual Equivalent

The direct wrapper resolves to this config stack:

```bash
conda run --no-capture-output -n tokengt python scripts/train_stage.py \
  --config config/model/ugm_250m_tokengt.yaml \
  --config config/data/real_full_selected_mix_250m.yaml \
  --config config/generated/real_full_selected_context_compact.yaml \
  --config config/train/real_full_selected_250m_local.yaml \
  --config logs/direct_training/<RUN_ID>/direct_250m_training_override.yaml \
  --config config/model/overrides/hybrid_flash_mhta_backend.yaml \
  --config config/train/overrides/uma_contact_geometry_loss.yaml \
  --config config/train/overrides/uma_internal_coordinates.yaml \
  --config config/train/overrides/uma_coordinate_head.yaml \
  --config config/train/overrides/wandb_online.yaml
```

The order matters: model/data/train base configs first, run-local batch/epoch override next, backend and loss overrides after that, and W&B last.

## Function Readiness List

| Surface | Function or file | Training role | Status |
|---|---|---|---|
| BioSELFIES tokenization | `tokenize_bioselfies` | Bracket-token parsing with total fallback | ready |
| BioSELFIES serialization | `bioselfies_from_modalities` | Converts non-structural protein/DNA/RNA/SELFIES fields into strict symbolic input | ready |
| BioSELFIES vocab | `reference_bioselfies_tokens` | Adds BioSELFIES, hybrid patch, H-bond, and torsion records to reference vocabulary | ready |
| BioSELFIES graph | `add_bioselfies_graph` | Produces typed graph nodes/edges without structure labels | ready |
| Multimodal graphification | `graphify_multimodal` | Accepts `bioselfies`, `bio_selfies`, `BioSELFIES`, `input_representation: bioselfies`, and `bioselfies_only` | ready |
| Biomed BioSELFIES graphification | `graphify_protein_ec`, `graphify_bioactivity`, `graphify_biomolecular_complex_affinity` | Adds BioSELFIES views to UniProt, bioactivity, and complex-affinity rows without structure labels | ready |
| Generic graphification | `graphify_rows` | Routes explicit BioSELFIES rows into multimodal graphification and biomed rows into BioSELFIES-augmented graphifiers | ready |
| Sequence-only gate | `graph_structure_violations` | Allows BioSELFIES string molecule anchors while rejecting direct structure labels | ready |
| UMA query batching | `RandomOrderCollator` | Emits `UMA_COORD_QUERY:*` slots from symbolic records using sequence-derived heavy-atom protein slots and RDKit molecule atoms with explicit hydrogens where available | ready |
| All-atom Cartesian candidates | `multimodal_reference_tokens` and structure-dynamics graphification | Emits `ALL_ATOM_CARTESIAN:*`, `CARTESIAN_ATOM:*`, and `CARTESIAN_FRAME:*` output/action labels for UMA-scored generated coordinate proposals | ready |
| All-atom contact template graph | `build_all_atom_contact_template_graph`, `graphify_multimodal`, `graphify_biomolecular_complex_affinity` | Adds compact sequence/SELFIES-initialized `all_atom_template_atom` source nodes and `molecular_bond` edge tokens so TokenGT contact maps can include atom and bond tokens without expanding million-row affinity corpora into untrainable JSONL size. The default persisted template cap is 64 atoms; full-size all-atom trajectory export remains in the inference path. | ready |
| Coordinate readout | `RandomOrderTokenGT.coordinate_head` | Generates continuous coordinate proposals from hidden graph-of-thought state | ready |
| UMA force surrogate | `uma_coordinate_head_oracle_loss` | Scores generated coordinates with FairChem/UMA and backpropagates detached-force surrogate | ready |
| Internal-coordinate slots | `internal_coordinate_actions` and `RandomOrderCollator` | Emits `INTERNAL_COORD_QUERY:*` slots from symbolic sequence records | ready |
| Internal-coordinate readout | `RandomOrderTokenGT.internal_coordinate_head` | Generates torsion-like actions for protein, nucleic-acid, and ligand geometry | ready |
| Internal-coordinate UMA feedback | `uma_internal_coordinate_head_oracle_loss` | Builds generated coarse geometries from model actions and scores them with UMA forces | ready |
| Contact-map coupling | `uma_contact_alignment_loss` | Aligns emitted contact maps and embedding geometry with UMA-stage feedback records | ready |
| Structure-dynamics export | `scripts/infer.py`, `records_to_multimodel_pdb`, `write_mdtraj_trajectory` | Writes multi-model PDB with sequence-derived `HELIX`/cartoon-view remarks plus MD-style DCD/XYZ trajectory artifacts | ready |
| SFT GFlowNet | `config/train/gflownet_sft_4090.yaml` | Learns broad symbolic graph completions for function/annotation/assay rows | ready |
| Structure-dynamics GFlowNet | `config/train/structure_dynamics_gflownet_4090.yaml` | Learns oracle/contact/internal-coordinate/adaptive-patch/all-atom Cartesian graph construction and can derive candidates from legacy curated biomed rows | ready |
| UniProt features | `graphify_protein_ec` and `_add_uniprot_annotations` | Adds symbolic binding-site, active-site, cofactor, domain, GO, keyword, and PTM records | ready |
| Complex affinity | `graphify_biomolecular_complex_affinity` | Adds protein/protein, protein/nucleic-acid, protein/ligand, and arbitrary component affinity rows | ready |
| Training loop | `run_training_stage` | Combines token loss, UMA coordinate loss, contact loss, tqdm, JSONL metrics, checkpoints, W&B | ready |
| Direct wrapper | `scripts/train_full_selected_250m_oracle_dynamics_direct.sh` | Jumps straight to phase-1 training with all oracle-dynamics overrides | ready |

The contact-map tensors are source-token maps. They become all-atom and bond-aware when the source graph includes the all-atom template nodes and `molecular_bond` edge tokens. The 8192-source-token path budgets this template so ordinary sequence/BioSELFIES/context tokens are retained; full untruncated atom-plus-bond attention for very large proteins would require a larger context window or a chunked atom-patch schedule.

The structure-dynamics inference smoke writes `.pdb`, `.dcd`, `.xyz`, and `.json` outputs. The PDB is multi-frame (`MODEL`/`ENDMDL`), includes `HELIX` secondary-structure records and `REMARK 902 VIEWER REPRESENTATION: CARTOON RECOMMENDED`, and includes `CONECT` records from generated/derived bonds. The DCD writer uses MDTraj with Angstrom-to-nanometer conversion at the file boundary.
| Full wrapper | `scripts/run_full_phase1_phase2_training_250m_oracle_dynamics.sh` | Runs preflight/readiness plus full phase-1/phase-2 path | ready |

## W&B Metrics To Watch

- `train/loss`, `train/token_loss`, `train/acc`
- `uma_coordinate/*`
- `uma_internal/*`
- `uma_contact/*`
- `folding_contact/*`
- `hybrid_attention/*`
- `tropical_attention/*`
- `flash_attention/*`
- `train/samples_per_sec`, `train/tokens_per_sec`, if present in the run

## Related Data And GFlowNet Commands

Prepare UniProt feature and binding-site exports:

```bash
conda run -n tokengt python scripts/prepare_science_sources.py \
  --kind uniprot_features \
  --input /path/to/uniprot_features.tsv \
  --dataset-name uniprot_features_local_export \
  --output data/processed/uniprot_features_local_export/all.jsonl
```

Prepare biomolecular-complex affinity rows:

```bash
conda run -n tokengt python scripts/prepare_science_sources.py \
  --kind biomolecular_affinity \
  --input /path/to/complex_affinity.tsv \
  --dataset-name biomolecular_complex_affinity_local \
  --output data/processed/biomolecular_complex_affinity_local/all.jsonl
```

Curate those local exports and jump directly to the 250M training stack:

```bash
# Replace these with real files on this machine; do not use /path/to literally.
UNIPROT_FEATURES_INPUTS="$PWD/data/local/uniprot_features.tsv" \
AFFINITY_INPUTS="$PWD/data/local/complex_affinity.tsv" \
TRAIN_PHASES=all \
./scripts/train_biomed_annotations_affinity_direct.sh
```

The wrapper writes/uses `data/processed/biomed_annotations_affinity/{train,val,test}.jsonl`, checks integrity, and then dispatches:

- `config/model/ugm_250m_tokengt.yaml`
- `config/data/biomed_annotations_affinity_250m.yaml`
- `config/train/biomed_annotations_affinity_250m.yaml`
- `config/train/biomed_annotations_affinity_gflownet_sft_4090.yaml`
- `config/train/biomed_annotations_affinity_structure_dynamics_gflownet_4090.yaml`

Set `TRAIN_PHASES=sft`, `TRAIN_PHASES=gflownet_sft`, or `TRAIN_PHASES=structure_dynamics_gflownet` to run only one stage.

If the graphified files already exist and are nonempty, leave `UNIPROT_FEATURES_INPUTS` and `AFFINITY_INPUTS` unset:

```bash
TRAIN_PHASES=all ./scripts/train_biomed_annotations_affinity_direct.sh
```

To include the original full selected public corpus in the same direct run, add `INCLUDE_ORIGINAL_FULL_SELECTED=1`. This appends `data/processed/real_full_selected_mix/train.jsonl`, `val.jsonl`, and `test.jsonl` to the UniProt and biomolecular-affinity graph files, then curates the combined corpus under `data/processed/biomed_annotations_affinity_plus_original_full_selected/`:

```bash
PREPARE_FULL_BIOMED_SOURCES=0 \
PREPARE_UNIPROT=0 \
PREPARE_AFFINITY=0 \
CURATE_DATA=force \
FAST_CURATE=1 \
RESUME_CURATE=1 \
INCLUDE_ORIGINAL_FULL_SELECTED=1 \
TRAIN_PHASES=all \
./scripts/run_full_biomed_annotations_affinity_training.sh
```

Train the SFT GFlowNet and the structure-dynamics GFlowNet separately:

```bash
conda run -n tokengt python scripts/train_stage.py \
  --config config/data/multimodal_graphs_4090.yaml \
  --config config/train/gflownet_sft_4090.yaml

conda run -n tokengt python scripts/train_stage.py \
  --config config/data/multimodal_graphs_4090.yaml \
  --config config/train/structure_dynamics_gflownet_4090.yaml
```

## Preconditions

Before the direct wrapper, this should already be true:

```bash
test -s outputs/real_full_selected_250m_local/vocab.jsonl
test -s data/processed/real_full_selected_mix/train.jsonl
test -d data/external_repos/fairchem
conda run --no-capture-output -n tokengt python scripts/download_uma_weights.py \
  --repo data/external_repos/fairchem \
  --model-name uma-s-1p2 \
  --task-name omol \
  --device cuda
```

If those checks have not been run recently, use the full preflight wrapper instead of the direct wrapper.
