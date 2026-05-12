# OMG/gLM2 and ESM Contact Integration

## Sources Read

- ESM contact-prediction notebook: `https://github.com/facebookresearch/esm/blob/main/examples/contact_prediction.ipynb`
- OMG/gLM2 preprint: `https://www.biorxiv.org/content/10.1101/2024.08.14.607850v1`
- OMG/gLM2 OpenReview/PDF mirror used for local reading: `https://openreview.net/forum?id=WlRX9OfW6y`
- OMG dataset: `https://huggingface.co/datasets/tattabio/OMG`
- gLM2 650M model card: `https://huggingface.co/tattabio/gLM2_650M`

## Implemented Path

- `scripts/build_esm_contact_priors.py` implements a cache-first ESM2 contact-prior builder using the notebook pattern: ESM alphabet batching plus `model.predict_contacts`.
- `src/iska_reasoner/data/multimodal.py` graphifies contact priors into `esm_predicted_contact` nodes, residue-pair edges, `CONTACT_PATCH:esm_prior`, `ESM_CONTACT:enabled`, and binned `ESM_CONTACT:*` tokens.
- `scripts/infer.py` can compute the same ESM contact priors during structure-dynamics inference with `--esm-contact-prior`.
- `scripts/prepare_omg_subsample.py` creates a diverse mixed-modality OMG subsample with CDS amino-acid segments, intergenic nucleotide segments, strand/order metadata, and optional contact records.
- `src/iska_reasoner/data/graphify.py` graphifies OMG rows into `omg_mixed_contig`, `omg_cds`, `omg_igs`, BioSELFIES component nodes, and `CONTACT_PATCH:categorical_jacobian` / `JACOBIAN_CONTACT:*` records.
- `scripts/build_categorical_jacobian_contacts.py` converts cached categorical-Jacobian matrices/contact lists into contact records. It also has an optional approximate gLM2 hidden-state perturbation path for local experiments.
- Biomolecular complex affinity graphification now emits affinity-weighted PPI/contact priors, including `AFFINITY_CONTACT:*`, `PPI_CONTACT:affinity_weighted`, and `CONTACT_PATCH:affinity_weighted_interface`.

## Training Boundary

All four bio modalities are represented as SELFIES/BioSELFIES-side fields capped at 8192 tokens:

- protein: `protein_bioselfies`
- DNA: `dna_bioselfies`
- RNA: `rna_bioselfies`
- molecule: `molecule_selfies`

These contact-prior paths are not supervised structure labels. They are sequence/model-derived candidate priors for graph construction and UMA-scored structure-dynamics generation. Actual PDB/mmCIF/SDF coordinates, contact-map labels, forces, and MD trajectories remain excluded from the strict policy training path.

## Commands

```bash
conda run -n tokengt python scripts/build_esm_contact_priors.py \
  --input data/processed/uniprot_features_local_export/all.jsonl \
  --output data/processed/uniprot_features_local_export/all.esm_contacts.jsonl \
  --model esm2_t33_650M_UR50D \
  --device cuda \
  --top-k 256 \
  --min-probability 0.2 \
  --min-separation 6
```

```bash
conda run -n tokengt python scripts/prepare_omg_subsample.py \
  --target-rows 20000 \
  --scan-limit 500000 \
  --require-intergenic \
  --output data/processed/omg_diverse_intergenic/raw.jsonl \
  --graph-output data/processed/omg_diverse_intergenic/all.jsonl
```

```bash
ENABLE_LONG_ALL_ATOM_CARTESIAN_HEAD=1 \
MODEL_CONFIG=config/model/ugm_120m_tokengt_8192_selfies.yaml \
FULL_TRAIN_BATCH_SIZE=1 \
FULL_TRAIN_EVAL_BATCH_SIZE=1 \
FULL_TRAIN_GRAD_ACCUM=36 \
./scripts/train_full_selected_250m_oracle_dynamics_direct.sh
```

Use the 250M model config for this last command when memory allows. The 120M config is the long-context fallback for 8192-token SELFIES/all-atom coordinate runs.
