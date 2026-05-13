# Multimodal Bio-LM Dataset Utilization Plan

This plan maps the best currently useful multimodal bio-language-model data ideas into UGM without changing the strict structure-dynamics boundary. The immediate goal is not to import another model architecture wholesale. It is to use the papers' dataset constructions as a disciplined source map for UGM graph rows: sequence/function alignment, nucleic-acid/protein co-training, binding-site annotations, and biomolecular-complex affinity records.

## Model References Read

| Reference | Modalities | Data signal to use in UGM | Boundary |
|---|---|---|---|
| [LucaOne](https://www.nature.com/articles/s42256-025-01044-4) / [preprint activity](https://sciety.org/articles/activity/10.1101/2024.05.10.592927) | DNA/RNA/protein sequence plus taxonomy and protein annotations | Unified nucleic-acid/protein sequence rows, taxonomy, UniProt keywords, InterPro domains/sites, GO/EC/function fields, central-dogma pairs | Use sequence and annotations directly; keep LucaOne-style structure-coordinate supervision out of strict UGM policy training |
| [ProTrek](https://www.nature.com/articles/s41587-025-02836-0) / [GitHub](https://github.com/westlake-repl/ProTrek) | Protein sequence, structure strings, natural-language function | Sequence/function text pairs, UniProt text descriptions, retrieval-style positive/negative function pairs, optional external embedding metadata | Use text/function and sequence pairs now; reserve structure-string alignment for separately labeled structure-enabled ablations |
| [BioT5+](https://arxiv.org/abs/2402.17810) / [ACL PDF](https://aclanthology.org/2024.findings-acl.71.pdf) | Molecules, protein text, biomedical text, IUPAC/SELFIES | Molecule captioning/instruction patterns, IUPAC and SELFIES aliases, numerical tokenization for affinity/assay values | Use molecule/text and numeric-label formatting; do not let text captions become unsourced structure labels |
| [OneProt](https://arxiv.org/abs/2411.04863) / [PLOS Computational Biology](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1013679) | Sequence, structure, binding pockets, text | Binding-site alignment as the design reference for UniProt `BINDING`, `ACT_SITE`, `METAL`, `DNA_BIND`, pocket, and cofactor rows | Use binding-site and pocket annotations as symbolic sequence features; structure/pocket coordinates remain evaluation or opt-in structure-enabled data |

## Why These Three Plus OneProt

LucaOne is the best fit for the UGM foundation curriculum because it trains one biological language model over nucleic acids and proteins, and its public paper describes data from RefSeq, UniRef50, UniProt, ColabFoldDB, taxonomy, keywords, InterPro features, and downstream DNA/RNA/protein tasks. The UGM use is the non-coordinate part of that construction: make DNA/RNA/protein graph rows share a common vocabulary and let sequence annotations create typed graph targets.

ProTrek is the best fit for protein sequence/function retrieval and graph-of-thought supervision because it aligns sequence, structure, and natural-language function. For strict UGM training, the usable part is sequence-function and function-sequence supervision. Its structure channel is valuable as a future ablation and as an evaluation comparator, but it should not leak structure targets into the strict UMA-oracle path.

BioT5+ is the best fit for text/molecule/protein instruction and numeric assay formatting. UGM needs reliable graph records for IUPAC, SELFIES, SMILES, protein function text, assay labels, and affinity values. BioT5+'s core lesson is to represent biological text, molecular strings, and numeric fields jointly rather than treating numeric labels as unstructured prose.

OneProt is the binding-site-specific reference. It explicitly aligns sequence, structure, binding-site/pocket, and text encoders. UGM should borrow the binding-site data idea, not the coordinate supervision: binding residues, active residues, cofactor residues, metal/DNA/nucleotide binding records, and pocket tags become symbolic graph annotations connected to sequence positions.

## Implemented UGM Ingest Surfaces

The current implementation adds the local-data contracts needed for this plan:

- `scripts/prepare_science_sources.py --kind uniprot_features` normalizes UniProt TSV/CSV/JSON/JSONL exports with sequence, accession, protein names, gene names, organism/taxon, GO, keywords, subcellular location, cofactor, catalytic activity, subunit, and feature fields.
- `src/iska_reasoner/data/graphify.py` converts UniProt binding/active/metal/DNA/calcium/site/domain/PTM/variant features into `UNIPROT:*` graph tokens and binding-site edges without requiring coordinates.
- `scripts/prepare_science_sources.py --kind complex_affinity`, `biomolecular_affinity`, `ppi_affinity`, or `protein_na_affinity` normalizes biomolecular-complex affinity rows.
- `graphify_biomolecular_complex_affinity` supports protein-protein, protein-RNA, protein-DNA, ligand-protein, ligand-RNA/DNA, and arbitrary `components` rows with `Kd`, `Ki`, `IC50`, `kon`, `koff`, or `dG`-style fields plus temperature, pH, buffer, and assay metadata.
- All four bio input modalities now get SELFIES/BioSELFIES-side metadata: `protein_bioselfies`, `dna_bioselfies`, `rna_bioselfies`, and `molecule_selfies`, each capped at 8192 BioSELFIES tokens. SMILES-only molecule rows are converted to SELFIES when the optional `selfies` package is installed.
- ESM-style contact priors can be cached with `scripts/build_esm_contact_priors.py` and graphified into `CONTACT_PATCH:esm_prior` and `ESM_CONTACT:*` records.
- OMG/gLM2-style mixed metagenomic context can be subsampled with `scripts/prepare_omg_subsample.py`, preserving CDS/IGS order, strand metadata, intergenic sequence rows, and categorical-Jacobian contact records.
- Cached categorical-Jacobian contacts can be converted with `scripts/build_categorical_jacobian_contacts.py` and used as `CONTACT_PATCH:categorical_jacobian` / `JACOBIAN_CONTACT:*` priors.
- `data/manifests/datasets.yaml` now declares local manifest-only entries for `uniprot_features_local_export` and `biomolecular_complex_affinity_local` so these sources are intentionally user-provided rather than silently downloaded.

## Contact-Prediction Additions

The ESM contact-prior path follows the public ESM contact-prediction notebook pattern: tokenize the protein sequence with the ESM alphabet, run an ESM2 model, call `predict_contacts`, and cache a residue-pair probability matrix. UGM then treats high-probability pairs as predicted contact priors, not as ground-truth structure labels. These records are suitable for both training graphification and inference-time structure-dynamics prompts.

The OMG/gLM2 path follows the paper's core data idea: mixed contigs contain amino-acid CDS segments and nucleotide intergenic segments, with order and strand orientation preserved. The UGM subsampler keeps a diverse subset of contigs with intergenic sequence present, so the model sees regulatory-context rows instead of only protein-only rows. Categorical-Jacobian contacts are represented as candidate contact records over CDS/residue spans. When the contact comes from protein-protein or complex context and an affinity value exists, `graphify_biomolecular_complex_affinity` also emits affinity-weighted PPI contact priors (`AFFINITY_CONTACT:*`, `PPI_CONTACT:*`, and `CONTACT_PATCH:affinity_weighted_interface`).

These additions are still sequence-only/oracle-supervised in the strict addendum sense. They do not copy PDB/mmCIF/SDF coordinates, contact-map labels, force labels, or MD trajectories into policy training.

## Data Source Priorities

1. **UniProtKB reviewed feature exports.**
   - Recommended fields: `Entry`, `Reviewed`, `Protein names`, `Gene Names`, `Organism`, `Organism ID`, `Sequence`, `EC number`, `Gene Ontology IDs`, `Keywords`, `Features`, `Binding site`, `Active site`, `Metal binding`, `DNA binding`, `Subcellular location [CC]`, `Cofactor`, `Catalytic activity`, `Subunit structure`.
   - Output row type: `function_description` plus `UNIPROT:*` target tokens.
   - Strict status: allowed, because these are symbolic sequence annotations and residue-index features, not coordinate labels.

2. **BindingDB protein-ligand affinity.**
   - Recommended fields: ligand SMILES/SELFIES, target sequence or UniProt accession, target name, `Kd/Ki/IC50/EC50`, units, assay type, publication metadata.
   - Output row type: `biomolecular_complex_affinity` when a biomolecular target and ligand are both present.
   - Strict status: allowed as non-structure scalar affinity supervision; use license/terms review before large ingestion.

3. **SKEMPI 2.0 and PPB-Affinity-style protein-protein data.**
   - Recommended fields: receptor/ligand chain sequences or resolved accession mappings, mutation string, wild-type and mutant `KD`, `kon`, `koff`, `dG/ddG`, temperature, assay/method, source paper.
   - Output row type: `ppi_affinity` or `biomolecular_affinity`.
   - Strict status: scalar affinity is allowed; complex coordinates/PDB chains must stay out of strict policy training unless explicitly using a structure-enabled ablation.

4. **PDBbind/BioLiP-like protein-ligand tables.**
   - Recommended fields: ligand SMILES/SELFIES, protein sequence or accession, affinity, units, ligand name, binding-site residue annotations if available.
   - Output row type: `complex_affinity` plus optional `UNIPROT:*` or `BINDING_SITE:*` symbolic sites.
   - Strict status: affinity and symbolic site labels can be used; coordinates are evaluation/future-phase only.

5. **ProTrek/BioT5+/LucaOne-derived sequence/function rows.**
   - Recommended fields: `protein_sequence`, `dna_sequence` or `rna_sequence` when available, `function_description`, taxonomy, keywords, GO/EC, IUPAC, SELFIES, and text prompt/completion.
   - Output row type: `function_description` or multimodal graph-to-graph.
   - Strict status: allowed if no structure labels or generated structure-token targets are included.

## Training Use

UGM now has two separate GFlowNet tracks:

- **SFT GFlowNet (`gflownet.mode: sft`)** learns diverse symbolic graph completions over ordinary SFT target tokens. Use this for function descriptions, UniProt annotations, molecule captions, assay metadata, tool/proof/text graph rows, and sequence/function alignment.
- **Structure-dynamics GFlowNet (`gflownet.mode: structure_dynamics`)** filters candidates to oracle and dynamics records such as `ALL_ATOM_CARTESIAN:*`, `ALL_ATOM_CONTACT:*`, `ALL_ATOM_BOND:*`, `CARTESIAN_ATOM:*`, `CARTESIAN_FRAME:*`, `INTERNAL_COORD:*`, `ADAPTIVE_PATCH:*`, `CONTACT_PATCH:*`, `TOKEN_MOTION:*`, temperature tokens, and UMA/force records. Use this after the model has learned the symbolic substrate and the UMA/FairChem preflight passes. If an older curated corpus lacks the explicit dynamics target family, the trainer derives these candidates from protein/DNA/RNA/SELFIES nodes so the failed GFlowNet phase can restart without repeating curation.

The standard SFT phase can train on UniProt feature rows and affinity rows directly. The structure-dynamics phase then uses those annotations as context: binding residues and complex components become places where BioSELFIES-conditioned all-atom Cartesian coordinate proposals, internal-coordinate actions, contact patches, adaptive atom patches, and UMA-scored geometry proposals should concentrate.

The coordinate head is enabled in model-training phases, not in the standalone GFlowNet trainer. The normal 4090 path uses `config/train/overrides/uma_coordinate_head.yaml` plus `config/train/overrides/uma_internal_coordinates.yaml`. Full-size all-atom Cartesian experiments use `config/train/overrides/uma_all_atom_cartesian_head_8192.yaml`, which exposes up to 8192 all-atom coordinate-query slots while scoring a tractable FairChem/UMA subset per feedback call. If the 250M profile is too large at 8192 context, switch only the model config to `config/model/ugm_120m_tokengt_8192_selfies.yaml`.

Contact maps and attention maps are not a separate residue-only data structure. They are TokenGT source-token maps. The all-atom contact-template update makes structure-dynamics rows source-tokenize an unfolded atom/bond template derived from protein/DNA/RNA/SELFIES strings, so attention can include all-atom template nodes and covalent bond edge tokens under the active source-token budget. Affinity and PPI rows use the same template path, so `AFFINITY_CONTACT:*` and `PPI_CONTACT:*` priors can condition bond-aware atom/contact reasoning.

## Current Full Local Corpus Status

The current local biomed source set is fully materialized as local TSV plus graph JSONL:

- `data/local/uniprot_features.tsv`: 574,627 UniProtKB reviewed feature data rows, plus one header row.
- `data/local/complex_affinity.tsv`: 1,836,729 biomolecular complex-affinity data rows, plus one header row.
- `data/processed/uniprot_features_local_export/all.jsonl`: graphified UniProt feature rows, about 36 GB.
- `data/processed/biomolecular_complex_affinity_local/all.jsonl`: graphified complex-affinity rows, about 66 GB.

Together this is 2,411,356 trainable data rows before curation and any exact duplicate removal. The full wrapper should normally be resumed from the graph JSONL files with `FAST_CURATE=1` and `RESUME_CURATE=1` once those files exist. That path performs exact raw-row deduplication, entity splitting, direct JSONL line copying, and resumable temp split/state writes, which avoids the previous failure mode where curation attempted to hold the entire 100 GB-scale graph corpus in memory and avoids losing all curation progress after an interruption.

The full public selected corpus remains available at `data/processed/real_full_selected_mix/` with 7,328,008 rows across its original train, validation, and test files. When `INCLUDE_ORIGINAL_FULL_SELECTED=1`, the direct wrapper adds all three original split files to the UniProt feature and biomolecular-affinity graph files, then recures the combined corpus into `data/processed/biomed_annotations_affinity_plus_original_full_selected/`. That combined input is 9,739,364 rows before exact duplicate removal and is the correct mode when the goal is "the original dataset also, with all of the data."

Expected runtime on the current 24 GB RTX 4090 workstation:

- fast curation from completed graph JSONL: about one hour at roughly 600 rows/sec after warmup;
- fast curation with the original full selected corpus included: several hours, because it reads about 187 GB of JSONL before writing the new curated splits;
- split integrity scan: minutes, dominated by reading the curated JSONL;
- 250M SFT full epoch: about 60k optimizer steps for biomed-only or about 240k optimizer steps for combined-original mode with effective batch 36, usually multi-day on one RTX 4090 depending on step time;
- SFT GFlowNet and structure-dynamics GFlowNet: 3k steps each by default, shorter than the SFT phase.

## Commands

Prepare UniProt feature rows:

```bash
conda run -n tokengt python scripts/prepare_science_sources.py \
  --kind uniprot_features \
  --input /path/to/uniprot_features.tsv \
  --dataset-name uniprot_features_local_export \
  --output data/processed/uniprot_features_local_export/all.jsonl \
  --limit 100000
```

Prepare biomolecular complex affinity rows:

```bash
conda run -n tokengt python scripts/prepare_science_sources.py \
  --kind biomolecular_affinity \
  --input /path/to/complex_affinity.tsv \
  --dataset-name biomolecular_complex_affinity_local \
  --output data/processed/biomolecular_complex_affinity_local/all.jsonl \
  --limit 100000
```

Train the symbolic SFT GFlowNet:

```bash
conda run -n tokengt python scripts/train_stage.py \
  --config config/data/multimodal_graphs_4090.yaml \
  --config config/train/gflownet_sft_4090.yaml
```

Train the dedicated UniProt plus biomolecular-affinity stack directly:

```bash
# Replace these with real files on this machine; do not use /path/to literally.
UNIPROT_FEATURES_INPUTS="$PWD/data/local/uniprot_features.tsv" \
AFFINITY_INPUTS="$PWD/data/local/complex_affinity.tsv" \
TRAIN_PHASES=all \
./scripts/train_biomed_annotations_affinity_direct.sh
```

This command prepares local graph JSONL if needed, curates `data/processed/biomed_annotations_affinity`, checks split integrity, trains the 250M SFT model config, then runs the SFT GFlowNet and structure-dynamics GFlowNet configs when `TRAIN_PHASES=all`.

Resume from completed graph JSONL without repeating TSV preparation or graphification:

```bash
PREPARE_FULL_BIOMED_SOURCES=0 \
PREPARE_UNIPROT=0 \
PREPARE_AFFINITY=0 \
CURATE_DATA=force \
FAST_CURATE=1 \
RESUME_CURATE=1 \
TRAIN_PHASES=all \
./scripts/run_full_biomed_annotations_affinity_training.sh
```

Resume from completed graph JSONL and include the original full selected public corpus in the same train/validation/test curation:

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

Train the structure-dynamics GFlowNet:

```bash
conda run -n tokengt python scripts/train_stage.py \
  --config config/data/multimodal_graphs_4090.yaml \
  --config config/train/structure_dynamics_gflownet_4090.yaml
```

Train the strict direct oracle-dynamics path:

```bash
./scripts/train_full_selected_250m_oracle_dynamics_direct.sh
```

## Validation Rules

- UniProt binding-site rows should create sequence nodes, `uniprot_feature` nodes, and `marks_binding_site` edges.
- Complex affinity rows should create at least two `biomolecule_component` or ligand/protein/nucleic-acid component nodes, a `biomolecular_complex` node, a `binding_affinity` node, and component-to-complex edges.
- SFT GFlowNet candidate vocab should include broad symbolic target tokens.
- Structure-dynamics GFlowNet candidate vocab should include BioSELFIES/all-atom Cartesian/internal-coordinate/contact/adaptive-patch/oracle tokens and exclude ordinary prose-only target tokens.
- Strict oracle-dynamics training should keep `coordinate_loss_weight: 0.0`; UMA coordinate and internal-coordinate losses must be oracle-supervised, not coordinate-label supervised.

## Next Extensions

- Add optional UniProt REST/FTP downloader with explicit query, field list, license log, and rate-limit handling.
- Add normalized affinity converters for `Kd/Ki/IC50/kon/koff/dG/ddG` with unit-aware scalar bins and continuous values.
- Add an entity-aware merger that links UniProt accessions in BindingDB/SKEMPI/PPB-Affinity rows to local UniProt feature rows.
- Add W&B panels for binding-site coverage, affinity measure distribution, source mix, and GFlowNet action coverage over binding-site-centered patches.
