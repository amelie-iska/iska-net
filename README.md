# Iska Universal Graph Model Scaffold

<img src="./assets/UGM-1.png" alt="Universal Graph Model" width="900">

[![Paper PDF](https://img.shields.io/badge/Paper-link%20PDF-1f6feb?style=for-the-badge&logo=arxiv&logoColor=white)](./assets/human_learning_transformer_learning_review_dataset_expanded.pdf)
[![Dynamics Addendum](https://img.shields.io/badge/Addendum-oracle%20dynamics-1f6feb?style=for-the-badge&logo=arxiv&logoColor=white)](./assets/main.pdf)

**TLDR:** *The comman catch phrase "all protein structure information is encoded in the sequence" is, in essence, and in pragmatic terms, not true; in particular, one cannot leave out consideration of forces acting on biomolecules, the all-atom structure and their energies, and, of course, any interactions that may be present; we reasoned a "universal graph model" trained on many modalities including sequence based atomic inputs only, along with forces and energies for all-atom structure dynamics would be sufficient for a generative dynamics model with additional conditioning on various language modalities, including graph structured language such as Hebrew, would yield a model that could be trained to reason on all modailites, with an embedding space GoT-reasoning GFlowNets style feedback training paradigm, on top of the pretraining corpus, leading to the present model*

This repo is a training scaffold for the **Universal Graph Model (UGM)**: a clopen-style model that is open to graph-structured information across modalities while seeking closure under a universal architecture. The central goal is that the broadly optimized transformer substrate can become a standardized graph reasoner through a simple tokenization and positional/identifier encoding of both vertices and edges, without requiring specialized graph attention for every domain. That same graph-token interface also makes the ideas in [Tropical Quivers of Archs](https://github.com/amelie-iska/Tropical_Quivers_of_Archs) and the NeurIPS 2025 [Tropical Attention](https://github.com/amelie-iska/Tropical-Attention) implementation natural to standardize: complex directed model graphs can be composed at the continuous embedding-space level rather than only through discrete token streams. UGM is therefore a TokenGT-style graph-to-graph model type for language, reasoning, tools, SELFIES/SMILES molecules, proteins, DNA/RNA, graph-state reasoning, temperature-conditioned UMA-oracle feedback, GFlowNet training, graph/tree/chain-of-thought supervision, continuous latent reasoning, optional Tropical Attention, and persistent-topology/tropical-geometry diagnostics.

**Why include MHTA (multi-head tropical attention)?**
The [Tropical Attention](https://arxiv.org/abs/2505.17190) paper gives the exact reason MHTA belongs in UGM as an optional graph-reasoning backend: it proves that MHTA stacks universally approximate tropical circuits and realize tropical transitive closure by composition. In the paper's words, MHTA is a universal approximator of max-plus dynamic programming for combinatorial optimization with closure; the relevant proof references are Theorem C.3, Corollary C.3.1, and Theorem 3.2.

That matters because many graph reasoning tasks are max-plus or min-plus computations in disguise: shortest paths, reachability, matching, dependency closure, Viterbi-style proof/state selection, verifier-guided branch choice, and graph-of-thought expansion. The paper also reports stronger out-of-distribution generalization across length and value shifts, with robustness to perturbative noise, compared with softmax and recurrent baselines on combinatorial reasoning tasks. UGM therefore exposes MHTA as a pure backend with `model.attention_backend: tropical` and as the default enabled training path with `model.attention_backend: hybrid_flash_tropical`, which runs a FlashAttention-2/SDPA softmax branch in every encoder layer and activates MHTA only on configured layers. The default full-training override uses `hybrid_tropical_layers: [-1]`, so the final layer receives the max-plus/Hilbert-projective computation while earlier layers stay on the fast attention path.

Proof-level details are in the paper PDF linked above and in the local TeX source. See the subsection `Multi-head Tropical Attention as an optional UGM backend` in [`assets/human_learning_transformer_learning_review_dataset_expanded.tex`](./assets/human_learning_transformer_learning_review_dataset_expanded.tex), which includes:

- the definition of tropical projective representatives and the Hilbert projective metric;
- the external proof references from the Tropical Attention paper: Theorem C.3, Corollary C.3.1, and Theorem 3.2;
- a projective-invariance lemma for `d_H`;
- a proposition showing that one MHTA head realizes a weighted max gate;
- a theorem proving that the MHTA block is piecewise-linear over tropical cells;
- a corollary explaining why bounded-horizon max-plus dynamic programs can be unrolled by stacked MHTA layers.

**Why we include Hebrew in the training corpus:**
Hebrew is included in both early and later training because it gives UGM a naturally occurring graph-reasoning problem inside language. English has rich syntax, but many English cues are carried by near-linear word order and concatenative morphology in most situations. Hebrew adds non-concatenative root-template morphology, clitic attachment, agreement, pronominal suffixes, optional diacritics, and frequent ambiguity in unvocalized forms. A single surface form can require the model to infer a latent root, template, part of speech, inflectional features, argument structure, and discourse context; in addition, one Hebrew root can generate a family of forms that are distant as strings but close as a lexical-semantic graph.

For a UGM-style model, those phenomena are direct supervision for graph reasoning. The helpful representation is not only a left-to-right path: it is a typed hypergraph whose nodes may include surface tokens, roots, radicals, templates, morphemes, lemmas, agreement features, predicates, and syntactic roles. Training on Hebrew rewards the model for preserving incidence relations, resolving many-to-one and one-to-many mappings, following multi-hop dependencies, and remaining invariant to surface changes that do not change the underlying root or role. These are the same operations UGM needs for proof graphs, tool traces, molecule strings, residue motifs, and oracle-scored candidate records.

The point is not that Hebrew replaces English or that language-specific heuristics should dominate the model. The point is that a multilingual curriculum containing Hebrew exposes the model to relational structure that English-only training can often solve with positional shortcuts. Hebrew therefore acts as a compact stress test and training signal for UGM's intended bias: treat sequences as renderings of structured graph records, and learn the latent graph that explains the surface string.

The implemented path is intentionally practical for one RTX 4090: graph-rich examples, a compact TokenGT-style transformer, random-order autoregressive graph-token decoding, staged training, validation, inference, and a local GFlowNet trajectory-balance stage.

## What Is Implemented

- Typed graph JSON schema for language, math, proof, code, tool, and molecule examples.
- Dataset acquisition manifest and small downloaded samples in `data/raw/`.
- Graphification scripts for synthetic, math, code, Lean/proof, and molecule rows.
- Random-order autoregressive decoding with `<POS>` query tokens.
- Compact TokenGT-style model using node/edge endpoint identifier embeddings, optional gradient checkpointing, local LoRA adapters, a pure Multi-Head Tropical Attention backend, and a hybrid Flash-eligible SDPA plus MHTA backend.
- Training runner with tqdm, JSONL metrics, checkpoints, resume support, validation, AMP, gradient clipping, schedulers, and optional W&B.
- Topology summaries, tropical logit diagnostics, Tropical Attention metrics, verifier adapters, and curation tooling.
- Domain vertical slices for code/unit tests, Lean availability/compile checks, and RDKit-backed molecule graphs when RDKit is installed.
- PLAN-D/PLAN-G science-data slices for SFM/NatureLM reference vocabulary and UniGenX-style molecule/material graphs.
- PLAN-E Hebrew morphology/root slices with UD Hebrew HTB, Hebrew QA, Nakdimon diacritization, root-template graphs, and root-extension GFlowNet training.
- PLAN-F/PLAN-G deferred-component closure: optional advanced topology backends, tropical attention/parser utilities, autoregressive coordinate/property graph tokens plus a gated continuous coordinate head, audio feature extraction, local SFM/NatureLM and UniGenX science-source preparation, bioactivity/docking/protein graphification, safer verifier execution, stronger curation, and context-aware learned-backward GFlowNets.
- PLAN-H UGM multimodal graph-to-graph phase: sequence-first vocabulary for text/protein/SELFIES/SMILES/DNA/RNA/tool/oracle records, local-source preparation, continuous temperature conditioning, UMA-conditioned coupling/motion bins, function-description alignment, and oracle-feedback GFlowNet rewards.
- BioSELFIES-style symbolic graphification for the oracle-dynamics addendum: `bioselfies`, `bio_selfies`, or `input_representation: bioselfies` rows decode into typed protein/DNA/RNA/SELFIES/atom/link/modification/patch/H-bond/torsion/thought graph records. UniProt feature, bioactivity, and biomolecular-complex affinity rows now also receive a BioSELFIES view automatically, including ligand SELFIES where a ligand SMILES string is available. If the optional `selfies` package is absent, simple ligand SMILES still receive a conservative fallback SELFIES-style serialization such as `[C][C][O]`, so protein-ligand and protein-DNA-ligand affinity rows remain in the SELFIES/BioSELFIES path. The decoder is total, so unsupported tokens become explicit `bioselfies_unknown` nodes rather than parser failures. This is a symbolic interface only; it does not introduce coordinates, distance labels, force labels, energy labels, PDB/mmCIF/SDF files, conformer libraries, or MD trajectories.
- All-atom Cartesian structure-dynamics candidate tokens for the strict oracle path: sequence/BioSELFIES rows can emit `ALL_ATOM_CARTESIAN:*`, `CARTESIAN_ATOM:protein:*`, `CARTESIAN_ATOM:nucleic_acid:*`, `CARTESIAN_ATOM:ligand:*`, and `CARTESIAN_FRAME:*` targets. These are output/action labels for model-generated coordinate proposals and UMA force scoring, not supervised coordinate labels.
- All-atom contact-template source graphs for structure-dynamics rows: protein/DNA/RNA/SELFIES inputs now derive a sequence-initialized unfolded all-atom template with `all_atom_template_atom` nodes and `molecular_bond` edge tokens. Attention/contact maps are still TokenGT source-token maps; with this path enabled, those maps include atom tokens and covalent bond edge tokens under the configured `max_source_tokens` budget.
- Internal-coordinate action slots for structure-dynamics training: symbolic protein/RNA/DNA/SELFIES rows can create `INTERNAL_COORD_QUERY:*` source slots, and the model emits torsion-like actions such as `protein_phi`, `protein_psi`, `protein_omega`, side-chain chi, nucleic-acid torsions, sugar pucker, and ligand torsion. These actions are trained through UMA-scored generated coarse geometries, not copied structure labels.
- Separate GFlowNet tracks for SFT and structure-dynamics: `gflownet.mode: sft` learns diverse symbolic graph completions, while `gflownet.mode: structure_dynamics` filters candidates to BioSELFIES/all-atom Cartesian, internal-coordinate, contact-patch, adaptive-patch, temperature, token-motion, and UMA/oracle records. The structure-dynamics trainer can also derive those candidates from older curated biomed rows, so a failed GFlowNet phase can be restarted without re-running multi-hour curation when the SFT stages are already complete.
- UniProt feature and binding-site graphification: local UniProt TSV/CSV/JSON/JSONL exports can add accessions, names, organism/taxon, GO, keywords, EC, domains, PTMs, variants, cofactors, catalytic activity, subcellular location, subunit text, binding sites, active sites, metal-binding sites, DNA-binding sites, and other sequence features.
- Biomolecular-complex affinity graphification: local rows for protein-protein, protein-RNA, protein-DNA, protein-ligand, ligand-nucleic-acid, antibody-antigen, receptor-ligand, or arbitrary component complexes can carry `Kd`, `Ki`, `IC50`, `kon`, `koff`, or `dG`-style values with units, temperature, pH, buffer, and assay metadata. These rows emit `AFFINITY_CONTACT:*`, `COMPLEX_CONTACT:*`, `PPI_CONTACT:*`, and `CONTACT_PATCH:affinity_weighted_interface` records so PPI and multimodal biomolecular interaction data are trained through the same SFT and structure-dynamics GFlowNet path.
- Full motif vocabulary path for sequence-first multimodal training: PROSITE, InterPro, Rfam, core sequence motifs, safe `SEQ_MOTIF_FROM_STRUCTURE:*` vocabulary entries, and optional non-structure molecule descriptors are parsed into graph-record vocabulary tokens; row-local structure motifs from coordinates/contact labels are evaluation/future-phase only.
- Structure/dynamics sources are validation-only by default. Actual PDB/mmCIF/SDF/trajectory coordinate labels, energy labels, and force labels remain disabled. The active coordinate path uses model-generated coordinates scored by UMA as an online oracle.
- Verifier-aware GFlowNet graph-of-thought trajectory-balance trainer plus rollout validation.
- Config-driven validation and inference CLIs.
- Smoke tests.
- `torchgfn` cloned under `data/external_repos/torchgfn` for reference.
- `Tropical-Attention` cloned under `data/external_repos/Tropical-Attention` for the upstream MHTA reference implementation.

## Environment

Create the project environment:

```bash
bash scripts/setup_conda_env.sh
conda activate iska-ugm
```

The implementation was also smoke-tested in the existing `tokengt` conda env:

```bash
conda run -n tokengt pytest -q
```

Check local readiness, including CUDA, optional domain packages, Lean, SFM/UniGenX repos, and reference-token files:

```bash
conda run -n tokengt python scripts/check_readiness.py
```

## Data

Generate deterministic synthetic graph data:

```bash
conda run -n tokengt python scripts/graphify_data.py \
  --synthetic --count 512 \
  --output data/processed/synthetic_graphs/train.jsonl
```

Acquire a small public sample:

```bash
conda run -n tokengt python scripts/acquire_datasets.py \
  --dataset gsm8k_main_train --limit 64
```

Graphify a raw sample:

```bash
conda run -n tokengt python scripts/graphify_data.py \
  --input data/raw/gsm8k_main_train/train.jsonl \
  --output data/processed/gsm8k_main_train/train.jsonl \
  --dataset-name gsm8k_main_train
```

Curate, deduplicate, and split graphified data:

```bash
conda run -n tokengt python scripts/curate_data.py \
  --input data/processed/mixed_graphs/train.jsonl \
  --output-dir data/processed/curated_graphs \
  --val-ratio 0.1 \
  --test-ratio 0.1 \
  --near-dedup-threshold 0.9 \
  --blocked-license-pattern "noncommercial"
```

Dataset details are in `planning/DATASETS.md` and `data/manifests/datasets.yaml`.

The executable dataset catalog plan is in `planning/DATASET-CATALOG-IMPLEMENTATION-PLAN.md`. Validate the catalog after any manifest, acquisition, graphification, vocabulary, or split change:

```bash
conda run -n tokengt python scripts/validate_dataset_catalog.py --no-progress
```

This writes `data/manifests/dataset_catalog_status.json` and `planning/DATASET-CATALOG-STATUS.md`. The previous completed public-corpus baseline had 19 active public Hugging Face parquet entries, 7,328,008 graph examples, and 1,107,708,497 untruncated model-sequence graph tokens. The current expanded default selection adds ranked graph-reasoning, scientific sequence/function, math verifier, and curated general text sources under a 5B untruncated graph-token guard.

The highest-priority additions are included by default in `data/manifests/datasets.yaml`: OpenAI GraphWalks, GraphWiz GraphInstruct-RFT-72K, PubChem10M SELFIES, UniRef50/UniProt sequence rows, UniProt function text descriptions, Rfam, RNAcentral, DNA coding regions, OpenMathReasoning TIR, OpenMathReasoning GenSelect, and a DCLM baseline 1B slice. GraphWalks is intentionally first because it most directly trains long-context graph-state reasoning. The bio-scale row caps are set for million-row modality coverage where the public source supports it: ConvergeBio UniRef50 at 3M rows for protein sequences, PubChem10M SELFIES at 8M rows for molecule strings, Rfam at 3M rows, RNAcentral at 3M rows, and DNA coding regions at 3M requested rows. The current public DNA coding split is source-limited below 3M rows, so the bio-scale runner records that as a warning rather than silently pretending the DNA target was met. OpenMathReasoning TIR remains capped at 1.3M and GenSelect at 300k.

Run the complete full selected-corpus training sequence with live tqdm streaming, per-stage logs, a master log, progress/status files, and runner-level W&B events:

```bash
scripts/run_full_training_sequence.sh
```

For the full phase 1 plus phase 2 curriculum, use the wrapper with the correct default budget, structure-training gate, W&B online logging, and training-first behavior:

```bash
./scripts/run_full_phase1_phase2_training.sh
```

To hold the same corpus, objectives, context limits, validation cadence, UMA preflight, and runner behavior while using the 250M-parameter UGM config, run:

```bash
./scripts/run_full_phase1_phase2_training_250m.sh
```

The script writes to `logs/full_training_sequence/<RUN_ID>/` and updates `logs/full_training_sequence/latest`. It uses `data/processed/real_full_selected_mix/`, honors manifest-local row caps such as the RNA/DNA caps, and enforces `MAX_GRAPH_TOKENS=5000000000` by default. Standard softmax training uses the 2x context config at `config/generated/real_full_selected_context_2x.yaml`; `ENABLE_TROPICAL_ATTENTION=1` defaults to the compact exact-coverage context config at `config/generated/real_full_selected_context_compact.yaml` because MHTA memory scales quadratically with sequence length. The phase wrapper defaults to `TRAINING_FIRST=1`, `SKIP_REFERENCE_REFRESH_IF_READY=1`, `SKIP_INTERPRO_MOTIF_DOWNLOAD=1`, and `REQUIRE_UMA_WEIGHTS=1`, so if the graph corpus and reference vocabularies already exist it proceeds to training while still verifying/downloading the required FairChem/UMA weights before oracle stages. Phase-1 training is full-corpus by default: the runner writes a per-run override with `max_steps: full_epoch` and `FULL_TRAIN_EPOCHS=1.0`, so optimizer steps are computed from the actual train split length, batch size, and gradient accumulation rather than from a fixed smoke-test cap. Training stages automatically append `config/train/overrides/wandb_online.yaml`; shell stages and commands also log W&B runner events when `WANDB_ENABLED=1`.

Common controls:

```bash
DRY_RUN=1 scripts/run_full_training_sequence.sh
START_AT=03 scripts/run_full_training_sequence.sh
STOP_AFTER=02 scripts/run_full_training_sequence.sh
VALIDATION_DEVICE=cpu scripts/run_full_training_sequence.sh
MAX_GRAPH_TOKENS=4500000000 scripts/run_full_phase1_phase2_training.sh
WANDB_MODE=offline scripts/run_full_phase1_phase2_training.sh
WANDB_ENABLED=0 scripts/run_full_phase1_phase2_training.sh
UMA_SCORE_SMOKE=1 scripts/run_full_phase1_phase2_training.sh
ENABLE_UMA_INTERNAL_COORDINATES=1 ./scripts/run_full_phase1_phase2_training_250m_oracle_dynamics.sh
FULL_TRAIN_EPOCHS=2.0 scripts/run_full_phase1_phase2_training.sh
FULL_TRAIN_EVAL_MAX_BATCHES=full scripts/run_full_phase1_phase2_training.sh
FULL_TRAIN_NUM_WORKERS=12 FULL_TRAIN_PREFETCH_FACTOR=6 scripts/run_full_phase1_phase2_training.sh
FULL_TRAIN_BATCH_SIZE=6 FULL_TRAIN_GRAD_ACCUM=6 ./scripts/run_full_phase1_phase2_training_250m.sh
FULL_TRAIN_SKIP_POLICY_CHECK=1 ./scripts/run_full_phase1_phase2_training_250m.sh
ENABLE_TROPICAL_ATTENTION=1 FULL_TRAIN_BATCH_SIZE=2 FULL_TRAIN_GRAD_ACCUM=18 ./scripts/run_full_phase1_phase2_training_250m.sh
./scripts/train_full_selected_250m_direct.sh
FULL_TRAIN_MAX_STEPS=20 ./scripts/train_full_selected_250m_direct.sh
ENABLE_TROPICAL_ATTENTION=1 FULL_TRAIN_MAX_STEPS=20 ./scripts/train_full_selected_250m_direct.sh
ENABLE_TROPICAL_ATTENTION=1 ENABLE_UMA_COORDINATE_HEAD=1 FULL_TRAIN_MAX_STEPS=20 ./scripts/train_full_selected_250m_direct.sh
./scripts/run_full_phase1_phase2_training_250m_oracle_dynamics.sh
./scripts/train_full_selected_250m_oracle_dynamics_direct.sh
MODEL_CONFIG=config/model/ugm_250m_tokengt.yaml scripts/run_full_phase1_phase2_training.sh
TRAINING_FIRST=0 SKIP_INTERPRO_MOTIF_DOWNLOAD=0 scripts/run_full_phase1_phase2_training.sh
```

Audit dataset sizes against the local machine before scaling:

```bash
conda run -n tokengt python scripts/audit_dataset_capacity.py
```

The audit writes `planning/DATASET-CAPACITY-AUDIT.md` and `data/manifests/dataset_capacity_audit.json`. On the current machine, the manifest sees about 2.5 TB free disk after downloads, about 49.6 GB available RAM, and an RTX 4090 with 24 GB VRAM.

Download the full public Hugging Face selected splits named by the manifest into parquet snapshots:

```bash
conda run -n tokengt python scripts/download_hf_selected_splits.py \
  --manifest data/manifests/datasets.yaml \
  --out-dir data/raw_hf_full \
  --max-total-gib 32
```

The expanded selected-split snapshot is expected to be materially larger than the old 2.8 GB baseline. Manifest-only entries, local-file entries, gated/very large entries, and user-provided reconstruction sources are not bulk-downloaded by this command.

Build the full real-data graph corpus from the downloaded parquet snapshots. This command honors manifest-local row caps and reports global, per-dataset, and per-parquet-file tqdm progress with live split, dataset, and error counters:

```bash
conda run -n tokengt python scripts/graphify_full_parquet_manifest.py \
  --manifest data/manifests/datasets.yaml \
  --raw-full-dir data/raw_hf_full \
  --output-dir data/processed/real_full_selected_mix \
  --val-ratio 0.01 \
  --test-ratio 0.01 \
  --batch-size 8192 \
  --progress-every 10000
```

Count full-corpus graph tokens and enforce the 5B untruncated graph-token cap:

```bash
conda run -n tokengt python scripts/count_graph_tokens.py \
  --data-dir data/processed/real_full_selected_mix \
  --output data/processed/real_full_selected_mix/token_counts.json \
  --progress-every 100000 \
  --max-model-sequence-tokens-total 5000000000
```

Generate the context-window audit and 2x context config:

```bash
conda run -n tokengt python scripts/inspect_context_requirements.py \
  --data-dir data/processed/real_full_selected_mix \
  --output data/processed/real_full_selected_mix/context_requirements.json \
  --context-multiplier 2.0 \
  --write-context-config config/generated/real_full_selected_context_2x.yaml
```

Verify that the split files on disk match `summary.json` before training. This catches interrupted graphification runs:

```bash
conda run -n tokengt python scripts/check_dataset_integrity.py \
  --data-dir data/processed/real_full_selected_mix \
  --output data/processed/real_full_selected_mix/integrity.json
```

After a completed integrity-checked graphification, the full public selected-split graph corpus is `data/processed/real_full_selected_mix/`. The old baseline counts are preserved in `planning/FULL-PRETRAINING-DATASET.md`; the expanded counts are produced by the next full run and must report `within_model_sequence_token_budget: true` before training starts.

When the 4090 is free, train on the full selected corpus. The default full selected-corpus training config now uses one full pass over the train split:

```bash
conda run -n tokengt python scripts/train_stage.py \
  --config config/model/max_4090_tokengt.yaml \
  --config config/data/real_full_selected_mix.yaml \
  --config config/generated/real_full_selected_context_2x.yaml \
  --config config/train/real_full_selected_local.yaml
```

`config/train/real_full_selected_local.yaml` sets `max_steps: full_epoch` and `full_epochs: 1.0`. The trainer computes optimizer steps from `ceil(train_examples / (batch_size * gradient_accumulation_steps))`, so a full epoch remains a full pass over the train split after batch-size changes. In-training validation is intentionally capped with `eval_max_batches: 512`; the stage still runs full validation and test after training through `scripts/validate_stage.py`. The full runner defaults to eight train DataLoader workers, pinned memory, persistent workers, prefetching, and cached JSONL byte offsets because the full corpus is an 88GB JSONL file and CPU graph parsing can otherwise leave the GPU underfed. Phase-1 full-corpus training backpropagates the autoregressive token objective; topology values remain logged as diagnostics, while weighted topology auxiliary losses are reserved for later normalized ablation stages. The trainer fails fast and writes an emergency checkpoint if any loss becomes non-finite.

The 250M setup uses `config/model/ugm_250m_tokengt.yaml`, `config/data/real_full_selected_mix_250m.yaml`, `config/train/real_full_selected_250m_local.yaml`, `config/validate/real_full_selected_250m_validation.yaml`, `config/validate/real_full_selected_250m_test.yaml`, and `config/inference/real_full_selected_250m_inference.yaml`. Its artifacts live under `outputs/real_full_selected_250m_local/` so it can be trained beside the default 576M run. Standard softmax 250M training defaults to `batch_size: 6`, `eval_batch_size: 6`, and `gradient_accumulation_steps: 6`. Hybrid FlashAttention-2/SDPA plus MHTA training defaults to compact 926-token context with layer-sparse MHTA: FlashAttention/SDPA runs in every layer, and MHTA runs in the final layer unless `hybrid_tropical_layers` is overridden. The hybrid config also sets `tropical_query_chunk_size: 96`, which computes the exact MHTA Hilbert-distance operation in query blocks rather than materializing the full `[batch, heads, tokens, tokens, head_dim]` tensor at once. The direct wrapper uses `batch_size: 2`, `eval_batch_size: 2`, and `gradient_accumulation_steps: 18` for the sparse hybrid path; set `FULL_TRAIN_BATCH_SIZE=1 FULL_TRAIN_GRAD_ACCUM=36` if another process is sharing the GPU or if local activation memory is tighter. After the corpus, UMA weights, and vocab have already been checked or built once, `./scripts/train_full_selected_250m_direct.sh` jumps straight to `train_stage.py` with the same 250M configs, reuses the existing vocab, and skips the full runner preamble. Its `SKIP_POLICY_CHECK=1` default should only be used after the same unchanged corpus has passed the full sequence-only policy scan.

### Sequence-First Multimodal And FairChem/UMA Oracle Conditioning

The current multimodal phase uses local sequence/string exports from `data/local/multimodal/`. The `--synthetic-if-empty` flag is reserved for smoke tests only; production training should omit it so missing reviewed rows fail visibly. First-run molecular training is SELFIES/SMILES and protein/DNA/RNA sequence-first: actual structure files, coordinate trajectories, energies, and forces are excluded from supervised training. When the optional UMA coordinate head is enabled, protein sequences receive sequence-derived heavy-atom Cartesian query slots, starting with backbone atoms and continuing into residue side-chain heavy atoms up to the configured cap. Molecule strings use RDKit-derived atom symbols with explicit hydrogens when RDKit is available. Those coordinates are generated by the model and scored by UMA; they are not copied from PDB, SDF, mmCIF, or trajectory files.

Temperature conditioning is continuous over roughly `300K..400K`. The graphifier stores both stable anchor/bin tokens and continuous Kelvin features, and FairChem/UMA oracle calls use that same continuous temperature when scoring candidate graph states.

Function-description training is part of the first curriculum. Use SFM/NatureLM-style sequence science rows, sanitized UniGenX sequence/function metadata, ProTrek-style protein sequence/function pairs, UniProt/InterPro/GO/EC annotations, and assay/target annotations as graph-to-graph function-description examples. These rows may include `function_description` and `sequence_motifs_from_structure`, but they must not include atoms, coordinates, contact maps, PDB/mmCIF/SDF files, energies, forces, or trajectories.

UMA influences graph-state evolution through fine-grained binned records only in the UMA candidate-scoring stage. Ordinary sequence reconstruction and function-description rows may carry temperature and function nodes, but they do not emit AF-style 64-bin `ATTN_BIN:*`, `TOKEN_COUPLING:uma:*`, `UMA_INFLUENCE:uma:*`, `TOKEN_MOTION:uma:*`, `UMA_TRAJ_BIN:*`, or `SEQ_STRUCT_DYN_PROXY:*` targets unless an oracle stage flag or structure-dynamics task is active. In that stage, GFlowNet training learns temperature-conditioned coupling, motion, and trajectory policies from SELFIES/FASTA inputs only.

The stage is still aimed at actual structure-dynamics generation. The model should propose typed atom/bond/coordinate/frame graph records; the restriction is that those predictions are trained through UMA/verifier/GFlowNet feedback rather than supervised copying of structure files, energy/force labels, or MD frames. Generated-token PDB rendering is optional and not required for this pass.

The internal-coordinate path is still the preferred structure-dynamics quality upgrade. Instead of making the coordinate head carry all geometry, the input collator adds source-side action slots such as `INTERNAL_COORD_QUERY:protein_phi`, `INTERNAL_COORD_QUERY:protein_psi`, `INTERNAL_COORD_QUERY:protein_omega`, side-chain chi, nucleic-acid torsions, sugar pucker, and ligand torsion. In parallel, the all-atom Cartesian token family names the coordinate proposal slots that the model must learn to construct. The model reads both families from the graph-of-thought embedding state, proposes torsion-like actions and Cartesian atom positions, builds a generated geometry, and receives UMA energy/force feedback. This keeps training autoregressive and oracle-supervised while making the geometry more physically structured than raw Cartesian slots alone.

The coordinate head is a readout from embedding-space graph-of-thought state, not a replacement for it. UGM continues to learn graph reasoning through hidden thought states, graph-token autoregression, attention/contact fields, and verifier/oracle records. The UMA coordinate objective backpropagates through the coordinate readout into the same hidden embeddings and attention layers. With `uma_coordinate_dynamics_steps > 1`, each reasoning iteration is treated like a short physical-time step: the current generated coordinates are scored by UMA at the graph's continuous temperature, detached UMA forces roll the candidate forward, and the next score is logged and trained against the same latent graph state. Low-temperature examples emphasize stabilization and refinement; high-temperature examples keep broader contact and coordinate support.

Mechanistically, the UMA stage treats evolving attention maps as contact fields. When `emit_attention_contact_maps` is enabled for the MHTA or hybrid FlashAttention/MHTA backend, MHTA Hilbert-distance scores are converted into row-normalized contact maps and fused with Euclidean hidden-state geometry plus Jensen-Shannon hidden-state geometry. The resulting field is a sequence-conditioned contact hypothesis, not a copied structure label. Structure-dynamics graphification now also inserts an all-atom contact template: each unfolded atom is a source node and each covalent bond is a `molecular_bond` source edge, so TokenGT attention can attend to residues/bases/SELFIES tokens, atom tokens, and bond edge tokens in one graph. With `config/train/overrides/uma_contact_geometry_loss.yaml`, rows that carry explicit UMA feedback records train both the contact map and embedding geometry: low-temperature rows prefer sharper stabilization/refinement support, while high-temperature rows prefer higher contact-map and embedding-geometry entropy, broader token-motion records such as `explore`, `diversify`, and `expand`, and more diverse GFlowNet terminal states. FairChem/UMA then scores candidate molecules by energy and force at the graph-specified Kelvin temperature; this oracle reward is what reinforces the graph-state path.

For long inputs, the all-atom contact template is budgeted. With the 8192-source-token override, the graphifier reserves source tokens for ordinary sequence/BioSELFIES/context nodes and then fills the remaining budget with atom nodes plus bond edge tokens. Full all-atom Cartesian output files can still be generated for the configured trajectory atom cap; a complete untruncated all-atom attention map for very large proteins would require a larger context window or a patch/chunked atom-contact schedule.

The live UMA backend is FairChem from the Amelie Schreiber fork at `data/external_repos/fairchem`. Repository acquisition alone does not download gated UMA weights. To run real oracle scoring, install/resolve FairChem dependencies, request access to `facebook/UMA` on Hugging Face, authenticate with `hf auth login`, and resolve the default UMA-S-1.2 checkpoint plus OMol reference tables:

```bash
conda run -n tokengt python scripts/acquire_model_files.py --repo-name fairchem

conda run -n tokengt python scripts/download_uma_weights.py \
  --repo data/external_repos/fairchem \
  --model-name uma-s-1p2 \
  --task-name omol \
  --device cuda

conda run -n tokengt python scripts/check_uma_oracle.py \
  --repo data/external_repos/fairchem \
  --smiles "CCO" \
  --temperature 325 \
  --backend fairchem \
  --strict
```

`scripts/download_uma_weights.py` uses FairChem's model registry and Hugging Face cache, downloading `facebook/UMA` files when they are absent and reporting their resolved local paths. The full training runners call this preflight when `REQUIRE_UMA_WEIGHTS=1`; set `UMA_SCORE_SMOKE=1` to run a strict `CCO` ASE/FairChem scoring check before training. `UGM_UMA_BACKEND=proxy` is available only for unit tests and local smoke checks that must not download gated UMA weights. The production oracle GFlowNet configs use `oracle.backend: fairchem` with `strict: true`, so a missing FairChem clone, missing Hugging Face access, or invalid candidate causes a clear failure rather than silent proxy scoring.

For the optional contact-map/geometry coupling during MHTA training, append the UMA geometry override through the runner's extra-config hook:

```bash
ENABLE_TROPICAL_ATTENTION=1 \
EXTRA_TRAIN_CONFIGS="config/train/overrides/uma_contact_geometry_loss.yaml" \
./scripts/train_full_selected_250m_direct.sh
```

For diagnostics without backpropagating through the contact maps, use `EXTRA_TRAIN_CONFIGS="config/model/overrides/attention_contact_maps.yaml config/train/overrides/folding_contact_diagnostics.yaml"` instead.

```bash
conda run -n tokengt python scripts/prepare_multimodal_sources.py \
  --input-dir data/local/multimodal \
  --synthetic-if-empty \
  --molecular-input-policy sequence_only \
  --output data/processed/multimodal_graphs/all.jsonl

conda run -n tokengt python scripts/curate_data.py \
  --input data/processed/multimodal_graphs/all.jsonl \
  --output-dir data/processed/multimodal_graphs \
  --val-ratio 0.2 \
  --test-ratio 0.1

conda run -n tokengt python scripts/train_stage.py \
  --config config/model/max_4090_tokengt.yaml \
  --config config/data/multimodal_graphs_4090.yaml \
  --config config/train/multimodal_phase2_4090.yaml

conda run -n tokengt python scripts/validate_stage.py \
  --config config/validate/multimodal_4090_validation.yaml

conda run -n tokengt python scripts/validate_stage.py \
  --config config/validate/multimodal_4090_test.yaml

conda run -n tokengt python scripts/check_dataset_policy.py \
  --data-dir data/processed/multimodal_graphs \
  --sequence-only-molecules

conda run -n tokengt python scripts/train_stage.py \
  --config config/data/multimodal_graphs_4090.yaml \
  --config config/train/multimodal_oracle_gflownet_4090.yaml
```

### PLAN-D/PLAN-G Science Data

The corrected science references are SFM/NatureLM (`https://github.com/amelie-iska/SFM`, paper `https://arxiv.org/abs/2502.07527`) and UniGenX (`https://github.com/amelie-iska/UniGenX`, paper `https://arxiv.org/abs/2503.06687`).

Clone the corrected SFM/NatureLM and UniGenX reference repositories, clone the Tropical-Attention reference implementation, and extract science domain/token dictionaries into the training vocabulary:

```bash
conda run -n tokengt python scripts/acquire_model_files.py --repo-name sfm
conda run -n tokengt python scripts/acquire_model_files.py --repo-name unigenx
conda run -n tokengt python scripts/acquire_model_files.py --repo-name tropical_attention

conda run -n tokengt python scripts/extract_reference_tokens.py \
  --sfm-dir data/external_repos/sfm \
  --unigenx-dir data/external_repos/unigenx \
  --output data/processed/reference_tokens/naturelm_unigenx_tokens.txt
```

Acquire tiny UniGenX-style public samples:

```bash
conda run -n tokengt python scripts/acquire_datasets.py \
  --dataset unigenx_qm9_train --limit 4

conda run -n tokengt python scripts/acquire_datasets.py \
  --dataset unigenx_materials_crystal_system --limit 4
```

### PLAN-E Hebrew Data

Acquire tiny Hebrew HF samples:

```bash
for ds in \
  hebrew_sefaria_train \
  hebrew_synthetic_medical_train \
  hebrew_wikianswers_lists \
  hebrew_wikianswers_queries \
  hebrew_alpaca_train \
  talmud_hebrew_train \
  hebrew_wikipedia_train
do
  conda run -n tokengt python scripts/acquire_datasets.py --dataset "$ds" --limit 8
done
```

Clone structured Hebrew repositories:

```bash
conda run -n tokengt python scripts/acquire_datasets.py --dataset hebrew_ud_htb
conda run -n tokengt python scripts/acquire_datasets.py --dataset hebrew_qa_nnlp
conda run -n tokengt python scripts/acquire_datasets.py --dataset hebrew_nakdimon
```

Graphify HF rows, then prepare UD/QA/Nakdimon/root-extension data:

```bash
conda run -n tokengt python scripts/graphify_data.py \
  --input data/raw/hebrew_alpaca_train/train.jsonl \
  --output data/processed/hebrew_alpaca_train/train.jsonl \
  --dataset-name hebrew_alpaca_train

conda run -n tokengt python scripts/prepare_hebrew_sources.py \
  --ud-limit 16 --qa-limit 16 --nakdimon-limit 8 --root-count 32
```

Merge the Hebrew graph sources into a training mix:

```bash
conda run -n tokengt python scripts/merge_jsonl.py \
  --input data/processed/hebrew_sefaria_train/train.jsonl \
  --input data/processed/hebrew_synthetic_medical_train/train.jsonl \
  --input data/processed/hebrew_wikianswers_lists/train.jsonl \
  --input data/processed/hebrew_wikianswers_queries/train.jsonl \
  --input data/processed/hebrew_alpaca_train/train.jsonl \
  --input data/processed/talmud_hebrew_train/train.jsonl \
  --input data/processed/hebrew_wikipedia_train/train.jsonl \
  --input data/processed/hebrew_ud_htb/train.jsonl \
  --input data/processed/hebrew_qa_nnlp/train.jsonl \
  --input data/processed/hebrew_nakdimon/train.jsonl \
  --input data/processed/hebrew_root_synthetic/train.jsonl \
  --output data/processed/hebrew_mix/all.jsonl

conda run -n tokengt python scripts/curate_data.py \
  --input data/processed/hebrew_mix/all.jsonl \
  --output-dir data/processed/hebrew_mix \
  --val-ratio 0.15 \
  --test-ratio 0.0
```

The Verb Complements Lexicon is manifest-only until an official downloadable source is provided. If a TSV/CSV with `verb_root`, `verb_binyan`, `verb_LexiconItem`, and `complement_LexiconItem` columns is placed under `data/raw/hebrew_verb_complements_lexicon/`, `scripts/prepare_hebrew_sources.py` will ingest it.

Refresh SFM and UniGenX GitHub metadata:

```bash
conda run -n tokengt python scripts/acquire_model_files.py \
  --repo-name sfm

conda run -n tokengt python scripts/acquire_model_files.py \
  --repo-name unigenx
```

Graphify and curate the tiny science mix:

```bash
conda run -n tokengt python scripts/graphify_data.py \
  --input data/raw/unigenx_qm9_train/train.jsonl \
  --output data/processed/unigenx_qm9_train/train.jsonl \
  --dataset-name unigenx_qm9_train

conda run -n tokengt python scripts/graphify_data.py \
  --input data/raw/unigenx_materials_crystal_system/train.jsonl \
  --output data/processed/unigenx_materials_crystal_system/train.jsonl \
  --dataset-name unigenx_materials_crystal_system

conda run -n tokengt python scripts/merge_jsonl.py \
  --input data/processed/unigenx_qm9_train/train.jsonl \
  --input data/processed/unigenx_materials_crystal_system/train.jsonl \
  --output data/processed/science_mix/all.jsonl

conda run -n tokengt python scripts/curate_data.py \
  --input data/processed/science_mix/all.jsonl \
  --output-dir data/processed/science_mix \
  --val-ratio 0.2 \
  --test-ratio 0.0
```

### PLAN-F Deferred Science Paths

Acquire and prepare the tractable official NatureLM/SFM public-source slices. This downloads PubChem extras, UniProt Swiss-Prot, selected RefSeq slices, records Materials Project as API-gated when `MP_API_KEY` is absent, and graphifies the non-large selected sources into train/validation/test JSONL splits. The default split policy is entity-aware for scientific data; pass `--split-policy row_hash` only for smoke tests that intentionally use row-level splitting.

```bash
conda run -n tokengt python scripts/acquire_naturelm_sources.py \
  --prepare \
  --val-ratio 0.01 \
  --test-ratio 0.01 \
  --log-dir logs/naturelm_acquisition \
  --wandb-mode offline

conda run -n tokengt python scripts/check_dataset_integrity.py \
  --data-dir data/processed/naturelm_public_sources \
  --output data/processed/naturelm_public_sources/integrity.json

conda run -n tokengt python scripts/count_graph_tokens.py \
  --data-dir data/processed/naturelm_public_sources \
  --line-counts \
  --output data/processed/naturelm_public_sources/token_counts.json

conda run -n tokengt python scripts/check_identifier_stats.py \
  --data-dir data/processed/naturelm_public_sources \
  --output data/processed/naturelm_public_sources/identifier_stats.json

conda run -n tokengt python scripts/inspect_context_requirements.py \
  --data-dir data/processed/naturelm_public_sources \
  --context-multiplier 2.0 \
  --output data/processed/naturelm_public_sources/context_requirements.json \
  --write-context-config config/generated/naturelm_public_sources_context_2x.yaml
```

If a corpus was prepared with an older row-hash split, resplit it without re-graphifying the raw sources:

```bash
conda run -n tokengt python scripts/resplit_graph_jsonl.py \
  --input-dir data/processed/naturelm_public_sources \
  --output-dir data/processed/naturelm_public_sources_entity \
  --split-policy entity \
  --val-ratio 0.01 \
  --test-ratio 0.01
```

Large paths are deliberately explicit. Full PubChem CID-SMILES graphification requires `--prepare-large`; UniProt TrEMBL requires `--include-large --prepare-large`; Materials Project requires `MP_API_KEY` plus a local API export.

```bash
conda run -n tokengt python scripts/train_stage.py \
  --config config/model/max_4090_tokengt.yaml \
  --config config/data/naturelm_public_sources_entity.yaml \
  --config config/generated/naturelm_public_sources_context_2x.yaml \
  --config config/train/real_full_selected_local.yaml
```

Acquire and graphify a small public protein/ligand binding sample:

```bash
conda run -n tokengt python scripts/acquire_datasets.py \
  --dataset binding_affinity_public --limit 16

conda run -n tokengt python scripts/graphify_data.py \
  --input data/raw/binding_affinity_public/train.jsonl \
  --output data/processed/binding_affinity_public/train.jsonl \
  --dataset-name binding_affinity_public
```

Prepare local SFM/NatureLM and UniGenX-style sources without reconstructing huge corpora by default:

```bash
conda run -n tokengt python scripts/prepare_science_sources.py \
  --kind uniprot \
  --input /path/to/uniprot.fasta \
  --output data/processed/local_uniprot/train.jsonl \
  --limit 1000

conda run -n tokengt python scripts/prepare_science_sources.py \
  --kind uniprot_features \
  --input /path/to/uniprot_features.tsv \
  --dataset-name uniprot_features_local_export \
  --output data/processed/uniprot_features_local_export/all.jsonl \
  --limit 100000

conda run -n tokengt python scripts/prepare_science_sources.py \
  --kind bindingdb \
  --input /path/to/BindingDB.tsv \
  --output data/processed/local_bindingdb/train.jsonl \
  --limit 1000

conda run -n tokengt python scripts/prepare_science_sources.py \
  --kind pdbbind \
  --input /path/to/pdbbind_rows.jsonl \
  --output data/processed/local_pdbbind/train.jsonl

conda run -n tokengt python scripts/prepare_science_sources.py \
  --kind biomolecular_affinity \
  --input /path/to/complex_affinity.tsv \
  --dataset-name biomolecular_complex_affinity_local \
  --output data/processed/biomolecular_complex_affinity_local/all.jsonl \
  --limit 100000
```

`--kind uniprot_features` accepts UniProt-style fields such as `Entry`, `Reviewed`, `Protein names`, `Gene Names`, `Organism`, `Organism ID`, `Sequence`, `EC number`, `Gene Ontology IDs`, `Keywords`, `Features`, `Binding site`, `Active site`, `Metal binding`, `DNA binding`, `Subcellular location [CC]`, `Cofactor`, `Catalytic activity`, and `Subunit structure`. The graphifier emits symbolic `UNIPROT:*` records and binding-site edges, not coordinates.

`--kind biomolecular_affinity`, `complex_affinity`, `ppi_affinity`, and `protein_na_affinity` accept protein, RNA, DNA, ligand, antibody/antigen, receptor/ligand, or arbitrary `components` rows with affinity fields such as `Kd`, `Ki`, `IC50`, `kon`, `koff`, `delta_g`, or `dG`, plus units, temperature, pH, buffer, and assay metadata.

To make the full local UniProt feature export and full local binding-affinity source trainable as one corpus, then jump directly into the 250M SFT plus two-GFlowNet path:

```bash
./scripts/run_full_biomed_annotations_affinity_training.sh
```

The wrapper first materializes `data/local/uniprot_features.tsv` from the UniProtKB reviewed feature stream and `data/local/complex_affinity.tsv` from the local full-selected `binding_affinity_public` parquet, then graphifies both files, curates `data/processed/biomed_annotations_affinity/{train,val,test}.jsonl`, checks split integrity, and starts training. `TRAIN_PHASES=sft` trains only `config/train/biomed_annotations_affinity_250m.yaml`; `TRAIN_PHASES=gflownet_sft` or `TRAIN_PHASES=structure_dynamics_gflownet` run the corresponding GFlowNet configs. Use `PREPARE_FULL_BIOMED_SOURCES=force PREPARE_UNIPROT=force PREPARE_AFFINITY=force CURATE_DATA=force` to force a complete rebuild. For already graphified full local corpora, the wrapper defaults to `FAST_CURATE=1` and `RESUME_CURATE=1`, which use exact raw-row deduplication, entity splitting, direct JSONL line copying, and resumable temp split/state files so interrupted curation can continue instead of starting over.

To include the original full selected public corpus as well, set `INCLUDE_ORIGINAL_FULL_SELECTED=1`. This appends `data/processed/real_full_selected_mix/{train,val,test}.jsonl` to the UniProt feature and biomolecular-affinity graph files, writes the combined curated corpus to `data/processed/biomed_annotations_affinity_plus_original_full_selected/`, and trains under `outputs/biomed_annotations_affinity_plus_original_250m/` by default:

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

The current full local biomed source set contains 2,411,356 data rows: 574,627 UniProt feature rows and 1,836,729 biomolecular complex-affinity rows. The original full selected public graph corpus contains 7,328,008 rows across its train/validation/test splits. With `INCLUDE_ORIGINAL_FULL_SELECTED=1`, the combined curation input is 9,739,364 rows before exact duplicate removal. The local biomed graph JSONL files are about 36 GB and 66 GB before train/validation/test curation; the original full selected graph JSONL adds about 85 GB. On the current workstation, fast curation of the 2.41M-row biomed-only corpus is about one hour, while the combined 9.74M-row corpus should be budgeted for several hours because it reads about 187 GB of JSONL before writing the new curated splits. The 250M SFT phase is the long part: with effective batch 36, biomed-only is about 60k optimizer steps and the combined corpus is about 240k optimizer steps for one full epoch, so expect a multi-day single-GPU run. The two GFlowNet follow-up phases are capped at 3k steps each by their configs.

For sequence/function-description alignment, use the sequence-only kinds:

```bash
conda run -n tokengt python scripts/prepare_science_sources.py \
  --kind protrek \
  --input /path/to/protrek_or_function_pairs.jsonl \
  --dataset-name protrek_sequence_function \
  --output data/processed/protrek_sequence_function/all.jsonl

conda run -n tokengt python scripts/prepare_science_sources.py \
  --kind naturelm \
  --input /path/to/sfm_or_naturelm_sequence_function.jsonl \
  --dataset-name naturelm_sequence_function \
  --output data/processed/naturelm_sequence_function/all.jsonl

conda run -n tokengt python scripts/prepare_science_sources.py \
  --kind protein_function \
  --input /path/to/unigenx_sequence_function.jsonl \
  --dataset-name unigenx_sequence_function \
  --output data/processed/unigenx_sequence_function/all.jsonl
```

Expected fields include `protein_sequence` or `sequence`, plus `function_description`, `function`, `description`, `annotation`, `summary`, `completion`, or `output`.

Supported `--kind` values include `pubchem`, `uniprot`, `uniprot_features`, `refseq`, `ncbi`, `materials_project`, `chembl`, `bindingdb`, `bioactivity`, `pdbbind`, `docking`, `complex_affinity`, `biomolecular_affinity`, `ppi_affinity`, `protein_na_affinity`, and `ec`. These create graph-token rows for molecules, proteins, UniProt sequence features, EC-number conditioning, protein-ligand docking, complex affinity, assay/bioactivity records, and materials.

Local audio feature extraction remains available for user-provided rows with `local_audio_path` or `audio_path`, but no external audio corpus is part of the active SFM/NatureLM or UniGenX pipeline.

Optional large NatureLM Hugging Face checkpoints remain explicit and are not pulled by the default GitHub metadata commands:

```bash
conda run -n tokengt python scripts/acquire_model_files.py --repo-name fairchem

conda run -n tokengt python scripts/download_uma_weights.py \
  --repo data/external_repos/fairchem \
  --model-name uma-s-1p2 \
  --task-name omol

conda run -n tokengt python scripts/acquire_model_files.py \
  --repo-name naturelm_8x7b_hf_checkpoint \
  --download \
  --pattern "README.md" \
  --max-file-mb 1
```

Use larger `--max-file-mb` and explicit `--pattern` values only when you intentionally want multi-GB checkpoint files.

### PLAN-H UGM Multimodal Graphs

Build the UGM reference-token extension and the actual motif vocabulary. This downloads public PROSITE, InterPro, CATH, and Rfam metadata into `data/raw_motifs/public/`, writes `data/processed/reference_tokens/motif_graph_tokens.txt`, and merges those motif tokens into `data/processed/reference_tokens/multimodal_graph_tokens.txt`:

```bash
conda run -n tokengt python scripts/build_multimodal_vocab.py \
  --download-public-motifs \
  --output data/processed/reference_tokens/multimodal_graph_tokens.txt
```

The current local full public motif build produced 74,789 motif records and 148,669 motif tokens: 55,644 sequence records, 10,987 structure records, and 8,158 structure-derived sequence records from core defaults, PROSITE, InterPro, CATH, and Rfam. Set `SKIP_INTERPRO_MOTIF_DOWNLOAD=1` only when intentionally doing a fast smoke run from already cached non-InterPro sources.

Create a neutral synthetic multimodal graph-to-graph smoke set:

```bash
conda run -n tokengt python scripts/prepare_multimodal_sources.py \
  --synthetic \
  --count 32 \
  --output data/processed/multimodal_graphs/all.jsonl

conda run -n tokengt python scripts/curate_data.py \
  --input data/processed/multimodal_graphs/all.jsonl \
  --output-dir data/processed/multimodal_graphs \
  --val-ratio 0.2 \
  --test-ratio 0.1
```

Prepare local JSON/JSONL/CSV/TSV/FASTA rows with first-run fields such as `protein_sequence`, `selfies`, `smiles`, `dna_sequence`, `rna_sequence`, `temperature`, `oracle`, `sequence_motifs`, `sequence_motifs_from_structure`, `function_description`, function labels, and assay/property annotations. Keep `atoms`, `bonds`, `frames`, `coordinates`, `energy`, `forces`, row-local `structure_motifs`, and row-local `structure_derived_sequence_motifs` out of training rows unless a later structure phase is explicitly enabled:

```bash
conda run -n tokengt python scripts/prepare_multimodal_sources.py \
  --input /path/to/mixed_rows.jsonl \
  --dataset-name local_multimodal_graph_to_graph \
  --molecular-input-policy sequence_only \
  --output data/processed/local_multimodal/train.jsonl \
  --limit 1000
```

## Train

Tiny graph pretraining:

```bash
conda run -n tokengt python scripts/train_stage.py \
  --config config/model/tiny_tokengt.yaml \
  --config config/data/synthetic_graphs.yaml \
  --config config/train/graph_pretrain_tiny.yaml
```

Tiny GFlowNet graph-of-thought stage:

```bash
conda run -n tokengt python scripts/train_stage.py \
  --config config/data/synthetic_graphs.yaml \
  --config config/train/gflownet_got_tiny.yaml
```

Tiny topology auxiliary stage:

```bash
conda run -n tokengt python scripts/train_stage.py \
  --config config/model/tiny_tokengt.yaml \
  --config config/data/synthetic_graphs.yaml \
  --config config/train/topology_aux_tiny.yaml
```

Graph-state evolution ablation for the first topology/persistence/tropical run. This compares baseline random-order graph decoding, graph/topology plus hidden-topology collapse guidance, tropical annealing, and the combined variant. It logs hidden-state distogram/H0-persistence proxies, graph topology summaries, tropical logit diagnostics, quality metrics, diversity proxies, and elapsed time per variant.

```bash
conda run -n tokengt python scripts/run_graph_state_ablation.py \
  --model-config config/model/tiny_tokengt.yaml \
  --data-config config/data/synthetic_graphs.yaml \
  --output outputs/graph_state_ablation/summary.json
```

Use `--dry-run` to print the exact variant commands without training. The graph-state implementation treats reasoning as an evolving graph with optional latent thought nodes and verifier/tool observations; chain-of-thought text is only a possible rendering of that state.

Optional Multi-Head Tropical Attention smoke stage. The pure backend replaces the self-attention kernel inside the compact TokenGT encoder with masked max-plus/Hilbert-projective MHTA. The training wrappers use the faster hybrid backend by default when `ENABLE_TROPICAL_ATTENTION=1`: every layer runs a FlashAttention-2 package path when the installed CUDA extension supports the mask/dtype, otherwise a PyTorch scaled-dot-product attention fallback, and only configured layers add an MHTA branch. `config/model/overrides/hybrid_flash_mhta_backend.yaml` defaults to `hybrid_tropical_layers: [-1]`, which means final-layer MHTA for any model depth, and `tropical_query_chunk_size: 96`, which preserves exact MHTA scores while lowering peak activation memory on 24GB GPUs. Use zero-based indices such as `[2, 5]` for a 6-layer model when you want a denser tropical path. The encoder logs `flash_attention/*`, `hybrid_attention/*`, and active-layer `tropical_attention/*` metrics to JSONL and W&B when W&B is enabled; `hybrid_attention/tropical_active` reports the fraction of hybrid layers that actually ran MHTA in that pass.

```bash
conda run -n tokengt python scripts/train_stage.py \
  --config config/model/tiny_tokengt_tropical.yaml \
  --config config/data/synthetic_graphs.yaml \
  --config config/train/graph_pretrain_tropical_attention_tiny.yaml
```

For an existing model config, the backend can also be enabled explicitly with:

```bash
conda run -n tokengt python scripts/train_stage.py \
  --config config/model/tiny_tokengt.yaml \
  --config config/train/overrides/tropical_attention_backend.yaml \
  --config config/data/synthetic_graphs.yaml \
  --config config/train/graph_pretrain_tiny.yaml
```

For the full selected public corpus, use the shell switch rather than editing YAML by hand. The switch appends `config/model/overrides/hybrid_flash_mhta_backend.yaml` by default, leaving the existing data, context, and training configs in place. On this 24GB RTX 4090, the current direct 250M sparse-hybrid default is compact context with micro-batch 2 and gradient accumulation 18. To force pure MHTA for ablation, set `TROPICAL_ATTENTION_CONFIG=config/model/overrides/tropical_attention_backend.yaml`.

```bash
ENABLE_TROPICAL_ATTENTION=1 \
FULL_TRAIN_BATCH_SIZE=2 \
FULL_TRAIN_EVAL_BATCH_SIZE=2 \
FULL_TRAIN_GRAD_ACCUM=18 \
./scripts/run_full_phase1_phase2_training_250m.sh
```

To jump directly to training after the corpus, context config, UMA weights, and vocab already exist:

```bash
ENABLE_TROPICAL_ATTENTION=1 ./scripts/train_full_selected_250m_direct.sh
```

Most time-efficient 250M command with both attention branches enabled:

```bash
ENABLE_TROPICAL_ATTENTION=1 \
FULL_TRAIN_BATCH_SIZE=2 \
FULL_TRAIN_EVAL_BATCH_SIZE=2 \
FULL_TRAIN_GRAD_ACCUM=18 \
SKIP_POLICY_CHECK=1 \
./scripts/train_full_selected_250m_direct.sh
```

Most complete strict oracle-dynamics command, with hybrid Flash/MHTA, UMA coordinate-force feedback, and contact-map/embedding-geometry alignment enabled:

```bash
./scripts/train_full_selected_250m_oracle_dynamics_direct.sh
```

Direct UniProt plus biomolecular-affinity training, with full local source preparation, graphification, curation, SFT, SFT-GFlowNet, and structure-dynamics GFlowNet enabled:

```bash
./scripts/run_full_biomed_annotations_affinity_training.sh
```

This uses `config/model/ugm_250m_tokengt.yaml`, `config/data/biomed_annotations_affinity_250m.yaml`, `config/train/biomed_annotations_affinity_250m.yaml`, `config/train/biomed_annotations_affinity_gflownet_sft_4090.yaml`, and `config/train/biomed_annotations_affinity_structure_dynamics_gflownet_4090.yaml`. The coordinate and internal-coordinate heads are model-stage overrides: they are enabled during the TokenGT SFT/model-training phase, where the model can actually emit coordinate readouts and receive UMA force feedback. The standalone SFT-GFlowNet and structure-dynamics GFlowNet phases train token-set construction policies over symbolic/contact/internal-coordinate/all-atom Cartesian candidate vocabularies. The full local source set is 2,411,356 data rows: 574,627 UniProt feature rows plus 1,836,729 biomolecular complex-affinity rows. Add `INCLUDE_ORIGINAL_FULL_SELECTED=1` to include the original 7,328,008-row selected public corpus in the same curated training dataset.

For the current bio-scale requirement, use the all-atom contact runner that adds million-row sequence sources per modality before launching the same 250M SFT plus two-GFlowNet stack:

```bash
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-bio-scale-all-atom-contact" \
BIO_SEQUENCE_TARGET_ROWS_PER_MODALITY=3000000 \
PROTEIN_SEQUENCE_TARGET_ROWS=3000000 \
STRUCTURE_DYNAMICS_TARGET_ROWS=2500 \
STATIC_STRUCTURE_TARGET_ROWS=25000 \
TRAIN_PHASES=all \
./scripts/run_bio_scale_all_atom_contact_training.sh
```

This runner downloads and graphifies the public protein/molecule/RNA/DNA sequence sources (`ConvergeBio/uniref50`, `PubChem10M_SELFIES`, UniProt function text, Rfam, RNAcentral 8192, and DNA coding regions), checks modality counts with `scripts/check_bio_scale_targets.py`, then passes those extra graph JSONL files into `scripts/run_all_atom_contact_biomed_retrain.sh`. The expected targets are at least 3M protein rows, 3M molecule SELFIES rows, and 3M RNA rows when the sources are available. DNA is requested at 3M but currently source-limited by the public coding-region split; the target check reports that explicitly. The structure-dynamics GFlowNet is trained on a dedicated 2,500-row dynamics subset, while static structure/contact prediction uses a larger 25,000-row subset by default. Set `PREPARE_PROTEIN_SCALE_REST=1` only if you intentionally want the slower UniProtKB REST feature stream for the 3M protein scale source; the default uses UniRef50 parquet for practical throughput and keeps the reviewed UniProt feature export for binding-site/function annotations.

If `data/local/uniprot_features.tsv` and `data/local/complex_affinity.tsv` already exist and you only want to rebuild graphification/curation before training:

```bash
PREPARE_FULL_BIOMED_SOURCES=0 \
PREPARE_UNIPROT=force \
PREPARE_AFFINITY=force \
CURATE_DATA=force \
TRAIN_PHASES=all \
./scripts/run_full_biomed_annotations_affinity_training.sh
```

If `data/processed/uniprot_features_local_export/all.jsonl` and `data/processed/biomolecular_complex_affinity_local/all.jsonl` already exist and you want to resume directly at fast curation plus training:

```bash
PREPARE_FULL_BIOMED_SOURCES=0 \
PREPARE_UNIPROT=0 \
PREPARE_AFFINITY=0 \
CURATE_DATA=force \
FAST_CURATE=1 \
TRAIN_PHASES=all \
./scripts/run_full_biomed_annotations_affinity_training.sh
```

To train on the original full selected corpus plus the new UniProt feature and biomolecular-affinity rows:

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

Fast curation should finish in about one hour on the current machine for the biomed-only corpus. The combined-original mode reads the 7.33M-row `real_full_selected_mix` corpus too, so budget several hours for curation and integrity checking before training. If curation is interrupted, rerun the same command with `RESUME_CURATE=1` and it will reuse `.train.jsonl.tmp`, `.val.jsonl.tmp`, `.test.jsonl.tmp`, and `.curate_resume_state.json` in the output directory. Resume now normalizes temporary split files before appending, so an interrupted write cannot concatenate two JSON objects onto one JSONL line. If an older interrupted run already produced that failure, repair the existing split files without rerunning curation:

```bash
conda run --no-capture-output -n tokengt python scripts/repair_jsonl_concatenation.py \
  --path data/processed/biomed_annotations_affinity_plus_original_full_selected/train.jsonl \
  --path data/processed/biomed_annotations_affinity_plus_original_full_selected/val.jsonl \
  --path data/processed/biomed_annotations_affinity_plus_original_full_selected/test.jsonl
```

After repair, rerun `scripts/check_dataset_integrity.py`; it reads curated `split_sizes` as well as full-corpus `counts`. Full SFT is one full epoch over the curated corpus, about 60k optimizer steps for biomed-only or about 240k optimizer steps for combined-original mode with the default effective batch of 36, followed by the two 3k-step GFlowNet phases.

The oracle-dynamics 250M wrapper above defaults to `FULL_TRAIN_BATCH_SIZE=1`, `FULL_TRAIN_GRAD_ACCUM=36`, `ENABLE_TROPICAL_ATTENTION=1`, `ENABLE_UMA_COORDINATE_HEAD=1`, and `EXTRA_TRAIN_CONFIGS+=config/train/overrides/uma_contact_geometry_loss.yaml`. It is intentionally conservative for a 24GB RTX 4090. This default coordinate-head path exposes short UMA coordinate-query windows and is the practical training default.

For full-size all-atom Cartesian structure-dynamics model training, use the long 8192-token BioSELFIES/coordinate override. This keeps supervised coordinate labels off, enables the coordinate and internal-coordinate heads for the model-training phase, exposes up to 8192 all-atom coordinate-query slots, and scores a tractable oracle subset per feedback call:

```bash
ENABLE_LONG_ALL_ATOM_CARTESIAN_HEAD=1 \
FULL_TRAIN_BATCH_SIZE=1 \
FULL_TRAIN_EVAL_BATCH_SIZE=1 \
FULL_TRAIN_GRAD_ACCUM=36 \
./scripts/train_full_selected_250m_oracle_dynamics_direct.sh
```

If the 250M model is too large at 8192 context on the local 4090, use the smaller long-context config without changing the data path:

```bash
ENABLE_LONG_ALL_ATOM_CARTESIAN_HEAD=1 \
MODEL_CONFIG=config/model/ugm_120m_tokengt_8192_selfies.yaml \
FULL_TRAIN_BATCH_SIZE=1 \
FULL_TRAIN_EVAL_BATCH_SIZE=1 \
FULL_TRAIN_GRAD_ACCUM=36 \
./scripts/train_full_selected_250m_oracle_dynamics_direct.sh
```

The same long all-atom override is available for the UniProt/affinity direct wrapper:

```bash
ENABLE_LONG_ALL_ATOM_CARTESIAN_HEAD=1 \
PREPARE_FULL_BIOMED_SOURCES=0 \
PREPARE_UNIPROT=0 \
PREPARE_AFFINITY=0 \
CURATE_DATA=0 \
INCLUDE_ORIGINAL_FULL_SELECTED=1 \
TRAIN_PHASES=sft \
./scripts/run_full_biomed_annotations_affinity_training.sh
```

The all-atom contact-template graph changes both source tokens and structure-dynamics target/action tokens. For that reason, reuse of the previous `20260503T204341Z` SFT checkpoint is not safe without explicit embedding and output-head resizing/remapping. The persisted contact template is compact by default: it keeps the 8192 BioSELFIES/source-token budget but caps emitted atom-template nodes at 64 unless `all_atom_template_max_atoms` is explicitly supplied on a row. Full-size 500-residue all-atom Cartesian outputs are generated by the inference/export path, not by writing every possible atom into every million-row training JSONL record. The retrain path should rebuild graphification and use a fresh vocab/output directory:

```bash
RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)-all-atom-contact
ENABLE_LONG_ALL_ATOM_CARTESIAN_HEAD=1 \
OUTPUT_DIR=outputs/biomed_annotations_affinity_plus_original_250m_all_atom_contact \
VOCAB_PATH=outputs/biomed_annotations_affinity_plus_original_250m_all_atom_contact/vocab.jsonl \
REUSE_VOCAB=false \
TRAIN_BATCH_SIZE=1 \
TRAIN_EVAL_BATCH_SIZE=1 \
TRAIN_GRAD_ACCUM=36 \
PREPARE_FULL_BIOMED_SOURCES=0 \
PREPARE_UNIPROT=force \
PREPARE_AFFINITY=force \
CURATE_DATA=force \
FAST_CURATE=1 \
RESUME_CURATE=1 \
INCLUDE_ORIGINAL_FULL_SELECTED=1 \
TRAIN_PHASES=all \
./scripts/run_full_biomed_annotations_affinity_training.sh
```

The current machine does not have `tmux`, so the full run is launched with `nohup`. The robust runner below writes newly graphified all-atom-contact biomed sources to separate output directories, then lets the existing direct trainer handle curation, integrity checking, SFT, SFT-GFlowNet, and structure-dynamics GFlowNet:

```bash
mkdir -p logs/biomed_direct_training
RUN_ID=20260513T003800Z-all-atom-contact
RUN_ID="$RUN_ID" \
PREPARE_UNIPROT=auto \
PREPARE_AFFINITY=auto \
CURATE_DATA=force \
TRAIN_PHASES=all \
nohup ./scripts/run_all_atom_contact_biomed_retrain.sh \
  > logs/biomed_direct_training/20260513T003800Z-all-atom-contact.nohup.log 2>&1 &
echo $! > logs/biomed_direct_training/20260513T003800Z-all-atom-contact.pid
```

To skip the multi-hour curation when a newly graphified and integrity-checked all-atom-contact corpus already exists, set `PREPARE_UNIPROT=0 PREPARE_AFFINITY=0 CURATE_DATA=0` and keep `REUSE_VOCAB=false` for the first all-atom-contact training run.

After a stable standard run, try:

```bash
FULL_TRAIN_BATCH_SIZE=2 FULL_TRAIN_EVAL_BATCH_SIZE=2 FULL_TRAIN_GRAD_ACCUM=18 \
./scripts/train_full_selected_250m_oracle_dynamics_direct.sh
```

If you want the full readiness/integrity/policy/UMA preflight before training, use:

```bash
./scripts/run_full_phase1_phase2_training_250m_oracle_dynamics.sh
```

Code, Lean, and chemistry vertical-slice SFT smoke stages:

```bash
conda run -n tokengt python scripts/train_stage.py \
  --config config/model/tiny_lora_checkpointed.yaml \
  --config config/data/code_graphs.yaml \
  --config config/train/code_sft_tiny.yaml

conda run -n tokengt python scripts/train_stage.py \
  --config config/model/tiny_lora_checkpointed.yaml \
  --config config/data/lean_graphs.yaml \
  --config config/train/lean_sft_tiny.yaml

conda run -n tokengt python scripts/train_stage.py \
  --config config/model/tiny_lora_checkpointed.yaml \
  --config config/data/chem_graphs.yaml \
  --config config/train/chem_sft_tiny.yaml
```

PLAN-D science SFT smoke stage:

```bash
conda run -n tokengt python scripts/train_stage.py \
  --config config/model/tiny_lora_checkpointed.yaml \
  --config config/data/science_mix.yaml \
  --config config/train/science_sft_tiny.yaml
```

UGM uses a single autoregressive graph-token decoder for symbolic and structure-candidate records. Generated frame coordinates are first discretized into identity-bearing target records such as `COORD:f0:a17:x:pos_near`, so frame, atom slot, axis, and coordinate bin are predicted through the same random-order `<POS>` objective as text, SELFIES, proof, tool, and oracle records. The optional continuous coordinate head is configured for UMA feedback rather than supervised structure labels: when `model.coordinate_head_enabled=true`, source-side `UMA_COORD_QUERY:*` slots receive continuous `(x,y,z)` proposals, and `loss.uma_coordinate_oracle_weight` trains those proposals from UMA energy/force feedback evaluated on the generated candidates. `loss.coordinate_loss_weight` remains `0.0` in the provided overrides, so no PDB/SDF/mmCIF/MD coordinate targets are used. The head is a geometry readout from embedding-space GoT reasoning; it does not replace random-order graph decoding.

BioSELFIES-only rows are available for the strict symbolic-input ablation described in `assets/main.tex`. Set `input_representation: bioselfies` or `bioselfies_only: true` on raw multimodal rows; if no `bioselfies` string is supplied, graphification serializes non-structural `protein_sequence`, `dna_sequence`, `rna_sequence`, and `selfies` fields into bracketed BioSELFIES tokens. The resulting graph still feeds the same UMA coordinate-query path: amino-acid BioSELFIES components create residue nodes, and those residue nodes create coarse backbone `UMA_COORD_QUERY:N/C/C/O` slots for oracle-scored generated coordinates.

UMA coordinate-head force-feedback override:

```bash
ENABLE_TROPICAL_ATTENTION=1 \
ENABLE_UMA_COORDINATE_HEAD=1 \
SKIP_POLICY_CHECK=1 \
./scripts/train_full_selected_250m_direct.sh
```

For a manual invocation, add `--config config/train/overrides/uma_coordinate_head.yaml` after the base train config and before the W&B override. Use `config/train/overrides/coordinate_head.yaml` only to enable the head with supervised coordinate NLL still off.

PLAN-H UGM multimodal phase-2 4090 stage:

```bash
conda run -n tokengt python scripts/train_stage.py \
  --config config/model/max_4090_tokengt.yaml \
  --config config/data/multimodal_graphs_4090.yaml \
  --config config/train/multimodal_phase2_4090.yaml
```

PLAN-H oracle-feedback GFlowNet 4090 stage:

```bash
conda run -n tokengt python scripts/train_stage.py \
  --config config/data/multimodal_graphs_4090.yaml \
  --config config/train/multimodal_oracle_gflownet_4090.yaml
```

Separate SFT and structure-dynamics GFlowNet stages:

```bash
conda run -n tokengt python scripts/train_stage.py \
  --config config/data/multimodal_graphs_4090.yaml \
  --config config/train/gflownet_sft_4090.yaml

conda run -n tokengt python scripts/train_stage.py \
  --config config/data/multimodal_graphs_4090.yaml \
  --config config/train/structure_dynamics_gflownet_4090.yaml
```

The SFT GFlowNet keeps broad symbolic target-token candidates for function, tool, text, math, molecule, UniProt, and assay rows. The structure-dynamics GFlowNet narrows candidates to `INTERNAL_COORD:*`, `ADAPTIVE_PATCH:*`, `CONTACT_PATCH:*`, temperature, token-motion, and UMA/oracle records, so it trains graph-state construction for contact maps, adaptive patches, internal-coordinate proposals, and oracle-conditioned dynamics instead of generic text completion.

Structure/dynamics graph-to-graph phase is disabled for the first run. Do not run these training configs unless a later explicit structure-file phase is approved; the active structure-dynamics proxy path is the sequence-only multimodal/GFlowNet stage above.

```bash
# future-phase only:
# ENABLE_STRUCTURE_TRAINING=1 conda run -n tokengt python scripts/train_stage.py \
#   --config config/model/max_4090_tokengt.yaml \
#   --config config/data/structure_dynamics_graphs.yaml \
#   --config config/train/structure_dynamics_4090.yaml
```

PLAN-E Hebrew SFT and root-extension GFlowNet smoke stages:

```bash
conda run -n tokengt python scripts/train_stage.py \
  --config config/model/tiny_lora_checkpointed.yaml \
  --config config/data/hebrew_mix.yaml \
  --config config/train/hebrew_sft_tiny.yaml

conda run -n tokengt python scripts/train_stage.py \
  --config config/data/hebrew_roots.yaml \
  --config config/train/hebrew_root_gflownet_tiny.yaml
```

For 4090-scale experiments, start from:

```bash
config/model/max_4090_tokengt.yaml
config/model/ugm_250m_tokengt.yaml
```

With `max_vocab_size: 262144`, this profile is 576,767,128 trainable parameters with `max_seq_len: 1024`, `hidden_dim: 1024`, `num_layers: 24`, `num_heads: 16`, and `ffn_dim: 4096`.

The 250M UGM profile keeps `max_seq_len: 1024`, `max_nodes: 1024`, and `max_slots: 256`, with structure-candidate coordinates represented as autoregressive graph tokens. It uses `hidden_dim: 768`, `num_layers: 6`, `num_heads: 12`, and `ffn_dim: 3072`.

The ready-to-run 4090 configs are:

```bash
config/train/science_sft_4090.yaml
config/train/hebrew_sft_4090.yaml
config/train/gflownet_got_4090.yaml
config/train/gflownet_sft_4090.yaml
config/train/hebrew_root_gflownet_4090.yaml
config/train/multimodal_phase2_4090.yaml
config/train/multimodal_oracle_gflownet_4090.yaml
config/train/structure_dynamics_gflownet_4090.yaml
config/train/structure_dynamics_4090.yaml                # disabled future-phase gate
config/train/structure_dynamics_oracle_gflownet_4090.yaml # disabled future-phase gate
```

Use `planning/TRAINING-SEQUENCE.md` for the full start-to-finish training command sequence. Use `planning/RUNBOOK-4090.md` for additional 4090 scaling, W&B, validation, and inference notes. Check `planning/LICENSE-REVIEW.md` before increasing dataset limits.

## Validate

Config-driven validation:

```bash
conda run -n tokengt python scripts/validate_stage.py \
  --config config/validate/domain_validation.yaml \
  --device cpu
```

PLAN-D science validation:

```bash
conda run -n tokengt python scripts/validate_stage.py \
  --config config/validate/science_validation.yaml \
  --device cpu
```

PLAN-E Hebrew validation:

```bash
conda run -n tokengt python scripts/validate_stage.py \
  --config config/validate/hebrew_validation.yaml \
  --device cpu

conda run -n tokengt python scripts/validate_gflownet.py \
  --config config/data/hebrew_roots.yaml \
  --config config/train/hebrew_root_gflownet_tiny.yaml \
  --checkpoint outputs/hebrew_root_gflownet_tiny/gflownet_final.pt \
  --data data/processed/hebrew_root_synthetic/train.jsonl \
  --device cpu \
  --output outputs/hebrew_root_gflownet_tiny/validation.json
```

PLAN-H UGM multimodal validation:

```bash
conda run -n tokengt python scripts/validate_stage.py \
  --config config/validate/multimodal_4090_validation.yaml \
  --device cpu

conda run -n tokengt python scripts/validate_stage.py \
  --config config/validate/multimodal_4090_test.yaml \
  --device cpu

conda run -n tokengt python scripts/validate_gflownet.py \
  --config config/data/multimodal_graphs_4090.yaml \
  --config config/train/multimodal_oracle_gflownet_4090.yaml \
  --config config/validate/multimodal_4090_gflownet_validation.yaml \
  --device cpu
```

Structure/dynamics validation:

```bash
conda run -n tokengt python scripts/validate_stage.py \
  --config config/validate/structure_dynamics_validation.yaml \
  --device cpu

conda run -n tokengt python scripts/validate_stage.py \
  --config config/validate/structure_dynamics_test.yaml \
  --device cpu

conda run -n tokengt python scripts/validate_gflownet.py \
  --config config/data/structure_dynamics_graphs.yaml \
  --config config/train/structure_dynamics_oracle_gflownet_4090.yaml \
  --config config/validate/structure_dynamics_gflownet_validation.yaml \
  --device cpu
```

Full selected-corpus validation after the full checkpoint exists:

```bash
conda run -n tokengt python scripts/validate_stage.py \
  --config config/validate/real_full_selected_validation.yaml \
  --device cpu
```

Full selected-corpus test check:

```bash
conda run -n tokengt python scripts/validate_stage.py \
  --config config/validate/real_full_selected_test.yaml \
  --device cpu
```

The full selected-corpus validation/test configs point at `data/processed/real_full_selected_mix/` and the full checkpoint under `outputs/real_full_selected_local/`.

Repeatable quality assessment:

```bash
conda run -n tokengt python scripts/quality_assess.py
```

Direct validation:

```bash
conda run -n tokengt python scripts/validate_stage.py \
  --checkpoint outputs/synthetic_graph_pretrain/checkpoint_final.pt \
  --vocab outputs/synthetic_graph_pretrain/vocab.jsonl \
  --data data/processed/synthetic_graphs/train.jsonl \
  --batch-size 8 \
  --device cpu
```

Validate a saved GFlowNet rollout policy:

```bash
conda run -n tokengt python scripts/validate_gflownet.py \
  --config config/train/gflownet_got_tiny.yaml \
  --config config/validate/gflownet_validation.yaml \
  --device cpu
```

GFlowNet configs now support:

- `gflownet.mode: sft`: broad symbolic target-token graph completion;
- `gflownet.mode: structure_dynamics`: oracle/contact/internal-coordinate/adaptive-patch target filtering;
- `gflownet.use_context`: topology-conditioned policy inputs;
- `gflownet.learn_backward_policy`: learned backward policy instead of uniform reverse actions;
- `gflownet.subtrajectory_weight`: auxiliary subtrajectory-balance loss.

The saved checkpoint records the candidate token set, context dimension, forward policy, optional backward policy, trajectory-balance state, and subtrajectory-balance state.

## Inference

```bash
conda run -n tokengt python scripts/infer.py \
  --config config/inference/science_tiny_inference.yaml \
  --text "Create a graph reasoning sketch for a protein-to-molecule design task." \
  --max-steps 8 --retries 2 \
  --device cpu
```

UGM multimodal inference smoke:

```bash
conda run -n tokengt python scripts/infer.py \
  --config config/inference/multimodal_4090_inference.yaml \
  --prompt "Generate graph records for a mixed protein and ligand input." \
  --protein-sequence "MKTW" \
  --selfies "[C][=O][O]" \
  --dna-sequence "ATGC" \
  --temperature-k 315.5 \
  --max-steps 8 \
  --device cpu
```

The multimodal inference path also accepts `--multimodal-json` and `--multimodal-json-file /path/to/row.json`. For the first run, keep these rows sequence-first and avoid `atoms`, `bonds`, `frames`, `energy`, and `forces`.

Structure/dynamics inference is validation-only unless `ENABLE_STRUCTURE_TRAINING=1` is set deliberately:

```bash
conda run -n tokengt python scripts/infer.py \
  --config config/inference/structure_dynamics_inference.yaml \
  --multimodal-json-file path/to/example_structure_dynamics_row.json \
  --render-input-pdb \
  --device cpu \
  --output outputs/structure_dynamics_4090/infer_structure_dynamics_sequence.json
```

Full selected-corpus checkpoint inference:

```bash
conda run -n tokengt python scripts/infer.py \
  --config config/inference/real_full_selected_inference.yaml \
  --text "Create a graph reasoning sketch for a protein ligand binding question." \
  --max-steps 8 \
  --device cpu \
  --output outputs/real_full_selected_local/infer_smoke.json
```

Run this after the full checkpoint has been produced.

## Ready-To-Roll Status

Current UGM smoke status:

- Data split: 23 train, 5 validation, 4 test examples under `data/processed/multimodal_graphs/`.
- Phase-2 checkpoint: `outputs/multimodal_phase2_tiny/checkpoint_final.pt`.
- Oracle-feedback GFlowNet checkpoint: `outputs/multimodal_oracle_gflownet_tiny/gflownet_final.pt`.
- File-based inference smoke output: `outputs/multimodal_phase2_tiny/infer_file_smoke.json`.
- Full selected public graph corpus target: `data/processed/real_full_selected_mix/` with expected completed counts of 7,181,690 train, 73,044 validation, and 73,274 test graph examples; `scripts/check_dataset_integrity.py` must pass before training.
- Full selected-corpus training target: `outputs/real_full_selected_local/checkpoint_final.pt`.
- Structure/dynamics future-phase targets are intentionally absent unless `ENABLE_STRUCTURE_TRAINING=1` is approved and documented.
- `conda run -n tokengt pytest -q`: 40 passed, 3 warnings.
- `conda run -n tokengt python scripts/quality_assess.py`: `ready_to_roll: true`.
- `conda run -n tokengt python scripts/check_readiness.py --json`: CUDA, Lean, RDKit, topology packages, SFM/NatureLM references, UniGenX references, and reference tokens are present.

The existing tiny checkpoints validate functionality only. Useful scientific behavior requires the full selected-corpus training run, reviewed local/gated data where applicable, a production tokenizer, real oracle integrations, constrained graph decoding, and benchmark evaluation. See `planning/REAL-DATA-TRAINING-STATUS.md` for the real-data execution log and remaining scale limits.

## Metrics

Metrics are documented in `planning/METRICS.md`.

W&B is disabled in tiny configs. To enable it:

```bash
conda run -n tokengt python scripts/train_stage.py \
  --config config/model/tiny_lora_checkpointed.yaml \
  --config config/data/science_mix.yaml \
  --config config/train/science_sft_tiny.yaml \
  --config config/train/overrides/wandb_online.yaml
```

Use `config/train/overrides/wandb_offline.yaml` for local dry runs. The 4090 configs enable W&B online by default. Every run also writes `metrics.jsonl` into its output directory.

## Tests

```bash
conda run -n tokengt pytest -q
```

Current smoke coverage:

- graph schema round trip;
- random-order collator labels and masks;
- tiny model forward/backward;
- topology/tropical/verifier diagnostics;
- curation dedup/split logic;
- GFlowNet trajectory-balance backward pass.
- SFT and structure-dynamics GFlowNet candidate filtering/reward paths.
- code/Lean/chemistry domain adapters;
- local LoRA plus checkpointing forward/backward.
- SFM/NatureLM and UniGenX reference-token extraction plus science metrics.
- Hebrew morphology/root graphification, UD CoNLL-U parsing, Verb Complements TSV support, root-extension graphs, and collator target-slot reservation.
- PLAN-F advanced topology/tropical utilities, autoregressive coordinate/property graph tokens, protein/docking/bioactivity graphifiers, audio feature extraction, stronger curation filters, and context/backward/subtrajectory GFlowNet.
- PLAN-H UGM multimodal vocabulary, graphification, optional input-row PDB rendering, numeric extraction, oracle-feedback reward, and collator coverage.
- Motif vocabulary parsing for PROSITE, InterPro, CATH, Rfam, local motif rows, and structure-derived sequence motifs from atom/frame rows.
- UGM scientific/oracle random-order policies and multimodal inference/QA wiring.
- UniProt feature/binding-site graphification, biomolecular-complex affinity graphification, internal-coordinate UMA action slots, UMA internal-coordinate oracle proxy loss, BioSELFIES conversion for biomed rows, all-atom Cartesian structure-dynamics candidate tokens, and legacy-corpus fallback derivation for structure-dynamics GFlowNet candidates.
- readiness probe for optional packages, Lean, CUDA, and reference data.

## Contact-Prior And OMG/gLM2 Preparation

Structure-dynamics training can now add two sequence-only contact-prior families before curation. These are still strict symbolic inputs: they add predicted contact records and candidate tokens, not coordinate labels.

ESM-style protein contact priors follow the public ESM notebook pattern: load an ESM2 model, run `model.predict_contacts(batch_tokens)`, cache the contact matrix, and graphify top residue-pair priors. Precompute and augment an existing protein JSONL like this:

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

For inference, enable the same contact prior directly from `scripts/infer.py`:

```bash
conda run -n tokengt python scripts/infer.py \
  --config config/inference/biomed_annotations_affinity_plus_original_250m_inference.yaml \
  --output-modality structure_dynamics \
  --task structure_dynamics_proxy \
  --protein-sequence "MKTWYV" \
  --temperature-k 325 \
  --esm-contact-prior \
  --esm-contact-device cuda \
  --trajectory-frames 32 \
  --trajectory-max-atoms 512 \
  --structure-output-prefix outputs/inference/structure_dynamics/esm_contact_example \
  --device cuda \
  --output outputs/inference/structure_dynamics/esm_contact_example.json
```

OMG/gLM2-style mixed metagenomic context preparation is available through a diverse subsampler. It keeps CDS amino-acid segments, intergenic nucleotide segments, strand/order metadata, and optional categorical-Jacobian contact records. Use local OMG JSONL when available, or stream a bounded scan from Hugging Face:

```bash
conda run -n tokengt python scripts/prepare_omg_subsample.py \
  --target-rows 20000 \
  --scan-limit 500000 \
  --require-intergenic \
  --output data/processed/omg_diverse_intergenic/raw.jsonl \
  --graph-output data/processed/omg_diverse_intergenic/all.jsonl
```

Cached categorical-Jacobian contact matrices or contact lists can be converted into graph rows:

```bash
conda run -n tokengt python scripts/build_categorical_jacobian_contacts.py \
  --input data/local/glm2_jacobian/example_contacts.json \
  --output data/processed/glm2_jacobian_contacts/example_contacts.json \
  --top-k 512 \
  --min-score 0.05
```

Protein-protein contact training should include affinity data when available. `graphify_biomolecular_complex_affinity` converts `Kd`, `Ki`, `IC50`, `kon`, `koff`, and `dG` fields into `AFFINITY_CONTACT:*`, `PPI_CONTACT:*`, and `CONTACT_PATCH:affinity_weighted_interface` tokens so the structure-dynamics GFlowNet can prioritize contact paths that are consistent with measured complex strength. The same graphifier handles multimodal interactions such as protein-DNA-ligand rows: protein, nucleic-acid, and ligand components all receive component SELFIES/BioSELFIES views, all-atom contact-template nodes, and affinity-weighted `COMPLEX_CONTACT:*` candidates.

## Current Structure-Dynamics Restart Command

The completed SFT and SFT-GFlowNet phases under run `20260503T204341Z` do not need to be repeated. The structure-dynamics GFlowNet failure happened before training, during candidate construction, so the restart command is the failed phase only:

```bash
RUN_ID=20260512T202638Z \
conda run --no-capture-output -n tokengt python scripts/train_stage.py \
  --config config/data/biomed_annotations_affinity_250m.yaml \
  --config logs/biomed_direct_training/20260503T204341Z/biomed_annotations_affinity_data_override.yaml \
  --config config/train/biomed_annotations_affinity_structure_dynamics_gflownet_4090.yaml \
  --config logs/biomed_direct_training/20260503T204341Z/biomed_annotations_affinity_structure_dynamics_gflownet_override.yaml \
  --config config/train/overrides/wandb_online.yaml
```

For the newer all-atom contact-template implementation, retrain from a fresh vocab as shown above rather than using the old checkpoint.

## Latest All-Atom Contact Verification

Commands run after enabling atom-node plus bond-edge contact templates:

```bash
conda run --no-capture-output -n tokengt pytest -q tests/test_multimodal_graphs.py tests/test_gflownet_smoke.py

rm -rf /tmp/iska_all_atom_contact_smoke && mkdir -p /tmp/iska_all_atom_contact_smoke
conda run --no-capture-output -n tokengt python scripts/prepare_multimodal_sources.py \
  --synthetic --count 8 \
  --dataset-name all_atom_contact_smoke \
  --output /tmp/iska_all_atom_contact_smoke/all.jsonl
conda run --no-capture-output -n tokengt python scripts/curate_data.py \
  --input /tmp/iska_all_atom_contact_smoke/all.jsonl \
  --output-dir /tmp/iska_all_atom_contact_smoke/curated \
  --val-ratio 0.25 \
  --test-ratio 0.0 \
  --split-policy row_hash
conda run --no-capture-output -n tokengt python scripts/train_stage.py \
  --config config/model/tiny_tokengt.yaml \
  --config config/data/multimodal_graphs.yaml \
  --config config/train/multimodal_phase2_tiny.yaml \
  --config /tmp/iska_all_atom_contact_smoke/train_override.yaml

conda run --no-capture-output -n tokengt pytest -q
```

Result: focused multimodal/GFlowNet tests passed, the tiny two-step training smoke completed, and the full test suite passed (`115 passed, 23 warnings`).

Additional compatibility checks after adding multimodal affinity fallback SELFIES and all-atom trajectory export:

```bash
conda run --no-capture-output -n tokengt pytest -q tests/test_multimodal_graphs.py tests/test_gflownet_smoke.py

conda run --no-capture-output -n tokengt python scripts/train_stage.py \
  --config config/model/tiny_tokengt.yaml \
  --config config/data/multimodal_graphs.yaml \
  --config config/train/multimodal_phase2_tiny.yaml \
  --config config/train/overrides/coordinate_head.yaml \
  --config /tmp/iska_structure_export_smoke/train_override.yaml

conda run --no-capture-output -n tokengt python scripts/infer.py \
  --checkpoint /tmp/iska_structure_export_smoke/output/checkpoint_final.pt \
  --vocab /tmp/iska_structure_export_smoke/output/vocab.jsonl \
  --output-modality structure_dynamics \
  --task structure_dynamics_proxy \
  --prompt "Generate UMA-scored all-atom Cartesian structure-dynamics records." \
  --protein-sequence "MKTWYV" \
  --selfies "[C][=O][O]" \
  --dna-sequence "ATGCGTAC" \
  --temperature-k 325 \
  --trajectory-frames 4 \
  --trajectory-max-atoms 64 \
  --trajectory-formats dcd,xyz \
  --trajectory-oracle-backend proxy \
  --structure-output-prefix /tmp/iska_structure_export_smoke/infer/example \
  --max-steps 16 \
  --max-source-tokens 512 \
  --device cuda \
  --output /tmp/iska_structure_export_smoke/infer/example.json

conda run --no-capture-output -n tokengt pytest -q
```

Result: focused tests passed (`35 passed`); structure-dynamics inference wrote `.pdb`, `.dcd`, `.xyz`, and `.json`; the PDB contains `MODEL`, `ENDMDL`, `HELIX`, `CONECT`, and `REMARK 902 VIEWER REPRESENTATION: CARTOON RECOMMENDED`; the full test suite passed (`116 passed, 23 warnings`).

## Key Files

- `planning/PLAN-A.md`: implementation plan and research decisions.
- `planning/BACKGROUND-RESEARCH.md`: source and dataset research notes.
- `planning/PLAN-D.md`: earlier science-data integration plan, superseded for NatureLM/UniGenX source correction by PLAN-G.
- `planning/PLAN-G.md`: corrected SFM/NatureLM and UniGenX GitHub integration.
- `planning/PLAN-H.md`: UGM multimodal graph-to-graph and oracle-feedback integration.
- `planning/BIOMOLECULAR-ORACLE-DYNAMICS-PLAN.md`: implementation map for the BioSELFIES, sequence-only, UMA coordinate-force, contact-map, and GFlowNet approach in the addendum.
- `planning/ORACLE-DYNAMICS-TRAINING-RUNBOOK.md`: concrete commands, config stack, W&B metrics, and function-readiness checklist for the addendum training path.
- `planning/MULTIMODAL-BIO-LM-DATASET-UTILIZATION-PLAN.md`: LucaOne, ProTrek, BioT5+, OneProt, UniProt feature, and biomolecular-complex affinity data utilization plan.
- `planning/OMG-GLM2-ESM-CONTACT-INTEGRATION.md`: ESM contact-prior, OMG/gLM2 intergenic-context, categorical-Jacobian, affinity-contact, and 8192 BioSELFIES integration notes.
- `planning/TROPICAL-ATTENTION-INTEGRATION-PLAN.md`: source audit, mathematical contract, config toggles, hyperparameter guidance, metrics, risks, and validation criteria for optional MHTA training.
- `planning/UGM-SYNTHETIC-FAUX-CODE-AUDIT.md`: first-party synthetic/proxy/faux-path inventory and replacement plan.
- `planning/FULL-PRETRAINING-DATASET.md`: complete selected public pretraining corpus, token counts, per-dataset totals, and completeness boundary.
- `planning/TRAINING-SEQUENCE.md`: ordered commands for readiness, data prep, full selected graph pretraining, follow-on stages, validation, inference, and final QA.
- `planning/REAL-DATA-TRAINING-STATUS.md`: real-data audit, implementation plan, training run, validation/test/inference results, and ready-to-roll status.
- `planning/PLAN-E.md`: Hebrew morphology, shoresh, and root-extension reasoning integration.
- `planning/PLAN-F.md`: deferred-component closure plan.
- `planning/ARCHITECTURE.md`: model/training architecture.
- `planning/DATASETS.md`: acquisition and curation protocol.
- `planning/LICENSE-REVIEW.md`: dataset/source scale policy and provenance checklist.
- `planning/RUNBOOK-4090.md`: 4090 training, validation, W&B, and inference runbook.
- `planning/METRICS.md`: metric namespaces and meanings.
- `assets/main.tex`: oracle-guided biomolecular dynamics addendum, including BioSELFIES/hybrid-tokenization and no-structure-training boundaries.
- `src/iska_reasoner/models/random_order_tokengt.py`: model.
- `src/iska_reasoner/data/dataset.py`: random-order collator.
- `src/iska_reasoner/topology/`: graph topology and distogram-style summaries.
- `src/iska_reasoner/tropical/`: annealing, logit selection diagnostics, masked MHTA, and tropical transformer encoder utilities.
- `src/iska_reasoner/oracles/`: live external oracle adapters, including FairChem/UMA.
- `src/iska_reasoner/inference/contact_priors.py`: optional ESM contact-prior inference/cache helpers.
- `src/iska_reasoner/inference/categorical_jacobian.py`: categorical-Jacobian contact helper utilities.
- `scripts/prepare_science_sources.py`: local PubChem/UniProt/RefSeq/Materials/ChEMBL/BindingDB/PDBbind/EC preparation.
- `scripts/prepare_multimodal_sources.py`: local/synthetic UGM multimodal graph-record preparation.
- `scripts/audit_dataset_capacity.py`: manifest size and local capacity audit.
- `scripts/download_hf_selected_splits.py`: full public HF selected-split parquet downloader.
- `scripts/graphify_full_parquet_manifest.py`: streaming graphification for downloaded HF parquet snapshots.
- `scripts/build_multimodal_vocab.py`: UGM multimodal reference-token writer.
- `scripts/check_uma_oracle.py`: FairChem/UMA clone/import/scoring readiness check.
- `scripts/download_uma_weights.py`: FairChem/UMA gated checkpoint and reference-table download/verification.
- `scripts/build_esm_contact_priors.py`: ESM2 contact-prior cache/JSONL augmentation for structure-dynamics graph rows.
- `scripts/prepare_omg_subsample.py`: diverse OMG/gLM2-style CDS/intergenic mixed-modality subsampling and graphification.
- `scripts/build_categorical_jacobian_contacts.py`: categorical-Jacobian contact-list/matrix conversion for graph contact priors.
- `scripts/build_motif_vocab.py`: standalone sequence, structure, and structure-derived sequence motif vocabulary builder.
- `scripts/quality_assess.py`: repeatable UGM ready-to-roll assessment.
- `src/iska_reasoner/data/multimodal.py`: multimodal graphifier, vocabulary families, and PDB renderer.
- `src/iska_reasoner/data/motifs.py`: public/local motif parsers, motif tokenization, and structure-derived sequence motif extraction.
- `scripts/extract_reference_tokens.py`: SFM/NatureLM and UniGenX token extraction for extra vocabulary.
- `scripts/check_readiness.py`: local environment and data readiness probe.
- `src/iska_reasoner/data/reference_repos.py`: GitHub reference-repo parsing helpers.
- `scripts/train_qlora_external.py`: opt-in external QLoRA dependency gate.
- `scripts/probe_upstream_tokengt.py`: upstream TokenGT/Fairseq readiness probe.
- `src/iska_reasoner/tools/`: verifier adapters.
- `src/iska_reasoner/data/curate.py`: dedup, split, and curation scoring.
- `src/iska_reasoner/gflownet/`: trajectory-balance stage.

## Inference Quick Reference

All inference commands use `scripts/infer.py`. Pass either a config with `inference.checkpoint` and `inference.vocab`, or pass `--checkpoint` and `--vocab` directly. Use `--device cuda` for trained 4090 checkpoints and `--device cpu` only for tiny smoke checkpoints.

Text or general graph reasoning:

```bash
conda run -n tokengt python scripts/infer.py \
  --config config/inference/real_full_selected_250m_inference.yaml \
  --text "Create a graph reasoning sketch for a protein ligand binding question." \
  --max-steps 16 \
  --device cuda \
  --output outputs/inference/text_graph_reasoning.json
```

Raw graph JSON:

```bash
conda run -n tokengt python scripts/infer.py \
  --config config/inference/real_full_selected_250m_inference.yaml \
  --graph-json-file data/local/inference/example_graph.json \
  --max-steps 32 \
  --device cuda \
  --output outputs/inference/graph_json_completion.json
```

Protein sequence and function-description modality:

```bash
conda run -n tokengt python scripts/infer.py \
  --config config/inference/multimodal_4090_inference.yaml \
  --task function_description \
  --prompt "Generate graph records and a function hypothesis for this protein." \
  --protein-sequence "MKTWYV" \
  --temperature-k 315 \
  --max-steps 32 \
  --device cuda \
  --output outputs/inference/protein_function.json
```

Small-molecule SELFIES or SMILES modality:

```bash
conda run -n tokengt python scripts/infer.py \
  --config config/inference/multimodal_4090_inference.yaml \
  --task molecule_reasoning \
  --prompt "Generate graph records for this molecule." \
  --selfies "[C][=O][O]" \
  --smiles "CC(=O)O" \
  --temperature-k 325 \
  --max-steps 32 \
  --device cuda \
  --output outputs/inference/molecule_graph.json
```

DNA or RNA modality:

```bash
conda run -n tokengt python scripts/infer.py \
  --config config/inference/multimodal_4090_inference.yaml \
  --task sequence_annotation \
  --prompt "Generate graph records for nucleic-acid sequence reasoning." \
  --dna-sequence "ATGCGTAC" \
  --rna-sequence "AUGCGUAC" \
  --temperature-k 310 \
  --max-steps 32 \
  --device cuda \
  --output outputs/inference/nucleic_acid_graph.json
```

Mixed multimodal row, including BioSELFIES input:

```bash
conda run -n tokengt python scripts/infer.py \
  --config config/inference/multimodal_4090_inference.yaml \
  --multimodal-json-file data/local/inference/mixed_bioselfies_row.json \
  --max-steps 48 \
  --device cuda \
  --output outputs/inference/mixed_multimodal_graph.json
```

Structure-dynamics output modality:

```bash
conda run -n tokengt python scripts/infer.py \
  --config config/inference/multimodal_4090_inference.yaml \
  --output-modality structure_dynamics \
  --task structure_dynamics_proxy \
  --prompt "Generate UMA-scored all-atom Cartesian structure-dynamics records." \
  --protein-sequence "MKTWYV" \
  --selfies "[C][=O][O]" \
  --temperature-k 325 \
  --trajectory-frames 16 \
  --trajectory-max-atoms 64 \
  --trajectory-formats dcd,xyz \
  --trajectory-oracle-backend fairchem \
  --structure-output-prefix outputs/inference/structure_dynamics/example \
  --max-steps 64 \
  --device cuda \
  --output outputs/inference/structure_dynamics/example.json
```

When `--output-modality structure_dynamics` is used, the CLI now writes:

- `outputs/inference/structure_dynamics/example.pdb`: multi-model PDB with one `MODEL`/`ENDMDL` block per generated frame.
- `outputs/inference/structure_dynamics/example.dcd`: MD-style trajectory written through MDTraj, suitable for tools that accept PDB topology plus DCD coordinates.
- `outputs/inference/structure_dynamics/example.xyz`: portable text trajectory for quick inspection and fallback tooling.
- `outputs/inference/structure_dynamics/example.json`: generated graph tokens, verifier metrics, and trajectory export metadata.

Use strict FairChem/UMA behavior when you want missing UMA weights or force rollout failures to stop the run:

```bash
conda run -n tokengt python scripts/infer.py \
  --config config/inference/multimodal_4090_inference.yaml \
  --output-modality structure_dynamics \
  --task structure_dynamics_proxy \
  --protein-sequence "MKTWYV" \
  --temperature-k 325 \
  --trajectory-oracle-backend fairchem \
  --trajectory-strict-oracle \
  --structure-output-prefix outputs/inference/structure_dynamics/strict_uma_example \
  --device cuda \
  --output outputs/inference/structure_dynamics/strict_uma_example.json
```

For local smoke tests without gated UMA weights, use the deterministic proxy oracle explicitly:

```bash
conda run -n tokengt python scripts/infer.py \
  --config config/inference/multimodal_tiny_inference.yaml \
  --output-modality structure_dynamics \
  --task structure_dynamics_proxy \
  --protein-sequence "MKTW" \
  --temperature-k 325 \
  --trajectory-frames 4 \
  --trajectory-formats dcd,xyz \
  --trajectory-oracle-backend proxy \
  --structure-output-prefix outputs/inference/structure_dynamics/proxy_smoke \
  --device cpu \
  --output outputs/inference/structure_dynamics/proxy_smoke.json
```

The structure-dynamics files are generated model/oracle artifacts. They are not supervised PDB/SDF/mmCIF/MD labels copied from training data.

## Latest All-Atom Contact Train Command

The verification train command run after the all-atom contact-template update was:

```bash
conda run --no-capture-output -n tokengt python scripts/train_stage.py \
  --config config/model/tiny_tokengt.yaml \
  --config config/data/multimodal_graphs.yaml \
  --config config/train/multimodal_phase2_tiny.yaml \
  --config /tmp/iska_all_atom_contact_smoke/train_override.yaml
```

For the real full retrain, do not resume the old `20260503T204341Z` SFT checkpoint unless embedding/head resizing is implemented. Rebuild graphification and train with a fresh vocab/output directory:

```bash
ENABLE_LONG_ALL_ATOM_CARTESIAN_HEAD=1 \
OUTPUT_DIR=outputs/biomed_annotations_affinity_plus_original_250m_all_atom_contact \
VOCAB_PATH=outputs/biomed_annotations_affinity_plus_original_250m_all_atom_contact/vocab.jsonl \
REUSE_VOCAB=false \
PREPARE_FULL_BIOMED_SOURCES=0 \
PREPARE_UNIPROT=force \
PREPARE_AFFINITY=force \
CURATE_DATA=force \
FAST_CURATE=1 \
RESUME_CURATE=1 \
INCLUDE_ORIGINAL_FULL_SELECTED=1 \
TRAIN_PHASES=all \
./scripts/run_full_biomed_annotations_affinity_training.sh
```

For the current bio-scale run with million-row sequence targets, 25k static-structure/contact rows, and 2,500 structure-dynamics rows, the launch command is:

```bash
RUN_ID=20260513T011100Z-bio-scale-all-atom-contact \
BIO_SEQUENCE_TARGET_ROWS_PER_MODALITY=3000000 \
PROTEIN_SEQUENCE_TARGET_ROWS=3000000 \
STRUCTURE_DYNAMICS_TARGET_ROWS=2500 \
STATIC_STRUCTURE_TARGET_ROWS=25000 \
TRAIN_PHASES=all \
setsid ./scripts/run_bio_scale_all_atom_contact_training.sh \
  > logs/biomed_direct_training/20260513T011100Z-bio-scale-all-atom-contact.nohup.log 2>&1 < /dev/null &
echo $! > logs/biomed_direct_training/20260513T011100Z-bio-scale-all-atom-contact.pid
```

Tail that run with:

```bash
tail -f logs/biomed_direct_training/20260513T011100Z-bio-scale-all-atom-contact.nohup.log
```
