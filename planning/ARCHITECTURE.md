# Architecture

## Core Representation

The project implements a compact TokenGT-style random-order graph model for the **Universal Graph Model (UGM)** type. The input prefix is a typed graph:

- node tokens: `N|<type>|<value>`;
- edge tokens: `E|<type>|<src>|<dst>`;
- graph/special tokens: `<GRAPH>`, `<SEP>`.

Node and edge tokens are both first-class transformer tokens. Each graph token carries:

- token ID from a project vocabulary;
- kind ID: special, node, edge, position, target;
- endpoint IDs: node tokens use `(i, i)`, edge tokens use `(src, dst)`;
- slot ID for random-order target positions.

This mirrors the important TokenGT idea from `./tokengt`: edge tokens encode incidence and are processed by ordinary attention. The local implementation avoids Fairseq as a hard dependency so smoke tests and 4090 experiments are easier to run.

## Random-Order Autoregression

The decoder sequence after the graph prefix is:

```text
<SEP>, <POS(slot_a)>, target_token_a, <POS(slot_b)>, target_token_b, ...
```

Training labels are attached only to `<POS>` positions. The content token appears after its position query, so the model must predict the target token before it can attend to that token. The causal mask preserves the sampled reveal-order factorization.

Supported order sources:

- natural order;
- dependency-style order;
- verifier-enabling order;
- uniform random order.

## Model

`src/iska_reasoner/models/random_order_tokengt.py` contains `RandomOrderTokenGT`:

- token embedding;
- kind embedding;
- sequence position embedding;
- slot embedding;
- endpoint identifier embeddings projected into the hidden dimension;
- PyTorch `TransformerEncoder`;
- tied LM head for target graph-token prediction;
- small value head reserved for verifier/reward extensions.
- optional conditional numeric diffusion head for coordinate/property/assay targets;
- optional gradient checkpointing across encoder layers;
- optional project-local LoRA adapters for linear layers.

## Training Stages

`scripts/train_stage.py` dispatches stages from YAML:

- `graph_pretrain`: random-order graph-token prediction;
- `tool_sft`: same runner, intended for tool/repair graph data;
- `code_sft`: code graph-token prediction with unit-test validation metrics;
- `lean_sft`: Lean/proof graph-token prediction with availability and compile metrics;
- `chem_sft`: molecule graph-token prediction with RDKit-backed validation metrics;
- `science_sft`: PLAN-D/PLAN-G science graph-token prediction over SFM/NatureLM reference vocabulary and UniGenX-style molecule/material rows;
- `multimodal_phase2`: PLAN-H UGM graph-to-graph training over text, protein, SELFIES/SMILES, DNA/RNA, function descriptions, safe sequence motifs, continuous temperature, UMA oracle records, attention/coupling bins, token-motion priors, tool records, and reasoning graph states. Generated atom/bond/coordinate/frame candidate records are active graph outputs; actual structure-file-derived coordinates, trajectories, direct energy/force labels, and PDB/mmCIF/SDF rows remain evaluation/future-phase training sources in the first run;
- `hebrew_sft`: PLAN-E Hebrew morphology/root graph-token prediction over text, instruction, QA, UD morphosyntax, and diacritization graphs;
- `topology_aux`: random-order training with a lightweight topology prediction head;
- `gflownet_got`: project-local trajectory-balance training over graph-of-thought token sets.

Current topology support includes graph component count, cycle rank, edge-type entropy, H0 total persistence over graph shortest-path distances, and Laplacian algebraic connectivity. PLAN-F adds optional `ripser`/`gudhi` persistent-homology summaries when those libraries are installed, plus fallback persistent-Laplacian-style spectra over shortest-path filtrations.

Tropical support includes temperature schedules, logit entropy, top-1 margin, and top-1 confidence for supervised target positions. PLAN-F also adds a standalone max-plus `TropicalAttention` module, activation-cell transition signatures, and a maximum-spanning-arborescence parser for tropical dependency-selection experiments.

## GFlowNet Stage

The local GFlowNet stage is in `src/iska_reasoner/gflownet/`.

- `GraphSetPolicy`: policy over add-token actions in a graph-of-thought state.
- optional topology-context features concatenated to policy state;
- optional learned backward policy for reverse-action probabilities;
- `sample_trajectories`: samples action trajectories, records forward log-probs, backward log-probs, rewards, lengths, validity, entropy, and terminal states.
- `TrajectoryBalanceLoss`: implements `(logZ + logPF - logPB - logR)^2`.
- `SubtrajectoryBalanceLoss`: auxiliary prefix-flow loss used when `gflownet.subtrajectory_weight > 0`.
- `GraphEditActionSpace`: graph edit labels for add/delete/stop experiments beyond the default add-token action set.

Rewards are verifier-aware: terminal token sets are scored against graph example targets with token recall, exact-set match, numeric answer matching, Python snippet checks, optional RDKit availability, and FairChem/UMA oracle feedback for multimodal oracle stages. The deterministic oracle proxy is only an explicit smoke-test backend.

The cloned `data/external_repos/torchgfn` repo is retained as a reference implementation and provenance source.

`scripts/validate_gflownet.py` reloads a saved policy, samples terminal graph-token sets, and reports reward, verifier, diversity, and domain metrics.

## Domain Adapters

The domain adapters are deliberately small and inspectable:

- code graphification extracts canonical solutions, entry points, imports, and tests; validation can run local pytest in a temporary directory with a timeout;
- Lean graphification extracts statements, proofs, and imports; validation probes `lean --version` and compiles a temporary file when Lean is installed;
- molecule graphification uses RDKit atom/bond graphs when available and falls back to SMILES character atom-symbol graphs otherwise.
- SFM/NatureLM integration extracts domain/reference tokens from the SFM `NatureLM` README and combines them with local science reconstruction rows.
- User-provided local audio rows can attach audio metadata when `local_audio_path` or `audio_path` is supplied, but no external audio corpus is part of the SFM/NatureLM or UniGenX pipeline.
- UniGenX-style graphification is sanitized for the first run: molecule rows keep SMILES/SELFIES-style strings and non-structure metadata, while atom-coordinate, energy, force, and structure fields are excluded from training; material rows become prompt, completion, formula, and Materials Project ID graphs when allowed.
- PLAN-F science graphification also supports protein/EC rows, protein-ligand docking rows, ChEMBL/BindingDB-style bioactivity rows, and local PubChem/UniProt/RefSeq/Materials/PDBbind preparation through `scripts/prepare_science_sources.py`.
- PLAN-H UGM graphification maps first-run mixed scientific rows into typed graph records for natural-language prompts, protein residues, SELFIES/SMILES tokens, DNA/RNA bases, sequence motifs, safe structure-derived sequence-motif vocabulary tokens, function-description targets, continuous temperature conditioning, UMA oracle records, attention-coupling bins, and token-motion priors. Generated coordinate/frame candidate records can appear as sampler outputs, but rows derived from actual structure files, energy/force labels, or MD frames are evaluation/future-phase records under the current sequence-only policy.
- Motif graphification maps public/local motif fields into `sequence_motif`, `structure_motif`, and `structure_derived_sequence_motif` nodes. The vocabulary builder parses PROSITE, InterPro, CATH, Rfam, local motif rows, and structure-derived sequence windows from atom/frame rows into first-class graph-record tokens.
- Hebrew graphification normalizes final letters, strips niqqud for analysis, infers heuristic shoresh/root candidates, emits radical and template nodes, parses UD CoNLL-U morphology/dependencies, and supports optional Verb Complements TSV rows with `verb_root` and complement fields.

These adapters are not production sandboxes or full domain platforms. They are rigorous smoke paths for the vertical slices described in PLAN-C.

PLAN-F implements the previously deferred UniGenX-style numeric head. The random-order LM still predicts symbolic graph tokens, while the numeric diffusion head receives graph-level hidden state and denoises extracted numeric values such as coordinates, molecular properties, and assay values.

PLAN-E keeps production Hebrew morphology out of scope. The root/template graphifier is a controlled heuristic unless UD features or local `verb_root` fields provide stronger evidence. This makes the root-extension GFlowNet stage useful for graph-of-thought mechanics without pretending the heuristic roots are gold annotations.

## UGM Multimodal Phase

`config/train/multimodal_phase2_4090.yaml` trains the UGM graph-record objective with random-order reveal over output records. `config/data/multimodal_graphs_4090.yaml` protects both SFM/NatureLM plus UniGenX reference tokens and the UGM multimodal token families:

- ordinary scientific text and reasoning/tool records;
- protein residue, sequence motif, and safe `SEQ_MOTIF_FROM_STRUCTURE:*` vocabulary records;
- SELFIES/SMILES and chemistry fragment records;
- DNA/RNA base and motif records;
- continuous temperature records;
- UMA oracle, attention-bin, token-coupling, token-motion, and sequence-structure-dynamics proxy records.

`config/train/multimodal_oracle_gflownet_4090.yaml` uses the same target-token set as an oracle-feedback GFlowNet stage. The production reward path calls the FairChem/UMA oracle adapter over sequence-grounded candidate molecule graph states. Direct PDB/mmCIF/SDF/trajectory supervision remains outside the first run.

The structure/dynamics phase is currently an evaluation/future-phase gate using `scripts/prepare_structure_dynamics_sources.py`, `config/data/structure_dynamics_graphs.yaml`, `config/train/structure_dynamics_4090.yaml`, and `config/train/structure_dynamics_oracle_gflownet_4090.yaml`. Those train configs are disabled by default. PDB/mmCIF/SDF/trajectory-derived records are reserved for validation, leakage audits, and later explicitly approved structure-file training.

## Hebrew Root GFlowNet Stage

`config/train/hebrew_root_gflownet_tiny.yaml` uses the existing trajectory-balance implementation over root-extension target tokens. Each state is a set of selected graph tokens such as:

- `HEBREW:root:<shoresh>`;
- `HEBREW:radical:<letter>`;
- `HEBREW:derived:<surface-form>`;
- `HEBREW:binyan:<pattern>` when available.

The policy learns to sample diverse terminal sets that reconstruct the root family graph. Validation reports `gflownet_val/hebrew/*` metrics, including root, radical, binyan, and derived-form counts.

## Collator Target Reservation

The random-order collator now reserves sequence room for target `<POS>` query slots before truncating graph-source tokens. This prevents long Hebrew texts from filling the whole sequence and producing all-ignore-label batches, which would otherwise yield undefined cross-entropy.

## Validation and Inference

- `scripts/validate_stage.py` computes loss, perplexity, random-order token accuracy, topology summaries, tropical diagnostics, and verifier metrics.
- `scripts/validate_stage.py` also accepts validation YAML configs.
- `scripts/infer.py` converts text or graph JSON to a graph prefix and generates target graph tokens greedily or by sampling, then reports verifier diagnostics. It supports config files and a simple verifier-guided retry loop.
- `scripts/profile_model.py` reports local parameter counts and a rough memory footprint for 4090 planning.

## 4090 Scaling

The default configs are deliberately small. For 4090-scale experiments, use `config/model/small_4090_tokengt.yaml` and increase sequence length slowly. Full 15B training and dense 256K context training are outside the scope of a single 24GB GPU; the intended path is graph memory, retrieval, and adapter training.
