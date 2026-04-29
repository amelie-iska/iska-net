# Dataset-Expanded Paper Update Plan

Created: 2026-04-29

## Objective

Update `assets/human_learning_transformer_learning_review_dataset_expanded.tex` so it preserves the newer dataset-expansion material while importing the prior graph-to-graph terminology and architecture corrections from `assets/human_learning_transformer_learning_review_graph_to_graph.tex`. Then bring the codebase into alignment with the new paper commitments: NatureLM/SFM and UniGenX acquisition readiness, explicit train/validation/test readiness, leakage-aware split planning, progress/logging/W&B reporting, and orthogonal vertex/edge identifier support for TokenGT-style graph tokens.

Additional April 29 constraint: the newest paper and implementation must describe the first molecular run as sequence-first. SELFIES/SMILES and protein/DNA/RNA sequences are trainable; actual structure files, dynamics trajectories, coordinate labels, force labels, and direct physics supervision are excluded. UMA is used as an external oracle/reward source conditioned on continuous temperature \(T\approx 300\)--\(400\,\mathrm K\), not as direct energy/force-label training.

## Audit Findings

1. `dataset_expanded.tex` contains newer dataset expansion sections that are not present in `graph_to_graph.tex`, including additional dataset sources, GoT/ToT/CoT/GFlowNet conversion, entity-aware splitting, a 2B-token curriculum, validation tables, and replication details. These should be retained.
2. `dataset_expanded.tex` regressed terminology from the previous graph-to-graph version:
   - The legacy architecture macro should be replaced by `\UGM`.
   - Abstract, bond-type section, validation tables, conclusion, and sampler pseudocode still use legacy architecture names.
   - The legacy integrated-extension label should become `sec:integrated_ugm_extension`.
3. `graph_to_graph.tex` already replaced narrow sequence-only terminology with Universal Graph Model (UGM). The newest paper must preserve that language.
4. The dataset expansion audit describes the completed public selected corpus, but it does not yet distinguish:
   - SFM/NatureLM and UniGenX reference repos/token files.
   - UniGenX public graph examples already included in the full selected public corpus.
   - Newly acquired NatureLM-style public source rows from PubChem, UniProt, and RefSeq.
   - Materials Project as API-gated and not train-ready without `MP_API_KEY`.
5. The split policy section correctly rejects row-level SHA1 as sufficient for scientific multimodal learning, but it needs a sharper implementation distinction:
   - Current completed HF selected public corpus: row-hash 98/1/1 split.
   - New public-source acquisition smoke/full prep: deterministic splits now, upgraded to entity-aware split registry as sources receive cluster/scaffold/family keys.
   - Future scientific expansion: sequence/structure/scaffold/family/time-aware grouping is mandatory.
6. The paper has duplicate `\appendix` declarations and duplicate conclusion sections inherited from merged drafts. This is not a hard compile error, but it creates confusing structure. The safe near-term fix is to leave section content intact while adding a note in the plan for a later structural consolidation; a more invasive appendix reorganization should be done as a dedicated editorial pass.
7. Code currently encodes endpoint IDs for source graph tokens, but it does not expose a separate vertex/edge identifier stream. Node tokens use endpoint pair `(i, i)` and edge tokens use endpoint pair `(src, dst)`. This is incidence-aware but not a complete vertex/edge identifier implementation.
8. Dataset acquisition now has `scripts/acquire_naturelm_sources.py` and gzip-aware `scripts/prepare_science_sources.py`, but provenance writing should be made append/per-source safe and W&B/logging should be added for dataset acquisition summaries.

## Paper Patch Plan

1. Replace the legacy architecture macro with `\newcommand{\UGM}{\textsc{UGM}}`.
2. Replace all legacy architecture text, macro calls, labels, and sampler names in the newest TeX:
   - legacy architecture macro -> `\UGM`.
   - legacy sampler name -> `sample_ugm`.
   - legacy integrated-extension label -> `sec:integrated_ugm_extension`.
   - Abstract sentence should explicitly define UGM as Universal Graph Model, not a sequence-only protein model.
3. Preserve the dataset expansion section but update the current corpus audit:
   - Add NatureLM/SFM + UniGenX reference-token status.
   - Add UniGenX train-ready counts from the completed full selected public corpus.
   - Add NatureLM public-source acquisition status as a newly prepared corpus separate from reference-token-only status.
   - State that PubChem full CID-SMILES is downloaded but high-cardinality graphification is opt-in.
   - State that Materials Project remains credential/API-gated.
4. Add explicit orthogonal identifier language in the TokenGT recap and implementation blueprint:
   - Vertex IDs, edge IDs, and endpoint IDs are separate structural channels.
   - Full mathematical orthogonality requires identifier dimension at least identifier count; otherwise the implementation uses orthogonally initialized or hash-signed approximate identifiers.
5. Expand entity-aware split details:
   - Current row-hash splits are acceptable for smoke and broad public text graph corpora.
   - Scientific sources graduate to split keys by UniRef/MMseqs sequence clusters, Foldseek/CATH/ECOD structure clusters, Bemis-Murcko/InChIKey molecule groups, Rfam/RNA3DB RNA groups, KG release/time/entity neighborhoods, Hebrew root/template groups, and graph canonical hashes.
6. Leave duplicate appendix/conclusion structure untouched in this patch unless it blocks compilation. Log it as follow-up editorial cleanup.
7. Rewrite structure/dynamics language into sequence-first graph-state language:
   - Molecular inputs are SELFIES/SMILES and biological sequences.
   - Optional string-derived geometric descriptors are off by default.
   - Structure/dynamics datasets are evaluation-only until a later explicit phase.
   - UMA temperature conditioning is continuous; 300/325/350/375/400K are curriculum anchors, not the complete conditioning space.

## Code Implementation Plan

1. Finish current NatureLM/SFM public-source acquisition and preparation.
2. Run integrity, token-count, and catalog validation checks over the resulting `data/processed/naturelm_public_sources/` corpus.
3. Add `config/data/naturelm_public_sources.yaml` so the new train/val/test JSONL files are directly trainable and use NatureLM/UniGenX plus multimodal reference vocabularies.
4. Patch `scripts/acquire_naturelm_sources.py`:
   - Use append/per-source provenance instead of overwriting `PROVENANCE.jsonl` for shared raw directories.
   - Add `--log-dir`, structured run logs, summary writing, and optional W&B logging.
   - Keep tqdm progress bars for downloads and graphification.
5. Implement graph-token identifier support:
   - Extend graph tokenization to return endpoint IDs and structural identifier IDs.
   - Assign node tokens vertex identifiers and edge tokens edge identifiers in a separate stream.
   - Pass `identifier_ids` through `EncodedExample`, collator batches, training, validation, and inference.
   - Add a fixed orthogonally initialized identifier embedding stream in `RandomOrderTokenGT`.
6. Update configs to allocate identifier dimensions for 4090/tiny/small models.
7. Add/adjust tests:
   - Collator emits nonzero node and edge identifier IDs.
   - Edge identifier IDs do not collide with vertex identifier IDs inside a graph when capacity allows.
   - Model forward pass consumes `identifier_ids`.
8. Run verification:
   - `python -m py_compile` for modified scripts/modules.
   - targeted unit tests for graph schema/collator/model/acquisition dry run.
   - full `pytest -q` if runtime remains reasonable.
   - TeX terminology scan: no legacy architecture term remains in the newest paper.
   - Dataset integrity and token-count reports exist for NatureLM public sources.
9. Enforce sequence-first molecular training:
   - Add/maintain a phase policy that removes structure fields in first-run graphification.
   - Keep continuous temperature features in sequence-only rows.
   - Allow UMA-oracle feedback records while blocking direct energy/force/coordinate labels.

## Completion Criteria

- Newest paper contains UGM terminology and expanded dataset/split discussion.
- NatureLM/SFM + UniGenX data/reference status is represented in paper, configs, manifests, and local reports.
- `naturelm_public_sources` has train/validation/test JSONL, summary, integrity, and token-count artifacts.
- Training code accepts identifier IDs and logs metrics through existing W&B hooks.
- Dataset acquisition emits tqdm progress, structured logs, and optional W&B summary metrics.
- Unit tests and integrity checks pass or any remaining blocker is explicitly recorded with concrete next action.
