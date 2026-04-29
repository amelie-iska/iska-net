# License and Provenance Review

Date: 2026-04-29

This is an engineering provenance checklist, not legal advice. The rule for this repo is conservative: scale a dataset only after its upstream license, redistribution rights, attribution requirements, and benchmark-contamination risk have been reviewed for the intended use.

## Scale Policy

- `ok_for_local_smoke`: suitable for tests, tiny local training, and pipeline validation.
- `ok_for_research_scale`: suitable for larger local research runs after attribution/provenance is retained.
- `review_before_scale`: keep tiny/local until upstream terms are checked.
- `manifest_only`: do not acquire by default; use only after storage, access, and license review.
- `do_not_redistribute`: generated artifacts may stay local; do not publish derived rows/checkpoints without a separate review.

## Current Sources

| Source | Local integration | Manifest/license note | Current status | Scale decision |
| --- | --- | --- | --- | --- |
| SFM / NatureLM repo (`amelie-iska/SFM`) | `data/external_repos/sfm`, reference tokens and checkpoint links | MIT in manifest | Reference repo only; no full NatureLM corpus included | `ok_for_research_scale` for reference-token use |
| UniGenX repo (`amelie-iska/UniGenX`) | `data/external_repos/unigenx`, dictionaries/tokenizer/model reference | MIT in manifest | Reference repo only; no large checkpoint shards by default | `ok_for_research_scale` for dictionary/token use |
| NatureLM HF checkpoints | optional metadata entries | `review-upstream` | Metadata-only by default | `review_before_scale`; explicit bounded downloads only |
| GSM8K | math smoke data | MIT | Small sample and validation | `ok_for_research_scale` |
| BigCodeBench | code vertical slice | Apache-2.0 | Small sample and validation | `ok_for_research_scale` with attribution |
| ProofNetSharp | Lean/proof vertical slice | MIT | Small sample and validation | `ok_for_research_scale` |
| Lean Workbook | Lean/proof manifest/sample path | Apache-2.0 in research notes | Optional sample path | `ok_for_research_scale` with attribution |
| OpenMathInstruct-2 | math instruction manifest/sample path | CC-BY-4.0 | Optional scaled sample path | `ok_for_research_scale` with attribution |
| NuminaMath CoT/TIR | math/tool instruction manifest/sample path | Apache-2.0 | Optional sample path | `ok_for_research_scale` |
| MoleculeNet Lipophilicity | chemistry smoke data | HF metadata sparse/unknown | Tiny smoke only | `review_before_scale` |
| QM9 via `yairschiff/qm9` | UniGenX-style molecule rows | `review-upstream` | Tiny smoke data | `review_before_scale` |
| Materials crystal-system dataset | UniGenX-style material rows | `review-upstream` | Tiny smoke data | `review_before_scale` |
| Binding affinity public sample | bioactivity smoke data | `review-upstream` | Tiny smoke data | `review_before_scale` |
| ChEMBL local exports | local preparation only | source-specific terms | Not auto-downloaded | `review_before_scale` |
| BindingDB local exports | local preparation only | source-specific terms | Not auto-downloaded | `review_before_scale` |
| PDBbind/docking local rows | local preparation only | source-specific terms | Not auto-downloaded | `review_before_scale` |
| PubChem/UniProt/RefSeq/NCBI/Materials Project local rows | local SFM science reconstruction | source-specific terms | Not auto-downloaded | `review_before_scale` |
| UGM multimodal synthetic rows | local generated smoke rows | project-generated | Tiny smoke data | `ok_for_local_smoke` |
| PROSITE/InterPro/CATH/Rfam public motif metadata | UGM sequence, structure, and structure-derived motif vocabulary | source-specific terms and attribution required | Downloader implemented; local public metadata snapshot present under `data/raw_motifs/public/` | `review_before_scale` before release/checkpoint publication |
| UniProt and other local motif rows | UGM protein sequence/function/motif preparation | source-specific terms | Local user-provided only | `review_before_scale` |
| RCSB PDB/wwPDB/AlphaFold DB local structure rows | UGM atom/bond/coordinate/frame preparation | source-specific terms | Not auto-downloaded | `review_before_scale` |
| PubChem/ChEMBL/ZINC/GEOM/SPICE/OMol25 local molecule/conformer rows | UGM molecule, conformer, energy, and force preparation | source-specific terms | Not auto-downloaded | `review_before_scale` |
| RNAcentral/RNA 3D Hub/NAKB local rows | UGM DNA/RNA sequence, motif, and structure preparation | source-specific terms | Not auto-downloaded | `review_before_scale` |
| ATLAS/mdCATH local trajectory rows | UGM temperature-conditioned trajectory preparation | source-specific terms | Not auto-downloaded | `review_before_scale` |
| The Stack v2 | code pretraining manifest | gated/other | Manifest-only | `manifest_only` |
| ZINC20 | molecule pretraining manifest | MIT in manifest | Manifest-only | `review_before_scale` before large acquisition |

## Hebrew Sources

| Source | Local integration | Manifest/license note | Current status | Scale decision |
| --- | --- | --- | --- | --- |
| Sefaria Hebrew | HF sample rows | row-level/upstream license fields | Tiny local sample | `review_before_scale`; preserve row license/title/source |
| Synthetic Hebrew medical text | HF sample rows | sparse dataset card | Tiny local sample | `review_before_scale` |
| WikiAnswers Hebrew | HF sample rows | sparse/upstream review | Tiny local sample | `review_before_scale` |
| Hebrew Alpaca | HF sample rows | sparse dataset card | Tiny local sample | `review_before_scale` |
| Talmud Hebrew | HF sample rows | sparse dataset card | Tiny local sample | `review_before_scale` |
| Hebrew Wikipedia | HF sample rows | MIT in research notes | Tiny local sample | `ok_for_research_scale` after attribution check |
| UD Hebrew HTB | Git clone / CoNLL-U parsing | CC BY-NC-SA 4.0 in research notes | Local clone/sample | `do_not_redistribute`; noncommercial/share-alike constraints |
| NNLP-IL Hebrew QA | Git clone / QA graph parsing | repository license present | Local clone/sample | `review_before_scale`; verify license text before release |
| Nakdimon | Git clone / diacritization pairs | MIT in research notes | Local clone/sample | `ok_for_research_scale` after submodule license check |
| Verb Complements Lexicon | local TSV/CSV only | no stable public download found | Manifest/local-file only | `manifest_only` until official source is supplied |

## Required Before Any Large Training Run

1. Save the exact dataset manifest, git commits, and raw-row acquisition summaries with the run output.
2. Keep `data/raw/*/PROVENANCE.jsonl`, graphified row metadata, and `outputs/*/vocab.jsonl`.
3. Decontaminate against intended validation/test benchmarks before scaling.
4. Block or isolate noncommercial/share-alike sources when training checkpoints may be redistributed.
5. For synthetic teacher traces, store teacher model, date, prompt template, verifier outcome, and leakage checks.
6. Do not publish checkpoints trained on `review_before_scale`, `manifest_only`, or `do_not_redistribute` sources without a separate release review.

## Current Recommendation

For the next 4090 run, use:

- SFM/NatureLM and UniGenX reference-token extraction;
- GSM8K/ProofNetSharp/BigCodeBench for general smoke-scale reasoning;
- Hebrew only as a local research slice unless source-specific release rights are reviewed;
- QM9/material/bioactivity rows only as local smoke or after upstream review.

This keeps the repo trainable while preventing accidental scale-up on sparse-license or noncommercial sources.
