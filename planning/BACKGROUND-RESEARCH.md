# Background Research Notes

Date: 2026-04-28

This note records the implementation-relevant research pass for `PLAN-A`.

## TokenGT

- Upstream paper/repo: Tokenized Graph Transformer (TokenGT), "Pure Transformers are Powerful Graph Learners"; implementation at `https://github.com/jw9730/tokengt`.
- Local checkout exists at `./tokengt`, remote `https://github.com/amelie-iska/tokengt.git`, commit `42d6e91`.
- Relevant local files:
  - `tokengt/large-scale-regression/tokengt/modules/tokenizer.py`
  - `tokengt/large-scale-regression/tokengt/modules/tokengt_graph_encoder.py`
  - `tokengt/large-scale-regression/tokengt/data/collator.py`
- Decision: use the TokenGT graph-token contract as the base and keep the upstream checkout as reference. Implement a compact local PyTorch model because the upstream large-scale path depends on Fairseq and an uninitialized submodule.

## GFlowNets

- Reference library: `https://github.com/GFNOrg/torchgfn`.
- Documentation confirms modular PyTorch support for GFlowNet variants, including trajectory balance.
- Trajectory balance reference: Malkin et al., "Trajectory Balance: Improved Credit Assignment in GFlowNets" (`https://arxiv.org/abs/2201.13259`).
- Decision: clone `torchgfn` into `data/external_repos/torchgfn` for reference/provenance, but implement project-local trajectory balance over graph-of-thought actions so the semantics match the paper.

## Random-Order Autoregression

- XLNet permutation language modeling motivates order-agnostic autoregressive factorization with attention masks.
- sigma-GPT ("A New Approach to Autoregressive Models") motivates adding explicit output position/order information so generation can proceed in arbitrary order.
- RandAR shows random-order decoder-only autoregression can work when each prediction is paired with a position instruction token, though it is vision-focused.
- Graph generation work emphasizes that autoregressive graph output order affects quality.
- Decision: implement graph-token random-order autoregression with structural IDs plus reveal-order IDs, not a direct clone of any vision or sequence-only implementation.

## Dataset Findings

Hugging Face dataset availability checked by Hub metadata and Dataset Viewer split calls.

| Area | Dataset | Config/Split | License/Status | Default use |
| --- | --- | --- | --- | --- |
| Math | `openai/gsm8k` | `main/train`, `main/test` | MIT | smoke + validation |
| Math | `nvidia/OpenMathInstruct-2` | `default/train`, `train_1M`, `train_2M`, `train_5M` | CC-BY-4.0, large | sample only |
| Math | `AI-MO/NuminaMath-CoT` | `default/train`, `default/test` | Apache-2.0 | sample |
| Tool math | `AI-MO/NuminaMath-TIR` | `default/train`, `default/test` | Apache-2.0 | sample |
| Code | `bigcode/bigcodebench` | `default/v0.1.4` | Apache-2.0 | smoke + validation |
| Code | `deepmind/code_contests` | dataset card available | CC-BY-4.0 | sample only |
| Code | `bigcode/the-stack-v2` | gated | other/gated, huge | manifest only |
| Lean | `internlm/Lean-Workbook` | `default/train` | Apache-2.0 | sample |
| Lean | `PAug/ProofNetSharp` | `default/valid`, `default/test` | MIT | sample + validation |
| Chemistry | `scikit-fingerprints/MoleculeNet_Lipophilicity` | `default/train` | license unknown in HF metadata | smoke only with provenance warning |
| Molecules | `zpn/zinc20` | huge | MIT | manifest only |

## PLAN-E Hebrew Research Addendum

The PLAN-E pass checked the requested Hebrew corpora through Hugging Face metadata, Dataset Viewer split inspection, GitHub repository inspection, and local clone validation.

| Area | Source | Config/Split or Path | License/Status | Integration decision |
| --- | --- | --- | --- | --- |
| Classical Hebrew | `sivan22/sefaria-hebrew` | `default/train`; fields include `language`, `title`, `versionSource`, `versionTitle`, `license`, `text` | upstream license field per row; HF size tag 1M<n<10M | sampled through manifest; graphified as `hebrew_classical_text` with title/source evidence nodes and root/template heuristic nodes |
| Hebrew medical text | `cp500/synthetic_hebrew_medical_text` | `default/train`; field `text` | dataset card sparse; review upstream before large training | sampled through manifest; graphified as `hebrew_medical_pretraining` |
| Hebrew WikiAnswers | `imvladikon/wikianswers_hebrew` | `lists/train`, `queries/train`; fields `record_id`, `set_id`, `text` | HF sentence-similarity dataset, language:he | sampled through manifest; graphified as `hebrew_question_similarity` with set/record metadata |
| Hebrew instruction data | `ashercn97/hebrew_alpaca` | `default/train`; fields `instruction`, `input`, `output` | HF dataset card sparse; review upstream before release-scale use | sampled through manifest; graphified as `hebrew_instruction_sft` with answer target tokens |
| Talmud Hebrew | `guyhadad01/Talmud-Hebrew` | `default/train`; fields `id`, `content` | HF dataset card sparse; review upstream before large training | sampled through manifest; graphified as `hebrew_classical_text` |
| Hebrew Wikipedia | `YanFren/Hebrew_wikipedia` | `default/train`; field `text` | HF metadata reports MIT | sampled through manifest; graphified as `hebrew_pretraining` |
| Hebrew Treebank replacement | `UniversalDependencies/UD_Hebrew-HTB` | `he_htb-ud-{train,dev,test}.conllu` | CC BY-NC-SA 4.0 | cloned into `data/raw/hebrew_ud_htb/repo`; parsed as CoNLL-U morphosyntax graphs with lemmas, dependencies, `HebBinyan`, root/template nodes |
| Hebrew QA | `NNLP-IL/Hebrew-Question-Answering-Dataset` | `data/data v1.0` and `data/data v1.1` JSON files | repository includes license file | cloned into `data/raw/hebrew_qa_nnlp/repo`; parsed as QA graph examples with answer target tokens |
| Nakdimon / Dicta-style vocalization | `elazarg/nakdimon` | repository plus `hebrew_diacritized` submodule text files | repository includes MIT license | cloned into `data/raw/hebrew_nakdimon/repo`; sampled as undotted/diacritized graph pairs for morphology-aware intermediate training |
| Verb Complements Lexicon | no stable public download found during implementation pass | expected TSV/CSV with `verb_root` and verb/complement columns | manifest-only until user supplies source file | implemented optional local TSV/CSV ingestion under `data/raw/hebrew_verb_complements_lexicon` |
| Semitic Root Encoding data | no public dataset dump found during implementation pass | SRE-style method rather than local corpus | represented as a construction target | implemented heuristic root/template graphification plus synthetic root-extension graphs for GFlowNet graph-of-thought training |

Key design choices from this pass:

- Gold morphological annotation is preferred where available: UD Hebrew-HTB supplies lemmas, dependency structure, and Hebrew-specific features such as `HebBinyan`; optional Verb Complements rows can provide `verb_root`.
- For broad text corpora without root annotations, the implementation uses a conservative heuristic root/template extractor. These are training features and auxiliary targets, not gold linguistic labels.
- Root-extension GFlowNet training is separated from ordinary Hebrew SFT. It uses graph states whose targets include root, radical, binyan, and derived-form tokens so trajectory balance trains graph-of-thought paths over Semitic morphology.
- All Hebrew acquisitions are provenance-tracked and limits are configurable. Sparse-license or card-sparse HF datasets are kept suitable for local research and require upstream review before large redistribution.

## PLAN-F Deferred-Component Addendum

The deferred science/data items are now implemented as local or opt-in paths rather than default bulk downloads.

- SFM/NatureLM full-corpus reconstruction is represented by local preparation modes for PubChem, UniProt, RefSeq/NCBI, and Materials Project rows through `scripts/prepare_science_sources.py`; the SFM repo itself is used for reference README/checkpoint/domain-token provenance.
- UniGenX-style numeric modeling is used as methodology background, but the active UGM implementation keeps numeric and structure-candidate values in the autoregressive graph-token stream. The corrected UniGenX source is the `amelie-iska/UniGenX` GitHub repository, whose dictionary/tokenizer files are extracted into the local vocabulary.
- User-provided local audio metadata extraction remains available for rows with `local_audio_path` or `audio_path`, but no external audio corpus is part of the SFM/NatureLM or UniGenX science pipeline.
- Protein-ligand docking and EC-number protein generation are represented by graphifiers for local PDBbind/docking rows and FASTA/CSV EC rows.
- ChEMBL/BindingDB ingestion is represented by local CSV/TSV normalization plus the small public `jglaser/binding_affinity` Hugging Face sample for smoke testing.
- Full secure microVM execution, full Fairseq TokenGT training, and external QLoRA are not invoked by default. The repo now has subprocess resource limits, an upstream TokenGT readiness probe, and a QLoRA dependency-gated entry point.

## PLAN-G SFM/NatureLM and UniGenX Correction

The corrected NatureLM source for this project is `https://github.com/amelie-iska/SFM`. The NatureLM paper is arXiv `2502.07527`, "NatureLM: Deciphering the Language of Nature for Scientific Discovery"; it describes a sequence-based science foundation model spanning small molecules, materials, proteins, DNA, and RNA, with model sizes reported at 1B, 8B, and 46.7B. The SFM checkout contains `NatureLM/README.md` with Hugging Face checkpoint references for `microsoft/NatureLM-8x7B` and `microsoft/NatureLM-8x7B-Inst`.

The corrected UniGenX source is `https://github.com/amelie-iska/UniGenX`. The paper is arXiv `2503.06687`, "UniGenX: Unified Generation of Sequence and Structure with Autoregressive Diffusion"; it couples autoregressive next-token modeling with conditional diffusion for sequence/structure generation over molecules, materials, and proteins. The local integration extracts `unigenx/data/dict*.txt` and tokenizer special tokens into `data/processed/reference_tokens/naturelm_unigenx_tokens.txt`, then includes that path in science configs through `data.extra_vocab_paths`.

## PLAN-H Multimodal Graph-to-Graph Addendum

The extended paper review adds a second training phase over mixed graph records rather than a separate molecular architecture. The implementation uses neutral multimodal graph-to-graph terminology.

Architecture and baseline research:

- AlphaFold 3, RoseTTAFold All-Atom, Boltz, and Chai-1 are treated as specialist all-atom biomolecular baselines. They motivate the output record families but are not cloned as architecture.
- ESM3 and ProGen2 motivate protein sequence/structure/function language modeling.
- SELFIES and Group SELFIES motivate robust small-molecule tokens and fragment motifs.
- NatureLM/SFM remains the multi-domain sequence/source framing for small molecules, materials, proteins, DNA, and RNA.
- UniGenX remains a key reference for coupling symbolic autoregression with numeric/structure generation. UGM deliberately chooses the autoregressive graph-record variant for the current implementation, including identity-bearing coordinate tokens rather than a parallel numeric diffusion loss.
- UMA/OMol25, GEOM, and SPICE motivate oracle-feedback scoring, evaluation protocols, and later-phase physics supervision. In the first run, UMA is an external temperature-conditioned oracle/reward source rather than a direct energy/force-label dataset.
- UniProt, InterPro, PROSITE, CATH, RNAcentral, Rfam, NatureLM/SFM-style science rows, UniGenX-style sequence/function rows, and ProTrek-style sequence/function rows are intended first-run sequence/function families, subject to license and contamination review. RNA 3D Hub, PDB/RCSB, NAKB, ATLAS, mdCATH, GEOM, SPICE, and OMol/UMA-family resources are evaluation, oracle, or future-phase sources unless reduced to safe train-split sequence motif vocabularies.

Implementation decisions:

- Add `src/iska_reasoner/data/multimodal.py` for neutral graph-record vocabulary, graphification, synthetic rows, and optional input-row PDB rendering for explicit evaluation examples.
- Add `src/iska_reasoner/data/motifs.py`, `scripts/build_motif_vocab.py`, and the public motif download path for PROSITE, InterPro, CATH, and Rfam. Motif tokens become `SEQ_MOTIF:*`, `STRUCT_MOTIF:*`, and `STRUCT_DERIVED_SEQ_MOTIF:*` graph records.
- Add `scripts/build_multimodal_vocab.py` and `scripts/prepare_multimodal_sources.py`.
- Add phase-2 configs for multimodal SFT and oracle-feedback GFlowNet training.
- Add multimodal verifier metrics and a real FairChem/UMA oracle adapter for candidate scoring. The deterministic proxy path remains only for unit tests and smoke runs via `UGM_UMA_BACKEND=proxy`; direct structure-file supervision remains blocked in the first run.
- Keep the core TokenGT-style model unchanged. Geometry is represented through graph records, numeric targets, and losses rather than a special coordinate head.

## Environment

- `conda` is available.
- Existing `tokengt` env:
  - Python 3.11.15
  - PyTorch 2.8.0+cu128
  - `tqdm`, `wandb`, `networkx`, and `pyyaml` present
  - `datasets` and `transformers` missing
- Decision: create a new reproducible `iska-ugm` conda env in `environment.yml`, but keep scripts usable from existing `tokengt` env after dependency installation.

## Source Links

- TokenGT: `https://github.com/jw9730/tokengt`
- TokenGT paper: `https://arxiv.org/abs/2207.02505`
- torchgfn: `https://github.com/GFNOrg/torchgfn`
- torchgfn paper page: `https://huggingface.co/papers/2305.14594`
- GFlowNet trajectory balance: `https://arxiv.org/abs/2201.13259`
- OpenMathInstruct-2: `https://huggingface.co/datasets/nvidia/OpenMathInstruct-2`
- GSM8K: `https://huggingface.co/datasets/openai/gsm8k`
- BigCodeBench: `https://huggingface.co/datasets/bigcode/bigcodebench`
- Lean Workbook: `https://huggingface.co/datasets/internlm/Lean-Workbook`
- ProofNetSharp: `https://huggingface.co/datasets/PAug/ProofNetSharp`
- MoleculeNet Lipophilicity: `https://huggingface.co/datasets/scikit-fingerprints/MoleculeNet_Lipophilicity`
- RandAR: `https://arxiv.org/abs/2412.01827`
- sigma-GPT: `https://arxiv.org/abs/2404.09562`
- XLNet: `https://huggingface.co/docs/transformers/model_doc/xlnet`
- Sefaria Hebrew dataset: `https://huggingface.co/datasets/sivan22/sefaria-hebrew`
- Synthetic Hebrew medical text: `https://huggingface.co/datasets/cp500/synthetic_hebrew_medical_text`
- WikiAnswers Hebrew: `https://huggingface.co/datasets/imvladikon/wikianswers_hebrew`
- Hebrew Alpaca: `https://huggingface.co/datasets/ashercn97/hebrew_alpaca`
- Talmud Hebrew: `https://huggingface.co/datasets/guyhadad01/Talmud-Hebrew`
- Hebrew Wikipedia: `https://huggingface.co/datasets/YanFren/Hebrew_wikipedia`
- UD Hebrew-HTB: `https://github.com/UniversalDependencies/UD_Hebrew-HTB`
- NNLP-IL Hebrew QA: `https://github.com/NNLP-IL/Hebrew-Question-Answering-Dataset`
- Nakdimon: `https://github.com/elazarg/nakdimon`
- SFM / NatureLM repo: `https://github.com/amelie-iska/SFM`
- NatureLM paper: `https://arxiv.org/abs/2502.07527`
- UniGenX repo: `https://github.com/amelie-iska/UniGenX`
- UniGenX paper: `https://arxiv.org/abs/2503.06687`
- AlphaFold 3: `https://www.nature.com/articles/s41586-024-07487-w`
- RoseTTAFold All-Atom: `https://www.science.org/doi/10.1126/science.adl2528`
- Boltz repository: `https://github.com/jwohlwend/boltz`
- Chai-1 technical report: `https://chaiassets.com/chai-1/paper/technical_report_v1.pdf`
- ESM3 release notes: `https://www.evolutionaryscale.ai/blog/esm3-release`
- SELFIES repository: `https://github.com/aspuru-guzik-group/selfies`
- Group SELFIES: `https://pubs.rsc.org/en/content/articlelanding/2023/dd/d3dd00012e`
- OMol25: `https://arxiv.org/abs/2505.08762`
- GEOM: `https://www.nature.com/articles/s41597-022-01288-4`
- SPICE: `https://www.nature.com/articles/s41597-022-01882-6`
- UniProt 2025: `https://academic.oup.com/nar/article/53/D1/D609/7902999`
- InterPro: `https://academic.oup.com/nar/article/49/D1/D344/5958491`
- RNAcentral: `https://rnacentral.org/about-us`
- RNA 3D Hub: `https://rna.bgsu.edu/rna3dhub`
