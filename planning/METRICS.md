# Metrics and W&B Namespaces

All metrics are step-based. W&B is optional and disabled by default in the tiny configs. Local JSONL metrics are always written to each run directory.

## Graph/Tool/Proof SFT Stages

Namespace pattern:

```text
<stage>/train/<metric>
<stage>/val/<metric>
```

Metrics:

- `loss`: cross-entropy on random-order `<POS>` query positions.
- `token_accuracy`: accuracy on target graph-token predictions only.
- `total_loss`: CE plus enabled auxiliary losses.
- `topology_loss`: MSE for the lightweight topology prediction head.
- `numeric_diffusion_loss`: PLAN-F UniGenX-style denoising loss over extracted numeric coordinate/property targets when `model.numeric_dim > 0`.
- `perplexity`: `exp(loss)` on validation, clipped internally to avoid overflow.
- `grad_norm`: clipped gradient norm after accumulation.
- `lr`: current optimizer learning rate.
- `profile/parameter_count`: total local model parameters from `scripts/profile_model.py`.
- `profile/trainable_parameter_count`: trainable parameters after optional LoRA/freezing.
- `profile/estimated_param_memory_mb`: rough parameter-memory footprint, not activation memory.
- `tropical/logit_entropy`: entropy of target-position logits under the current tropical temperature.
- `tropical/top1_margin`: top-1 minus top-2 logit margin.
- `tropical/top1_confidence`: mean max softmax probability.
- `topology/*`: batch mean graph topology summaries.
- `verifier/*`: validation-time verifier reward/pass diagnostics.
- `code/*`, `lean/*`, `chem/*`, `science/*`, `hebrew/*`, `multimodal/*`: domain validation metrics when the batch contains those task families.

Important stages:

- `graph_pretrain/train/loss`
- `graph_pretrain/train/token_accuracy`
- `graph_pretrain/val/loss`
- `graph_pretrain/val/token_accuracy`
- `tool_sft/train/loss`
- `proof_code_sft/train/loss`

## GFlowNet Stage

Namespace:

```text
gflownet/<metric>
```

Metrics:

- `tb_loss`: trajectory-balance residual loss.
- `trajectory_balance_loss`: raw trajectory-balance component before subtrajectory weighting.
- `subtrajectory_balance_loss`: auxiliary subtrajectory-balance component.
- `logZ`: learned log partition value.
- `subtb_logZ`: learned log partition value for the auxiliary subtrajectory-balance module.
- `reward_mean`: mean terminal reward.
- `reward_max`: maximum terminal reward in the batch.
- `trajectory_len`: mean sampled trajectory length.
- `terminal_valid_rate`: exact terminal graph-set validity rate.
- `verifier_exact_rate`: exact terminal target-token set rate.
- `token_recall`: mean target-token recall.
- `unique_terminal_states`: number of unique terminal states in the logged batch.
- `action_entropy`: average forward policy entropy.
- `action_coverage`: fraction of candidate actions selected at least once in the logged batch.
- `context_enabled`: whether topology context features are supplied to the policy.
- `backward_policy_learned`: whether a learned backward policy is active.
- `grad_norm`: clipped gradient norm.
- `gflownet/code/*`, `gflownet/lean/*`, `gflownet/chem/*`: domain metrics on sampled terminal token sets when those tasks appear in the batch.
- `gflownet/science/*`: reserved domain metric namespace for PLAN-D/PLAN-G graph-of-thought runs.
- `gflownet/hebrew/*`: root-extension and Hebrew graph-of-thought metrics for PLAN-E runs.
- `gflownet/multimodal/*`: UGM oracle-feedback metrics for PLAN-H graph-record trajectory runs.

## Domain Metrics

Code metrics:

- `code/has_tests_rate`: examples with embedded test nodes or test metadata.
- `code/test_count_mean`: mean number of extracted tests.
- `code/attempt_rate`: examples where code and tests were both present and execution was attempted.
- `code/pass_rate`: attempted examples that passed local pytest execution.
- `code/python_error_rate`: attempted examples that failed or timed out.

Lean metrics:

- `lean/available`: local `lean` binary availability.
- `lean/source_present_rate`: examples with Lean statement/proof source.
- `lean/compile_attempt_rate`: compile attempt rate when Lean and source are both present.
- `lean/compile_success_rate`: successful local Lean compilations.
- `lean/error_rate`: attempted Lean compilations that failed or timed out.

Chemistry metrics:

- `chem/rdkit_available`: local RDKit import availability.
- `chem/smiles_present_rate`: examples with a SMILES value.
- `chem/smiles_valid_rate`: RDKit-valid SMILES rate when RDKit is available.
- `chem/atom_count_mean`: mean RDKit atom count or fallback atom-symbol count.
- `chem/bond_count_mean`: mean RDKit bond count.
- `chem/medicinal_filter_pass_rate`: deterministic medicinal-chemistry alert filter pass rate.
- `chem/reactive_alert_count_mean`: count of simple reactive-fragment alerts. This is a triage signal, not a toxicity claim.

Local audio metrics:

- `audio/task_present_rate`: examples with a local audio task label.
- `audio/taxonomy_node_count_mean`: mean count of taxonomy nodes when the user-provided row includes taxonomy metadata.
- `audio/noncommercial_license_rate`: examples whose local license string indicates non-commercial or NC terms.
- `audio/audio_feature_node_count_mean`: local audio feature nodes when `local_audio_path` was supplied.
- `audio/audio_feature_available_rate`: rate of successful local audio metadata extraction.

Science/UniGenX-style metrics:

- `science/coordinate_node_count_mean`: mean number of coordinate nodes, mainly QM9 atom-position rows.
- `science/protein_ligand_coordinate_node_count_mean`: protein/ligand coordinate nodes for docking-style rows.
- `science/property_node_count_mean`: mean number of numeric molecule property nodes.
- `science/material_formula_count_mean`: mean number of material formula nodes.
- `science/molecule_smiles_present_rate`: examples with molecule SMILES nodes.
- `science/protein_sequence_present_rate`: examples with protein sequence nodes.
- `science/ec_number_present_rate`: examples with EC-number conditioning nodes.
- `science/pocket_atom_count_mean`: mean pocket atom nodes for docking-style rows.

Biomedical/bioactivity metrics:

- `biomed/assay_value_present_rate`: examples with assay or affinity value nodes.
- `biomed/assay_numeric_present_rate`: assay values containing parseable numbers.
- `biomed/target_node_count_mean`: target/protein node count.
- `biomed/smiles_present_rate`: examples with ligand SMILES nodes.

UGM multimodal metrics:

- `multimodal/modality_count_mean`: mean number of modalities present among text, protein, SELFIES/SMILES, DNA, RNA, all-atom, and trajectory records.
- `multimodal/protein_present_rate`, `multimodal/selfies_present_rate`, `multimodal/dna_present_rate`, `multimodal/rna_present_rate`: modality coverage rates.
- `multimodal/all_atom_present_rate`: examples with atom/bond output graph records.
- `multimodal/trajectory_present_rate`: examples with conformer/frame records.
- `multimodal/atom_count_mean`: mean atom node count.
- `multimodal/bond_count_mean`: mean molecular-bond edge count.
- `multimodal/bond_type_coverage_rate`: fraction of molecular bonds carrying a first-class `bond_type`.
- `multimodal/coordinate_node_count_mean`: mean coordinate record count.
- `multimodal/distance_node_count_mean`: mean distance/distogram record count.
- `multimodal/frame_count_mean`: mean trajectory-frame count.
- `multimodal/energy_record_rate`: examples with energy records.
- `multimodal/force_record_rate`: examples with force records.
- `multimodal/temperature_conditioned_rate`: examples with temperature conditioning.
- `multimodal/pdb_target_rate`: examples with PDB serializer target records.
- `multimodal/oracle_feedback_target_rate`: examples with oracle-feedback target records.

Hebrew metrics:

- `hebrew/root_node_count_mean`: mean number of shoresh/root nodes.
- `hebrew/unique_root_count_mean`: mean number of unique root values.
- `hebrew/template_node_count_mean`: mean number of heuristic or explicit template nodes.
- `hebrew/radical_node_count_mean`: mean number of radical nodes attached to roots.
- `hebrew/lemma_node_count_mean`: mean number of lemma nodes, mainly from UD CoNLL-U data.
- `hebrew/binyan_node_count_mean`: mean number of Hebrew binyan/pattern nodes.
- `hebrew/derived_form_count_mean`: mean number of root-extension generated form nodes.
- `hebrew/diacritized_pair_rate`: examples containing a diacritized Hebrew node.
- `hebrew/qa_or_instruction_rate`: examples with an answer/instruction-style node.

## Validation

`scripts/validate_stage.py` prints JSON:

- `validation/loss`
- `validation/perplexity`
- `validation/token_accuracy`
- `validation/topology/*`
- `validation/tropical/*`
- `validation/verifier/*`
- `validation/code/*`
- `validation/lean/*`
- `validation/chem/*`
- `validation/science/*`
- `validation/hebrew/*`
- `validation/multimodal/*`

GFlowNet validation from `scripts/validate_gflownet.py` uses:

- `gflownet_val/reward_mean`
- `gflownet_val/trajectory_len`
- `gflownet_val/terminal_valid_rate`
- `gflownet_val/unique_terminal_states`
- `gflownet_val/action_entropy`
- `gflownet_val/action_coverage`
- `gflownet_val/context_enabled`
- `gflownet_val/verifier/*`
- `gflownet_val/code/*`, `gflownet_val/lean/*`, `gflownet_val/chem/*`
- `gflownet_val/hebrew/*`
- `gflownet_val/multimodal/*`

## Interpretation

For tiny smoke runs, loss and perplexity are only functional checks. Meaningful quality assessment starts once:

- train/validation splits are independent;
- dataset contamination has been checked;
- target graph-token vocab has enough support;
- verifier rewards are domain-specific rather than synthetic proxies.
