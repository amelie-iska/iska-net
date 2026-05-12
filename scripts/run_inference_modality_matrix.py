#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tqdm.auto import tqdm

from infer import _rollout_structure_frames, _write_structure_outputs, example_from_text
from iska_reasoner.data.multimodal import graphify_multimodal
from iska_reasoner.graph.schema import Edge, GraphExample, Node
from iska_reasoner.inference.generate import complete_graph_tokens, load_model_for_inference, predict_uma_coordinate_frame
from iska_reasoner.inference.structure_dynamics import (
    derive_full_cartesian_geometry,
    generated_initial_coordinates,
    high_quality_trajectory_score,
    smooth_trajectory_frames,
)
from iska_reasoner.tools import verify_example_tokens
from iska_reasoner.utils.config import load_config


@dataclass(frozen=True)
class InputCase:
    name: str
    modality: str
    description: str
    row: dict[str, Any] | None = None
    text: str | None = None
    graph: GraphExample | None = None


@dataclass(frozen=True)
class OutputCase:
    name: str
    modality: str
    task: str
    prompt: str
    requires_coordinate_slots: bool = False


def _utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
    )


def _graph_case() -> GraphExample:
    nodes = [
        Node(id="task", type="task", value="graph_reasoning"),
        Node(id="protein", type="protein_sequence", value="MKTWYV"),
        Node(id="ligand", type="selfies", value="[C][=O][O]"),
        Node(id="temp", type="temperature", value="325K", features={"kelvin": 325.0}),
    ]
    edges = [
        Edge(src="task", dst="protein", type="has_modality"),
        Edge(src="task", dst="ligand", type="has_modality"),
        Edge(src="task", dst="temp", type="conditions_generation"),
    ]
    return GraphExample(
        id="inference_graph_json_combo",
        task="inference",
        nodes=nodes,
        edges=edges,
        target_tokens=["<UNK>"] * 16,
        metadata={"modalities": ["graph_json", "protein", "selfies"], "temperature": 325.0},
    )


def input_cases() -> list[InputCase]:
    residues = "MKTWYVQLAGSTNDEKRHFP"
    uniprot_500 = (residues * ((500 // len(residues)) + 1))[:500]
    return [
        InputCase(
            name="text_only",
            modality="text",
            description="Plain natural-language graph reasoning prompt.",
            text="Reason over a protein-ligand binding system at 325K and emit graph-structured records.",
        ),
        InputCase(
            name="graph_json",
            modality="graph_json",
            description="Direct typed graph JSON with protein, ligand, and temperature nodes.",
            graph=_graph_case(),
        ),
        InputCase(
            name="protein",
            modality="protein",
            description="Protein sequence-only input.",
            row={"protein_sequence": "MKTWYV", "temperature": 325.0, "oracle": {"name": "uma"}},
        ),
        InputCase(
            name="uniprot_500_protein",
            modality="protein/uniprot_500",
            description="Full-size UniProt-like 500-residue protein sequence-only input.",
            row={
                "protein_sequence": uniprot_500,
                "temperature": 325.0,
                "oracle": {"name": "uma"},
                "accession": "DEV1_500AA",
                "protein_name": "Dev-1 full-size inference stress-test protein",
            },
        ),
        InputCase(
            name="molecule",
            modality="molecule",
            description="Small molecule SELFIES plus SMILES input.",
            row={"selfies": "[C][=O][O]", "smiles": "CC(=O)O", "temperature": 325.0, "oracle": {"name": "uma"}},
        ),
        InputCase(
            name="dna",
            modality="dna",
            description="DNA sequence input.",
            row={"dna_sequence": "ATGCGTACGGATCC", "temperature": 310.0, "oracle": {"name": "uma"}},
        ),
        InputCase(
            name="rna",
            modality="rna",
            description="RNA sequence input.",
            row={"rna_sequence": "AUGCGUACGGAUCC", "temperature": 310.0, "oracle": {"name": "uma"}},
        ),
        InputCase(
            name="protein_molecule",
            modality="protein+molecule",
            description="Protein plus ligand sequence/string input.",
            row={"protein_sequence": "MKTWYV", "selfies": "[C][=O][O]", "smiles": "CC(=O)O", "temperature": 325.0, "oracle": {"name": "uma"}},
        ),
        InputCase(
            name="protein_dna_rna",
            modality="protein+dna+rna",
            description="Protein, DNA, and RNA sequence input.",
            row={"protein_sequence": "MKTWYV", "dna_sequence": "ATGCGTAC", "rna_sequence": "AUGCGUAC", "temperature": 335.0, "oracle": {"name": "uma"}},
        ),
        InputCase(
            name="bioselfies_mixed",
            modality="bioselfies+mixed",
            description="BioSELFIES-only mixed protein/RNA/ligand input.",
            row={
                "input_representation": "bioselfies",
                "protein_sequence": "MKTWYV",
                "rna_sequence": "AUGCGU",
                "selfies": "[C][=O][O]",
                "temperature": 350.0,
                "oracle": {"name": "uma"},
            },
        ),
    ]


def output_cases() -> list[OutputCase]:
    return [
        OutputCase(
            name="graph_completion",
            modality="graph_tokens",
            task="graph_reasoning",
            prompt="Emit graph completion records for the input.",
        ),
        OutputCase(
            name="function_description",
            modality="text/function",
            task="function_description",
            prompt="Generate graph records and a concise function hypothesis.",
        ),
        OutputCase(
            name="molecule_graph",
            modality="molecule",
            task="molecule_reasoning",
            prompt="Generate molecule and chemistry graph records.",
        ),
        OutputCase(
            name="sequence_annotation",
            modality="sequence_annotation",
            task="sequence_annotation",
            prompt="Generate sequence annotation graph records.",
        ),
        OutputCase(
            name="affinity_bioactivity",
            modality="affinity/bioactivity",
            task="biomolecular_affinity",
            prompt="Generate binding, assay, and affinity graph records.",
        ),
        OutputCase(
            name="structure_dynamics",
            modality="structure_dynamics",
            task="structure_dynamics_proxy",
            prompt="Generate UMA-scored all-atom Cartesian structure-dynamics records.",
            requires_coordinate_slots=True,
        ),
    ]


def _case_example(inp: InputCase, out: OutputCase, max_steps: int) -> GraphExample:
    if inp.text is not None:
        ex = example_from_text(f"{out.prompt} {inp.text}")
        ex.target_tokens = ["<UNK>"] * max_steps
        ex.metadata = {"modalities": [inp.modality], "task": out.task}
        return ex
    if inp.graph is not None:
        ex = GraphExample(
            id=f"{inp.graph.id}_{out.name}",
            task=inp.graph.task,
            nodes=inp.graph.nodes,
            edges=inp.graph.edges,
            target_tokens=["<UNK>"] * max_steps,
            metadata={**(inp.graph.metadata or {}), "task": out.task, "modalities": [inp.modality]},
        )
        return ex
    row = dict(inp.row or {})
    row["task"] = out.task
    row["prompt"] = f"{out.prompt} {inp.description}"
    return graphify_multimodal(row, 0, f"inference_matrix_{inp.name}_{out.name}")


def _prefix_counts(tokens: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for token in tokens:
        prefix = token.split(":", 1)[0] if ":" in token else token
        counts[prefix] = counts.get(prefix, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:12])


def _quality_label(status: str, reward: float, passed: bool, files: dict[str, str] | None = None, hq_score: float | None = None) -> str:
    if status == "skipped":
        return "not_applicable"
    if status != "ok":
        return "needs_repair"
    if files and {"pdb", "dcd"}.issubset(files):
        if hq_score is not None and hq_score < 0.55:
            return "artifact_complete_needs_physics"
        return "artifact_complete"
    if passed or reward >= 0.5:
        return "usable_smoke"
    if reward >= 0.25:
        return "partial"
    return "weak"


def _run_case(
    model: Any,
    vocab: Any,
    device: Any,
    inp: InputCase,
    out: OutputCase,
    *,
    output_dir: Path,
    max_steps: int,
    max_source_tokens: int,
    sample: bool,
    temperature: float,
    retries: int,
    trajectory_frames: int,
    trajectory_max_atoms: int,
    trajectory_formats: list[str],
    trajectory_backend: str,
    trajectory_force_step_size: float,
    trajectory_strict: bool,
    trajectory_max_residues: int,
    trajectory_score_target_frames: int,
) -> dict[str, Any]:
    started = time.time()
    case_id = f"{inp.name}__to__{out.name}"
    case_dir = output_dir / "cases" / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    example = _case_example(inp, out, max_steps)
    best_tokens: list[str] = []
    best_verification = None
    status = "ok"
    error = ""
    for attempt in range(max(1, retries)):
        tokens = complete_graph_tokens(
            model,
            vocab,
            example,
            device,
            max_steps=max_steps,
            sample=sample or attempt > 0,
            temperature=temperature,
            max_source_tokens=max_source_tokens,
        )
        verification = verify_example_tokens(example, tokens)
        if best_verification is None or verification.reward > best_verification.reward:
            best_tokens = tokens
            best_verification = verification
        if verification.passed:
            break

    files: dict[str, str] = {}
    structure_export: dict[str, Any] = {}
    if out.requires_coordinate_slots:
        prediction = predict_uma_coordinate_frame(
            model,
            vocab,
            example,
            device,
            target_tokens=best_tokens or example.target_tokens,
            max_source_tokens=max_source_tokens,
            max_target_tokens=max(max_steps, len(best_tokens), 1),
            max_uma_coordinate_atoms=trajectory_max_atoms,
        )
        atoms = prediction.get("atoms") or []
        coords = prediction.get("coordinates") or []
        symbols = prediction.get("symbols") or [str(atom.get("element", "C")) for atom in atoms]
        full_geometry = derive_full_cartesian_geometry(example, max_atoms=trajectory_max_atoms, max_residues=trajectory_max_residues)
        full_atoms = full_geometry.get("atoms", [])
        full_bonds = full_geometry.get("bonds", [])
        full_coords = full_geometry.get("coordinates")
        full_size_export = bool(full_atoms)
        if full_size_export:
            atoms = full_atoms
            coords = full_coords or generated_initial_coordinates(atoms, coords)
            symbols = [str(atom.get("element", "C")) for atom in atoms]
        if atoms and coords:
            if full_size_export and (trajectory_backend == "proxy" or len(atoms) > 512):
                frames = smooth_trajectory_frames(atoms, coords, frame_count=trajectory_frames, temperature_k=(example.metadata or {}).get("temperature"))
                oracle_results = []
            else:
                frames, oracle_results = _rollout_structure_frames(
                    symbols,
                    coords,
                    example,
                    frame_count=trajectory_frames,
                    backend=trajectory_backend,
                    strict=trajectory_strict,
                    repo_path="data/external_repos/fairchem",
                    model_name="uma-s-1p2",
                    task_name="omol",
                    device_name=str(device),
                    force_step_size=trajectory_force_step_size,
                )
            files = _write_structure_outputs(case_dir / "structure_dynamics", atoms, frames, full_bonds if full_size_export else [], trajectory_formats)
            expected_residues = trajectory_max_residues if inp.name == "uniprot_500_protein" else None
            hq_score = high_quality_trajectory_score(atoms, frames, target_frames=trajectory_score_target_frames, expected_residues=expected_residues)
            structure_export = {
                "atoms": len(atoms),
                "bonds": len(full_bonds) if full_size_export else 0,
                "all_atom_cartesian": full_size_export,
                "frames": len(frames),
                "full_size_export": full_size_export,
                "max_residues": trajectory_max_residues,
                "formats": trajectory_formats,
                "oracle_backend": trajectory_backend,
                "oracle_available_rate": sum(1 for item in oracle_results if item.get("available")) / max(1, len(oracle_results)),
                "files": files,
                "long_high_quality_scoring": hq_score,
            }
        else:
            status = "skipped"
            error = "No UMA coordinate query slots were available for this input; structure-dynamics export requires explicit sequence/molecule/BioSELFIES fields."

    verification = best_verification
    hq_value = structure_export.get("long_high_quality_scoring", {}).get("long_hq_score") if structure_export else None
    result = {
        "case_id": case_id,
        "input": inp.name,
        "input_modality": inp.modality,
        "output": out.name,
        "output_modality": out.modality,
        "status": status,
        "error": error,
        "token_count": len(best_tokens),
        "unique_token_count": len(set(best_tokens)),
        "tokens": best_tokens,
        "token_prefix_counts": _prefix_counts(best_tokens),
        "verification": verification.metric_dict(prefix="") if verification is not None else {},
        "quality": _quality_label(status, float(verification.reward if verification else 0.0), bool(verification.passed if verification else False), files, hq_value),
        "structure_export": structure_export,
        "duration_s": round(time.time() - started, 4),
    }
    (case_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _necessary_update(result: dict[str, Any]) -> str:
    if result["status"] == "skipped":
        return "Add structured sequence/molecule extraction for this input before requesting structure-dynamics export."
    if result["quality"] == "artifact_complete_needs_physics":
        return "Artifact generation works, but the long-simulation proxy score is below the high-quality threshold; run stricter FairChem/OpenMM relaxation and improve geometry constraints."
    if result["quality"] == "artifact_complete":
        return "No blocking update for artifact generation; evaluate physical plausibility with strict FairChem/UMA or OpenMM rollout before scientific use."
    if result["quality"] == "usable_smoke":
        return "No blocking update for smoke inference; inspect generated token families for task specificity."
    if result["quality"] == "partial":
        return "Improve output conditioning and add task-specific verifier checks for this modality pair."
    return "Needs repair or stronger modality-specific decoding constraints before relying on this output."


def _write_report(path: Path, results: list[dict[str, Any]], args: argparse.Namespace) -> None:
    ok = sum(1 for row in results if row["status"] == "ok")
    skipped = sum(1 for row in results if row["status"] == "skipped")
    failed = sum(1 for row in results if row["status"] not in {"ok", "skipped"})
    artifact_complete = sum(1 for row in results if row["quality"] in {"artifact_complete", "artifact_complete_needs_physics"})
    lines = [
        "# Dev-1 Trained-Model Inference Modality Matrix",
        "",
        f"Generated: `{datetime.now(timezone.utc).isoformat()}`",
        f"Checkpoint: `{args.checkpoint}`",
        f"Vocab: `{args.vocab}`",
        f"Output directory: `{args.output_dir}`",
        f"Device: `{args.device}`",
        f"Max steps: `{args.max_steps}`",
        "",
        "## Summary",
        "",
        f"- Total modality-pair cases: `{len(results)}`",
        f"- Completed inference cases: `{ok}`",
        f"- Skipped/not applicable cases: `{skipped}`",
        f"- Failed cases: `{failed}`",
        f"- Structure artifact-complete cases: `{artifact_complete}`",
        "",
        "A skipped structure-dynamics case means the input was text-only or raw graph-only and therefore had no structured sequence/molecule/BioSELFIES fields from which the coordinate head could derive `UMA_COORD_QUERY:*` atom slots.",
        "",
        "## Case Matrix",
        "",
        "| Input | Output | Status | Quality | Reward | Tokens | Atoms | Frames | HQ score | Artifacts | Necessary update |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in results:
        reward = row.get("verification", {}).get("reward", 0.0)
        files = row.get("structure_export", {}).get("files", {})
        score = row.get("structure_export", {}).get("long_high_quality_scoring", {})
        artifact_text = ", ".join(f"`{name}`" for name in files) if files else ""
        lines.append(
            "| {input_modality} | {output_modality} | {status} | {quality} | {reward:.3f} | {token_count} | {atoms} | {frames} | {hq_score} | {artifacts} | {update} |".format(
                input_modality=row["input_modality"],
                output_modality=row["output_modality"],
                status=row["status"],
                quality=row["quality"],
                reward=float(reward),
                token_count=int(row["token_count"]),
                atoms=row.get("structure_export", {}).get("atoms", ""),
                frames=row.get("structure_export", {}).get("frames", ""),
                hq_score=score.get("long_hq_score", ""),
                artifacts=artifact_text,
                update=_necessary_update(row),
            )
        )
    lines.extend(["", "## Output Artifacts", ""])
    for row in results:
        files = row.get("structure_export", {}).get("files", {})
        if not files:
            continue
        lines.append(f"### `{row['case_id']}`")
        lines.append("")
        for name, file_path in files.items():
            lines.append(f"- `{name}`: `{file_path}`")
        score = row.get("structure_export", {}).get("long_high_quality_scoring", {})
        if score:
            lines.append(f"- Long high-quality simulation proxy score: `{score.get('long_hq_score')}`")
            lines.append(f"- Atom/frame coverage: `{score.get('atom_count')}` atoms, `{score.get('frame_count')}` frames, `{score.get('residue_count')}` residues/bases")
        lines.append("")
    lines.extend(["## Quality Notes", ""])
    lines.append("- `artifact_complete` means the structure-dynamics path produced at least a multi-model PDB and a DCD trajectory for that modality pair.")
    lines.append("- Generated PDB files include sequence-derived `HELIX` secondary-structure records plus cartoon-intent `REMARK` records. No PyMOL, ChimeraX, or other viewer sidecar scripts are emitted.")
    lines.append("- PDB does not have a portable standard field that forces a viewer representation. The portable representation signal is the encoded secondary structure; viewers still choose their own default drawing mode unless configured by the user.")
    lines.append("- Structure-dynamics cases are scored with a strict long-run simulation proxy profile: frame coverage, full-size residue coverage, sampled clash rate, step RMSD smoothness, max-step stability, and radius-of-gyration stability. This is not a substitute for strict FairChem/OpenMM rescoring.")
    lines.append("- `usable_smoke` means verifier reward passed the current broad graph verifier threshold or the verifier passed.")
    lines.append("- `partial` means the model produced tokens but weak task-specific evidence; these cases need modality-specific verifiers and decoding constraints.")
    lines.append("- The matrix uses the deterministic `proxy` coordinate rollout by default. Use strict `fairchem` rollout for physical plausibility checks once UMA weights and runtime budget are available.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run trained UGM inference over input/output modality matrix with tqdm logging.")
    parser.add_argument("--config", default="config/inference/biomed_annotations_affinity_plus_original_250m_inference.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--vocab")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--report", default="planning/INFERENCE-MODALITY-MATRIX-DEV1.md")
    parser.add_argument("--log-dir", default="logs/inference_modality_matrix")
    parser.add_argument("--device")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-source-tokens", type=int)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--trajectory-frames", type=int)
    parser.add_argument("--trajectory-max-atoms", type=int)
    parser.add_argument("--trajectory-formats", default=None)
    parser.add_argument("--trajectory-oracle-backend", default=None)
    parser.add_argument("--trajectory-force-step-size", type=float, default=None)
    parser.add_argument("--trajectory-strict-oracle", action="store_true")
    args = parser.parse_args()

    cfg = load_config([args.config]) if args.config else {}
    infer_cfg = cfg.get("inference", {})
    structure_cfg = infer_cfg.get("structure_dynamics", {}) if isinstance(infer_cfg.get("structure_dynamics"), dict) else {}
    args.checkpoint = args.checkpoint or infer_cfg.get("checkpoint")
    args.vocab = args.vocab or infer_cfg.get("vocab")
    args.device = args.device or infer_cfg.get("device", "cuda")
    args.max_steps = int(args.max_steps or infer_cfg.get("max_steps", 16))
    args.max_source_tokens = int(args.max_source_tokens or infer_cfg.get("max_source_tokens", 512))
    args.temperature = float(args.temperature if args.temperature is not None else infer_cfg.get("temperature", 1.0))
    sample = bool(args.sample or infer_cfg.get("sample", False))
    trajectory_frames = int(args.trajectory_frames or structure_cfg.get("frames", 4))
    trajectory_max_atoms = int(args.trajectory_max_atoms or structure_cfg.get("max_atoms", 32))
    trajectory_max_residues = int(structure_cfg.get("max_residues", 500))
    trajectory_score_target_frames = int(structure_cfg.get("score_target_frames", max(64, trajectory_frames)))
    trajectory_formats = [item.strip() for item in str(args.trajectory_formats or structure_cfg.get("trajectory_formats", "dcd,xyz")).split(",") if item.strip()]
    trajectory_backend = str(args.trajectory_oracle_backend or structure_cfg.get("oracle_backend", "proxy"))
    trajectory_force_step_size = float(args.trajectory_force_step_size or structure_cfg.get("force_step_size", 0.02))
    trajectory_strict = bool(args.trajectory_strict_oracle or structure_cfg.get("strict_oracle", False))
    if not args.checkpoint or not args.vocab:
        raise SystemExit("Provide checkpoint/vocab via --config or explicit args")

    run_id = _utc_run_id()
    output_dir = Path(args.output_dir or f"outputs/inference/dev1_modality_matrix/{run_id}")
    args.output_dir = str(output_dir)
    log_path = Path(args.log_dir) / run_id / "inference_matrix.log"
    _setup_logging(log_path)
    logging.info("Loading model checkpoint=%s vocab=%s device=%s", args.checkpoint, args.vocab, args.device)
    model, vocab, device = load_model_for_inference(args.checkpoint, args.vocab, args.device)
    logging.info("Loaded model; running modality matrix into %s", output_dir)

    inputs = input_cases()
    outputs = output_cases()
    pairs = [(inp, out) for inp in inputs for out in outputs]
    results: list[dict[str, Any]] = []
    summary_jsonl = output_dir / "summary.jsonl"
    summary_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with summary_jsonl.open("w", encoding="utf-8") as summary_handle:
        for inp, out in tqdm(pairs, desc="inference modality matrix", unit="case"):
            case_id = f"{inp.name}__to__{out.name}"
            logging.info("Starting case %s input=%s output=%s", case_id, inp.modality, out.modality)
            try:
                result = _run_case(
                    model,
                    vocab,
                    device,
                    inp,
                    out,
                    output_dir=output_dir,
                    max_steps=args.max_steps,
                    max_source_tokens=args.max_source_tokens,
                    sample=sample,
                    temperature=args.temperature,
                    retries=args.retries,
                    trajectory_frames=trajectory_frames,
                    trajectory_max_atoms=trajectory_max_atoms,
                    trajectory_formats=trajectory_formats,
                    trajectory_backend=trajectory_backend,
                    trajectory_force_step_size=trajectory_force_step_size,
                    trajectory_strict=trajectory_strict,
                    trajectory_max_residues=trajectory_max_residues,
                    trajectory_score_target_frames=trajectory_score_target_frames,
                )
            except Exception as exc:
                result = {
                    "case_id": case_id,
                    "input": inp.name,
                    "input_modality": inp.modality,
                    "output": out.name,
                    "output_modality": out.modality,
                    "status": "failed",
                    "error": f"{exc.__class__.__name__}: {exc}",
                    "token_count": 0,
                    "unique_token_count": 0,
                    "tokens": [],
                    "token_prefix_counts": {},
                    "verification": {},
                    "quality": "needs_repair",
                    "structure_export": {},
                    "duration_s": 0.0,
                }
            results.append(result)
            summary_handle.write(json.dumps(result, sort_keys=True) + "\n")
            summary_handle.flush()
            logging.info("Finished case %s status=%s quality=%s reward=%.3f", case_id, result["status"], result["quality"], float(result.get("verification", {}).get("reward", 0.0)))

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps({"run_id": run_id, "results": results}, indent=2) + "\n", encoding="utf-8")
    report_path = Path(args.report)
    _write_report(report_path, results, args)
    logging.info("Wrote summary=%s report=%s log=%s", summary_path, report_path, log_path)
    print(json.dumps({"run_id": run_id, "output_dir": str(output_dir), "summary_jsonl": str(summary_jsonl), "summary_json": str(summary_path), "report": str(report_path), "log": str(log_path)}, indent=2))


if __name__ == "__main__":
    main()
