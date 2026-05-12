#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iska_reasoner.graph.schema import Edge, GraphExample, Node
from iska_reasoner.data.multimodal import graphify_multimodal, records_to_multimodel_pdb, records_to_xyz_trajectory, write_mdtraj_trajectory
from iska_reasoner.inference.generate import complete_graph_tokens, load_model_for_inference, predict_uma_coordinate_frame
from iska_reasoner.oracles import score_uma_coordinate_candidate
from iska_reasoner.tools import verify_example_tokens
from iska_reasoner.utils.config import load_config


def example_from_text(text: str) -> GraphExample:
    words = text.split()[:64]
    nodes = [Node(id=f"w{i}", type="token", value=word) for i, word in enumerate(words)]
    edges = [Edge(src=f"w{i}", dst=f"w{i+1}", type="next_token") for i in range(max(0, len(words) - 1))]
    return GraphExample(id="inference_text", task="inference", nodes=nodes, edges=edges, target_tokens=["<UNK>"])


def _temperature_kelvin(example: GraphExample) -> float | None:
    for node in example.nodes:
        if node.type != "temperature":
            continue
        for key in ("kelvin", "kelvin_clamped", "temperature_k"):
            try:
                return float(node.features[key])
            except Exception:
                pass
        try:
            return float(str(node.value).rstrip("Kk"))
        except Exception:
            return None
    try:
        value = (example.metadata or {}).get("temperature")
        return float(value) if value is not None else None
    except Exception:
        return None


def _fallback_displacement(coords: list[list[float]], frame_idx: int, temperature_k: float | None) -> list[list[float]]:
    temp_scale = 1.0 + max(0.0, min(100.0, float(temperature_k or 300.0) - 300.0)) / 100.0
    step = 0.015 * temp_scale
    moved: list[list[float]] = []
    for atom_idx, xyz in enumerate(coords):
        phase = frame_idx + atom_idx * 0.37
        moved.append(
            [
                float(xyz[0]) + step * math.sin(phase),
                float(xyz[1]) + step * math.cos(phase * 0.7),
                float(xyz[2]) + step * math.sin(phase * 0.43 + 0.5),
            ]
        )
    return moved


def _rollout_structure_frames(
    symbols: list[str],
    initial_coords: list[list[float]],
    example: GraphExample,
    *,
    frame_count: int,
    backend: str,
    strict: bool,
    repo_path: str,
    model_name: str,
    task_name: str,
    device_name: str,
    force_step_size: float,
) -> tuple[list[list[list[float]]], list[dict[str, Any]]]:
    frames = [[list(map(float, xyz[:3])) for xyz in initial_coords]]
    oracle_results: list[dict[str, Any]] = []
    temperature_k = _temperature_kelvin(example)
    current = frames[0]
    for frame_idx in range(1, max(1, int(frame_count))):
        result = score_uma_coordinate_candidate(
            symbols,
            current,
            temperature_k=temperature_k,
            backend=backend,
            strict=strict,
            repo_path=repo_path,
            model_name=model_name,
            task_name=task_name,
            device=device_name,
        )
        oracle_results.append(asdict(result))
        if result.available and result.forces_ev_per_a:
            forces = result.forces_ev_per_a[: len(current)]
            current = [
                [
                    float(xyz[0]) + force_step_size * float(force[0]),
                    float(xyz[1]) + force_step_size * float(force[1]),
                    float(xyz[2]) + force_step_size * float(force[2]),
                ]
                for xyz, force in zip(current, forces, strict=False)
            ]
        elif strict:
            raise RuntimeError(f"UMA coordinate rollout failed: {result.message}")
        else:
            current = _fallback_displacement(current, frame_idx, temperature_k)
        frames.append(current)
    return frames, oracle_results


def _write_structure_outputs(
    prefix: Path,
    atoms: list[dict[str, Any]],
    frames: list[list[list[float]]],
    bonds: list[dict[str, Any]],
    formats: list[str],
) -> dict[str, str]:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    pdb_path = prefix.with_suffix(".pdb")
    pdb_path.write_text(records_to_multimodel_pdb(atoms, frames, bonds), encoding="utf-8")
    written["pdb"] = str(pdb_path)
    for fmt in formats:
        fmt = fmt.strip().lower().lstrip(".")
        if not fmt or fmt == "pdb":
            continue
        if fmt == "xyz":
            xyz_path = prefix.with_suffix(".xyz")
            xyz_path.write_text(records_to_xyz_trajectory(atoms, frames), encoding="utf-8")
            written["xyz"] = str(xyz_path)
        elif fmt in {"dcd", "xtc", "trr", "nc"}:
            traj_path = prefix.with_suffix(f".{fmt}")
            write_mdtraj_trajectory(traj_path, atoms, frames, bonds)
            written[fmt] = str(traj_path)
        else:
            raise ValueError(f"Unsupported trajectory format: {fmt}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Run random-order graph token inference.")
    parser.add_argument("--config", action="append", help="YAML config path. Values are used as defaults.")
    parser.add_argument("--checkpoint")
    parser.add_argument("--vocab")
    parser.add_argument("--text")
    parser.add_argument("--graph-json")
    parser.add_argument("--graph-json-file")
    parser.add_argument("--multimodal-json", help="JSON row with prompt/protein_sequence/selfies/dna_sequence/rna_sequence/atoms/bonds/frames fields.")
    parser.add_argument("--multimodal-json-file", help="Path to a JSON row with prompt/protein_sequence/selfies/dna_sequence/rna_sequence/atoms/bonds/frames fields.")
    parser.add_argument("--task", default="structure_generation")
    parser.add_argument("--prompt")
    parser.add_argument("--protein-sequence")
    parser.add_argument("--selfies")
    parser.add_argument("--smiles")
    parser.add_argument("--dna-sequence")
    parser.add_argument("--rna-sequence")
    parser.add_argument("--temperature-k", type=float)
    parser.add_argument("--render-input-pdb", action="store_true", help="Render atoms/frames from the provided multimodal input row when available.")
    parser.add_argument("--output-modality", help="Optional requested output modality, e.g. structure_dynamics.")
    parser.add_argument("--structure-output-prefix", help="Prefix for structure-dynamics files. Writes .pdb plus requested trajectory formats.")
    parser.add_argument("--trajectory-formats", default=None, help="Comma-separated MD trajectory formats to write beside the PDB. Default: dcd,xyz.")
    parser.add_argument("--trajectory-frames", type=int, default=None, help="Number of frames for structure-dynamics export.")
    parser.add_argument("--trajectory-max-atoms", type=int, default=None, help="Maximum coordinate-query atoms to export.")
    parser.add_argument("--trajectory-force-step-size", type=float, default=None, help="UMA force rollout step size in Angstrom per eV/A proxy units.")
    parser.add_argument("--trajectory-oracle-backend", default=None, help="Coordinate rollout oracle backend: fairchem/uma or proxy.")
    parser.add_argument("--trajectory-strict-oracle", action="store_true", help="Fail if UMA/FairChem force rollout is unavailable.")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-source-tokens", type=int)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--device")
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--output", help="Optional JSON output path.")
    args = parser.parse_args()
    cfg = load_config(args.config) if args.config else {}
    infer_cfg = cfg.get("inference", {})
    checkpoint = args.checkpoint or infer_cfg.get("checkpoint")
    vocab_path = args.vocab or infer_cfg.get("vocab")
    if not checkpoint or not vocab_path:
        raise SystemExit("Provide --checkpoint/--vocab or --config with inference.checkpoint and inference.vocab")
    max_steps = args.max_steps if args.max_steps is not None else int(infer_cfg.get("max_steps", 8))
    max_source_tokens = args.max_source_tokens if args.max_source_tokens is not None else int(infer_cfg.get("max_source_tokens", 128))
    sample = args.sample or bool(infer_cfg.get("sample", False))
    temperature = args.temperature if args.temperature is not None else float(infer_cfg.get("temperature", 1.0))
    device_name = args.device or infer_cfg.get("device", "cuda")
    retries = max(1, int(args.retries or infer_cfg.get("retries", 1)))
    output_modality = args.output_modality or infer_cfg.get("output_modality")
    model, vocab, device = load_model_for_inference(checkpoint, vocab_path, device_name)
    multimodal_row = None
    if args.graph_json_file:
        example = GraphExample.from_dict(json.loads(Path(args.graph_json_file).read_text(encoding="utf-8")))
    elif args.graph_json:
        example = GraphExample.from_dict(json.loads(args.graph_json))
    elif args.multimodal_json_file:
        multimodal_row = json.loads(Path(args.multimodal_json_file).read_text(encoding="utf-8"))
        example = graphify_multimodal(multimodal_row, 0, "inference_multimodal_graph_to_graph")
    elif args.multimodal_json:
        multimodal_row = json.loads(args.multimodal_json)
        example = graphify_multimodal(multimodal_row, 0, "inference_multimodal_graph_to_graph")
    elif any([args.prompt, args.protein_sequence, args.selfies, args.smiles, args.dna_sequence, args.rna_sequence, args.temperature_k]):
        multimodal_row = {
            "task": args.task,
            "prompt": args.prompt or args.text or "Generate graph-structured scientific output.",
            "protein_sequence": args.protein_sequence or "",
            "selfies": args.selfies or "",
            "smiles": args.smiles or "",
            "dna_sequence": args.dna_sequence or "",
            "rna_sequence": args.rna_sequence or "",
            "temperature": args.temperature_k,
        }
        example = graphify_multimodal(multimodal_row, 0, "inference_multimodal_graph_to_graph")
    elif args.text:
        example = example_from_text(args.text)
    else:
        raise SystemExit("Provide --text or --graph-json")
    best_tokens = []
    best_verification = None
    for attempt in range(retries):
        tokens = complete_graph_tokens(model, vocab, example, device, max_steps, sample or attempt > 0, temperature, max_source_tokens=max_source_tokens)
        verification = verify_example_tokens(example, tokens)
        if best_verification is None or verification.reward > best_verification.reward:
            best_tokens = tokens
            best_verification = verification
        if verification.passed:
            break
    verification = best_verification
    output = {"tokens": best_tokens, "verification": verification.metric_dict(prefix="")}
    if args.render_input_pdb and multimodal_row:
        atoms = multimodal_row.get("atoms") or []
        frames = multimodal_row.get("frames") or []
        bonds = multimodal_row.get("bonds") or []
        if atoms and frames:
            output["rendered_input_pdb"] = records_to_multimodel_pdb(atoms, frames, bonds)
    task_text = " ".join(str(item or "") for item in [output_modality, args.task, (example.metadata or {}).get("task")]).lower()
    wants_structure_dynamics = ("structure" in task_text and "dynamic" in task_text) or "trajectory" in task_text
    wants_structure_dynamics = wants_structure_dynamics or output_modality in {"structure_dynamics", "trajectory", "all_atom_trajectory"}
    if wants_structure_dynamics:
        structure_cfg = infer_cfg.get("structure_dynamics", {}) if isinstance(infer_cfg.get("structure_dynamics"), dict) else {}
        prefix_text = args.structure_output_prefix or structure_cfg.get("output_prefix")
        if prefix_text is None and args.output:
            prefix_text = str(Path(args.output).with_suffix(""))
        prefix = Path(prefix_text or "outputs/inference/structure_dynamics_generated")
        formats_text = args.trajectory_formats or structure_cfg.get("trajectory_formats") or "dcd,xyz"
        formats = [item.strip() for item in str(formats_text).split(",") if item.strip()]
        frame_count = int(args.trajectory_frames or structure_cfg.get("frames", 8))
        max_atoms = int(args.trajectory_max_atoms or structure_cfg.get("max_atoms", 64))
        force_step_size = float(args.trajectory_force_step_size or structure_cfg.get("force_step_size", 0.02))
        oracle_backend = str(args.trajectory_oracle_backend or structure_cfg.get("oracle_backend", "fairchem"))
        oracle_repo = str(structure_cfg.get("oracle_repo", "data/external_repos/fairchem"))
        oracle_model = str(structure_cfg.get("oracle_model", "uma-s-1p2"))
        oracle_task = str(structure_cfg.get("oracle_task", "omol"))
        oracle_device = str(structure_cfg.get("oracle_device", device_name))
        strict_oracle = bool(args.trajectory_strict_oracle or structure_cfg.get("strict_oracle", False))
        prediction = predict_uma_coordinate_frame(
            model,
            vocab,
            example,
            device,
            target_tokens=best_tokens or example.target_tokens,
            max_source_tokens=max_source_tokens,
            max_target_tokens=max(max_steps, len(best_tokens), 1),
            max_uma_coordinate_atoms=max_atoms,
        )
        atoms = prediction.get("atoms") or []
        coords = prediction.get("coordinates") or []
        symbols = prediction.get("symbols") or [str(atom.get("element", "C")) for atom in atoms]
        if not atoms or not coords:
            raise SystemExit("Structure-dynamics export requires a checkpoint with coordinate_head_enabled=true and nonempty UMA coordinate query slots.")
        frames, oracle_results = _rollout_structure_frames(
            symbols,
            coords,
            example,
            frame_count=frame_count,
            backend=oracle_backend,
            strict=strict_oracle,
            repo_path=oracle_repo,
            model_name=oracle_model,
            task_name=oracle_task,
            device_name=oracle_device,
            force_step_size=force_step_size,
        )
        written = _write_structure_outputs(prefix, atoms, frames, [], formats)
        output["structure_dynamics_export"] = {
            "files": written,
            "frames": len(frames),
            "atoms": len(atoms),
            "trajectory_formats": formats,
            "oracle_backend": oracle_backend,
            "strict_oracle": strict_oracle,
            "oracle_rollout": oracle_results,
            "note": "Coordinates are model-generated from UMA coordinate query slots; PDB/DCD/XYZ are generated trajectory artifacts, not supervised structure labels.",
        }
    payload = json.dumps(output, indent=2)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
