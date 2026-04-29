#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iska_reasoner.graph.schema import Edge, GraphExample, Node
from iska_reasoner.data.multimodal import graphify_multimodal, records_to_multimodel_pdb
from iska_reasoner.inference.generate import complete_graph_tokens, load_model_for_inference
from iska_reasoner.tools import verify_example_tokens
from iska_reasoner.utils.config import load_config


def example_from_text(text: str) -> GraphExample:
    words = text.split()[:64]
    nodes = [Node(id=f"w{i}", type="token", value=word) for i, word in enumerate(words)]
    edges = [Edge(src=f"w{i}", dst=f"w{i+1}", type="next_token") for i in range(max(0, len(words) - 1))]
    return GraphExample(id="inference_text", task="inference", nodes=nodes, edges=edges, target_tokens=["<UNK>"])


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
    payload = json.dumps(output, indent=2)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
