#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tqdm.auto import tqdm

from iska_reasoner.graph.schema import GraphExample
from iska_reasoner.inference.contact_priors import predict_esm_contacts


def _sequence_from_graph_row(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    for key in ("protein_sequence", "sequence", "aa_sequence"):
        value = metadata.get(key) or row.get(key)
        if value:
            return "".join(ch for ch in str(value).upper() if ch.isalpha())
    residues: list[str] = []
    for node in row.get("nodes", []) or []:
        if node.get("type") == "protein_sequence" and node.get("value"):
            return "".join(ch for ch in str(node["value"]).upper() if ch.isalpha())
        if node.get("type") == "amino_acid" and node.get("value"):
            residues.append(str(node["value"]).strip().upper()[:1])
    return "".join(residues)


def _iter_rows(path: Path) -> Iterable[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".json"}:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            if suffix == ".json":
                payload = json.load(handle)
                rows = payload if isinstance(payload, list) else payload.get("rows", [])
                for row in rows:
                    yield dict(row)
            else:
                for line in handle:
                    line = line.strip()
                    if line:
                        yield json.loads(line)
        return
    dialect = "excel-tab" if suffix in {".tsv", ".tab"} else "excel"
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle, dialect=dialect)
        for row in reader:
            yield dict(row)


def _contact_bin(probability: float) -> str:
    idx = int(max(0, min(63, round(float(probability) * 63))))
    return f"b{idx:02d}"


def _augment_graph_row(row: dict[str, Any], contacts: list[dict[str, Any]]) -> dict[str, Any]:
    if not row.get("nodes"):
        row["esm_contacts"] = contacts
        return row
    nodes = list(row.get("nodes") or [])
    edges = list(row.get("edges") or [])
    target_tokens = list(row.get("target_tokens") or [])
    existing_ids = {str(node.get("id")) for node in nodes}
    root_id = "esm_contact_prior"
    if root_id not in existing_ids:
        nodes.append({"id": root_id, "type": "esm_contact_prior", "value": "protein_contact_map", "features": {"source": "ESM", "count": len(contacts)}})
        edges.append({"src": "task", "dst": root_id, "type": "has_contact_prior", "features": {}})
        existing_ids.add(root_id)
    target_tokens.extend(["CONTACT_PATCH:esm_prior", "ESM_CONTACT:enabled"])
    for idx, contact in enumerate(contacts):
        try:
            i = int(contact["i"])
            j = int(contact["j"])
            prob = float(contact["probability"])
        except Exception:
            continue
        bin_label = _contact_bin(prob)
        node_id = f"esm_contact_{idx}"
        if node_id not in existing_ids:
            nodes.append(
                {
                    "id": node_id,
                    "type": "esm_contact_pair",
                    "value": f"{i}:{j}:{bin_label}",
                    "features": {"i": i, "j": j, "probability": round(prob, 6), "bin": bin_label, "source": contact.get("source", "esm_contact_regression")},
                }
            )
            edges.append({"src": root_id, "dst": node_id, "type": "contains_contact_pair", "features": {}})
        src = f"protein_{i - 1}"
        dst = f"protein_{j - 1}"
        if src in existing_ids and dst in existing_ids:
            edges.append({"src": src, "dst": dst, "type": "esm_predicted_contact", "features": {"probability": round(prob, 6), "bin": bin_label}})
        target_tokens.append(f"ESM_CONTACT:{bin_label}")
    row["nodes"] = nodes
    row["edges"] = edges
    row["target_tokens"] = list(dict.fromkeys(str(token) for token in target_tokens))
    row.setdefault("metadata", {})["esm_contact_prior"] = {"source": "ESM", "contacts": len(contacts)}
    GraphExample.from_dict(row)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute/cache ESM contact priors and optionally augment JSONL graph rows.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir", default="data/processed/esm_contact_priors/cache")
    parser.add_argument("--model", default="esm2_t33_650M_UR50D")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top-k", type=int, default=256)
    parser.add_argument("--min-probability", type=float, default=0.2)
    parser.add_argument("--min-separation", type=int, default=6)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as out:
        for row in tqdm(_iter_rows(input_path), desc="esm/contact_priors", unit="row"):
            sequence = _sequence_from_graph_row(row)
            existing_contacts = row.get("esm_contacts") or row.get("contact_priors") or row.get("predicted_contacts")
            if existing_contacts:
                contacts = existing_contacts.get("contacts", []) if isinstance(existing_contacts, dict) else existing_contacts
                row = _augment_graph_row(row, list(contacts))
            elif sequence:
                result = predict_esm_contacts(
                    sequence,
                    model_name=args.model,
                    device=args.device,
                    top_k=args.top_k,
                    min_probability=args.min_probability,
                    min_separation=args.min_separation,
                    cache_dir=args.cache_dir,
                    strict=args.strict,
                )
                if result.contacts:
                    row = _augment_graph_row(row, result.contacts)
                else:
                    row.setdefault("metadata", {})["esm_contact_prior"] = result.to_dict()
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
            if args.limit and count >= args.limit:
                break
    print(json.dumps({"rows": count, "output": str(output_path)}, indent=2))


if __name__ == "__main__":
    main()
