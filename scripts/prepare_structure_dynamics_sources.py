#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tqdm.auto import tqdm

from iska_reasoner.data.graphify import graphify_rows
from iska_reasoner.data.multimodal import iter_synthetic_multimodal_examples
from iska_reasoner.utils.io import ensure_dir, read_jsonl, write_jsonl


def _atom_element(name: str, fallback: str = "C") -> str:
    text = re.sub(r"[^A-Za-z]", "", name).strip()
    if not text:
        return fallback
    if len(text) >= 2 and text[:2].capitalize() in {"Cl", "Br", "Na", "Mg", "Ca", "Fe", "Zn", "Mn", "Cu", "Co", "Ni"}:
        return text[:2].capitalize()
    return text[0].upper()


def parse_pdb(path: Path) -> dict[str, Any]:
    atoms: list[dict[str, Any]] = []
    frames: list[list[list[float]]] = []
    current: list[list[float]] = []
    serial_to_index: dict[int, int] = {}
    bonds: list[dict[str, Any]] = []
    saw_model = False
    in_model = False

    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.rstrip("\n")
        record = line[:6].strip().upper()
        if record == "MODEL":
            saw_model = True
            in_model = True
            current = []
            continue
        if record == "ENDMDL":
            if current:
                frames.append(current)
            current = []
            in_model = False
            continue
        if record in {"ATOM", "HETATM"}:
            try:
                serial = int(line[6:11])
            except Exception:
                serial = len(current) + 1
            name = line[12:16].strip() or f"A{len(current) + 1}"
            residue = line[17:20].strip() or ("MOL" if record == "HETATM" else "UNK")
            chain = (line[21:22].strip() or "A")[:1]
            try:
                residue_index = int(line[22:26])
            except Exception:
                residue_index = 1
            try:
                coord = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            except Exception:
                fields = line.split()
                coord = [float(fields[-6]), float(fields[-5]), float(fields[-4])] if len(fields) >= 9 else [0.0, 0.0, 0.0]
            element = (line[76:78].strip() if len(line) >= 78 else "") or _atom_element(name)
            if not frames and (not saw_model or in_model):
                if serial not in serial_to_index:
                    serial_to_index[serial] = len(atoms)
                    atoms.append(
                        {
                            "element": element,
                            "name": name,
                            "residue": residue,
                            "chain": chain,
                            "residue_index": residue_index,
                            "record_type": record,
                        }
                    )
            current.append(coord)
            continue
        if record == "CONECT":
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                src_serial = int(parts[1])
            except Exception:
                continue
            for dst_text in parts[2:]:
                try:
                    dst_serial = int(dst_text)
                except Exception:
                    continue
                if src_serial in serial_to_index and dst_serial in serial_to_index:
                    src = serial_to_index[src_serial]
                    dst = serial_to_index[dst_serial]
                    if src != dst:
                        bonds.append({"src": min(src, dst), "dst": max(src, dst), "bond_type": "single"})
    if current:
        frames.append(current)
    dedup_bonds = list({(b["src"], b["dst"], b["bond_type"]): b for b in bonds}.values())
    sequence = "".join(atom.get("residue", "X")[0] for atom in atoms if atom.get("name") == "CA")[:512]
    return {
        "prompt": f"Generate all-atom graph records and trajectory serialization for {path.name}.",
        "task": "conformer_trajectory" if len(frames) > 1 else "structure_generation",
        "protein_sequence": sequence,
        "atoms": atoms,
        "bonds": dedup_bonds,
        "frames": frames,
        "temperature": 300,
        "oracle": "pdb_geometry",
        "function_description": "Local structure/dynamics evaluation row parsed from PDB coordinates.",
        "source_path": str(path),
    }


def iter_csv(path: Path, limit: int | None = None) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t" if path.suffix.lower() in {".tsv", ".tab"} else ",")
        for idx, row in enumerate(reader):
            if limit is not None and idx >= limit:
                break
            yield dict(row)


def iter_json_or_jsonl(path: Path, limit: int | None = None) -> Iterable[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        for idx, row in enumerate(read_jsonl(path)):
            if limit is not None and idx >= limit:
                break
            yield row
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload if isinstance(payload, list) else payload.get("rows", [])
    for idx, row in enumerate(rows):
        if limit is not None and idx >= limit:
            break
        if isinstance(row, dict):
            yield row


def iter_fasta(path: Path, limit: int | None = None) -> Iterable[dict[str, Any]]:
    header = ""
    seq: list[str] = []
    emitted = 0
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if seq:
                yield {"prompt": header, "protein_sequence": "".join(seq), "task": "structure_generation", "temperature": 300}
                emitted += 1
                if limit is not None and emitted >= limit:
                    return
            header = line[1:]
            seq = []
        else:
            seq.append(line)
    if seq and (limit is None or emitted < limit):
        yield {"prompt": header, "protein_sequence": "".join(seq), "task": "structure_generation", "temperature": 300}


def iter_rows(paths: list[Path], limit: int | None) -> Iterable[dict[str, Any]]:
    emitted = 0
    for path in paths:
        remaining = None if limit is None else max(0, limit - emitted)
        if remaining == 0:
            return
        suffix = path.suffix.lower()
        if suffix in {".pdb", ".ent"}:
            iterator = [parse_pdb(path)]
        elif suffix in {".fa", ".fasta", ".faa", ".fna"}:
            iterator = iter_fasta(path, remaining)
        elif suffix in {".csv", ".tsv", ".tab"}:
            iterator = iter_csv(path, remaining)
        else:
            iterator = iter_json_or_jsonl(path, remaining)
        for row in iterator:
            yield row
            emitted += 1
            if limit is not None and emitted >= limit:
                return


def collect_paths(inputs: list[str] | None, input_dirs: list[str] | None, patterns: list[str]) -> list[Path]:
    paths = [Path(item) for item in inputs or []]
    for directory in input_dirs or []:
        base = Path(directory)
        for pattern in patterns:
            paths.extend(sorted(base.rglob(pattern)))
    return sorted(dict.fromkeys(path for path in paths if path.exists() and path.is_file()))


STRUCTURE_FILE_SUFFIXES = {".pdb", ".ent", ".cif", ".mmcif", ".bcif", ".sdf", ".mol2", ".xtc", ".trr", ".dcd"}


def assert_no_structure_file_training(paths: list[Path], purpose: str, allow_structure_file_training: bool) -> None:
    if purpose != "train" or allow_structure_file_training:
        return
    blocked = [str(path) for path in paths if path.suffix.lower() in STRUCTURE_FILE_SUFFIXES]
    if not blocked:
        raise SystemExit(
            "Structure/dynamics row preparation is blocked for training by default. "
            "Use sequence/SELFIES rows for early training; run this script with --purpose validation, --purpose test, or --purpose eval for structure-side checks."
        )
    if blocked:
        sample = "\n".join(blocked[:20])
        raise SystemExit(
            "Actual structure files are blocked for training by default. "
            "Use sequence/SELFIES training data first, and reserve PDB/mmCIF/SDF/trajectory files for validation, test, "
            f"or later explicitly approved stages. Blocked files:\n{sample}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare full local structure/dynamics graph-to-graph rows.")
    parser.add_argument("--input", action="append", help="PDB/JSON/JSONL/CSV/TSV/FASTA source file.")
    parser.add_argument("--input-dir", action="append", help="Directory scanned recursively for supported source files.")
    parser.add_argument("--pattern", action="append", default=["*.pdb", "*.ent", "*.jsonl", "*.json", "*.csv", "*.tsv", "*.fa", "*.fasta", "*.faa", "*.fna"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-name", default="local_structure_dynamics_graph_to_graph")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--synthetic-if-empty", action="store_true")
    parser.add_argument("--synthetic-count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--purpose", choices=["train", "validation", "test", "eval"], default="eval")
    parser.add_argument("--allow-structure-file-training", action="store_true", help="Explicit override for future stages; off by default.")
    args = parser.parse_args()

    paths = collect_paths(args.input, args.input_dir, args.pattern)
    assert_no_structure_file_training(paths, args.purpose, args.allow_structure_file_training)
    rows = list(tqdm(iter_rows(paths, args.limit), desc="structure/rows")) if paths else []
    if not rows and args.synthetic_if_empty:
        rows = list(iter_synthetic_multimodal_examples(count=args.synthetic_count, seed=args.seed))
    if not rows:
        raise SystemExit("No structure/dynamics rows found. Provide --input/--input-dir or use --synthetic-if-empty.")
    graphs = list(tqdm(graphify_rows(rows, args.dataset_name), total=len(rows), desc="structure/graphify"))
    ensure_dir(Path(args.output).parent)
    count = write_jsonl(args.output, graphs)
    print(json.dumps({"rows": len(rows), "graphs": count, "output": args.output, "sources": len(paths)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
