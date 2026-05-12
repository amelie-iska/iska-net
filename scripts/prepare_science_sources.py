#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from pathlib import Path
from typing import Any, Iterable, TextIO

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tqdm.auto import tqdm

from iska_reasoner.data.bioselfies import modality_bioselfies_fields
from iska_reasoner.data.graphify import graphify_rows
from iska_reasoner.utils.io import ensure_dir, write_jsonl


def _open_text(path: Path) -> TextIO:
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="")
    return path.open("r", encoding="utf-8", errors="replace", newline="")


def _suffixes(path: Path) -> list[str]:
    return [suffix.lower() for suffix in path.suffixes]


def _has_suffix(path: Path, candidates: set[str]) -> bool:
    return any(suffix in candidates for suffix in _suffixes(path))


def _looks_like_pubchem_cid_smiles(path: Path) -> bool:
    name = path.name.lower()
    return "cid-smiles" in name or "cid_smiles" in name


def iter_pubchem_cid_smiles(path: Path, limit: int | None = None) -> Iterable[dict[str, Any]]:
    with _open_text(path) as handle:
        for i, line in enumerate(handle):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            pieces = line.split(None, 1)
            if len(pieces) != 2:
                continue
            yield {"cid": pieces[0], "smiles": pieces[1], "Title": f"PubChem CID {pieces[0]}"}


def iter_csv(path: Path, limit: int | None = None) -> Iterable[dict[str, Any]]:
    if _looks_like_pubchem_cid_smiles(path):
        yield from iter_pubchem_cid_smiles(path, limit)
        return
    delimiter = "\t" if _has_suffix(path, {".tsv", ".tab"}) else ","
    with _open_text(path) as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for i, row in enumerate(reader):
            if limit is not None and i >= limit:
                break
            yield dict(row)


def iter_json_or_jsonl(path: Path, limit: int | None = None) -> Iterable[dict[str, Any]]:
    if _has_suffix(path, {".jsonl"}):
        with _open_text(path) as handle:
            for i, line in enumerate(handle):
                if limit is not None and i >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
        return
    with _open_text(path) as handle:
        payload = json.loads(handle.read())
    rows = payload if isinstance(payload, list) else payload.get("rows", [])
    for i, row in enumerate(rows):
        if limit is not None and i >= limit:
            break
        if isinstance(row, dict):
            yield row


def iter_fasta(path: Path, limit: int | None = None) -> Iterable[dict[str, Any]]:
    header = ""
    seq: list[str] = []
    count = 0
    with _open_text(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if seq:
                    yield fasta_row(header, "".join(seq))
                    count += 1
                    if limit is not None and count >= limit:
                        return
                header = line[1:]
                seq = []
            else:
                seq.append(line)
    if seq and (limit is None or count < limit):
        yield fasta_row(header, "".join(seq))


def fasta_row(header: str, sequence: str) -> dict[str, Any]:
    ec = ""
    for part in header.replace("|", " ").split():
        if part.count(".") == 3 and all(piece.isdigit() or piece == "-" for piece in part.split(".")):
            ec = part
            break
    return {"protein_sequence": sequence, "sequence": sequence, "ec_number": ec, "description": header}


def normalize_row(row: dict[str, Any], kind: str) -> dict[str, Any]:
    out = dict(row)
    if kind == "pubchem":
        out.setdefault("smiles", row.get("CanonicalSMILES") or row.get("canonical_smiles") or row.get("SMILES"))
        out.setdefault("prompt", row.get("Title") or row.get("name") or row.get("description") or "")
    elif kind in {"uniprot", "uniprot_features", "refseq", "ncbi", "ec"}:
        out.setdefault("protein_sequence", row.get("Sequence") or row.get("sequence") or row.get("protein_sequence"))
        out.setdefault("accession", row.get("Entry") or row.get("accession") or row.get("primaryAccession") or row.get("uniprot_id"))
        out.setdefault("ec_number", row.get("EC number") or row.get("ec_number") or row.get("EC"))
        out.setdefault("protein_name", row.get("Protein names") or row.get("protein_name") or row.get("recommended_name"))
        out.setdefault("gene_names", row.get("Gene Names") or row.get("gene_names") or row.get("gene"))
        out.setdefault("organism", row.get("Organism") or row.get("organism"))
        out.setdefault("taxonomy_id", row.get("Organism ID") or row.get("Organism (ID)") or row.get("taxonomy_id") or row.get("taxid"))
        out.setdefault("go_terms", row.get("Gene Ontology IDs") or row.get("Gene Ontology (GO)") or row.get("go_terms") or row.get("go"))
        out.setdefault("keywords", row.get("Keywords") or row.get("keywords"))
        out.setdefault("features", row.get("Features") or row.get("features") or row.get("uniprot_features"))
        out.setdefault("binding_sites", row.get("Binding site") or row.get("binding_sites") or row.get("binding_site"))
        out.setdefault("subcellular_location", row.get("Subcellular location [CC]") or row.get("subcellular_location"))
        out.setdefault("cofactor", row.get("Cofactor") or row.get("cofactor"))
        out.setdefault("catalytic_activity", row.get("Catalytic activity") or row.get("catalytic_activity"))
        out.setdefault("subunit", row.get("Subunit structure") or row.get("subunit"))
        function_text = (
            row.get("function_description")
            or row.get("function")
            or row.get("description")
            or row.get("protein_description")
            or row.get("annotation")
            or row.get("text")
        )
        if function_text:
            out.setdefault("function_description", function_text)
            out.setdefault("task", "function_description")
            out.setdefault("prompt", "Generate sequence-grounded protein function records.")
    elif kind in {"sfm", "naturelm", "protrek", "protein_function"}:
        out.setdefault("protein_sequence", row.get("Sequence") or row.get("sequence") or row.get("protein_sequence") or row.get("aa_sequence"))
        function_text = (
            row.get("function_description")
            or row.get("function")
            or row.get("text")
            or row.get("description")
            or row.get("protein_description")
            or row.get("annotation")
            or row.get("summary")
            or row.get("completion")
            or row.get("output")
        )
        if function_text:
            out.setdefault("function_description", function_text)
        out.setdefault("task", row.get("task") or "function_description")
        out.setdefault("prompt", row.get("prompt") or "Generate sequence-grounded protein function records.")
    elif kind in {"omg", "omg_mixed", "open_metagenome"}:
        out.setdefault("task", row.get("task") or "omg_mixed_metagenomic_context")
        out.setdefault("prompt", row.get("prompt") or "Generate mixed CDS/IGS metagenomic context graph records.")
    elif kind in {"materials", "materials_project"}:
        out.setdefault("formula", row.get("formula_pretty") or row.get("formula") or row.get("material_formula"))
        out.setdefault("mpid", row.get("material_id") or row.get("mpid") or row.get("Materials Project ID"))
        out.setdefault("completion", row.get("crystal_system") or row.get("completion") or row.get("label") or "")
    elif kind in {"chembl", "bindingdb", "bioactivity"}:
        out.setdefault("smiles", row.get("canonical_smiles") or row.get("Ligand SMILES") or row.get("SMILES"))
        out.setdefault("protein_sequence", row.get("Target Sequence") or row.get("target_sequence") or row.get("sequence"))
        out.setdefault("standard_value", row.get("Standard Value") or row.get("Ki") or row.get("Kd") or row.get("IC50"))
        out.setdefault("standard_units", row.get("Standard Units") or row.get("units"))
        out.setdefault("standard_type", row.get("Standard Type") or row.get("type"))
    elif kind in {"pdbbind", "docking"}:
        out.setdefault("ligand_smiles", row.get("ligand_smiles") or row.get("smiles") or row.get("SMILES"))
        out.setdefault("protein_sequence", row.get("protein_sequence") or row.get("sequence"))
        out.setdefault("affinity", row.get("affinity") or row.get("binding_affinity"))
    elif kind in {"complex_affinity", "biomolecular_affinity", "ppi_affinity", "protein_na_affinity"}:
        out.setdefault("components", row.get("components") or row.get("complex_components"))
        out.setdefault("protein_sequence_a", row.get("protein_sequence_a") or row.get("protein_a_sequence") or row.get("sequence_a") or row.get("protein1_sequence"))
        out.setdefault("protein_sequence_b", row.get("protein_sequence_b") or row.get("protein_b_sequence") or row.get("sequence_b") or row.get("protein2_sequence"))
        out.setdefault("rna_sequence", row.get("rna_sequence") or row.get("rna") or row.get("aptamer_sequence"))
        out.setdefault("dna_sequence", row.get("dna_sequence") or row.get("dna"))
        out.setdefault("ligand_smiles", row.get("ligand_smiles") or row.get("smiles") or row.get("SMILES"))
        out.setdefault("affinity", row.get("affinity") or row.get("binding_affinity") or row.get("Kd") or row.get("Ki") or row.get("IC50"))
        out.setdefault("affinity_type", row.get("affinity_type") or row.get("measure") or row.get("standard_type"))
        out.setdefault("affinity_units", row.get("affinity_units") or row.get("units") or row.get("standard_units"))
        out.setdefault("interaction_type", row.get("interaction_type") or row.get("complex_type") or row.get("assay_type"))
    if kind in {
        "pubchem",
        "uniprot",
        "uniprot_features",
        "refseq",
        "ncbi",
        "ec",
        "sfm",
        "naturelm",
        "protrek",
        "protein_function",
        "chembl",
        "bindingdb",
        "bioactivity",
        "pdbbind",
        "docking",
        "complex_affinity",
        "biomolecular_affinity",
        "ppi_affinity",
        "protein_na_affinity",
    }:
        for key, value in modality_bioselfies_fields(out).items():
            out.setdefault(key, value)
        out.setdefault("force_bioselfies_inputs", True)
    return out


def iter_rows(paths: list[Path], kind: str, limit: int | None) -> Iterable[dict[str, Any]]:
    emitted = 0
    for path in paths:
        if _has_suffix(path, {".fa", ".fasta", ".faa", ".fna"}):
            iterator = iter_fasta(path, None if limit is None else limit - emitted)
        elif _looks_like_pubchem_cid_smiles(path) or _has_suffix(path, {".csv", ".tsv", ".tab"}):
            iterator = iter_csv(path, None if limit is None else limit - emitted)
        else:
            iterator = iter_json_or_jsonl(path, None if limit is None else limit - emitted)
        for row in iterator:
            yield normalize_row(row, kind)
            emitted += 1
            if limit is not None and emitted >= limit:
                return


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare local NatureLM/UniGenX-style science sources into graph JSONL.")
    parser.add_argument("--input", action="append", required=True, help="CSV/TSV/JSON/JSONL/FASTA source file.")
    parser.add_argument(
        "--kind",
        required=True,
        choices=[
            "pubchem",
            "uniprot",
            "uniprot_features",
            "refseq",
            "ncbi",
            "sfm",
            "naturelm",
            "protrek",
            "protein_function",
            "omg",
            "omg_mixed",
            "open_metagenome",
            "materials",
            "materials_project",
            "chembl",
            "bindingdb",
            "bioactivity",
            "pdbbind",
            "docking",
            "complex_affinity",
            "biomolecular_affinity",
            "ppi_affinity",
            "protein_na_affinity",
            "ec",
        ],
    )
    parser.add_argument("--dataset-name", help="Override dataset name used for graphification.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--expected-rows", type=int, help="Optional progress-bar total when streaming large compressed files.")
    args = parser.parse_args()
    paths = [Path(p) for p in args.input]
    dataset_name = args.dataset_name or f"local_{args.kind}"
    ensure_dir(Path(args.output).parent)
    rows = iter_rows(paths, args.kind, args.limit)
    total = args.expected_rows if args.expected_rows is not None else args.limit
    graphs = tqdm(graphify_rows(rows, dataset_name), total=total, desc=f"science/{args.kind}", unit="ex")
    count = write_jsonl(args.output, graphs)
    print(json.dumps({"rows": count, "graphs": count, "output": args.output}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
