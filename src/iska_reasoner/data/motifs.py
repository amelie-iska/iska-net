from __future__ import annotations

import csv
import gzip
import json
import math
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable
from urllib.request import Request, urlopen

from tqdm.auto import tqdm


DEFAULT_SEQUENCE_MOTIFS = [
    "coiled_coil",
    "signal_peptide",
    "transmembrane_helix",
    "low_complexity",
    "zinc_finger",
    "leucine_zipper",
    "kinase_domain",
    "rossmann_fold_binding_loop",
    "p_loop_nucleotide_binding",
    "helix_turn_helix",
    "ef_hand",
    "sh3_domain",
    "pdz_domain",
    "rna_recognition_motif",
    "homeobox",
]

DEFAULT_STRUCTURE_MOTIFS = [
    "helix",
    "sheet",
    "turn",
    "loop",
    "domain",
    "active_site",
    "binding_pocket",
    "pharmacophore",
    "aromatic_ring",
    "hydrogen_bond_donor",
    "hydrogen_bond_acceptor",
    "rna_hairpin",
    "rna_internal_loop",
    "rna_pseudoknot",
    "dna_helix",
    "cath_domain",
    "cath_superfamily",
    "3di_fragment",
    "contact_patch",
]

DEFAULT_STRUCTURE_DERIVED_SEQUENCE_MOTIFS = [
    "helix_window",
    "strand_window",
    "loop_window",
    "contact_patch_sequence",
    "binding_pocket_sequence",
    "domain_boundary_sequence",
    "3di_like_sequence_fragment",
]


@dataclass(slots=True, frozen=True)
class MotifRecord:
    kind: str
    source: str
    accession: str
    name: str = ""
    description: str = ""
    parent: str = ""

    def token(self) -> str:
        prefix = {
            "sequence": "SEQ_MOTIF",
            "structure": "STRUCT_MOTIF",
            "structure_derived_sequence": "STRUCT_DERIVED_SEQ_MOTIF",
        }.get(self.kind, "MOTIF")
        source = normalize_fragment(self.source)
        accession = normalize_fragment(self.accession or self.name or "unknown")
        return f"{prefix}:{source}:{accession}"

    def sequence_safe_token(self) -> str | None:
        """Return an early-phase-safe token for structure-derived sequence motifs.

        These tokens describe sequence motifs imported from a frozen motif
        vocabulary whose origin may be structural, for example CATH or 3Di-like
        annotations. They do not imply that the current training row contains
        coordinates, atom records, contact maps, or structure files.
        """

        if self.kind != "structure_derived_sequence":
            return None
        source = normalize_fragment(self.source)
        accession = normalize_fragment(self.accession or self.name or "unknown")
        return f"SEQ_MOTIF_FROM_STRUCTURE:{source}:{accession}"

    def name_token(self) -> str | None:
        if not self.name:
            return None
        prefix = {
            "sequence": "SEQ_MOTIF_NAME",
            "structure": "STRUCT_MOTIF_NAME",
            "structure_derived_sequence": "STRUCT_DERIVED_SEQ_MOTIF_NAME",
        }.get(self.kind, "MOTIF_NAME")
        return f"{prefix}:{normalize_fragment(self.name)}"


def normalize_fragment(value: Any, max_len: int = 96) -> str:
    text = str(value or "").strip()
    text = text.replace("'", "").replace('"', "")
    text = re.sub(r"[^A-Za-z0-9_.:+-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return (text or "unknown")[:max_len]


def default_motif_records() -> list[MotifRecord]:
    records: list[MotifRecord] = []
    records.extend(MotifRecord("sequence", "core", name, name) for name in DEFAULT_SEQUENCE_MOTIFS)
    records.extend(MotifRecord("structure", "core", name, name) for name in DEFAULT_STRUCTURE_MOTIFS)
    records.extend(
        MotifRecord("structure_derived_sequence", "core", name, name)
        for name in DEFAULT_STRUCTURE_DERIVED_SEQUENCE_MOTIFS
    )
    return records


def motif_records_to_tokens(records: Iterable[MotifRecord], include_names: bool = True) -> list[str]:
    tokens: list[str] = []
    for record in records:
        tokens.append(record.token())
        safe_token = record.sequence_safe_token()
        if safe_token:
            tokens.append(safe_token)
        if include_names:
            name_token = record.name_token()
            if name_token:
                tokens.append(name_token)
    return sorted(dict.fromkeys(tokens))


def _read_text(path: Path) -> str:
    data = path.read_bytes()
    if path.suffix == ".gz":
        data = gzip.decompress(data)
    return data.decode("utf-8", errors="ignore")


def parse_prosite_dat(path: str | Path) -> list[MotifRecord]:
    records: list[MotifRecord] = []
    current: dict[str, str] = {}
    for raw in _read_text(Path(path)).splitlines():
        code = raw[:2]
        value = raw[5:].strip() if len(raw) > 5 else ""
        if code == "ID":
            name = value.split(";", 1)[0].strip()
            entry_type = value.split(";", 1)[1].strip(" .") if ";" in value else ""
            current["name"] = name
            current["entry_type"] = entry_type
        elif code == "AC":
            current["accession"] = value.strip(";.")
        elif code == "DE":
            current["description"] = value
        elif raw.startswith("//"):
            if current.get("accession") or current.get("name"):
                records.append(
                    MotifRecord(
                        kind="sequence",
                        source="prosite",
                        accession=current.get("accession", current.get("name", "")),
                        name=current.get("name", ""),
                        description=current.get("description", current.get("entry_type", "")),
                    )
                )
            current = {}
    return records


def parse_interpro_json(path: str | Path) -> list[MotifRecord]:
    payload = json.loads(_read_text(Path(path)))
    rows = payload.get("results", payload if isinstance(payload, list) else [])
    records: list[MotifRecord] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else row
        accession = str(metadata.get("accession") or row.get("accession") or "").strip()
        if not accession:
            continue
        entry_type = str(metadata.get("type") or metadata.get("entry_type") or "").lower()
        kind = "structure" if any(term in entry_type for term in ["homologous_superfamily", "domain"]) and "cath" in str(metadata).lower() else "sequence"
        records.append(
            MotifRecord(
                kind=kind,
                source="interpro",
                accession=accession,
                name=str(metadata.get("name") or ""),
                description=str(metadata.get("description") or metadata.get("type") or ""),
            )
        )
    return records


def parse_interpro_jsonl(path: str | Path) -> list[MotifRecord]:
    records: list[MotifRecord] = []
    for line in _read_text(Path(path)).splitlines():
        if line.strip():
            records.extend(parse_interpro_json_payload(json.loads(line)))
    return records


def parse_interpro_json_payload(payload: dict[str, Any] | list[Any]) -> list[MotifRecord]:
    rows = payload.get("results", payload if isinstance(payload, list) else [])
    records: list[MotifRecord] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else row
        accession = str(metadata.get("accession") or row.get("accession") or "").strip()
        if accession:
            records.append(MotifRecord("sequence", "interpro", accession, str(metadata.get("name") or ""), str(metadata.get("type") or "")))
    return records


def parse_cath_names(path: str | Path) -> list[MotifRecord]:
    records: list[MotifRecord] = []
    for raw in _read_text(Path(path)).splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        accession = parts[0]
        desc = parts[1] if len(parts) > 1 else ""
        if re.match(r"^\d+(?:\.\d+)*$", accession):
            records.append(MotifRecord("structure", "cath", accession, desc[:80], desc))
            records.append(MotifRecord("structure_derived_sequence", "cath", accession, desc[:80], desc, parent=accession))
    return records


def parse_rfam_family(path: str | Path) -> list[MotifRecord]:
    records: list[MotifRecord] = []
    for raw in _read_text(Path(path)).splitlines():
        if not raw.strip() or raw.startswith("#"):
            continue
        parts = raw.split("\t")
        accession = parts[0].strip() if parts else ""
        if not accession.startswith("RF"):
            continue
        identifier = parts[3].strip() if len(parts) > 3 else accession
        description = parts[18].strip() if len(parts) > 18 else identifier
        records.append(MotifRecord("sequence", "rfam", accession, identifier, description))
    return records


def parse_motif_rows(path: str | Path) -> list[MotifRecord]:
    path = Path(path)
    text = _read_text(path)
    records: list[MotifRecord] = []
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    elif path.suffix == ".json":
        payload = json.loads(text)
        rows = payload if isinstance(payload, list) else payload.get("rows", [])
    elif path.suffix.lower() in {".csv", ".tsv", ".tab"}:
        rows = list(csv.DictReader(text.splitlines(), delimiter="\t" if path.suffix.lower() in {".tsv", ".tab"} else ","))
    else:
        rows = [{"motif": line.strip()} for line in text.splitlines() if line.strip() and not line.startswith("#")]
    for row in rows:
        if not isinstance(row, dict):
            continue
        for key, kind in [
            ("sequence_motifs", "sequence"),
            ("sequence_motif", "sequence"),
            ("prosite", "sequence"),
            ("interpro", "sequence"),
            ("rfam", "sequence"),
            ("sequence_motifs_from_structure", "structure_derived_sequence"),
            ("sequence_motif_from_structure", "structure_derived_sequence"),
            ("structure_vocab_sequence_motifs", "structure_derived_sequence"),
            ("structure_vocab_sequence_motif", "structure_derived_sequence"),
            ("structure_motifs", "structure"),
            ("structure_motif", "structure"),
            ("cath", "structure"),
            ("structure_derived_sequence_motifs", "structure_derived_sequence"),
            ("structure_derived_sequence_motif", "structure_derived_sequence"),
        ]:
            values = row.get(key)
            if values is None:
                continue
            if isinstance(values, str):
                values = re.split(r"[,;]\s*", values) if "," in values or ";" in values else [values]
            for value in values if isinstance(values, list) else [values]:
                if isinstance(value, dict):
                    accession = value.get("accession") or value.get("id") or value.get("name") or ""
                    name = value.get("name") or value.get("description") or accession
                else:
                    accession = value
                    name = value
                records.append(MotifRecord(kind, "local", str(accession), str(name)))
        if row.get("motif"):
            records.append(MotifRecord("sequence", "local", str(row["motif"]), str(row.get("name") or row["motif"])))
    return records


def _download(url: str, output_path: Path, timeout: int = 60) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url, headers={"User-Agent": "iska-net-motif-builder/1.0"})
    with urlopen(req, timeout=timeout) as response:
        output_path.write_bytes(response.read())
    return output_path


PROSITE_URL = "https://ftp.expasy.org/databases/prosite/prosite.dat"
INTERPRO_URL = "https://www.ebi.ac.uk/interpro/api/entry/interpro/?page_size=200"
CATH_NAMES_URL = "https://download.cathdb.info/cath/releases/latest-release/cath-classification-data/cath-names.txt"
RFAM_FAMILY_URL = "https://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/database_files/family.txt.gz"


def download_public_motif_sources(output_dir: str | Path, include_interpro: bool = True) -> dict[str, Path]:
    output_dir = Path(output_dir)
    paths = {
        "prosite": _download(PROSITE_URL, output_dir / "prosite.dat"),
        "cath": _download(CATH_NAMES_URL, output_dir / "cath-names.txt"),
        "rfam": _download(RFAM_FAMILY_URL, output_dir / "rfam-family.txt.gz"),
    }
    if include_interpro:
        rows: list[dict[str, Any]] = []
        next_url: str | None = INTERPRO_URL
        pbar = tqdm(desc="motifs/interpro_pages", unit="page")
        total_count = None
        while next_url:
            req = Request(next_url, headers={"User-Agent": "iska-net-motif-builder/1.0"})
            with urlopen(req, timeout=60) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if total_count is None and payload.get("count"):
                total_count = int(payload["count"])
                pbar.total = math.ceil(total_count / 200)
                pbar.refresh()
            rows.extend(payload.get("results", []))
            next_url = payload.get("next")
            pbar.update(1)
            pbar.set_postfix(entries=len(rows))
        pbar.close()
        interpro_path = output_dir / "interpro_entries.json"
        interpro_path.write_text(json.dumps({"results": rows}, sort_keys=True), encoding="utf-8")
        paths["interpro"] = interpro_path
    return paths


def parse_motif_source(path: str | Path) -> list[MotifRecord]:
    path = Path(path)
    name = path.name.lower()
    if "prosite" in name and name.endswith(".dat"):
        return parse_prosite_dat(path)
    if "interpro" in name and path.suffix in {".json", ".gz"}:
        return parse_interpro_json(path)
    if "interpro" in name and path.suffix == ".jsonl":
        return parse_interpro_jsonl(path)
    if "cath" in name and "name" in name:
        return parse_cath_names(path)
    if "rfam" in name or "family" in name:
        return parse_rfam_family(path)
    return parse_motif_rows(path)


def derive_structure_sequence_motifs_from_atoms(row: dict[str, Any], source: str = "local_structure") -> list[MotifRecord]:
    atoms = row.get("atoms") or []
    frames = row.get("frames") or row.get("trajectory") or []
    if not isinstance(atoms, list) or not atoms:
        return []
    ca_atoms: list[tuple[int, dict[str, Any]]] = []
    for idx, atom in enumerate(atoms):
        if not isinstance(atom, dict):
            continue
        name = str(atom.get("name") or atom.get("atom") or "").strip().upper()
        if name in {"CA", "C4'", "C4P", "P"}:
            ca_atoms.append((idx, atom))
    if not ca_atoms:
        ca_atoms = [(idx, atom) for idx, atom in enumerate(atoms[:64]) if isinstance(atom, dict)]
    coords: list[list[float]] = []
    if isinstance(frames, list) and frames:
        frame0 = frames[0].get("coordinates") if isinstance(frames[0], dict) else frames[0]
        if isinstance(frame0, list):
            coords = frame0
    records: list[MotifRecord] = []
    for center in range(len(ca_atoms)):
        left = max(0, center - 2)
        right = min(len(ca_atoms), center + 3)
        window = ca_atoms[left:right]
        residues = [str(atom.get("residue") or atom.get("resname") or atom.get("element") or "X")[:3] for _, atom in window]
        seq_code = "-".join(residues)
        contact_degree = 0
        idx_center = ca_atoms[center][0]
        if coords and idx_center < len(coords):
            c0 = coords[idx_center]
            for idx_other, _ in ca_atoms:
                if idx_other == idx_center or idx_other >= len(coords):
                    continue
                c1 = coords[idx_other]
                try:
                    dist = math.sqrt(sum((float(c0[k]) - float(c1[k])) ** 2 for k in range(3)))
                except Exception:
                    continue
                if dist <= 8.0:
                    contact_degree += 1
        degree_bin = "contact_low" if contact_degree <= 2 else "contact_mid" if contact_degree <= 6 else "contact_high"
        accession = f"{seq_code}:{degree_bin}"
        records.append(MotifRecord("structure", source, accession, name=accession))
        records.append(MotifRecord("structure_derived_sequence", source, accession, name=accession, parent=accession))
    return records


def build_motif_vocabulary(
    source_paths: Iterable[str | Path] = (),
    include_defaults: bool = True,
    include_names: bool = True,
) -> tuple[list[str], list[MotifRecord]]:
    records: list[MotifRecord] = default_motif_records() if include_defaults else []
    for path in source_paths:
        records.extend(parse_motif_source(path))
    tokens = motif_records_to_tokens(records, include_names=include_names)
    return tokens, records


def write_motif_vocabulary(
    output_path: str | Path,
    source_paths: Iterable[str | Path] = (),
    summary_path: str | Path | None = None,
    include_defaults: bool = True,
    include_names: bool = True,
) -> dict[str, Any]:
    tokens, records = build_motif_vocabulary(source_paths, include_defaults=include_defaults, include_names=include_names)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(tokens) + "\n", encoding="utf-8")
    summary = {
        "output": str(output),
        "tokens": len(tokens),
        "records": len(records),
        "by_kind": {},
        "by_source": {},
    }
    for record in records:
        summary["by_kind"][record.kind] = summary["by_kind"].get(record.kind, 0) + 1
        summary["by_source"][record.source] = summary["by_source"].get(record.source, 0) + 1
    if summary_path:
        path = Path(summary_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary
