from __future__ import annotations

import re
import os
import resource
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from iska_reasoner.graph.schema import GraphExample
from iska_reasoner.oracles import score_uma_oracle_candidate


@dataclass(slots=True)
class VerificationResult:
    passed: bool
    reward: float
    exact_token_match: bool
    token_recall: float
    extra_token_rate: float
    numeric_match: bool
    python_passed: bool
    lean_available: bool
    rdkit_available: bool
    message: str = ""

    def metric_dict(self, prefix: str = "verifier/") -> dict[str, float]:
        data = asdict(self)
        out: dict[str, float] = {}
        for key, value in data.items():
            if isinstance(value, bool):
                out[f"{prefix}{key}"] = 1.0 if value else 0.0
            elif isinstance(value, (int, float)):
                out[f"{prefix}{key}"] = float(value)
        return out


@dataclass(slots=True)
class PythonTestResult:
    attempted: bool
    passed: bool
    test_count: int
    message: str = ""


@dataclass(slots=True)
class LeanResult:
    available: bool
    attempted: bool
    passed: bool
    version: str = ""
    message: str = ""


@dataclass(slots=True)
class MoleculeResult:
    rdkit_available: bool
    valid: bool
    atom_count: int = 0
    bond_count: int = 0
    message: str = ""


def _answer_values(tokens: Iterable[str]) -> list[str]:
    values = []
    for token in tokens:
        if token.startswith("ANSWER:"):
            values.append(token.split(":", 1)[1].strip())
    return values


def _numbers(text: str) -> list[str]:
    return re.findall(r"[-+]?\d+(?:\.\d+)?", text)


def numeric_match(target_tokens: list[str], predicted_tokens: list[str]) -> bool:
    target_answers = " ".join(_answer_values(target_tokens))
    pred_answers = " ".join(_answer_values(predicted_tokens))
    target_nums = _numbers(target_answers)
    pred_nums = _numbers(pred_answers)
    return bool(target_nums and pred_nums and target_nums[-1] == pred_nums[-1])


def run_python_snippet(snippet: str, timeout_s: float = 3.0) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory(prefix="iska_verify_") as tmp:
        path = Path(tmp) / "snippet.py"
        path.write_text(snippet, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, str(path)],
                cwd=tmp,
                text=True,
                capture_output=True,
                timeout=timeout_s,
                preexec_fn=_resource_limiter(),
            )
        except subprocess.TimeoutExpired:
            return False, "python timeout"
    return proc.returncode == 0, (proc.stdout + proc.stderr)[-500:]


def run_python_tests(code: str, tests: list[str], timeout_s: float = 5.0) -> PythonTestResult:
    if not code.strip() or not tests:
        return PythonTestResult(attempted=False, passed=False, test_count=len(tests), message="missing code or tests")
    with tempfile.TemporaryDirectory(prefix="iska_code_verify_") as tmp:
        base = Path(tmp)
        (base / "solution.py").write_text(code, encoding="utf-8")
        test_body = "\n\n".join(tests)
        if "solution" not in test_body:
            test_body = "from solution import *\n\n" + test_body
        (base / "test_solution.py").write_text(test_body, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", str(base / "test_solution.py")],
                cwd=tmp,
                text=True,
                capture_output=True,
                timeout=timeout_s,
                preexec_fn=_resource_limiter(),
            )
        except subprocess.TimeoutExpired:
            return PythonTestResult(attempted=True, passed=False, test_count=len(tests), message="python tests timeout")
    return PythonTestResult(
        attempted=True,
        passed=proc.returncode == 0,
        test_count=len(tests),
        message=(proc.stdout + proc.stderr)[-800:],
    )


def _resource_limiter(memory_mb: int = 512, cpu_seconds: int = 8):
    if os.name != "posix":
        return None

    def limit() -> None:
        mem = int(memory_mb * 1024 * 1024)
        try:
            resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
        except Exception:
            pass
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        except Exception:
            pass
        try:
            resource.setrlimit(resource.RLIMIT_FSIZE, (10 * 1024 * 1024, 10 * 1024 * 1024))
        except Exception:
            pass

    return limit


@lru_cache(maxsize=1)
def lean_version() -> tuple[bool, str]:
    lean = shutil.which("lean")
    if not lean:
        return False, ""
    try:
        proc = subprocess.run([lean, "--version"], text=True, capture_output=True, timeout=3.0)
    except Exception:
        return True, "version probe failed"
    return True, (proc.stdout or proc.stderr).strip()


def compile_lean_source(source: str, timeout_s: float = 8.0) -> LeanResult:
    available, version = lean_version()
    if not source.strip():
        return LeanResult(available=available, attempted=False, passed=False, version=version, message="missing source")
    if not available:
        return LeanResult(available=False, attempted=False, passed=False, version=version, message="lean unavailable")
    with tempfile.TemporaryDirectory(prefix="iska_lean_verify_") as tmp:
        path = Path(tmp) / "Check.lean"
        path.write_text(source, encoding="utf-8")
        try:
            proc = subprocess.run(["lean", str(path)], cwd=tmp, text=True, capture_output=True, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return LeanResult(available=True, attempted=True, passed=False, version=version, message="lean timeout")
    return LeanResult(
        available=True,
        attempted=True,
        passed=proc.returncode == 0,
        version=version,
        message=(proc.stdout + proc.stderr)[-800:],
    )


def rdkit_molecule_stats(smiles: str) -> MoleculeResult:
    try:
        from rdkit import Chem
        from rdkit import RDLogger

        RDLogger.DisableLog("rdApp.*")
    except Exception:
        return MoleculeResult(rdkit_available=False, valid=False, message="rdkit unavailable")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return MoleculeResult(rdkit_available=True, valid=False, message="invalid smiles")
    return MoleculeResult(
        rdkit_available=True,
        valid=True,
        atom_count=int(mol.GetNumAtoms()),
        bond_count=int(mol.GetNumBonds()),
        message="ok",
    )


def rdkit_smiles_valid(smiles: str) -> tuple[bool, bool]:
    result = rdkit_molecule_stats(smiles)
    return result.valid, result.rdkit_available


def _node_values(example: GraphExample, *types: str) -> list[str]:
    type_set = set(types)
    return [node.value for node in example.nodes if node.type in type_set and node.value]


def _metadata_tests(metadata: dict[str, Any]) -> list[str]:
    raw = metadata.get("tests") or metadata.get("test") or metadata.get("unit_tests") or []
    if isinstance(raw, str):
        return [raw] if raw.strip() else []
    if isinstance(raw, list):
        return [str(item) for item in raw if str(item).strip()]
    return []


def _predicted_python(predicted_tokens: list[str]) -> str:
    chunks = []
    for token in predicted_tokens:
        if token.startswith("CODE:python:"):
            chunks.append(token.split(":", 2)[2])
        elif token.startswith("CODE:raw:"):
            chunks.append(token.split(":", 2)[2])
    return "\n".join(chunks)


def code_metrics_for_example(example: GraphExample, predicted_tokens: list[str]) -> dict[str, float]:
    tests = _metadata_tests(example.metadata) + _node_values(example, "test", "unit_test")
    code = _predicted_python(predicted_tokens) or str(example.metadata.get("canonical_solution") or "")
    result = run_python_tests(code, tests) if tests and code else PythonTestResult(False, False, len(tests), "not attempted")
    return {
        "code/has_tests_rate": 1.0 if tests else 0.0,
        "code/test_count_mean": float(len(tests)),
        "code/pass_rate": 1.0 if result.attempted and result.passed else 0.0,
        "code/python_error_rate": 1.0 if result.attempted and not result.passed else 0.0,
        "code/attempt_rate": 1.0 if result.attempted else 0.0,
    }


def lean_source_for_example(example: GraphExample) -> str:
    source = str(example.metadata.get("lean_source") or "").strip()
    if source:
        return source
    formal = "\n".join(_node_values(example, "lean_statement"))
    proof = "\n".join(_node_values(example, "lean_proof"))
    return "\n".join(part for part in [formal, proof] if part.strip()).strip()


def lean_metrics_for_example(example: GraphExample) -> dict[str, float]:
    source = lean_source_for_example(example)
    result = compile_lean_source(source) if source else LeanResult(available=lean_version()[0], attempted=False, passed=False, message="no source")
    return {
        "lean/available": 1.0 if result.available else 0.0,
        "lean/source_present_rate": 1.0 if source else 0.0,
        "lean/compile_attempt_rate": 1.0 if result.attempted else 0.0,
        "lean/compile_success_rate": 1.0 if result.attempted and result.passed else 0.0,
        "lean/error_rate": 1.0 if result.attempted and not result.passed else 0.0,
    }


def smiles_for_example(example: GraphExample, predicted_tokens: list[str]) -> str:
    for token in predicted_tokens:
        if token.startswith("SMILES:"):
            value = token.split(":", 1)[1].strip()
            if value and value != "valid_candidate":
                return value
    for key in ("smiles", "SMILES", "Smiles"):
        if example.metadata.get(key):
            return str(example.metadata[key])
    values = _node_values(example, "smiles")
    return values[0] if values else ""


def chem_metrics_for_example(example: GraphExample, predicted_tokens: list[str]) -> dict[str, float]:
    smiles = smiles_for_example(example, predicted_tokens)
    result = rdkit_molecule_stats(smiles) if smiles else MoleculeResult(False, False, message="missing smiles")
    atom_count = result.atom_count
    bond_count = result.bond_count
    if not result.rdkit_available:
        atom_count = int(example.metadata.get("atom_count") or atom_count)
        bond_count = int(example.metadata.get("bond_count") or bond_count)
    filters = medicinal_safety_filters(smiles)
    return {
        "chem/rdkit_available": 1.0 if result.rdkit_available else 0.0,
        "chem/smiles_present_rate": 1.0 if smiles else 0.0,
        "chem/smiles_valid_rate": 1.0 if result.valid else 0.0,
        "chem/atom_count_mean": float(atom_count),
        "chem/bond_count_mean": float(bond_count),
        "chem/medicinal_filter_pass_rate": 1.0 if filters["passes"] else 0.0,
        "chem/reactive_alert_count_mean": float(filters["alert_count"]),
    }


def medicinal_safety_filters(smiles: str) -> dict[str, Any]:
    """Very small medicinal-chemistry triage filters.

    These are not clinical or toxicology claims. They are deterministic alerts
    for obvious reactive fragments so training/validation can track whether
    molecule proposals need downstream review.
    """

    if not smiles:
        return {"passes": False, "alert_count": 0, "alerts": ["missing_smiles"]}
    patterns = {
        "acid_chloride": "C(=O)Cl",
        "isocyanate": "N=C=O",
        "azide": "N=[N+]=[N-]",
        "nitroso": "N=O",
        "alkyl_halide": "CCl",
    }
    alerts = [name for name, pattern in patterns.items() if pattern in smiles]
    return {"passes": len(alerts) == 0, "alert_count": len(alerts), "alerts": alerts}


def local_audio_metrics_for_example(example: GraphExample) -> dict[str, float]:
    taxonomy_nodes = [node for node in example.nodes if node.type.startswith("taxonomy_")]
    audio_feature_nodes = [node for node in example.nodes if node.type == "audio_features"]
    audio_feature_available = any(bool(node.features.get("available")) for node in audio_feature_nodes)
    license_value = str(example.metadata.get("license") or "").lower()
    return {
        "audio/task_present_rate": 1.0 if example.metadata.get("task") else 0.0,
        "audio/taxonomy_node_count_mean": float(len(taxonomy_nodes)),
        "audio/noncommercial_license_rate": 1.0 if "nc" in license_value or "noncommercial" in license_value else 0.0,
        "audio/audio_feature_node_count_mean": float(len(audio_feature_nodes)),
        "audio/audio_feature_available_rate": 1.0 if audio_feature_available else 0.0,
    }


def science_metrics_for_example(example: GraphExample) -> dict[str, float]:
    coordinate_nodes = [node for node in example.nodes if node.type == "coordinate_3d"]
    protein_coordinate_nodes = [node for node in example.nodes if node.type in {"protein_coordinate", "ligand_coordinate"}]
    property_nodes = [node for node in example.nodes if node.type == "molecule_property"]
    formula_nodes = [node for node in example.nodes if node.type == "material_formula"]
    smiles_present = any(node.type == "smiles" and node.value for node in example.nodes)
    protein_nodes = [node for node in example.nodes if node.type == "protein_sequence"]
    ec_nodes = [node for node in example.nodes if node.type == "ec_number"]
    pocket_atoms = [node for node in example.nodes if node.type == "pocket_atom"]
    return {
        "science/coordinate_node_count_mean": float(len(coordinate_nodes)),
        "science/protein_ligand_coordinate_node_count_mean": float(len(protein_coordinate_nodes)),
        "science/property_node_count_mean": float(len(property_nodes)),
        "science/material_formula_count_mean": float(len(formula_nodes)),
        "science/molecule_smiles_present_rate": 1.0 if smiles_present else 0.0,
        "science/protein_sequence_present_rate": 1.0 if protein_nodes else 0.0,
        "science/ec_number_present_rate": 1.0 if ec_nodes else 0.0,
        "science/pocket_atom_count_mean": float(len(pocket_atoms)),
    }


def _example_temperature_kelvin(example: GraphExample) -> float | None:
    for node in example.nodes:
        if node.type != "temperature":
            continue
        value = node.features.get("kelvin") or node.features.get("kelvin_clamped")
        try:
            return float(value)
        except Exception:
            text = str(node.value).rstrip("Kk")
            try:
                return float(text)
            except Exception:
                return None
    metadata_value = (example.metadata or {}).get("temperature")
    try:
        return float(metadata_value)
    except Exception:
        return None


def _temperature_norm(example: GraphExample) -> float | None:
    temp_k = _example_temperature_kelvin(example)
    if temp_k is None:
        return None
    return max(0.0, min(1.0, (float(temp_k) - 300.0) / 100.0))


def _has_prefixed_token(tokens: set[str], prefix: str) -> bool:
    return any(token.startswith(prefix) for token in tokens)


def _temperature_diversity_bonus(example: GraphExample, pred_set: set[str]) -> float:
    temp_norm = _temperature_norm(example)
    if temp_norm is None:
        return 0.0
    diversity_hits = sum(
        [
            _has_prefixed_token(pred_set, "TOKEN_MOTION:uma:diversify:"),
            _has_prefixed_token(pred_set, "TOKEN_MOTION:uma:explore:"),
            _has_prefixed_token(pred_set, "TOKEN_MOTION:uma:expand:"),
            _has_prefixed_token(pred_set, "UMA_TRAJ_BIN:diversify:"),
            _has_prefixed_token(pred_set, "UMA_TRAJ_BIN:explore:"),
            _has_prefixed_token(pred_set, "UMA_TRAJ_BIN:expand:"),
            _has_prefixed_token(pred_set, "UMA_INFLUENCE:uma:diversity_pressure:"),
        ]
    ) / 7.0
    stability_hits = sum(
        [
            _has_prefixed_token(pred_set, "TOKEN_MOTION:uma:stabilize:"),
            _has_prefixed_token(pred_set, "TOKEN_MOTION:uma:refine:"),
            _has_prefixed_token(pred_set, "TOKEN_MOTION:uma:contract:"),
            _has_prefixed_token(pred_set, "UMA_TRAJ_BIN:stabilize:"),
            _has_prefixed_token(pred_set, "UMA_TRAJ_BIN:refine:"),
            _has_prefixed_token(pred_set, "UMA_TRAJ_BIN:contract:"),
            _has_prefixed_token(pred_set, "UMA_INFLUENCE:uma:score_sharpness:"),
        ]
    ) / 7.0
    high_temp_bonus = temp_norm * diversity_hits
    low_temp_bonus = (1.0 - temp_norm) * stability_hits
    return 0.08 * max(high_temp_bonus, low_temp_bonus)


def multimodal_metrics_for_example(example: GraphExample) -> dict[str, float]:
    modalities = example.metadata.get("modalities") or []
    if not isinstance(modalities, list):
        modalities = []
    atom_nodes = [node for node in example.nodes if node.type in {"atom", "all_atom_template_atom"}]
    bond_edges = [edge for edge in example.edges if edge.type == "molecular_bond"]
    typed_bonds = [edge for edge in bond_edges if edge.features.get("bond_type")]
    coordinate_nodes = [node for node in example.nodes if node.type == "coordinate_3d"]
    distance_nodes = [node for node in example.nodes if node.type == "distance_record"]
    frame_nodes = [node for node in example.nodes if node.type == "trajectory_frame"]
    energy_nodes = [node for node in example.nodes if node.type == "energy_record"]
    force_nodes = [node for node in example.nodes if node.type == "force_record"]
    temperature_nodes = [node for node in example.nodes if node.type == "temperature"]
    attention_bin_nodes = [node for node in example.nodes if node.type == "attention_coupling_bin"]
    coupling_bin_nodes = [node for node in example.nodes if node.type == "uma_coupling_strength_bin"]
    influence_bin_nodes = [node for node in example.nodes if node.type == "uma_influence_bin"]
    token_motion_nodes = [node for node in example.nodes if node.type == "token_motion_prior"]
    proxy_nodes = [node for node in example.nodes if node.type == "sequence_structure_dynamics_proxy"]
    target_set = set(example.target_tokens)
    return {
        "multimodal/modality_count_mean": float(len(set(str(item) for item in modalities))),
        "multimodal/protein_present_rate": 1.0 if "protein" in modalities else 0.0,
        "multimodal/selfies_present_rate": 1.0 if "selfies" in modalities else 0.0,
        "multimodal/dna_present_rate": 1.0 if "dna" in modalities else 0.0,
        "multimodal/rna_present_rate": 1.0 if "rna" in modalities else 0.0,
        "multimodal/all_atom_present_rate": 1.0 if "all_atom" in modalities else 0.0,
        "multimodal/trajectory_present_rate": 1.0 if "trajectory" in modalities else 0.0,
        "multimodal/atom_count_mean": float(len(atom_nodes)),
        "multimodal/bond_count_mean": float(len(bond_edges)),
        "multimodal/bond_type_coverage_rate": float(len(typed_bonds) / max(1, len(bond_edges))),
        "multimodal/coordinate_node_count_mean": float(len(coordinate_nodes)),
        "multimodal/distance_node_count_mean": float(len(distance_nodes)),
        "multimodal/frame_count_mean": float(len(frame_nodes)),
        "multimodal/energy_record_rate": 1.0 if energy_nodes else 0.0,
        "multimodal/force_record_rate": 1.0 if force_nodes else 0.0,
        "multimodal/temperature_conditioned_rate": 1.0 if temperature_nodes else 0.0,
        "multimodal/attention_bin_count_mean": float(len(attention_bin_nodes)),
        "multimodal/uma_coupling_bin_count_mean": float(len(coupling_bin_nodes)),
        "multimodal/uma_influence_bin_count_mean": float(len(influence_bin_nodes)),
        "multimodal/token_motion_prior_count_mean": float(len(token_motion_nodes)),
        "multimodal/sequence_structure_dynamics_proxy_rate": 1.0 if proxy_nodes else 0.0,
        "multimodal/pdb_target_rate": 1.0 if any(tok.startswith("PDB:") for tok in target_set) else 0.0,
        "multimodal/oracle_feedback_target_rate": 1.0 if "UGM:oracle:uma_feedback" in target_set else 0.0,
    }


def multimodal_oracle_reward(example: GraphExample, predicted_tokens: list[str]) -> float:
    """Return the UMA oracle reward for oracle-feedback graph completion.

    Production oracle stages use the FairChem UMA backend by default. A
    deterministic token-completeness proxy is still available only when
    explicitly requested with ``UGM_UMA_BACKEND=proxy`` for unit tests and
    smoke runs that must not download gated UMA weights.
    """

    if "multimodal" not in example.task.lower():
        return 0.0
    target_set = set(example.target_tokens)
    pred_set = set(predicted_tokens)
    family_prefixes = [
        "SELFIES:",
        "SMILES:",
        "AA:",
        "DNA:",
        "RNA:",
        "SEQ_MOTIF:",
        "SEQ_MOTIF_FROM_STRUCTURE:",
        "TEMP:",
        "TEMP_ANCHOR:",
        "TEMP_BIN:",
        "ATTN_BIN:",
        "ATTN_COARSE:",
        "TOKEN_COUPLING:uma:",
        "UMA_INFLUENCE:uma:",
        "TOKEN_MOTION:uma:",
        "UMA_TRAJ_BIN:",
        "SEQ_STRUCT_DYN_PROXY:",
        "UGM:oracle:",
    ]
    active_families = [prefix for prefix in family_prefixes if any(tok.startswith(prefix) for tok in target_set)]
    if not active_families:
        return 0.0
    family_scores = []
    for prefix in active_families:
        target_family = {tok for tok in target_set if tok.startswith(prefix)}
        pred_family = {tok for tok in pred_set if tok.startswith(prefix)}
        family_scores.append(len(target_family & pred_family) / max(1, len(target_family)))
    serializer_bonus = 0.10 if "UGM:serializer:selfies" in target_set and "UGM:serializer:selfies" in pred_set else 0.0
    oracle_bonus = 0.15 if "UGM:oracle:uma_feedback" in target_set and "UGM:oracle:uma_feedback" in pred_set else 0.0
    graph_bonus = 0.10 if "UGM:graph_to_graph" in pred_set else 0.0
    temp_nodes = [node for node in example.nodes if node.type == "temperature"]
    temp_bonus = 0.05 if temp_nodes and any(tok.startswith("TEMP:") or tok.startswith("TEMP_BIN:") for tok in pred_set) else 0.0
    coupling_bonus = 0.05 if any(tok.startswith("TOKEN_COUPLING:uma:") for tok in pred_set) else 0.0
    influence_bonus = 0.05 if any(tok.startswith("UMA_INFLUENCE:uma:") for tok in pred_set) else 0.0
    motion_bonus = 0.05 if any(tok.startswith("TOKEN_MOTION:uma:") for tok in pred_set) else 0.0
    trajectory_bonus = 0.05 if any(tok.startswith("UMA_TRAJ_BIN:") for tok in pred_set) else 0.0
    proxy_bonus = 0.05 if "SEQ_STRUCT_DYN_PROXY:uma_scored" in pred_set else 0.0
    temperature_diversity_bonus = _temperature_diversity_bonus(example, pred_set)
    base = min(
        0.75,
        0.45 * (sum(family_scores) / max(1, len(family_scores)))
        + serializer_bonus
        + oracle_bonus
        + graph_bonus
        + temp_bonus
        + coupling_bonus
        + influence_bonus
        + motion_bonus
        + trajectory_bonus
        + proxy_bonus
        + temperature_diversity_bonus,
    )
    temp_k = _example_temperature_kelvin(example)
    if temp_k is not None:
        temp_k = max(300.0, min(400.0, temp_k))
        # High T is more permissive; low T sharpens reward around high-validity
        # candidates. This value is passed as the explicit proxy reward only
        # when the caller opts into UGM_UMA_BACKEND=proxy.
        exponent = 350.0 / temp_k
        base = min(0.75, 0.75 * (max(base, 0.0) / 0.75) ** exponent)
    return score_uma_oracle_candidate(example, predicted_tokens, proxy_reward=base).reward


def biomed_metrics_for_example(example: GraphExample) -> dict[str, float]:
    assay_nodes = [node for node in example.nodes if node.type in {"assay_value", "binding_affinity"}]
    target_nodes = [node for node in example.nodes if node.type in {"target_name", "protein_sequence"}]
    smiles_present = any(node.type == "smiles" and node.value for node in example.nodes)
    numeric_values = []
    for node in assay_nodes:
        numeric_values.extend(_numbers(node.value))
    return {
        "biomed/assay_value_present_rate": 1.0 if assay_nodes else 0.0,
        "biomed/assay_numeric_present_rate": 1.0 if numeric_values else 0.0,
        "biomed/target_node_count_mean": float(len(target_nodes)),
        "biomed/smiles_present_rate": 1.0 if smiles_present else 0.0,
    }


def hebrew_metrics_for_example(example: GraphExample) -> dict[str, float]:
    root_nodes = [node for node in example.nodes if node.type == "hebrew_root" and node.value]
    template_nodes = [node for node in example.nodes if node.type == "hebrew_template" and node.value]
    radical_nodes = [node for node in example.nodes if node.type == "hebrew_radical" and node.value]
    lemma_nodes = [node for node in example.nodes if node.type == "hebrew_lemma" and node.value]
    binyan_nodes = [node for node in example.nodes if node.type == "hebrew_binyan" and node.value]
    derived_nodes = [node for node in example.nodes if node.type == "hebrew_derived_form" and node.value]
    dotted_nodes = [node for node in example.nodes if node.type == "hebrew_diacritized" and node.value]
    qa_like = 1.0 if any(node.type in {"hebrew_answer", "answer"} for node in example.nodes) else 0.0
    return {
        "hebrew/root_node_count_mean": float(len(root_nodes)),
        "hebrew/unique_root_count_mean": float(len({node.value for node in root_nodes})),
        "hebrew/template_node_count_mean": float(len(template_nodes)),
        "hebrew/radical_node_count_mean": float(len(radical_nodes)),
        "hebrew/lemma_node_count_mean": float(len(lemma_nodes)),
        "hebrew/binyan_node_count_mean": float(len(binyan_nodes)),
        "hebrew/derived_form_count_mean": float(len(derived_nodes)),
        "hebrew/diacritized_pair_rate": 1.0 if dotted_nodes else 0.0,
        "hebrew/qa_or_instruction_rate": qa_like,
    }


def domain_metric_dict(example: GraphExample, predicted_tokens: list[str]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    task = example.task.lower()
    if "code" in task:
        metrics.update(code_metrics_for_example(example, predicted_tokens))
    if "proof" in task or "lean" in task:
        metrics.update(lean_metrics_for_example(example))
    if "molecule" in task or "chem" in task:
        metrics.update(chem_metrics_for_example(example, predicted_tokens))
    if "local_audio" in task or task == "audio":
        metrics.update(local_audio_metrics_for_example(example))
    if "unigenx" in task:
        metrics.update(science_metrics_for_example(example))
    if "multimodal" in task:
        metrics.update(multimodal_metrics_for_example(example))
    if "biomed" in task or "bioactivity" in task:
        metrics.update(biomed_metrics_for_example(example))
        metrics.update(chem_metrics_for_example(example, predicted_tokens))
    if "hebrew" in task:
        metrics.update(hebrew_metrics_for_example(example))
    return metrics


def verify_example_tokens(example: GraphExample, predicted_tokens: list[str]) -> VerificationResult:
    target_set = set(example.target_tokens)
    pred_set = set(predicted_tokens)
    recall = len(target_set & pred_set) / max(1, len(target_set))
    extra_rate = len(pred_set - target_set) / max(1, len(pred_set))
    exact = target_set == pred_set
    numeric_ok = numeric_match(example.target_tokens, predicted_tokens)

    python_ok = False
    for token in predicted_tokens:
        if token.startswith("CODE:python:") or token.startswith("CODE:raw:"):
            code = token.split(":", 2)[2]
            python_ok, _ = run_python_snippet(code)
            break

    rdkit_available = False
    rdkit_ok = False
    for token in predicted_tokens + [node.value for node in example.nodes if node.type == "smiles"]:
        if token.startswith("SMILES:"):
            token = token.split(":", 1)[1]
        if any(ch.isalpha() for ch in token) and any(ch in token for ch in "CNOPSFIBrc[]=#()123456789"):
            rdkit_ok, rdkit_available = rdkit_smiles_valid(token)
            break

    lean_available = lean_version()[0]
    passed = exact or numeric_ok or python_ok or rdkit_ok
    oracle_reward = multimodal_oracle_reward(example, predicted_tokens)
    passed = passed or oracle_reward >= 0.45
    reward = 0.05 + 0.65 * recall + 0.2 * float(exact) + 0.1 * float(numeric_ok or python_ok or rdkit_ok) + oracle_reward - 0.15 * extra_rate
    reward = max(1e-4, min(1.5, reward))
    return VerificationResult(
        passed=passed,
        reward=reward,
        exact_token_match=exact,
        token_recall=recall,
        extra_token_rate=extra_rate,
        numeric_match=numeric_ok,
        python_passed=python_ok,
        lean_available=lean_available,
        rdkit_available=rdkit_available,
        message="ok" if passed else "partial_or_failed",
    )
