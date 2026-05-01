from __future__ import annotations

import math
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from iska_reasoner.graph.schema import GraphExample


DEFAULT_FAIRCHEM_REPO = "data/external_repos/fairchem"
DEFAULT_UMA_MODEL = "uma-s-1p2"
DEFAULT_UMA_TASK = "omol"
DEFAULT_UMA_DEVICE = "cuda"


@dataclass(slots=True)
class UmaOracleResult:
    backend: str
    available: bool
    reward: float
    score: float
    temperature_k: float | None = None
    smiles: str = ""
    atom_count: int = 0
    energy_ev: float | None = None
    energy_per_atom_ev: float | None = None
    force_rms_ev_per_a: float | None = None
    forces_ev_per_a: list[list[float]] | None = None
    model_name: str = DEFAULT_UMA_MODEL
    task_name: str = DEFAULT_UMA_TASK
    repo_path: str = DEFAULT_FAIRCHEM_REPO
    message: str = ""

    def metric_dict(self, prefix: str = "uma/") -> dict[str, float]:
        data = asdict(self)
        out: dict[str, float] = {}
        for key, value in data.items():
            if isinstance(value, bool):
                out[f"{prefix}{key}"] = 1.0 if value else 0.0
            elif isinstance(value, (int, float)) and math.isfinite(float(value)):
                out[f"{prefix}{key}"] = float(value)
        return out


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_repo_path(path: str | Path | None = None) -> Path:
    raw = Path(path or os.environ.get("UGM_FAIRCHEM_REPO", DEFAULT_FAIRCHEM_REPO))
    if raw.is_absolute():
        return raw
    return _project_root() / raw


def _git_commit(repo: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return proc.stdout.strip()
    except Exception:
        return ""


def _prepend_fairchem_src(repo: Path) -> None:
    src = repo / "src"
    if src.exists() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


def fairchem_repo_status(repo_path: str | Path | None = None) -> dict[str, Any]:
    repo = _resolve_repo_path(repo_path)
    status: dict[str, Any] = {
        "repo_path": str(repo),
        "exists": repo.exists(),
        "is_git_repo": (repo / ".git").exists(),
        "commit": _git_commit(repo) if repo.exists() else "",
        "src_exists": (repo / "src").exists(),
        "importable": False,
        "available_models": [],
        "error": "",
    }
    if not repo.exists():
        status["error"] = "fairchem repository is not cloned"
        return status
    try:
        _prepend_fairchem_src(repo)
        from fairchem.core.calculate import pretrained_mlip  # type: ignore

        status["importable"] = True
        status["available_models"] = sorted(str(item) for item in getattr(pretrained_mlip, "available_models", []))
    except Exception as exc:
        status["error"] = f"{exc.__class__.__name__}: {exc}"
    return status


def _token_value(tokens: list[str], prefix: str) -> str:
    for token in tokens:
        if token.startswith(prefix):
            value = token.split(":", 1)[1].strip()
            if value and value != "valid_candidate":
                return value
    return ""


def _node_value(example: GraphExample, *node_types: str) -> str:
    wanted = set(node_types)
    for node in example.nodes:
        if node.type in wanted and node.value:
            return str(node.value)
    return ""


def candidate_smiles(example: GraphExample, predicted_tokens: list[str]) -> str:
    token_smiles = _token_value(predicted_tokens, "SMILES:")
    if token_smiles:
        return token_smiles
    for key in ("smiles", "SMILES", "canonical_smiles"):
        value = (example.metadata or {}).get(key)
        if value:
            return str(value)
    return _node_value(example, "smiles")


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
        return float((example.metadata or {}).get("temperature"))
    except Exception:
        return None


def _atoms_from_smiles(smiles: str):
    if not smiles:
        raise ValueError("missing SMILES for FairChem/UMA scoring")
    from ase import Atoms  # type: ignore
    from rdkit import Chem  # type: ignore
    from rdkit.Chem import AllChem  # type: ignore

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")
    charge = int(Chem.GetFormalCharge(mol))
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 17
    params.useRandomCoords = True
    embed_status = AllChem.EmbedMolecule(mol, params)
    if embed_status != 0:
        raise ValueError(f"RDKit conformer embedding failed for SMILES: {smiles}")
    if AllChem.MMFFHasAllMoleculeParams(mol):
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
    else:
        AllChem.UFFOptimizeMolecule(mol, maxIters=200)
    conf = mol.GetConformer()
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    positions = [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]
    atoms = Atoms(symbols=symbols, positions=positions, pbc=False)
    atoms.info["charge"] = charge
    atoms.info["spin"] = 1
    return atoms


def _atoms_from_symbols_positions(symbols: list[str], positions: list[list[float]], charge: int = 0, spin: int = 1):
    if not symbols:
        raise ValueError("missing atom symbols for FairChem/UMA scoring")
    if len(symbols) != len(positions):
        raise ValueError(f"symbol/position length mismatch: {len(symbols)} != {len(positions)}")
    from ase import Atoms  # type: ignore

    atoms = Atoms(symbols=symbols, positions=positions, pbc=False)
    atoms.info["charge"] = int(charge)
    atoms.info["spin"] = int(spin)
    return atoms


@lru_cache(maxsize=4)
def _fairchem_calculator(repo_path: str, model_name: str, task_name: str, device: str):
    repo = _resolve_repo_path(repo_path)
    _prepend_fairchem_src(repo)
    from fairchem.core import FAIRChemCalculator, pretrained_mlip  # type: ignore

    predictor = pretrained_mlip.get_predict_unit(model_name, device=device)
    return FAIRChemCalculator(predictor, task_name=task_name)


def _reward_from_energy_force(energy_ev: float, atom_count: int, force_rms: float | None, temperature_k: float | None) -> tuple[float, float]:
    n = max(1, atom_count)
    energy_per_atom = energy_ev / n
    finite_energy = math.isfinite(energy_per_atom)
    finite_forces = force_rms is not None and math.isfinite(force_rms)
    if not finite_energy:
        return 0.0, energy_per_atom
    force_term = 1.0 / (1.0 + max(force_rms or 0.0, 0.0))
    positive_energy_penalty = 1.0 / (1.0 + max(energy_per_atom, 0.0) * 0.05)
    raw = max(0.0, min(1.0, 0.65 * force_term + 0.35 * positive_energy_penalty))
    if not finite_forces:
        raw *= 0.7
    if temperature_k is not None:
        temp = max(300.0, min(400.0, float(temperature_k)))
        raw = raw ** (350.0 / temp)
    return max(1e-4, min(0.75, 0.75 * raw)), energy_per_atom


def _proxy_force_result(symbols: list[str], positions: list[list[float]], temperature_k: float | None) -> UmaOracleResult:
    """Deterministic smoke-test oracle over candidate coordinates.

    This is not used as a scientific oracle. It gives tests a differentiable
    force direction with no FairChem dependency.
    """
    forces: list[list[float]] = []
    energy = 0.0
    temp = max(300.0, min(400.0, float(temperature_k or 300.0)))
    spread = 1.0 + (temp - 300.0) / 100.0
    for pos in positions:
        xyz = [float(pos[i]) if i < len(pos) else 0.0 for i in range(3)]
        energy += 0.5 * sum((coord / spread) ** 2 for coord in xyz)
        forces.append([-(coord / (spread**2)) for coord in xyz])
    force_rms = math.sqrt(sum(sum(axis * axis for axis in force) for force in forces) / max(1, len(forces)))
    reward, energy_per_atom = _reward_from_energy_force(energy, len(symbols), force_rms, temperature_k)
    return UmaOracleResult(
        backend="proxy",
        available=True,
        reward=reward,
        score=reward / 0.75,
        temperature_k=temperature_k,
        atom_count=len(symbols),
        energy_ev=energy,
        energy_per_atom_ev=energy_per_atom,
        force_rms_ev_per_a=force_rms,
        forces_ev_per_a=forces,
        message="explicit deterministic proxy coordinate oracle; use only for tests and smoke runs",
    )


def _score_atoms_with_fairchem(
    atoms: Any,
    *,
    repo: Path,
    model: str,
    task: str,
    dev: str,
    temperature_k: float | None,
) -> UmaOracleResult:
    status = fairchem_repo_status(repo)
    if not status["exists"] or not status["importable"]:
        raise RuntimeError(status.get("error") or "FairChem repository is unavailable")
    calc = _fairchem_calculator(str(repo), model, task, dev)
    atoms.calc = calc
    energy_ev = float(atoms.get_potential_energy())
    forces = atoms.get_forces()
    force_rms = float((forces**2).sum(axis=1).mean() ** 0.5) if len(forces) else None
    reward, energy_per_atom = _reward_from_energy_force(energy_ev, len(atoms), force_rms, temperature_k)
    return UmaOracleResult(
        backend="fairchem",
        available=True,
        reward=reward,
        score=reward / 0.75,
        temperature_k=temperature_k,
        atom_count=int(len(atoms)),
        energy_ev=energy_ev,
        energy_per_atom_ev=energy_per_atom,
        force_rms_ev_per_a=force_rms,
        forces_ev_per_a=forces.tolist() if hasattr(forces, "tolist") else None,
        model_name=model,
        task_name=task,
        repo_path=str(repo),
        message="FairChem UMA score",
    )


def _proxy_result(example: GraphExample, predicted_tokens: list[str], proxy_reward: float | None) -> UmaOracleResult:
    reward = float(proxy_reward if proxy_reward is not None else 0.0)
    return UmaOracleResult(
        backend="proxy",
        available=True,
        reward=max(0.0, min(0.75, reward)),
        score=max(0.0, min(1.0, reward / 0.75 if reward else 0.0)),
        temperature_k=_temperature_kelvin(example),
        smiles=candidate_smiles(example, predicted_tokens),
        message="explicit deterministic proxy backend; use only for tests and smoke runs",
    )


def score_uma_oracle_candidate(
    example: GraphExample,
    predicted_tokens: list[str],
    *,
    backend: str | None = None,
    proxy_reward: float | None = None,
    strict: bool | None = None,
    repo_path: str | Path | None = None,
    model_name: str | None = None,
    task_name: str | None = None,
    device: str | None = None,
) -> UmaOracleResult:
    selected_backend = (backend or os.environ.get("UGM_UMA_BACKEND") or "fairchem").lower()
    strict_mode = bool(int(os.environ.get("UGM_UMA_STRICT", "0"))) if strict is None else strict
    repo = _resolve_repo_path(repo_path)
    model = model_name or os.environ.get("UGM_UMA_MODEL") or DEFAULT_UMA_MODEL
    task = task_name or os.environ.get("UGM_UMA_TASK") or DEFAULT_UMA_TASK
    dev = device or os.environ.get("UGM_UMA_DEVICE") or DEFAULT_UMA_DEVICE
    if selected_backend == "proxy":
        return _proxy_result(example, predicted_tokens, proxy_reward)
    if selected_backend not in {"fairchem", "uma"}:
        raise ValueError(f"Unknown UMA oracle backend: {selected_backend}")

    smiles = candidate_smiles(example, predicted_tokens)
    temp_k = _temperature_kelvin(example)
    try:
        atoms = _atoms_from_smiles(smiles)
        result = _score_atoms_with_fairchem(atoms, repo=repo, model=model, task=task, dev=dev, temperature_k=temp_k)
        result.smiles = smiles
        return result
    except Exception as exc:
        if strict_mode:
            raise
        return UmaOracleResult(
            backend="fairchem",
            available=False,
            reward=0.0,
            score=0.0,
            temperature_k=temp_k,
            smiles=smiles,
            model_name=model,
            task_name=task,
            repo_path=str(repo),
            message=f"{exc.__class__.__name__}: {exc}",
        )


def score_uma_coordinate_candidate(
    symbols: list[str],
    positions: list[list[float]],
    *,
    temperature_k: float | None = None,
    backend: str | None = None,
    strict: bool | None = None,
    repo_path: str | Path | None = None,
    model_name: str | None = None,
    task_name: str | None = None,
    device: str | None = None,
) -> UmaOracleResult:
    """Score model-proposed coordinates with UMA without structure labels."""
    selected_backend = (backend or os.environ.get("UGM_UMA_BACKEND") or "fairchem").lower()
    strict_mode = bool(int(os.environ.get("UGM_UMA_STRICT", "0"))) if strict is None else strict
    repo = _resolve_repo_path(repo_path)
    model = model_name or os.environ.get("UGM_UMA_MODEL") or DEFAULT_UMA_MODEL
    task = task_name or os.environ.get("UGM_UMA_TASK") or DEFAULT_UMA_TASK
    dev = device or os.environ.get("UGM_UMA_DEVICE") or DEFAULT_UMA_DEVICE
    if selected_backend == "proxy":
        return _proxy_force_result(symbols, positions, temperature_k)
    if selected_backend not in {"fairchem", "uma"}:
        raise ValueError(f"Unknown UMA oracle backend: {selected_backend}")
    try:
        atoms = _atoms_from_symbols_positions(symbols, positions)
        return _score_atoms_with_fairchem(atoms, repo=repo, model=model, task=task, dev=dev, temperature_k=temperature_k)
    except Exception as exc:
        if strict_mode:
            raise
        return UmaOracleResult(
            backend="fairchem",
            available=False,
            reward=0.0,
            score=0.0,
            temperature_k=temperature_k,
            atom_count=len(symbols),
            model_name=model,
            task_name=task,
            repo_path=str(repo),
            message=f"{exc.__class__.__name__}: {exc}",
        )
