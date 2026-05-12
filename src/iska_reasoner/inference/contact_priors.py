from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ContactPriorResult:
    available: bool
    source: str
    sequence_length: int
    contacts: list[dict[str, Any]]
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _cache_path(cache_dir: str | Path, sequence: str, model_name: str, top_k: int, min_probability: float, min_separation: int) -> Path:
    key = hashlib.sha256(f"{model_name}|{top_k}|{min_probability}|{min_separation}|{sequence}".encode("utf-8")).hexdigest()[:24]
    return Path(cache_dir) / f"esm_contacts_{key}.json"


def _top_contacts(matrix: Any, *, top_k: int, min_probability: float, min_separation: int) -> list[dict[str, Any]]:
    try:
        import torch
    except Exception:
        torch = None  # type: ignore
    if torch is not None and hasattr(matrix, "detach"):
        matrix = matrix.detach().float().cpu()
        n = int(matrix.shape[0])
        records: list[tuple[float, int, int]] = []
        for i in range(n):
            for j in range(i + 1, n):
                if j - i < min_separation:
                    continue
                prob = float(matrix[i, j].item())
                if prob >= min_probability:
                    records.append((prob, i + 1, j + 1))
        records.sort(reverse=True)
        return [{"i": i, "j": j, "probability": round(prob, 6), "source": "esm_contact_regression"} for prob, i, j in records[:top_k]]
    n = len(matrix)
    records = []
    for i in range(n):
        for j in range(i + 1, n):
            if j - i < min_separation:
                continue
            prob = float(matrix[i][j])
            if prob >= min_probability:
                records.append((prob, i + 1, j + 1))
    records.sort(reverse=True)
    return [{"i": i, "j": j, "probability": round(prob, 6), "source": "esm_contact_regression"} for prob, i, j in records[:top_k]]


def predict_esm_contacts(
    sequence: str,
    *,
    model_name: str = "esm2_t33_650M_UR50D",
    device: str = "cuda",
    top_k: int = 256,
    min_probability: float = 0.20,
    min_separation: int = 6,
    cache_dir: str | Path | None = None,
    strict: bool = False,
) -> ContactPriorResult:
    """Predict protein residue contacts using the optional fair-esm package.

    This follows the ESM contact-prediction example pattern: load an ESM-2
    model, convert one sequence to tokens, and call ``model.predict_contacts``.
    The function is intentionally optional and cache-first because the 650M
    model is too expensive to run inside a training dataloader.
    """
    clean = "".join(ch for ch in str(sequence).upper() if ch.isalpha())
    if not clean:
        return ContactPriorResult(False, "esm", 0, [], "empty protein sequence")
    if cache_dir is not None:
        path = _cache_path(cache_dir, clean, model_name, top_k, min_probability, min_separation)
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                return ContactPriorResult(
                    bool(payload.get("available", True)),
                    str(payload.get("source", "esm_cache")),
                    int(payload.get("sequence_length", len(clean))),
                    list(payload.get("contacts", [])),
                    str(payload.get("message", "loaded cached ESM contacts")),
                )
            except Exception:
                pass
    try:
        import torch
        import esm
    except Exception as exc:
        if strict:
            raise
        return ContactPriorResult(False, "esm", len(clean), [], f"fair-esm unavailable: {exc}")
    try:
        factory = getattr(esm.pretrained, model_name)
        model, alphabet = factory()
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        model = model.eval().to(device)
        batch_converter = alphabet.get_batch_converter()
        _labels, _strs, tokens = batch_converter([("protein", clean)])
        tokens = tokens.to(device)
        with torch.no_grad():
            contacts = model.predict_contacts(tokens)[0]
        records = _top_contacts(contacts, top_k=top_k, min_probability=min_probability, min_separation=min_separation)
        result = ContactPriorResult(True, model_name, len(clean), records, f"predicted {len(records)} ESM contacts")
        if cache_dir is not None:
            path = _cache_path(cache_dir, clean, model_name, top_k, min_probability, min_separation)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
        return result
    except Exception as exc:
        if strict:
            raise
        return ContactPriorResult(False, model_name, len(clean), [], f"ESM contact prediction failed: {exc}")
