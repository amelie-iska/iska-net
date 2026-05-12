from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class JacobianContactResult:
    available: bool
    source: str
    contacts: list[dict[str, Any]]
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def contacts_from_categorical_jacobian(
    scores: list[list[float]],
    *,
    spans: list[tuple[int, int, str]] | None = None,
    top_k: int = 256,
    min_score: float = 0.0,
    min_separation: int = 6,
) -> list[dict[str, Any]]:
    """Convert a categorical-Jacobian score map into contact-prior records.

    The score map can come from gLM2, Evo, an MSA/Potts comparator, or a cached
    Additional File. Positions are 1-based in the returned records.
    """
    n = len(scores)
    span_labels: list[str] = ["unknown"] * (n + 1)
    for start, end, label in spans or []:
        for pos in range(max(1, start), min(n, end) + 1):
            span_labels[pos] = label
    contacts: list[tuple[float, int, int, str]] = []
    for i in range(n):
        row = scores[i]
        for j in range(i + 1, min(n, len(row))):
            if j - i < min_separation and span_labels[i + 1] == span_labels[j + 1]:
                continue
            score = float(row[j])
            if score < min_score:
                continue
            kind = "inter_element" if span_labels[i + 1] != span_labels[j + 1] else "intra_element"
            contacts.append((score, i + 1, j + 1, kind))
    contacts.sort(reverse=True)
    return [{"src": i, "dst": j, "score": round(score, 6), "kind": kind, "source": "categorical_jacobian"} for score, i, j, kind in contacts[:top_k]]


def load_categorical_jacobian_contacts(path: str | Path, *, top_k: int = 256, min_score: float = 0.0, min_separation: int = 6) -> JacobianContactResult:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "contacts" in payload:
        contacts = list(payload.get("contacts") or [])[:top_k]
        return JacobianContactResult(True, "cached_categorical_jacobian_contacts", contacts, f"loaded {len(contacts)} cached contacts")
    scores = payload.get("scores") if isinstance(payload, dict) else payload
    spans_raw = payload.get("spans", []) if isinstance(payload, dict) else []
    spans = [(int(item["start"]), int(item["end"]), str(item.get("label", "element"))) for item in spans_raw if isinstance(item, dict)]
    contacts = contacts_from_categorical_jacobian(scores, spans=spans, top_k=top_k, min_score=min_score, min_separation=min_separation)
    return JacobianContactResult(True, "cached_categorical_jacobian_matrix", contacts, f"converted {len(contacts)} contacts from Jacobian matrix")


def compute_glm2_hidden_state_jacobian_contacts(
    sequence: str,
    *,
    spans: list[tuple[int, int, str]] | None = None,
    model_name: str = "tattabio/gLM2_650M",
    device: str = "cuda",
    top_k: int = 128,
    mutation_alphabet: str = "ACDEFGHIKLMNPQRSTVWY",
    max_positions: int = 256,
    strict: bool = False,
) -> JacobianContactResult:
    """Optional gLM2 perturbation contact prior.

    This is intentionally expensive and disabled by default. It approximates the
    categorical-Jacobian idea by mutating positions and measuring hidden-state
    response norms across all positions. For production-scale use, precompute
    and cache maps offline.
    """
    clean = str(sequence)
    if not clean:
        return JacobianContactResult(False, model_name, [], "empty sequence")
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except Exception as exc:
        if strict:
            raise
        return JacobianContactResult(False, model_name, [], f"transformers/torch unavailable: {exc}")
    try:
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32, trust_remote_code=True).eval().to(device)
        enc = tokenizer([clean], return_tensors="pt").to(device)
        with torch.no_grad():
            base = model(enc.input_ids, output_hidden_states=True).last_hidden_state[0].float()
        seq_len = min(max_positions, base.shape[0])
        scores = torch.zeros(seq_len, seq_len, dtype=torch.float32)
        chars = list(clean)
        for pos in range(seq_len):
            original = chars[pos]
            substitute = next((aa for aa in mutation_alphabet if aa != original), original)
            if substitute == original:
                continue
            chars[pos] = substitute
            mut_enc = tokenizer(["".join(chars)], return_tensors="pt").to(device)
            with torch.no_grad():
                mutated = model(mut_enc.input_ids, output_hidden_states=True).last_hidden_state[0].float()
            delta = (mutated[:seq_len] - base[:seq_len]).pow(2).mean(dim=-1).sqrt().cpu()
            scores[pos, : len(delta)] = delta
            chars[pos] = original
        contacts = contacts_from_categorical_jacobian(scores.tolist(), spans=spans, top_k=top_k, min_score=0.0, min_separation=6)
        return JacobianContactResult(True, model_name, contacts, f"computed {len(contacts)} approximate hidden-state Jacobian contacts")
    except Exception as exc:
        if strict:
            raise
        return JacobianContactResult(False, model_name, [], f"gLM2 Jacobian computation failed: {exc}")
