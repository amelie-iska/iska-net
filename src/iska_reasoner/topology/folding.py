from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def attention_contact_field(
    attention_maps: torch.Tensor,
    token_mask: torch.Tensor | None = None,
    *,
    normalize: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Convert attention maps into a symmetric contact-like coupling field.

    The input may be shaped as ``[N, N]``, ``[M, N, N]``,
    ``[B, H, N, N]``, or ``[B, L, H, N, N]``. Head/layer dimensions are
    averaged. The returned tensor is ``[B, N, N]`` with zero diagonal.
    """

    if attention_maps.numel() == 0:
        raise ValueError("attention_maps must be non-empty")
    attn = attention_maps.float()
    if attn.dim() == 2:
        field = attn.unsqueeze(0)
    elif attn.dim() == 3:
        if token_mask is not None and token_mask.dim() == 2 and attn.size(0) == token_mask.size(0):
            field = attn
        else:
            field = attn.mean(dim=0, keepdim=True)
    elif attn.dim() == 4:
        field = attn.mean(dim=1)
    elif attn.dim() == 5:
        field = attn.mean(dim=(1, 2))
    else:
        leading = int(torch.tensor(attn.shape[:-2]).prod().item())
        field = attn.reshape(leading, attn.size(-2), attn.size(-1)).mean(dim=0, keepdim=True)
    field = 0.5 * (field + field.transpose(-1, -2))
    field = _mask_and_zero_diag(field, token_mask)
    if normalize:
        denom = field.amax(dim=(-2, -1), keepdim=True).clamp_min(eps)
        field = field / denom
    return field.clamp(0.0, 1.0)


def embedding_contact_fields(
    hidden_states: torch.Tensor,
    token_mask: torch.Tensor | None = None,
    *,
    euclidean_scale: float | None = None,
    js_scale: float = 0.25,
    exact_js_max_tokens: int = 256,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build Euclidean and Jensen-Shannon contact fields from hidden states."""

    hidden = hidden_states.float()
    if hidden.dim() == 2:
        hidden = hidden.unsqueeze(0)
    if hidden.dim() != 3:
        raise ValueError("hidden_states must have shape [N,D] or [B,N,D]")
    distances = torch.cdist(hidden, hidden, p=2)
    if euclidean_scale is None:
        upper_mask = torch.triu(torch.ones_like(distances, dtype=torch.bool), diagonal=1)
        if token_mask is not None:
            mask = _as_batch_mask(token_mask, hidden.size(0), hidden.size(1), hidden.device)
            upper_mask = upper_mask & (mask.unsqueeze(1) & mask.unsqueeze(2))
        upper = distances[upper_mask]
        scale = upper.mean().clamp_min(eps) if upper.numel() else torch.tensor(1.0, device=hidden.device)
    else:
        scale = torch.tensor(float(euclidean_scale), device=hidden.device).clamp_min(eps)
    euclidean_field = torch.exp(-torch.square(distances / scale))

    probs = F.softmax(hidden, dim=-1)
    if hidden.size(1) <= int(exact_js_max_tokens):
        js = _pairwise_js_distance(probs)
        js_scale_t = torch.tensor(float(js_scale), device=hidden.device).clamp_min(eps)
        js_field = torch.exp(-torch.square(js / js_scale_t))
    else:
        js_field = _probability_overlap_contact(probs, eps=eps)

    return _mask_and_zero_diag(euclidean_field, token_mask), _mask_and_zero_diag(js_field, token_mask)


def folding_contact_field(
    *,
    attention_maps: torch.Tensor | None = None,
    hidden_states: torch.Tensor | None = None,
    token_mask: torch.Tensor | None = None,
    attention_weight: float = 0.60,
    embedding_weight: float = 0.25,
    js_weight: float = 0.15,
) -> torch.Tensor:
    """Fuse attention, Euclidean hidden geometry, and JS geometry into contacts.

    This is a sequence-only fold-contact proxy. It does not use ground-truth
    coordinates; it summarizes whether evolving attention and hidden-state
    geometry are forming persistent long-range coupling hypotheses that a
    downstream graph decoder can render as atom/coordinate/frame records.
    """

    pieces: list[torch.Tensor] = []
    weights: list[float] = []
    if attention_maps is not None:
        pieces.append(attention_contact_field(attention_maps, token_mask=token_mask))
        weights.append(float(attention_weight))
    if hidden_states is not None:
        emb_field, js_field = embedding_contact_fields(hidden_states, token_mask=token_mask)
        pieces.extend([emb_field, js_field])
        weights.extend([float(embedding_weight), float(js_weight)])
    if not pieces:
        raise ValueError("attention_maps or hidden_states must be provided")
    batch = max(piece.size(0) for piece in pieces)
    expanded = [piece.expand(batch, -1, -1) if piece.size(0) == 1 and batch > 1 else piece for piece in pieces]
    total_weight = sum(max(0.0, weight) for weight in weights)
    if total_weight <= 0:
        total_weight = float(len(expanded))
        weights = [1.0] * len(expanded)
    field = sum(max(0.0, weight) * piece for weight, piece in zip(weights, expanded)) / total_weight
    return _mask_and_zero_diag(field.clamp(0.0, 1.0), token_mask)


@torch.no_grad()
def folding_contact_metrics(contact_field: torch.Tensor, token_mask: torch.Tensor | None = None, prefix: str = "folding_contact/") -> dict[str, float]:
    """Small metrics for attention-derived fold-contact fields."""

    field = contact_field.float()
    if field.dim() == 2:
        field = field.unsqueeze(0)
    upper_values = []
    for idx in range(field.size(0)):
        mat = field[idx]
        mask = torch.triu(torch.ones_like(mat, dtype=torch.bool), diagonal=1)
        if token_mask is not None:
            batch_mask = _as_batch_mask(token_mask, field.size(0), field.size(-1), field.device)[idx]
            mask = mask & (batch_mask.unsqueeze(0) & batch_mask.unsqueeze(1))
        values = mat[mask]
        if values.numel():
            upper_values.append(values)
    if not upper_values:
        return {
            f"{prefix}mean": 0.0,
            f"{prefix}std": 0.0,
            f"{prefix}entropy": 0.0,
            f"{prefix}density_05": 0.0,
            f"{prefix}density_08": 0.0,
            f"{prefix}top_contact_mean": 0.0,
            f"{prefix}effective_contact_count": 0.0,
        }
    values = torch.cat(upper_values)
    hist = torch.histc(values.detach().cpu(), bins=16, min=0.0, max=1.0)
    probs = hist.float() / hist.sum().clamp_min(1.0)
    entropy = -(probs * probs.clamp_min(1e-8).log()).sum()
    top_k = max(1, int(0.05 * values.numel()))
    top_mean = values.topk(top_k).values.mean()
    return {
        f"{prefix}mean": float(values.mean().item()),
        f"{prefix}std": float(values.std(unbiased=False).item()),
        f"{prefix}entropy": float(entropy.item()),
        f"{prefix}density_05": float(values.ge(0.5).float().mean().item()),
        f"{prefix}density_08": float(values.ge(0.8).float().mean().item()),
        f"{prefix}top_contact_mean": float(top_mean.item()),
        f"{prefix}effective_contact_count": float(torch.exp(entropy).item()),
    }


def folding_attention_coordinate_consistency_loss(
    contact_field: torch.Tensor,
    coordinates: torch.Tensor,
    token_mask: torch.Tensor | None = None,
    *,
    contact_radius: float = 8.0,
    softness: float = 1.0,
) -> torch.Tensor:
    """Self-consistency between inferred contacts and generated coordinates.

    This loss is for generated candidate coordinates, not ground-truth
    structure labels. It makes a rendered PDB hypothesis agree with the
    contact field induced by attention/embedding/JS geometry.
    """

    field = contact_field.float()
    coords = coordinates.float()
    if field.dim() == 2:
        field = field.unsqueeze(0)
    if coords.dim() == 2:
        coords = coords.unsqueeze(0)
    if field.size(0) == 1 and coords.size(0) > 1:
        field = field.expand(coords.size(0), -1, -1)
    if coords.size(0) == 1 and field.size(0) > 1:
        coords = coords.expand(field.size(0), -1, -1)
    distances = torch.cdist(coords, coords, p=2)
    target = torch.sigmoid((float(contact_radius) - distances) / max(float(softness), 1e-6))
    mask = torch.triu(torch.ones_like(field, dtype=torch.bool), diagonal=1)
    if token_mask is not None:
        batch_mask = _as_batch_mask(token_mask, field.size(0), field.size(-1), field.device)
        mask = mask & (batch_mask.unsqueeze(1) & batch_mask.unsqueeze(2))
    if not bool(mask.any()):
        return field.sum() * 0.0
    return F.mse_loss(field[mask], target[mask])


def uma_contact_alignment_loss(
    contact_field: torch.Tensor,
    hidden_states: torch.Tensor,
    examples: list[Any],
    token_mask: torch.Tensor | None = None,
    *,
    embedding_weight: float = 0.5,
    temperature_diversity_weight: float = 0.5,
    max_tokens: int = 192,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Align contact and embedding geometry to UMA feedback records.

    The target is derived only from explicit UMA-stage target records such as
    ``ATTN_BIN:*:bNN``, ``TOKEN_COUPLING:uma:*:bNN``,
    ``UMA_INFLUENCE:uma:*:bNN``, and ``UMA_TRAJ_BIN:*:bNN``. This keeps
    ordinary sequence/function rows free of hidden structure supervision while
    allowing oracle-scored rows to shape both contact-map density and embedding
    geometry.
    """

    if not examples:
        zero = hidden_states.sum() * 0.0
        return zero, {
            "uma_contact/alignment_examples": 0.0,
            "uma_contact/target_strength": 0.0,
            "uma_contact/contact_strength": 0.0,
            "uma_contact/embedding_strength": 0.0,
            "uma_contact/temperature_diversity_target": 0.0,
            "uma_contact/contact_entropy": 0.0,
            "uma_contact/embedding_entropy": 0.0,
        }

    field = contact_field.float()
    hidden = hidden_states.float()
    if field.dim() == 2:
        field = field.unsqueeze(0)
    if hidden.dim() == 2:
        hidden = hidden.unsqueeze(0)
    batch = min(field.size(0), hidden.size(0), len(examples))
    if batch == 0:
        zero = hidden_states.sum() * 0.0
        return zero, {
            "uma_contact/alignment_examples": 0.0,
            "uma_contact/target_strength": 0.0,
            "uma_contact/contact_strength": 0.0,
            "uma_contact/embedding_strength": 0.0,
            "uma_contact/temperature_diversity_target": 0.0,
            "uma_contact/contact_entropy": 0.0,
            "uma_contact/embedding_entropy": 0.0,
        }
    field = field[:batch]
    hidden = hidden[:batch]
    mask = _as_batch_mask(token_mask, batch, field.size(-1), field.device) if token_mask is not None else torch.ones(batch, field.size(-1), dtype=torch.bool, device=field.device)
    if field.size(-1) > max(2, int(max_tokens)):
        idx = _sample_token_indices(field.size(-1), int(max_tokens), field.device)
        field = field.index_select(-2, idx).index_select(-1, idx)
        hidden = hidden.index_select(1, idx)
        mask = mask.index_select(1, idx)
    emb_field, js_field = embedding_contact_fields(hidden, token_mask=mask)
    embedding_proxy = 0.5 * (emb_field + js_field)

    losses: list[torch.Tensor] = []
    targets: list[float] = []
    diversity_targets: list[float] = []
    contact_strengths: list[torch.Tensor] = []
    embedding_strengths: list[torch.Tensor] = []
    contact_entropies: list[torch.Tensor] = []
    embedding_entropies: list[torch.Tensor] = []
    for row, example in enumerate(examples[:batch]):
        target = _uma_feedback_strength(example)
        if target is None:
            continue
        pair_mask = torch.triu(torch.ones(field.size(-1), field.size(-1), dtype=torch.bool, device=field.device), diagonal=1)
        pair_mask = pair_mask & (mask[row].unsqueeze(0) & mask[row].unsqueeze(1))
        if not bool(pair_mask.any()):
            continue
        target_t = field.new_tensor(float(target))
        contact_strength = field[row][pair_mask].mean()
        embedding_strength = embedding_proxy[row][pair_mask].mean()
        loss = F.mse_loss(contact_strength, target_t)
        if embedding_weight > 0:
            loss = loss + float(embedding_weight) * F.mse_loss(embedding_strength, target_t)
        diversity_target = _temperature_diversity_target(example)
        if diversity_target is not None and temperature_diversity_weight > 0:
            diversity_t = field.new_tensor(float(diversity_target))
            contact_entropy = _normalized_pair_entropy(field[row], pair_mask)
            embedding_entropy = _normalized_pair_entropy(embedding_proxy[row], pair_mask)
            loss = loss + float(temperature_diversity_weight) * F.mse_loss(contact_entropy, diversity_t)
            if embedding_weight > 0:
                loss = loss + float(temperature_diversity_weight) * float(embedding_weight) * F.mse_loss(embedding_entropy, diversity_t)
            diversity_targets.append(float(diversity_target))
            contact_entropies.append(contact_entropy.detach())
            embedding_entropies.append(embedding_entropy.detach())
        losses.append(loss)
        targets.append(float(target))
        contact_strengths.append(contact_strength.detach())
        embedding_strengths.append(embedding_strength.detach())

    if not losses:
        zero = hidden_states.sum() * 0.0
        return zero, {
            "uma_contact/alignment_examples": 0.0,
            "uma_contact/target_strength": 0.0,
            "uma_contact/contact_strength": 0.0,
            "uma_contact/embedding_strength": 0.0,
            "uma_contact/temperature_diversity_target": 0.0,
            "uma_contact/contact_entropy": 0.0,
            "uma_contact/embedding_entropy": 0.0,
        }
    loss = torch.stack(losses).mean()
    return loss, {
        "uma_contact/alignment_examples": float(len(losses)),
        "uma_contact/target_strength": float(sum(targets) / max(1, len(targets))),
        "uma_contact/contact_strength": float(torch.stack(contact_strengths).mean().item()),
        "uma_contact/embedding_strength": float(torch.stack(embedding_strengths).mean().item()),
        "uma_contact/temperature_diversity_target": float(sum(diversity_targets) / max(1, len(diversity_targets))) if diversity_targets else 0.0,
        "uma_contact/contact_entropy": float(torch.stack(contact_entropies).mean().item()) if contact_entropies else 0.0,
        "uma_contact/embedding_entropy": float(torch.stack(embedding_entropies).mean().item()) if embedding_entropies else 0.0,
    }


def _pairwise_js_distance(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = probs.clamp_min(eps)
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(eps)
    q = p.unsqueeze(1)
    p2 = p.unsqueeze(2)
    m = 0.5 * (p2 + q)
    js = 0.5 * (p2 * (p2.log() - m.clamp_min(eps).log())).sum(dim=-1)
    js = js + 0.5 * (q * (q.log() - m.clamp_min(eps).log())).sum(dim=-1)
    return js.clamp_min(0.0).sqrt()


def _probability_overlap_contact(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Memory-safe probability-geometry proxy for long hidden sequences.

    Exact pairwise Jensen-Shannon distance materializes ``[B,N,N,D]``. That is
    fine for small diagnostics and not fine inside 926-token training batches.
    For long sequences, use normalized probability overlap as a bounded
    contact-like field with the same shape.
    """

    p = probs.float().clamp_min(eps)
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(eps)
    overlap = torch.bmm(p, p.transpose(1, 2))
    denom = torch.linalg.vector_norm(p, ord=2, dim=-1).clamp_min(eps)
    overlap = overlap / (denom.unsqueeze(2) * denom.unsqueeze(1)).clamp_min(eps)
    return overlap.clamp(0.0, 1.0)


def _sample_token_indices(n: int, max_tokens: int, device: torch.device) -> torch.Tensor:
    if n <= max_tokens:
        return torch.arange(n, device=device)
    return torch.linspace(0, n - 1, steps=max_tokens, device=device).round().to(torch.long).unique(sorted=True)


def _as_batch_mask(mask: torch.Tensor, batch: int, n: int, device: torch.device) -> torch.Tensor:
    out = mask.to(device=device, dtype=torch.bool)
    if out.dim() == 1:
        out = out.unsqueeze(0)
    if out.size(0) == 1 and batch > 1:
        out = out.expand(batch, -1)
    return out[:, :n]


def _mask_and_zero_diag(field: torch.Tensor, token_mask: torch.Tensor | None) -> torch.Tensor:
    out = field.clone()
    if token_mask is not None:
        mask = _as_batch_mask(token_mask, out.size(0), out.size(-1), out.device)
        out = out * (mask.unsqueeze(1) & mask.unsqueeze(2)).to(out.dtype)
    eye = torch.eye(out.size(-1), dtype=torch.bool, device=out.device)
    out = out.masked_fill(eye.unsqueeze(0), 0.0)
    return out


def _uma_feedback_strength(example: Any) -> float | None:
    tokens = getattr(example, "target_tokens", None) or []
    strengths: list[float] = []
    uma_prefixes = (
        "ATTN_BIN:",
        "ATTN_COARSE:",
        "TOKEN_COUPLING:uma:",
        "UMA_INFLUENCE:uma:",
        "TOKEN_MOTION:uma:",
        "UMA_TRAJ_BIN:",
    )
    for token in tokens:
        if not isinstance(token, str) or not token.startswith(uma_prefixes):
            continue
        value = _feedback_token_value(token)
        if value is not None:
            strengths.append(value)
    if any(token == "SEQ_STRUCT_DYN_PROXY:uma_scored" for token in tokens):
        strengths.append(0.5)
    if not strengths:
        return None
    return max(0.0, min(1.0, sum(strengths) / len(strengths)))


def _temperature_diversity_target(example: Any) -> float | None:
    if _uma_feedback_strength(example) is None:
        return None
    temp_k = None
    for node in getattr(example, "nodes", []) or []:
        if getattr(node, "type", "") != "temperature":
            continue
        features = getattr(node, "features", {}) or {}
        for key in ("kelvin", "kelvin_clamped", "temperature_k"):
            try:
                temp_k = float(features[key])
                break
            except Exception:
                pass
        if temp_k is None:
            try:
                temp_k = float(str(getattr(node, "value", "")).rstrip("Kk"))
            except Exception:
                pass
        if temp_k is not None:
            break
    if temp_k is None:
        metadata = getattr(example, "metadata", {}) or {}
        try:
            temp_k = float(metadata.get("temperature"))
        except Exception:
            return None
    return max(0.0, min(1.0, (float(temp_k) - 300.0) / 100.0))


def _feedback_token_value(token: str) -> float | None:
    tail = token.rsplit(":", 1)[-1]
    if len(tail) == 3 and tail.startswith("b") and tail[1:].isdigit():
        return int(tail[1:]) / 63.0
    coarse = {
        "low": 0.20,
        "medium": 0.50,
        "high": 0.80,
        "critical": 1.00,
        "stabilize": 0.70,
        "refine": 0.75,
        "diversify": 0.35,
        "explore": 0.40,
        "contract": 0.65,
        "expand": 0.45,
        "trajectory_follow": 0.80,
        "oracle_reject": 0.10,
        "oracle_accept": 0.90,
    }
    return coarse.get(tail)


def _normalized_pair_entropy(values: torch.Tensor, pair_mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    selected = values[pair_mask].float().clamp_min(0.0)
    if selected.numel() <= 1:
        return selected.sum() * 0.0
    probs = selected + eps
    probs = probs / probs.sum().clamp_min(eps)
    entropy = -(probs * probs.clamp_min(eps).log()).sum()
    return entropy / torch.log(values.new_tensor(float(selected.numel())).clamp_min(2.0))
