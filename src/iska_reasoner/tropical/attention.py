from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # Optional FlashAttention-2 package path; PyTorch SDPA remains the fallback.
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import pad_input, unpad_input
except Exception:  # pragma: no cover - depends on local CUDA extension availability.
    flash_attn_func = None
    flash_attn_varlen_func = None
    pad_input = None
    unpad_input = None


class TropicalAttention(nn.Module):
    """Max-plus style attention for algorithmic-selection experiments.

    In soft mode this is ordinary temperature-scaled dot-product attention.
    In hard mode the output is the value vector at the maximum score. This is
    intentionally small and standalone so it can be used in ablations without
    replacing the main PyTorch transformer stack.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 1, hard: bool = False, temperature: float = 1.0):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.hard = hard
        self.temperature = temperature
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / max(self.head_dim**0.5, 1.0)
        scores = scores / max(float(self.temperature), 1e-6)
        if mask is not None:
            scores = scores.masked_fill(~mask[:, None, None, :].bool(), -1e9)
        if self.hard:
            index = scores.argmax(dim=-1)
            weights = torch.zeros_like(scores).scatter_(-1, index.unsqueeze(-1), 1.0)
        else:
            weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(weights, v).transpose(1, 2).reshape(batch, seq_len, self.hidden_dim)
        return self.out_proj(out), weights


class HeadwiseTropicalLinear(nn.Module):
    """Head-specific max-plus linear map.

    For input ``x[b, h, s, i]`` and weights ``W[h, o, i]``, the output is
    ``y[b, h, s, o] = max_i x[b, h, s, i] + W[h, o, i]``.
    """

    def __init__(self, num_heads: int, input_dim: int, output_dim: int, init_std: float = 0.02):
        super().__init__()
        self.num_heads = num_heads
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.empty(num_heads, output_dim, input_dim))
        nn.init.normal_(self.weight, mean=0.0, std=init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected [batch, heads, seq, dim], got {tuple(x.shape)}")
        return (x.unsqueeze(-2) + self.weight.unsqueeze(0).unsqueeze(2)).amax(dim=-1)


class MultiHeadTropicalAttention(nn.Module):
    """Masked Multi-Head Tropical Attention (MHTA).

    This module implements the paper's practical kernel:

    1. Euclidean Q/K/V projections.
    2. log-ReLU tropicalization with optional learned shift and projective
       normalization.
    3. Optional head-wise max-plus linear projections.
    4. Hilbert-projective distance scores.
    5. Max-plus value aggregation and Euclidean devaluation.

    It is intentionally explicit and quadratic in sequence length. Use it for
    graph/algorithmic reasoning ablations before enabling it for long-context
    training.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        use_projection: bool = True,
        use_norm_shift: bool = True,
        projective_normalize: bool = True,
        symmetric: bool = True,
        context_clamp: float = 20.0,
        score_floor: float = -10_000.0,
        query_chunk_size: int = 0,
        eps: float = 1e-6,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.use_projection = use_projection
        self.use_norm_shift = use_norm_shift
        self.projective_normalize = projective_normalize
        self.symmetric = symmetric
        self.context_clamp = float(context_clamp)
        self.score_floor = float(score_floor)
        self.query_chunk_size = int(query_chunk_size)
        self.eps = float(eps)

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        if use_projection:
            self.query_trop = HeadwiseTropicalLinear(num_heads, self.head_dim, self.head_dim)
            self.key_trop = HeadwiseTropicalLinear(num_heads, self.head_dim, self.head_dim)
            self.value_trop = HeadwiseTropicalLinear(num_heads, self.head_dim, self.head_dim)
        else:
            self.query_trop = None
            self.key_trop = None
            self.value_trop = None
        self.lambda_param = nn.Parameter(torch.zeros(1, 1, hidden_dim)) if use_norm_shift else None
        self.last_metrics: dict[str, torch.Tensor] = {}
        self.emit_contact_map = False
        self.detach_contact_map = True
        self.last_contact_map: torch.Tensor | None = None

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        return x.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _projective_normalize(self, x: torch.Tensor) -> torch.Tensor:
        return x - x.amax(dim=-1, keepdim=True)

    def _tropicalize(self, x: torch.Tensor, tropical_linear: HeadwiseTropicalLinear | None) -> torch.Tensor:
        x = torch.log1p(F.relu(x) + self.eps)
        if self.lambda_param is not None:
            x = x - self.lambda_param.to(dtype=x.dtype, device=x.device)
        x = self._split_heads(x)
        if self.projective_normalize:
            x = self._projective_normalize(x)
        if tropical_linear is not None:
            x = tropical_linear(x)
            if self.projective_normalize:
                x = self._projective_normalize(x)
        return x

    def _mask_scores(
        self,
        scores: torch.Tensor,
        attn_mask: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        disallowed = None
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                disallowed = attn_mask.to(device=scores.device).view(1, 1, scores.size(-2), scores.size(-1))
            else:
                scores = scores + attn_mask.to(device=scores.device, dtype=scores.dtype).view(1, 1, scores.size(-2), scores.size(-1))
        if key_padding_mask is not None:
            key_block = key_padding_mask.to(device=scores.device, dtype=torch.bool).view(scores.size(0), 1, 1, scores.size(-1))
            disallowed = key_block if disallowed is None else (disallowed | key_block)
        if disallowed is not None:
            scores = scores.masked_fill(disallowed, self.score_floor)
        return scores, disallowed

    def _score_and_context_chunk(
        self,
        q_chunk: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        diff = q_chunk.unsqueeze(3) - k.unsqueeze(2)
        if self.symmetric:
            distance = diff.amax(dim=-1) - diff.amin(dim=-1)
            scores = -distance
        else:
            min_diff = diff.amin(dim=-1)
            scores = -(diff.sum(dim=-1) - self.head_dim * min_diff)
        scores, disallowed = self._mask_scores(scores, attn_mask=attn_mask, key_padding_mask=key_padding_mask)

        context_terms = scores.unsqueeze(-1) + v.unsqueeze(2)
        if disallowed is not None:
            context_terms = context_terms.masked_fill(disallowed.unsqueeze(-1), self.score_floor)
        context = context_terms.amax(dim=3)
        if self.context_clamp > 0:
            context = context.clamp(min=-self.context_clamp, max=self.context_clamp)
        return context, scores, disallowed

    def _slice_attention_mask(self, attn_mask: torch.Tensor | None, start: int, end: int) -> torch.Tensor | None:
        if attn_mask is None:
            return None
        if attn_mask.dim() >= 2 and attn_mask.size(-2) >= end:
            return attn_mask[..., start:end, :]
        return attn_mask

    def _forward_tropical(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.size(2)
        chunk_size = int(self.query_chunk_size)
        if chunk_size <= 0 or seq_len <= chunk_size:
            context, scores, disallowed = self._score_and_context_chunk(q, k, v, attn_mask, key_padding_mask)
            self.last_metrics = self._collect_metrics(scores, context, disallowed, key_padding_mask)
            if self.emit_contact_map:
                self.last_contact_map = self._contact_map_from_scores(
                    scores,
                    disallowed,
                    key_padding_mask,
                    detach=self.detach_contact_map,
                )
            else:
                self.last_contact_map = None
            return context, scores.detach()

        context_chunks: list[torch.Tensor] = []
        score_chunks: list[torch.Tensor] = []
        disallowed_chunks: list[torch.Tensor] = []
        for start in range(0, seq_len, chunk_size):
            end = min(seq_len, start + chunk_size)
            mask_chunk = self._slice_attention_mask(attn_mask, start, end)
            context, scores, disallowed = self._score_and_context_chunk(
                q[:, :, start:end, :],
                k,
                v,
                mask_chunk,
                key_padding_mask,
            )
            context_chunks.append(context)
            score_chunks.append(scores)
            if disallowed is not None:
                disallowed_chunks.append(disallowed)
        full_context = torch.cat(context_chunks, dim=2)
        full_scores = torch.cat(score_chunks, dim=2)
        full_disallowed = torch.cat(disallowed_chunks, dim=2) if disallowed_chunks else None
        self.last_metrics = self._collect_metrics(full_scores, full_context, full_disallowed, key_padding_mask)
        if self.emit_contact_map:
            self.last_contact_map = self._contact_map_from_scores(
                full_scores,
                full_disallowed,
                key_padding_mask,
                detach=self.detach_contact_map,
            )
        else:
            self.last_contact_map = None
        return full_context, full_scores.detach()

    def _collect_metrics(
        self,
        scores: torch.Tensor,
        context: torch.Tensor,
        disallowed: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            score_values = scores.detach().float()
            if key_padding_mask is not None:
                valid_query = ~key_padding_mask.to(device=scores.device, dtype=torch.bool).view(scores.size(0), 1, scores.size(-2))
            else:
                valid_query = torch.ones(scores.size(0), 1, scores.size(-2), device=scores.device, dtype=torch.bool)
            valid_query = valid_query.expand(-1, scores.size(1), -1)
            if disallowed is not None:
                allowed = ~disallowed.expand(scores.size(0), scores.size(1), scores.size(-2), scores.size(-1))
                allowed = allowed & valid_query.unsqueeze(-1)
            else:
                allowed = valid_query.unsqueeze(-1).expand_as(score_values)
            if allowed.any():
                valid_scores = score_values[allowed]
                score_mean = valid_scores.mean()
                score_std = valid_scores.std(unbiased=False)
                distance_mean = (-valid_scores).mean()
            else:
                zero = score_values.new_tensor(0.0)
                score_mean = score_std = distance_mean = zero

            masked_scores = score_values.masked_fill(~allowed, float("-inf"))
            allowed_counts = allowed.sum(dim=-1)
            comparable_query = valid_query & allowed_counts.ge(2)
            top2 = torch.topk(masked_scores, k=min(2, masked_scores.size(-1)), dim=-1).values
            margins = top2[..., 0] - top2[..., 1] if top2.size(-1) > 1 else top2[..., 0]
            if comparable_query.any():
                top1_margin = margins[comparable_query].mean()
            else:
                top1_margin = score_values.new_tensor(0.0)
            if valid_query.any():
                probs = torch.softmax(masked_scores.masked_fill(~allowed, -80.0).clamp(min=-80.0, max=80.0), dim=-1)
                selection_confidence = probs.max(dim=-1).values[valid_query].mean()
                argmax_idx = masked_scores.argmax(dim=-1)[valid_query]
                unique_argmax_rate = score_values.new_tensor(float(torch.unique(argmax_idx).numel()) / max(1, score_values.size(-1)))
            else:
                selection_confidence = score_values.new_tensor(0.0)
                unique_argmax_rate = score_values.new_tensor(0.0)
            return {
                "tropical_attention/score_mean": score_mean,
                "tropical_attention/score_std": score_std,
                "tropical_attention/distance_mean": distance_mean,
                "tropical_attention/top1_margin": top1_margin,
                "tropical_attention/selection_confidence": selection_confidence,
                "tropical_attention/unique_argmax_rate": unique_argmax_rate,
                "tropical_attention/context_abs_mean": context.detach().float().abs().mean(),
            }

    def _contact_map_from_scores(
        self,
        scores: torch.Tensor,
        disallowed: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None,
        *,
        detach: bool = True,
    ) -> torch.Tensor:
        """Turn MHTA Hilbert-distance scores into attention-like contacts.

        MHTA scores are negative projective distances: larger values mean a
        stronger selected relation. For contact-map diagnostics we only need a
        detached, row-normalized coupling field, so this softmax does not alter
        the MHTA max-plus aggregation path.
        """

        score_values = scores.detach().float() if detach else scores.float()
        if key_padding_mask is not None:
            valid_query = ~key_padding_mask.to(device=scores.device, dtype=torch.bool).view(scores.size(0), 1, scores.size(-2))
        else:
            valid_query = torch.ones(scores.size(0), 1, scores.size(-2), device=scores.device, dtype=torch.bool)
        valid_query = valid_query.expand(-1, scores.size(1), -1)
        if disallowed is not None:
            allowed = ~disallowed.expand(scores.size(0), scores.size(1), scores.size(-2), scores.size(-1))
            allowed = allowed & valid_query.unsqueeze(-1)
        else:
            allowed = valid_query.unsqueeze(-1).expand_as(score_values)
        stable_scores = score_values.masked_fill(~allowed, -80.0).clamp(min=-80.0, max=80.0)
        contacts = torch.softmax(stable_scores, dim=-1)
        contacts = contacts * allowed.to(contacts.dtype)
        denom = contacts.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        contacts = contacts / denom
        return contacts.masked_fill(~valid_query.unsqueeze(-1), 0.0)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = self._tropicalize(self.q_proj(x), self.query_trop)
        k = self._tropicalize(self.k_proj(x), self.key_trop)
        v = self._tropicalize(self.v_proj(x), self.value_trop)

        context, scores = self._forward_tropical(q, k, v, attn_mask, key_padding_mask)
        context = torch.expm1(context)
        batch, _, seq_len, _ = context.shape
        context = context.transpose(1, 2).reshape(batch, seq_len, self.hidden_dim)
        return self.out_proj(self.dropout(context)), scores


class FlashSDPAAttention(nn.Module):
    """Flash-eligible softmax attention branch.

    This branch uses PyTorch's scaled-dot-product attention API. On CUDA,
    PyTorch dispatches this call to the fastest enabled SDPA kernel supported
    by the shape, dtype, mask, and device, including FlashAttention kernels
    when available. It intentionally does not request attention weights, because
    materializing them would disable the fast path.
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.dropout_p = float(dropout)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.last_metrics: dict[str, torch.Tensor] = {}
        self.last_contact_map: torch.Tensor | None = None
        if torch.cuda.is_available():
            try:
                torch.backends.cuda.enable_flash_sdp(True)
                torch.backends.cuda.enable_mem_efficient_sdp(True)
                torch.backends.cuda.enable_math_sdp(True)
            except Exception:
                pass

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        return x.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _additive_mask(
        self,
        batch: int,
        q_len: int,
        k_len: int,
        dtype: torch.dtype,
        device: torch.device,
        attn_mask: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        additive: torch.Tensor | None = None
        disallowed: torch.Tensor | None = None
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                disallowed = attn_mask.to(device=device).view(1, 1, q_len, k_len)
            else:
                additive = attn_mask.to(device=device, dtype=dtype).view(1, 1, q_len, k_len)
        if key_padding_mask is not None:
            key_block = key_padding_mask.to(device=device, dtype=torch.bool).view(batch, 1, 1, k_len)
            disallowed = key_block if disallowed is None else (disallowed | key_block)
        if disallowed is not None:
            if additive is None:
                additive = torch.zeros(batch, 1, q_len, k_len, device=device, dtype=dtype)
            elif additive.size(0) == 1 and batch > 1:
                additive = additive.expand(batch, -1, -1, -1).clone()
            additive = additive.masked_fill(disallowed, torch.finfo(dtype).min)
        return additive

    def _is_causal_mask(self, attn_mask: torch.Tensor | None, seq_len: int) -> bool:
        if attn_mask is None:
            return False
        if attn_mask.dtype != torch.bool or attn_mask.shape != (seq_len, seq_len):
            return False
        expected = torch.ones(seq_len, seq_len, device=attn_mask.device, dtype=torch.bool).triu(diagonal=1)
        return bool(torch.equal(attn_mask, expected))

    def _can_use_flash_attn_package(self, q: torch.Tensor, attn_mask: torch.Tensor | None) -> bool:
        if flash_attn_func is None or flash_attn_varlen_func is None or pad_input is None or unpad_input is None:
            return False
        if not q.is_cuda or q.dtype not in {torch.float16, torch.bfloat16}:
            return False
        if attn_mask is None:
            return True
        return self._is_causal_mask(attn_mask, q.size(1))

    def _flash_attn_package_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch, seq_len, _, _ = q.shape
        causal = self._is_causal_mask(attn_mask, seq_len)
        dropout_p = self.dropout_p if self.training else 0.0
        if key_padding_mask is None or not key_padding_mask.any():
            return flash_attn_func(q, k, v, dropout_p=dropout_p, causal=causal)
        valid_mask = ~key_padding_mask.to(device=q.device, dtype=torch.bool)
        q_unpad, indices, cu_seqlens, max_seqlen, _ = unpad_input(q, valid_mask)
        k_unpad, _, _, _, _ = unpad_input(k, valid_mask)
        v_unpad, _, _, _, _ = unpad_input(v, valid_mask)
        out_unpad = flash_attn_varlen_func(
            q_unpad,
            k_unpad,
            v_unpad,
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            dropout_p=dropout_p,
            causal=causal,
        )
        return pad_input(out_unpad, indices, batch, seq_len)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim)
        self.last_contact_map = None
        package_kernel_used = False
        package_kernel_failed = False
        if self._can_use_flash_attn_package(q, attn_mask):
            try:
                out = self._flash_attn_package_forward(q, k, v, attn_mask, key_padding_mask)
                package_kernel_used = True
            except Exception:
                package_kernel_failed = True
                out = None
        else:
            out = None
        if out is None:
            q_sdpa = q.transpose(1, 2)
            k_sdpa = k.transpose(1, 2)
            v_sdpa = v.transpose(1, 2)
            additive_mask = self._additive_mask(
                batch=batch,
                q_len=seq_len,
                k_len=seq_len,
                dtype=q_sdpa.dtype,
                device=q_sdpa.device,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
            )
            out = F.scaled_dot_product_attention(
                q_sdpa,
                k_sdpa,
                v_sdpa,
                attn_mask=additive_mask,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=False,
            ).transpose(1, 2)
        out = out.reshape(batch, seq_len, self.hidden_dim)
        out = self.out_proj(out)
        self.last_metrics = {
            "flash_attention/enabled": out.new_tensor(1.0),
            "flash_attention/package_kernel": out.new_tensor(1.0 if package_kernel_used else 0.0),
            "flash_attention/package_kernel_failed": out.new_tensor(1.0 if package_kernel_failed else 0.0),
            "flash_attention/sdpa_requested": out.new_tensor(0.0 if package_kernel_used else 1.0),
        }
        return out


class TropicalTransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with MHTA as the self-attention kernel."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        norm_first: bool = True,
        **attention_kwargs: object,
    ):
        super().__init__()
        self.norm_first = norm_first
        self.self_attn = MultiHeadTropicalAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            **attention_kwargs,
        )
        self.linear1 = nn.Linear(hidden_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.last_attention_metrics: dict[str, torch.Tensor] = {}
        self.last_attention_contact_map: torch.Tensor | None = None

    def _sa_block(self, x: torch.Tensor, src_mask: torch.Tensor | None, src_key_padding_mask: torch.Tensor | None) -> torch.Tensor:
        out, _ = self.self_attn(x, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        self.last_attention_metrics = self.self_attn.last_metrics
        self.last_attention_contact_map = self.self_attn.last_contact_map
        return self.dropout1(out)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout2(self.linear2(self.dropout(F.gelu(self.linear1(x)))))

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.norm_first:
            src = src + self._sa_block(self.norm1(src), src_mask, src_key_padding_mask)
            src = src + self._ff_block(self.norm2(src))
            return src
        src = self.norm1(src + self._sa_block(src, src_mask, src_key_padding_mask))
        return self.norm2(src + self._ff_block(src))


class HybridFlashTropicalTransformerEncoderLayer(nn.Module):
    """Encoder layer that uses both Flash-eligible SDPA and MHTA.

    The softmax branch uses PyTorch SDPA, which can dispatch to FlashAttention
    kernels. The tropical branch uses Hilbert-projective MHTA. The two outputs
    are merged before the residual update so the layer contains both algebras
    without trying to force FlashAttention into the max-plus MHTA computation.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        norm_first: bool = True,
        hybrid_softmax_weight: float = 1.0,
        hybrid_tropical_weight: float = 1.0,
        enable_tropical: bool = True,
        **attention_kwargs: object,
    ):
        super().__init__()
        self.norm_first = norm_first
        self.hybrid_softmax_weight = float(hybrid_softmax_weight)
        self.hybrid_tropical_weight = float(hybrid_tropical_weight)
        self.enable_tropical = bool(enable_tropical)
        self.softmax_attn = FlashSDPAAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
        self.tropical_attn = MultiHeadTropicalAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            **attention_kwargs,
        )
        self.linear1 = nn.Linear(hidden_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.last_attention_metrics: dict[str, torch.Tensor] = {}
        self.last_attention_contact_map: torch.Tensor | None = None

    def _sa_block(self, x: torch.Tensor, src_mask: torch.Tensor | None, src_key_padding_mask: torch.Tensor | None) -> torch.Tensor:
        softmax_out = self.softmax_attn(x, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        metrics = dict(self.softmax_attn.last_metrics)
        self.last_attention_contact_map = None
        if self.enable_tropical:
            tropical_out, _ = self.tropical_attn(x, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
            merged = self.hybrid_softmax_weight * softmax_out + self.hybrid_tropical_weight * tropical_out
            metrics.update(self.tropical_attn.last_metrics)
            self.last_attention_contact_map = self.tropical_attn.last_contact_map
            tropical_active = 1.0
        else:
            merged = self.hybrid_softmax_weight * softmax_out
            tropical_active = 0.0
        self.last_attention_metrics = {
            **metrics,
            "hybrid_attention/enabled": merged.new_tensor(1.0),
            "hybrid_attention/softmax_weight": merged.new_tensor(self.hybrid_softmax_weight),
            "hybrid_attention/tropical_weight": merged.new_tensor(self.hybrid_tropical_weight),
            "hybrid_attention/tropical_active": merged.new_tensor(tropical_active),
        }
        return self.dropout1(merged)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout2(self.linear2(self.dropout(F.gelu(self.linear1(x)))))

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.norm_first:
            src = src + self._sa_block(self.norm1(src), src_mask, src_key_padding_mask)
            src = src + self._ff_block(self.norm2(src))
            return src
        src = self.norm1(src + self._sa_block(src, src_mask, src_key_padding_mask))
        return self.norm2(src + self._ff_block(src))


class TropicalTransformerEncoder(nn.Module):
    """Small stack wrapper matching the subset of nn.TransformerEncoder used by UGM."""

    def __init__(
        self,
        layer: nn.Module,
        num_layers: int,
        norm: nn.Module | None = None,
        *,
        collect_contact_maps: bool = False,
        detach_contact_maps: bool = True,
    ):
        super().__init__()
        import copy

        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
        self.norm = norm
        self.collect_contact_maps = bool(collect_contact_maps)
        self.detach_contact_maps = bool(detach_contact_maps)
        self.last_attention_metrics: dict[str, torch.Tensor] = {}
        self.last_attention_maps: torch.Tensor | None = None
        for encoder_layer in self.layers:
            for module in encoder_layer.modules():
                if isinstance(module, MultiHeadTropicalAttention):
                    module.emit_contact_map = self.collect_contact_maps
                    module.detach_contact_map = self.detach_contact_maps

    def forward(
        self,
        src: torch.Tensor,
        mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output = src
        buckets: dict[str, list[torch.Tensor]] = defaultdict(list)
        contact_maps: list[torch.Tensor] = []
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            for key, value in layer.last_attention_metrics.items():
                buckets[key].append(value)
            if self.collect_contact_maps:
                contact_map = getattr(layer, "last_attention_contact_map", None)
                if contact_map is not None:
                    contact_maps.append(contact_map.detach() if self.detach_contact_maps else contact_map)
        if self.norm is not None:
            output = self.norm(output)
        self.last_attention_metrics = {
            key: torch.stack(values).mean()
            for key, values in buckets.items()
            if values
        }
        self.last_attention_maps = torch.stack(contact_maps, dim=1) if contact_maps else None
        if any(key.startswith("tropical_attention/") for key in self.last_attention_metrics):
            self.last_attention_metrics["tropical_attention/enabled"] = output.new_tensor(1.0)
        return output


@dataclass(slots=True)
class TropicalCellSignature:
    active_rate: float
    transition_rate: float
    unique_cells: float

    def metric_dict(self, prefix: str = "tropical/cell_") -> dict[str, float]:
        return {
            f"{prefix}active_rate": self.active_rate,
            f"{prefix}transition_rate": self.transition_rate,
            f"{prefix}unique_cells": self.unique_cells,
        }


@torch.no_grad()
def activation_cell_signature(hidden_states: torch.Tensor, threshold: float = 0.0) -> TropicalCellSignature:
    """Approximate tropical cell transitions by hidden-coordinate signs."""

    if hidden_states.numel() == 0:
        return TropicalCellSignature(0.0, 0.0, 0.0)
    signs = hidden_states.float().gt(threshold)
    active_rate = signs.float().mean().item()
    if signs.size(1) <= 1:
        transition_rate = 0.0
    else:
        transition_rate = signs[:, 1:].ne(signs[:, :-1]).float().mean().item()
    flat = signs.reshape(signs.size(0), signs.size(1), -1)
    # A compact signature: pack up to 32 sign bits for approximate cell counts.
    bits = flat[..., : min(32, flat.size(-1))].to(torch.int64)
    powers = (2 ** torch.arange(bits.size(-1), device=bits.device, dtype=torch.int64)).view(1, 1, -1)
    packed = (bits * powers).sum(dim=-1)
    unique_cells = float(torch.unique(packed).numel())
    return TropicalCellSignature(float(active_rate), float(transition_rate), unique_cells)


def tropical_max_spanning_arborescence(nodes: list[str], edge_scores: dict[tuple[str, str], float], root: str | None = None) -> list[tuple[str, str, float]]:
    """Select a maximum-score directed dependency tree.

    NetworkX supplies Chu-Liu/Edmonds through maximum_spanning_arborescence.
    The optional root is enforced by adding a high-scoring synthetic root edge
    and then removing it from the returned tree.
    """

    if not nodes:
        return []
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    synthetic_root = "__ROOT__"
    if root is not None:
        graph.add_node(synthetic_root)
        graph.add_edge(synthetic_root, root, weight=max(edge_scores.values(), default=0.0) + 1.0)
    for (src, dst), score in edge_scores.items():
        if src != dst and src in nodes and dst in nodes:
            graph.add_edge(src, dst, weight=float(score))
    if graph.number_of_edges() == 0:
        return []
    try:
        tree = nx.maximum_spanning_arborescence(graph, attr="weight")
    except Exception:
        return []
    out = []
    for src, dst, data in tree.edges(data=True):
        if src == synthetic_root or dst == synthetic_root:
            continue
        out.append((str(src), str(dst), float(data.get("weight", 0.0))))
    return out
