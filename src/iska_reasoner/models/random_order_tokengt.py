from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from iska_reasoner.models.numeric_diffusion import ConditionalNumericDiffusionHead
from iska_reasoner.tropical import TropicalTransformerEncoder, TropicalTransformerEncoderLayer


@dataclass
class RandomOrderTokenGTConfig:
    vocab_size: int
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    ffn_dim: int = 512
    dropout: float = 0.1
    max_seq_len: int = 256
    max_nodes: int = 256
    max_slots: int = 128
    num_kinds: int = 8
    endpoint_dim: int = 32
    identifier_dim: int = 32
    max_identifiers: int = 0
    freeze_identifier_embeddings: bool = False
    source_numeric_dim: int = 4
    topology_dim: int = 7
    numeric_dim: int = 0
    numeric_diffusion_steps: int = 100
    gradient_checkpointing: bool = False
    lora_rank: int = 0
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    freeze_base_for_lora: bool = False
    attention_backend: str = "standard"
    tropical_use_projection: bool = True
    tropical_use_norm_shift: bool = True
    tropical_projective_normalize: bool = True
    tropical_symmetric: bool = True
    tropical_context_clamp: float = 20.0
    tropical_score_floor: float = -10_000.0
    tropical_eps: float = 1e-6


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        self.rank = rank
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5**0.5)
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_b(self.lora_a(self.dropout(x))) * self.scaling

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.base.bias


def _set_child(module: nn.Module, child_name: str, child: nn.Module) -> None:
    parts = child_name.split(".")
    parent = module
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], child)


class RandomOrderTokenGT(nn.Module):
    """Compact TokenGT-style random-order autoregressive model.

    The model keeps TokenGT's key idea that nodes and edges are both tokens.
    Source graph tokens receive both endpoint identifier embeddings and explicit
    structural identifier embeddings. Node tokens use endpoint pair `(i, i)`
    and identifier `i`; edge tokens use endpoint pair `(src, dst)` and an edge
    identifier from a disjoint range. Target graph tokens are decoded in sampled
    reveal orders with `<POS>` query tokens whose hidden states carry labels.
    """

    def __init__(self, cfg: RandomOrderTokenGTConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.hidden_dim)
        self.kind_embed = nn.Embedding(cfg.num_kinds, cfg.hidden_dim)
        self.position_embed = nn.Embedding(cfg.max_seq_len, cfg.hidden_dim)
        self.slot_embed = nn.Embedding(cfg.max_slots + 1, cfg.hidden_dim)
        self.endpoint_embed = nn.Embedding(cfg.max_nodes + 1, cfg.endpoint_dim)
        self.endpoint_proj = nn.Linear(2 * cfg.endpoint_dim, cfg.hidden_dim, bias=False)
        self.max_identifiers = cfg.max_identifiers if cfg.max_identifiers > 0 else (2 * cfg.max_nodes)
        self.identifier_embed = nn.Embedding(self.max_identifiers + 1, cfg.identifier_dim)
        self.identifier_proj = nn.Linear(cfg.identifier_dim, cfg.hidden_dim, bias=False)
        self.source_numeric_proj = nn.Linear(cfg.source_numeric_dim, cfg.hidden_dim, bias=False) if cfg.source_numeric_dim > 0 else None
        self._init_identifier_embeddings()
        if cfg.freeze_identifier_embeddings:
            self.identifier_embed.weight.requires_grad_(False)

        backend = cfg.attention_backend.strip().lower()
        if backend in {"standard", "softmax", "torch"}:
            layer = nn.TransformerEncoderLayer(
                d_model=cfg.hidden_dim,
                nhead=cfg.num_heads,
                dim_feedforward=cfg.ffn_dim,
                dropout=cfg.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.num_layers)
        elif backend in {"tropical", "mhta", "tropical_attention"}:
            layer = TropicalTransformerEncoderLayer(
                hidden_dim=cfg.hidden_dim,
                num_heads=cfg.num_heads,
                ffn_dim=cfg.ffn_dim,
                dropout=cfg.dropout,
                norm_first=True,
                use_projection=cfg.tropical_use_projection,
                use_norm_shift=cfg.tropical_use_norm_shift,
                projective_normalize=cfg.tropical_projective_normalize,
                symmetric=cfg.tropical_symmetric,
                context_clamp=cfg.tropical_context_clamp,
                score_floor=cfg.tropical_score_floor,
                eps=cfg.tropical_eps,
            )
            self.encoder = TropicalTransformerEncoder(layer, num_layers=cfg.num_layers)
        else:
            raise ValueError(f"Unknown attention_backend={cfg.attention_backend!r}; expected 'standard' or 'tropical'")
        self.norm = nn.LayerNorm(cfg.hidden_dim)
        self.lm_head = nn.Linear(cfg.hidden_dim, cfg.vocab_size, bias=False)
        self.value_head = nn.Linear(cfg.hidden_dim, 1)
        self.topology_head = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.topology_dim),
        )
        self.numeric_head = (
            ConditionalNumericDiffusionHead(cfg.hidden_dim, cfg.numeric_dim, cfg.numeric_diffusion_steps)
            if cfg.numeric_dim > 0
            else None
        )
        self.lm_head.weight = self.token_embed.weight
        if cfg.lora_rank > 0:
            try:
                torch.backends.mha.set_fastpath_enabled(False)
            except Exception:
                pass
            self.apply_lora(rank=cfg.lora_rank, alpha=cfg.lora_alpha, dropout=cfg.lora_dropout)
            if cfg.freeze_base_for_lora:
                self.freeze_non_lora_parameters()

    def _init_identifier_embeddings(self) -> None:
        """Initialize graph identifier rows as an orthogonal or semi-orthogonal table."""
        with torch.no_grad():
            self.identifier_embed.weight.zero_()
            rows = self.identifier_embed.weight[1:]
            if rows.numel() == 0:
                return
            if rows.size(1) >= rows.size(0):
                rows.copy_(torch.eye(rows.size(1), device=rows.device, dtype=rows.dtype)[: rows.size(0)])
            else:
                nn.init.orthogonal_(rows)
                rows.copy_(F.normalize(rows, dim=1))

    def apply_lora(self, rank: int, alpha: float, dropout: float) -> None:
        replacements: list[tuple[str, nn.Linear]] = []
        for name, module in self.named_modules():
            if (
                isinstance(module, nn.Linear)
                and name not in {"lm_head"}
                and not name.endswith(".base")
                and not name.endswith("self_attn.out_proj")
            ):
                replacements.append((name, module))
        for name, module in replacements:
            _set_child(self, name, LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout))

    def freeze_non_lora_parameters(self) -> None:
        for name, param in self.named_parameters():
            param.requires_grad = "lora_" in name

    def _encode(self, x: torch.Tensor, causal_mask: torch.Tensor | None, key_padding_mask: torch.Tensor) -> torch.Tensor:
        if not self.cfg.gradient_checkpointing or not self.training:
            return self.encoder(x, mask=causal_mask, src_key_padding_mask=key_padding_mask)
        hidden = x
        for layer in self.encoder.layers:
            def layer_forward(inp: torch.Tensor, layer=layer) -> torch.Tensor:
                return layer(inp, src_mask=causal_mask, src_key_padding_mask=key_padding_mask)
            hidden = checkpoint(layer_forward, hidden, use_reentrant=False)
        if self.encoder.norm is not None:
            hidden = self.encoder.norm(hidden)
        return hidden

    def forward(
        self,
        input_ids: torch.Tensor,
        kind_ids: torch.Tensor,
        slot_ids: torch.Tensor,
        endpoint_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        identifier_ids: torch.Tensor | None = None,
        source_numeric_features: torch.Tensor | None = None,
        causal_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        topology_targets: torch.Tensor | None = None,
        numeric_targets: torch.Tensor | None = None,
        numeric_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        bsz, seq_len = input_ids.shape
        device = input_ids.device
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, seq_len)
        endpoint_ids = endpoint_ids.clamp(0, self.cfg.max_nodes)
        endpoint_pair = self.endpoint_embed(endpoint_ids).reshape(bsz, seq_len, -1)
        if identifier_ids is None:
            identifier_ids = torch.zeros_like(input_ids)
        identifier_ids = identifier_ids.clamp(0, self.max_identifiers)

        x = (
            self.token_embed(input_ids)
            + self.kind_embed(kind_ids.clamp(0, self.cfg.num_kinds - 1))
            + self.position_embed(positions.clamp(0, self.cfg.max_seq_len - 1))
            + self.slot_embed(slot_ids.clamp(0, self.cfg.max_slots))
            + self.endpoint_proj(endpoint_pair)
            + self.identifier_proj(self.identifier_embed(identifier_ids))
        )
        if self.source_numeric_proj is not None:
            if source_numeric_features is None:
                source_numeric_features = torch.zeros(
                    bsz,
                    seq_len,
                    self.cfg.source_numeric_dim,
                    device=device,
                    dtype=x.dtype,
                )
            x = x + self.source_numeric_proj(source_numeric_features.to(device=device, dtype=x.dtype))
        key_padding_mask = ~attention_mask
        hidden = self._encode(x, causal_mask, key_padding_mask)
        hidden = self.norm(hidden)
        logits = self.lm_head(hidden)
        values = self.value_head(hidden).squeeze(-1)
        topology_pred = self.topology_head(hidden[:, 0])
        output = {"logits": logits, "hidden_states": hidden, "values": values, "topology_pred": topology_pred}
        attention_metrics = getattr(self.encoder, "last_attention_metrics", None)
        if attention_metrics:
            output["attention_metrics"] = attention_metrics
        if labels is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)
            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                mask = labels.ne(-100)
                acc = (pred[mask] == labels[mask]).float().mean() if mask.any() else torch.tensor(0.0, device=device)
            output["loss"] = loss
            output["token_accuracy"] = acc
        if topology_targets is not None:
            output["topology_loss"] = F.mse_loss(topology_pred, topology_targets.to(topology_pred.dtype))
        if self.numeric_head is not None and numeric_targets is not None and numeric_mask is not None:
            numeric_out = self.numeric_head(hidden[:, 0], numeric_targets.to(hidden.dtype), numeric_mask)
            output.update(numeric_out)
        return output

    def score_next_tokens(self, batch: dict[str, torch.Tensor], candidate_ids: torch.Tensor) -> torch.Tensor:
        """Score candidate token IDs at the last supervised `<POS>` position."""
        out = self.forward(
            input_ids=batch["input_ids"],
            kind_ids=batch["kind_ids"],
            slot_ids=batch["slot_ids"],
            endpoint_ids=batch["endpoint_ids"],
            identifier_ids=batch.get("identifier_ids"),
            source_numeric_features=batch.get("source_numeric_features"),
            attention_mask=batch["attention_mask"],
            causal_mask=batch.get("causal_mask"),
        )
        pos_mask = batch["kind_ids"].eq(4) & batch["attention_mask"]
        last_pos = pos_mask.long().argmax(dim=1)
        for i in range(pos_mask.size(0)):
            idxs = torch.where(pos_mask[i])[0]
            if len(idxs) > 0:
                last_pos[i] = idxs[-1]
        logits = out["logits"][torch.arange(batch["input_ids"].size(0), device=candidate_ids.device), last_pos]
        return logits.gather(1, candidate_ids)


def build_model_from_config(model_cfg: dict[str, int | float]) -> RandomOrderTokenGT:
    cfg = RandomOrderTokenGTConfig(**model_cfg)
    return RandomOrderTokenGT(cfg)
