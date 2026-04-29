from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalNumericDiffusionHead(nn.Module):
    """Small conditional DDPM-style head for numeric graph fields.

    This implements the UniGenX-style idea that symbolic autoregression can be
    paired with a diffusion objective for numeric values. It predicts Gaussian
    noise added to normalized numeric targets conditioned on the graph hidden
    state and a diffusion timestep.
    """

    def __init__(self, hidden_dim: int, numeric_dim: int, diffusion_steps: int = 100, time_dim: int = 64):
        super().__init__()
        if numeric_dim <= 0:
            raise ValueError("numeric_dim must be positive")
        self.numeric_dim = numeric_dim
        self.diffusion_steps = max(2, int(diffusion_steps))
        self.time_embed = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + numeric_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, numeric_dim),
        )
        betas = torch.linspace(1e-4, 0.02, self.diffusion_steps)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("alpha_bar", alpha_bar, persistent=False)
        self.time_dim = time_dim

    def timestep_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.time_dim // 2
        freqs = torch.exp(-math.log(10_000) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / max(1, half - 1))
        args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if emb.size(1) < self.time_dim:
            emb = F.pad(emb, (0, self.time_dim - emb.size(1)))
        return emb

    def forward(self, context: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        batch = targets.size(0)
        device = targets.device
        timesteps = torch.randint(0, self.diffusion_steps, (batch,), device=device)
        alpha_bar = self.alpha_bar[timesteps].to(targets.dtype).view(batch, 1)
        noise = torch.randn_like(targets)
        noisy = alpha_bar.sqrt() * targets + (1.0 - alpha_bar).sqrt() * noise
        time_context = self.time_embed(self.timestep_embedding(timesteps).to(context.dtype))
        pred = self.net(torch.cat([context, noisy, time_context], dim=-1))
        denom = mask.float().sum().clamp_min(1.0)
        loss = ((pred - noise).pow(2) * mask.float()).sum() / denom
        return {"numeric_diffusion_loss": loss, "numeric_noise_pred": pred, "numeric_timesteps": timesteps}

    @torch.no_grad()
    def sample(self, context: torch.Tensor, shape: tuple[int, int] | None = None) -> torch.Tensor:
        if shape is None:
            shape = (context.size(0), self.numeric_dim)
        x = torch.randn(shape, device=context.device, dtype=context.dtype)
        for t in reversed(range(self.diffusion_steps)):
            timestep = torch.full((shape[0],), t, device=context.device, dtype=torch.long)
            time_context = self.time_embed(self.timestep_embedding(timestep).to(context.dtype))
            pred_noise = self.net(torch.cat([context, x, time_context], dim=-1))
            alpha_bar = self.alpha_bar[t].to(context.dtype)
            x = (x - (1.0 - alpha_bar).sqrt() * pred_noise) / alpha_bar.sqrt().clamp_min(1e-6)
            if t > 0:
                x = x + torch.randn_like(x) * 0.01
        return x

