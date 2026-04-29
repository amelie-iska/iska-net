from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TrajectoryBatch:
    forward_logprobs: torch.Tensor
    backward_logprobs: torch.Tensor
    step_forward_logprobs: torch.Tensor
    step_backward_logprobs: torch.Tensor
    rewards: torch.Tensor
    lengths: torch.Tensor
    terminal_valid: torch.Tensor
    action_entropy: torch.Tensor
    terminal_state: torch.Tensor
    action_counts: torch.Tensor


class GraphSetPolicy(nn.Module):
    """Small policy for set-valued graph-of-thought construction.

    State is a binary vector over candidate graph tokens. The policy scores
    add-token actions. This is intentionally separate from the main LM so the
    trajectory-balance objective is easy to inspect and test.
    """

    def __init__(self, num_actions: int, hidden_dim: int = 128, context_dim: int = 0):
        super().__init__()
        self.context_dim = context_dim
        self.net = nn.Sequential(
            nn.Linear(num_actions + context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, state: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        x = state.float()
        if self.context_dim > 0:
            if context is None:
                context = torch.zeros(state.size(0), self.context_dim, device=state.device, dtype=state.dtype)
            x = torch.cat([x, context.to(state.dtype)], dim=-1)
        return self.net(x)


class TrajectoryBalanceLoss(nn.Module):
    def __init__(self, init_log_z: float = 0.0):
        super().__init__()
        self.log_z = nn.Parameter(torch.tensor(float(init_log_z)))

    def forward(self, batch: TrajectoryBatch) -> torch.Tensor:
        log_reward = torch.log(batch.rewards.clamp_min(1e-8))
        residual = self.log_z + batch.forward_logprobs - batch.backward_logprobs - log_reward
        return residual.pow(2).mean()


class SubtrajectoryBalanceLoss(nn.Module):
    """Lightweight subtrajectory-balance auxiliary objective.

    We use terminal rewards as the available potential and constrain every
    prefix flow to be numerically compatible with the same terminal scale. This
    is not a replacement for a learned state-flow model, but it supplies the
    subtrajectory credit-assignment pressure that the earlier scaffold lacked.
    """

    def __init__(self, init_log_z: float = 0.0):
        super().__init__()
        self.log_z = nn.Parameter(torch.tensor(float(init_log_z)))

    def forward(self, batch: TrajectoryBatch) -> torch.Tensor:
        log_reward = torch.log(batch.rewards.clamp_min(1e-8)).unsqueeze(1)
        forward_prefix = torch.cumsum(batch.step_forward_logprobs, dim=1)
        backward_prefix = torch.cumsum(batch.step_backward_logprobs, dim=1)
        residual = self.log_z + forward_prefix - backward_prefix - log_reward
        valid_steps = torch.arange(batch.step_forward_logprobs.size(1), device=batch.lengths.device).unsqueeze(0) < batch.lengths.long().unsqueeze(1)
        return residual.pow(2).masked_select(valid_steps).mean()


class GraphEditActionSpace:
    """Utility for graph-edit action labels beyond add-token actions."""

    def __init__(self, tokens: list[str], allow_delete: bool = True, allow_stop: bool = True):
        self.tokens = tokens
        self.actions = [f"ADD:{tok}" for tok in tokens]
        if allow_delete:
            self.actions.extend(f"DELETE:{tok}" for tok in tokens)
        if allow_stop:
            self.actions.append("STOP")

    def decode(self, action_index: int) -> tuple[str, str | None]:
        label = self.actions[action_index]
        if label == "STOP":
            return "STOP", None
        op, value = label.split(":", 1)
        return op, value


def sample_trajectories(
    policy: GraphSetPolicy,
    target_mask: torch.Tensor,
    max_steps: int,
    epsilon: float = 0.05,
    backward_policy: GraphSetPolicy | None = None,
    context: torch.Tensor | None = None,
) -> TrajectoryBatch:
    device = target_mask.device
    batch_size, num_actions = target_mask.shape
    state = torch.zeros(batch_size, num_actions, device=device)
    forward_logprobs = torch.zeros(batch_size, device=device)
    backward_logprobs = torch.zeros(batch_size, device=device)
    entropy_sum = torch.zeros(batch_size, device=device)
    lengths = torch.zeros(batch_size, device=device)
    step_forward: list[torch.Tensor] = []
    step_backward: list[torch.Tensor] = []
    action_counts = torch.zeros(num_actions, device=device)

    for _ in range(max_steps):
        logits = policy(state, context=context)
        available = state.lt(0.5)
        masked_logits = logits.masked_fill(~available, -1e9)
        probs = torch.softmax(masked_logits, dim=-1)
        if epsilon > 0:
            uniform = available.float() / available.float().sum(dim=-1, keepdim=True).clamp_min(1.0)
            probs = (1.0 - epsilon) * probs + epsilon * uniform
        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample()
        logp = dist.log_prob(action)
        forward_logprobs += logp
        step_forward.append(logp)
        entropy_sum += dist.entropy()
        next_state = state.clone()
        next_state[torch.arange(batch_size, device=device), action] = 1.0
        state = next_state
        lengths += 1
        action_counts.scatter_add_(0, action, torch.ones(action.shape, device=device, dtype=action_counts.dtype))

        selected = state.ge(0.5)
        if backward_policy is None:
            selected_counts = state.sum(dim=-1).clamp_min(1.0)
            blogp = -torch.log(selected_counts)
        else:
            backward_logits = backward_policy(state, context=context).masked_fill(~selected, -1e9)
            backward_dist = torch.distributions.Categorical(logits=backward_logits)
            blogp = backward_dist.log_prob(action)
        backward_logprobs += blogp
        step_backward.append(blogp)

        if torch.all(state.ge(target_mask)):
            break

    selected_target = (state * target_mask).sum(dim=-1)
    target_count = target_mask.sum(dim=-1).clamp_min(1.0)
    extra = (state * (1.0 - target_mask)).sum(dim=-1)
    missing = (target_mask * (1.0 - state)).sum(dim=-1)
    terminal_valid = (extra.eq(0) & missing.eq(0)).float()
    recall_reward = selected_target / target_count
    penalty = 0.05 * extra
    rewards = (0.05 + terminal_valid + 0.45 * recall_reward - penalty).clamp_min(1e-4)
    if step_forward:
        step_forward_tensor = torch.stack(step_forward, dim=1)
        step_backward_tensor = torch.stack(step_backward, dim=1)
    else:
        step_forward_tensor = torch.zeros(batch_size, 0, device=device)
        step_backward_tensor = torch.zeros(batch_size, 0, device=device)
    return TrajectoryBatch(
        forward_logprobs=forward_logprobs,
        backward_logprobs=backward_logprobs,
        step_forward_logprobs=step_forward_tensor,
        step_backward_logprobs=step_backward_tensor,
        rewards=rewards,
        lengths=lengths,
        terminal_valid=terminal_valid,
        action_entropy=entropy_sum / lengths.clamp_min(1.0),
        terminal_state=state.detach(),
        action_counts=action_counts.detach(),
    )
