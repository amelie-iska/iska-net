import torch

from iska_reasoner.gflownet.trainer import _temperature_diversity_bonuses
from iska_reasoner.gflownet.trajectory import GraphSetPolicy, TrajectoryBalanceLoss, sample_trajectories
from iska_reasoner.graph.schema import GraphExample, Node


def test_trajectory_balance_smoke():
    target_mask = torch.tensor([[1, 1, 0, 0], [1, 0, 1, 0]], dtype=torch.float32)
    policy = GraphSetPolicy(num_actions=4, hidden_dim=16)
    loss_fn = TrajectoryBalanceLoss()
    traj = sample_trajectories(policy, target_mask, max_steps=3, epsilon=0.1)
    loss = loss_fn(traj)
    assert loss.isfinite()
    loss.backward()
    assert loss_fn.log_z.grad is not None


def test_temperature_diversity_bonus_prefers_high_temperature_terminal_variation():
    examples = [
        GraphExample(
            id="hot0",
            task="multimodal",
            nodes=[Node(id="temperature", type="temperature", value="400K", features={"kelvin": 400.0})],
            edges=[],
            target_tokens=[],
        ),
        GraphExample(
            id="hot1",
            task="multimodal",
            nodes=[Node(id="temperature", type="temperature", value="400K", features={"kelvin": 400.0})],
            edges=[],
            target_tokens=[],
        ),
        GraphExample(
            id="cool",
            task="multimodal",
            nodes=[Node(id="temperature", type="temperature", value="300K", features={"kelvin": 300.0})],
            edges=[],
            target_tokens=[],
        ),
    ]
    terminal = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    bonuses, metrics = _temperature_diversity_bonuses(examples, terminal, weight=0.1)
    assert bonuses[0] > 0.0
    assert bonuses[1] > 0.0
    assert bonuses[2] == 0.0
    assert metrics["high_temperature_unique_terminal_states"] == 2.0
    assert metrics["high_temperature_terminal_hamming"] > 0.0
