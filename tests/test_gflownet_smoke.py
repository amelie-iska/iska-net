import torch

from iska_reasoner.gflownet.trajectory import GraphSetPolicy, TrajectoryBalanceLoss, sample_trajectories


def test_trajectory_balance_smoke():
    target_mask = torch.tensor([[1, 1, 0, 0], [1, 0, 1, 0]], dtype=torch.float32)
    policy = GraphSetPolicy(num_actions=4, hidden_dim=16)
    loss_fn = TrajectoryBalanceLoss()
    traj = sample_trajectories(policy, target_mask, max_steps=3, epsilon=0.1)
    loss = loss_fn(traj)
    assert loss.isfinite()
    loss.backward()
    assert loss_fn.log_z.grad is not None

