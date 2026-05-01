import torch

from iska_reasoner.topology import (
    folding_attention_coordinate_consistency_loss,
    folding_contact_field,
    folding_contact_metrics,
    hidden_js_geometry_loss,
    hidden_state_topology_metrics,
    hidden_topology_collapse_loss,
    uma_contact_alignment_loss,
)


def test_hidden_state_topology_metrics_are_bounded_and_finite():
    hidden = torch.randn(2, 12, 8)
    mask = torch.ones(2, 12, dtype=torch.bool)
    metrics = hidden_state_topology_metrics(hidden, mask, max_points=6, bins=4)
    assert metrics["hidden_topology/point_count_mean"] == 6.0
    assert metrics["hidden_topology/h0_total_persistence_mean"] >= 0.0
    assert metrics["hidden_topology/distogram_entropy_mean"] >= 0.0
    assert metrics["hidden_topology/js_distance_mean_mean"] >= 0.0
    assert metrics["hidden_topology/js_distogram_entropy_mean"] >= 0.0
    assert -1.0 <= metrics["hidden_topology/geometry_js_correlation_mean"] <= 1.0


def test_hidden_topology_collapse_loss_backpropagates():
    hidden = torch.zeros(1, 4, 3, requires_grad=True)
    mask = torch.ones(1, 4, dtype=torch.bool)
    loss = hidden_topology_collapse_loss(hidden, mask, margin=0.5, max_points=4)
    assert loss.item() > 0.0
    loss.backward()
    assert hidden.grad is not None


def test_hidden_js_geometry_loss_backpropagates():
    hidden = torch.zeros(1, 4, 3, requires_grad=True)
    mask = torch.ones(1, 4, dtype=torch.bool)
    loss = hidden_js_geometry_loss(hidden, mask, margin=0.05, max_points=4)
    assert loss.item() > 0.0
    loss.backward()
    assert hidden.grad is not None


def test_folding_contact_field_fuses_attention_embedding_and_js_geometry():
    hidden = torch.tensor(
        [
            [
                [0.0, 0.0, 5.0],
                [0.0, 0.1, 4.8],
                [5.0, 0.0, 0.0],
                [4.8, 0.1, 0.0],
            ]
        ]
    )
    attention = torch.zeros(1, 2, 1, 4, 4)
    attention[:, :, :, 0, 1] = 0.9
    attention[:, :, :, 1, 0] = 0.9
    attention[:, :, :, 2, 3] = 0.8
    attention[:, :, :, 3, 2] = 0.8
    mask = torch.ones(1, 4, dtype=torch.bool)

    contact = folding_contact_field(attention_maps=attention, hidden_states=hidden, token_mask=mask)
    assert contact.shape == (1, 4, 4)
    assert torch.allclose(torch.diagonal(contact[0]), torch.zeros(4))
    assert contact[0, 0, 1] > contact[0, 0, 2]
    assert contact[0, 2, 3] > contact[0, 1, 2]

    metrics = folding_contact_metrics(contact, mask)
    assert metrics["folding_contact/top_contact_mean"] > metrics["folding_contact/mean"]
    assert metrics["folding_contact/density_05"] > 0.0

    folded_coords = torch.tensor([[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [10.0, 0.0, 0.0], [11.5, 0.0, 0.0]]])
    scrambled_coords = torch.tensor([[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [1.5, 0.0, 0.0], [11.5, 0.0, 0.0]]])
    folded_loss = folding_attention_coordinate_consistency_loss(contact, folded_coords, mask, contact_radius=3.0)
    scrambled_loss = folding_attention_coordinate_consistency_loss(contact, scrambled_coords, mask, contact_radius=3.0)
    assert folded_loss < scrambled_loss


def test_uma_contact_alignment_loss_uses_oracle_feedback_records():
    class Example:
        target_tokens = [
            "ATTN_BIN:sequence_to_motion:b48",
            "TOKEN_COUPLING:uma:sequence_oracle:b48",
            "UMA_INFLUENCE:uma:trajectory_physics:b48",
            "SEQ_STRUCT_DYN_PROXY:uma_scored",
        ]

    contact = torch.full((1, 4, 4), 0.2, requires_grad=True)
    contact = contact - torch.diag_embed(torch.diagonal(contact, dim1=-2, dim2=-1))
    hidden = torch.randn(1, 4, 8, requires_grad=True)
    mask = torch.ones(1, 4, dtype=torch.bool)

    loss, metrics = uma_contact_alignment_loss(contact, hidden, [Example()], mask)
    assert loss.item() > 0.0
    assert metrics["uma_contact/alignment_examples"] == 1.0
    assert metrics["uma_contact/target_strength"] > 0.5
    loss.backward()
    assert hidden.grad is not None
