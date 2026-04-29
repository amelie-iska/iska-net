import torch

from iska_reasoner.data.synthetic import iter_synthetic_examples
from iska_reasoner.tools import verify_example_tokens
from iska_reasoner.topology import TOPOLOGY_FEATURE_NAMES, summarize_graph, topology_feature_tensor
from iska_reasoner.tropical import TropicalSchedule, logit_diagnostics


def test_topology_summary_features_are_finite():
    examples = list(iter_synthetic_examples(3))
    summary = summarize_graph(examples[0])
    assert summary.node_count > 0
    features = topology_feature_tensor(examples)
    assert features.shape == (3, len(TOPOLOGY_FEATURE_NAMES))
    assert torch.isfinite(features).all()


def test_tropical_diagnostics_and_schedule():
    logits = torch.tensor([[[3.0, 1.0, 0.0], [0.0, 2.0, 1.0]]])
    labels = torch.tensor([[0, -100]])
    metrics = logit_diagnostics(logits, labels, temperature=0.5)
    assert metrics["tropical/top1_margin"] > 0
    schedule = TropicalSchedule(temperature=1.0, temperature_min=0.25, anneal_steps=10)
    assert schedule.value(0) == 1.0
    assert schedule.value(10) == 0.25


def test_verifier_rewards_exact_target_tokens():
    example = next(iter_synthetic_examples(1))
    result = verify_example_tokens(example, example.target_tokens)
    assert result.passed
    assert result.exact_token_match
    assert result.reward > 1.0

