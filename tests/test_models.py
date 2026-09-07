import pytest
import torch

from surgvu.models import build_model


def test_model_emits_one_logit_per_class():
    model = build_model(num_outputs=12, pretrained=False)
    out = model(torch.zeros(2, 3, 64, 64))
    assert out.shape == (2, 12)


def test_model_emits_raw_logits_not_probabilities():
    """The loss functions apply their own sigmoid/softmax. A model that
    already squashed its output would be trained through it twice, which
    flattens gradients and looks like slow convergence rather than a bug."""
    torch.manual_seed(0)
    model = build_model(num_outputs=12, pretrained=False)
    out = model(torch.randn(8, 3, 64, 64))
    assert out.min() < 0.0 or out.max() > 1.0


def test_task_head_width_is_independent_of_tool_head_width():
    assert build_model(num_outputs=8, pretrained=False)(
        torch.zeros(1, 3, 64, 64)).shape == (1, 8)


def test_unknown_backbone_fails_loudly():
    with pytest.raises(ValueError, match="unknown backbone"):
        build_model(num_outputs=12, backbone="not_a_real_net", pretrained=False)
