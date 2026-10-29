"""Model helpers that don't need downloaded weights."""

import pytest
import torch
from torch import nn

from adai_sticker.models import PatchEmbedding


@pytest.mark.parametrize("bias,size", [(True, 32), (False, 32), (True, 30)])
def test_patch_embedding_matches_the_convolution(bias, size):
    torch.manual_seed(0)
    conv = nn.Conv2d(3, 8, 4, stride=4, bias=bias)
    fast = PatchEmbedding(conv)
    x = torch.rand(2, 3, size, size, requires_grad=True)
    expected = conv(x)
    (grad_expected,) = torch.autograd.grad(expected.square().sum(), x)
    out = fast(x)
    (grad,) = torch.autograd.grad(out.square().sum(), x)
    assert out.shape == expected.shape
    assert torch.allclose(out, expected, atol=1e-5)
    assert torch.allclose(grad, grad_expected, atol=1e-5)
    assert set(fast.state_dict()) == set(conv.state_dict())


def test_only_patch_convolutions_are_replaced():
    assert PatchEmbedding.replaces(nn.Conv2d(3, 8, 16, stride=16))
    assert not PatchEmbedding.replaces(nn.Conv2d(3, 8, 7, stride=2, padding=3))
    assert not PatchEmbedding.replaces(nn.Linear(3, 8))
