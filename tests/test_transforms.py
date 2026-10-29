import math

import pytest
import torch

from adai_sticker.transforms import (
    EOTConfig,
    Placement,
    gaussian_blur,
    neutral_photometric,
    place,
    render_scenes,
    sample_photometric,
    sample_placement,
)


def _placement(n, scale=0.5, angle=0.0, center=(0.0, 0.0), tilt=(0.0, 0.0)):
    return Placement(
        torch.full((n,), scale), torch.full((n,), angle),
        torch.tensor([center] * n, dtype=torch.float32), torch.tensor([tilt] * n, dtype=torch.float32),
    )


def test_centered_square_sticker_covers_expected_area():
    bg = torch.zeros(2, 3, 100, 100)
    rgb = torch.ones(1, 3, 20, 20)
    alpha = torch.ones(1, 1, 20, 20)
    scenes, scene_alpha = place(bg, rgb, alpha, _placement(2, scale=0.5))
    assert scene_alpha.mean().item() == pytest.approx(0.25, abs=0.02)
    assert scenes[0, :, 50, 50].min().item() == pytest.approx(1.0, abs=1e-4)
    assert scenes[0, :, 2, 2].max().item() == pytest.approx(0.0, abs=1e-6)


def test_rotation_preserves_area():
    bg = torch.zeros(1, 3, 128, 128)
    rgb, alpha = torch.ones(1, 3, 32, 32), torch.ones(1, 1, 32, 32)
    _, a0 = place(bg, rgb, alpha, _placement(1, scale=0.4))
    _, a45 = place(bg, rgb, alpha, _placement(1, scale=0.4, angle=math.pi / 4))
    assert a45.mean().item() == pytest.approx(a0.mean().item(), rel=0.05)


def test_offset_moves_sticker():
    bg = torch.zeros(1, 3, 100, 100)
    rgb, alpha = torch.ones(1, 3, 10, 10), torch.ones(1, 1, 10, 10)
    _, a = place(bg, rgb, alpha, _placement(1, scale=0.2, center=(0.5, -0.5)))
    ys, xs = torch.nonzero(a[0, 0] > 0.5, as_tuple=True)
    assert xs.float().mean().item() == pytest.approx(75, abs=2)
    assert ys.float().mean().item() == pytest.approx(25, abs=2)


def test_render_is_differentiable_wrt_sticker():
    cfg = EOTConfig()
    gen = torch.Generator().manual_seed(0)
    rgb = torch.rand(1, 3, 64, 64, requires_grad=True)
    alpha = torch.ones(1, 1, 64, 64)
    bg = torch.rand(4, 3, 96, 96)
    scenes, _ = render_scenes(bg, rgb, alpha, sample_placement(4, cfg, gen),
                              sample_photometric(4, cfg, gen), cfg.gamut)
    scenes.sum().backward()
    assert rgb.grad is not None and rgb.grad.abs().sum() > 0
    assert scenes.min() >= 0 and scenes.max() <= 1


def test_neutral_photometric_is_identity_apart_from_gamut():
    bg = torch.rand(2, 3, 64, 64)
    rgb, alpha = torch.rand(1, 3, 32, 32), torch.zeros(1, 1, 32, 32)
    scenes, _ = render_scenes(bg, rgb, alpha, _placement(2), neutral_photometric(2))
    assert torch.allclose(scenes, bg, atol=1e-5)


def test_blur_keeps_shape_and_mean():
    x = torch.rand(2, 3, 40, 40)
    y = gaussian_blur(x, 1.5)
    assert y.shape == x.shape
    assert y.mean().item() == pytest.approx(x.mean().item(), abs=0.01)
