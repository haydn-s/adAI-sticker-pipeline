import math

import pytest
import torch

from adai_sticker.masks import REGION_PRESETS, SHAPES, region_mask, shape_mask
from adai_sticker.palette import PALETTES, parse_palette
from adai_sticker.sticker import Sticker, StickerDesign


@pytest.mark.parametrize("shape", SHAPES)
def test_shapes_are_soft_masks_with_sensible_area(shape):
    m = shape_mask(shape, 64)
    assert m.shape == (64, 64)
    assert 0 <= float(m.min()) and float(m.max()) <= 1
    assert 0.25 < float(m.mean()) <= 1.0
    if shape != "square":
        # The outline should not touch the image corners.
        assert float(m[0, 0]) == 0 and float(m[-1, -1]) == 0


def test_circle_area_matches_geometry():
    area = float(shape_mask("circle", 256).mean())
    assert area == pytest.approx(math.pi / 4 * 0.98**2, abs=0.01)


@pytest.mark.parametrize("preset", REGION_PRESETS)
def test_region_presets_grow_with_amount(preset):
    small = float(region_mask(preset, 64, 0.15).mean())
    large = float(region_mask(preset, 64, 0.5).mean())
    assert 0 < small < large <= 1


def test_none_mode_renders_inside_unit_range():
    s = Sticker(StickerDesign(resolution=48, shape="star"))
    rgb, alpha = s.render()
    assert rgb.shape == (1, 3, 48, 48) and alpha.shape == (1, 1, 48, 48)
    assert 0 <= float(rgb.min()) and float(rgb.max()) <= 1


def test_palette_hard_render_uses_only_palette_colors():
    s = Sticker(StickerDesign(resolution=32, palette="primary"))
    rgb, _ = s.render()
    palette = parse_palette("primary")
    pixels = rgb[0].reshape(3, -1).T
    dist = torch.cdist(pixels, palette).min(1).values
    assert float(dist.max()) < 1e-4
    assert len(palette) == len(PALETTES["primary"])


def test_palette_straight_through_passes_gradients():
    s = Sticker(StickerDesign(resolution=32, palette="sunset"))
    s.hard = True
    rgb, _ = s()
    rgb.sum().backward()
    assert s.param.grad is not None and float(s.param.grad.abs().sum()) > 0


def test_region_mode_preserves_cover_outside_region(cover_path):
    s = Sticker(StickerDesign(resolution=64, cover_mode="region", cover_image=cover_path,
                              region="border", region_amount=0.25))
    rgb, _ = s.render()
    keep = s.region[0, 0] == 0
    assert keep.any()
    assert torch.allclose(rgb[0][:, keep], s.cover[0][:, keep])


def test_blend_mode_respects_epsilon(cover_path):
    eps = 0.05
    s = Sticker(StickerDesign(resolution=64, cover_mode="blend", cover_image=cover_path, epsilon=eps))
    with torch.no_grad():
        s.param.fill_(10.0)  # saturate tanh -> maximal perturbation
    rgb, _ = s.render()
    assert float((rgb - s.cover).abs().max()) <= eps + 1e-6
    stats = s.perturbation_stats()
    assert stats["max_abs_change"] <= eps + 1e-6
    assert 0 < stats["ssim"] < 1


def test_ssim_of_identical_images_is_one(cover_path):
    from adai_sticker.sticker import load_image, ssim

    img = load_image(cover_path, 48)[None]
    assert ssim(img, img) == pytest.approx(1.0, abs=1e-5)
    assert ssim(img, torch.rand_like(img)) < 0.5


def test_auto_palette_requires_cover():
    with pytest.raises(ValueError):
        Sticker(StickerDesign(palette="auto"))


def test_blend_rejects_palette(cover_path):
    with pytest.raises(ValueError):
        Sticker(StickerDesign(cover_mode="blend", cover_image=cover_path, palette="primary"))


def test_high_resolution_render_and_state_roundtrip(cover_path):
    s = Sticker(StickerDesign(resolution=32, cover_mode="region", cover_image=cover_path,
                              palette="auto", palette_size=4), seed=3)
    rgb_hi, alpha_hi = s.render(128)
    assert rgb_hi.shape == (1, 3, 128, 128) and alpha_hi.shape == (1, 1, 128, 128)
    clone = Sticker.from_state(s.state())
    assert torch.allclose(clone.render()[0], s.render()[0])
