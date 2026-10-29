"""Integration tests with a tiny stand-in model and synthetic backgrounds (fast, offline)."""

import json

import pytest
import torch

from adai_sticker import pipeline
from adai_sticker.data import BackgroundSet
from adai_sticker.evaluate import EvalSettings, Evaluator
from adai_sticker.explain import attention_share, gradcam
from adai_sticker.labels import BlockedTargetError, resolve_target
from adai_sticker.losses import target_loss
from adai_sticker.optimize import AttackConfig, optimize_sticker
from adai_sticker.sticker import Sticker, StickerDesign
from adai_sticker.transforms import neutral_photometric, render_scenes, sample_placement

from .conftest import TinyClassifier

SMALL_EVAL = EvalSettings(
    n_images=8, grid_images=4, batch_size=8, grid_angles=(-90, 0, 90), grid_scales=(0.3, 0.5),
    sweeps={"brightness": (0.6, 1.0), "jpeg": (100, 40)},
)


def test_optimizer_lowers_loss_on_tiny_model(tiny_model):
    target = resolve_target("toaster")
    backgrounds = BackgroundSet.synthetic(16, 64)
    cfg = AttackConfig(steps=60, batch_size=8, scene_size=64, lr=0.5)
    gen = torch.Generator().manual_seed(5)
    bg, _ = backgrounds.sample(8, 64, "cpu", gen)
    placement = sample_placement(8, cfg.eot, gen)

    def fixed_loss(sticker: Sticker) -> float:
        with torch.no_grad():
            rgb, alpha = sticker()
            scenes, _ = render_scenes(bg, rgb, alpha, placement, neutral_photometric(8))
            return float(target_loss(tiny_model(scenes), target.imagenet_index))

    sticker = Sticker(StickerDesign(resolution=32))
    before = fixed_loss(sticker)
    optimize_sticker(sticker, [tiny_model], target, backgrounds, cfg)
    assert fixed_loss(sticker) < before


def test_evaluator_battery_shapes(tiny_model):
    sticker = Sticker(StickerDesign(resolution=32))
    ev = Evaluator(BackgroundSet.synthetic(12, 72), resolve_target("toaster"), SMALL_EVAL)
    out = ev.full(tiny_model, sticker)
    assert set(out["controls"]) == {"no sticker", "random-noise sticker", "adversarial sticker"}
    assert len(out["grid"]["success_rate"]) == 2 and len(out["grid"]["success_rate"][0]) == 3
    assert set(out["sweeps"]) == {"brightness", "jpeg"}
    for m in out["controls"].values():
        assert 0 <= m["success_rate"] <= 1 and m["n"] == 8


def test_gradcam_and_attention_share(tiny_model):
    imgs = torch.rand(3, 3, 64, 64)
    cams, method = gradcam(tiny_model, imgs, 859)
    assert method == "Grad-CAM" and cams.shape == (3, 64, 64)
    alpha = torch.zeros(3, 1, 64, 64)
    alpha[..., :32, :] = 1
    share = attention_share(cams, alpha, method)
    assert 0 <= share.share_on_sticker <= 1
    assert share.sticker_area == pytest.approx(0.5)


def test_make_sticker_end_to_end(tmp_path, monkeypatch, cover_path):
    monkeypatch.setattr(pipeline, "load_model", lambda key, device=None: TinyClassifier(key).eval())
    request = pipeline.StickerRequest(
        target="banana",
        models=["resnet50"],
        eval_models=["mobilenet_v3_large"],
        design=StickerDesign(resolution=32, cover_mode="region", cover_image=cover_path, palette="auto",
                             palette_size=4),
        attack=AttackConfig(steps=3, batch_size=4, scene_size=64),
        eval=SMALL_EVAL,
        backgrounds="synthetic",
        train_images=8,
    )
    run = tmp_path / "run"
    summary = pipeline.make_sticker(request, run, device="cpu", say=lambda _m: None)
    for name in ("sticker.png", "sticker_900px.png", "print_sheet.pdf", "examples.png", "report.md",
                 "report.json", "report_card.png", "fig_grid.png", "fig_sweeps.png", "fig_transfer.png",
                 "fig_gradcam.png", "assets/cover_image.png"):
        assert (run / name).exists(), name
    assert set(summary["evaluation"]) == {"resnet50", "mobilenet_v3_large"}
    data = json.loads((run / "report.json").read_text())
    assert data["target"]["label"] == "banana"

    # A saved run reloads to the identical sticker.
    sticker, target, req = pipeline.load_run(run)
    assert target.label == "banana" and req.design.cover_mode == "region"
    rgb, _ = sticker.render()
    assert rgb.shape == (1, 3, 32, 32)


def test_make_sticker_refuses_blocked_target(tmp_path):
    request = pipeline.StickerRequest(target="assault rifle", backgrounds="synthetic")
    with pytest.raises(BlockedTargetError):
        pipeline.make_sticker(request, tmp_path / "blocked", device="cpu", say=lambda _m: None)
    assert not (tmp_path / "blocked").exists()
