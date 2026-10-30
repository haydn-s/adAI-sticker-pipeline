import json

import pytest
import torch

from adai_sticker import cli, models, pipeline
from adai_sticker.data import BackgroundSet
from adai_sticker.defense import (
    SALIENCY_ATTACKS,
    BlurDefense,
    JPEGDefense,
    SaliencyMaskDefense,
    StraightThroughSaliency,
    _center_batch,
    attack_saliency,
    build_defense,
)
from adai_sticker.evaluate import EvalSettings, Evaluator
from adai_sticker.explain import differentiable_gradcam, gradcam, share_on_sticker
from adai_sticker.labels import resolve_target
from adai_sticker.optimize import AttackConfig
from adai_sticker.sticker import Sticker, StickerDesign
from adai_sticker.transforms import neutral_photometric, render_scenes, sample_placement

from .conftest import TinyClassifier


def test_build_defense_parses_specs(tiny_model):
    assert isinstance(build_defense("blur:2", tiny_model), BlurDefense)
    assert build_defense("blur:2", tiny_model).sigma == 2.0
    assert build_defense("jpeg", tiny_model).quality == 50
    assert isinstance(build_defense("saliency:0.2", tiny_model), SaliencyMaskDefense)
    with pytest.raises(ValueError):
        build_defense("magic", tiny_model)


def test_defended_models_keep_interface(tiny_model):
    x = torch.rand(2, 3, 64, 64)
    for d in (BlurDefense(tiny_model), JPEGDefense(tiny_model, 30)):
        assert d(x).shape == (2, 1000)
        assert d.key.startswith("resnet50+")
        assert d.class_names == tiny_model.class_names


def test_blur_defense_is_differentiable(tiny_model):
    x = torch.rand(2, 3, 64, 64, requires_grad=True)
    BlurDefense(tiny_model)(x).sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0


def test_saliency_defense_calibration_controls_false_positives(tiny_model):
    clean = torch.rand(40, 3, 64, 64)
    d = SaliencyMaskDefense(tiny_model, area=0.1, false_positive_rate=0.25)
    with pytest.raises(RuntimeError):
        d(clean[:2])
    d.calibrate(clean)
    d(clean)
    assert d.last_flagged.float().mean().item() == pytest.approx(0.25, abs=0.05)


def test_evaluator_reports_flags_and_clean_accuracy(tiny_model):
    bgs = BackgroundSet.synthetic(10, 72)
    bgs.labels[:5] = 0  # pretend half the photos are tench
    ev = Evaluator(bgs, resolve_target("toaster"), EvalSettings(n_images=10, batch_size=5))
    d = SaliencyMaskDefense(tiny_model)
    d.calibrate(bgs.center(bgs.images, 224).float() / 255)
    rgb, alpha = Sticker(StickerDesign(resolution=32)).render()
    placement, photometric = ev._random_conditions()
    m = ev.measure(d, rgb, alpha, placement, photometric)
    assert m.flagged_rate is not None and 0 <= m.flagged_rate <= 1
    assert 0 <= ev.clean_accuracy(tiny_model) <= 1
    assert "flagged_rate" not in ev.measure(tiny_model, rgb, alpha, placement, photometric).to_dict()


def test_differentiable_gradcam_matches_gradcam(tiny_model):
    x = torch.rand(3, 3, 64, 64, requires_grad=True)
    cams, logits = differentiable_gradcam(tiny_model, x, 859)
    assert torch.allclose(cams, gradcam(tiny_model, x, 859)[0], atol=1e-6)
    assert logits.shape == (3, 1000)
    cams.sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0


def test_straight_through_mask_matches_defense_and_passes_gradient(tiny_model):
    d = SaliencyMaskDefense(tiny_model)
    d.threshold = 0.0  # flag every image, so the hotspot is always grayed out
    x = torch.rand(2, 3, 64, 64)
    region = d.mask(d._explain(x))[:, 0] > 0
    assert region.any()

    # The real defense hands back no gradient where it grays out the image ...
    naive = x.clone().requires_grad_(True)
    d(naive)[:, 859].sum().backward()
    assert naive.grad.abs().amax(1)[region].max() == 0

    # ... the straight-through version computes the same logits but does.
    st = StraightThroughSaliency(d)
    adaptive = x.clone().requires_grad_(True)
    out = st(adaptive)
    with torch.no_grad():
        assert torch.allclose(out, d(x), atol=1e-5)
    out[:, 859].sum().backward()
    assert adaptive.grad.abs().amax(1)[region].max() > 0


def test_attention_attack_moves_gradcam_off_sticker(tiny_model):
    target = resolve_target("toaster")
    bgs = BackgroundSet.synthetic(16, 64)
    cfg = AttackConfig(steps=40, batch_size=8, scene_size=64, lr=0.3)
    gen = torch.Generator().manual_seed(3)
    bg, _ = bgs.sample(8, 64, "cpu", gen)
    placement = sample_placement(8, cfg.eot, gen)

    def share(sticker: Sticker) -> float:
        with torch.no_grad():
            rgb, alpha = sticker()
        scenes, scene_alpha = render_scenes(bg, rgb, alpha, placement, neutral_photometric(8))
        return float(share_on_sticker(gradcam(tiny_model, scenes, 859)[0], scene_alpha).mean())

    d = SaliencyMaskDefense(tiny_model)
    d.calibrate(_center_batch(bgs, 64, "cpu"))
    sticker = Sticker(StickerDesign(resolution=32))
    before = share(sticker)
    attack_saliency(sticker, d, target, bgs, cfg, "attention", attention_weight=5.0)
    assert share(sticker) < 0.5 * before
    with pytest.raises(ValueError):
        attack_saliency(sticker, d, target, bgs, cfg, "magic")


def test_defend_cli_reports_adaptive_attacks(tmp_path, monkeypatch):
    tiny = lambda key, device=None: TinyClassifier(key).eval()  # noqa: E731
    monkeypatch.setattr(pipeline, "load_model", tiny)
    monkeypatch.setattr(models, "load_model", tiny)
    request = pipeline.StickerRequest(
        target="toaster",
        design=StickerDesign(resolution=32),
        attack=AttackConfig(steps=2, batch_size=4, scene_size=64),
        eval=EvalSettings(n_images=8, batch_size=8, scene_size=64),
        backgrounds="synthetic",
        train_images=8,
        full_battery=False,
    )
    run = tmp_path / "run"
    pipeline.make_sticker(request, run, device="cpu", say=lambda _m: None)
    argv = ["defend", str(run), "--defense", "blur", "--defense", "saliency", "--adaptive-steps", "2",
            "--adaptive-attack", "bpda", "--adaptive-attack", "attention", "--device", "cpu"]
    assert cli.main(argv) == 0

    result = json.loads((run / "defense.json").read_text())
    attacks = [r.get("attack") for r in result["rows"]]
    assert attacks == [None, None, "re-optimized through blur", None,
                       SALIENCY_ATTACKS["bpda"], SALIENCY_ATTACKS["attention"]]
    for row in result["rows"]:
        assert 0 <= row["attention_on_sticker"] <= 1
        if row["attacker"] == "adaptive attacker":
            assert 0 <= row["success_without_defense"] <= 1
    assert 0 < result["sticker_area"] < 1 and result["saliency_threshold"] > 0
    md = (run / "defense.md").read_text()
    assert "## Adaptive attacks on the saliency detector" in md
    assert (run / "fig_defense.png").exists()
