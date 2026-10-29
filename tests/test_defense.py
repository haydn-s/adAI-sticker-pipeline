import pytest
import torch

from adai_sticker.data import BackgroundSet
from adai_sticker.defense import (
    BlurDefense,
    JPEGDefense,
    SaliencyMaskDefense,
    build_defense,
)
from adai_sticker.evaluate import EvalSettings, Evaluator
from adai_sticker.labels import resolve_target
from adai_sticker.sticker import Sticker, StickerDesign


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
