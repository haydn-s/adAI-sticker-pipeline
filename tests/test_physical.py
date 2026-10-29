import numpy as np
import pytest

from adai_sticker.demo import Prediction, annotate, center_square, frame_to_tensor, predict
from adai_sticker.labels import resolve_target
from adai_sticker.physical import load_metadata, summarize_by_condition


def test_center_square_crops_the_middle():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:, 80:560] = 255
    sq = center_square(frame)
    assert sq.shape == (480, 480, 3) and sq.min() == 255


def test_annotate_produces_view_plus_panel(tiny_model):
    square = (np.random.default_rng(0).random((300, 300, 3)) * 255).astype(np.uint8)
    x = frame_to_tensor(square, 224, "cpu")
    pred = predict(tiny_model, x)
    target = resolve_target("toaster")
    out = annotate(square, pred, target, 859, "ResNet-50", fps=12.3, cam=np.random.rand(7, 7))
    assert out.shape == (540, 540 + 440, 3) and out.dtype == np.uint8


def test_annotate_marks_fooled_when_target_wins():
    square = np.full((100, 100, 3), 128, dtype=np.uint8)
    pred = Prediction(["toaster", "banana", "a", "b", "c"], [0.9, 0.05, 0.02, 0.02, 0.01],
                      [859, 954, 1, 2, 3])
    fooled = annotate(square, pred, resolve_target("toaster"), 859, "ResNet-50")
    not_fooled = annotate(square, Prediction(pred.names[::-1], pred.probs, pred.indices[::-1]),
                          resolve_target("toaster"), 859, "ResNet-50")
    # The FOOLED banner is solid green in the panel; the not-fooled box is not.
    green = np.array([0, 131, 0])
    assert (np.abs(fooled.astype(int) - green).sum(-1) < 10).sum() > 1000
    assert (np.abs(not_fooled.astype(int) - green).sum(-1) < 10).sum() < 50


def test_metadata_and_condition_summary(tmp_path):
    meta_path = tmp_path / "shoot.csv"
    meta_path.write_text("photo,distance_cm,lighting\na.jpg,50,indoor\nb.jpg,100,indoor\nc.jpg,50,outdoor\n")
    meta = load_metadata(meta_path)
    assert meta["a.jpg"] == {"distance_cm": "50", "lighting": "indoor"}
    rows = [{"photo": k, **v, "target_top1": k != "b.jpg"} for k, v in meta.items()]
    lines = summarize_by_condition(rows, ["distance_cm", "lighting"])
    text = "\n".join(lines)
    assert "by distance_cm" in text and "100%  (n=2)" in text and "0%  (n=1)" in text


def test_metadata_requires_photo_column(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("file,distance\nx.jpg,1\n")
    with pytest.raises(ValueError):
        load_metadata(bad)


def test_eval_photos_end_to_end(tmp_path, monkeypatch, tiny_model):
    from PIL import Image

    from adai_sticker import cli, models

    photos = tmp_path / "photos"
    (photos / "annotated").mkdir(parents=True)
    for name in ("p1.jpg", "p2.jpg"):
        Image.fromarray((np.random.default_rng(1).random((64, 80, 3)) * 255).astype(np.uint8)).save(photos / name)
    Image.new("RGB", (10, 10)).save(photos / "annotated" / "view.jpg")  # must be skipped
    (tmp_path / "meta.csv").write_text("photo,distance_cm\np1.jpg,50\np2.jpg,100\n")
    monkeypatch.setattr(models, "load_model", lambda key, device=None: tiny_model)
    code = cli.main(["eval-photos", "--photos", str(photos), "--target", "toaster",
                     "--meta", str(tmp_path / "meta.csv"), "--device", "cpu"])
    assert code == 0
    lines = (photos / "results_resnet50.csv").read_text().strip().splitlines()
    assert len(lines) == 3 and lines[0].startswith("photo,distance_cm")
