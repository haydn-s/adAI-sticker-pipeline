"""Web API tests: a tiny stand-in model and synthetic photos, so no downloads are needed."""

import io
import json
import time

import pytest
from PIL import Image

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from adai_sticker import pipeline  # noqa: E402
from adai_sticker.web.server import create_app  # noqa: E402

from .conftest import TinyClassifier  # noqa: E402

TINY_PRESETS = {
    "quick": {
        "steps": 3, "eval_images": 6, "grid_images": 3, "train_images": 8, "label": "Tiny",
        "attack": {"batch_size": 4, "scene_size": 64},
        "eval": {"batch_size": 6, "grid_angles": (0, 90), "grid_scales": (0.3,),
                 "sweeps": {"blur": (0.0, 1.0)}},
    }
}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "load_model", lambda key, device=None: TinyClassifier(key).eval())
    app = create_app(runs_root=tmp_path / "runs", device="cpu", backgrounds="synthetic",
                     presets=TINY_PRESETS)
    return TestClient(app)


def _settings(**kw):
    base = {"model": "resnet50", "target": "toaster", "ethics_ack": True, "test_other_models": False}
    return {"settings": json.dumps({**base, **kw})}


def _png_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (80, 60), (120, 90, 60)).save(buf, format="PNG")
    return buf.getvalue()


def test_options_and_static_page(client):
    opts = client.get("/api/options").json()
    assert opts["allow_generate"] is True
    assert {m["key"] for m in opts["models"]} >= {"resnet50", "clip-vit-b-32"}
    assert any(b["category"] == "dehumanizing labels" for b in opts["blocked_categories"])
    page = client.get("/")
    assert page.status_code == 200 and "Make a sticker that fools an AI" in page.text


def test_check_target_policy(client):
    ok = client.post("/api/check-target", json={"target": "toaster"}).json()
    assert ok["allowed"] and ok["label"] == "toaster"
    blocked = client.post("/api/check-target", json={"target": "revolver"}).json()
    assert not blocked["allowed"] and blocked["category"] == "weapons and military equipment"
    free = client.post("/api/check-target", json={"target": "a bowl of ramen", "model": "clip-vit-b-32"}).json()
    assert free["allowed"] and free["free_text"]
    not_clip = client.post("/api/check-target", json={"target": "a bowl of ramen", "model": "resnet50"}).json()
    assert not not_clip["allowed"]


def test_label_search_and_shape_png(client):
    res = client.get("/api/labels?q=guitar").json()["results"]
    assert {r["label"] for r in res} == {"acoustic guitar", "electric guitar"}
    assert client.get("/api/shape/star.png").headers["content-type"] == "image/png"


def test_design_preview_with_cover(client):
    res = client.post("/api/design-preview", data={"shape": "heart", "cover_mode": "region"},
                      files={"cover": ("c.png", _png_bytes(), "image/png")})
    assert res.status_code == 200 and res.headers["content-type"] == "image/png"


@pytest.mark.parametrize("settings,error", [
    ({"ethics_ack": False}, "responsible-use"),
    ({"target": "assault rifle"}, "blocked"),
    ({"cover_mode": "region"}, "Upload a photo"),
    ({"shape": "triangle"}, "Unknown shape"),
])
def test_job_validation(client, settings, error):
    res = client.post("/api/jobs", data=_settings(**settings))
    assert res.status_code == 400
    msg = res.json()["error"]
    assert error.lower() in msg.lower() or "weapon" in msg.lower()


def test_job_runs_to_results(client):
    res = client.post("/api/jobs", data=_settings(cover_mode="region", palette="auto"),
                      files={"cover": ("c.png", _png_bytes(), "image/png")})
    assert res.status_code == 200, res.text
    job_id = res.json()["id"]
    deadline = time.time() + 120
    status = {}
    while time.time() < deadline:
        status = client.get(f"/api/jobs/{job_id}").json()
        if status["status"] in ("done", "failed"):
            break
        time.sleep(0.2)
    assert status["status"] == "done", status
    assert status["success_curve"], "progress should be recorded"
    report = client.get(f"/runs/{job_id}/report.json").json()
    assert report["target"]["label"] == "toaster" and report["design"]["cover_mode"] == "region"
    assert client.get(f"/api/jobs/{job_id}/preview.png").status_code == 200
    assert client.get(f"/runs/{job_id}/print_sheet.pdf").status_code == 200
    items = client.get("/api/gallery").json()["items"]
    assert items[0]["id"] == job_id


def test_gallery_only_mode_refuses_jobs(tmp_path):
    client = TestClient(create_app(runs_root=tmp_path / "runs", allow_generate=False))
    assert client.get("/api/options").json()["allow_generate"] is False
    assert client.post("/api/jobs", data=_settings()).status_code == 403
