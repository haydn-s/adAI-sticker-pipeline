"""Streamlit app tests, run headless with Streamlit's AppTest.

A tiny stand-in model and synthetic photos replace the real ones, so nothing is downloaded.
"""

import shutil
import time
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from adai_sticker import pipeline  # noqa: E402
from adai_sticker.web import common, streamlit_ui  # noqa: E402

from .conftest import TinyClassifier  # noqa: E402

APP = str(Path(__file__).parents[1] / "deploy" / "streamlit_app.py")
EXAMPLE = Path(__file__).parents[1] / "docs" / "examples" / "toaster-resnet50"
TINY_PRESETS = {
    "quick": {
        "steps": 3, "eval_images": 6, "grid_images": 3, "train_images": 8, "label": "Tiny",
        "attack": {"batch_size": 4, "scene_size": 64},
        "eval": {"batch_size": 6, "grid_angles": (0, 90), "grid_scales": (0.3,),
                 "sweeps": {"blur": (0.0, 1.0)}},
    }
}


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Point the app at temporary folders, a tiny model and synthetic photos."""
    runs, gallery = tmp_path / "runs", tmp_path / "gallery"
    shutil.copytree(EXAMPLE, gallery / EXAMPLE.name)
    monkeypatch.setenv("ADAI_DEVICE", "cpu")
    monkeypatch.setenv("ADAI_BACKGROUNDS", "synthetic")
    monkeypatch.setenv("ADAI_RUNS_DIR", str(runs))
    monkeypatch.setenv("ADAI_GALLERY_DIR", str(gallery))
    monkeypatch.delenv("ADAI_GALLERY_ONLY", raising=False)
    monkeypatch.delenv("ADAI_SHOW_ALL_RUNS", raising=False)
    monkeypatch.setattr(pipeline, "load_model", lambda key, device=None: TinyClassifier(key).eval())
    monkeypatch.setattr(common, "CPU_PRESETS", TINY_PRESETS)
    streamlit_ui._manager.clear()
    yield {"runs": runs, "gallery": gallery}
    streamlit_ui._manager.clear()


def _app(**state) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=30)
    for key, value in state.items():
        at.session_state[key] = value
    return at.run()


def _button(at: AppTest, label: str):
    return next(b for b in at.button if b.label == label)


def _text(at: AppTest) -> str:
    parts = [el.value for kind in ("title", "header", "subheader", "markdown", "caption")
             for el in getattr(at, kind)]
    return "\n".join(str(p) for p in parts)


def test_welcome_requires_agreement(env):
    at = _app()
    assert not at.exception
    assert at.title[0].value == "Make a sticker that fools an AI"
    assert _button(at, "Start").disabled
    at.checkbox(key="ack").check().run()
    _button(at, "Start").click().run()
    assert at.header[0].value == "Which AI do you want to fool?"


def test_later_steps_need_earlier_ones(env):
    at = _app(_goto=("design", {}))
    assert "This step needs the earlier ones first." in [i.value for i in at.info]


def test_blocked_target_explains_why(env):
    at = _app(ack=True, model="resnet50", _goto=("target", {}))
    at.selectbox(key="_label_search").select(413).run()  # assault rifle
    assert not at.exception
    assert "blocked" in at.error[0].value and "weapons" in at.error[0].value
    assert _button(at, "Next: design it").disabled
    at.selectbox(key="_label_search").select(859).run()  # toaster
    assert "toaster" in at.success[0].value
    assert not _button(at, "Next: design it").disabled


def test_free_text_only_for_clip(env):
    at = _app(ack=True, model="resnet50", _goto=("target", {}))
    assert not at.text_input
    at = _app(ack=True, model="clip-vit-b-32", _goto=("target", {}))
    assert at.text_input[0].label.startswith("Or, because CLIP reads text")


def test_make_sticker_end_to_end(env):
    target = {"allowed": True, "label": "toaster", "free_text": False}
    at = _app(ack=True, model="resnet50", target=target, test_others=False,
              _goto=("make", {}))
    _button(at, "Make my sticker").click().run()
    assert not at.exception and at.session_state["job_id"]
    deadline = time.time() + 120
    while time.time() < deadline and at.session_state["job_id"] is not None:
        time.sleep(0.3)
        at.run()
    assert not at.exception
    assert at.session_state["job_id"] is None, "the sticker should have finished"
    run_dir = Path(at.session_state["run_dir"])
    assert (run_dir / "report.json").exists() and (run_dir / "print_sheet.pdf").exists()
    assert run_dir.name in at.session_state["my_runs"]
    assert "sticker vs. ResNet-50" in _text(at)
    assert {d.label for d in at.get("download_button")} >= {"Print sheet (PDF)", "Full report"}


def test_results_link_opens_a_finished_run(env):
    at = AppTest.from_file(APP, default_timeout=30)
    at.query_params["run"] = EXAMPLE.name
    at.session_state["_goto"] = ("results", {"run": EXAMPLE.name})
    at.run()
    assert not at.exception
    assert "67%" in _text(at)
    at = AppTest.from_file(APP, default_timeout=30)
    at.query_params["run"] = "../../etc"
    at.session_state["_goto"] = ("results", {"run": "../../etc"})
    at.run()
    assert "to see its results here" in [i.value for i in at.info][0]


def test_gallery_shows_only_your_own_stickers(env):
    for run_id in ("mine", "someone-else"):
        shutil.copytree(EXAMPLE, env["runs"] / run_id)
    at = _app(my_runs=["mine"], _goto=("gallery", {}))
    keys = {b.key for b in at.button}
    assert {"open_mine", f"open_{EXAMPLE.name}"} <= keys
    assert "open_someone-else" not in keys


def test_gallery_only_mode(env, monkeypatch):
    monkeypatch.setenv("ADAI_GALLERY_ONLY", "1")
    streamlit_ui._manager.clear()
    at = _app()
    assert _button(at, "Browse the gallery")
    target = {"allowed": True, "label": "toaster", "free_text": False}
    at = _app(ack=True, model="resnet50", target=target, _goto=("make", {}))
    assert "gallery mode" in at.info[0].value
    assert "Make my sticker" not in {b.label for b in at.button}
