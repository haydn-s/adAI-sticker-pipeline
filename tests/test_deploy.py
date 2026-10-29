"""Deployment files: Streamlit Community Cloud installs deploy/requirements.txt, not pyproject."""

import os
import re
import time
from pathlib import Path

import pytest

from adai_sticker.web import common
from adai_sticker.web.jobs import JobManager

ROOT = Path(__file__).parents[1]
tomllib = pytest.importorskip("tomllib")


def _name(requirement: str) -> str:
    return re.split(r"[<>=!~\[; ]", requirement.strip(), maxsplit=1)[0].lower().replace("_", "-")


def test_deploy_requirements_match_pyproject():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    needed = project["dependencies"] + project["optional-dependencies"]["streamlit"]
    lines = (ROOT / "deploy" / "requirements.txt").read_text().splitlines()
    listed = {_name(line): line.strip() for line in lines if line.strip() and line[0] not in "#-"}
    for req in needed:
        assert listed.get(_name(req)) == req, f"deploy/requirements.txt should list {req!r}"
    assert set(listed) == {_name(r) for r in needed}, "deploy/requirements.txt has extras"
    assert "--extra-index-url https://download.pytorch.org/whl/cpu" in lines


def test_streamlit_config_is_at_repo_root():
    config = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text())
    assert config["server"]["maxUploadSize"] == 10
    assert config["theme"]["light"]["primaryColor"] == "#2a78d6"


def test_presets_follow_the_device():
    assert common.presets_for("cpu") is common.CPU_PRESETS
    assert common.presets_for("cuda") is common.PRESETS
    assert set(common.CPU_PRESETS) == set(common.PRESETS)
    for preset in common.CPU_PRESETS.values():
        request = common.build_request(common.Choices(target="toaster"), preset)
        assert request.attack.batch_size < 24 and request.eval.n_images < 150


def test_prune_removes_old_finished_runs(tmp_path):
    jobs = JobManager(tmp_path, device="cpu", make_fn=lambda *a, **k: {})
    old_id, old_dir = jobs.new_job_dir()
    new_id, _ = jobs.new_job_dir()
    (old_dir / "upload.jpg").write_bytes(b"photo")
    stale = time.time() - 48 * 3600
    os.utime(old_dir, (stale, stale))
    assert jobs.prune(max_age_hours=24) == 1
    assert not old_dir.exists() and (tmp_path / new_id).exists()
