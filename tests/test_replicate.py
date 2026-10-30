"""The replication driver's statistics: pooled summaries and paired per-run comparisons."""

import csv
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
replicate = pytest.importorskip("replicate")


def _row(target, seed, name, success, ssim="", mode="region-stripes", **extra):
    return {"target": target, "seed": seed, "name": name, "mode": mode, "setting": "",
            "success_rate": str(success), "ssim": str(ssim), **extra}


def test_pool_tags_rows_and_skips_missing_runs(tmp_path):
    run = tmp_path / "toaster" / "seed1"
    run.mkdir(parents=True)
    with (run / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "success_rate"])
        writer.writeheader()
        writer.writerow({"name": "full", "success_rate": "0.5"})
    rows = replicate.pool(tmp_path, ["toaster", "banana"], [0, 1])
    assert rows == [{"target": "toaster", "seed": 1, "name": "full", "success_rate": "0.5"}]


def test_summarize_per_target_and_pooled():
    rows = [_row("a", 0, "full", 0.6), _row("a", 1, "full", 0.8), _row("b", 0, "full", 0.1, ssim=0.5)]
    summary = replicate.summarize(rows, ["name"], ["success_rate", "ssim"], ["a", "b"])
    by_target = {r["target"]: r for r in summary}
    assert [r["target"] for r in summary] == ["a", "b", "all"]
    assert by_target["a"]["n"] == 2
    assert by_target["a"]["success_rate_mean"] == pytest.approx(0.7)
    assert np.isnan(by_target["a"]["ssim_mean"])  # blank cells are skipped, not zeros
    assert by_target["all"]["success_rate_min"] == pytest.approx(0.1)
    assert by_target["all"]["success_rate_max"] == pytest.approx(0.8)
    assert by_target["all"]["ssim_mean"] == pytest.approx(0.5)


def test_paired_matches_target_and_seed():
    rows = [_row("a", 0, "x", 0.5), _row("a", 0, "y", 0.2), _row("a", 1, "x", 0.4),
            _row("b", 0, "x", 0.1), _row("b", 0, "y", 0.3)]
    diff = replicate.paired(rows, {"name": "x"}, {"name": "y"})
    assert sorted(diff.tolist()) == pytest.approx([-0.2, 0.3])  # ("a", 1) has no partner


def test_stripes_interpolated_at_blend_similarity():
    rows = [_row("a", 0, "s1", 0.6, ssim=0.2), _row("a", 0, "s2", 0.2, ssim=0.6),
            _row("a", 0, "blend-32", 0.3, ssim=0.4, mode="blend"),
            _row("a", 1, "s1", 0.6, ssim=0.2), _row("a", 1, "s2", 0.2, ssim=0.6),
            _row("a", 1, "blend-32", 0.3, ssim=0.9, mode="blend")]  # outside the stripes range
    assert replicate.stripes_minus_blend(rows, "32").tolist() == pytest.approx([0.4 - 0.3])


def test_held_out_means_need_every_held_out_model():
    rows = [{"target": "a", "seed": 0, "source": "resnet50", "eval_model": m, "success_rate": v,
             **{k: 0 for k in replicate.TRANSFER_METRICS if k != "success_rate"}}
            for m, v in zip(replicate.HELD_OUT, (0.1, 0.2, 0.6), strict=True)]
    rows.append({**rows[0], "eval_model": "resnet50", "success_rate": 0.9})
    (derived,) = replicate.held_out_means(rows)
    assert derived["success_rate"] == pytest.approx(0.3)
    assert replicate.held_out_means(rows[1:]) == []
