"""Experiment: do stickers transfer to models they never saw, and do ensembles help?

Optimizes one sticker per "source" configuration (a single model or an ensemble)
and evaluates every sticker on every model. Writes ``results.csv``,
``results.json`` and ``fig_transfer_matrix.png`` (sources x evaluated models) to
``--out``.

    uv run python experiments/transfer_matrix.py --target toaster
    uv run python experiments/transfer_matrix.py --sources resnet50 resnet50+convnext_tiny
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from adai_sticker import report as rpt
from adai_sticker.evaluate import EvalSettings
from adai_sticker.models import available_models
from adai_sticker.optimize import AttackConfig
from adai_sticker.pipeline import StickerRequest, make_sticker
from adai_sticker.sticker import StickerDesign

DEFAULT_SOURCES = [
    "resnet50",
    "convnext_tiny",
    "vit_b_16",
    "resnet50+efficientnet_b0+convnext_tiny",
]


def run(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    eval_models = args.eval_models or available_models()
    matrix: dict[str, dict[str, float]] = {}
    rows = []
    for source in args.sources:
        models = source.split("+")
        request = StickerRequest(
            target=args.target, models=models, design=StickerDesign(shape=args.shape),
            attack=AttackConfig(steps=args.steps, seed=args.seed),
            eval_models=[m for m in eval_models if m not in models],
            eval=EvalSettings(n_images=args.eval_images), full_battery=False,
        )
        run_dir = out / source.replace("+", "_")
        saved = run_dir / "report.json"
        if saved.exists() and not args.fresh:  # finished in an earlier, interrupted run
            print(f"\n=== source: {source} (reusing {saved}) ===", flush=True)
            summary = json.loads(saved.read_text())
        else:
            print(f"\n=== source: {source} ===", flush=True)
            summary = make_sticker(request, run_dir, say=lambda m: print("  " + m, flush=True))
        matrix[source] = {k: v["headline"]["success_rate"] for k, v in summary["evaluation"].items()}
        history = json.loads((run_dir / "history.json").read_text())
        for key, ev in summary["evaluation"].items():
            noise = ev["controls"]["random-noise sticker"]
            rows.append({"source": source, "eval_model": key, "seen": key in models,
                         "success_rate": ev["headline"]["success_rate"],
                         "disruption_rate": ev["headline"]["disruption_rate"],
                         "clean_success": ev["controls"]["no sticker"]["success_rate"],
                         "noise_success": noise["success_rate"],
                         "noise_disruption": noise["disruption_rate"],
                         "train_peak_target_prob": max(h["target_prob"] for h in history)})

    with (out / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "results.json").write_text(json.dumps({"args": vars(args), "matrix": matrix}, indent=2))
    plot(matrix, eval_models, out / "fig_transfer_matrix.png", args.target)


def plot(matrix: dict[str, dict[str, float]], eval_models: list[str], path: Path, target: str) -> None:
    rpt._style()
    sources = list(matrix)
    data = np.array([[matrix[s].get(m, np.nan) for m in eval_models] for s in sources])
    fig, ax = plt.subplots(figsize=(1.2 * len(eval_models) + 3.2, 0.6 * len(sources) + 1.8), dpi=150)
    im = ax.imshow(data, cmap=rpt.SEQUENTIAL, vmin=0, vmax=1, aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(len(eval_models)), [rpt.model_name(m) for m in eval_models], rotation=30, ha="right")
    ax.set_yticks(range(len(sources)),
                  [" + ".join(rpt.model_name(k) for k in s.split("+")) for s in sources])
    for i, s in enumerate(sources):
        for j, m in enumerate(eval_models):
            v = data[i, j]
            if np.isnan(v):
                continue
            seen = m in s.split("+")
            ax.text(j, i, f"{v:.0%}" + ("*" if seen else ""), ha="center", va="center", fontsize=8,
                    color="#ffffff" if v > 0.55 else rpt.TEXT)
    ax.set_xlabel("Evaluated on   (* = optimized against this model)")
    ax.set_ylabel("Optimized against")
    ax.set_title(f'Transfer of "{target}" stickers', loc="left")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.outline.set_visible(False)
    cbar.ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    cbar.ax.tick_params(labelsize=8, colors=rpt.TEXT_2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", default="toaster")
    p.add_argument("--sources", nargs="+", default=DEFAULT_SOURCES,
                   help="model keys; join with + for an ensemble")
    p.add_argument("--eval-models", nargs="+", choices=available_models())
    p.add_argument("--shape", default="circle")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--eval-images", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/exp-transfer")
    p.add_argument("--fresh", action="store_true",
                   help="remake every sticker instead of reusing finished ones in --out")
    return p


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
