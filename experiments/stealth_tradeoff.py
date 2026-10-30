"""Experiment: how much attack strength does a sticker give up as it looks more natural?

Builds one cover-photo sticker per stealth setting:

* region mode — editable stripes / border covering more or less of the sticker (``--amounts``)
* blend mode  — per-pixel budget epsilon in 1/255 units (``--epsilons``)

plus a fully adversarial reference sticker, and evaluates each on held-out photos.
Writes ``results.csv``, ``results.json`` and ``fig_tradeoff.png`` to ``--out``, and
reuses stickers already finished there (``--fresh`` to redo).

    uv run python experiments/stealth_tradeoff.py --target toaster --cover dog.jpg
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from adai_sticker import report as rpt
from adai_sticker.data import find_imagenette
from adai_sticker.evaluate import EvalSettings
from adai_sticker.optimize import AttackConfig
from adai_sticker.pipeline import StickerRequest, make_sticker
from adai_sticker.sticker import StickerDesign


def default_cover() -> str:
    """First English springer photo from Imagenette val (a dog, as in the project pitch)."""
    root = find_imagenette()
    if root is None:
        sys.exit("No --cover given and Imagenette not found; run `adai-sticker fetch-data`.")
    return str(sorted((root / "val" / "n02102040").iterdir())[0])


def run(args: argparse.Namespace) -> list[dict]:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cover = args.cover or default_cover()
    settings: list[tuple[str, str | None, float | None]] = [("none", None, None)]
    settings += [("region", r, a) for r in args.regions for a in args.amounts]
    settings += [("blend", None, e) for e in args.epsilons]

    rows = []
    for mode, region, value in settings:
        if mode == "none":
            design = StickerDesign(shape=args.shape)
            name = "full"
        elif mode == "region":
            design = StickerDesign(shape=args.shape, cover_mode="region", cover_image=cover,
                                   region=region, region_amount=value,
                                   palette="auto" if args.auto_palette else None)
            name = f"region-{region}-{value:g}"
        else:
            design = StickerDesign(shape=args.shape, cover_mode="blend", cover_image=cover,
                                   epsilon=value / 255)
            name = f"blend-{value:g}"
        request = StickerRequest(
            target=args.target, models=[args.model], design=design,
            attack=AttackConfig(steps=args.steps, seed=args.seed),
            eval=EvalSettings(n_images=args.eval_images), full_battery=False,
        )
        saved = out / name / "report.json"
        if saved.exists() and not args.fresh:  # finished in an earlier, interrupted run
            print(f"\n=== {name} (reusing {saved}) ===", flush=True)
            summary = json.loads(saved.read_text())
        else:
            print(f"\n=== {name} ===", flush=True)
            summary = make_sticker(request, out / name, say=lambda m: print("  " + m, flush=True))
        head = summary["evaluation"][args.model]["headline"]
        controls = summary["evaluation"][args.model]["controls"]
        stats = summary["stealth"]
        history = json.loads((out / name / "history.json").read_text())
        rows.append({
            "name": name,
            "mode": mode if mode != "region" else f"region-{region}",
            "setting": value if value is not None else "",
            "success_rate": head["success_rate"],
            "top5_rate": head["top5_rate"],
            "disruption_rate": head["disruption_rate"],
            "changed_fraction": stats.get("changed_fraction", 1.0),
            "mean_abs_change": stats.get("mean_abs_change", ""),
            "ssim": stats.get("ssim", ""),
            "cover_only_success": controls.get("cover photo only", {}).get("success_rate", ""),
            "cover_only_disruption": controls.get("cover photo only", {}).get("disruption_rate", ""),
            "noise_success": controls["random-noise sticker"]["success_rate"],
            "noise_disruption": controls["random-noise sticker"]["disruption_rate"],
            # Near zero means the optimization stalled rather than found a weak sticker.
            "train_peak_target_prob": max(h["target_prob"] for h in history),
        })
        print(f"  success {head['success_rate']:.0%}  ssim {stats.get('ssim', float('nan')):.3f}")

    with (out / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "results.json").write_text(json.dumps({"args": vars(args), "cover": cover, "rows": rows},
                                                 indent=2))
    plot(rows, out / "fig_tradeoff.png", args)
    return rows


def plot(rows: list[dict], path: Path, args: argparse.Namespace) -> None:
    rpt._style()
    fig, ax = plt.subplots(figsize=(6.8, 4.2), dpi=150)
    series = [(m, f"Region: {m.split('-', 1)[1]} (amount)") for m in dict.fromkeys(
        r["mode"] for r in rows if r["mode"].startswith("region-"))]
    series.append(("blend", "Blend (ε, /255)"))
    for i, (mode, label) in enumerate(series):
        key = "setting"
        pts = sorted((r for r in rows if r["mode"] == mode), key=lambda r: -r["ssim"])
        if not pts:
            continue
        xs = [r["ssim"] for r in pts]
        ys = [r["success_rate"] for r in pts]
        ax.plot(xs, ys, color=rpt.SERIES[i], lw=2, marker="o", ms=6, mec=rpt.SURFACE, mew=1.5,
                label=label)
        for r in (pts[0], pts[-1]):
            ax.annotate(f"{r[key]:g}", (r["ssim"], r["success_rate"]), textcoords="offset points",
                        xytext=(6, 4), fontsize=8, color=rpt.TEXT_2)
    full = next((r for r in rows if r["mode"] == "none"), None)
    if full:
        ax.axhline(full["success_rate"], color=rpt.TEXT_2, lw=1)
        ax.text(0.01, full["success_rate"] + 0.015, f"fully adversarial sticker: {full['success_rate']:.0%}",
                transform=ax.get_yaxis_transform(), fontsize=8, color=rpt.TEXT_2)
    ax.set_xlabel("Similarity to the original photo (SSIM; 1 = identical)")
    ax.set_ylabel("Attack success rate")
    ax.set_ylim(0, 1.05)
    rpt._pct_axis(ax)
    ax.set_title(f'Stealth vs. strength: "{args.target}" vs. {rpt.model_name(args.model)}', loc="left")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", default="toaster")
    p.add_argument("--model", default="resnet50")
    p.add_argument("--cover", help="cover photo (default: an Imagenette dog photo)")
    p.add_argument("--shape", default="circle")
    p.add_argument("--regions", nargs="+", default=["stripes", "border"])
    p.add_argument("--amounts", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.45, 0.6])
    p.add_argument("--epsilons", type=float, nargs="+", default=[4, 8, 16, 32, 64])
    p.add_argument("--auto-palette", action="store_true", help="match region colors to the cover")
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--eval-images", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/exp-stealth")
    p.add_argument("--fresh", action="store_true",
                   help="remake every sticker instead of reusing finished ones in --out")
    return p


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
