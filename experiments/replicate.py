"""Repeat the stealth and transfer experiments over several targets and seeds.

Runs each experiment script once per (target, seed) into
``--runs/<experiment>/<target>/seed<k>/``, reusing stickers that already finished
there (``--fresh`` to redo), then pools the runs into ``--out/<experiment>/``:

* ``runs.csv``     one row per sticker (per sticker and evaluated model, for transfer)
* ``summary.csv``  mean, min and max of every metric per cell, for each target and
  over all targets (``target = all``)
* ``fig_tradeoff.png`` / ``fig_transfer.png``  summary figures with min-max error bars

It also prints the summary tables and the paired, per-run comparisons quoted in
``docs/RESULTS.md``. To keep GPU time bounded it runs a subset of each experiment's
settings (``STEALTH_ARGS`` and ``TRANSFER_ARGS``).

    uv run python experiments/replicate.py                      # run everything, then summarize
    uv run python experiments/replicate.py --experiments stealth --targets banana --seeds 1
    uv run python experiments/replicate.py --summarize-only     # no GPU: re-pool and redraw
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import stealth_tradeoff
import transfer_matrix
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch

from adai_sticker import report as rpt
from adai_sticker.models import available_models

TARGETS = ["toaster", "banana", "jellyfish"]  # none is an Imagenette class: no test photo is a free win
SEEDS = [0, 1, 2]
# The stripes preset (the strongest region in the single run) at three amounts, three blend budgets,
# and every transfer source except ViT-B/16, which our attack can't fool even with full access.
STEALTH_ARGS = ["--regions", "stripes", "--amounts", "0.2", "0.4", "0.6", "--epsilons", "16", "32", "64"]
ENSEMBLE = "resnet50+efficientnet_b0+convnext_tiny"
TRANSFER_ARGS = ["--sources", "resnet50", "convnext_tiny", ENSEMBLE]
EXPERIMENTS = {"stealth": (stealth_tradeoff, STEALTH_ARGS), "transfer": (transfer_matrix, TRANSFER_ARGS)}
HELD_OUT = ["mobilenet_v3_large", "vit_b_16", "clip-vit-b-32"]  # seen by none of the transfer sources
STALL_PROB = 0.01  # a run whose training target probability never beats this stalled

STEALTH_CELL = ["name", "mode", "setting"]
STEALTH_METRICS = ["success_rate", "top5_rate", "disruption_rate", "ssim", "changed_fraction",
                   "cover_only_success", "cover_only_disruption", "noise_success", "noise_disruption",
                   "train_peak_target_prob"]
TRANSFER_CELL = ["source", "eval_model", "seen"]
TRANSFER_METRICS = ["success_rate", "disruption_rate", "clean_success", "noise_success",
                    "noise_disruption", "train_peak_target_prob"]


def run_experiments(args: argparse.Namespace) -> None:
    for name in args.experiments:
        script, settings = EXPERIMENTS[name]
        for seed in args.seeds:
            for target in args.targets:
                out = Path(args.runs) / name / target / f"seed{seed}"
                cli = [*settings, "--target", target, "--seed", str(seed), "--out", str(out)]
                print(f"\n##### {name}: {target!r}, seed {seed} #####", flush=True)
                script.run(script.build_parser().parse_args(cli + ["--fresh"] * args.fresh))


# ---------------------------------------------------------------------------
# Pooling and statistics
# ---------------------------------------------------------------------------


def _num(x) -> float:
    return float(x) if x not in ("", None) else math.nan


def pool(runs: Path, targets: list[str], seeds: list[int]) -> list[dict]:
    """Every run's ``results.csv`` rows, tagged with target and seed. Missing runs are skipped."""
    rows = []
    for target in targets:
        for seed in seeds:
            path = runs / target / f"seed{seed}" / "results.csv"
            if not path.exists():
                print(f"  (missing {path})")
                continue
            with path.open() as f:
                rows += [{"target": target, "seed": seed, **r} for r in csv.DictReader(f)]
    return rows


def summarize(rows: list[dict], cell: list[str], metrics: list[str], targets: list[str]) -> list[dict]:
    """Mean, min and max of each metric per cell, for each target and over all of them ("all")."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        key = tuple(r[k] for k in cell)
        groups[(r["target"], *key)].append(r)
        groups[("all", *key)].append(r)
    cells = list(dict.fromkeys(tuple(r[k] for k in cell) for r in rows))
    out = []
    for target in [*targets, "all"]:
        for key in cells:
            members = groups.get((target, *key))
            if not members:
                continue
            row = {"target": target, **dict(zip(cell, key, strict=True)), "n": len(members)}
            for m in metrics:
                vals = np.array([_num(r.get(m)) for r in members])
                vals = vals[~np.isnan(vals)]
                for stat, fn in (("mean", np.mean), ("min", np.min), ("max", np.max)):
                    row[f"{m}_{stat}"] = float(fn(vals)) if len(vals) else math.nan
            out.append(row)
    return out


def held_out_means(rows: list[dict]) -> list[dict]:
    """Per run and source, each metric averaged over the models no source was optimized against."""
    by_run: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        if r["eval_model"] in HELD_OUT:
            by_run[(r["target"], r["seed"], r["source"])].append(r)
    return [
        {"target": t, "seed": s, "source": src, "eval_model": "held-out mean", "seen": "False",
         **{m: float(np.mean([_num(c[m]) for c in cells])) for m in TRANSFER_METRICS}}
        for (t, s, src), cells in by_run.items() if len(cells) == len(HELD_OUT)
    ]


def _pick(rows: list[dict], where: dict, metric: str) -> dict[tuple, float]:
    return {(r["target"], r["seed"]): _num(r[metric]) for r in rows
            if all(r[k] == v for k, v in where.items())}


def paired(rows: list[dict], a: dict, b: dict, metric: str = "success_rate",
           metric_b: str | None = None) -> np.ndarray:
    """Per-run difference ``a - b`` between two cells (or two metrics) of the same target and seed."""
    va, vb = _pick(rows, a, metric), _pick(rows, b, metric_b or metric)
    return np.array([va[k] - vb[k] for k in va if k in vb])


def stripes_minus_blend(rows: list[dict], epsilon: str) -> np.ndarray:
    """Per run: stripes success interpolated at the blend sticker's SSIM, minus the blend's success."""
    out = []
    for key, blend in _pick(rows, {"name": f"blend-{epsilon}"}, "ssim").items():
        pts = sorted((_num(r["ssim"]), _num(r["success_rate"])) for r in rows
                     if r["mode"] == "region-stripes" and (r["target"], r["seed"]) == key)
        if len(pts) > 1 and pts[0][0] <= blend <= pts[-1][0]:
            stripes = np.interp(blend, *zip(*pts, strict=True))
            out.append(stripes - _pick(rows, {"name": f"blend-{epsilon}"}, "success_rate")[key])
    return np.array(out)


def _diff(label: str, d: np.ndarray) -> str:
    if not len(d):
        return f"  {label}: no pairs"
    return (f"  {label}: mean {d.mean():+.1%}, range {d.min():+.1%} to {d.max():+.1%}, "
            f"positive in {int((d > 0).sum())}/{len(d)} runs")


def _cell(row: dict, m: str) -> str:
    return f"{row[f'{m}_mean']:.0%} ({row[f'{m}_min']:.0%}–{row[f'{m}_max']:.0%})"


def write_csv(rows: list[dict], path: Path) -> None:
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}
                         for r in rows)


# ---------------------------------------------------------------------------
# Stealth
# ---------------------------------------------------------------------------


def report_stealth(rows: list[dict], targets: list[str], out: Path) -> None:
    summary = summarize(rows, STEALTH_CELL, STEALTH_METRICS, targets)
    write_csv(rows, out / "runs.csv")
    write_csv(summary, out / "summary.csv")
    stalled = [r for r in rows if _num(r.get("train_peak_target_prob")) < STALL_PROB]
    plot_stealth(summary, targets, out / "fig_tradeoff.png", stalled)

    pooled = [r for r in summary if r["target"] == "all"]
    get = {(r["target"], r["name"]): r for r in summary}
    print(f"\nStealth, pooled over {len(targets)} targets (mean, min–max over runs):")
    print("| Sticker | n | SSIM | success | " + " | ".join(targets) + " | top 5 | correct label lost |")
    for r in pooled:
        per = [get[(t, r["name"])] for t in targets if (t, r["name"]) in get]
        ssim = ("—" if math.isnan(r["ssim_mean"])
                else f"{r['ssim_mean']:.2f} ({r['ssim_min']:.2f}–{r['ssim_max']:.2f})")
        print(f"| {r['name']} | {r['n']} | {ssim} | {_cell(r, 'success_rate')} | "
              + " | ".join(_cell(p, "success_rate") for p in per)
              + f" | {_cell(r, 'top5_rate')} | {_cell(r, 'disruption_rate')} |")
    cover = next(r for r in pooled if r["mode"] != "none")
    full = next(r for r in pooled if r["mode"] == "none")
    print(f"  controls: cover only success {_cell(cover, 'cover_only_success')}, "
          f"disruption {_cell(cover, 'cover_only_disruption')}; noise sticker success "
          f"{_cell(full, 'noise_success')}, disruption {_cell(full, 'noise_disruption')}")

    print("\nStealth, paired per run:")
    print(_diff("fully adversarial − stripes 60%", paired(rows, {"name": "full"},
                                                          {"name": "region-stripes-0.6"})))
    for eps in ("32", "64"):
        print(_diff(f"stripes at blend-{eps}'s SSIM − blend {eps}", stripes_minus_blend(rows, eps)))
    print(_diff("blend 64 − blend 32", paired(rows, {"name": "blend-64"}, {"name": "blend-32"})))
    for name in ("region-stripes-0.2", "region-stripes-0.6", "blend-16", "blend-32", "blend-64"):
        print(_diff(f"{name}: top 5 − top 1", paired(rows, {"name": name}, {"name": name},
                                                     "top5_rate", "success_rate")))
        print(_diff(f"{name}: correct label lost − cover photo alone",
                    paired(rows, {"name": name}, {"name": name}, "disruption_rate",
                           "cover_only_disruption")))
    print(f"  stalled (training target probability never above {STALL_PROB:.0%}): " + (", ".join(
        f"{r['target']}/seed{r['seed']}/{r['name']} (peak {_num(r['train_peak_target_prob']):.1%})"
        for r in stalled) or "none"))


def plot_stealth(summary: list[dict], targets: list[str], path: Path, stalled: list[dict]) -> None:
    rpt._style()
    n_stalled = defaultdict(int)
    for r in stalled:
        n_stalled[(r["target"], r["name"])] += 1
    fig, axes = plt.subplots(1, len(targets), figsize=(rpt.FIG_WIDTH, 3.6), dpi=150, sharey=True,
                             squeeze=False)
    series = [("region-stripes", "Stripes (share of sticker)"), ("blend", "Blend (ε, /255)")]
    for ax, target in zip(axes[0], targets, strict=True):
        cells = [r for r in summary if r["target"] == target]
        full = next((r for r in cells if r["mode"] == "none"), None)
        if full:
            ax.axhspan(full["success_rate_min"], full["success_rate_max"], color=rpt.GRID, lw=0)
            ax.axhline(full["success_rate_mean"], color=rpt.TEXT_2, lw=1)
            ax.text(0.02, min(full["success_rate_max"], 0.97) + 0.015,
                    f"fully adversarial: {full['success_rate_mean']:.0%}",
                    transform=ax.get_yaxis_transform(), fontsize=7.5, color=rpt.TEXT_2)
        for i, (mode, label) in enumerate(series):
            pts = sorted((r for r in cells if r["mode"] == mode), key=lambda r: r["ssim_mean"])
            if not pts:
                continue
            x = np.array([r["ssim_mean"] for r in pts])
            y = np.array([r["success_rate_mean"] for r in pts])
            xerr = [x - [r["ssim_min"] for r in pts], [r["ssim_max"] for r in pts] - x]
            yerr = [y - [r["success_rate_min"] for r in pts], [r["success_rate_max"] for r in pts] - y]
            ax.errorbar(x, y, yerr=yerr, xerr=xerr, color=rpt.SERIES[i], lw=2, marker="o", ms=5,
                        mec=rpt.SURFACE, mew=1.5, elinewidth=1, capsize=2, label=label)
            for r in pts:
                v = float(r["setting"])
                k = n_stalled[(target, r["name"])]
                ax.annotate((f"{v:.0%}" if mode != "blend" else f"{v:g}")
                            + (f" ({k}/{r['n']} stalled)" if k else ""),
                            (r["ssim_mean"], r["success_rate_mean"]), textcoords="offset points",
                            xytext=(-5, 5) if k else (5, 4), ha="right" if k else "left",
                            fontsize=7.5, color=rpt.TEXT_2)
        ax.set_title(f'"{target}"', loc="left", fontsize=10)
        ax.set_xlabel("Similarity to the photo (SSIM)")
        ax.set_xlim(0.1, 0.85)
    axes[0][0].set_ylabel("Attack success rate")
    axes[0][0].set_ylim(0, 1.05)
    rpt._pct_axis(axes[0][0])
    axes[0][0].legend(loc="upper right")
    n = max(r["n"] for r in summary if r["target"] != "all")
    fig.suptitle(f"Stealth vs. strength against ResNet-50: mean of {n} seeds, bars span min–max",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", color=rpt.TEXT)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Transfer
# ---------------------------------------------------------------------------


def _source_name(source: str, short: bool = False) -> str:
    if short and "+" in source:
        return f"Ensemble of {len(source.split('+'))}"
    return " + ".join(rpt.model_name(k) for k in source.split("+"))


def report_transfer(rows: list[dict], targets: list[str], out: Path) -> None:
    derived = held_out_means(rows)
    summary = summarize(rows + derived, TRANSFER_CELL, TRANSFER_METRICS, targets)
    write_csv(rows, out / "runs.csv")
    write_csv(summary, out / "summary.csv")
    sources = list(dict.fromkeys(r["source"] for r in rows))
    models = [m for m in available_models() if any(r["eval_model"] == m for r in rows)]
    plot_transfer(summary, derived, sources, models, out / "fig_transfer.png")

    get = {(r["target"], r["source"], r["eval_model"]): r for r in summary}
    print(f"\nTransfer, pooled over {len(targets)} targets (mean, min–max over runs; * = seen):")
    print("| Optimized against | " + " | ".join(models) + " | held-out mean |")
    for s in sources:
        cells = [_cell(get[("all", s, m)], "success_rate") + ("*" if m in s.split("+") else "")
                 for m in [*models, "held-out mean"]]
        print(f"| {_source_name(s)} | " + " | ".join(cells) + " |")
    print("\nHeld-out mean success per target:")
    for s in sources:
        print(f"| {_source_name(s)} | "
              + " | ".join(_cell(get[(t, s, "held-out mean")], "success_rate") for t in targets) + " |")
    print("\nControls (max over runs and sources): clean success "
          + ", ".join(f"{m} {max(_num(r['clean_success']) for r in rows if r['eval_model'] == m):.1%}"
                      for m in models)
          + "\n  noise-sticker success "
          + ", ".join(f"{m} {max(_num(r['noise_success']) for r in rows if r['eval_model'] == m):.1%}"
                      for m in models)
          + "\n  noise-sticker disruption (mean) "
          + ", ".join(f"{m} {np.mean([_num(r['noise_disruption']) for r in rows if r['eval_model'] == m]):.0%}"
                      for m in models))
    print("  sticker disruption (mean) per source: " + "; ".join(
        f"{_source_name(s, True)}: " + ", ".join(
            f"{m} {get[('all', s, m)]['disruption_rate_mean']:.0%}" for m in models) for s in sources))

    for s in sources:
        own = min((r for r in rows if r["source"] == s and r["seen"] == "True"),
                  key=lambda r: _num(r["success_rate"]))
        print(f"  weakest {_source_name(s, True)} run on a model it saw: {own['target']}/seed{own['seed']} "
              f"{own['eval_model']} {_num(own['success_rate']):.0%} (training peak target probability "
              f"{_num(own.get('train_peak_target_prob')):.0%})")

    print("\nTransfer, paired per run:")
    held = {"eval_model": "held-out mean"}
    both = rows + derived
    for a, b in ((ENSEMBLE, "resnet50"), (ENSEMBLE, "convnext_tiny"), ("convnext_tiny", "resnet50")):
        print(_diff(f"held-out mean: {_source_name(a, True)} − {_source_name(b)}",
                    paired(both, {**held, "source": a}, {**held, "source": b})))
    for m in ("resnet50", "convnext_tiny"):
        print(_diff(f"on {m}: ensemble − {m} alone",
                    paired(rows, {"source": ENSEMBLE, "eval_model": m}, {"source": m, "eval_model": m})))
    for s in ("resnet50", "convnext_tiny"):
        unseen = [m for m in models if m not in s.split("+")]
        best = defaultdict(int)
        for key in _pick(rows, {"source": s, "eval_model": unseen[0]}, "success_rate"):
            scores = {m: _pick(rows, {"source": s, "eval_model": m}, "success_rate")[key] for m in unseen}
            best[max(scores, key=scores.get)] += 1
        print(f"  {s} sticker: best-transfer model per run: {dict(best)}")


def plot_transfer(summary: list[dict], derived: list[dict], sources: list[str], models: list[str],
                  path: Path) -> None:
    rpt._style()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(rpt.FIG_WIDTH, 4.0), dpi=150, sharey=True,
                                  gridspec_kw={"width_ratios": [3.2, 1]})
    get = {(r["source"], r["eval_model"]): r for r in summary if r["target"] == "all"}
    width = 0.8 / len(sources)
    for i, source in enumerate(sources):
        xs = np.arange(len(models)) + (i - (len(sources) - 1) / 2) * width
        cells = [get[(source, m)] for m in models]
        mean = np.array([c["success_rate_mean"] for c in cells])
        seen = [m in source.split("+") for m in models]
        color = rpt.SERIES[i]
        ax.bar(xs, mean, width * 0.88, color=[to_rgba(color, 0.3) if s else color for s in seen],
               edgecolor=color, linewidth=1)
        ax.errorbar(xs, mean, yerr=[mean - [c["success_rate_min"] for c in cells],
                                    [c["success_rate_max"] for c in cells] - mean],
                    fmt="none", ecolor=rpt.TEXT_2, elinewidth=1, capsize=2)
        for x, c, s in zip(xs, cells, seen, strict=True):
            if s:
                ax.text(x, c["success_rate_max"] + 0.02, "*", ha="center", fontsize=9, color=rpt.TEXT_2)
    ax.set_xticks(range(len(models)), [rpt.model_name(m) for m in models], rotation=20, ha="right")
    ax.set_ylabel("Attack success rate")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="x", visible=False)
    rpt._pct_axis(ax)
    ax.set_title("Every model (pale bar, * = optimized against it)", loc="left", fontsize=9.5)

    # Right: the same comparison paired by run, on the models no source saw.
    runs = sorted({(r["target"], r["seed"]) for r in derived})
    lookup = {(r["target"], r["seed"], r["source"]): r["success_rate"] for r in derived}
    for run in runs:
        ys = [lookup.get((*run, s), np.nan) for s in sources]
        ax2.plot(range(len(sources)), ys, color=to_rgba(rpt.TEXT_2, 0.3), lw=1, zorder=1)
    for i, source in enumerate(sources):
        ys = [lookup[(*run, source)] for run in runs if (*run, source) in lookup]
        ax2.scatter([i] * len(ys), ys, s=16, color=rpt.SERIES[i], edgecolor=rpt.SURFACE, lw=0.8, zorder=2)
        ax2.plot([i - 0.25, i + 0.25], [np.mean(ys)] * 2, color=rpt.TEXT, lw=2, zorder=3)
    ax2.set_xticks(range(len(sources)), [_source_name(s, True) for s in sources], rotation=20, ha="right")
    ax2.set_xlim(-0.5, len(sources) - 0.5)
    ax2.grid(axis="x", visible=False)
    ax2.set_title("Held-out models,\nper run", loc="left", fontsize=9.5)
    n = max(c["n"] for c in summary if c["target"] == "all")
    fig.suptitle(f"Transfer: mean of {n} runs (targets × seeds), bars span min–max", x=0.01,
                 ha="left", fontsize=11, fontweight="bold", color=rpt.TEXT)
    fig.legend([Patch(color=rpt.SERIES[i]) for i in range(len(sources))],
               [_source_name(s) for s in sources], loc="upper left", bbox_to_anchor=(0.01, 0.94),
               ncol=len(sources), title="Optimized against:", title_fontsize=8, alignment="left")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiments", nargs="+", choices=list(EXPERIMENTS), default=list(EXPERIMENTS))
    p.add_argument("--targets", nargs="+", default=TARGETS)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--runs", default="runs/replicate", help="where each run's sticker folders go")
    p.add_argument("--out", default="docs/results", help="where pooled CSVs and figures go")
    p.add_argument("--fresh", action="store_true", help="remake stickers instead of reusing them")
    p.add_argument("--summarize-only", action="store_true", help="skip optimization; pool and plot")
    args = p.parse_args()
    if not args.summarize_only:
        run_experiments(args)
    reports = {"stealth": report_stealth, "transfer": report_transfer}
    for name in args.experiments:
        rows = pool(Path(args.runs) / name, args.targets, args.seeds)
        if not rows:
            print(f"\nNo finished {name} runs under {args.runs}.")
            continue
        out = Path(args.out) / name
        out.mkdir(parents=True, exist_ok=True)
        reports[name](rows, args.targets, out)


if __name__ == "__main__":
    main()
