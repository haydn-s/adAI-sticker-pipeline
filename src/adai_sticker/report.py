"""Figures and the plain-language report that accompany every sticker.

Chart styling follows one small system so every figure reads the same: a light
neutral surface, text in two ink tones, hairline recessive grids, 2px lines,
categorical colors assigned in a fixed order, and a single-hue blue ramp for
magnitudes (the rotation x scale heatmap).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from .models import MODEL_INFO  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQUENTIAL = LinearSegmentedColormap.from_list("blue_ramp", BLUE_RAMP)
FIG_WIDTH = 9.0  # inches; one width for every figure so text scales the same on the report card

SWEEP_TITLES = {
    "brightness": ("Lighting", "brightness ×"),
    "contrast": ("Contrast", "contrast ×"),
    "blur": ("Blur", "blur σ (px)"),
    "noise": ("Sensor noise", "noise σ"),
    "perspective": ("Viewing angle", "perspective tilt"),
    "jpeg": ("JPEG compression", "quality"),
    "color_cast": ("Color cast", "cast strength"),
}


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_2,
        "axes.titlecolor": TEXT,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "xtick.color": TEXT_2,
        "ytick.color": TEXT_2,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "font.size": 9,
    })


def _pct_axis(ax, axis: str = "y") -> None:
    fmt = matplotlib.ticker.PercentFormatter(1.0, decimals=0)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def model_name(key: str) -> str:
    return MODEL_INFO[key].display_name if key in MODEL_INFO else key


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def fig_grid(grid: dict, path: Path, title: str) -> Path:
    _style()
    data = np.array(grid["success_rate"])
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 3.8), dpi=150)
    im = ax.imshow(data, cmap=SEQUENTIAL, vmin=0, vmax=1, aspect="auto", origin="lower")
    ax.grid(False)
    ax.set_xticks(range(len(grid["angles"])), [f"{a:g}°" for a in grid["angles"]])
    ax.set_yticks(range(len(grid["scales"])), [f"{s:.0%}" for s in grid["scales"]])
    ax.set_xlabel("Sticker rotation")
    ax.set_ylabel("Sticker width (% of photo)")
    ax.set_title(title, loc="left")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.outline.set_visible(False)
    cbar.ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    cbar.ax.tick_params(labelsize=8, colors=TEXT_2)
    cbar.set_label("Attack success rate", color=TEXT_2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_sweeps(sweeps: dict, path: Path, title: str) -> Path:
    _style()
    factors = list(sweeps)
    cols = 4
    rows = (len(factors) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(FIG_WIDTH, 2.3 * rows), dpi=150, sharey=True)
    axes = np.atleast_1d(axes).flatten()
    for ax, factor in zip(axes, factors, strict=False):
        sw = sweeps[factor]
        xs = sw["values"]
        ys = [r["success_rate"] for r in sw["results"]]
        positions = list(range(len(xs)))
        ax.plot(positions, ys, color=SERIES[0], lw=2, solid_capstyle="round", solid_joinstyle="round",
                marker="o", ms=5, mec=SURFACE, mew=1.5)
        ax.set_xticks(positions, [f"{x:g}" for x in xs])
        name, xlabel = SWEEP_TITLES.get(factor, (factor, factor))
        ax.set_title(name, loc="left", fontsize=10)
        ax.set_xlabel(xlabel)
        ax.set_ylim(0, 1.02)
        _pct_axis(ax)
    for ax in axes[len(factors):]:
        ax.axis("off")
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold", color=TEXT)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _hbar(ax, labels: list[str], values: list[float], colors: list[str]) -> None:
    y = np.arange(len(labels))
    ax.barh(y, values, height=0.5, color=colors)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.12)
    _pct_axis(ax, "x")
    ax.grid(axis="y", visible=False)
    for yi, v in zip(y, values, strict=True):
        ax.text(v + 0.015, yi, f"{v:.0%}", va="center", fontsize=8, color=TEXT)


def fig_transfer(results: dict[str, float], trained_on: list[str], path: Path, title: str) -> Path:
    _style()
    keys = sorted(results, key=lambda k: (k not in trained_on, -results[k]))
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 0.4 * len(keys) + 1.3), dpi=150)
    colors = [SERIES[0] if k in trained_on else SERIES[1] for k in keys]
    _hbar(ax, [model_name(k) for k in keys], [results[k] for k in keys], colors)
    ax.set_xlabel("Attack success rate (held-out photos, random conditions)")
    ax.set_title(title, loc="left")
    handles = [
        matplotlib.patches.Patch(color=SERIES[0], label="Optimized against"),
        matplotlib.patches.Patch(color=SERIES[1], label="Never seen (transfer)"),
    ]
    ax.legend(handles=handles, loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_controls(controls: dict[str, dict], path: Path, title: str) -> Path:
    _style()
    labels = list(controls)
    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH, 0.42 * len(labels) + 1.4), dpi=150, sharey=True)
    _hbar(axes[0], labels, [controls[k]["success_rate"] for k in labels], [SERIES[0]] * len(labels))
    axes[0].set_title("Photo labeled as the target", loc="left", fontsize=10)
    _hbar(axes[1], labels, [controls[k]["disruption_rate"] for k in labels], [SERIES[0]] * len(labels))
    axes[1].set_title("Correct label lost", loc="left", fontsize=10)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold", color=TEXT)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_training(history: list[dict], path: Path) -> Path:
    _style()
    steps = [h["step"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH, 2.6), dpi=150)
    axes[0].plot(steps, [h["success_rate"] for h in history], color=SERIES[0], lw=2)
    axes[0].set_title("Success on training batches", loc="left", fontsize=10)
    axes[0].set_ylim(0, 1.02)
    _pct_axis(axes[0])
    axes[1].plot(steps, [h["attack_loss"] for h in history], color=SERIES[0], lw=2)
    axes[1].set_title("Attack loss (lower is stronger)", loc="left", fontsize=10)
    for ax in axes:
        ax.set_xlabel("Optimization step")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_gradcam(clean, clean_cams, stickered, sticker_cams, captions, path: Path, method: str) -> Path:
    """One row per example: photo and explanation without, then with, the sticker."""
    _style()
    n = len(clean)
    fig, axes = plt.subplots(n, 4, figsize=(FIG_WIDTH, 2.45 * n + 0.5), dpi=150, squeeze=False)
    headers = ("Without sticker", f"{method}: its answer", "With sticker", f"{method}: target")
    for i in range(n):
        panels = (
            (clean[i], None, captions[i][0]),
            (clean[i], clean_cams[i], ""),
            (stickered[i], None, captions[i][1]),
            (stickered[i], sticker_cams[i], ""),
        )
        for j, (img, cam, cap) in enumerate(panels):
            ax = axes[i, j]
            ax.imshow(img.permute(1, 2, 0).cpu().numpy())
            if cam is not None:
                ax.imshow(cam.cpu().numpy(), cmap="magma", alpha=0.55, vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if i == 0:
                ax.set_title(headers[j], fontsize=10, loc="left", color=TEXT, fontweight="bold")
            if cap:
                ax.set_xlabel(cap, fontsize=8, color=TEXT_2, loc="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def report_card(parts: list[Path], header: str, subheader: str, path: Path, width: int = 1500) -> Path:
    """Stack figures under a header into one shareable image."""
    imgs = []
    for p in parts:
        if p is None or not Path(p).exists():
            continue
        im = Image.open(p).convert("RGB")
        imgs.append(im.resize((width - 60, int(im.height * (width - 60) / im.width)), Image.Resampling.LANCZOS))
    head_h = 150
    total_h = head_h + sum(im.height + 30 for im in imgs) + 30
    card = Image.new("RGB", (width, total_h), SURFACE)
    draw = ImageDraw.Draw(card)
    draw.text((30, 30), header, fill=TEXT, font=ImageFont.load_default(size=44))
    draw.text((30, 92), subheader, fill=TEXT_2, font=ImageFont.load_default(size=24))
    y = head_h
    for im in imgs:
        card.paste(im, (30, y))
        y += im.height + 30
    card.save(path)
    return path


# ---------------------------------------------------------------------------
# Plain-language report
# ---------------------------------------------------------------------------


def _pct(x: float) -> str:
    return f"{x:.0%}"


def describe_grid(grid: dict) -> list[str]:
    data = np.array(grid["success_rate"])
    scales = grid["scales"]
    by_scale = data.mean(1)
    lines = []
    reliable = [s for s, v in zip(scales, by_scale, strict=True) if v >= 0.5]
    if reliable:
        lines.append(
            f"The sticker succeeds on at least half of photos once it is {min(reliable):.0%} of the "
            f"photo's width or larger (averaged over every rotation)."
        )
    else:
        lines.append("At no tested size does the sticker succeed on half of photos on average.")
    by_angle = data.mean(0)
    worst = int(np.argmin(by_angle))
    best = int(np.argmax(by_angle))
    lines.append(
        f"Rotation matters less than size: averaged over sizes, success ranges from "
        f"{_pct(by_angle[worst])} (at {grid['angles'][worst]:g}°) to {_pct(by_angle[best])} "
        f"(at {grid['angles'][best]:g}°)."
    )
    return lines


def markdown_report(summary: dict) -> str:
    t = summary["target"]["label"]
    primary = summary["models"][0]
    prim_eval = summary["evaluation"][primary]
    head = prim_eval["headline"]
    controls = prim_eval["controls"]
    design = summary["design"]
    L: list[str] = []
    L.append(f'# Sticker report: "{t}" vs. {model_name(primary)}')
    L.append("")
    L.append(
        f"**Bottom line:** across {head['n']} held-out photos with random size, rotation, position, "
        f"lighting, blur and print color drift, {model_name(primary)} labeled the scene "
        f'**"{t}" {_pct(head["success_rate"])}** of the time with this sticker, versus '
        f'{_pct(controls["no sticker"]["success_rate"])} with no sticker and '
        f'{_pct(controls["random-noise sticker"]["success_rate"])} with a random-noise sticker of the '
        "same shape."
    )
    L.append("")
    L.append(
        f"Even when the target label wasn't reached, the sticker made the model lose a correct answer "
        f"{_pct(head['disruption_rate'])} of the time."
    )
    L.append("")
    L.append("![Report card](report_card.png)")
    L.append("")
    L.append("## What was made")
    L.append("")
    L.append(f"- **Target label:** {t}")
    L.append(f"- **Optimized against:** {', '.join(model_name(m) for m in summary['models'])}")
    L.append(f"- **Shape:** {design['shape']}")
    mode = design["cover_mode"]
    if mode == "none":
        L.append("- **Cover mode:** none (the whole sticker is adversarial)")
    elif mode == "region":
        L.append(f"- **Cover mode:** region — the cover photo is kept except in the '{design['region']}' "
                 f"region ({design['region_amount']:.0%} amount)")
    else:
        L.append(f"- **Cover mode:** blend — every pixel of the cover photo may move by at most "
                 f"{design['epsilon']:.2f} ({design['epsilon'] * 255:.0f}/255)")
    L.append(f"- **Colors:** {design['palette'] if design['palette'] else 'unconstrained RGB'}")
    if "stealth" in summary:
        st = summary["stealth"]
        L.append(f"- **Changed from cover photo:** {st.get('changed_fraction', 1):.0%} of pixels, "
                 f"mean change {st.get('mean_abs_change', 0) * 255:.1f}/255")
    L.append(f"- **Optimization:** {summary['attack']['steps']} steps in {summary['seconds']:.0f} s")
    L.append("")
    L.append("## Controls")
    L.append("")
    L.append("| Condition | Labeled as target | Top-5 contains target | Correct label lost |")
    L.append("|---|---|---|---|")
    for name, m in controls.items():
        L.append(f"| {name} | {_pct(m['success_rate'])} | {_pct(m['top5_rate'])} | {_pct(m['disruption_rate'])} |")
    L.append("")
    if "grid" in prim_eval:
        L.append("## Rotation and size")
        L.append("")
        L.extend(f"- {line}" for line in describe_grid(prim_eval["grid"]))
        L.append("")
        L.append("![Rotation x size](fig_grid.png)")
        L.append("")
    if "sweeps" in prim_eval:
        L.append("## Imaging conditions")
        L.append("")
        L.append("Each row varies one factor from mild to severe (sticker at "
                 f"{summary['eval_settings']['default_scale']:.0%} of photo width, random rotation).")
        L.append("")
        L.append("| Factor | Values | Success rate |")
        L.append("|---|---|---|")
        for factor, sw in prim_eval["sweeps"].items():
            vals = ", ".join(f"{v:g}" for v in sw["values"])
            rates = ", ".join(_pct(r["success_rate"]) for r in sw["results"])
            L.append(f"| {SWEEP_TITLES.get(factor, (factor,))[0]} | {vals} | {rates} |")
        L.append("")
        L.append("![Condition sweeps](fig_sweeps.png)")
        L.append("")
    if len(summary["evaluation"]) > 1:
        L.append("## Transfer to other models")
        L.append("")
        L.append("| Model | Optimized against? | Labeled as target | Correct label lost |")
        L.append("|---|---|---|---|")
        for key, ev in summary["evaluation"].items():
            trained = "yes" if key in summary["models"] else "no"
            L.append(f"| {model_name(key)} | {trained} | {_pct(ev['headline']['success_rate'])} | "
                     f"{_pct(ev['headline']['disruption_rate'])} |")
        if summary.get("skipped_models"):
            L.append("")
            L.append("Not evaluated (free-text targets only exist for CLIP): "
                     + ", ".join(model_name(k) for k in summary["skipped_models"]))
        L.append("")
        L.append("![Transfer](fig_transfer.png)")
        L.append("")
    if "explanation" in summary:
        ex = summary["explanation"]
        L.append("## Where the model looks")
        L.append("")
        L.append(
            f"The sticker covers {_pct(ex['sticker_area'])} of each photo but receives "
            f"**{_pct(ex['share_on_sticker'])}** of the model's attention ({ex['method']}) for the "
            f"target class — {ex['ratio']:.1f}x more than its size alone would predict."
        )
        L.append("")
        L.append("![Grad-CAM](fig_gradcam.png)")
        L.append("")
    L.append("## Limitations")
    L.append("")
    L.append("- These numbers come from *simulated* photos. Printing and photographing the sticker is the "
             "real test; use the print sheet and `adai-sticker eval-photos` to measure it.")
    L.append("- The simulator's printer model is a generic approximation, not a calibrated profile of a "
             "specific printer.")
    L.append("- Success against one model does not imply success against others; see the transfer table.")
    L.append("- Results depend on the background photos; Imagenette covers only 10 everyday categories.")
    L.append("")
    L.append("## Responsible use")
    L.append("")
    L.append("This sticker was made to teach how fragile image classifiers can be. Labels that could "
             "endanger or demean people are blocked by the pipeline's safety policy. Please don't use "
             "adversarial stickers to deceive safety-critical systems or to harm, harass, or misidentify "
             "anyone.")
    L.append("")
    return "\n".join(L)
