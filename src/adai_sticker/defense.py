"""Defenses: can a model be protected from adversarial stickers, and at what cost?

Every defense wraps a victim model and exposes the same :class:`VictimModel`
interface, so the evaluation battery runs on defended models unchanged. Each one
is judged on two numbers: how much it lowers the attack's success, and how much
accuracy it costs on ordinary photos.

* ``blur``     — Gaussian-blur every input. The evaluation sweeps show the
  sticker's power lives in fine detail, so blurring weakens it; it also blurs the
  real objects. Differentiable, so an *adaptive* attacker can optimize through it.
* ``jpeg``     — re-compress every input as JPEG (a classic input-purification
  defense). Not differentiable.
* ``saliency`` — use the model's own explanation as a patch detector. When
  Grad-CAM attention is unusually concentrated in a small area (as it is on a
  sticker), gray out that area and classify again. The concentration threshold is
  calibrated on clean photos so only a set fraction of them get masked. Related
  to Local Gradients Smoothing (Naseer et al., 2019) and Jedi (Tarchoun et al., 2023).

Defenses tested only against the sticker they were built to catch overstate their
protection (Athalye, Carlini & Wagner, 2018; Tramèr et al., 2020), so each one can
also be attacked by an *adaptive* attacker who knows it is there:

* blur is differentiable, so the sticker is simply re-optimized through it;
* the saliency mask is a hard, non-differentiable decision, so it gets three
  attacks (:data:`SALIENCY_ATTACKS`): ``bpda`` optimizes through a soft stand-in
  for the mask (straight-through, in the spirit of BPDA), ``attention`` adds a loss
  that pushes Grad-CAM attention off the sticker while keeping the target label
  (Dombrowski et al., 2019 show explanations can be steered this way), and
  ``combined`` adds the straight-through objective to the attention attack.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import torch
import torch.nn.functional as F

from .data import BackgroundSet
from .evaluate import _jpeg
from .explain import differentiable_gradcam, gradcam, share_on_sticker
from .labels import Target
from .models import VictimModel
from .optimize import AttackConfig, AttackResult, optimize_sticker
from .sticker import Sticker
from .transforms import gaussian_blur

DEFENSES = ("blur", "jpeg", "saliency")
SIZE_SWEEP = (0.25, 0.35, 0.45, 0.55, 0.65)  # sticker width / photo width
SALIENCY_ATTACKS = {
    "bpda": "BPDA (straight-through mask)",
    "attention": "attention-evasion loss",
    "combined": "BPDA + attention loss",
}


class Defended(VictimModel):
    """Base class: a wrapped model with a preprocessing step."""

    name = "defense"

    def __init__(self, base: VictimModel):
        super().__init__()
        self.base = base
        self.input_size = base.input_size

    @property
    def key(self) -> str:  # type: ignore[override]
        return f"{self.base.key}+{self.label}"

    @property
    def label(self) -> str:
        return self.name

    @property
    def class_names(self) -> list[str]:
        return self.base.class_names

    @property
    def gradcam_layer(self):
        return self.base.gradcam_layer

    def class_index(self, target: Target) -> int:
        return self.base.class_index(target)

    def preprocess(self, images: torch.Tensor) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.base(self.preprocess(images))


class BlurDefense(Defended):
    name = "blur"

    def __init__(self, base: VictimModel, sigma: float = 1.5):
        super().__init__(base)
        self.sigma = sigma

    @property
    def label(self) -> str:
        return f"blur σ={self.sigma:g}"

    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        return gaussian_blur(images, self.sigma)


class JPEGDefense(Defended):
    name = "jpeg"

    def __init__(self, base: VictimModel, quality: int = 50):
        super().__init__(base)
        self.quality = quality

    @property
    def label(self) -> str:
        return f"JPEG q={self.quality}"

    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        return _jpeg(images, self.quality)


class SaliencyMaskDefense(Defended):
    """Gray out the region the model fixates on when its attention is suspiciously concentrated."""

    name = "saliency"

    def __init__(self, base: VictimModel, area: float = 0.12, false_positive_rate: float = 0.1):
        super().__init__(base)
        self.area = area  # fraction of the image masked when the detector fires
        self.false_positive_rate = false_positive_rate
        self.threshold: float | None = None
        self.last_flagged: torch.Tensor | None = None

    @property
    def label(self) -> str:
        return "saliency mask"

    def concentration(self, cams: torch.Tensor) -> torch.Tensor:
        """Share of explanation mass inside the top ``area`` fraction of pixels."""
        flat = cams.flatten(1)
        k = max(1, int(self.area * flat.shape[1]))
        top = flat.topk(k, dim=1).values.sum(1)
        return top / flat.sum(1).clamp_min(1e-8)

    def _explain(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            top1 = self.base(images).argmax(-1)
        cams, _ = gradcam(self.base, images, top1)
        return cams

    @torch.no_grad()
    def calibrate(self, clean_images: torch.Tensor, batch_size: int = 50) -> float:
        """Set the threshold so ``false_positive_rate`` of clean photos get masked."""
        scores = torch.cat([
            self.concentration(self._explain(clean_images[i:i + batch_size]))
            for i in range(0, clean_images.shape[0], batch_size)
        ])
        self.threshold = float(torch.quantile(scores.float().cpu(), 1 - self.false_positive_rate))
        return self.threshold

    def mask(self, cams: torch.Tensor, soft_tau: float | None = None) -> torch.Tensor:
        """Where to gray out, (B, 1, H, W): the hotspot of every flagged image.

        With ``soft_tau`` the values are unchanged but gradients flow through sigmoid
        relaxations of both thresholds (hotspot cutoff and trigger), straight-through
        style. Only adaptive attacks use this.
        """
        if self.threshold is None:
            raise RuntimeError("Call calibrate() on clean photos before using the saliency defense.")
        conc = self.concentration(cams)
        flagged = conc > self.threshold
        self.last_flagged = flagged.detach().cpu()
        flat = cams.flatten(1)
        k = max(1, int(self.area * flat.shape[1]))
        cutoff = flat.topk(k, dim=1).values[:, -1].detach().view(-1, 1, 1)
        region = _dilate((cams >= cutoff).float()[:, None]) * flagged.float().view(-1, 1, 1, 1)
        if soft_tau is None:
            return region
        soft = _dilate(torch.sigmoid((cams - cutoff) / soft_tau)[:, None])
        soft = soft * torch.sigmoid((conc - self.threshold) / soft_tau).view(-1, 1, 1, 1)
        return region + soft - soft.detach()

    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        return _gray_out(images, self.mask(self._explain(images)))


def _dilate(region: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(region, 9, stride=1, padding=4)  # cover the hotspot's edges too


def _gray_out(images: torch.Tensor, region: torch.Tensor) -> torch.Tensor:
    return images * (1 - region) + 0.5 * region


def build_defense(spec: str, base: VictimModel) -> Defended:
    """Parse ``name`` or ``name:value`` (blur:<sigma>, jpeg:<quality>, saliency:<area>)."""
    name, _, value = spec.partition(":")
    if name == "blur":
        return BlurDefense(base, float(value) if value else 1.5)
    if name == "jpeg":
        return JPEGDefense(base, int(value) if value else 50)
    if name == "saliency":
        return SaliencyMaskDefense(base, float(value) if value else 0.12)
    raise ValueError(f"Unknown defense '{name}'. Choose from: {', '.join(DEFENSES)}")


# ---------------------------------------------------------------------------
# Adaptive attacks on the saliency mask
# ---------------------------------------------------------------------------


class StraightThroughSaliency(Defended):
    """The saliency defense with a differentiable stand-in for its mask (BPDA).

    The forward pass is the real defense: same Grad-CAM, same trigger, same grayed-out
    hotspot. The backward pass replaces the two hard thresholds with sigmoids, so the
    attacker learns how to move the hotspot off the sticker or keep attention below
    the trigger, instead of seeing a zero gradient wherever the sticker is masked
    (Athalye, Carlini & Wagner, 2018).
    """

    def __init__(self, defense: SaliencyMaskDefense, tau: float = 0.05):
        super().__init__(defense.base)
        self.defense = defense
        self.tau = tau

    @property
    def label(self) -> str:
        return f"{self.defense.label} (straight-through)"

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        cams, _ = differentiable_gradcam(self.base, images)
        return self.base(_gray_out(images, self.defense.mask(cams, soft_tau=self.tau)))


def attention_penalty(model: VictimModel, class_idx: int,
                      weight: float) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Loss term: ``weight`` x the share of ``class_idx``'s Grad-CAM mass on the sticker."""

    def loss(scenes: torch.Tensor, scene_alpha: torch.Tensor) -> torch.Tensor:
        cams, _ = differentiable_gradcam(model, scenes, class_idx)
        return weight * share_on_sticker(cams, scene_alpha).mean()

    return loss


def attack_saliency(
    sticker: Sticker,
    defense: SaliencyMaskDefense,
    target: Target,
    backgrounds: BackgroundSet,
    config: AttackConfig,
    kind: str = "bpda",
    attention_weight: float = 2.0,
    tau: float = 0.05,
    **kwargs,
) -> AttackResult:
    """Re-optimize ``sticker`` against a calibrated saliency defense (see :data:`SALIENCY_ATTACKS`).

    ``bpda`` optimizes the defended prediction through :class:`StraightThroughSaliency`.
    ``attention`` keeps the undefended prediction on target while pushing the target's
    Grad-CAM attention off the sticker. ``combined`` is ``attention`` plus the defended
    prediction: the straight-through gradient alone is too weak to hold the label
    against a large attention weight.
    """
    if kind not in SALIENCY_ATTACKS:
        raise ValueError(f"Unknown adaptive attack '{kind}'. Choose from: {', '.join(SALIENCY_ATTACKS)}")
    through = StraightThroughSaliency(defense, tau)
    models: list[VictimModel] = {"bpda": [through], "attention": [defense.base],
                                 "combined": [defense.base, through]}[kind]
    extra = None
    if kind in ("attention", "combined"):
        extra = attention_penalty(defense.base, defense.class_index(target), attention_weight)
    return optimize_sticker(sticker, models, target, backgrounds, config, extra_loss=extra, **kwargs)


# ---------------------------------------------------------------------------
# Evaluating defenses on a saved sticker
# ---------------------------------------------------------------------------


def evaluate_defenses(
    run_dir,
    specs: list[str] | None = None,
    adaptive_steps: int = 0,
    saliency_attacks: list[str] | None = None,
    attention_weight: float = 2.0,
    soft_tau: float = 0.05,
    device: torch.device | str | None = None,
    say=print,
) -> dict:
    """Measure each defense's protection and its cost; optionally attack it adaptively.

    With ``adaptive_steps > 0`` the sticker is re-optimized through the blur defense
    and attacked with each of ``saliency_attacks`` (default: all of
    :data:`SALIENCY_ATTACKS`) against the saliency defense, with ``attention_weight``
    and ``soft_tau`` passed to :func:`attack_saliency`. Every sticker also gets
    its Grad-CAM attention share, so you can see whether an attack really adapted.
    Writes ``defense.json``, ``defense.md`` and ``fig_defense.png`` into ``run_dir``.
    """
    import json
    from pathlib import Path

    from . import report as rpt
    from .data import load_backgrounds
    from .evaluate import Evaluator
    from .explain import attention_share
    from .models import default_device, load_model
    from .pipeline import _examples, load_run

    run_dir = Path(run_dir)
    device = torch.device(device) if device is not None else default_device()
    specs = specs or ["blur", "jpeg", "saliency"]
    saliency_attacks = list(SALIENCY_ATTACKS) if saliency_attacks is None else saliency_attacks
    for kind in saliency_attacks:
        if kind not in SALIENCY_ATTACKS:
            raise ValueError(f"Unknown adaptive attack '{kind}'. Choose from: {', '.join(SALIENCY_ATTACKS)}")
    sticker, target, request = load_run(run_dir)
    sticker.to(device).eval()
    base = load_model(request.models[0], device)
    t_idx = base.class_index(target)
    val = load_backgrounds("val", request.backgrounds, limit=max(request.eval.n_images * 2, 600),
                           seed=request.eval.seed)
    evaluator = Evaluator(val, target, request.eval, request.attack.eot)
    placement, photometric = evaluator._random_conditions()

    def attention(s: Sticker) -> tuple[float, float, str]:
        """Share of the target's Grad-CAM mass on the sticker, as in the main report."""
        _, scenes, scene_alpha, *_ = _examples(base, s, target, evaluator, n=48)
        cams, method = gradcam(base, scenes, t_idx)
        share = attention_share(cams, scene_alpha, method)
        return share.share_on_sticker, share.sticker_area, share.method

    def assess(model: VictimModel, s: Sticker, attack: str | None = None) -> dict:
        rgb, alpha = s.render()
        clean = evaluator.measure(model, None, None, placement, photometric)
        attacked = evaluator.measure(model, rgb, alpha, placement, photometric)
        row = {
            "defense": getattr(model, "label", "none"),
            "attacker": "original sticker" if attack is None else "adaptive attacker",
            "clean_accuracy": evaluator.clean_accuracy(model),
            "success_rate": attacked.success_rate,
            "disruption_rate": attacked.disruption_rate,
            "attention_on_sticker": attention(s)[0],
        }
        if attack is not None:
            row["attack"] = attack
            row["success_without_defense"] = evaluator.measure(base, rgb, alpha, placement,
                                                               photometric).success_rate
        if attacked.flagged_rate is not None:
            row["flagged_on_sticker"] = attacked.flagged_rate
            row["flagged_on_clean"] = clean.flagged_rate
        return row

    def adapt() -> tuple[Sticker, AttackConfig]:
        adapted = Sticker.from_state(sticker.state())
        adapted.hard = False
        return adapted, replace(request.attack, steps=adaptive_steps, lr=request.attack.lr * 0.5)

    rows = [assess(base, sticker)]
    _, area, method = attention(sticker)
    say(f"no defense: success {rows[0]['success_rate']:.0%}, clean accuracy {rows[0]['clean_accuracy']:.0%}, "
        f"attention on sticker {rows[0]['attention_on_sticker']:.0%}")
    train = load_backgrounds("train", request.backgrounds, limit=300, seed=request.attack.seed + 99)
    result = {"target": target.to_dict(), "model": base.key, "rows": rows,
              "adaptive_steps": adaptive_steps, "sticker_area": area, "attention_method": method,
              "saliency_attack_settings": {"attention_weight": attention_weight, "soft_tau": soft_tau}}
    for spec in specs:
        defended = build_defense(spec, base)
        if isinstance(defended, SaliencyMaskDefense):
            calib = _center_batch(train, request.eval.scene_size, device)
            result["saliency_threshold"] = defended.calibrate(calib)
        row = assess(defended, sticker)
        rows.append(row)
        say(f"{defended.label}: success {row['success_rate']:.0%}, clean accuracy {row['clean_accuracy']:.0%}")
        if isinstance(defended, SaliencyMaskDefense):
            result["saliency_by_size"] = _by_size(evaluator, base, defended, sticker, photometric)
            result["trained_scale"] = list(request.attack.eot.scale)
            say("  by sticker width: " + ", ".join(
                f"{r['scale']:.0%} → {r['success_with_defense']:.0%}" for r in result["saliency_by_size"]))
        if adaptive_steps <= 0:
            continue

        if isinstance(defended, BlurDefense):
            say(f"adaptive attacker vs. {defended.label} ({adaptive_steps} steps)")
            adapted, cfg = adapt()
            optimize_sticker(adapted, [defended], target, train, cfg)
            row = assess(defended, adapted, "re-optimized through blur")
            rows.append(row)
            say(f"  adaptive: success {row['success_rate']:.0%}")
        elif isinstance(defended, SaliencyMaskDefense):
            if base.gradcam_layer is None:
                say(f"  skipping adaptive attacks: {base.key} has no Grad-CAM layer")
                continue
            for kind in saliency_attacks:
                say(f"adaptive attacker ({SALIENCY_ATTACKS[kind]}) vs. {defended.label} "
                    f"({adaptive_steps} steps)")
                adapted, cfg = adapt()
                attack_saliency(adapted, defended, target, train, cfg, kind,
                                attention_weight=attention_weight, tau=soft_tau)
                row = assess(defended, adapted, SALIENCY_ATTACKS[kind])
                rows.append(row)
                say(f"  success {row['success_rate']:.0%} with the defense, "
                    f"{row['success_without_defense']:.0%} without; flagged {row['flagged_on_sticker']:.0%}; "
                    f"attention on sticker {row['attention_on_sticker']:.0%}")

    (run_dir / "defense.json").write_text(json.dumps(result, indent=2))
    fig_defense(rows, run_dir / "fig_defense.png", f"Defending {rpt.model_name(base.key)}",
                sticker_area=area)
    (run_dir / "defense.md").write_text(defense_markdown(result))
    return result


def _by_size(evaluator, base: VictimModel, defense: SaliencyMaskDefense, sticker: Sticker,
             photometric) -> list[dict]:
    """Success with and without the detector at fixed sticker widths (random rotation and position)."""
    rgb, alpha = sticker.render()
    n = evaluator.settings.grid_images
    out = []
    for scale in SIZE_SWEEP:
        placement = evaluator._fixed_placement(scale)
        defended = evaluator.measure(defense, rgb, alpha, placement, photometric, n_images=n)
        out.append({
            "scale": scale,
            "success_without_defense": evaluator.measure(base, rgb, alpha, placement, photometric,
                                                         n_images=n).success_rate,
            "success_with_defense": defended.success_rate,
            "flagged": defended.flagged_rate,
        })
    return out


def _center_batch(bgs, size: int, device) -> torch.Tensor:
    return bgs.center(bgs.images, size).to(device).float() / 255.0


def _is_adaptive(row: dict) -> bool:
    return row["attacker"] == "adaptive attacker"


def _row_label(row: dict) -> str:
    if not _is_adaptive(row):
        return row["defense"]
    return f"{row['defense']} vs. {row.get('attack', 'adaptive attacker')}"


def fig_defense(rows: list[dict], path, title: str, sticker_area: float | None = None):
    import matplotlib.pyplot as plt

    from . import report as rpt

    rpt._style()
    labels = [_row_label(r) for r in rows]
    colors = [rpt.SERIES[1] if _is_adaptive(r) else rpt.SERIES[0] for r in rows]
    panels = [("clean_accuracy", "Accuracy on ordinary photos ↑"), ("success_rate", "Sticker success ↓")]
    if all("attention_on_sticker" in r for r in rows):
        panels.append(("attention_on_sticker", "Grad-CAM attention on sticker"))
    fig, axes = plt.subplots(1, len(panels), figsize=(rpt.FIG_WIDTH + 1.5 * (len(panels) - 2),
                                                      0.45 * len(rows) + 1.6), dpi=150, sharey=True)
    for ax, (key, panel_title) in zip(axes, panels, strict=True):
        rpt._hbar(ax, labels, [r[key] for r in rows], colors)
        ax.set_title(panel_title, loc="left", fontsize=10)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles = []
    if any(_is_adaptive(r) for r in rows):
        handles += [Patch(color=rpt.SERIES[0], label="Original sticker"),
                    Patch(color=rpt.SERIES[1], label="Sticker re-optimized against the defense")]
    if len(panels) == 3 and sticker_area is not None:
        axes[2].axvline(sticker_area, color=rpt.TEXT_2, lw=1, ls="--")
        handles.append(Line2D([], [], color=rpt.TEXT_2, lw=1, ls="--", label="Sticker's share of the image"))
    if handles:
        fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold", color=rpt.TEXT)
    fig.tight_layout(rect=(0, 0.08 if handles else 0, 1, 1))
    fig.savefig(path)
    plt.close(fig)
    return path


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{x:.0%}"


def defense_markdown(result: dict) -> str:
    from . import report as rpt

    t = result["target"]["label"]
    rows = result["rows"]
    lines = [f'# Defending {rpt.model_name(result["model"])} against the "{t}" sticker', ""]
    lines.append("| Defense | Attacker | Clean accuracy | Sticker success | Correct label lost | "
                 "Flagged (sticker / clean) | Attention on sticker |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in rows:
        flagged = (f"{r['flagged_on_sticker']:.0%} / {r['flagged_on_clean']:.0%}"
                   if "flagged_on_sticker" in r else "—")
        attacker = f"adaptive: {r['attack']}" if "attack" in r else r["attacker"]
        lines.append(f"| {r['defense']} | {attacker} | {r['clean_accuracy']:.0%} | "
                     f"{r['success_rate']:.0%} | {r['disruption_rate']:.0%} | {flagged} | "
                     f"{_pct(r.get('attention_on_sticker'))} |")
    lines += [
        "",
        "![Defenses](fig_defense.png)",
        "",
        "**Reading this table.** A good defense lowers sticker success without lowering "
        "clean accuracy. The *adaptive* rows re-optimize the sticker knowing the "
        "defense is there. Defenses that only look strong against the original sticker "
        "often fail once the attacker adapts: an arms race.",
        "",
        "Clean accuracy is measured on the held-out Imagenette photos (10 classes) "
        "as a 1,000-way ImageNet prediction. The saliency defense's trigger threshold is "
        "calibrated on training photos so about 10% of clean photos get masked. "
        "*Attention on sticker* is the share of the target's Grad-CAM mass that falls on "
        f"the sticker in 48 held-out photos, where the sticker covers {_pct(result.get('sticker_area'))} "
        "of the image.",
        "",
    ]
    lines += _saliency_section(result)
    lines += _size_section(result)
    return "\n".join(lines)


def _saliency_section(result: dict) -> list[str]:
    rows = result["rows"]
    original = next((r for r in rows if r["defense"] == "saliency mask" and not _is_adaptive(r)), None)
    adaptive = [r for r in rows if r["defense"] == "saliency mask" and _is_adaptive(r)]
    if original is None or not adaptive:
        return []
    undefended = rows[0]
    best = max(adaptive, key=lambda r: r["success_rate"])
    cost = undefended["clean_accuracy"] - original["clean_accuracy"]
    lines = [
        "## Adaptive attacks on the saliency detector",
        "",
        f"Each attacker starts from the original sticker and re-optimizes it for "
        f"{result['adaptive_steps']} steps on training photos, knowing the detector and its "
        "threshold:",
        "",
        "- **BPDA (straight-through mask)**: optimize the *defended* prediction. The forward "
        "pass is the real detector; the backward pass swaps its two hard thresholds for "
        "sigmoids, so the gradient says how to move or shrink the masked hotspot "
        "(Athalye, Carlini & Wagner, 2018).",
        "- **Attention-evasion loss**: keep the *undefended* prediction on target while "
        "penalizing the share of Grad-CAM attention on the sticker.",
        "- **BPDA + attention loss**: the attention attack, also optimizing the defended "
        "prediction through the straight-through mask.",
        "",
        "| Sticker | Attention on sticker | Flagged | Success with detector | Success without detector |",
        "|---|---|---|---|---|",
        f"| original | {_pct(original.get('attention_on_sticker'))} | "
        f"{_pct(original.get('flagged_on_sticker'))} | {original['success_rate']:.0%} | "
        f"{undefended['success_rate']:.0%} |",
    ]
    for r in adaptive:
        lines.append(f"| adaptive: {r['attack']} | {_pct(r.get('attention_on_sticker'))} | "
                     f"{_pct(r.get('flagged_on_sticker'))} | {r['success_rate']:.0%} | "
                     f"{_pct(r.get('success_without_defense'))} |")
    lines += [
        "",
        f"The detector costs {cost * 100:.0f} points of clean accuracy "
        f"({undefended['clean_accuracy']:.0%} → {original['clean_accuracy']:.0%}) whichever "
        f"sticker it faces. Against the original sticker it holds success to "
        f"{original['success_rate']:.0%}; the strongest adaptive attack here "
        f"({best['attack']}) reaches {best['success_rate']:.0%} "
        f"(undefended: {undefended['success_rate']:.0%}).",
        "",
    ]
    return lines


def _size_section(result: dict) -> list[str]:
    sizes = result.get("saliency_by_size")
    if not sizes:
        return []
    lo, hi = result.get("trained_scale", (None, None))
    trained = (f" The sticker was optimized and tested at widths of {lo:.0%}–{hi:.0%}."
               if lo is not None else "")
    lines = [
        "## Does sticker size matter to the saliency detector?",
        "",
        "The original sticker at fixed widths (share of the photo's width), with random "
        f"rotation and position.{trained}",
        "",
        "| Sticker width | Success without detector | Success with detector | Flagged |",
        "|---|---|---|---|",
    ]
    for r in sizes:
        lines.append(f"| {r['scale']:.0%} | {r['success_without_defense']:.0%} | "
                     f"{r['success_with_defense']:.0%} | {_pct(r.get('flagged'))} |")
    lines += [
        "",
        "The detector grays out the top 12% of Grad-CAM pixels plus a margin, which is more "
        "than a small sticker covers. A sticker big enough to spread the model's attention "
        "over a larger area no longer looks *concentrated*, so the detector stops firing.",
        "",
    ]
    return lines
