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
"""

from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn.functional as F

from .evaluate import _jpeg
from .explain import gradcam
from .labels import Target
from .models import VictimModel
from .transforms import gaussian_blur

DEFENSES = ("blur", "jpeg", "saliency")


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

    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        if self.threshold is None:
            raise RuntimeError("Call calibrate() on clean photos before using the saliency defense.")
        cams = self._explain(images)
        flagged = self.concentration(cams) > self.threshold
        self.last_flagged = flagged.detach().cpu()
        flat = cams.flatten(1)
        k = max(1, int(self.area * flat.shape[1]))
        cutoff = flat.topk(k, dim=1).values[:, -1:, None]
        region = (cams >= cutoff.view(-1, 1, 1)).float()[:, None]
        region = F.max_pool2d(region, 9, stride=1, padding=4)  # cover the hotspot's edges too
        region = region * flagged.float().view(-1, 1, 1, 1)
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
# Evaluating defenses on a saved sticker
# ---------------------------------------------------------------------------


def evaluate_defenses(
    run_dir,
    specs: list[str] | None = None,
    adaptive_steps: int = 0,
    device: torch.device | str | None = None,
    say=print,
) -> dict:
    """Measure each defense's protection and its cost; optionally attack it adaptively.

    Writes ``defense.json``, ``defense.md`` and ``fig_defense.png`` into ``run_dir``.
    """
    from pathlib import Path

    from . import report as rpt
    from .data import load_backgrounds
    from .evaluate import Evaluator
    from .models import default_device, load_model
    from .optimize import optimize_sticker
    from .pipeline import load_run
    from .sticker import Sticker

    run_dir = Path(run_dir)
    device = torch.device(device) if device is not None else default_device()
    specs = specs or ["blur", "jpeg", "saliency"]
    sticker, target, request = load_run(run_dir)
    sticker.to(device).eval()
    base = load_model(request.models[0], device)
    val = load_backgrounds("val", request.backgrounds, limit=max(request.eval.n_images * 2, 600),
                           seed=request.eval.seed)
    evaluator = Evaluator(val, target, request.eval, request.attack.eot)
    placement, photometric = evaluator._random_conditions()

    def assess(model: VictimModel, s: Sticker, attacker: str) -> dict:
        rgb, alpha = s.render()
        clean = evaluator.measure(model, None, None, placement, photometric)
        attacked = evaluator.measure(model, rgb, alpha, placement, photometric)
        row = {
            "defense": getattr(model, "label", "none"),
            "attacker": attacker,
            "clean_accuracy": evaluator.clean_accuracy(model),
            "success_rate": attacked.success_rate,
            "disruption_rate": attacked.disruption_rate,
        }
        if attacked.flagged_rate is not None:
            row["flagged_on_sticker"] = attacked.flagged_rate
            row["flagged_on_clean"] = clean.flagged_rate
        return row

    rows = [assess(base, sticker, "original sticker")]
    say(f"no defense: success {rows[0]['success_rate']:.0%}, clean accuracy {rows[0]['clean_accuracy']:.0%}")
    train = load_backgrounds("train", request.backgrounds, limit=300, seed=request.attack.seed + 99)
    for spec in specs:
        defended = build_defense(spec, base)
        if isinstance(defended, SaliencyMaskDefense):
            calib = _center_batch(train, request.eval.scene_size, device)
            defended.calibrate(calib)
        row = assess(defended, sticker, "original sticker")
        rows.append(row)
        say(f"{defended.label}: success {row['success_rate']:.0%}, clean accuracy {row['clean_accuracy']:.0%}")

        if adaptive_steps > 0 and isinstance(defended, BlurDefense):
            say(f"adaptive attacker vs. {defended.label} ({adaptive_steps} steps)")
            adapted = Sticker.from_state(sticker.state())
            adapted.hard = False
            cfg = replace(request.attack, steps=adaptive_steps, lr=request.attack.lr * 0.5)
            optimize_sticker(adapted, [defended], target, train, cfg)
            row = assess(defended, adapted, "adaptive attacker")
            rows.append(row)
            say(f"  adaptive: success {row['success_rate']:.0%}")

    result = {"target": target.to_dict(), "model": base.key, "rows": rows,
              "adaptive_steps": adaptive_steps}
    import json

    (run_dir / "defense.json").write_text(json.dumps(result, indent=2))
    fig_defense(rows, run_dir / "fig_defense.png", f"Defending {rpt.model_name(base.key)}")
    (run_dir / "defense.md").write_text(defense_markdown(result))
    return result


def _center_batch(bgs, size: int, device) -> torch.Tensor:
    return bgs.center(bgs.images, size).to(device).float() / 255.0


def _row_label(row: dict) -> str:
    label = row["defense"]
    return f"{label} (adaptive attacker)" if row["attacker"] == "adaptive attacker" else label


def fig_defense(rows: list[dict], path, title: str):
    import matplotlib.pyplot as plt

    from . import report as rpt

    rpt._style()
    labels = [_row_label(r) for r in rows]
    colors = [rpt.SERIES[1] if r["attacker"] == "adaptive attacker" else rpt.SERIES[0] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(rpt.FIG_WIDTH, 0.45 * len(rows) + 1.6), dpi=150, sharey=True)
    rpt._hbar(axes[0], labels, [r["clean_accuracy"] for r in rows], colors)
    axes[0].set_title("Accuracy on ordinary photos ↑", loc="left", fontsize=10)
    rpt._hbar(axes[1], labels, [r["success_rate"] for r in rows], colors)
    axes[1].set_title("Sticker success ↓", loc="left", fontsize=10)
    has_adaptive = any(r["attacker"] == "adaptive attacker" for r in rows)
    if has_adaptive:
        from matplotlib.patches import Patch

        fig.legend(handles=[Patch(color=rpt.SERIES[0], label="Original sticker"),
                            Patch(color=rpt.SERIES[1], label="Sticker re-optimized against the defense")],
                   loc="lower center", ncol=2, frameon=False)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold", color=rpt.TEXT)
    fig.tight_layout(rect=(0, 0.08 if has_adaptive else 0, 1, 1))
    fig.savefig(path)
    plt.close(fig)
    return path


def defense_markdown(result: dict) -> str:
    from . import report as rpt

    t = result["target"]["label"]
    lines = [f'# Defending {rpt.model_name(result["model"])} against the "{t}" sticker', ""]
    lines.append("| Defense | Attacker | Clean accuracy | Sticker success | Correct label lost | "
                 "Flagged (sticker / clean) |")
    lines.append("|---|---|---|---|---|---|")
    for r in result["rows"]:
        flagged = (f"{r['flagged_on_sticker']:.0%} / {r['flagged_on_clean']:.0%}"
                   if "flagged_on_sticker" in r else "—")
        lines.append(f"| {r['defense']} | {r['attacker']} | {r['clean_accuracy']:.0%} | "
                     f"{r['success_rate']:.0%} | {r['disruption_rate']:.0%} | {flagged} |")
    lines += [
        "",
        "![Defenses](fig_defense.png)",
        "",
        "**Reading this table.** A good defense lowers sticker success without lowering "
        "clean accuracy. The *adaptive attacker* rows re-optimize the sticker knowing the "
        "defense is there. Defenses that only look strong against the original sticker "
        "often fail once the attacker adapts: an arms race.",
        "",
        "Clean accuracy is measured on the held-out Imagenette photos (10 classes) "
        "as a 1,000-way ImageNet prediction. The saliency defense's trigger threshold is "
        "calibrated on training photos so about 10% of clean photos get masked.",
        "",
    ]
    return "\n".join(lines)
