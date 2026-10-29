"""The evaluation battery: how well, and under what conditions, does a sticker work?

All measurements use held-out photos (Imagenette *val*) the sticker never saw
during optimization, and fixed random seeds so results are reproducible. The
battery has four parts:

1. **Robustness grid** — attack success rate over rotation x sticker size.
2. **Condition sweeps** — one imaging factor at a time (lighting, blur, noise,
   perspective tilt, JPEG compression, color cast), each from mild to severe.
3. **Transfer** — success against models the sticker was not optimized on.
4. **Controls** — the same scenes with no sticker, a random-noise sticker of the
   same shape, and (for cover modes) the untouched cover photo, which separates
   the effect of the optimization from mere occlusion.

Metrics per condition:

* ``success_rate`` — fraction of photos whose top-1 prediction is the target.
* ``top5_rate``    — fraction with the target among the top 5.
* ``target_prob``  — mean softmax probability of the target.
* ``disruption_rate`` — among photos the model got right without a sticker,
  the fraction it now gets wrong (an untargeted view of the same attack).
"""

from __future__ import annotations

import io
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
from PIL import Image

from .data import BackgroundSet
from .labels import Target
from .models import VictimModel
from .sticker import Sticker
from .transforms import (
    EOTConfig,
    Photometric,
    Placement,
    neutral_photometric,
    render_scenes,
    sample_photometric,
    sample_placement,
)


@dataclass
class EvalSettings:
    n_images: int = 300  # held-out photos per condition
    grid_images: int = 100  # photos per rotation x scale cell
    batch_size: int = 50
    scene_size: int = 224
    default_scale: float = 0.35  # sticker width / image width when not swept
    seed: int = 1234
    grid_angles: tuple[float, ...] = tuple(range(-180, 181, 30))
    grid_scales: tuple[float, ...] = (0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5)
    sweeps: dict[str, tuple[float, ...]] = field(default_factory=lambda: {
        "brightness": (0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6),
        "contrast": (0.4, 0.6, 0.8, 1.0, 1.2, 1.4),
        "blur": (0.0, 0.5, 1.0, 1.5, 2.0, 3.0),
        "noise": (0.0, 0.02, 0.04, 0.06, 0.08, 0.12),
        "perspective": (0.0, 0.1, 0.2, 0.3, 0.4, 0.5),
        "jpeg": (100, 80, 60, 40, 20, 10),
        "color_cast": (0.0, 0.05, 0.1, 0.15, 0.2, 0.3),
    })

    def to_dict(self) -> dict:
        return asdict(self)


SWEEP_DESCRIPTIONS = {
    "brightness": "Scene brightness multiplier (1.0 = unchanged; <1 darker, >1 brighter).",
    "contrast": "Scene contrast multiplier (1.0 = unchanged).",
    "blur": "Gaussian blur sigma in pixels (camera out of focus / motion).",
    "noise": "Gaussian sensor-noise standard deviation (low light, cheap cameras).",
    "perspective": "Perspective tilt of the sticker (viewed at an angle).",
    "jpeg": "JPEG quality (lower = heavier compression, as in messaging apps).",
    "color_cast": "Warm color cast strength on the printed sticker (printer or lighting tint).",
}


@dataclass
class Metrics:
    success_rate: float
    top5_rate: float
    target_prob: float
    disruption_rate: float
    n: int
    flagged_rate: float | None = None  # detector-style defenses: share of photos flagged

    def to_dict(self) -> dict:
        d = asdict(self)
        if d["flagged_rate"] is None:
            del d["flagged_rate"]
        return d


def _jpeg(images: torch.Tensor, quality: int) -> torch.Tensor:
    if quality >= 100:
        return images
    out = []
    for img in images.detach().cpu():
        arr = (img.permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="JPEG", quality=int(quality))
        buf.seek(0)
        dec = np.asarray(Image.open(buf).convert("RGB"), dtype=np.float32) / 255.0
        out.append(torch.from_numpy(dec).permute(2, 0, 1))
    return torch.stack(out).to(images.device)


class Evaluator:
    """Runs measurements for one sticker against one or more models on held-out photos."""

    def __init__(
        self,
        backgrounds: BackgroundSet,
        target: Target,
        settings: EvalSettings | None = None,
        eot: EOTConfig | None = None,
    ):
        self.settings = settings or EvalSettings()
        self.eot = eot or EOTConfig()
        self.target = target
        bgs = backgrounds.without_label(target.imagenet_index)
        n = min(self.settings.n_images, len(bgs))
        self.backgrounds = BackgroundSet(bgs.images[:n], bgs.labels[:n], bgs.source)
        self._clean_cache: dict[str, torch.Tensor] = {}

    # -- core measurement --------------------------------------------------

    @torch.no_grad()
    def _predict(self, model: VictimModel, scenes: torch.Tensor) -> torch.Tensor:
        return model(scenes).float().softmax(-1)

    def clean_accuracy(self, model: VictimModel) -> float:
        """Top-1 accuracy on the held-out photos whose true label is known."""
        known = self.backgrounds.labels >= 0
        if not bool(known.any()):
            return float("nan")
        preds = self.clean_predictions(model)
        return float((preds[known] == self.backgrounds.labels[known]).float().mean())

    @torch.no_grad()
    def clean_predictions(self, model: VictimModel) -> torch.Tensor:
        if model.key not in self._clean_cache:
            device = next(model.parameters()).device
            preds = [
                self._predict(model, imgs).argmax(-1).cpu()
                for imgs, _ in self.backgrounds.batches(self.settings.batch_size, self.settings.scene_size, device)
            ]
            self._clean_cache[model.key] = torch.cat(preds)
        return self._clean_cache[model.key]

    @torch.no_grad()
    def measure(
        self,
        model: VictimModel,
        rgb: torch.Tensor | None,
        alpha: torch.Tensor | None,
        placement_fn: Callable[[int, torch.Generator], Placement],
        photometric_fn: Callable[[int, torch.Generator], Photometric],
        n_images: int | None = None,
        post: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> Metrics:
        """Success metrics for one condition. ``rgb=None`` measures clean photos."""
        s = self.settings
        device = next(model.parameters()).device
        t_idx = model.class_index(self.target)
        gen = torch.Generator().manual_seed(s.seed)
        clean = self.clean_predictions(model)
        n_total = min(n_images or len(self.backgrounds), len(self.backgrounds))

        hits = top5 = disrupted = correct_clean = flagged = 0
        has_detector = hasattr(model, "last_flagged")
        prob_sum = 0.0
        for start in range(0, n_total, s.batch_size):
            stop = min(start + s.batch_size, n_total)
            imgs = BackgroundSet.center(self.backgrounds.images[start:stop], s.scene_size)
            imgs = imgs.to(device).float() / 255.0
            labels = self.backgrounds.labels[start:stop]
            b = imgs.shape[0]
            placement, photometric = placement_fn(b, gen), photometric_fn(b, gen)
            if rgb is None:
                scenes = imgs
            else:
                scenes, _ = render_scenes(imgs, rgb, alpha, placement, photometric, self.eot.gamut, gen)
            if post is not None:
                scenes = post(scenes)
            probs = self._predict(model, scenes).cpu()
            if has_detector:
                flagged += int(model.last_flagged.sum())
            pred = probs.argmax(-1)
            hits += int((pred == t_idx).sum())
            top5 += int((probs.topk(5, -1).indices == t_idx).any(-1).sum())
            prob_sum += float(probs[:, t_idx].sum())
            was_right = clean[start:stop] == labels
            correct_clean += int(was_right.sum())
            disrupted += int((was_right & (pred != labels)).sum())
        return Metrics(
            success_rate=hits / n_total,
            top5_rate=top5 / n_total,
            target_prob=prob_sum / n_total,
            disruption_rate=disrupted / max(1, correct_clean),
            n=n_total,
            flagged_rate=flagged / n_total if has_detector else None,
        )

    # -- condition builders ------------------------------------------------

    def _fixed_placement(self, scale: float, angle_deg: float | None = None, tilt: float = 0.0):
        def fn(n: int, gen: torch.Generator) -> Placement:
            if angle_deg is None:
                angle = (torch.rand(n, generator=gen) * 2 - 1) * math.pi
            else:
                angle = torch.full((n,), math.radians(angle_deg))
            reach = max(0.0, 1 - scale)
            center = (torch.rand(n, 2, generator=gen) * 2 - 1) * reach
            # Random tilt direction with fixed magnitude.
            theta = torch.rand(n, generator=gen) * 2 * math.pi
            tilt_xy = torch.stack([torch.cos(theta), torch.sin(theta)], 1) * tilt
            return Placement(torch.full((n,), scale), angle, center, tilt_xy)
        return fn

    @staticmethod
    def _neutral(n: int, gen: torch.Generator) -> Photometric:
        return neutral_photometric(n)

    def _random_conditions(self):
        """Training-like random conditions, for the headline 'in the wild' number."""
        return (
            lambda n, gen: sample_placement(n, self.eot, gen),
            lambda n, gen: sample_photometric(n, self.eot, gen),
        )

    # -- battery -----------------------------------------------------------

    def headline(self, model: VictimModel, sticker: Sticker) -> Metrics:
        rgb, alpha = sticker.render()
        return self.measure(model, rgb, alpha, *self._random_conditions())

    def robustness_grid(self, model: VictimModel, sticker: Sticker) -> dict:
        s = self.settings
        rgb, alpha = sticker.render()
        grid = []
        for scale in s.grid_scales:
            row = []
            for angle in s.grid_angles:
                m = self.measure(model, rgb, alpha, self._fixed_placement(scale, angle),
                                 self._neutral, n_images=s.grid_images)
                row.append(m.success_rate)
            grid.append(row)
        return {"angles": list(s.grid_angles), "scales": list(s.grid_scales), "success_rate": grid}

    def sweep(self, model: VictimModel, sticker: Sticker, factor: str) -> dict:
        s = self.settings
        rgb, alpha = sticker.render()
        values = s.sweeps[factor]
        results = []
        for v in values:
            placement = self._fixed_placement(s.default_scale, None, v if factor == "perspective" else 0.0)
            post = None

            def photometric(n: int, gen: torch.Generator, v=v) -> Photometric:
                p = neutral_photometric(n)
                if factor == "brightness":
                    p.brightness = torch.full((n,), float(v))
                elif factor == "contrast":
                    p.contrast = torch.full((n,), float(v))
                elif factor == "blur":
                    p.blur_sigma = float(v)
                elif factor == "noise":
                    p.noise_std = torch.full((n,), float(v))
                elif factor == "color_cast":
                    p.gain = torch.tensor([1 + v, 1.0, 1 - v]).expand(n, 3).clone()
                return p

            if factor == "jpeg":
                post = lambda x, q=int(v): _jpeg(x, q)  # noqa: E731
            m = self.measure(model, rgb, alpha, placement, photometric, post=post)
            results.append(m.to_dict())
        return {"factor": factor, "values": list(values), "description": SWEEP_DESCRIPTIONS[factor],
                "results": results}

    def controls(self, model: VictimModel, sticker: Sticker) -> dict[str, dict]:
        """Clean photos vs. a random sticker vs. the cover photo vs. the adversarial sticker."""
        placement, photometric = self._random_conditions()
        rgb, alpha = sticker.render()
        out = {"no sticker": self.measure(model, None, None, placement, photometric).to_dict()}
        gen = torch.Generator().manual_seed(self.settings.seed + 1)
        noise = torch.rand(rgb.shape, generator=gen).to(rgb)
        out["random-noise sticker"] = self.measure(model, noise, alpha, placement, photometric).to_dict()
        if sticker.design.cover_mode != "none":
            cover = sticker.cover.to(rgb)
            out["cover photo only"] = self.measure(model, cover, alpha, placement, photometric).to_dict()
        out["adversarial sticker"] = self.measure(model, rgb, alpha, placement, photometric).to_dict()
        return out

    def full(
        self,
        model: VictimModel,
        sticker: Sticker,
        sweeps: bool = True,
        grid: bool = True,
        progress: Callable[[str], None] | None = None,
    ) -> dict:
        """Run the whole battery for one model."""
        say = progress or (lambda _msg: None)
        result: dict = {"model": model.key}
        say(f"[{model.key}] controls")
        result["controls"] = self.controls(model, sticker)
        result["headline"] = result["controls"]["adversarial sticker"]
        if grid:
            say(f"[{model.key}] rotation x scale grid")
            result["grid"] = self.robustness_grid(model, sticker)
        if sweeps:
            result["sweeps"] = {}
            for factor in self.settings.sweeps:
                say(f"[{model.key}] sweep: {factor}")
                result["sweeps"][factor] = self.sweep(model, sticker, factor)
        return result
