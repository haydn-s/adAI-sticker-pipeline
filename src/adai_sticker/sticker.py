"""The sticker: trainable parameters plus the user's design choices.

A sticker is rendered as an RGB image and an alpha (shape) mask. What the
optimizer is allowed to change depends on the design:

* ``cover_mode="none"``   — every pixel inside the outline is adversarial.
* ``cover_mode="region"`` — a cover photo is kept as-is except inside an
  editable region (a border, stripes, a custom painted mask, ...).
* ``cover_mode="blend"``  — the whole cover photo is nudged by a perturbation
  bounded by ``epsilon`` per pixel; a larger budget is stronger but less natural.

Adversarial pixels are either free RGB or constrained to a color palette. Palette
stickers are optimized in free RGB while a penalty pulls colors toward the
palette; then every pixel snaps to its nearest palette color in the forward pass,
with a straight-through estimator carrying gradients back to the free colors.
The final printed sticker is therefore exactly what was evaluated.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from .masks import load_mask, region_mask
from .palette import palette_from_image, parse_palette

CoverMode = Literal["none", "region", "blend"]


@dataclass
class StickerDesign:
    """Everything the user chooses about how the sticker looks."""

    resolution: int = 224
    shape: str = "circle"
    palette: str | list[str] | None = None  # None = unconstrained RGB; "auto" = from cover
    palette_size: int = 8  # colors extracted when palette == "auto"
    cover_mode: CoverMode = "none"
    cover_image: str | None = None
    region: str = "stripes"  # region preset or path to a mask image (region mode)
    region_amount: float = 0.4
    epsilon: float = 0.12  # per-pixel L-infinity budget (blend mode), in [0, 1] units

    def validate(self) -> None:
        if self.cover_mode not in ("none", "region", "blend"):
            raise ValueError(f"Unknown cover mode '{self.cover_mode}'.")
        if self.cover_mode != "none" and not self.cover_image:
            raise ValueError(f"cover_mode='{self.cover_mode}' needs a cover_image.")
        if self.cover_mode == "blend" and self.palette is not None:
            raise ValueError("Blend mode perturbs the cover photo directly; palettes don't apply.")
        if self.palette == "auto" and not self.cover_image:
            raise ValueError("palette='auto' derives colors from the cover image; provide one.")
        if not 0 < self.epsilon <= 1:
            raise ValueError("epsilon must be in (0, 1].")
        if self.resolution < 32:
            raise ValueError("resolution must be at least 32 pixels.")

    def to_dict(self) -> dict:
        return asdict(self)


def load_image(path: str | Path, size: int) -> torch.Tensor:
    """Load an RGB image, center-crop to square, resize -> (3, size, size) in [0, 1]."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    side = min(w, h)
    img = img.crop(((w - side) // 2, (h - side) // 2, (w - side) // 2 + side, (h - side) // 2 + side))
    img = img.resize((size, size), Image.Resampling.LANCZOS)
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)


def ssim(a: torch.Tensor, b: torch.Tensor, weight: torch.Tensor | None = None) -> float:
    """Mean structural similarity (Wang et al., 2004) with an 11px Gaussian window."""
    from .transforms import gaussian_blur

    c1, c2 = 0.01**2, 0.03**2
    mu_a, mu_b = gaussian_blur(a, 1.5), gaussian_blur(b, 1.5)
    var_a = gaussian_blur(a * a, 1.5) - mu_a**2
    var_b = gaussian_blur(b * b, 1.5) - mu_b**2
    cov = gaussian_blur(a * b, 1.5) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2))
    if weight is None:
        return float(s.mean())
    w = weight.expand_as(s)
    return float((s * w).sum() / w.sum().clamp_min(1.0))


def _logit(p: torch.Tensor) -> torch.Tensor:
    p = p.clamp(1e-4, 1 - 1e-4)
    return torch.log(p / (1 - p))


class Sticker(nn.Module):
    def __init__(self, design: StickerDesign, seed: int = 0):
        super().__init__()
        design.validate()
        self.design = design
        res = design.resolution
        gen = torch.Generator().manual_seed(seed)

        self.register_buffer("mask", load_mask(design.shape, res)[None, None])
        cover = load_image(design.cover_image, res) if design.cover_image else torch.zeros(3, res, res)
        self.register_buffer("cover", cover[None])
        if design.cover_mode == "region":
            region = region_mask(design.region, res, design.region_amount)
        else:
            region = torch.ones(res, res)
        self.register_buffer("region", region[None, None])

        palette = self._resolve_palette(cover)
        self.register_buffer("palette", palette if palette is not None else torch.empty(0, 3))
        self.hard = False

        if design.cover_mode == "blend":
            # delta = epsilon * tanh(w); start from the untouched cover photo.
            self.param = nn.Parameter(torch.zeros(1, 3, res, res))
        else:
            # Free RGB in logit space (palette stickers quantize these colors).
            init = torch.rand(1, 3, res, res, generator=gen) * 0.8 + 0.1
            self.param = nn.Parameter(_logit(init))

    def _resolve_palette(self, cover: torch.Tensor) -> torch.Tensor | None:
        spec = self.design.palette
        if spec is None:
            return None
        if spec == "auto":
            return palette_from_image(cover, k=self.design.palette_size)
        return parse_palette(spec)

    @property
    def uses_palette(self) -> bool:
        return self.palette.numel() > 0

    # -- rendering ---------------------------------------------------------

    def quantize(self, rgb: torch.Tensor) -> torch.Tensor:
        """Snap every pixel to its nearest palette color."""
        b, _, h, w = rgb.shape
        pixels = rgb.permute(0, 2, 3, 1).reshape(-1, 3)
        nearest = torch.cdist(pixels, self.palette).argmin(1)
        return self.palette[nearest].reshape(b, h, w, 3).permute(0, 3, 1, 2)

    def _adversarial_layer(self, param: torch.Tensor, hard: bool) -> torch.Tensor:
        free = torch.sigmoid(param)
        if not self.uses_palette:
            return free
        if not hard:
            return free
        snapped = self.quantize(free)
        if not torch.is_grad_enabled():
            return snapped  # exact palette colors for export
        return free + (snapped - free).detach()  # straight-through

    def palette_distance(self) -> torch.Tensor:
        """Mean distance from the free colors to the palette (pulls colors toward it)."""
        from .losses import non_printability

        return non_printability(torch.sigmoid(self.param), self.palette, self.mask * self.region)

    def forward(self, hard: bool | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Render at optimization resolution -> (rgb (1,3,H,W), alpha (1,1,H,W))."""
        hard = self.hard if hard is None else hard
        return self._compose(self.param, self.cover, self.region, self.mask, hard)

    def _compose(self, param, cover, region, mask, hard) -> tuple[torch.Tensor, torch.Tensor]:
        mode = self.design.cover_mode
        if mode == "blend":
            rgb = (cover + self.design.epsilon * torch.tanh(param)).clamp(0, 1)
        else:
            adv = self._adversarial_layer(param, hard)
            rgb = cover * (1 - region) + adv * region if mode == "region" else adv
        return rgb, mask

    @torch.no_grad()
    def render(self, resolution: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Final (hard) render, optionally re-rendered crisply at a print resolution.

        Outlines, cover photos and region masks are re-generated at the target
        resolution from the design; only the learned adversarial layer is upsampled
        (nearest-neighbor for palettes so colors stay exact, bicubic otherwise).
        """
        res = resolution or self.design.resolution
        if res == self.design.resolution:
            return self.forward(hard=True)
        d = self.design
        device = self.param.device
        mask = load_mask(d.shape, res)[None, None].to(device)
        cover = (load_image(d.cover_image, res)[None] if d.cover_image
                 else torch.zeros(1, 3, res, res)).to(device)
        region = (region_mask(d.region, res, d.region_amount) if d.cover_mode == "region"
                  else torch.ones(res, res))[None, None].to(device)
        if self.uses_palette:
            param = F.interpolate(self.param, size=(res, res), mode="nearest")
        else:
            param = F.interpolate(self.param, size=(res, res), mode="bicubic", align_corners=False)
        return self._compose(param, cover, region, mask, hard=True)

    @torch.no_grad()
    def perturbation_stats(self) -> dict[str, float]:
        """How far the sticker strays from its cover photo (stealth measures).

        ``ssim`` is the structural similarity to the cover photo inside the outline
        (1.0 = indistinguishable); the other numbers are in [0, 1] pixel units.
        """
        rgb, alpha = self.forward(hard=True)
        if self.design.cover_mode == "none":
            return {"changed_fraction": 1.0}
        diff = (rgb - self.cover).abs() * alpha
        inside = alpha.sum().clamp_min(1.0)
        changed = (diff.amax(1, keepdim=True) > 1 / 255).float()
        return {
            "changed_fraction": float((changed * alpha).sum() / inside),
            "mean_abs_change": float(diff.sum() / (inside * 3)),
            "max_abs_change": float(diff.max()),
            "ssim": ssim(rgb, self.cover, alpha),
        }

    # -- persistence -------------------------------------------------------

    def state(self) -> dict:
        return {
            "design": self.design.to_dict(),
            "param": self.param.detach().cpu(),
            "palette": self.palette.detach().cpu(),
        }

    @classmethod
    def from_state(cls, state: dict) -> Sticker:
        design = StickerDesign(**state["design"])
        sticker = cls(design)
        with torch.no_grad():
            sticker.param.copy_(state["param"])
            if state["palette"].numel():
                sticker.palette.copy_(state["palette"])
        sticker.hard = True
        return sticker
