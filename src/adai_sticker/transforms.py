"""Expectation over Transformation (EOT): put the sticker into many simulated photos.

Following Athalye et al. (2018) and Brown et al. (2017), the sticker is optimized
on the *average* over random placements and imaging conditions, so it keeps
working when it is printed, stuck somewhere, and photographed. Each simulated
photo is built in three differentiable stages:

1. **Print**   — color gain/offset jitter and gamut compression on the sticker.
2. **Place**   — scale, rotation, position and a perspective tilt (a homography),
   then alpha-composited onto a background photo.
3. **Camera**  — brightness, contrast, blur and sensor noise on the whole scene.

The same functions take explicit parameters, so the evaluator can sweep one
factor at a time (e.g. rotation x scale grids).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F


@dataclass
class EOTConfig:
    scale: tuple[float, float] = (0.25, 0.45)  # sticker width / image width
    rotation: float = 180.0  # max |degrees|
    perspective: float = 0.25  # max homography tilt term
    print_gain: float = 0.08  # per-channel color gain jitter (printer drift)
    print_offset: float = 0.04  # per-channel additive jitter
    gamut: tuple[float, float] = (0.04, 0.96)  # printable range the inks reach
    brightness: tuple[float, float] = (0.7, 1.3)
    contrast: tuple[float, float] = (0.75, 1.25)
    blur_sigma: tuple[float, float] = (0.0, 1.2)  # in output pixels
    noise_std: float = 0.03

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Placement:
    """Per-image geometric parameters (all tensors of shape (B,) or (B, 2))."""

    scale: torch.Tensor
    angle: torch.Tensor  # radians
    center: torch.Tensor  # (B, 2) in normalized [-1, 1] scene coords
    tilt: torch.Tensor  # (B, 2) homography perspective terms


@dataclass
class Photometric:
    """Per-image print and camera parameters."""

    gain: torch.Tensor  # (B, 3)
    offset: torch.Tensor  # (B, 3)
    brightness: torch.Tensor  # (B,)
    contrast: torch.Tensor  # (B,)
    noise_std: torch.Tensor  # (B,)
    blur_sigma: float  # one value per batch (keeps the blur a single conv)


def _uniform(n: int, lo: float, hi: float, gen: torch.Generator | None) -> torch.Tensor:
    return torch.rand(n, generator=gen) * (hi - lo) + lo


def sample_placement(n: int, cfg: EOTConfig, gen: torch.Generator | None = None) -> Placement:
    scale = _uniform(n, *cfg.scale, gen)
    angle = _uniform(n, -1.0, 1.0, gen) * math.radians(cfg.rotation)
    reach = (1 - scale).clamp_min(0)[:, None]
    center = (torch.rand(n, 2, generator=gen) * 2 - 1) * reach
    tilt = (torch.rand(n, 2, generator=gen) * 2 - 1) * cfg.perspective
    return Placement(scale, angle, center, tilt)


def sample_photometric(n: int, cfg: EOTConfig, gen: torch.Generator | None = None) -> Photometric:
    return Photometric(
        gain=1 + (torch.rand(n, 3, generator=gen) * 2 - 1) * cfg.print_gain,
        offset=(torch.rand(n, 3, generator=gen) * 2 - 1) * cfg.print_offset,
        brightness=_uniform(n, *cfg.brightness, gen),
        contrast=_uniform(n, *cfg.contrast, gen),
        noise_std=torch.rand(n, generator=gen) * cfg.noise_std,
        blur_sigma=float(_uniform(1, *cfg.blur_sigma, gen)),
    )


def neutral_photometric(n: int) -> Photometric:
    """No print/camera distortion (used as the baseline in evaluation sweeps)."""
    return Photometric(
        gain=torch.ones(n, 3), offset=torch.zeros(n, 3), brightness=torch.ones(n),
        contrast=torch.ones(n), noise_std=torch.zeros(n), blur_sigma=0.0,
    )


def _to(t: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return t.to(device=ref.device, dtype=ref.dtype)


def apply_print(rgb: torch.Tensor, p: Photometric, gamut: tuple[float, float]) -> torch.Tensor:
    """Simulate printing: color drift, then compression into the printable range."""
    lo, hi = gamut
    rgb = rgb * _to(p.gain, rgb)[:, :, None, None] + _to(p.offset, rgb)[:, :, None, None]
    return lo + (hi - lo) * rgb.clamp(0, 1)


def _inverse_homography(pl: Placement, ref: torch.Tensor) -> torch.Tensor:
    """3x3 maps from scene coords back to sticker coords, one per image.

    Forward map (sticker -> scene) is T @ R @ S @ P where P adds the perspective
    tilt; we build its inverse analytically: P^-1 @ S^-1 @ R^-1 @ T^-1.
    """
    n = pl.scale.shape[0]
    s, a = _to(pl.scale, ref), _to(pl.angle, ref)
    c, t = _to(pl.center, ref), _to(pl.tilt, ref)
    eye = torch.eye(3, device=ref.device, dtype=ref.dtype).expand(n, 3, 3)

    t_inv = eye.clone()
    t_inv[:, 0, 2], t_inv[:, 1, 2] = -c[:, 0], -c[:, 1]
    r_inv = eye.clone()
    cos, sin = torch.cos(a), torch.sin(a)
    r_inv[:, 0, 0], r_inv[:, 0, 1], r_inv[:, 1, 0], r_inv[:, 1, 1] = cos, sin, -sin, cos
    s_inv = eye.clone()
    s_inv[:, 0, 0], s_inv[:, 1, 1] = 1 / s, 1 / s
    p_inv = eye.clone()
    p_inv[:, 2, 0], p_inv[:, 2, 1] = -t[:, 0], -t[:, 1]
    return p_inv @ s_inv @ r_inv @ t_inv


def place(
    background: torch.Tensor, rgb: torch.Tensor, alpha: torch.Tensor, pl: Placement
) -> tuple[torch.Tensor, torch.Tensor]:
    """Warp the sticker into each background. Returns (scenes, scene_alpha).

    ``rgb``/``alpha`` may have batch size 1 (broadcast) or B (per-image print jitter).
    """
    b, _, h, w = background.shape
    # Anti-alias: shrink the sticker close to its on-screen size before warping so the
    # optimizer can't rely on high-frequency detail a camera would never resolve.
    target = int(math.ceil(float(pl.scale.max()) * max(h, w) * 1.5))
    if target < rgb.shape[-1]:
        size = (target, target)
        rgb = F.interpolate(rgb, size=size, mode="bilinear", align_corners=False, antialias=True)
        alpha = F.interpolate(alpha, size=size, mode="bilinear", align_corners=False, antialias=True)
    rgb, alpha = rgb.expand(b, -1, -1, -1), alpha.expand(b, -1, -1, -1)

    ys = (torch.arange(h, device=background.device, dtype=background.dtype) + 0.5) / h * 2 - 1
    xs = (torch.arange(w, device=background.device, dtype=background.dtype) + 0.5) / w * 2 - 1
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    pts = torch.stack([gx, gy, torch.ones_like(gx)], dim=-1).reshape(1, -1, 3)  # (1, HW, 3)
    src = pts @ _inverse_homography(pl, background).transpose(1, 2)  # (B, HW, 3)
    denom = src[..., 2:3]
    denom = torch.where(denom.abs() < 1e-6, torch.full_like(denom, 1e-6), denom)
    grid = (src[..., :2] / denom).reshape(b, h, w, 2)

    warped = F.grid_sample(rgb, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    warped_alpha = F.grid_sample(alpha, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    scenes = background * (1 - warped_alpha) + warped * warped_alpha
    return scenes, warped_alpha


def _gaussian_kernel(sigma: float, ref: torch.Tensor) -> torch.Tensor:
    radius = max(1, int(math.ceil(3 * sigma)))
    x = torch.arange(-radius, radius + 1, device=ref.device, dtype=ref.dtype)
    k = torch.exp(-(x**2) / (2 * sigma**2))
    return k / k.sum()


def gaussian_blur(images: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma < 0.3:
        return images
    k = _gaussian_kernel(sigma, images)
    c = images.shape[1]
    pad = k.numel() // 2
    x = F.pad(images, (pad, pad, 0, 0), mode="reflect")
    x = F.conv2d(x, k.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)
    x = F.pad(x, (0, 0, pad, pad), mode="reflect")
    return F.conv2d(x, k.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)


def apply_camera(
    scenes: torch.Tensor, p: Photometric, gen: torch.Generator | None = None
) -> torch.Tensor:
    x = gaussian_blur(scenes, p.blur_sigma)
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - mean) * _to(p.contrast, x)[:, None, None, None] + mean
    x = x * _to(p.brightness, x)[:, None, None, None]
    if float(p.noise_std.max()) > 0:
        noise = torch.randn(x.shape, generator=gen).to(x) if gen is not None else torch.randn_like(x)
        x = x + noise * _to(p.noise_std, x)[:, None, None, None]
    return x.clamp(0, 1)


def render_scenes(
    background: torch.Tensor,
    rgb: torch.Tensor,
    alpha: torch.Tensor,
    placement: Placement,
    photometric: Photometric,
    gamut: tuple[float, float] = (0.0, 1.0),
    gen: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Full print -> place -> camera simulation. Returns (scenes, scene_alpha)."""
    printed = apply_print(rgb.expand(background.shape[0], -1, -1, -1), photometric, gamut)
    scenes, scene_alpha = place(background, printed, alpha, placement)
    return apply_camera(scenes, photometric, gen), scene_alpha
