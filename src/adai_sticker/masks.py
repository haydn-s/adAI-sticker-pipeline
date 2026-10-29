"""Sticker outlines (shape masks) and editable-region masks.

All masks are float tensors of shape (H, W) in [0, 1], rendered with 4x
supersampling so edges are anti-aliased at any resolution. Because shapes are
analytic, the same design can be re-rendered crisply at print resolution.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

SHAPES = ("circle", "square", "rounded", "hexagon", "star", "heart")
REGION_PRESETS = ("border", "center", "stripes", "corners")

_SUPERSAMPLE = 4


def _grid(size: int) -> tuple[np.ndarray, np.ndarray]:
    """Supersampled pixel-center coordinates in [-1, 1], y pointing down."""
    n = size * _SUPERSAMPLE
    coords = (np.arange(n) + 0.5) / n * 2 - 1
    return np.meshgrid(coords, coords, indexing="xy")


def _downsample(inside: np.ndarray, size: int) -> torch.Tensor:
    t = torch.from_numpy(inside.astype(np.float32))[None, None]
    return F.avg_pool2d(t, _SUPERSAMPLE)[0, 0].reshape(size, size)


def _point_in_polygon(x: np.ndarray, y: np.ndarray, verts: list[tuple[float, float]]) -> np.ndarray:
    """Even-odd rule point-in-polygon test, vectorized over a grid."""
    inside = np.zeros_like(x, dtype=bool)
    n = len(verts)
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n]
        crosses = (y1 > y) != (y2 > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_int = (x2 - x1) * (y - y1) / (y2 - y1) + x1
        inside ^= crosses & (x < x_int)
    return inside


def _regular_polygon(n: int, radius: float, phase: float) -> list[tuple[float, float]]:
    return [
        (radius * math.cos(phase + 2 * math.pi * k / n), radius * math.sin(phase + 2 * math.pi * k / n))
        for k in range(n)
    ]


def shape_mask(shape: str, size: int) -> torch.Tensor:
    """Render a named sticker outline at ``size`` x ``size`` pixels."""
    x, y = _grid(size)
    if shape == "circle":
        inside = x**2 + y**2 <= 0.98**2
    elif shape == "square":
        inside = np.ones_like(x, dtype=bool)
    elif shape == "rounded":
        r, half = 0.3, 0.98
        dx = np.maximum(np.abs(x) - (half - r), 0)
        dy = np.maximum(np.abs(y) - (half - r), 0)
        inside = (dx**2 + dy**2 <= r**2) & (np.abs(x) <= half) & (np.abs(y) <= half)
    elif shape == "hexagon":
        inside = _point_in_polygon(x, y, _regular_polygon(6, 0.98, -math.pi / 2))
    elif shape == "star":
        verts = []
        for k in range(10):
            radius = 0.98 if k % 2 == 0 else 0.45
            angle = -math.pi / 2 + math.pi * k / 5
            verts.append((radius * math.cos(angle), radius * math.sin(angle) + 0.06))
        inside = _point_in_polygon(x, y, verts)
    elif shape == "heart":
        # Implicit heart curve (x^2 + y^2 - 1)^3 - x^2 y^3 <= 0, flipped and fit to the box.
        hx, hy = x * 1.2, -y * 1.2 + 0.12
        inside = (hx**2 + hy**2 - 1) ** 3 - hx**2 * hy**3 <= 0
    else:
        raise ValueError(f"Unknown shape '{shape}'. Choose from: {', '.join(SHAPES)}")
    return _downsample(inside, size)


def mask_from_image(path: str | Path, size: int) -> torch.Tensor:
    """Load a custom outline: the alpha channel if present, else bright = inside."""
    img = Image.open(path)
    if img.mode in ("RGBA", "LA") or "transparency" in img.info:
        channel = img.convert("RGBA").getchannel("A")
    else:
        channel = img.convert("L")
    channel = channel.resize((size, size), Image.Resampling.LANCZOS)
    return torch.from_numpy(np.asarray(channel, dtype=np.float32) / 255.0)


def load_mask(shape_or_path: str, size: int) -> torch.Tensor:
    if shape_or_path in SHAPES:
        return shape_mask(shape_or_path, size)
    if Path(shape_or_path).exists():
        return mask_from_image(shape_or_path, size)
    raise ValueError(
        f"'{shape_or_path}' is neither a built-in shape ({', '.join(SHAPES)}) nor an image file."
    )


def region_mask(preset_or_path: str, size: int, amount: float = 0.3) -> torch.Tensor:
    """Where adversarial pixels may go in *region* mode (1 = editable).

    ``amount`` controls how much of the sticker the preset frees up (its stealth
    knob): border width, center radius, stripe duty cycle, or corner size.
    """
    if preset_or_path not in REGION_PRESETS:
        if Path(preset_or_path).exists():
            return mask_from_image(preset_or_path, size)
        raise ValueError(
            f"'{preset_or_path}' is neither a region preset ({', '.join(REGION_PRESETS)}) "
            "nor an image file."
        )
    if not 0 < amount <= 1:
        raise ValueError("Region amount must be in (0, 1].")
    x, y = _grid(size)
    r = np.sqrt(x**2 + y**2)
    if preset_or_path == "border":
        # Ring hugging the outline: everything farther than (1 - amount) from the center,
        # using the max of radial and box distance so it follows circles and squares alike.
        inside = np.maximum(r, np.maximum(np.abs(x), np.abs(y)) * 0.98) >= (1 - amount)
    elif preset_or_path == "center":
        inside = r <= amount
    elif preset_or_path == "stripes":
        period = 0.25
        inside = ((x + y) / math.sqrt(2) / period) % 1.0 < amount
    else:  # corners
        inside = (np.abs(x) >= 1 - amount) & (np.abs(y) >= 1 - amount)
        inside |= r >= 1 - amount * 0.5
    return _downsample(inside, size)
