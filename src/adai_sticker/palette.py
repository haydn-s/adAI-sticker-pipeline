"""Color palettes for palette-constrained stickers and the printability penalty."""

from __future__ import annotations

import torch

PALETTES: dict[str, list[str]] = {
    "primary": ["#ffffff", "#111111", "#00a0e0", "#e0007a", "#ffe600", "#e3241b", "#00933c", "#1d3f9a"],
    "sunset": ["#2b1b3d", "#5c2a6b", "#a63a79", "#e3567b", "#f88f6f", "#fcc86d", "#fdf1c9"],
    "ocean": ["#03132b", "#0a3a63", "#11679a", "#2a9fc9", "#7fd3e6", "#d5f4f7", "#f7e9c4"],
    "forest": ["#141f12", "#2f4a26", "#4e7a35", "#8fae4c", "#d7d48f", "#7a5230", "#c99a63"],
    "candy": ["#ff7eb6", "#ffb3d1", "#a685ff", "#7ee0ff", "#b6ffb0", "#fff59c", "#ffffff"],
    "grayscale": ["#0d0d0d", "#3a3a3a", "#666666", "#939393", "#c0c0c0", "#ededed"],
}


def hex_to_rgb(color: str) -> tuple[float, float, float]:
    c = color.strip().lstrip("#")
    if len(c) == 3:
        c = "".join(ch * 2 for ch in c)
    if len(c) != 6:
        raise ValueError(f"Invalid hex color '{color}'.")
    return tuple(int(c[i:i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]


def rgb_to_hex(rgb: torch.Tensor) -> str:
    vals = (rgb.clamp(0, 1) * 255).round().int().tolist()
    return "#" + "".join(f"{v:02x}" for v in vals)


def parse_palette(spec: str | list[str]) -> torch.Tensor:
    """Palette name, comma-separated hex list, or list of hex strings -> (K, 3) tensor."""
    if isinstance(spec, str):
        colors = PALETTES[spec] if spec in PALETTES else [c for c in spec.split(",") if c.strip()]
    else:
        colors = spec
    if len(colors) < 2:
        raise ValueError("A palette needs at least two colors.")
    return torch.tensor([hex_to_rgb(c) for c in colors], dtype=torch.float32)


def palette_from_image(image: torch.Tensor, k: int = 8, iters: int = 20, seed: int = 0) -> torch.Tensor:
    """k-means colors of an image (3, H, W) — lets adversarial regions match a cover photo."""
    pixels = image.reshape(3, -1).T
    gen = torch.Generator().manual_seed(seed)
    if pixels.shape[0] > 20000:
        pixels = pixels[torch.randperm(pixels.shape[0], generator=gen)[:20000]]
    centers = pixels[torch.randperm(pixels.shape[0], generator=gen)[:k]].clone()
    for _ in range(iters):
        assign = torch.cdist(pixels, centers).argmin(1)
        for j in range(k):
            members = pixels[assign == j]
            if len(members):
                centers[j] = members.mean(0)
    return centers


def printable_colors(levels: int = 6, low: float = 0.06, high: float = 0.94) -> torch.Tensor:
    """A proxy for the gamut a consumer printer reproduces reliably.

    Without a measured printer profile we use an RGB lattice compressed away from
    the extremes that inkjets struggle with. Replacing this with colors measured
    from a printed and photographed calibration chart (Sharif et al., 2016) is
    listed as future work.
    """
    axis = torch.linspace(low, high, levels)
    r, g, b = torch.meshgrid(axis, axis, axis, indexing="ij")
    return torch.stack([r.flatten(), g.flatten(), b.flatten()], dim=1)
