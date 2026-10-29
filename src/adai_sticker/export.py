"""Turn a finished sticker into files people can print, share, and inspect."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from .sticker import Sticker

PRINT_DPI = 300
LETTER_IN = (8.5, 11.0)


def to_pil_rgba(rgb: torch.Tensor, alpha: torch.Tensor) -> Image.Image:
    """(1,3,H,W) + (1,1,H,W) tensors in [0,1] -> RGBA PIL image."""
    rgba = torch.cat([rgb[0], alpha[0]], 0).clamp(0, 1)
    arr = (rgba.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(arr, mode="RGBA")


def to_pil_rgb(img: torch.Tensor) -> Image.Image:
    """(3,H,W) tensor in [0,1] -> RGB PIL image."""
    arr = (img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def save_sticker_images(sticker: Sticker, out_dir: Path, print_inches: float = 3.0) -> dict[str, str]:
    """Save the sticker at optimization resolution and at print resolution (PNG with alpha)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rgb, alpha = sticker.render()
    small = out_dir / "sticker.png"
    to_pil_rgba(rgb, alpha).save(small)
    px = int(print_inches * PRINT_DPI)
    rgb_hi, alpha_hi = sticker.render(px)
    hi = out_dir / f"sticker_{px}px.png"
    to_pil_rgba(rgb_hi, alpha_hi).save(hi, dpi=(PRINT_DPI, PRINT_DPI))
    return {"png": small.name, "png_print": hi.name}


def _cut_outline(alpha: torch.Tensor, offset_px: int, width_px: int) -> np.ndarray:
    """A ring just outside the sticker's outline, used as the cut line."""
    a = (alpha > 0.5).float()
    outer = F.max_pool2d(a, 2 * (offset_px + width_px) + 1, stride=1, padding=offset_px + width_px)
    inner = F.max_pool2d(a, 2 * offset_px + 1, stride=1, padding=offset_px)
    return ((outer - inner)[0, 0] > 0).cpu().numpy()


def print_sheet(
    sticker: Sticker,
    path: Path,
    title: str,
    subtitle: str,
    sizes_in: tuple[float, ...] = (4.0, 3.0, 2.0, 2.0, 1.5, 1.5),
) -> Path:
    """A US-Letter PDF with the sticker at several physical sizes and cut lines.

    Different sizes matter physically: a sticker only works when it covers
    enough of the camera's view, so the sheet lets people test distance vs. size.
    """
    page_w, page_h = int(LETTER_IN[0] * PRINT_DPI), int(LETTER_IN[1] * PRINT_DPI)
    page = Image.new("RGB", (page_w, page_h), "white")
    draw = ImageDraw.Draw(page)
    margin, gap = int(0.5 * PRINT_DPI), int(0.35 * PRINT_DPI)
    title_font = ImageFont.load_default(size=60)
    body_font = ImageFont.load_default(size=34)
    small_font = ImageFont.load_default(size=28)

    draw.text((margin, margin), title, fill="#0b0b0b", font=title_font)
    draw.text((margin, margin + 80), subtitle, fill="#52514e", font=body_font)
    y = margin + 170
    x = margin
    row_h = 0
    offset = int(0.04 * PRINT_DPI)
    pad = offset + 4  # room for the cut line outside the sticker's bounds
    rendered: dict[int, tuple[Image.Image, Image.Image]] = {}
    for size in sizes_in:
        px = int(size * PRINT_DPI)
        if px not in rendered:
            rgb, alpha = sticker.render(px)
            ring = _cut_outline(F.pad(alpha, (pad, pad, pad, pad)), offset_px=offset, width_px=3)
            ring_mask = Image.fromarray((ring * 255).astype(np.uint8), mode="L")
            rendered[px] = (to_pil_rgba(rgb, alpha), ring_mask)
        img, ring_mask = rendered[px]
        if x + px + 2 * pad > page_w - margin:
            x, y, row_h = margin, y + row_h + gap, 0
        if y + px + 2 * pad + 50 > page_h - margin - 120:
            break
        page.paste(img, (x + pad, y + pad), img)
        page.paste(Image.new("RGB", ring_mask.size, "#9a9996"), (x, y), ring_mask)
        draw.text((x + pad, y + px + 2 * pad + 8), f'{size:g}" sticker', fill="#52514e", font=small_font)
        x += px + 2 * pad + gap
        row_h = max(row_h, px + 2 * pad + 50)

    notes = [
        "Print at 100% / actual size (turn off 'fit to page'). Cut along the gray lines.",
        "Made with adAI Sticker Lab for education and research on adversarial AI.",
        "Do not use stickers to deceive safety-critical systems or to harm, harass, or misidentify people.",
    ]
    for i, line in enumerate(notes):
        draw.text((margin, page_h - margin - 110 + i * 40), line, fill="#52514e", font=small_font)
    page.save(path, "PDF", resolution=PRINT_DPI)
    return path


def scene_grid(
    scenes: torch.Tensor, captions: list[str], hits: list[bool], path: Path, cols: int = 4
) -> Path:
    """Example photos with the sticker in place, captioned with the model's prediction."""
    n = scenes.shape[0]
    rows = (n + cols - 1) // cols
    size = scenes.shape[-1]
    cap_h = 46
    sheet = Image.new("RGB", (cols * (size + 8) + 8, rows * (size + cap_h + 8) + 8), "#fcfcfb")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=14)
    bold = ImageFont.load_default(size=15)
    for i in range(n):
        r, c = divmod(i, cols)
        x, y = 8 + c * (size + 8), 8 + r * (size + cap_h + 8)
        sheet.paste(to_pil_rgb(scenes[i]), (x, y))
        # Status shown by word and color together, never color alone.
        status, color = ("FOOLED", "#008300") if hits[i] else ("not fooled", "#d03b3b")
        draw.rounded_rectangle((x, y + size + 6, x + 10, y + size + 16), radius=2, fill=color)
        draw.text((x + 16, y + size + 3), status, fill="#0b0b0b", font=bold)
        draw.text((x + 16, y + size + 22), f"model says: {captions[i]}", fill="#52514e", font=font)
    sheet.save(path)
    return path
