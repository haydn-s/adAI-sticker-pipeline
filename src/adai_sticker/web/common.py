"""The parts of the web app that don't depend on a web framework.

Both front ends use these: the FastAPI app (``adai-sticker serve``) and the Streamlit
app (``deploy/streamlit_app.py``). They offer the same choices, run the same safety
checks and send the same requests to the pipeline. Nothing here imports FastAPI, so a
Streamlit deployment doesn't need it.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

from ..evaluate import EvalSettings
from ..export import to_pil_rgba
from ..labels import BLOCK_RATIONALE, BLOCKED_IMAGENET, BlockedTargetError, resolve_target
from ..masks import REGION_PRESETS, SHAPES, region_mask, shape_mask
from ..models import MODEL_INFO, available_models, supports_free_text
from ..optimize import AttackConfig
from ..palette import PALETTES
from ..pipeline import StickerRequest
from ..sticker import StickerDesign, load_image

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
COVER_MODES = ("none", "region", "blend")

# Presets keep run time predictable for a live audience. These assume a GPU (CUDA or
# Apple MPS).
PRESETS = {
    "quick": {"steps": 400, "eval_images": 150, "grid_images": 40, "label": "Quick (~2 min)"},
    "standard": {"steps": 1000, "eval_images": 300, "grid_images": 100, "label": "Standard (~4 min)"},
}

# On two shared CPU cores (Streamlit Community Cloud) the GPU presets would take about
# an hour. The CPU presets train on smaller batches and test on fewer photos, with a
# coarser rotation x size grid and three to four levels per condition sweep.
_CPU_EVAL = {
    "batch_size": 24,
    "grid_angles": (-180, -120, -60, 0, 60, 120, 180),
    "grid_scales": (0.2, 0.3, 0.4, 0.5),
    "sweeps": {
        "brightness": (0.6, 1.0, 1.4),
        "contrast": (0.6, 1.0, 1.4),
        "blur": (0.0, 1.0, 2.0, 3.0),
        "noise": (0.0, 0.04, 0.08),
        "perspective": (0.0, 0.25, 0.5),
        "jpeg": (100, 60, 20),
        "color_cast": (0.0, 0.15, 0.3),
    },
}
CPU_PRESETS = {
    "quick": {
        "steps": 200, "eval_images": 48, "grid_images": 12, "train_images": 600,
        "attack": {"batch_size": 8}, "eval": _CPU_EVAL, "label": "Quick (~5 min, no GPU)",
    },
    "standard": {
        "steps": 400, "eval_images": 96, "grid_images": 24, "train_images": 1000,
        "attack": {"batch_size": 12}, "eval": _CPU_EVAL, "label": "Thorough (~12 min, no GPU)",
    },
}


def presets_for(device: torch.device | str) -> dict:
    """Run-length presets suited to the compute device."""
    return CPU_PRESETS if torch.device(device).type == "cpu" else PRESETS


@dataclass
class Choices:
    """Everything the user picks in the guided flow."""

    model: str = "resnet50"
    target: str = ""
    shape: str = "circle"
    palette: str | None = None
    cover_mode: str = "none"
    region: str = "stripes"
    region_amount: float = 0.4
    epsilon: float = 0.1  # largest per-pixel change in blend mode, in [0, 1] units
    preset: str = "quick"
    test_other_models: bool = True


def check_target(target: str, model: str = "resnet50") -> dict:
    """Apply the target policy, with a plain-language reason when a target is refused."""
    free_ok = model in MODEL_INFO and supports_free_text(model)
    try:
        resolved = resolve_target(target, allow_free_text=free_ok)
    except BlockedTargetError as err:
        return {"allowed": False, "category": err.decision.category, "reason": err.decision.reason}
    except ValueError as err:
        return {"allowed": False, "category": "unknown", "reason": str(err)}
    return {"allowed": True, "label": resolved.label, "free_text": resolved.is_free_text}


def validate(c: Choices, has_cover: bool, presets: dict) -> str | None:
    """What's wrong with these choices, in plain language, or None if they're fine."""
    if c.model not in MODEL_INFO:
        return f"Unknown model '{c.model}'."
    if c.shape not in SHAPES:
        return f"Unknown shape '{c.shape}'."
    if c.palette not in (None, "auto", *PALETTES):
        return f"Unknown palette '{c.palette}'."
    if c.cover_mode not in COVER_MODES:
        return f"Unknown cover mode '{c.cover_mode}'."
    if c.cover_mode != "none" and not has_cover:
        return "Upload a photo to make the sticker look normal."
    if c.palette == "auto" and not has_cover:
        return "'Match my photo' colors need a photo."
    if c.region not in REGION_PRESETS:
        return f"Unknown region '{c.region}'."
    if c.preset not in presets:
        return f"Unknown preset '{c.preset}'."
    check = check_target(c.target, c.model)
    if not check["allowed"]:
        return check["reason"]
    return None


def read_photo(data: bytes) -> Image.Image:
    """Decode an uploaded photo, or raise ValueError with a message for the user."""
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("Photos must be smaller than 10 MB.")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError) as err:
        raise ValueError("That file doesn't look like a photo (JPEG or PNG).") from err
    return img.convert("RGB")


def save_cover(img: Image.Image, run_dir: Path) -> Path:
    """Store the cover photo in the run folder, downsized to a sensible maximum."""
    img = img.copy()
    img.thumbnail((1600, 1600))
    path = Path(run_dir) / "upload.jpg"
    img.save(path, quality=92)
    return path


def build_request(c: Choices, preset: dict, cover_path: Path | None = None,
                  backgrounds: str | None = None) -> StickerRequest:
    """Turn validated choices and a run-length preset into a pipeline request."""
    others = [m for m in available_models() if m != c.model] if c.test_other_models else []
    return StickerRequest(
        target=c.target,
        models=[c.model],
        design=StickerDesign(
            shape=c.shape, palette=c.palette,
            cover_mode=c.cover_mode, cover_image=str(cover_path) if cover_path else None,
            region=c.region, region_amount=c.region_amount, epsilon=c.epsilon,
        ),
        attack=AttackConfig(steps=preset["steps"], **preset.get("attack", {})),
        eval_models=others,
        eval=EvalSettings(n_images=preset["eval_images"], grid_images=preset["grid_images"],
                          **preset.get("eval", {})),
        backgrounds=backgrounds,
        train_images=preset.get("train_images", 2000),
    )


def design_preview(shape: str, cover_mode: str = "none", region: str = "stripes",
                   region_amount: float = 0.4, cover: Image.Image | None = None,
                   size: int = 256) -> Image.Image:
    """Show which pixels the optimizer may change: orange hatching = adversarial."""
    mask = shape_mask(shape, size)
    if cover is not None and cover_mode != "none":
        tmp = io.BytesIO()
        cover.save(tmp, format="PNG")
        tmp.seek(0)
        rgb = load_image(tmp, size)
    else:
        rgb = torch.full((3, size, size), 0.92)
    if cover_mode == "region":
        editable = region_mask(region, size, min(max(region_amount, 0.05), 1.0))
    elif cover_mode == "blend":
        editable = torch.zeros(size, size)
    else:
        editable = torch.ones(size, size)
    yy, xx = np.mgrid[0:size, 0:size]
    hatch = torch.from_numpy(((xx + yy) // 6 % 2).astype("float32"))
    pattern = torch.stack([0.92 - 0.75 * hatch, 0.47 - 0.3 * hatch, 0.2 + 0.0 * hatch])
    out = rgb * (1 - editable) + pattern * editable
    return to_pil_rgba(out[None], mask[None, None])


def shape_icon(name: str, size: int = 96) -> Image.Image:
    """A flat blue silhouette of a sticker shape."""
    m = shape_mask(name, size)
    rgba = Image.new("RGBA", (size, size), (42, 120, 214, 0))
    rgba.putalpha(Image.fromarray((m.numpy() * 255).astype("uint8")))
    return rgba


def blocked_categories() -> list[dict]:
    """Blocked target categories with the reason and a few example labels each."""
    return [
        {"category": c, "reason": BLOCK_RATIONALE[c], "examples": list(e.values())[:6]}
        for c, e in BLOCKED_IMAGENET.items()
    ]


def gallery_items(roots: list[Path], limit: int = 60) -> list[dict]:
    """Finished runs under ``roots``, newest first, with the facts a gallery card shows."""
    items = []
    for root in roots:
        reports = sorted(Path(root).glob("*/report.json"), key=lambda p: p.stat().st_mtime,
                         reverse=True)
        for report in reports:
            data = json.loads(report.read_text())
            primary = data["models"][0]
            items.append({
                "id": report.parent.name,
                "dir": report.parent,
                "root": Path(root),
                "target": data["target"]["label"],
                "model": MODEL_INFO[primary].display_name if primary in MODEL_INFO else primary,
                "success_rate": data["evaluation"][primary]["headline"]["success_rate"],
                "cover_mode": data["design"]["cover_mode"],
            })
    return items[:limit]
