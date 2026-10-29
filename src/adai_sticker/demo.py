"""Live webcam demo: hold up a printed sticker and watch the model change its mind.

Run with ``adai-sticker demo --target toaster`` (needs the ``demo`` extra:
``uv sync --extra demo``). Keys:

    g      toggle the Grad-CAM overlay (where the model is looking)
    m      toggle mirror view
    space  freeze / unfreeze the frame
    s      save the current frame (raw photo + annotated view) for `eval-photos`
    q/esc  quit

Frame processing (:func:`annotate`) is pure NumPy/PIL so it can be tested
without a camera; OpenCV is only used to capture and display.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from matplotlib import colormaps
from PIL import Image, ImageDraw, ImageFont

from .explain import gradcam
from .labels import Target
from .models import VictimModel

SURFACE = (252, 252, 251)
TEXT = (11, 11, 11)
TEXT_2 = (82, 81, 78)
TRACK = (228, 227, 223)
BAR = (42, 120, 214)  # categorical slot 1
TARGET_BAR = (235, 104, 52)  # categorical slot 2
FOOLED = (0, 131, 0)

PANEL_W = 440


@dataclass
class Prediction:
    names: list[str]
    probs: list[float]
    indices: list[int]


def center_square(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    side = min(h, w)
    top, left = (h - side) // 2, (w - side) // 2
    return frame[top:top + side, left:left + side]


def frame_to_tensor(square_rgb: np.ndarray, size: int, device) -> torch.Tensor:
    img = Image.fromarray(square_rgb).resize((size, size), Image.Resampling.BILINEAR)
    t = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)[None]
    return t.to(device)


@torch.no_grad()
def predict(model: VictimModel, x: torch.Tensor, k: int = 5) -> Prediction:
    probs = model(x).float().softmax(-1)[0]
    top = probs.topk(k)
    names = model.class_names
    return Prediction([names[int(i)] for i in top.indices], top.values.tolist(), top.indices.tolist())


def overlay_cam(square_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    heat = (colormaps["magma"](cam)[..., :3] * 255).astype(np.float32)
    heat = np.asarray(Image.fromarray(heat.astype(np.uint8)).resize(square_rgb.shape[1::-1]),
                      dtype=np.float32)
    return (square_rgb * (1 - alpha) + heat * alpha).astype(np.uint8)


def annotate(
    square_rgb: np.ndarray,
    pred: Prediction,
    target: Target | None,
    target_idx: int | None,
    model_name: str,
    fps: float | None = None,
    cam: np.ndarray | None = None,
    view_size: int = 540,
) -> np.ndarray:
    """Compose the camera view and a side panel with the model's top guesses."""
    view = overlay_cam(square_rgb, cam) if cam is not None else square_rgb
    view_img = Image.fromarray(view).resize((view_size, view_size), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (view_size + PANEL_W, view_size), SURFACE)
    canvas.paste(view_img, (0, 0))
    draw = ImageDraw.Draw(canvas)
    big, mid, small = (ImageFont.load_default(size=s) for s in (26, 18, 14))

    x0, y = view_size + 24, 22
    draw.text((x0, y), model_name, fill=TEXT, font=big)
    y += 38
    draw.text((x0, y), "Top 5 guesses", fill=TEXT_2, font=small)
    y += 26
    bar_w = PANEL_W - 48
    for name, p, idx in zip(pred.names, pred.probs, pred.indices, strict=True):
        is_target = target_idx is not None and idx == target_idx
        label = f"{name}  ·  TARGET" if is_target else name
        draw.text((x0, y), label, fill=TEXT, font=mid)
        draw.text((x0 + bar_w - 48, y), f"{p:4.0%}", fill=TEXT, font=mid)
        y += 26
        draw.rounded_rectangle((x0, y, x0 + bar_w, y + 10), radius=4, fill=TRACK)
        if p > 0.005:
            draw.rounded_rectangle((x0, y, x0 + max(10, int(bar_w * p)), y + 10), radius=4,
                                   fill=TARGET_BAR if is_target else BAR)
        y += 26

    if target is not None:
        y += 10
        fooled = target_idx is not None and pred.indices[0] == target_idx
        if fooled:
            draw.rounded_rectangle((x0, y, x0 + bar_w, y + 64), radius=8, fill=FOOLED)
            draw.text((x0 + 14, y + 8), "FOOLED", fill=(255, 255, 255), font=big)
            draw.text((x0 + 14, y + 40), f'the model sees "{target.label}"', fill=(255, 255, 255),
                      font=small)
        else:
            draw.rounded_rectangle((x0, y, x0 + bar_w, y + 64), radius=8, outline=TRACK, width=2)
            draw.text((x0 + 14, y + 8), "Not fooled", fill=TEXT, font=big)
            draw.text((x0 + 14, y + 40), f'target: "{target.label}"', fill=TEXT_2, font=small)

    hints = "g Grad-CAM · m mirror · space freeze · s save · q quit"
    draw.text((x0, view_size - 46), hints, fill=TEXT_2, font=small)
    if fps is not None:
        draw.text((x0, view_size - 24), f"{fps:4.1f} fps", fill=TEXT_2, font=small)
    if cam is not None:
        caption = "Grad-CAM: where the model is looking"
        x1, y1, x2, y2 = draw.textbbox((16, 14), caption, font=mid)
        draw.rounded_rectangle((x1 - 8, y1 - 6, x2 + 8, y2 + 6), radius=6, fill=(11, 11, 11))
        draw.text((16, 14), caption, fill=(255, 255, 255), font=mid)
    return np.asarray(canvas)


def run_demo(
    model: VictimModel,
    target: Target | None,
    camera: int = 0,
    save_dir: Path | None = None,
    show_cam: bool = False,
) -> None:  # pragma: no cover - needs a camera and a display
    import cv2

    device = next(model.parameters()).device
    target_idx = model.class_index(target) if target is not None else None
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {camera}. Check camera permissions for your terminal.")
    save_dir = save_dir or Path("photos") / datetime.now().strftime("%Y%m%d-%H%M%S")
    window = "adAI Sticker Lab"
    mirror, frozen, last = True, False, None
    t_prev, fps = time.time(), None
    try:
        while True:
            if not frozen or last is None:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                square = center_square(frame_bgr[:, :, ::-1])
                if mirror:
                    square = square[:, ::-1]
                square = np.ascontiguousarray(square)
                x = frame_to_tensor(square, model.input_size, device)
                pred = predict(model, x)
                cam = None
                if show_cam:
                    cam = gradcam(model, x, pred.indices[0])[0][0].cpu().numpy()
                now = time.time()
                fps = 1 / max(1e-6, now - t_prev) if fps is None else 0.9 * fps + 0.1 / max(1e-6, now - t_prev)
                t_prev = now
                last = (square, pred, cam)
            square, pred, cam = last
            view = annotate(square, pred, target, target_idx, model.info.display_name, fps, cam)
            cv2.imshow(window, view[:, :, ::-1])
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("g"):
                show_cam = not show_cam
                last = None if not frozen else last
            elif key == ord("m"):
                mirror = not mirror
            elif key == ord(" "):
                frozen = not frozen
            elif key == ord("s"):
                save_dir.mkdir(parents=True, exist_ok=True)
                stamp = datetime.now().strftime("%H%M%S-%f")[:-3]
                Image.fromarray(square).save(save_dir / f"photo-{stamp}.jpg", quality=95)
                (save_dir / "annotated").mkdir(exist_ok=True)  # skipped by eval-photos
                Image.fromarray(view).save(save_dir / "annotated" / f"view-{stamp}.jpg", quality=90)
                print(f"saved {save_dir / f'photo-{stamp}.jpg'}  (model: {pred.names[0]} {pred.probs[0]:.0%})")
    finally:
        cap.release()
        cv2.destroyAllWindows()
