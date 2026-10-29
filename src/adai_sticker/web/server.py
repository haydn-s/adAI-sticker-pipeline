"""adAI Sticker Lab web app: a guided, six-step sticker builder for non-experts.

Run with ``adai-sticker serve`` (needs ``uv sync --extra web``). The API is small:

    GET  /api/options            models, featured targets, shapes, palettes, regions, presets
    GET  /api/labels?q=          search target labels
    POST /api/check-target       safety-policy check with a plain-language reason
    POST /api/design-preview     PNG preview of a design (cover photo + editable region)
    POST /api/jobs               start a sticker (multipart: JSON settings + optional cover)
    GET  /api/jobs/{id}          progress, live success curve, status
    GET  /api/jobs/{id}/preview.png   the sticker as it is being optimized
    GET  /api/gallery            finished stickers available to browse
    /runs/<id>/...               result files (report.json, PNGs, PDF, ...)

Safety: the target policy is enforced by the pipeline itself; the API also
requires an explicit ethics acknowledgement before any sticker is made.
"""

from __future__ import annotations

import io
import json
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from ..evaluate import EvalSettings
from ..export import to_pil_rgba
from ..labels import (
    BLOCK_RATIONALE,
    BLOCKED_IMAGENET,
    FEATURED_TARGETS,
    BlockedTargetError,
    resolve_target,
    search_labels,
)
from ..masks import REGION_PRESETS, SHAPES, region_mask, shape_mask
from ..models import MODEL_INFO, available_models, supports_free_text
from ..optimize import AttackConfig
from ..palette import PALETTES
from ..pipeline import ETHICS_NOTICE, StickerRequest
from ..sticker import StickerDesign, load_image
from .jobs import JobManager, QueueFullError

STATIC = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

# Presets keep run time predictable for a live audience.
PRESETS = {
    "quick": {"steps": 400, "eval_images": 150, "grid_images": 40, "label": "Quick (~2 min)"},
    "standard": {"steps": 1000, "eval_images": 300, "grid_images": 100, "label": "Standard (~4 min)"},
}


class StickerSettings(BaseModel):
    """What the browser sends to start a sticker."""

    model: str = "resnet50"
    target: str
    shape: str = "circle"
    palette: str | None = None
    cover_mode: str = "none"
    region: str = "stripes"
    region_amount: float = Field(0.4, gt=0, le=1)
    epsilon: float = Field(0.1, gt=0, le=0.5)
    preset: str = "quick"
    test_other_models: bool = True
    ethics_ack: bool = False


class TargetCheck(BaseModel):
    target: str
    model: str = "resnet50"


def _validate(settings: StickerSettings, has_cover: bool) -> None:
    if not settings.ethics_ack:
        raise HTTPException(400, "Please read and accept the responsible-use guidance first.")
    if settings.model not in MODEL_INFO:
        raise HTTPException(400, f"Unknown model '{settings.model}'.")
    if settings.shape not in SHAPES:
        raise HTTPException(400, f"Unknown shape '{settings.shape}'.")
    if settings.palette not in (None, "auto", *PALETTES):
        raise HTTPException(400, f"Unknown palette '{settings.palette}'.")
    if settings.cover_mode not in ("none", "region", "blend"):
        raise HTTPException(400, f"Unknown cover mode '{settings.cover_mode}'.")
    if settings.cover_mode != "none" and not has_cover:
        raise HTTPException(400, "Upload a photo to make the sticker look normal.")
    if settings.palette == "auto" and not has_cover:
        raise HTTPException(400, "'Match my photo' colors need a photo.")
    if settings.region not in REGION_PRESETS:
        raise HTTPException(400, f"Unknown region '{settings.region}'.")


async def _read_image(upload: UploadFile) -> Image.Image:
    data = await upload.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Photos must be smaller than 10 MB.")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError) as err:
        raise HTTPException(400, "That file doesn't look like a photo (JPEG or PNG).") from err
    return img.convert("RGB")


def _png(img: Image.Image) -> Response:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return Response(buf.getvalue(), media_type="image/png")


def create_app(
    runs_root: Path = Path("runs/web"),
    gallery_root: Path | None = None,
    allow_generate: bool = True,
    device: str | None = None,
    manager: JobManager | None = None,
    backgrounds: str | None = None,
    presets: dict | None = None,
) -> FastAPI:
    """Build the app. ``backgrounds`` and ``presets`` override the photo source and run lengths."""
    app = FastAPI(title="adAI Sticker Lab", docs_url="/api/docs", redoc_url=None)
    presets = presets or PRESETS
    runs_root = Path(runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)
    jobs = manager or (JobManager(runs_root, device) if allow_generate else None)

    @app.get("/api/options")
    def options() -> dict:
        return {
            "allow_generate": jobs is not None,
            "ethics_notice": ETHICS_NOTICE,
            "models": [asdict(MODEL_INFO[k]) for k in available_models()],
            "featured_targets": list(FEATURED_TARGETS),
            "shapes": list(SHAPES),
            "palettes": PALETTES,
            "regions": list(REGION_PRESETS),
            "presets": {k: v["label"] for k, v in presets.items()},
            "blocked_categories": [
                {"category": c, "reason": BLOCK_RATIONALE[c], "examples": list(e.values())[:6]}
                for c, e in BLOCKED_IMAGENET.items()
            ],
        }

    @app.get("/api/labels")
    def labels(q: str = "", limit: int = 12) -> dict:
        if not q.strip():
            return {"results": []}
        return {"results": [{"index": i, "label": n} for i, n in search_labels(q, limit=min(limit, 50))]}

    @app.post("/api/check-target")
    def check_target(body: TargetCheck) -> dict:
        free_ok = body.model in MODEL_INFO and supports_free_text(body.model)
        try:
            target = resolve_target(body.target, allow_free_text=free_ok)
        except BlockedTargetError as err:
            return {"allowed": False, "category": err.decision.category, "reason": err.decision.reason}
        except ValueError as err:
            return {"allowed": False, "category": "unknown", "reason": str(err)}
        return {"allowed": True, "label": target.label, "free_text": target.is_free_text}

    @app.get("/api/shape/{name}.png")
    def shape_png(name: str) -> Response:
        if name not in SHAPES:
            raise HTTPException(404)
        m = shape_mask(name, 96)
        rgba = Image.new("RGBA", (96, 96), (42, 120, 214, 0))
        rgba.putalpha(Image.fromarray((m.numpy() * 255).astype("uint8")))
        return _png(rgba)

    @app.post("/api/design-preview")
    async def design_preview(
        shape: str = Form("circle"),
        cover_mode: str = Form("none"),
        region: str = Form("stripes"),
        region_amount: float = Form(0.4),
        cover: UploadFile | None = File(None),
    ) -> Response:
        """Show which pixels the optimizer may change: hatched = adversarial."""
        if shape not in SHAPES or region not in REGION_PRESETS:
            raise HTTPException(400, "Unknown shape or region.")
        size = 256
        import numpy as np
        import torch

        mask = shape_mask(shape, size)
        if cover is not None and cover_mode != "none":
            img = await _read_image(cover)
            tmp = io.BytesIO()
            img.save(tmp, format="PNG")
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
        return _png(to_pil_rgba(out[None], mask[None, None]))

    @app.post("/api/jobs")
    async def create_job(settings: str = Form(...), cover: UploadFile | None = File(None)) -> dict:
        if jobs is None:
            raise HTTPException(403, "This deployment shows the gallery only; making stickers is turned off.")
        try:
            s = StickerSettings(**json.loads(settings))
        except (ValueError, TypeError) as err:
            raise HTTPException(400, f"Invalid settings: {err}") from err
        _validate(s, cover is not None)
        if s.preset not in presets:
            raise HTTPException(400, f"Unknown preset '{s.preset}'.")
        check = check_target(TargetCheck(target=s.target, model=s.model))
        if not check["allowed"]:
            raise HTTPException(400, check["reason"])

        job_id, run_dir = jobs.new_job_dir()
        cover_path = None
        if cover is not None and (s.cover_mode != "none" or s.palette == "auto"):
            img = await _read_image(cover)
            img.thumbnail((1600, 1600))
            cover_path = run_dir / "upload.jpg"
            img.save(cover_path, quality=92)
        preset = presets[s.preset]
        others = [m for m in available_models() if m != s.model] if s.test_other_models else []
        request = StickerRequest(
            target=s.target,
            models=[s.model],
            design=StickerDesign(
                shape=s.shape, palette=s.palette,
                cover_mode=s.cover_mode, cover_image=str(cover_path) if cover_path else None,
                region=s.region, region_amount=s.region_amount, epsilon=s.epsilon,
            ),
            attack=AttackConfig(steps=preset["steps"], **preset.get("attack", {})),
            eval_models=others,
            eval=EvalSettings(n_images=preset["eval_images"], grid_images=preset["grid_images"],
                              **preset.get("eval", {})),
            backgrounds=backgrounds,
            train_images=preset.get("train_images", 2000),
        )
        try:
            job = jobs.submit(job_id, run_dir, request)
        except QueueFullError as err:
            raise HTTPException(503, str(err)) from err
        return job.public(jobs.position(job_id))

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str) -> dict:
        job = jobs.get(job_id) if jobs else None
        if job is None:
            done = runs_root / job_id / "job.json"
            if done.exists():
                return json.loads(done.read_text())
            raise HTTPException(404, "No such sticker.")
        return job.public(jobs.position(job_id))

    @app.get("/api/jobs/{job_id}/preview.png")
    def job_preview(job_id: str) -> Response:
        job = jobs.get(job_id) if jobs else None
        if job is None or job.preview_png is None:
            sticker = runs_root / job_id / "sticker.png"
            if sticker.exists():
                return FileResponse(sticker)
            raise HTTPException(404)
        return Response(job.preview_png, media_type="image/png", headers={"Cache-Control": "no-store"})

    @app.get("/api/gallery")
    def gallery() -> dict:
        items = []
        roots = [("runs", runs_root)] + ([("gallery", Path(gallery_root))] if gallery_root else [])
        for mount, root in roots:
            for report in sorted(root.glob("*/report.json"), key=lambda p: p.stat().st_mtime, reverse=True):
                data = json.loads(report.read_text())
                primary = data["models"][0]
                items.append({
                    "id": report.parent.name,
                    "base": f"/{mount}/{report.parent.name}/",
                    "target": data["target"]["label"],
                    "model": MODEL_INFO[primary].display_name if primary in MODEL_INFO else primary,
                    "success_rate": data["evaluation"][primary]["headline"]["success_rate"],
                    "cover_mode": data["design"]["cover_mode"],
                })
        return {"items": items[:60]}

    app.mount("/runs", StaticFiles(directory=runs_root), name="runs")
    if gallery_root is not None:
        app.mount("/gallery", StaticFiles(directory=gallery_root), name="gallery")

    @app.exception_handler(HTTPException)
    async def http_error(_request, exc: HTTPException):
        return JSONResponse({"error": exc.detail}, status_code=exc.status_code)

    app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
    return app
