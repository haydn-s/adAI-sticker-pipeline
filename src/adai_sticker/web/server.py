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
from PIL import Image
from pydantic import BaseModel, Field

from ..labels import FEATURED_TARGETS, search_labels
from ..masks import REGION_PRESETS, SHAPES
from ..models import MODEL_INFO, available_models
from ..palette import PALETTES
from ..pipeline import ETHICS_NOTICE
from .common import (
    MAX_UPLOAD_BYTES,
    PRESETS,
    Choices,
    blocked_categories,
    build_request,
    check_target,
    design_preview,
    gallery_items,
    read_photo,
    save_cover,
    shape_icon,
    validate,
)
from .jobs import JobManager, QueueFullError

STATIC = Path(__file__).parent / "static"


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

    def choices(self) -> Choices:
        return Choices(**self.model_dump(exclude={"ethics_ack"}))


class TargetCheck(BaseModel):
    target: str
    model: str = "resnet50"


async def _read_image(upload: UploadFile) -> Image.Image:
    try:
        return read_photo(await upload.read(MAX_UPLOAD_BYTES + 1))
    except ValueError as err:
        too_big = "smaller than" in str(err)
        raise HTTPException(413 if too_big else 400, str(err)) from err


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
            "blocked_categories": blocked_categories(),
        }

    @app.get("/api/labels")
    def labels(q: str = "", limit: int = 12) -> dict:
        if not q.strip():
            return {"results": []}
        return {"results": [{"index": i, "label": n} for i, n in search_labels(q, limit=min(limit, 50))]}

    @app.post("/api/check-target")
    def check_target_route(body: TargetCheck) -> dict:
        return check_target(body.target, body.model)

    @app.get("/api/shape/{name}.png")
    def shape_png(name: str) -> Response:
        if name not in SHAPES:
            raise HTTPException(404)
        return _png(shape_icon(name))

    @app.post("/api/design-preview")
    async def design_preview_route(
        shape: str = Form("circle"),
        cover_mode: str = Form("none"),
        region: str = Form("stripes"),
        region_amount: float = Form(0.4),
        cover: UploadFile | None = File(None),
    ) -> Response:
        """Show which pixels the optimizer may change: hatched = adversarial."""
        if shape not in SHAPES or region not in REGION_PRESETS:
            raise HTTPException(400, "Unknown shape or region.")
        img = await _read_image(cover) if cover is not None and cover_mode != "none" else None
        return _png(design_preview(shape, cover_mode, region, region_amount, img))

    @app.post("/api/jobs")
    async def create_job(settings: str = Form(...), cover: UploadFile | None = File(None)) -> dict:
        if jobs is None:
            raise HTTPException(403, "This deployment shows the gallery only; making stickers is turned off.")
        try:
            s = StickerSettings(**json.loads(settings))
        except (ValueError, TypeError) as err:
            raise HTTPException(400, f"Invalid settings: {err}") from err
        if not s.ethics_ack:
            raise HTTPException(400, "Please read and accept the responsible-use guidance first.")
        choices = s.choices()
        problem = validate(choices, cover is not None, presets)
        if problem:
            raise HTTPException(400, problem)

        job_id, run_dir = jobs.new_job_dir()
        cover_path = None
        if cover is not None and (s.cover_mode != "none" or s.palette == "auto"):
            cover_path = save_cover(await _read_image(cover), run_dir)
        request = build_request(choices, presets[s.preset], cover_path, backgrounds)
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
        mounts = {runs_root: "runs", **({Path(gallery_root): "gallery"} if gallery_root else {})}
        items = gallery_items(list(mounts))
        for it in items:
            it["base"] = f"/{mounts[it.pop('root')]}/{it.pop('dir').name}/"
        return {"items": items}

    app.mount("/runs", StaticFiles(directory=runs_root), name="runs")
    if gallery_root is not None:
        app.mount("/gallery", StaticFiles(directory=gallery_root), name="gallery")

    @app.exception_handler(HTTPException)
    async def http_error(_request, exc: HTTPException):
        return JSONResponse({"error": exc.detail}, status_code=exc.status_code)

    app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
    return app
