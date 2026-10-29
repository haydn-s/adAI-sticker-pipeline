"""A single-GPU job queue: one sticker is optimized at a time, in a worker thread.

The web server submits jobs and polls their state; the worker runs the normal
pipeline (:func:`adai_sticker.pipeline.make_sticker`) and records progress, a
live preview of the sticker, and the final summary.
"""

from __future__ import annotations

import io
import json
import queue
import threading
import time
import traceback
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import torch

from ..export import to_pil_rgba
from ..optimize import StepInfo
from ..pipeline import StickerRequest, make_sticker
from ..sticker import Sticker

PREVIEW_EVERY = 10  # steps between live preview renders


class QueueFullError(RuntimeError):
    pass


@dataclass
class Job:
    id: str
    request: StickerRequest
    run_dir: Path
    status: str = "queued"  # queued | optimizing | evaluating | done | failed
    step: int = 0
    total_steps: int = 0
    success_curve: list[tuple[int, float]] = field(default_factory=list)
    message: str = "Waiting for the GPU"
    error: str | None = None
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    preview_png: bytes | None = None
    _sticker: Sticker | None = None

    def public(self, position: int | None = None) -> dict:
        d = {
            "id": self.id,
            "status": self.status,
            "step": self.step,
            "total_steps": self.total_steps,
            "success_curve": self.success_curve[-200:],
            "message": self.message,
            "error": self.error,
            "elapsed": (self.finished or time.time()) - (self.started or time.time()),
            "target": self.request.target,
            "models": self.request.models,
        }
        if position is not None:
            d["queue_position"] = position
        if self.status == "done":
            d["results"] = f"/runs/{self.id}/"
        return d


class JobManager:
    def __init__(
        self,
        runs_root: Path,
        device: torch.device | str | None = None,
        max_queue: int = 8,
        make_fn: Callable[..., dict] = make_sticker,
    ):
        self.runs_root = Path(runs_root)
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.make_fn = make_fn
        self.jobs: dict[str, Job] = {}
        self._queue: queue.Queue[Job] = queue.Queue(maxsize=max_queue)
        self._order: list[str] = []
        self._worker = threading.Thread(target=self._work, daemon=True, name="sticker-worker")
        self._worker.start()

    def new_job_dir(self) -> tuple[str, Path]:
        job_id = uuid.uuid4().hex[:16]
        run_dir = self.runs_root / job_id
        run_dir.mkdir(parents=True)
        return job_id, run_dir

    def submit(self, job_id: str, run_dir: Path, request: StickerRequest) -> Job:
        job = Job(job_id, request, run_dir, total_steps=request.attack.steps)
        try:
            self._queue.put_nowait(job)
        except queue.Full as err:
            raise QueueFullError("The sticker queue is full; please try again in a few minutes.") from err
        self.jobs[job_id] = job
        self._order.append(job_id)
        return job

    def position(self, job_id: str) -> int | None:
        waiting = [j for j in self._order if self.jobs[j].status == "queued"]
        return waiting.index(job_id) + 1 if job_id in waiting else None

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    # -- worker ------------------------------------------------------------

    def _work(self) -> None:
        while True:
            job = self._queue.get()
            try:
                self._run(job)
            finally:
                self._queue.task_done()

    def _run(self, job: Job) -> None:
        job.status = "optimizing"
        job.started = time.time()
        job.message = "Loading the model and practice photos"

        def on_start(sticker: Sticker) -> None:
            job._sticker = sticker
            self._render_preview(job)
            job.message = "Testing the sticker on practice photos and improving it"

        def on_step(info: StepInfo) -> None:
            job.step = info.step
            if info.step % PREVIEW_EVERY == 0 or info.step == info.total_steps:
                job.success_curve.append((info.step, info.success_rate))
                self._render_preview(job)
            if info.step == info.total_steps:
                job.status = "evaluating"
                job.message = "Measuring how well it works on photos it has never seen"

        def say(msg: str) -> None:
            if job.status == "evaluating":
                job.message = _friendly(msg)

        try:
            self.make_fn(job.request, job.run_dir, device=self.device, say=say, on_step=on_step,
                         on_start=on_start)
            job.status = "done"
            job.message = "Done"
        except Exception as err:  # noqa: BLE001 - report any failure to the user
            job.status = "failed"
            job.error = str(err) or type(err).__name__
            job.message = "Something went wrong"
            (job.run_dir / "error.txt").write_text(traceback.format_exc())
        finally:
            job.finished = time.time()
            job._sticker = None
            (job.run_dir / "job.json").write_text(json.dumps(job.public(), indent=2))

    def _render_preview(self, job: Job) -> None:
        if job._sticker is None:
            return
        rgb, alpha = job._sticker.render()
        buf = io.BytesIO()
        to_pil_rgba(rgb, alpha).save(buf, format="PNG")
        job.preview_png = buf.getvalue()


_FRIENDLY = (
    ("controls", "Comparing against photos with no sticker and a random sticker"),
    ("rotation x scale", "Trying every rotation and size"),
    ("sweep: brightness", "Testing different lighting"),
    ("sweep: contrast", "Testing different contrast"),
    ("sweep: blur", "Testing blurry photos"),
    ("sweep: noise", "Testing grainy photos"),
    ("sweep: perspective", "Testing viewing angles"),
    ("sweep: jpeg", "Testing compressed photos"),
    ("sweep: color_cast", "Testing color tints"),
    ("Grad-CAM", "Finding where the model looks"),
    ("figures", "Drawing the report"),
)


def _friendly(msg: str) -> str:
    for key, text in _FRIENDLY:
        if key in msg:
            model = msg[1:msg.index("]")] if msg.startswith("[") else None
            return f"{text} ({model})" if model and "controls" in key else text
    return msg
