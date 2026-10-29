"""adAI Sticker Lab as a Streamlit app, ready for Streamlit Community Cloud.

The steps, wording and safety checks match the FastAPI app (``adai-sticker serve``); both
use :mod:`adai_sticker.web.common`. A single background worker (:class:`JobManager`)
makes one sticker at a time while the page polls it. A long optimization never blocks
the page, and visitors wait in a queue instead of overloading the server.

Run locally with ``uv run --extra streamlit streamlit run deploy/streamlit_app.py``.
docs/DEPLOY.md covers hosting. Settings come from environment variables (on Community
Cloud, top-level secrets become environment variables):

    ADAI_GALLERY_ONLY=1      only browse finished stickers; nothing is computed
    ADAI_DEVICE=cpu          compute device (default: CUDA, then Apple MPS, then CPU)
    ADAI_RUNS_DIR            where new stickers are written (default runs/streamlit)
    ADAI_GALLERY_DIR         finished examples to show (default docs/examples)
    ADAI_BACKGROUNDS         practice photos: a folder, or "synthetic" (default Imagenette)
    ADAI_MAX_QUEUE=4         how many stickers may wait in line
    ADAI_RUN_TTL_HOURS=24    delete stickers and uploaded photos after this long
    ADAI_SHOW_ALL_RUNS=1     gallery lists every sticker on the server, not just the
                             visitor's own (handy for a live demo; off by default for privacy)
"""

from __future__ import annotations

import html
import json
import math
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import streamlit as st
import torch

from ..data import fetch_imagenette, find_imagenette
from ..labels import FEATURED_TARGETS, imagenet_classes
from ..masks import REGION_PRESETS, SHAPES
from ..models import MODEL_INFO, available_models, default_device, supports_free_text
from ..palette import PALETTES
from ..pipeline import make_sticker
from ..report import SERIES
from .common import (
    COVER_MODES,
    Choices,
    blocked_categories,
    build_request,
    check_target,
    design_preview,
    gallery_items,
    presets_for,
    read_photo,
    save_cover,
    validate,
)
from .jobs import JobManager, QueueFullError

ACCENT = SERIES[0]
RUN_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# Relative training cost on CPU (forward + backward FLOPs), for speed hints without a GPU.
CPU_SPEED = {
    "mobilenet_v3_large": "fast", "efficientnet_b0": "fast", "resnet50": "medium",
    "convnext_tiny": "medium", "clip-vit-b-32": "medium", "vit_b_16": "slow",
}


@dataclass(frozen=True)
class Settings:
    gallery_only: bool
    device: torch.device
    runs_root: Path
    gallery_root: Path
    backgrounds: str | None
    max_queue: int
    ttl_hours: float
    show_all_runs: bool

    @classmethod
    def from_env(cls) -> Settings:
        env = os.environ.get
        flag = lambda name: env(name, "").strip().lower() in ("1", "true", "yes")  # noqa: E731
        return cls(
            gallery_only=flag("ADAI_GALLERY_ONLY"),
            device=torch.device(env("ADAI_DEVICE") or default_device()),
            runs_root=Path(env("ADAI_RUNS_DIR", "runs/streamlit")),
            gallery_root=Path(env("ADAI_GALLERY_DIR", "docs/examples")),
            backgrounds=env("ADAI_BACKGROUNDS") or None,
            max_queue=int(env("ADAI_MAX_QUEUE", "4")),
            ttl_hours=float(env("ADAI_RUN_TTL_HOURS", "24")),
            show_all_runs=flag("ADAI_SHOW_ALL_RUNS"),
        )

    @property
    def presets(self) -> dict:
        return presets_for(self.device)


def _cpu_quota() -> int | None:
    """CPUs this container may use under a cgroup v2 quota, which torch doesn't detect."""
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        return None if quota == "max" else max(1, math.ceil(int(quota) / int(period)))
    except (OSError, ValueError):
        return None


def _make(request, run_dir, say=print, **kwargs) -> dict:
    """:func:`make_sticker`, fetching Imagenette first on a server that doesn't have it yet."""
    if request.backgrounds is None and find_imagenette() is None:
        say("Downloading practice photos (a one-time wait of a minute or two)")
        fetch_imagenette()
    return make_sticker(request, run_dir, say=say, **kwargs)


@st.cache_resource
def _manager() -> JobManager | None:
    """The process-wide sticker queue, shared by every visitor (None in gallery-only mode)."""
    cfg = Settings.from_env()
    if cfg.gallery_only:
        return None
    if cfg.device.type == "cpu":
        threads = os.environ.get("ADAI_TORCH_THREADS") or _cpu_quota()
        if threads:
            torch.set_num_threads(int(threads))
    return JobManager(cfg.runs_root, cfg.device, max_queue=cfg.max_queue, make_fn=_make)


# ---------------------------------------------------------------------------
# Session state and navigation
# ---------------------------------------------------------------------------

# Widget values that must survive moving between pages. Streamlit drops a widget's state
# when its page isn't shown, so these are re-saved at the start of every run.
WIDGET_DEFAULTS = {
    "ack": False,
    "shape": "circle",
    "cover_mode": "none",
    "region": "stripes",
    "region_pct": 40,
    "epsilon": 24,  # in 1/255 brightness steps
    "palette": "any",
    "preset": "quick",
    "test_others": True,
}


def _init_state(cfg: Settings) -> None:
    ss = st.session_state
    for key, value in WIDGET_DEFAULTS.items():
        if key == "test_others":
            value = cfg.device.type != "cpu"  # transfer tests add minutes without a GPU
        ss[key] = ss.get(key, value)
    for key in ("model", "target", "target_status", "cover", "job_id", "run_dir", "make_error"):
        ss.setdefault(key, None)
    ss.setdefault("my_runs", [])


def _go(page: str, **params: str) -> None:
    """Button callback: switch pages on the next run."""
    st.session_state["_goto"] = (page, params)


def _nav_buttons(back: str | None = None, forward: str | None = None, label: str = "",
                 enabled: bool = True) -> None:
    with st.container(horizontal=True, horizontal_alignment="right"):
        if back:
            st.button("Back", key=f"back_{back}", on_click=_go, args=(back,))
        if forward:
            st.button(label, key=f"next_{forward}", type="primary", disabled=not enabled,
                      on_click=_go, args=(forward,))


def _needs(*keys: str) -> bool:
    """Guard a step: True if earlier steps are done, else point the visitor back."""
    ss = st.session_state
    if Settings.from_env().gallery_only:
        st.info("This site is in gallery mode: making new stickers is turned off.")
        st.button("Open the gallery", type="primary", on_click=_go, args=("gallery",))
        return False
    if all(ss.get(k) for k in keys):
        return True
    first = {"ack": "welcome", "model": "model", "target": "target"}[next(k for k in keys if not ss.get(k))]
    st.info("This step needs the earlier ones first.")
    st.button("Go there", type="primary", on_click=_go, args=(first,))
    return False


def _pct(x: float) -> str:
    return f"{round(x * 100)}%"


def _model_name(key: str) -> str:
    return MODEL_INFO[key].display_name if key in MODEL_INFO else key


CSS = """
<style>
[data-testid="stMainBlockContainer"] { max-width: 1040px; }
.lede { font-size: 1.15rem; line-height: 1.55; margin-bottom: 1rem; }
.eyebrow { font-size: 0.85rem; font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase;
  opacity: 0.75; margin: 0; }
.hero-number { font-size: 4.5rem; font-weight: 700; line-height: 1; margin: 0.2rem 0 0.4rem;
  font-variant-numeric: tabular-nums; }
.summary { display: grid; grid-template-columns: max-content 1fr; gap: 6px 20px; margin: 0; }
.summary dt { opacity: 0.7; }
.summary dd { margin: 0; font-weight: 600; }
.swatches { display: flex; gap: 6px; flex-wrap: wrap; margin: -4px 0 8px; }
.swatches span { width: 24px; height: 24px; border-radius: 6px;
  border: 1px solid rgb(128 128 128 / 0.35); }
</style>
"""


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


def page_welcome() -> None:
    cfg = Settings.from_env()
    st.title("Make a sticker that fools an AI")
    st.markdown(
        '<div class="lede">Image-recognition AI can name a thousand kinds of objects in a photo, '
        "but it doesn't see the way you do. In a few minutes you'll design a sticker that "
        "makes an AI see something that isn't there, and find out why that's possible.</div>",
        unsafe_allow_html=True,
    )
    cards = (
        ("Why AI vision can be tricked", "Models learn shortcuts: patterns of texture and color "
         "that usually go with a label. A sticker can be packed with those patterns."),
        ("What makes an attack strong", "You'll test your sticker at every size, angle and "
         "lighting, and on AIs it was never designed for."),
        ("Where the AI is looking", "Heat maps show the AI staring at your sticker instead of "
         "the real object in the photo."),
    )
    for col, (title, body) in zip(st.columns(3), cards, strict=True):
        with col.container(border=True, height="stretch"):
            st.markdown(f"**{title}**")
            st.write(body)

    with st.container(border=True):
        st.subheader("Before you start")
        st.markdown(
            "Adversarial stickers are real security research, and they can cause real harm. "
            "By continuing you agree to:\n"
            "- Use stickers for learning, research, and fun with your own things.\n"
            "- **Never** use them to fool safety systems: cars, cameras, alarms or content "
            "filters.\n"
            "- **Never** use them to harass, mislabel or embarrass people."
        )
        with st.expander("Some target labels are blocked. Why?"):
            for b in blocked_categories():
                st.markdown(f"**{b['category'].capitalize()}**: {b['reason']} "
                            f"*(e.g. {', '.join(b['examples'])})*")
        if not cfg.gallery_only:
            st.caption(f"Photos you upload are used only to make your sticker. They are stored "
                       f"on this server with your results and deleted after "
                       f"{cfg.ttl_hours:g} hours.")
        st.checkbox("I understand and agree", key="ack")
        start = "gallery" if cfg.gallery_only else "model"
        st.button("Browse the gallery" if cfg.gallery_only else "Start", type="primary",
                  disabled=not st.session_state.ack, on_click=_go, args=(start,))
    if cfg.gallery_only:
        st.caption("This site is in gallery mode: you can explore finished stickers, but "
                   "making new ones is turned off.")


def _choose_model(key: str) -> None:
    ss = st.session_state
    ss.model = key
    if ss.target:  # re-check: free-text targets only work with CLIP
        _check_target(ss.target["label"])


def page_model() -> None:
    cfg = Settings.from_env()
    if not _needs("ack"):
        return
    st.header("Which AI do you want to fool?")
    st.write("Each of these models learned to recognize objects from over a million labeled "
             "photos. They were built in different ways and in different years. Bigger isn't "
             "always harder to fool.")
    if cfg.device.type == "cpu":
        st.caption("This server has no GPU, so the speed of each model matters. Fast models "
                   "finish in a few minutes; slow ones can take much longer.")
    keys = available_models()
    for row in range(0, len(keys), 3):
        for col, key in zip(st.columns(3), keys[row:row + 3], strict=False):
            info = MODEL_INFO[key]
            chosen = st.session_state.model == key
            with col.container(border=True, height="stretch"):
                st.markdown(f"#### {info.display_name}")
                with st.container(horizontal=True, gap="small"):
                    if info.family == "clip":
                        st.badge("reads any words", color="violet")
                    if cfg.device.type == "cpu":
                        speed = CPU_SPEED.get(key, "medium")
                        st.badge(f"{speed} without a GPU", color={
                            "fast": "green", "medium": "gray", "slow": "orange"}[speed])
                st.caption(f"{info.year} · {round(info.params_millions)} million parameters")
                st.write(info.description)
                st.button("Selected" if chosen else "Choose", key=f"model_{key}",
                          type="primary" if chosen else "secondary",
                          icon=":material/check:" if chosen else None,
                          on_click=_choose_model, args=(key,), width="stretch")
    _nav_buttons(forward="target", label="Next: pick a target",
                 enabled=st.session_state.model is not None)


def _check_target(label: str, display: str | None = None) -> None:
    ss = st.session_state
    res = check_target(label, ss.model or "resnet50")
    if res["allowed"]:
        ss.target = res
        ss.target_status = ("good", f"Target: “{res['label']}”. The sticker will try to make "
                                    f"{_model_name(ss.model)} see this.")
    else:
        ss.target = None
        shown = display or label
        ss.target_status = ("bad", res["reason"] if res["category"] == "unknown"
                            else f"“{shown}” is blocked ({res['category']}). {res['reason']}")


def _pick_featured() -> None:
    if (label := st.session_state.get("_featured")) is not None:
        _check_target(label)


def _pick_label() -> None:
    if (index := st.session_state.get("_label_search")) is not None:
        _check_target(str(index), imagenet_classes()[index])


def page_target() -> None:
    if not _needs("ack", "model"):
        return
    ss = st.session_state
    st.header("What should the AI see?")
    st.markdown("Pick something that clearly *isn't* in your photos. That's what makes the trick "
                "convincing. The sticker will try to make the AI say this label no matter what "
                "else is in the picture.")
    st.pills("Popular targets", FEATURED_TARGETS, key="_featured", on_change=_pick_featured)
    classes = imagenet_classes()
    st.selectbox("Or search all 1,000 labels", range(len(classes)), index=None,
                 format_func=lambda i: classes[i], key="_label_search", on_change=_pick_label,
                 placeholder="e.g. guitar, pizza, jellyfish")
    if supports_free_text(ss.model):
        with st.form("free_text", border=False, enter_to_submit=True):
            text = st.text_input("Or, because CLIP reads text, type your own words",
                                 max_chars=60, placeholder="e.g. a bowl of ramen")
            if st.form_submit_button("Use this") and text.strip():
                _check_target(text.strip())
    if ss.target_status:
        kind, message = ss.target_status
        (st.success if kind == "good" else st.error)(message)
    _nav_buttons(back="model", forward="design", label="Next: design it",
                 enabled=ss.target is not None)


def _store_cover() -> None:
    ss = st.session_state
    upload = ss.get("_cover_upload")
    ss.cover_error = None
    if upload is None:
        ss.cover = None
        return
    data = upload.getvalue()
    try:
        read_photo(data)
    except ValueError as err:
        ss.cover, ss.cover_error = None, str(err)
        return
    ss.cover = {"name": upload.name, "bytes": data}


def _remove_cover() -> None:
    st.session_state.cover = None


@st.cache_data(max_entries=64, show_spinner=False)
def _preview_png(shape: str, cover_mode: str, region: str, amount: float,
                 cover: bytes | None) -> bytes:
    import io

    img = design_preview(shape, cover_mode, region, amount,
                         read_photo(cover) if cover else None)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _palette_label(key: str) -> str:
    return {"any": "Any colors (strongest)", "auto": "Match my photo"}.get(key, key.capitalize())


def page_design() -> None:
    cfg = Settings.from_env()
    if not _needs("ack", "model", "target"):
        return
    ss = st.session_state
    st.header("Design your sticker")
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.segmented_control("Shape", SHAPES, key="shape", required=True,
                             format_func=str.capitalize)

        st.markdown("**Make it look normal?**")
        st.caption("A pure attack looks like strange, colorful noise. You can hide it in a photo "
                   "instead, but the more normal it looks, the weaker it gets. Finding that "
                   "balance is part of the experiment.")
        labels = {"none": "No: pure attack", "region": "Hide it in my photo",
                  "blend": "Blend it into my photo"}
        st.radio("Make it look normal?", COVER_MODES, key="cover_mode",
                 format_func=labels.get, label_visibility="collapsed", captions=[
                     "The strongest sticker. Every pixel is chosen to fool the AI.",
                     "Keep your photo and let the attack use only some areas, like stripes or "
                     "a border.",
                     "Every pixel of your photo changes a tiny bit. Subtle, but weaker.",
                 ])
        if ss.cover_mode != "none":
            if ss.cover:
                with st.container(horizontal=True, vertical_alignment="center"):
                    st.image(ss.cover["bytes"], width=56)
                    st.caption(f"Using your photo **{ss.cover['name']}**")
                    st.button("Remove", on_click=_remove_cover, type="tertiary")
            else:
                st.file_uploader("Your photo (JPEG or PNG, under 10 MB)",
                                 type=["jpg", "jpeg", "png"], key="_cover_upload",
                                 on_change=_store_cover, max_upload_size=10)
                if ss.get("cover_error"):
                    st.error(ss.cover_error)
        if ss.cover_mode == "region":
            st.selectbox("Where the attack goes", REGION_PRESETS, key="region",
                          format_func=str.capitalize)
            st.slider("How much of the sticker the attack may use", 10, 60, step=5,
                      key="region_pct", format="%d%%")
            st.caption("Lower is sneakier; higher is stronger.")
        elif ss.cover_mode == "blend":
            st.slider("How much each pixel may change (out of 255 brightness levels)", 4, 64,
                      step=4, key="epsilon")
            st.caption("Lower is sneakier; higher is stronger.")

        if ss.cover_mode == "blend":
            ss.palette = "any"  # palettes only apply to adversarial pixels
            st.markdown("**Colors**")
            st.caption("Blend mode keeps your photo's own colors.")
        else:
            options = ["any", *PALETTES] + (["auto"] if ss.cover else [])
            if ss.palette not in options:
                ss.palette = "any"
            st.radio("Colors", options, key="palette", format_func=_palette_label)
            colors = (PALETTES.get(ss.palette)
                      or ["#e3241b", "#ffe600", "#00933c", "#1d3f9a", "#e0007a"])
            if ss.palette != "auto":
                st.html('<div class="swatches">' + "".join(
                    f'<span style="background:{c}"></span>' for c in colors) + "</div>")

        st.selectbox("How thorough?", list(cfg.presets), key="preset",
                     format_func=lambda k: cfg.presets[k]["label"])
        n_other = len(available_models()) - 1
        st.checkbox(f"Also test it on the other {n_other} AIs (does the trick transfer?)",
                    key="test_others",
                    help="Adds a few minutes on a server without a GPU."
                    if cfg.device.type == "cpu" else None)

    with right, st.container(border=True):
        st.markdown("**Preview**")
        cover = ss.cover["bytes"] if ss.cover and ss.cover_mode != "none" else None
        st.image(_preview_png(ss.shape, ss.cover_mode, ss.region, ss.region_pct / 100, cover),
                 width=256)
        st.caption("Orange stripes show where the attack will be drawn. Everything else stays "
                   "as you see it.")
    _nav_buttons(back="target", forward="make", label="Next: make it")


def _choices() -> Choices:
    ss = st.session_state
    return Choices(
        model=ss.model, target=ss.target["label"] if ss.target else "", shape=ss.shape,
        palette=None if ss.palette == "any" else ss.palette, cover_mode=ss.cover_mode,
        region=ss.region, region_amount=ss.region_pct / 100, epsilon=ss.epsilon / 255,
        preset=ss.preset, test_other_models=ss.test_others,
    )


def _start_job() -> None:
    cfg = Settings.from_env()
    ss = st.session_state
    ss.make_error = None
    jobs = _manager()
    choices = _choices()
    if jobs is None:
        problem = "This site is in gallery mode: making new stickers is turned off."
    elif not ss.ack:
        problem = "Please accept the responsible-use guidance first."
    else:
        problem = validate(choices, ss.cover is not None, cfg.presets)
    if problem:
        ss.make_error = problem
        return
    jobs.prune(cfg.ttl_hours)
    job_id, run_dir = jobs.new_job_dir()
    cover_path = None
    if ss.cover and (choices.cover_mode != "none" or choices.palette == "auto"):
        cover_path = save_cover(read_photo(ss.cover["bytes"]), run_dir)
    request = build_request(choices, cfg.presets[choices.preset], cover_path, cfg.backgrounds)
    try:
        jobs.submit(job_id, run_dir, request)
    except QueueFullError as err:
        shutil.rmtree(run_dir, ignore_errors=True)
        ss.make_error = str(err)
        return
    ss.job_id = job_id


def _summary_html(cfg: Settings) -> str:
    ss = st.session_state
    look = {
        "none": "Pure attack",
        "region": f"Hidden in my photo ({ss.region}, {ss.region_pct}%)",
        "blend": f"Blended (±{ss.epsilon}/255)",
    }[ss.cover_mode]
    rows = [
        ("AI to fool", _model_name(ss.model)),
        ("Target label", ss.target["label"]),
        ("Shape", ss.shape),
        ("Look", look),
        ("Colors", _palette_label(ss.palette)),
        ("Thoroughness", cfg.presets[ss.preset]["label"]),
        ("Also test on other AIs", "yes" if ss.test_others else "no"),
    ]
    body = "".join(f"<dt>{html.escape(k)}</dt><dd>{html.escape(str(v))}</dd>" for k, v in rows)
    return f'<dl class="summary">{body}</dl>'


def _curve_chart(points: list[tuple[int, float]], total: int):
    import altair as alt
    import pandas as pd

    # Smooth the noisy per-batch rate with a short moving average, like the web app.
    rates = [p[1] for p in points]
    smooth = [sum(rates[max(0, i - 4):i + 1]) / len(rates[max(0, i - 4):i + 1])
              for i in range(len(rates))]
    df = pd.DataFrame({"step": [p[0] for p in points], "rate": smooth})
    base = alt.Chart(df).encode(
        x=alt.X("step:Q", title="Step", scale=alt.Scale(domain=[0, max(1, total)]),
                axis=alt.Axis(grid=False)),
        y=alt.Y("rate:Q", title=None, scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%", values=[0, 0.5, 1])),
    )
    hover = alt.selection_point(nearest=True, on="pointerover", fields=["step"], empty=False)
    area = base.mark_area(color=ACCENT, opacity=0.12)
    line = base.mark_line(color=ACCENT, strokeWidth=2, strokeCap="round", strokeJoin="round")
    dots = base.mark_point(color=ACCENT, filled=True, size=60).encode(
        opacity=alt.condition(hover, alt.value(1), alt.value(0)),
        tooltip=[alt.Tooltip("step:Q", title="Step"),
                 alt.Tooltip("rate:Q", title="Fooled", format=".0%")],
    ).add_params(hover)
    return (area + line + dots).properties(height=150)


@st.fragment(run_every=1.0)
def _progress() -> None:
    cfg = Settings.from_env()
    ss = st.session_state
    jobs = _manager()
    job = jobs.get(ss.job_id) if jobs and ss.job_id else None
    if job is None:  # the server restarted, or the run was cleaned up
        done = cfg.runs_root / str(ss.job_id) / "report.json"
        if done.exists():
            _open_run(done.parent)
            st.rerun()
        ss.job_id = None
        st.error("This sticker is no longer on the server. Please make it again.")
        return
    if job.status == "done":
        ss.my_runs.append(job.id)
        ss.job_id = None
        _open_run(job.run_dir)
        st.rerun()
    if job.status == "failed":
        st.error(f"Something went wrong: {job.error}")
        st.button("Try again", on_click=lambda: ss.update(job_id=None))
        return

    left, right = st.columns([1, 2], gap="large")
    with left:
        if job.preview_png:
            st.image(job.preview_png, width=224)
            st.caption("Live: the sticker as the computer improves it")
    with right:
        position = jobs.position(job.id)
        message = (f"Waiting in line (position {position}). Stickers are made one at a time."
                   if job.status == "queued" else job.message)
        st.markdown(f"**{message}**")
        frac = job.step / job.total_steps if job.total_steps else 0
        overall = 0.85 if job.status == "evaluating" else frac * 0.85
        st.progress(overall)
        elapsed = time.time() - (job.started or time.time())
        if job.status == "optimizing" and job.step:
            eta = job.eta()
            left_text = f" · about {max(1, round(eta / 60))} min left in this stage" if eta else ""
            st.caption(f"Step {job.step} of {job.total_steps} · {elapsed:.0f} s{left_text}")
        elif job.started:
            st.caption(f"{elapsed:.0f} s")
        if job.success_curve:
            st.markdown("**Fooling rate on practice photos**")
            st.altair_chart(_curve_chart(job.success_curve, job.total_steps),
                            width="stretch")


def page_make() -> None:
    cfg = Settings.from_env()
    if not _needs("ack", "model", "target"):
        return
    ss = st.session_state
    st.header("Make the sticker")
    st.html(_summary_html(cfg))
    if ss.job_id is None:
        if ss.make_error:
            st.error(ss.make_error)
        with st.container(horizontal=True, horizontal_alignment="right"):
            st.button("Back", on_click=_go, args=("design",))
            st.button("Make my sticker", type="primary", on_click=_start_job)
        return
    _progress()
    with st.container(border=True):
        target = html.escape(ss.target["label"])
        st.markdown(
            "**What's happening?** The computer pastes your sticker into batches of practice "
            "photos at random sizes, angles, positions and lighting, then checks what the AI "
            "sees. After each batch it nudges every pixel a little in whichever direction "
            f"makes the AI more likely to say *“{target}”*.\n\n"
            "Researchers call this *Expectation over Transformation*: the sticker gets "
            "optimized for the average photo, not one perfect photo, so it keeps working in "
            "the real world."
        )


def _open_run(run_dir: Path) -> None:
    st.session_state.run_dir = str(run_dir)
    _go("results", run=Path(run_dir).name)


def _find_run(cfg: Settings, run_id: str) -> Path | None:
    if not RUN_ID.match(run_id):
        return None
    for root in (cfg.runs_root, cfg.gallery_root):
        if (root / run_id / "report.json").exists():
            return root / run_id
    return None


def _current_run(cfg: Settings) -> Path | None:
    ss = st.session_state
    run_id = st.query_params.get("run")
    if run_id and (found := _find_run(cfg, run_id)):
        ss.run_dir = str(found)
        return found
    if ss.run_dir and (Path(ss.run_dir) / "report.json").exists():
        st.query_params["run"] = Path(ss.run_dir).name
        return Path(ss.run_dir)
    return None


def _download(run_dir: Path, name: str | None, label: str, mime: str) -> None:
    if name and (run_dir / name).exists():
        st.download_button(label, (run_dir / name).read_bytes(), file_name=name, mime=mime,
                           on_click="ignore", icon=":material/download:")


def _figure(path: Path, caption: str | None = None) -> bool:
    if path.exists():
        st.image(str(path), width="stretch", caption=caption)
        return True
    return False


def page_results() -> None:
    cfg = Settings.from_env()
    run_dir = _current_run(cfg)
    if run_dir is None:
        st.info("Make a sticker, or open one from the gallery, to see its results here.")
        st.button("Open the gallery", on_click=_go, args=("gallery",))
        return
    report = json.loads((run_dir / "report.json").read_text())
    files_path = run_dir / "files.json"
    files = json.loads(files_path.read_text()) if files_path.exists() else {}
    primary = report["models"][0]
    ev = report["evaluation"][primary]
    head, ctl = ev["headline"], ev["controls"]
    target = report["target"]["label"]
    model = _model_name(primary)

    left, right = st.columns([1, 3], gap="large", vertical_alignment="center")
    with left:
        st.image(str(run_dir / "sticker.png"), width=200)
    with right:
        st.markdown(
            f'<div class="eyebrow">“{html.escape(target)}” sticker vs. {html.escape(model)}</div>'
            f'<div class="hero-number">{_pct(head["success_rate"])}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f"of new photos made {model} say “{target}”. Without the sticker: "
            f"{_pct(ctl['no sticker']['success_rate'])}. With a random sticker of the same "
            f"shape: {_pct(ctl['random-noise sticker']['success_rate'])}."
        )
        with st.container(horizontal=True):
            _download(run_dir, files.get("print_sheet"), "Print sheet (PDF)", "application/pdf")
            _download(run_dir, files.get("png_print"), "Sticker (PNG, print size)", "image/png")
            _download(run_dir, "report_card.png", "Report card", "image/png")
            _download(run_dir, "report.md", "Full report", "text/markdown")

    st.subheader("Does it work?")
    if (run_dir / "examples.png").exists():
        st.write("These photos were never used to make the sticker. Each one shows what the AI "
                 "says with the sticker in place.")
        _figure(run_dir / "examples.png")
    else:  # curated examples leave out third-party photos
        st.write("Tested on photos that were never used to make the sticker, next to two "
                 "controls: the same photos with no sticker, and with random noise in its place.")
        _figure(run_dir / "fig_controls.png")
    for col, (name, m) in zip(st.columns(len(ctl)), ctl.items(), strict=True):
        with col.container(border=True):
            st.metric(name.capitalize(), _pct(m["success_rate"]))
            st.caption(f"labeled “{target}” · correct label lost {_pct(m['disruption_rate'])}")

    ex = report.get("explanation")
    if ex:
        st.subheader("Where is the AI looking?")
        st.write(
            f"The sticker covers about {_pct(ex['sticker_area'])} of each photo, but it gets "
            f"{_pct(ex['share_on_sticker'])} of the AI's attention when it answers “{target}”, "
            f"{ex['ratio']:.1f}× more than its size would suggest. Brighter colors in the heat "
            "maps mean \"this part mattered most.\""
        )
        _figure(run_dir / "fig_gradcam.png")

    if (run_dir / "fig_grid.png").exists() or (run_dir / "fig_sweeps.png").exists():
        st.subheader("What makes it fail?")
        st.write("The sticker was tested at every rotation and size, then under one change at a "
                 "time: lighting, blur, noise, viewing angle, compression and color tint.")
        _figure(run_dir / "fig_grid.png")
        _figure(run_dir / "fig_sweeps.png")

    if len(report["evaluation"]) > 1 and (run_dir / "fig_transfer.png").exists():
        st.subheader("Does it fool other AIs?")
        st.write("The sticker was only designed against one model. Here it is against AIs it "
                 "never saw.")
        _figure(run_dir / "fig_transfer.png")

    with st.container(border=True):
        st.subheader("Keep in mind")
        st.markdown(
            "- These results come from *simulated* photos. Printed stickers usually work less "
            "well. Print the PDF at actual size and try it with a camera.\n"
            "- Real systems often use defenses, several models, or human review.\n"
            "- Please use your sticker responsibly: never on safety systems, and never to "
            "target people."
        )
    if not cfg.gallery_only:
        with st.container(horizontal=True, horizontal_alignment="right"):
            st.button("Make another sticker", type="primary",
                      on_click=_go, args=("design" if st.session_state.ack else "welcome",))


def page_gallery() -> None:
    cfg = Settings.from_env()
    st.header("Gallery")
    st.write("Finished stickers. Open one to see its full report.")
    mine = set(st.session_state.my_runs)
    items = [it for it in gallery_items([cfg.runs_root]) if cfg.show_all_runs or it["id"] in mine]
    items += gallery_items([cfg.gallery_root])
    if not items:
        st.caption("No finished stickers yet.")
        return
    for row in range(0, len(items), 4):
        for col, it in zip(st.columns(4), items[row:row + 4], strict=False):
            with col.container(border=True):
                st.image(str(it["dir"] / "sticker.png"), width="stretch")
                st.markdown(f"### {_pct(it['success_rate'])}")
                st.caption(f"“{it['target']}” vs. {it['model']}"
                           + (" · yours" if it["id"] in mine else ""))
                st.button("Open", key=f"open_{it['id']}", on_click=_open_run,
                          args=(it["dir"],), width="stretch")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="adAI Sticker Lab", page_icon=":material/sell:",
                       layout="wide")
    cfg = Settings.from_env()
    _init_state(cfg)
    st.html(CSS)
    ss = st.session_state
    can_make = not cfg.gallery_only
    reachable = {
        "welcome": True,
        "model": can_make and ss.ack,
        "target": can_make and ss.ack and ss.model is not None,
        "design": can_make and ss.ack and ss.target is not None,
        "make": can_make and ss.ack and ss.target is not None,
        "results": ss.run_dir is not None or "run" in st.query_params,
        "gallery": True,
    }
    specs = {
        "welcome": (page_welcome, "Welcome"),
        "model": (page_model, "1 · Model"),
        "target": (page_target, "2 · Target"),
        "design": (page_design, "3 · Design"),
        "make": (page_make, "4 · Make"),
        "results": (page_results, "5 · Results"),
        "gallery": (page_gallery, "Gallery"),
    }
    pages = {
        name: st.Page(fn, title=title, url_path=name, default=name == "welcome",
                      visibility="visible" if reachable[name] else "hidden")
        for name, (fn, title) in specs.items()
    }
    nav = st.navigation(list(pages.values()), position="top")
    if goto := ss.pop("_goto", None):
        page, params = goto
        st.switch_page(pages[page], query_params=params or None)
    nav.run()
    st.caption("adAI Sticker Lab is a teaching tool for adversarial AI. "
               "[Source and documentation](https://github.com/haydn-s/adAI-sticker-pipeline).")
