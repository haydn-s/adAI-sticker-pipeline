"""End-to-end pipeline: design -> optimize -> evaluate -> explain -> export.

This is the single entry point the CLI (and, later, the web app) calls. Every
run writes a self-contained folder::

    runs/<name>/
      request.json        what was asked for (design, models, settings)
      sticker.pt          learned parameters (reloadable)
      assets/             copies of the cover photo / custom masks
      sticker.png         the sticker at optimization resolution (RGBA)
      sticker_900px.png   the sticker at print resolution (3" at 300 dpi)
      print_sheet.pdf     US-Letter sheet with several sizes and cut lines
      examples.png        held-out photos with the sticker and the model's label
      fig_*.png           evaluation figures
      report_card.png     all figures in one image
      report.md           plain-language summary
      report.json         every number behind the report
"""

from __future__ import annotations

import json
import math
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import torch

from . import report as rpt
from .data import load_backgrounds
from .evaluate import EvalSettings, Evaluator
from .explain import attention_share, gradcam
from .export import print_sheet, save_sticker_images, scene_grid
from .labels import Target, resolve_target
from .masks import REGION_PRESETS, SHAPES
from .models import VictimModel, default_device, load_model, supports_free_text
from .optimize import AttackConfig, AttackResult, StepInfo, optimize_sticker
from .sticker import Sticker, StickerDesign
from .transforms import EOTConfig, Placement, neutral_photometric, render_scenes

ETHICS_NOTICE = (
    "adAI Sticker Lab is a teaching tool. Adversarial stickers show how fragile image "
    "classifiers are; they must not be used to deceive safety-critical systems or to harm, "
    "harass, or misidentify people. Targets that could enable those harms are blocked."
)


@dataclass
class StickerRequest:
    target: str
    models: list[str] = field(default_factory=lambda: ["resnet50"])
    design: StickerDesign = field(default_factory=StickerDesign)
    attack: AttackConfig = field(default_factory=AttackConfig)
    eval_models: list[str] = field(default_factory=list)  # extra models for transfer
    eval: EvalSettings = field(default_factory=EvalSettings)
    backgrounds: str | None = None  # folder, "synthetic", or None for Imagenette
    train_images: int = 2000
    full_battery: bool = True  # grid + sweeps for the optimized models

    def to_dict(self) -> dict:
        return {
            "target": self.target,
            "models": self.models,
            "design": self.design.to_dict(),
            "attack": self.attack.to_dict(),
            "eval_models": self.eval_models,
            "eval": self.eval.to_dict(),
            "backgrounds": self.backgrounds,
            "train_images": self.train_images,
            "full_battery": self.full_battery,
        }

    @classmethod
    def from_dict(cls, d: dict) -> StickerRequest:
        attack = dict(d.get("attack", {}))
        eot = EOTConfig(**{k: tuple(v) if isinstance(v, list) else v
                           for k, v in attack.pop("eot", {}).items()})
        attack = {k: tuple(v) if isinstance(v, list) else v for k, v in attack.items()}
        ev = dict(d.get("eval", {}))
        ev = {k: tuple(v) if isinstance(v, list) else v for k, v in ev.items()}
        if "sweeps" in ev:
            ev["sweeps"] = {k: tuple(v) for k, v in ev["sweeps"].items()}
        return cls(
            target=d["target"],
            models=list(d.get("models", ["resnet50"])),
            design=StickerDesign(**d.get("design", {})),
            attack=AttackConfig(**attack, eot=eot),
            eval_models=list(d.get("eval_models", [])),
            eval=EvalSettings(**ev),
            backgrounds=d.get("backgrounds"),
            train_images=d.get("train_images", 2000),
            full_battery=d.get("full_battery", True),
        )


def resolve_request_target(request: StickerRequest) -> Target:
    """Resolve the target against the safety policy for the chosen models."""
    free_text_ok = all(supports_free_text(m) for m in request.models)
    return resolve_target(request.target, allow_free_text=free_text_ok)


def _copy_assets(design: StickerDesign, run_dir: Path) -> StickerDesign:
    """Copy user-supplied images into the run so it is self-contained."""
    assets = run_dir / "assets"
    d = StickerDesign(**design.to_dict())
    for attr, keep in (("cover_image", None), ("shape", SHAPES), ("region", REGION_PRESETS)):
        value = getattr(d, attr)
        if value is None or (keep is not None and value in keep):
            continue
        src = Path(value)
        if src.exists():
            assets.mkdir(parents=True, exist_ok=True)
            dst = assets / f"{attr}{src.suffix.lower()}"
            shutil.copyfile(src, dst)
            setattr(d, attr, str(dst.relative_to(run_dir)))
    return d


def _absolute_design(design_dict: dict, run_dir: Path) -> StickerDesign:
    d = StickerDesign(**design_dict)
    for attr in ("cover_image", "shape", "region"):
        value = getattr(d, attr)
        if value and (run_dir / value).exists():
            setattr(d, attr, str(run_dir / value))
    return d


def _save_json(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, default=float))


def save_run(result: AttackResult, request: StickerRequest, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    stored_design = _copy_assets(result.sticker.design, run_dir)
    state = result.sticker.state()
    state["design"] = stored_design.to_dict()
    state["target"] = result.target.to_dict()
    state["models"] = result.model_keys
    torch.save(state, run_dir / "sticker.pt")
    req = request.to_dict()
    req["design"] = stored_design.to_dict()
    _save_json(req, run_dir / "request.json")
    _save_json(result.history_dicts(), run_dir / "history.json")


def load_run(run_dir: str | Path) -> tuple[Sticker, Target, StickerRequest]:
    run_dir = Path(run_dir)
    state = torch.load(run_dir / "sticker.pt", map_location="cpu", weights_only=False)
    state["design"] = _absolute_design(state["design"], run_dir).to_dict()
    sticker = Sticker.from_state(state)
    target = Target(**state["target"])
    request = StickerRequest.from_dict(json.loads((run_dir / "request.json").read_text()))
    return sticker, target, request


# ---------------------------------------------------------------------------
# Evaluation + explanation of a finished sticker
# ---------------------------------------------------------------------------


@torch.no_grad()
def _examples(model: VictimModel, sticker: Sticker, target: Target, evaluator: Evaluator,
              n: int = 8, scale: float = 0.35, seed: int = 7):
    """Held-out photos with the sticker placed at a fixed size, plus the model's labels."""
    device = next(model.parameters()).device
    gen = torch.Generator().manual_seed(seed)
    bgs = evaluator.backgrounds
    n = min(n, len(bgs))
    idx = torch.randperm(len(bgs), generator=gen)[:n]
    imgs = bgs.center(bgs.images[idx], evaluator.settings.scene_size).to(device).float() / 255
    angle = (torch.rand(n, generator=gen) * 2 - 1) * math.pi * 0.25
    center = (torch.rand(n, 2, generator=gen) * 2 - 1) * (1 - scale) * 0.8
    placement = Placement(torch.full((n,), scale), angle, center, torch.zeros(n, 2))
    rgb, alpha = sticker.render()
    scenes, scene_alpha = render_scenes(imgs, rgb.to(device), alpha.to(device), placement,
                                        neutral_photometric(n), evaluator.eot.gamut)
    probs = model(scenes).float().softmax(-1)
    t_idx = model.class_index(target)
    names = model.class_names
    top = probs.max(-1)
    captions = [f"{names[int(i)]} ({float(p):.0%})" for p, i in zip(top.values, top.indices, strict=True)]
    hits = [int(i) == t_idx for i in top.indices]
    clean_probs = model(imgs).float().softmax(-1).max(-1)
    clean_captions = [f"{names[int(i)]} ({float(p):.0%})"
                      for p, i in zip(clean_probs.values, clean_probs.indices, strict=True)]
    return imgs, scenes, scene_alpha, captions, hits, clean_captions, clean_probs.indices


def explain(model: VictimModel, sticker: Sticker, target: Target, evaluator: Evaluator,
            run_dir: Path, n_share: int = 48) -> dict:
    imgs, scenes, scene_alpha, captions, _, clean_caps, clean_idx = _examples(
        model, sticker, target, evaluator, n=max(4, n_share)
    )
    t_idx = model.class_index(target)
    sticker_cams, method = gradcam(model, scenes, t_idx)
    share = attention_share(sticker_cams, scene_alpha, method)
    k = min(4, imgs.shape[0])
    clean_cams = torch.stack([gradcam(model, imgs[i:i + 1], int(clean_idx[i]))[0][0] for i in range(k)])
    rpt.fig_gradcam(
        imgs[:k], clean_cams, scenes[:k], sticker_cams[:k],
        [(f"says: {clean_caps[i]}", f"says: {captions[i]}") for i in range(k)],
        run_dir / "fig_gradcam.png", method,
    )
    return share.to_dict()


def evaluate_sticker(
    sticker: Sticker,
    target: Target,
    request: StickerRequest,
    run_dir: Path,
    device: torch.device,
    loaded: dict[str, VictimModel] | None = None,
    say: Callable[[str], None] = print,
    history: list[dict] | None = None,
    seconds: float = 0.0,
) -> dict:
    """Run the evaluation battery, explanations, figures and report for a sticker."""
    loaded = dict(loaded or {})
    sticker.to(device).eval()
    val = load_backgrounds("val", request.backgrounds, limit=max(request.eval.n_images * 2, 600),
                           seed=request.eval.seed)
    evaluator = Evaluator(val, target, request.eval, request.attack.eot)

    evaluation: dict[str, dict] = {}
    skipped: list[str] = []
    all_models = list(dict.fromkeys(request.models + request.eval_models))
    for key in all_models:
        if target.is_free_text and not supports_free_text(key):
            skipped.append(key)
            continue
        model = loaded.get(key) or load_model(key, device)
        loaded[key] = model
        battery = request.full_battery and key in request.models
        evaluation[key] = evaluator.full(model, sticker, sweeps=battery, grid=battery, progress=say)

    primary = loaded[request.models[0]]
    say("Explaining predictions (Grad-CAM)")
    explanation = explain(primary, sticker, target, evaluator, run_dir)

    imgs, scenes, _, captions, hits, _, _ = _examples(primary, sticker, target, evaluator, n=8)
    scene_grid(scenes, captions, hits, run_dir / "examples.png")

    summary = {
        "target": target.to_dict(),
        "models": request.models,
        "design": sticker.design.to_dict(),
        "attack": request.attack.to_dict(),
        "eval_settings": request.eval.to_dict(),
        "backgrounds": {"train": request.backgrounds or "imagenette/train",
                        "val": val.fingerprint()},
        "evaluation": evaluation,
        "skipped_models": skipped,
        "explanation": explanation,
        "stealth": sticker.perturbation_stats(),
        "seconds": seconds,
        "ethics_notice": ETHICS_NOTICE,
    }
    _save_json(summary, run_dir / "report.json")

    say("Drawing figures")
    title_model = rpt.model_name(request.models[0])
    figs = [run_dir / "examples.png"]
    prim_eval = evaluation[request.models[0]]
    figs.append(rpt.fig_controls(prim_eval["controls"], run_dir / "fig_controls.png",
                                 f"{title_model}: sticker vs. controls"))
    if "grid" in prim_eval:
        figs.append(rpt.fig_grid(prim_eval["grid"], run_dir / "fig_grid.png",
                                 f"{title_model}: success by rotation and size"))
    if "sweeps" in prim_eval:
        figs.append(rpt.fig_sweeps(prim_eval["sweeps"], run_dir / "fig_sweeps.png",
                                   f"{title_model}: success under changing conditions"))
    if len(evaluation) > 1:
        figs.append(rpt.fig_transfer({k: v["headline"]["success_rate"] for k, v in evaluation.items()},
                                     request.models, run_dir / "fig_transfer.png",
                                     "Does it fool other models?"))
    figs.append(run_dir / "fig_gradcam.png")
    if history:
        figs.append(rpt.fig_training(history, run_dir / "fig_training.png"))
    head = prim_eval["headline"]["success_rate"]
    rpt.report_card(
        figs,
        f'"{target.label}" sticker vs. {title_model}',
        f"Fools the model on {head:.0%} of held-out photos in simulation  ·  adAI Sticker Lab",
        run_dir / "report_card.png",
    )
    (run_dir / "report.md").write_text(rpt.markdown_report(summary))
    return summary


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def make_sticker(
    request: StickerRequest,
    run_dir: str | Path,
    device: torch.device | str | None = None,
    say: Callable[[str], None] = print,
    on_step: Callable[[StepInfo], None] | None = None,
    evaluate: bool = True,
) -> dict:
    run_dir = Path(run_dir)
    device = torch.device(device) if device is not None else default_device()
    target = resolve_request_target(request)  # raises BlockedTargetError before any work
    request.design.validate()

    say(f"Target: {target.label}  |  models: {', '.join(request.models)}  |  device: {device}")
    models = {k: load_model(k, device) for k in request.models}
    say("Loading background photos")
    train = load_backgrounds("train", request.backgrounds, limit=request.train_images,
                             seed=request.attack.seed)
    sticker = Sticker(request.design, seed=request.attack.seed)

    say(f"Optimizing for {request.attack.steps} steps")
    result = optimize_sticker(sticker, list(models.values()), target, train, request.attack,
                              on_step=on_step)
    save_run(result, request, run_dir)
    files = save_sticker_images(result.sticker, run_dir)
    files["print_sheet"] = print_sheet(
        result.sticker, run_dir / "print_sheet.pdf",
        title=f'Adversarial sticker: "{target.label}"',
        subtitle=f"Optimized against {', '.join(rpt.model_name(m) for m in request.models)}",
    ).name
    _save_json(files, run_dir / "files.json")
    if not evaluate:
        return {"run_dir": str(run_dir), "files": files}
    summary = evaluate_sticker(result.sticker, target, request, run_dir, device, models, say,
                               result.history_dicts(), result.seconds)
    summary["files"] = files
    return summary


def reevaluate(run_dir: str | Path, eval_models: list[str] | None = None,
               device: torch.device | str | None = None, say: Callable[[str], None] = print) -> dict:
    """Re-run the evaluation battery on a saved run (e.g. with more transfer models)."""
    run_dir = Path(run_dir)
    device = torch.device(device) if device is not None else default_device()
    sticker, target, request = load_run(run_dir)
    if eval_models is not None:
        request.eval_models = eval_models
    history_path = run_dir / "history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else None
    return evaluate_sticker(sticker, target, request, run_dir, device, say=say, history=history)


__all__ = [
    "ETHICS_NOTICE",
    "StickerRequest",
    "make_sticker",
    "reevaluate",
    "load_run",
]
