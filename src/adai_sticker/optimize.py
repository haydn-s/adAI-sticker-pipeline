"""The optimization loop that turns a sticker design into an adversarial sticker."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

import torch

from .data import BackgroundSet
from .labels import Target
from .losses import non_printability, target_loss, total_variation
from .models import VictimModel
from .palette import printable_colors
from .sticker import Sticker
from .transforms import EOTConfig, render_scenes, sample_photometric, sample_placement


@dataclass
class AttackConfig:
    steps: int = 1000
    batch_size: int = 24
    lr: float = 0.2
    loss: str = "ce"
    margin: float = 5.0
    tv_weight: float = 0.5
    nps_weight: float = 0.25  # only applied to free-color (non-palette) adversarial pixels
    scene_size: int = 224
    palette_pull: float = 1.0  # max weight pulling free colors toward the palette
    hard_palette_from: float = 0.5  # fraction of steps after which palette colors snap
    seed: int = 0
    eot: EOTConfig = field(default_factory=EOTConfig)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class StepInfo:
    step: int
    total_steps: int
    loss: float
    attack_loss: float
    tv: float
    target_prob: float
    success_rate: float  # fraction of this step's simulated photos classified as the target
    elapsed: float


@dataclass
class AttackResult:
    sticker: Sticker
    target: Target
    model_keys: list[str]
    config: AttackConfig
    history: list[StepInfo]
    seconds: float

    def history_dicts(self) -> list[dict]:
        return [asdict(h) for h in self.history]


def _cosine(step: int, total: int, start: float, end: float) -> float:
    return end + 0.5 * (start - end) * (1 + math.cos(math.pi * step / max(1, total)))


def optimize_sticker(
    sticker: Sticker,
    models: list[VictimModel],
    target: Target,
    backgrounds: BackgroundSet,
    config: AttackConfig | None = None,
    on_step: Callable[[StepInfo], None] | None = None,
    log_every: int = 25,
) -> AttackResult:
    """Optimize ``sticker`` so ``models`` see ``target`` in simulated photos.

    With several models the loss is averaged over the ensemble, which tends to
    produce stickers that transfer better to models never seen in training.
    """
    cfg = config or AttackConfig()
    device = next(models[0].parameters()).device
    sticker.to(device).train()
    targets = [m.class_index(target) for m in models]
    backgrounds = backgrounds.without_label(target.imagenet_index)
    gen = torch.Generator().manual_seed(cfg.seed)
    printable = printable_colors().to(device)
    use_nps = cfg.nps_weight > 0 and not sticker.uses_palette and sticker.design.cover_mode != "blend"

    opt = torch.optim.Adam([sticker.param], lr=cfg.lr)
    history: list[StepInfo] = []
    hard_step = int(cfg.hard_palette_from * cfg.steps)
    t0 = time.time()

    for step in range(cfg.steps):
        for group in opt.param_groups:
            group["lr"] = _cosine(step, cfg.steps, cfg.lr, cfg.lr * 0.1)
        if sticker.uses_palette:
            sticker.hard = step >= hard_step

        bg, _ = backgrounds.sample(cfg.batch_size, cfg.scene_size, device, gen)
        placement = sample_placement(cfg.batch_size, cfg.eot, gen)
        photometric = sample_photometric(cfg.batch_size, cfg.eot, gen)
        rgb, alpha = sticker()
        scenes, _ = render_scenes(bg, rgb, alpha, placement, photometric, cfg.eot.gamut)

        attack = torch.zeros((), device=device)
        probs = torch.zeros(cfg.batch_size, device=device)
        hits = torch.zeros(cfg.batch_size, device=device)
        for model, t_idx in zip(models, targets, strict=True):
            logits = model(scenes)
            attack = attack + target_loss(logits, t_idx, cfg.loss, cfg.margin) / len(models)
            with torch.no_grad():
                probs += logits.softmax(-1)[:, t_idx] / len(models)
                hits += (logits.argmax(-1) == t_idx).float() / len(models)

        if sticker.design.cover_mode == "blend":
            tv = total_variation(rgb - sticker.cover, alpha)
        else:
            tv = total_variation(rgb, alpha * sticker.region)
        loss = attack + cfg.tv_weight * tv
        if use_nps:
            loss = loss + cfg.nps_weight * non_printability(rgb, printable, alpha * sticker.region)
        if sticker.uses_palette:
            pull = cfg.palette_pull * min(1.0, step / max(1, hard_step))
            loss = loss + pull * sticker.palette_distance()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        info = StepInfo(
            step=step + 1,
            total_steps=cfg.steps,
            loss=float(loss.detach()),
            attack_loss=float(attack.detach()),
            tv=float(tv.detach()),
            target_prob=float(probs.mean()),
            success_rate=float(hits.mean()),
            elapsed=time.time() - t0,
        )
        if (step + 1) % log_every == 0 or step == 0 or step + 1 == cfg.steps:
            history.append(info)
        if on_step is not None:
            on_step(info)

    sticker.hard = True
    sticker.eval()
    return AttackResult(sticker, target, [m.key for m in models], cfg, history, time.time() - t0)
