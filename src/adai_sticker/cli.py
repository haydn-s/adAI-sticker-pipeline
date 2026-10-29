"""Command-line interface: ``adai-sticker <command>``."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import textwrap
from datetime import datetime
from pathlib import Path

from .labels import (
    BLOCK_RATIONALE,
    BLOCKED_IMAGENET,
    FEATURED_TARGETS,
    BlockedTargetError,
    resolve_target,
    search_labels,
)
from .masks import REGION_PRESETS, SHAPES
from .models import MODEL_INFO, available_models, supports_free_text
from .optimize import AttackConfig, StepInfo
from .palette import PALETTES


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "sticker"


def cmd_labels(args: argparse.Namespace) -> int:
    if args.blocked:
        for category, entries in BLOCKED_IMAGENET.items():
            print(f"\n{category.upper()}")
            print(textwrap.fill(BLOCK_RATIONALE[category], 88, initial_indent="  ", subsequent_indent="  "))
            print("  blocked: " + ", ".join(entries.values()))
        return 0
    if args.query:
        for idx, name in search_labels(args.query, limit=args.limit):
            print(f"{idx:4d}  {name}")
        return 0
    print("Featured targets: " + ", ".join(FEATURED_TARGETS))
    print("Search all 1,000 ImageNet labels with: adai-sticker labels <word>")
    return 0


def cmd_models(_args: argparse.Namespace) -> int:
    for key, info in MODEL_INFO.items():
        free = "  (accepts free-text targets)" if info.family == "clip" else ""
        print(f"{key:20s} {info.display_name} — {info.year}, {info.params_millions:.0f}M parameters{free}")
        print(textwrap.fill(info.description, 88, initial_indent=" " * 21, subsequent_indent=" " * 21))
    return 0


def cmd_fetch_data(args: argparse.Namespace) -> int:
    from .data import fetch_imagenette

    root = fetch_imagenette(Path(args.data_dir) if args.data_dir else None, size=args.size)
    print(f"Imagenette ready at {root}")
    return 0


def _progress_printer(every: int):
    def on_step(info: StepInfo) -> None:
        if info.step == 1 or info.step % every == 0 or info.step == info.total_steps:
            eta = info.elapsed / info.step * (info.total_steps - info.step)
            print(f"  step {info.step:5d}/{info.total_steps}  success {info.success_rate:5.0%}  "
                  f"target prob {info.target_prob:5.2f}  loss {info.attack_loss:6.3f}  "
                  f"({info.elapsed:4.0f}s, ~{eta:3.0f}s left)", flush=True)
    return on_step


def cmd_make(args: argparse.Namespace) -> int:
    from .evaluate import EvalSettings
    from .pipeline import ETHICS_NOTICE, StickerRequest, make_sticker
    from .sticker import StickerDesign

    print(textwrap.fill(ETHICS_NOTICE, 88) + "\n")
    models = args.model or ["resnet50"]
    eval_models = list(args.eval_model or [])
    if args.transfer:
        eval_models += [m for m in available_models() if m not in models]
    design = StickerDesign(
        resolution=args.resolution,
        shape=args.shape,
        palette=args.palette,
        palette_size=args.palette_size,
        cover_mode=args.cover_mode,
        cover_image=args.cover,
        region=args.region,
        region_amount=args.region_amount,
        epsilon=args.epsilon,
    )
    attack = AttackConfig(steps=args.steps, batch_size=args.batch_size, lr=args.lr, seed=args.seed,
                          tv_weight=args.tv_weight, loss=args.loss)
    ev = EvalSettings(n_images=args.eval_images, grid_images=max(20, args.eval_images // 3))
    if args.quick:
        attack.steps = min(attack.steps, 300)
        ev = EvalSettings(n_images=100, grid_images=30)
    request = StickerRequest(
        target=args.target, models=models, design=design, attack=attack,
        eval_models=eval_models, eval=ev, backgrounds=args.backgrounds,
        train_images=args.train_images, full_battery=not args.quick,
    )
    out = Path(args.out) if args.out else Path("runs") / (
        f"{datetime.now():%Y%m%d-%H%M%S}-{_slug(args.target)}-{_slug(models[0])}")
    try:
        summary = make_sticker(request, out, device=args.device,
                               on_step=_progress_printer(max(1, attack.steps // 20)),
                               evaluate=not args.no_eval)
    except BlockedTargetError as err:
        print(f"Blocked: {err}", file=sys.stderr)
        return 2
    except ValueError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 2
    print(f"\nDone. Results in {out}/")
    if "evaluation" in summary:
        for key, ev_res in summary["evaluation"].items():
            h = ev_res["headline"]
            print(f"  {key:20s} success {h['success_rate']:5.0%}   correct label lost {h['disruption_rate']:5.0%}")
        print(f"  Report: {out / 'report.md'}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from .pipeline import reevaluate

    eval_models = list(args.eval_model) if args.eval_model else None
    if args.transfer:
        req = json.loads((Path(args.run_dir) / "request.json").read_text())
        eval_models = [m for m in available_models() if m not in req["models"]]
    summary = reevaluate(args.run_dir, eval_models, device=args.device)
    for key, ev_res in summary["evaluation"].items():
        h = ev_res["headline"]
        print(f"  {key:20s} success {h['success_rate']:5.0%}   correct label lost {h['disruption_rate']:5.0%}")
    return 0


def cmd_defend(args: argparse.Namespace) -> int:
    from .defense import evaluate_defenses

    try:
        evaluate_defenses(args.run_dir, args.defense, adaptive_steps=args.adaptive_steps,
                          device=args.device)
    except ValueError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 2
    print(f"Defense report: {Path(args.run_dir) / 'defense.md'}")
    return 0


def cmd_eval_photos(args: argparse.Namespace) -> int:
    """Physical test: classify real photos of a printed sticker."""
    import torch

    from .data import IMAGE_EXTENSIONS
    from .models import default_device, load_model
    from .sticker import load_image

    device = torch.device(args.device) if args.device else default_device()
    try:
        target = resolve_target(args.target, allow_free_text=supports_free_text(args.model))
    except BlockedTargetError as err:
        print(f"Blocked: {err}", file=sys.stderr)
        return 2
    model = load_model(args.model, device)
    t_idx = model.class_index(target)
    names = model.class_names
    photos = sorted(p for p in Path(args.photos).rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not photos:
        print(f"No photos found in {args.photos}", file=sys.stderr)
        return 1
    rows = []
    with torch.no_grad():
        for p in photos:
            img = load_image(p, model.input_size)[None].to(device)
            probs = model(img).float().softmax(-1)[0]
            top = probs.topk(5)
            rows.append({
                "photo": str(p.relative_to(args.photos)),
                "prediction": names[int(top.indices[0])],
                "confidence": round(float(top.values[0]), 4),
                "target_prob": round(float(probs[t_idx]), 4),
                "target_top1": int(top.indices[0]) == t_idx,
                "target_top5": t_idx in top.indices.tolist(),
            })
    out = Path(args.out) if args.out else Path(args.photos) / f"results_{args.model}.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    hits = sum(r["target_top1"] for r in rows)
    top5 = sum(r["target_top5"] for r in rows)
    print(f"{len(rows)} photos  |  labeled '{target.label}': {hits} ({hits / len(rows):.0%})  |  "
          f"in top 5: {top5} ({top5 / len(rows):.0%})")
    print(f"Per-photo results: {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="adai-sticker",
        description="Design, optimize and evaluate adversarial stickers against image classifiers.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("labels", help="search target labels, or list blocked ones")
    s.add_argument("query", nargs="?")
    s.add_argument("--blocked", action="store_true", help="show blocked labels and why")
    s.add_argument("--limit", type=int, default=15)
    s.set_defaults(func=cmd_labels)

    s = sub.add_parser("models", help="list the models you can attack")
    s.set_defaults(func=cmd_models)

    s = sub.add_parser("fetch-data", help="download the Imagenette background photos")
    s.add_argument("--size", type=int, choices=(160, 320), default=320)
    s.add_argument("--data-dir")
    s.set_defaults(func=cmd_fetch_data)

    s = sub.add_parser("make", help="optimize, evaluate and export a sticker")
    s.add_argument("--target", required=True, help="ImageNet label (or free text for CLIP models)")
    s.add_argument("--model", action="append", choices=available_models(),
                   help="model(s) to optimize against; repeat for an ensemble (default resnet50)")
    s.add_argument("--eval-model", action="append", choices=available_models(),
                   help="extra model(s) to test transfer on")
    s.add_argument("--transfer", action="store_true", help="test transfer on every other model")
    g = s.add_argument_group("design")
    g.add_argument("--shape", default="circle", help=f"{', '.join(SHAPES)}, or a mask image path")
    g.add_argument("--palette", help=f"{', '.join(PALETTES)}, 'auto' (from cover), or hex list")
    g.add_argument("--palette-size", type=int, default=8)
    g.add_argument("--cover-mode", choices=("none", "region", "blend"), default="none")
    g.add_argument("--cover", help="cover photo for region/blend modes")
    g.add_argument("--region", default="stripes", help=f"{', '.join(REGION_PRESETS)}, or a mask image path")
    g.add_argument("--region-amount", type=float, default=0.4)
    g.add_argument("--epsilon", type=float, default=0.12, help="blend-mode budget per pixel, 0-1")
    g.add_argument("--resolution", type=int, default=224)
    g = s.add_argument_group("optimization")
    g.add_argument("--steps", type=int, default=1000)
    g.add_argument("--batch-size", type=int, default=24)
    g.add_argument("--lr", type=float, default=0.2)
    g.add_argument("--tv-weight", type=float, default=0.5)
    g.add_argument("--loss", choices=("ce", "margin"), default="ce")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--train-images", type=int, default=2000)
    g.add_argument("--backgrounds", help="folder of background photos (default: Imagenette)")
    g = s.add_argument_group("evaluation")
    g.add_argument("--eval-images", type=int, default=300)
    g.add_argument("--quick", action="store_true", help="fewer steps and a lighter evaluation")
    g.add_argument("--no-eval", action="store_true", help="skip evaluation and reports")
    s.add_argument("--out", help="output folder (default runs/<timestamp>-<target>-<model>)")
    s.add_argument("--device", help="cuda, mps or cpu (default: best available)")
    s.set_defaults(func=cmd_make)

    s = sub.add_parser("evaluate", help="re-run the evaluation battery on a saved run")
    s.add_argument("run_dir")
    s.add_argument("--eval-model", action="append", choices=available_models())
    s.add_argument("--transfer", action="store_true")
    s.add_argument("--device")
    s.set_defaults(func=cmd_evaluate)

    s = sub.add_parser("defend", help="test defenses against a saved sticker, and their cost")
    s.add_argument("run_dir")
    s.add_argument("--defense", action="append",
                   help="blur[:sigma], jpeg[:quality], saliency[:area]; repeat (default: all three)")
    s.add_argument("--adaptive-steps", type=int, default=300,
                   help="re-optimize the sticker against the blur defense (0 to skip)")
    s.add_argument("--device")
    s.set_defaults(func=cmd_defend)

    s = sub.add_parser("eval-photos", help="score real photos of a printed sticker (physical test)")
    s.add_argument("--photos", required=True, help="folder of photos")
    s.add_argument("--target", required=True)
    s.add_argument("--model", default="resnet50", choices=available_models())
    s.add_argument("--out", help="CSV path (default <photos>/results_<model>.csv)")
    s.add_argument("--device")
    s.set_defaults(func=cmd_eval_photos)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
