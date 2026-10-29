# Roadmap

The project is organized around its **evaluation**. Each milestone adds one layer of
evidence for the central question:

> **Research question.** How much attack success does an adversarial sticker give up
> as it is made to look more natural, and does that trade-off survive real-world
> printing and transfer to models it never saw? Does making one by hand help
> non-experts form an accurate mental model of adversarial vulnerability?

The four evidence layers:

1. **Digital robustness.** Attack success over rotation × size, and under lighting,
   blur, noise, viewing angle, compression and color cast, on held-out photos with
   controls.
2. **Transfer.** Success against models the sticker was never optimized on, plus
   ensembles as a way to improve it.
3. **Physical.** Printed stickers photographed in real scenes and scored.
4. **Learning outcome.** A small pre/post study with non-technical participants.

## Milestones

### M1: Core pipeline ✅
- Sticker design: shapes, palettes (fixed, custom, or derived from the cover photo),
  **region** and **blend** cover modes.
- EOT optimizer (print → place → camera simulation), ensembles, safety policy
  enforced in the core.
- Evaluation battery: controls, rotation × size grid, seven condition sweeps,
  transfer, Grad-CAM attention share.
- Exports: PNG (optimization and print resolution), print-sheet PDF with cut lines,
  examples, figures, report card, Markdown and JSON reports.
- `eval-photos` command for scoring photos of printed stickers.

### M2: Stealth vs. strength experiments
- Sweep the blend budget `epsilon` and the region `amount` for several targets and
  models. Plot success rate against how much of the photo changed (the headline
  trade-off curve for the poster).
- Ensemble vs. single-model optimization: does an ensemble improve transfer to
  held-out architectures (e.g. optimize on CNNs, test on ViT and CLIP)?
- Repeat over seeds and targets; report means and spread.
- Output: `experiments/` scripts plus `docs/RESULTS.md`, with figures generated from code.

### M3: Physical testing and live demo (tooling ✅, photo shoot pending)
- ✅ `adai-sticker demo`: live webcam view with top-5 guesses, a FOOLED banner, a
  Grad-CAM overlay, and frame saving for scoring.
- ✅ `eval-photos --meta`: per-condition breakdowns from a shoot CSV.
- ✅ Written protocol: [PHYSICAL_TESTING.md](PHYSICAL_TESTING.md).
- ⏳ Run the shoot and compare with the simulator.
- Printing protocol (paper, printer, sizes) and photo protocol (distances, angles,
  lighting), documented so others can repeat it.
- Score photos with `eval-photos`; compare physical success with the simulator's
  prediction for the same conditions.
- Optional printer color calibration (print a color chart, photograph it, fit a
  color transform) to replace the generic printer model.
- **Live webcam demo** for Project Expo: real-time top-5 predictions and a Grad-CAM
  overlay.

### M4: Defense mode
- Show at least one defense, e.g. adversarial training (an adversarially robust
  ResNet-50) or a patch-specific defense (masking or detecting the patch region), and
  measure how much success drops.
- Teaching goal: attacks and defenses are an arms race, and robustness costs accuracy.

### M5: Web app
- Guided flow that matches the six user-facing steps: pick a model → pick a target
  → design (shape, colors, cover photo, region or blend) → optimize with live
  progress → see the report → download PNG and PDF.
- Ethics acknowledgement before creation; refusals explained in plain language.
- Hosting: pre-made gallery that works without a GPU; live generation on a hosted GPU.

### M6: Learning study, poster and video
- Short pre/post concept quiz and attitude items (8–15 participants), with informed
  consent and no personal data collected.
- Poster (24"×24" or larger) with a live demo station; 5-minute video for online
  submission.
