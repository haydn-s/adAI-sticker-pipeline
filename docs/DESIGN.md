# Design notes

This document explains how the pipeline works and why it's built this way. For how
to use it, see the [README](../README.md).

## The pipeline at a glance

```
 user choices                optimization (EOT)                     evaluation & outputs
┌──────────────┐   ┌───────────────────────────────────────┐   ┌──────────────────────────┐
│ model(s)     │   │  sticker params ──► render RGB + alpha │   │ held-out photos (val)    │
│ target label │──►│        │                               │──►│ controls · grid · sweeps │
│ shape        │   │        ▼                               │   │ transfer · Grad-CAM      │
│ colors       │   │  print sim ─► place on photo ─► camera │   │ PNG · PDF · report       │
│ cover photo  │   │        │           (train photos)      │   └──────────────────────────┘
│ region/blend │   │        ▼                               │
└──────────────┘   │  model(s) ─► loss toward target ─► ∇   │
   safety check    └───────────────────────────────────────┘
```

| Module | Responsibility |
|---|---|
| `labels.py` | ImageNet names, target resolution, **safety policy** |
| `models.py` | One interface over torchvision classifiers and CLIP (inputs in [0, 1]) |
| `masks.py` | Analytic shape outlines and editable-region masks at any resolution |
| `palette.py` | Named and custom palettes, k-means palettes from a cover photo, printable gamut |
| `sticker.py` | Trainable sticker: none / region / blend modes, free or palette colors |
| `transforms.py` | Differentiable print → place → camera simulation |
| `losses.py` | Target loss, total variation, non-printability |
| `optimize.py` | The optimization loop, with progress callbacks for UIs |
| `evaluate.py` | The evaluation battery |
| `explain.py` | Grad-CAM / gradient saliency and the attention-share metric |
| `export.py`, `report.py` | Files, figures, report card, Markdown and JSON reports |
| `pipeline.py` | End-to-end orchestration and run folders |
| `demo.py`, `physical.py` | Live webcam demo; photo-shoot metadata and per-condition summaries |
| `cli.py` | `adai-sticker` command |

## Key decisions

### Cover photos are an input to optimization, not a final layer
Laying a "normal" photo over a finished sticker would overwrite the adversarial
pixels. So the cover photo is part of the optimization, in one of two ways:

- **Region mode.** The cover is kept exactly, except inside an editable region:
  stripes (default), a border, the center, the corners, or a custom painted mask.
  The adversarial pixels are visible but confined to that region. `region_amount` is
  the stealth knob. *Where* the region sits matters as much as its size. In early
  tests, stripes covering ~50% of the sticker succeeded quickly, while a border ring
  or central disk of similar area stalled. Stripes spread the attack across the
  sticker's whole footprint, and the full-sticker results show attacks need a
  minimum footprint (about 30% of the photo's width) to work.
- **Blend mode.** The whole cover photo is perturbed, with each pixel limited to
  `±epsilon` (parameterized as `epsilon * tanh(w)` so the limit holds exactly).
  `epsilon` is the stealth knob: small values look like the photo, large values are
  stronger.

Both knobs give a measurable **stealth vs. strength** trade-off, the main experiment
in milestone M2.

### Expectation over Transformation, in three stages
A sticker that works on one perfectly framed image is useless once printed. Each
optimization step builds a batch of simulated photos:

1. **Print.** Per-channel color gain and offset jitter (printer drift), then
   compression into a printable range (inks can't reach pure black or white).
2. **Place.** Random scale (25–45% of the photo width), full 360° rotation, position,
   and perspective tilt, applied as one homography with `grid_sample`. The result is
   alpha-composited onto a random training photo (random crop and flip).
3. **Camera.** Brightness, contrast, Gaussian blur and sensor noise on the whole scene.

Everything is differentiable, so gradients reach the sticker through the full
simulation. The same functions take explicit parameters, which lets the evaluator
vary one factor at a time.

**Anti-aliasing.** Before warping, the sticker is downsampled with an anti-aliasing
filter to roughly its on-screen size. Without this, the optimizer can hide signal in
pixel-level detail that `grid_sample` aliases but a real camera would never resolve.

### Palette stickers: optimize freely, then snap
Palette stickers are optimized as free RGB. During the first half of optimization,
a gradually increasing penalty pulls each color toward its nearest palette color.
After that, every pixel **snaps** to its nearest palette color in the forward pass.
A straight-through estimator passes gradients back to the underlying free colors, so
the attack keeps adapting to the snapped image. The printed sticker is exactly the
one that was evaluated.

We compared three schedules on ResNet-50, 400 steps, "toaster", "primary" palette
(success on training batches, last 100 steps):

| Schedule | Success |
|---|---|
| Free color (no palette, reference) | 58% |
| **Pull toward palette, snap at 50% (default)** | **42%** |
| Weaker pull, snap at 80% | 39% |
| Soft-to-hard vector quantization (softmax over color distances, annealed) | 0% |

The soft relaxation saturates and its gradients vanish, so it was dropped.
Restricting colors costs some attack strength, a trade-off users can see directly.
`palette="auto"` runs k-means on the cover photo so adversarial regions match its
colors.

### Losses
- **Target loss.** Cross-entropy toward the target, as in Brown et al. A capped
  logit-margin loss (Carlini & Wagner style) is available via `--loss margin`.
- **Total variation**, which favors smooth, printable patterns. In blend mode it's
  applied to the perturbation, not the photo.
- **Non-printability**, the distance to the nearest color in a printable lattice
  (Sharif et al., 2016). Applied only to free-color stickers; palettes are printable
  by construction.

### Evaluation design
- **Held-out photos.** Optimization uses the Imagenette *train* split; evaluation
  uses *val*. Photos whose true class is the target are excluded, since they would
  count as free wins.
- **Controls** on the same scenes: no sticker, a random-noise sticker of the same
  shape, the untouched cover photo (cover modes), and the adversarial sticker. This
  separates the effect of optimization from the effect of covering part of the image.
- **Two complementary metrics.** *Success rate*: the model outputs the target label.
  *Correct label lost*: among photos the model got right without a sticker, the share
  it now gets wrong. The second captures harm to the original object even when the
  target isn't reached.
- **Fixed seeds** for every condition, so results are reproducible and comparable
  across stickers.
- **Attention share.** The fraction of Grad-CAM mass that falls on the sticker,
  compared with the sticker's share of the image. This puts a number on "the model is
  staring at the sticker".

### Safety in the core
The target-label policy runs in `make_sticker` before any model is loaded, so every
front end inherits it. See [RESPONSIBLE_USE.md](RESPONSIBLE_USE.md).

## Known limitations

- **Simulation ≠ reality.** The printer and camera models are generic. Physical
  success has to be measured (milestone M3); expect it to be lower than simulated
  success.
- **Background photos.** Imagenette has 10 everyday categories. Success on other
  scenes, such as close-ups or very cluttered images, may differ. Pass
  `--backgrounds <folder>` to test your own.
- **White-box access.** The optimizer needs the victim model's gradients. Transfer
  results show how far that carries to unseen models; black-box attacks are out of
  scope.
- **CLIP prompts.** CLIP is scored against the 1,000 ImageNet labels (plus any
  free-text target) with seven prompt templates. In a real deployment it would be
  scored against a different label set.
- **Compute.** Optimization takes a few minutes on a laptop GPU (Apple M-series or
  CUDA) and much longer on a CPU. The web app will need a hosted GPU or a pre-made
  gallery.

## References

- Brown, Mané, Roy, Abadi, Gilmer. *Adversarial Patch.* 2017.
- Athalye, Engstrom, Ilyas, Kwok. *Synthesizing Robust Adversarial Examples.* ICML 2018.
- Eykholt et al. *Robust Physical-World Attacks on Deep Learning Visual Classification.* CVPR 2018.
- Sharif, Bhagavatula, Bauer, Reiter. *Accessorize to a Crime.* CCS 2016.
- Thys, Van Ranst, Goedemé. *Fooling Automated Surveillance Cameras.* CVPR Workshops 2019.
- Hu et al. *Naturalistic Physical Adversarial Patch for Object Detectors.* ICCV 2021.
- Selvaraju et al. *Grad-CAM.* ICCV 2017.
- Smilkov et al. *SmoothGrad.* 2017.
- Radford et al. *Learning Transferable Visual Models From Natural Language Supervision (CLIP).* ICML 2021.
- Goh et al. *Multimodal Neurons in Artificial Neural Networks.* Distill 2021.
- Xiang et al. *PatchGuard.* USENIX Security 2021.
- Howard. *Imagenette.* fast.ai, 2019.
