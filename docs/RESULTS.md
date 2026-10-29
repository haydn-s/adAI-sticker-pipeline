# Results

Evidence for the project's research question (see the [roadmap](ROADMAP.md)):

> How much attack success does an adversarial sticker give up as it is made to look
> more natural, and does that trade-off survive transfer to models it never saw?

Every number below comes from **held-out** photos (Imagenette *val*, 300 photos,
photos of the target class excluded). Conditions are randomized but seeded: random
size (25–45% of the photo width), any rotation, position, perspective tilt, lighting,
contrast, blur, sensor noise and printer color drift. Target: **"toaster"**. All
numbers are simulated; physical results will be added after the photo shoot
([protocol](PHYSICAL_TESTING.md)).

Reproduce with `uv run python experiments/<script>.py` (see
[experiments/README.md](../experiments/README.md)). Raw numbers are in
`docs/results/*/results.csv`.

## 1. Stealth vs. strength

*Can a sticker look like a normal photo and still fool the model?*

The cover photo is a dog (an English springer from Imagenette val). Each point is a
separate sticker optimized for 600 steps against ResNet-50:

- **Region: stripes / border.** The photo is kept exactly, except in diagonal
  stripes or a border ring covering *amount* of the sticker.
- **Blend.** Every pixel of the photo may change by at most ε/255.

Stealth is measured as **SSIM** to the original photo (1 = identical).

![Stealth vs. strength](results/stealth/fig_tradeoff.png)

| Sticker | Similarity (SSIM) | Says "toaster" | Toaster in top 5 | Correct label lost |
|---|---|---|---|---|
| Fully adversarial (reference) | — | **68%** | 99% | 63% |
| Stripes, 10% | 0.75 | 3% | 30% | 13% |
| Stripes, 20% | 0.64 | 16% | 76% | 23% |
| Stripes, 30% | 0.54 | 29% | 87% | 30% |
| Stripes, 45% | 0.39 | 43% | 96% | 39% |
| Stripes, 60% | 0.25 | **57%** | 98% | 51% |
| Border, 10% | 0.81 | 0% | 1% | 15% |
| Border, 30% | 0.49 | 3% | 22% | 16% |
| Border, 60% | 0.16 | 28% | 91% | 31% |
| Blend, ε = 4/255 | 0.96 | 0% | 0% | 12% |
| Blend, ε = 16/255 | 0.70 | 4% | 38% | 12% |
| Blend, ε = 32/255 | 0.49 | 23% | 74% | 26% |
| Blend, ε = 64/255 | 0.28 | 27% | 76% | 28% |
| *Cover photo alone (control)* | 1.00 | 0% | — | 11% |
| *Random-noise sticker (control)* | — | 0% | — | 9% |

**Findings**

1. **The trade-off is real and steep.** No sticker that stays close to the photo
   (SSIM ≥ 0.7) reaches even 5% top-1 success. Even with 60% of the sticker given
   over to the attack, the best design reaches 57%, 11 points short of the fully
   adversarial sticker.
2. **Where the attack goes matters more than how much.** At equal similarity, stripes
   beat blend, and blend beats a border, across the whole range. A 60% border changes
   more pixels than 60% stripes (84% vs. 64%) yet succeeds half as often (28% vs. 57%).
   Stripes spread adversarial texture across the sticker's whole footprint. A border
   pushes it to the edges, where it is small in the camera's view and gets cut by
   rotation. This matches the size threshold in the full-sticker robustness grid.
3. **Subtle blending saturates.** Doubling the blend budget from 32/255 to 64/255
   barely helps (23% → 27%). The dog photo's own features keep competing, and a
   bounded perturbation can't erase them.
4. **"Almost fooled" comes long before "fooled".** Top-5 success climbs far faster
   than top-1. Stripes at 20% put "toaster" in the top five 76% of the time but win
   only 16%. A system that acts on top-5 candidates, or on a low-confidence
   threshold, is much easier to attack than headline accuracy suggests.
5. **The stealthiest stickers do little beyond covering the scene.** The cover photo
   by itself makes the model lose the correct label 11% of the time, through
   occlusion alone. The most natural-looking stickers (blend ≤ 16/255, stripes 10%)
   reach only 12–13%. Disruption beyond occlusion appears once the attack becomes
   visible: 22% for a 20% border, 51% for 60% stripes.

**Caveats.** One target, one cover photo, one seed, and 600 optimization steps per
sticker. Stripes were the default region because they performed best in early
tests, so they're not an independent discovery. SSIM is a proxy for "looks
normal"; a human rating study would be the stronger measure.

## 2. Transfer and ensembles

*Results pending: the experiment is running.*

## 3. Defenses

See the defense-mode PR / `adai-sticker defend`: on the full "toaster" sticker, a
Grad-CAM-based patch detector cuts success from 67% to 4% at a 4-point clean-accuracy
cost. Blur cuts success to 42%, but an adaptive attacker restores it to 71%.
