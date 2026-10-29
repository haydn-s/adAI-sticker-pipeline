# Physical testing protocol

Simulated success is a prediction. This protocol checks it against reality: print
the sticker, photograph it under controlled conditions, and score the photos with the
same model. It's written so a classmate or reviewer can repeat it.

## 1. Print

1. Open `runs/<run>/print_sheet.pdf` and print at **100% / actual size**. Turn off
   "fit to page" so the 4", 3", 2" and 1.5" sizes come out exact.
2. Record the printer, paper (plain, matte photo, glossy, sticker paper) and print
   quality setting. Glossy paper adds glare, which is itself a useful condition to test.
3. Cut along the gray lines. Optionally mount on cardboard so the sticker stays flat.

## 2. Set up scenes

Use 3–5 everyday scenes with a clear main object, e.g. a banana on a desk, a mug,
a bookshelf, a houseplant. Photograph each scene **without** the sticker first. Those
photos are the "no sticker" control and should be labeled as the real object.

## 3. Photograph

Shoot with a phone camera or the live demo. With the demo, press **s** to save the
exact frame the model saw:

```bash
uv sync --extra demo
uv run adai-sticker demo --target toaster --save-dir photos/shoot-01
```

Vary **one factor at a time** from a reference setup (3" sticker, about 50 cm away,
facing the camera, indoor light):

| Factor | Levels |
|---|---|
| Sticker size (`size_in`) | 4, 3, 2, 1.5 |
| Distance (`distance_cm`) | 30, 50, 100, 150 |
| Viewing angle (`angle_deg`) | 0, 30, 60 |
| Rotation (`rotation_deg`) | 0, 90, 180 |
| Lighting (`lighting`) | indoor, window, dim, outdoor |
| Placement (`placement`) | on object, next to object, held in hand |

Take 3 photos at each level, moving the sticker slightly between shots. That's about
60–80 photos per sticker, which takes about 30 minutes.

## 4. Record conditions

Make a `shoot.csv` next to the photos, with a `photo` column and one column per
factor. Leave a cell empty when it's at the reference value or not applicable:

```csv
photo,scene,size_in,distance_cm,angle_deg,rotation_deg,lighting,placement
photo-101500-123.jpg,banana,3,50,0,0,indoor,next to object
photo-101512-456.jpg,banana,3,100,0,0,indoor,next to object
photo-101530-789.jpg,banana,3,50,30,0,indoor,next to object
```

## 5. Score

```bash
uv run adai-sticker eval-photos --photos photos/shoot-01 --target toaster \
    --model resnet50 --meta photos/shoot-01/shoot.csv
```

This prints the overall success rate plus a breakdown by every column in
`shoot.csv`, and writes per-photo results to `results_resnet50.csv`. Annotated demo
frames under `annotated/` are skipped automatically. Re-run with `--model` set to
other models to measure physical transfer.

## 6. Compare with simulation

The simulator's sweeps (`fig_grid.png`, `fig_sweeps.png`) predict how size, angle
and lighting should affect success. For the write-up, plot physical success next to
simulated success for the same factor. The gap between them is the
**simulation-to-reality gap**, and it's a result in its own right.

To relate distance to the simulator's "sticker width as % of photo": in a few
photos, measure the sticker's width in pixels and divide by the width of the
model's center crop (the shorter side of the photo).

## Responsible testing

- Photograph your own objects in private spaces. Don't capture people's faces or
  other people's property without consent.
- Don't place stickers on vehicles, road signs, or anything a safety system might
  watch.
