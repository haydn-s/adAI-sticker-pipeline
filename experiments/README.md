# Experiments

Scripts that produce the project's evidence (see the [roadmap](../docs/ROADMAP.md)).
Each one writes a folder under `runs/` containing one sub-run per configuration
(full sticker reports) plus `results.csv`, `results.json` and a summary figure.

| Script | Question | Approx. time (M-series GPU) |
|---|---|---|
| `stealth_tradeoff.py` | How much success does a sticker give up as it looks more like a normal photo (region amount, blend ε)? | ~2 min per setting |
| `transfer_matrix.py` | Do stickers fool models they never saw, and do ensembles help? | ~3–6 min per source on an M-series GPU |
| `replicate.py` | Do those answers hold across targets and seeds? Runs both scripts for 3 targets × 3 seeds on a subset of settings, then pools them (mean, min–max, paired per-run comparisons) | ~3.5 h in total; `--summarize-only` needs no GPU |

```bash
uv run python experiments/stealth_tradeoff.py --target toaster
uv run python experiments/transfer_matrix.py --target toaster
uv run python experiments/replicate.py                    # writes docs/results/{stealth,transfer}/
uv run python experiments/replicate.py --summarize-only   # re-pool finished runs and redraw
```

Both scripts reuse stickers already finished in their `--out` folder (`--fresh` to redo),
so an interrupted run picks up where it stopped, and `replicate.py` never remakes a
sticker it already has.

Both default to a dog photo from Imagenette's validation split as the cover image
(the project's running example). Pass `--cover` to use your own. Every run uses
fixed seeds, and `--seed` repeats an experiment with different randomness.
