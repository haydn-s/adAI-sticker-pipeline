# Experiments

Scripts that produce the project's evidence (see the [roadmap](../docs/ROADMAP.md)).
Each one writes a folder under `runs/` containing one sub-run per configuration
(full sticker reports) plus `results.csv`, `results.json` and a summary figure.

| Script | Question | Approx. time (M-series GPU) |
|---|---|---|
| `stealth_tradeoff.py` | How much success does a sticker give up as it looks more like a normal photo (region amount, blend ε)? | ~2 min per setting |
| `transfer_matrix.py` | Do stickers fool models they never saw, and do ensembles help? | ~5–15 min per source on an M-series GPU; reuses finished sources (`--fresh` to redo) |

```bash
uv run python experiments/stealth_tradeoff.py --target toaster
uv run python experiments/transfer_matrix.py --target toaster
```

Both default to a dog photo from Imagenette's validation split as the cover image
(the project's running example). Pass `--cover` to use your own. Every run uses
fixed seeds, and `--seed` repeats an experiment with different randomness.
