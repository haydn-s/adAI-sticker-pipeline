# Hosting the Sticker Lab on Streamlit

The guided app also runs as a Streamlit app, ready for
[Streamlit Community Cloud](https://streamlit.io/cloud). It has the same six steps,
wording and safety checks as the FastAPI app (`adai-sticker serve`). Both use
`src/adai_sticker/web/common.py`, so they stay in step.

| File | Purpose |
|---|---|
| `deploy/streamlit_app.py` | Entrypoint. Community Cloud runs it from the repo root. |
| `deploy/requirements.txt` | What Community Cloud installs, including CPU-only PyTorch. |
| `.streamlit/config.toml` | Theme (light and dark), 10 MB upload limit, toolbar. |
| `src/adai_sticker/web/streamlit_ui.py` | The app's pages. |

## Try it locally first

```bash
uv sync --extra streamlit
uv run streamlit run deploy/streamlit_app.py
```

On a Mac or a CUDA machine it uses the GPU and the same run lengths as the FastAPI
app. To preview what Community Cloud visitors will get, force the CPU:

```bash
ADAI_DEVICE=cpu uv run streamlit run deploy/streamlit_app.py
```

## Deploy to Community Cloud

1. Merge the web-app branches into `main` and push to GitHub.
2. Sign in at [share.streamlit.io](https://share.streamlit.io) with GitHub, then choose
   **Create app → Deploy a public app from GitHub**.
3. Choose the repository `haydn-s/adAI-sticker-pipeline`, branch `main`, and main file
   path **`deploy/streamlit_app.py`**.
4. Under **Advanced settings**, pick **Python 3.12**. Optionally paste settings into
   **Secrets** (see below), for example:

   ```toml
   ADAI_MAX_QUEUE = "4"
   ADAI_RUN_TTL_HOURS = "24"
   ```

5. Click **Deploy**. The first build takes several minutes, mostly installing PyTorch.

Why the entrypoint lives in `deploy/`: Community Cloud uses the dependency file next to
the entrypoint before the `uv.lock` at the repo root. The lock file would install
GPU PyTorch (several GB of CUDA libraries) and no Streamlit. `tests/test_deploy.py`
checks that `deploy/requirements.txt` stays in step with `pyproject.toml`.

## What visitors get on Community Cloud

Community Cloud gives an app about **2 CPU cores and up to 2.7 GB of memory, with no
GPU**. The Streamlit docs give these limits as approximate. The app detects the CPU
and switches to lighter run lengths:

| Preset | Steps | Photos per step | Test photos | Test grid |
|---|---|---|---|---|
| Quick (GPU) | 400 | 24 | 150 | 13 angles × 8 sizes |
| Quick (CPU) | CPU_QUICK_STEPS | CPU_QUICK_BATCH | 48 | 7 angles × 4 sizes |
| Thorough (CPU) | CPU_STD_STEPS | CPU_STD_BATCH | 96 | 7 angles × 4 sizes |

CALIBRATION_NOTES

On CPU, model cards are tagged fast, medium or slow. "Also test on the other 5 AIs" is
off by default there, because it adds several minutes.

The details:

- **One sticker at a time.** A single background worker makes stickers. Other
  visitors see their place in line (up to `ADAI_MAX_QUEUE` waiting). The page shows
  live progress and an estimate of the time left.
- **First sticker after a restart is slower.** The server downloads Imagenette, the
  practice and test photos (about 340 MB, a minute or two). Model weights (20–350 MB
  each) download the first time each model is used.
- **Sleeps when idle.** Apps with no traffic for 12 hours go to sleep; the first
  visit wakes them up, which takes a minute.
- **Files are temporary.** Stickers and uploaded photos are deleted after
  `ADAI_RUN_TTL_HOURS`, and whenever Community Cloud restarts the app.

### Before a user-testing session

1. Open the app 10 minutes early to wake it up.
2. Make one Quick sticker with MobileNet V3. This downloads the photos and weights,
   and shows the real speed on the day.
3. If it's too slow for your session, set `ADAI_GALLERY_ONLY = "1"` in Secrets. The
   app then becomes a browse-only gallery, which still has a printable example sticker.

## Settings

All settings are optional environment variables. On Community Cloud, put them in the
app's Secrets as top-level keys; Streamlit exposes those as environment variables.

| Variable | Default | Meaning |
|---|---|---|
| `ADAI_GALLERY_ONLY` | off | `1` = browse finished stickers only; nothing is computed |
| `ADAI_DEVICE` | auto | `cpu`, `cuda` or `mps`; auto picks CUDA, then Apple MPS, then CPU |
| `ADAI_MAX_QUEUE` | 4 | stickers allowed to wait in line |
| `ADAI_RUN_TTL_HOURS` | 24 | delete stickers and uploaded photos after this many hours |
| `ADAI_SHOW_ALL_RUNS` | off | `1` = the gallery lists every sticker on the server (useful for a live demo) |
| `ADAI_RUNS_DIR` | `runs/streamlit` | where new stickers are written |
| `ADAI_GALLERY_DIR` | `docs/examples` | finished example stickers to show |
| `ADAI_BACKGROUNDS` | Imagenette | a folder of practice photos, or `synthetic` for smoke tests |
| `ADAI_TORCH_THREADS` | detected | CPU threads for PyTorch (defaults to the container's CPU quota) |

## Privacy and safety

- Visitors must accept the responsible-use guidance before making anything. Blocked
  targets are refused inside the pipeline itself, with a plain-language reason.
- The gallery shows each visitor only their own stickers plus the curated examples.
  Other people's stickers, which may contain their uploaded photos, are never listed.
- A results link (`…/results?run=<id>`) opens that sticker for anyone who has the link.
  The id is random, 16 hex characters. Treat the link like an unlisted video.
- Uploaded photos are stored with the sticker on the server and deleted with it.

## If Community Cloud is too slow

The same app runs anywhere `streamlit run` works. On a machine with a GPU it switches
to the GPU run lengths automatically. For the class presentation, the FastAPI app on a
laptop (`adai-sticker serve`) or the Streamlit app run locally gives the fastest live
demo.

## Troubleshooting

- **Build fails resolving `torch`.** Check that `deploy/requirements.txt` still starts
  with the `--extra-index-url https://download.pytorch.org/whl/cpu` line.
- **"Oh no" or a restart in the middle of a sticker.** The app probably ran out of memory.
  The largest models (ViT-B/16 and CLIP) plus transfer testing are the heaviest load.
  Check the logs under **Manage app**. Consider lowering `ADAI_MAX_QUEUE` or switching
  to gallery-only mode.
- **A results link says the sticker is gone.** The app restarted or the run expired;
  make it again.
