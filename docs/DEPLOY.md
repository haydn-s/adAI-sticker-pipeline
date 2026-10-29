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

## What has been checked, and what hasn't

Checked before the first deploy:

- `deploy/requirements.txt` resolves for Linux x86-64 on Python 3.12 and 3.13, with
  both uv's and pip's index rules. It installs `torch 2.15.0+cpu` and no CUDA packages.
- In a fresh virtual environment with only those requirements (no project install, no
  FastAPI), the app renders its welcome, model and results pages.
- A full sticker on CPU, end to end in the browser, at phone and desktop widths, in
  dark mode.
- `tests/test_streamlit.py` drives the whole flow headless in CI, with a tiny stand-in
  model.

Not checked yet: an actual Community Cloud deployment. Its real speed, memory headroom
and the first-run Imagenette download will only show up there.

## What visitors get on Community Cloud

Community Cloud gives an app about **2 CPU cores and up to 2.7 GB of memory, with no
GPU**. The Streamlit docs give these limits as approximate. The app detects the
missing GPU and adapts in three ways.

**Only MobileNet V3 can be attacked.** It makes a working sticker in minutes. The
other five models still appear, marked "needs a GPU". Stickers are still *tested* on
them when "Also test on the other 5 AIs" is ticked. Optimizing against them on CPU
doesn't work well: in calibration runs, EfficientNet-B0 was 5× slower per step than
MobileNet V3 and reached only ~17% after 300 steps (11.5 minutes). ResNet-50 was
still near 0% after 400 steps of 16 photos (22 minutes). Set `ADAI_MODELS=all` to
allow every model anyway. CLIP's free-text targets need CLIP, so they need a GPU too.

**Lighter run lengths:**

| Preset | Steps | Photos per step | Test photos | Rotation × size grid |
|---|---|---|---|---|
| Quick (GPU) | 400 | 24 | 150 | 13 × 8 |
| Standard (GPU) | 1000 | 24 | 300 | 13 × 8 |
| Quick (CPU) | 300 | 12 | 48 | 7 × 4 |
| Thorough (CPU) | 500 | 16 | 96 | 7 × 4 |

**Transfer testing is off by default** because it adds several minutes.

How the CPU presets were chosen, with MobileNet V3 (unless noted) on this project's Mac,
limited to two CPU threads:

| Steps × photos per step | Fooling rate on practice photos at the end | Optimization time |
|---|---|---|
| 200 × 8 | ~0% (never got going) | ~2 min |
| 300 × 12 (Quick) | ~36%, still rising | 2.3–2.6 min |
| 400 × 16 | ~45% | not timed cleanly (ran alongside others) |
| 300 × 12, EfficientNet-B0 | ~17% | 11.5 min |

A complete Quick sticker made the way the app makes it (MobileNet V3, "toaster", 2
threads) took **3.7 minutes**: 2.5 optimizing and 1.2 testing. With "Also test on the
other 5 AIs" it took **5.6 minutes**. It fooled MobileNet on **25% and 44%** of held-out
photos in two runs, against 0% for both controls. At this budget, results vary a lot
from run to run.

Community Cloud's shared cores are likely slower than a recent Mac's. That hasn't been
measured yet, so time the first sticker after deploying. The app shows a live
time-left estimate during optimization, and the progress bar steps through each test.

**PyTorch is limited to 2 CPU threads** (`ADAI_TORCH_THREADS` overrides this). More
threads made these small batches slower: on a 6-core Mac, MobileNet V3 took 0.44 s
per step with 2 threads and 1.5 s with 6.

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
| `ADAI_MODELS` | auto | models visitors may attack, comma-separated, or `all`; auto = all with a GPU, only MobileNet V3 on CPU |
| `ADAI_MAX_QUEUE` | 4 | stickers allowed to wait in line |
| `ADAI_RUN_TTL_HOURS` | 24 | delete stickers and uploaded photos after this many hours |
| `ADAI_SHOW_ALL_RUNS` | off | `1` = the gallery lists every sticker on the server (useful for a live demo) |
| `ADAI_RUNS_DIR` | `runs/streamlit` | where new stickers are written |
| `ADAI_GALLERY_DIR` | `docs/examples` | finished example stickers to show |
| `ADAI_BACKGROUNDS` | Imagenette | a folder of practice photos, or `synthetic` for smoke tests |
| `ADAI_TORCH_THREADS` | 2 | CPU threads for PyTorch (1 if the container's CPU quota is 1) |

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
  Transfer testing loads every model in turn, including ViT-B/16 and CLIP (the largest).
  Check the logs under **Manage app**. Consider lowering `ADAI_MAX_QUEUE` or switching
  to gallery-only mode.
- **A results link says the sticker is gone.** The app restarted or the run expired;
  make it again.
