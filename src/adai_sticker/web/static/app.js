// adAI Sticker Lab front end: a six-step guided flow over the JSON API in server.py.

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => [...document.querySelectorAll(sel)];
const pct = (x) => `${Math.round(x * 100)}%`;

const state = {
  options: null,
  ack: false,
  model: null,
  target: null, // { label, free_text }
  shape: "circle",
  palette: null,
  coverMode: "none",
  coverFile: null,
  region: "stripes",
  regionAmount: 0.4,
  epsilon: 24, // in /255 units
  preset: "quick",
  testOthers: true,
  jobId: null,
  furthest: 0,
};

const STEPS = ["welcome", "model", "target", "design", "make", "results"];

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.error || `Request failed (${res.status})`);
  return body;
}

// -- navigation ----------------------------------------------------------------

function show(step) {
  $$(".step").forEach((s) => (s.hidden = s.id !== `step-${step}`));
  const idx = STEPS.indexOf(step);
  if (idx > state.furthest) state.furthest = idx;
  $$(".stepper button").forEach((b) => {
    const i = STEPS.indexOf(b.dataset.goto);
    b.classList.toggle("current", b.dataset.goto === step);
    b.disabled = i > state.furthest || (i > 0 && !state.ack);
  });
  if (step === "design") refreshPreview();
  if (step === "make") renderSummary();
  if (step === "gallery") loadGallery();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

document.addEventListener("click", (e) => {
  const t = e.target.closest("[data-goto],[data-next],[data-back]");
  if (!t || t.disabled) return;
  e.preventDefault();
  show(t.dataset.goto || t.dataset.next || t.dataset.back);
});

// -- welcome -------------------------------------------------------------------

function renderWelcome() {
  const list = $("#blocked-list");
  list.innerHTML = "";
  for (const b of state.options.blocked_categories) {
    const div = document.createElement("div");
    div.innerHTML = `<strong></strong>: <span></span> <span class="muted"></span>`;
    div.querySelector("strong").textContent = b.category;
    div.querySelectorAll("span")[0].textContent = b.reason;
    div.querySelectorAll("span")[1].textContent = `(e.g. ${b.examples.join(", ")})`;
    list.append(div);
  }
  if (!state.options.allow_generate) {
    $("#gallery-only-note").hidden = false;
    $("#start").textContent = "Browse the gallery";
  }
  $("#ethics-ack").addEventListener("change", (e) => {
    state.ack = e.target.checked;
    $("#start").disabled = !state.ack;
  });
  $("#start").addEventListener("click", () => show(state.options.allow_generate ? "model" : "gallery"));
}

// -- 1. model ------------------------------------------------------------------

function renderModels() {
  const wrap = $("#model-cards");
  for (const m of state.options.models) {
    const btn = document.createElement("button");
    btn.className = "card model-card";
    btn.innerHTML = `<h3></h3><div class="meta"></div><p></p>`;
    btn.querySelector("h3").textContent = m.display_name;
    if (m.family === "clip") {
      const badge = document.createElement("span");
      badge.className = "badge";
      badge.textContent = "reads any words";
      btn.querySelector("h3").append(badge);
    }
    btn.querySelector(".meta").textContent = `${m.year} · ${Math.round(m.params_millions)} million parameters`;
    btn.querySelector("p").textContent = m.description;
    btn.addEventListener("click", () => {
      state.model = m;
      $$(".model-card").forEach((c) => c.classList.toggle("selected", c === btn));
      $("#model-next").disabled = false;
      $("#free-text-box").hidden = m.family !== "clip";
      if (state.target) checkTarget(state.target.label); // re-check under the new model
    });
    wrap.append(btn);
  }
}

// -- 2. target -----------------------------------------------------------------

async function checkTarget(label, display = label) {
  const status = $("#target-status");
  const res = await api("/api/check-target", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ target: label, model: state.model?.key || "resnet50" }),
  });
  status.className = "status " + (res.allowed ? "good" : "bad");
  if (res.allowed) {
    state.target = res;
    status.textContent = `Target: “${res.label}”. The sticker will try to make ${state.model?.display_name || "the AI"} see this.`;
  } else {
    state.target = null;
    status.textContent = res.category === "unknown" ? res.reason : `“${display}” is blocked (${res.category}). ${res.reason}`;
  }
  $("#target-next").disabled = !state.target;
}

function renderTargets() {
  const chips = $("#featured");
  for (const label of state.options.featured_targets) {
    const b = document.createElement("button");
    b.textContent = label;
    b.addEventListener("click", () => checkTarget(label));
    chips.append(b);
  }
  let timer;
  $("#label-search").addEventListener("input", (e) => {
    clearTimeout(timer);
    timer = setTimeout(async () => {
      const q = e.target.value.trim();
      const list = $("#label-results");
      list.innerHTML = "";
      if (!q) return;
      const { results } = await api(`/api/labels?q=${encodeURIComponent(q)}`);
      for (const r of results) {
        const li = document.createElement("li");
        const b = document.createElement("button");
        b.textContent = r.label;
        b.addEventListener("click", () => { list.innerHTML = ""; checkTarget(String(r.index), r.label); });
        li.append(b);
        list.append(li);
      }
      if (!results.length) list.innerHTML = `<li class="muted small" style="padding:8px 12px">No matching labels</li>`;
    }, 200);
  });
  $("#free-text-use").addEventListener("click", () => {
    const v = $("#free-text").value.trim();
    if (v) checkTarget(v);
  });
}

// -- 3. design -----------------------------------------------------------------

function renderDesign() {
  const shapes = $("#shapes");
  for (const s of state.options.shapes) {
    const b = document.createElement("button");
    b.innerHTML = `<img alt="" src="/api/shape/${s}.png"><span>${s}</span>`;
    b.classList.toggle("selected", s === state.shape);
    b.addEventListener("click", () => {
      state.shape = s;
      $$("#shapes button").forEach((x) => x.classList.toggle("selected", x === b));
      refreshPreview();
    });
    shapes.append(b);
  }

  const palettes = $("#palettes");
  const addPalette = (key, label, colors) => {
    const b = document.createElement("button");
    b.dataset.palette = key ?? "";
    b.innerHTML = `<span></span><span class="swatches"></span>`;
    b.firstChild.textContent = label;
    for (const c of colors) {
      const sw = document.createElement("span");
      sw.style.background = c;
      b.lastChild.append(sw);
    }
    b.classList.toggle("selected", key === state.palette);
    b.addEventListener("click", () => {
      state.palette = key;
      $$("#palettes button").forEach((x) => x.classList.toggle("selected", x === b));
    });
    palettes.append(b);
    return b;
  };
  addPalette(null, "Any colors (strongest)", ["#e3241b", "#ffe600", "#00933c", "#1d3f9a", "#e0007a"]);
  for (const [k, colors] of Object.entries(state.options.palettes)) addPalette(k, k, colors);
  const auto = addPalette("auto", "Match my photo", ["#8a6d4f", "#c9b79a", "#3d4a2f", "#ece5d8"]);
  auto.disabled = true;

  $$("input[name=cover]").forEach((r) => r.addEventListener("change", () => {
    state.coverMode = r.value;
    $("#cover-controls").hidden = state.coverMode === "none";
    $("#region-controls").hidden = state.coverMode !== "region";
    $("#blend-controls").hidden = state.coverMode !== "blend";
    // Palettes only apply to adversarial pixels; blend mode perturbs the photo itself.
    $$("#palettes button").forEach((b) => (b.disabled = state.coverMode === "blend" && b.dataset.palette !== "" ||
      (b.dataset.palette === "auto" && !state.coverFile)));
    if (state.coverMode === "blend") selectPalette(null);
    refreshPreview();
  }));
  $("#cover-file").addEventListener("change", (e) => {
    state.coverFile = e.target.files[0] || null;
    const auto = $$("#palettes button").find((b) => b.dataset.palette === "auto");
    auto.disabled = !state.coverFile || state.coverMode === "blend";
    refreshPreview();
  });

  const region = $("#region");
  for (const r of state.options.regions) region.append(new Option(r, r, r === state.region, r === state.region));
  region.addEventListener("change", () => { state.region = region.value; refreshPreview(); });
  const amount = $("#region-amount");
  const showAmount = () => ($("#region-amount-out").textContent = pct(amount.value));
  amount.addEventListener("input", () => { state.regionAmount = Number(amount.value); showAmount(); refreshPreview(); });
  showAmount();
  const eps = $("#epsilon");
  const showEps = () => ($("#epsilon-out").textContent = `up to ${eps.value} of 255 brightness levels`);
  eps.addEventListener("input", () => { state.epsilon = Number(eps.value); showEps(); });
  showEps();

  const preset = $("#preset");
  for (const [k, label] of Object.entries(state.options.presets)) preset.append(new Option(label, k, k === state.preset, k === state.preset));
  preset.addEventListener("change", () => (state.preset = preset.value));
  $("#test-others").addEventListener("change", (e) => (state.testOthers = e.target.checked));
}

function selectPalette(key) {
  state.palette = key;
  $$("#palettes button").forEach((b) => b.classList.toggle("selected", (b.dataset.palette || null) === key));
}

let previewTimer;
let previewUrl;
function refreshPreview() {
  clearTimeout(previewTimer);
  previewTimer = setTimeout(async () => {
    const fd = new FormData();
    fd.append("shape", state.shape);
    fd.append("cover_mode", state.coverMode);
    fd.append("region", state.region);
    fd.append("region_amount", state.regionAmount);
    if (state.coverFile && state.coverMode !== "none") fd.append("cover", state.coverFile);
    const res = await fetch("/api/design-preview", { method: "POST", body: fd });
    if (!res.ok) return;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = URL.createObjectURL(await res.blob());
    $("#design-preview").src = previewUrl;
  }, 150);
}

// -- 4. make -------------------------------------------------------------------

function settingsPayload() {
  return {
    model: state.model.key,
    target: state.target.label,
    shape: state.shape,
    palette: state.palette,
    cover_mode: state.coverMode,
    region: state.region,
    region_amount: state.regionAmount,
    epsilon: state.epsilon / 255,
    preset: state.preset,
    test_other_models: state.testOthers,
    ethics_ack: state.ack,
  };
}

function renderSummary() {
  const modeText = { none: "Pure attack", region: `Hidden in my photo (${state.region}, ${pct(state.regionAmount)})`, blend: `Blended (±${state.epsilon}/255)` };
  const items = [
    ["AI to fool", state.model?.display_name],
    ["Target label", state.target?.label],
    ["Shape", state.shape],
    ["Look", modeText[state.coverMode]],
    ["Colors", state.palette ?? "any"],
    ["Thoroughness", state.options.presets[state.preset]],
  ];
  $("#make-summary").innerHTML = "";
  for (const [k, v] of items) {
    const div = document.createElement("div");
    div.innerHTML = "<dt></dt><dd></dd>";
    div.querySelector("dt").textContent = k;
    div.querySelector("dd").textContent = v ?? "—";
    $("#make-summary").append(div);
  }
  $("#explain-target").textContent = state.target ? `“${state.target.label}”` : "your target";
}

async function startJob() {
  $("#make-error").hidden = true;
  if (state.coverMode !== "none" && !state.coverFile) {
    showError("Upload a photo on the Design step, or choose “No: pure attack”.");
    return;
  }
  const fd = new FormData();
  fd.append("settings", JSON.stringify(settingsPayload()));
  if (state.coverFile && (state.coverMode !== "none" || state.palette === "auto")) fd.append("cover", state.coverFile);
  try {
    $("#make-go").disabled = true;
    const job = await api("/api/jobs", { method: "POST", body: fd });
    state.jobId = job.id;
    $("#make-actions").hidden = true;
    $("#progress").hidden = false;
    poll();
  } catch (err) {
    $("#make-go").disabled = false;
    showError(err.message);
  }
}

function showError(msg) {
  const el = $("#make-error");
  el.textContent = msg;
  el.hidden = false;
}

async function poll() {
  let job;
  try {
    job = await api(`/api/jobs/${state.jobId}`);
  } catch (err) {
    showError(err.message);
    return;
  }
  const frac = job.total_steps ? job.step / job.total_steps : 0;
  const overall = job.status === "evaluating" ? 0.85 : job.status === "done" ? 1 : frac * 0.85;
  $("#progress-fill").style.width = `${Math.round(overall * 100)}%`;
  $("#progress-message").textContent = job.status === "queued"
    ? `Waiting in line (position ${job.queue_position ?? "?"})` : job.message;
  $("#progress-detail").textContent = job.status === "optimizing"
    ? `Step ${job.step} of ${job.total_steps} · ${Math.round(job.elapsed)} s` : `${Math.round(job.elapsed)} s`;
  if (job.status !== "queued") $("#live-sticker").src = `/api/jobs/${state.jobId}/preview.png?t=${Date.now()}`;
  drawCurve(job.success_curve, job.total_steps);
  if (job.status === "done") return loadResults(`/runs/${state.jobId}/`);
  if (job.status === "failed") return showError(`Something went wrong: ${job.error}`);
  setTimeout(poll, 1200);
}

function drawCurve(points, total) {
  const svg = $("#curve");
  const W = 320, H = 120, L = 30, B = 18, T = 6;
  const x = (s) => L + (s / Math.max(1, total)) * (W - L - 6);
  const y = (v) => T + (1 - v) * (H - T - B);
  // Smooth the noisy per-batch rate with a short moving average.
  const sm = points.map((p, i) => {
    const win = points.slice(Math.max(0, i - 4), i + 1);
    return [p[0], win.reduce((a, q) => a + q[1], 0) / win.length];
  });
  let html = "";
  for (const v of [0, 0.5, 1]) {
    html += `<line class="grid" x1="${L}" x2="${W - 6}" y1="${y(v)}" y2="${y(v)}"/>`;
    html += `<text x="${L - 6}" y="${y(v) + 3}" text-anchor="end">${pct(v)}</text>`;
  }
  if (sm.length > 1) {
    const d = sm.map((p, i) => `${i ? "L" : "M"}${x(p[0]).toFixed(1)},${y(p[1]).toFixed(1)}`).join("");
    html += `<path class="area" d="${d}L${x(sm.at(-1)[0])},${y(0)}L${x(sm[0][0])},${y(0)}Z"/>`;
    html += `<path class="line" d="${d}"/>`;
  }
  svg.innerHTML = html;
}

// -- 5. results ----------------------------------------------------------------

async function loadResults(base) {
  const report = await api(`${base}report.json`);
  const files = await api(`${base}files.json`).catch(() => ({}));
  const primary = report.models[0];
  const ev = report.evaluation[primary];
  const head = ev.headline;
  const ctl = ev.controls;
  const modelName = (k) => state.options.models.find((m) => m.key === k)?.display_name ?? k;

  $("#result-sticker").src = `${base}sticker.png`;
  $("#result-eyebrow").textContent = `“${report.target.label}” sticker vs. ${modelName(primary)}`;
  $("#result-number").textContent = pct(head.success_rate);
  $("#result-sentence").textContent =
    `of new photos made ${modelName(primary)} say “${report.target.label}”. Without the sticker: ` +
    `${pct(ctl["no sticker"].success_rate)}. With a random sticker of the same shape: ${pct(ctl["random-noise sticker"].success_rate)}.`;

  const dl = $("#downloads");
  dl.innerHTML = "";
  const link = (href, text) => {
    const a = document.createElement("a");
    a.className = "button";
    a.href = href;
    a.download = "";
    a.textContent = text;
    dl.append(a);
  };
  if (files.print_sheet) link(`${base}${files.print_sheet}`, "Print sheet (PDF)");
  if (files.png_print) link(`${base}${files.png_print}`, "Sticker (PNG, print size)");
  link(`${base}report_card.png`, "Report card");
  link(`${base}report.md`, "Full report");

  const setFig = (id, file) => { const img = $(id); img.src = `${base}${file}`; img.closest("figure").hidden = false; };
  setFig("#fig-examples", "examples.png");
  setFig("#fig-gradcam", "fig_gradcam.png");
  if (ev.grid) setFig("#fig-grid", "fig_grid.png"); else $("#fig-grid").closest("figure").hidden = true;
  if (ev.sweeps) setFig("#fig-sweeps", "fig_sweeps.png"); else $("#fig-sweeps").closest("figure").hidden = true;
  const multi = Object.keys(report.evaluation).length > 1;
  $("#transfer-section").hidden = !multi;
  if (multi) setFig("#fig-transfer", "fig_transfer.png");

  const controls = $("#controls");
  controls.innerHTML = "";
  for (const [name, m] of Object.entries(ctl)) {
    const card = document.createElement("div");
    card.className = "card";
    card.innerHTML = `<div class="stat-label"></div><div class="stat-value"></div><div class="muted small"></div>`;
    card.querySelector(".stat-label").textContent = name;
    card.querySelector(".stat-value").textContent = pct(m.success_rate);
    card.querySelector(".small").textContent = `labeled “${report.target.label}” · correct label lost ${pct(m.disruption_rate)}`;
    controls.append(card);
  }

  const ex = report.explanation;
  $("#attention-sentence").textContent = ex
    ? `The sticker covers about ${pct(ex.sticker_area)} of each photo, but it gets ${pct(ex.share_on_sticker)} of the ` +
      `AI's attention when it answers “${report.target.label}”, ${ex.ratio.toFixed(1)}× more than its size would suggest. ` +
      `Brighter colors in the heat maps mean "this part mattered most."`
    : "";

  state.furthest = STEPS.indexOf("results");
  show("results");
}

async function loadGallery() {
  const wrap = $("#gallery");
  wrap.innerHTML = `<p class="muted">Loading…</p>`;
  const { items } = await api("/api/gallery");
  wrap.innerHTML = items.length ? "" : `<p class="muted">No finished stickers yet.</p>`;
  for (const it of items) {
    const b = document.createElement("button");
    b.className = "card";
    b.innerHTML = `<img alt=""><span class="pct"></span><span class="small"></span>`;
    b.querySelector("img").src = `${it.base}sticker.png`;
    b.querySelector(".pct").textContent = pct(it.success_rate);
    b.querySelector(".small").textContent = `“${it.target}” vs. ${it.model}`;
    b.addEventListener("click", () => loadResults(it.base));
    wrap.append(b);
  }
}

// -- boot ------------------------------------------------------------------------

async function boot() {
  state.options = await api("/api/options");
  renderWelcome();
  renderModels();
  renderTargets();
  renderDesign();
  $("#make-go").addEventListener("click", startJob);
  $("#again").addEventListener("click", () => {
    state.jobId = null;
    $("#make-actions").hidden = false;
    $("#make-go").disabled = false;
    $("#progress").hidden = true;
    $("#curve").innerHTML = "";
    $("#live-sticker").removeAttribute("src");
    show("design");
  });
  show("welcome");
}

boot().catch((err) => {
  document.querySelector("main").innerHTML = `<div class="status bad">Could not reach the server: ${err.message}</div>`;
});
