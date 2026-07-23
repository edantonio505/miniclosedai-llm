"use strict";
/* miniclosedai-llm control plane — vanilla JS, no build step.
   Talks to the FastAPI manager: list/add/start/stop/remove models, stream logs
   over SSE, run a vision test, copy the base_url for miniclosedai. */

const THEME_KEY = "miniclosedai-llm:theme";
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

// per-model UI state: id -> { node, es (EventSource), file (chosen test image) }
const cards = new Map();
// model ids that failed to download because a gated repo needs an HF token — once
// a token is saved these are retried automatically (see saveHFToken).
const gatedModels = new Set();
let refreshTimer = null;

// --------------------------------------------------------------------- helpers
async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let detail = `HTTP ${r.status}`;
    try { const b = await r.json(); if (b.detail !== undefined) detail = b.detail; } catch (e) {}
    // detail may be a string OR a structured object ({message, analysis}).
    const msg = (detail && detail.message) || (typeof detail === "string" ? detail : JSON.stringify(detail));
    const err = new Error(msg);
    err.status = r.status;
    err.detail = detail;
    throw err;
  }
  return r.status === 204 ? null : r.json();
}

let toastTimer = null;
function toast(msg, kind = "") {
  const el = $("#toast");
  el.textContent = msg;
  el.className = "toast" + (kind ? " is-" + kind : "");
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, 4000);
}

// Clipboard with a fallback: navigator.clipboard only exists on HTTPS/localhost,
// so over http://<LAN-IP> we fall back to a temporary textarea + execCommand.
function copyText(text) {
  if (navigator.clipboard && window.isSecureContext) {
    return navigator.clipboard.writeText(text);
  }
  return new Promise((resolve, reject) => {
    try {
      const ta = document.createElement("textarea");
      ta.value = text;
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.focus(); ta.select();
      const ok = document.execCommand("copy");
      document.body.removeChild(ta);
      ok ? resolve() : reject(new Error("copy failed"));
    } catch (e) { reject(e); }
  });
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// --------------------------------------------------------------------- theme
function applyTheme(mode) {
  const systemDark = matchMedia("(prefers-color-scheme: dark)").matches;
  const dark = mode === "dark" || (mode === "system" && systemDark);
  document.documentElement.classList.toggle("dark", dark);
  for (const k of ["light", "dark", "system"])
    $("#theme-icon-" + k).style.display = k === mode ? "" : "none";
  $("#theme-toggle").title = "Theme: " + mode;
}
function initTheme() {
  let mode = localStorage.getItem(THEME_KEY) || "system";
  applyTheme(mode);
  $("#theme-toggle").addEventListener("click", () => {
    const order = ["system", "light", "dark"];
    mode = order[(order.indexOf(mode) + 1) % order.length];
    localStorage.setItem(THEME_KEY, mode);
    applyTheme(mode);
  });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    if ((localStorage.getItem(THEME_KEY) || "system") === "system") applyTheme("system");
  });
}

// --------------------------------------------------------------------- banner
async function loadBanner() {
  const el = $("#banner");
  let h;
  try { h = await api("/api/health"); }
  catch (e) { el.className = "banner bad"; el.textContent = "Manager unreachable: " + e.message; return; }

  let gpuTxt = "";
  try {
    const g = await api("/api/gpu");
    if (g.gpus && g.gpus.length) {
      gpuTxt = g.gpus.map((x) => {
        const mem = (x.mem_total_mb == null)
          ? "unified memory" : `${x.mem_used_mb}/${x.mem_total_mb} MB`;
        return `GPU${x.index} ${x.name} — ${mem} (${x.util_pct}%)`;
      }).join(" · ");
    } else { gpuTxt = "GPU: " + (g.error || "not detected"); }
  } catch (e) { gpuTxt = "GPU: unknown"; }

  let cls = "ok", msg;
  if (h.no_engine) {
    cls = "bad";
    msg = "No launch engine — run ./setup_shim.sh (bare-metal, any model) or ./setup_llamacpp.sh (GGUF), or install Docker / `pip install vllm`.";
  } else if (!h.gpu_ok) {
    cls = "warn";
    msg = `Engine ready, but no GPU detected (${gpuTxt}). Models will fail to load until the GPU/driver works.`;
  } else {
    msg = "Ready.";
  }
  const engLabel = h.engine === "docker" ? "Docker engine"
    : h.engine === "native" ? "Native (vllm serve)"
    : h.engine === "shim" ? "Native (transformers shim)"
    : h.engine;
  const net = h.dashboard_url
    ? `<span class="gpu-readout">· reachable at ${escapeHtml(h.dashboard_url)}</span>` : "";
  const ggufTxt = h.llamacpp_ok ? "llama.cpp ready"
    : h.llamacpp_building ? `building llama.cpp… ${h.llamacpp_progress || ""}`.trim()
    : "run ./setup_llamacpp.sh";
  const gguf = `<span class="gpu-readout${h.llamacpp_building ? " building" : ""}">· GGUF/ternary: ${escapeHtml(ggufTxt)}</span>`;
  const shimTxt = h.shim_ok ? "shim ready"
    : h.shim_building ? `installing shim… ${h.shim_progress || ""}`.trim()
    : "run ./setup_shim.sh";
  const shim = `<span class="gpu-readout${h.shim_building ? " building" : ""}">· safetensors (bare-metal): ${escapeHtml(shimTxt)}</span>`;
  el.className = "banner " + cls;
  el.innerHTML =
    `<span class="engine-badge">${escapeHtml(engLabel)}</span>` +
    `<span class="pill">${escapeHtml(msg)}</span>` +
    `<span class="gpu-readout">${escapeHtml(gpuTxt)}</span>` +
    net + gguf + shim +
    (h.runpod ? `<span class="gpu-readout">· RunPod (base URLs use the pod proxy)</span>` : "");
}

// --------------------------------------------------------------------- hugging face token
function openTokenPanel() {
  $("#hf-token-panel").hidden = false;
  const inp = $("#hf-token-input");
  inp.focus();
  try { inp.scrollIntoView({ behavior: "smooth", block: "center" }); } catch (e) {}
}

async function loadHFToken() {
  const statusEl = $("#hf-token-status");
  let s;
  try { s = await api("/api/hf-token"); }
  catch (e) { statusEl.textContent = ""; return; }
  if (!s.present) {
    statusEl.textContent = "· no token set — required for gated models";
    statusEl.className = "hint warn-text";
    $("#hf-token-panel").hidden = false;   // nudge: reveal the input when none is set
  } else if (s.user) {
    statusEl.textContent = `· signed in as ${s.user} (${s.masked})`;
    statusEl.className = "hint";
  } else {
    statusEl.textContent = `· token set (${s.masked})` + (s.valid === false ? " — unverified" : "");
    statusEl.className = "hint";
  }
}

async function saveHFToken() {
  const inp = $("#hf-token-input");
  const msg = $("#hf-token-msg");
  const token = inp.value.trim();
  if (!token) { msg.hidden = false; msg.className = "hint warn-text"; msg.textContent = "Paste a token first."; return; }
  const btn = $("#hf-token-save");
  btn.classList.add("is-busy");
  msg.hidden = false; msg.className = "hint"; msg.textContent = "Saving…";
  try {
    const r = await api("/api/hf-token", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ token }),
    });
    inp.value = "";
    if (r.valid && r.user) {
      msg.className = "hint"; msg.textContent = `Saved — signed in as ${r.user}. It'll be used for the next download automatically.`;
      toast("Hugging Face token saved", "ok");
    } else {
      msg.className = "hint warn-text";
      msg.textContent = "Saved, but couldn't verify it with Hugging Face right now (network?). "
        + "It'll still be used on the next download.";
      toast("Token saved (unverified)", "ok");
    }
    if (r.warning) { msg.className = "hint warn-text"; msg.textContent += " " + r.warning; }
    await loadHFToken();
    // Auto-retry any model that failed because it was gated — the download now
    // continues on its own with the token we just saved.
    const toRetry = [...gatedModels];
    for (const id of toRetry) {
      try { await api(`/api/models/${encodeURIComponent(id)}/start`, { method: "POST" }); }
      catch (e) { /* leave it in error state; the user can hit Retry */ }
    }
    gatedModels.clear();
    if (toRetry.length) {
      msg.textContent += ` Retrying ${toRetry.length} model${toRetry.length > 1 ? "s" : ""} now…`;
      toast(`Retrying ${toRetry.length} model${toRetry.length > 1 ? "s" : ""} with the new token`, "ok");
    }
    // Gated model waiting to be analyzed? re-check it now that we're authenticated.
    if ($("#hf-id").value.trim()) onAnalyze();
    loadBanner();
    loadModels();
  } catch (e) {
    msg.className = "hint bad"; msg.textContent = e.message;
  } finally {
    btn.classList.remove("is-busy");
  }
}

async function clearHFToken() {
  if (!confirm("Remove the saved Hugging Face token? Gated models won't download until you set one again.")) return;
  try { await api("/api/hf-token", { method: "DELETE" }); }
  catch (e) { toast(e.message, "error"); return; }
  $("#hf-token-msg").hidden = true;
  toast("Token cleared", "ok");
  loadHFToken();
}

// --------------------------------------------------------------------- add model
function readAdvanced() {
  const num = (sel) => { const v = $(sel).value.trim(); return v === "" ? undefined : Number(v); };
  const str = (sel) => { const v = $(sel).value.trim(); return v === "" ? undefined : v; };
  const params = {};
  const maxlen = num("#adv-maxlen"); if (maxlen !== undefined) params.max_model_len = maxlen;
  const gpumem = num("#adv-gpumem"); if (gpumem !== undefined) params.gpu_memory_util = gpumem;
  const tp = num("#adv-tp"); if (tp !== undefined) params.tensor_parallel = tp;
  const maximg = num("#adv-maximg"); if (maximg !== undefined) params.max_images = maximg;
  const quant = str("#adv-quant"); if (quant !== undefined) params.quantization = quant;
  if ($("#adv-trust").checked) params.trust_remote_code = true;
  const mm = str("#adv-mmproc"); if (mm !== undefined) params.mm_processor_kwargs = mm;
  const hf = str("#adv-hfover"); if (hf !== undefined) params.hf_overrides = hf;
  const extra = str("#adv-extra"); if (extra !== undefined) params.extra_args = extra.split(/\s+/);
  return {
    served_name: str("#adv-served"),
    port: num("#adv-port"),
    params: Object.keys(params).length ? params : undefined,
  };
}

function fmtAnalysis(a) {
  const rows = [];
  const typ = a.multimodal ? "vision + text" : (a.is_llm ? "text LLM" : "⚠ not a text-gen model?");
  rows.push(["Type", typ]);
  if (a.params) rows.push(["Parameters", (a.params / 1e9).toFixed(1) + " B" + (a.dtype ? " · " + a.dtype : "")]);
  if (a.size_gb != null) rows.push(["Weights", "~" + a.size_gb + " GB"]);
  if (a.need_gb != null) rows.push(["Needs (est.)", "~" + a.need_gb + " GB"]);
  rows.push(["Available", a.available_gb + " GB" + (a.total_gb ? " / " + a.total_gb + " GB total" : "")]);
  if (a.gated) rows.push(["Gated", a.hf_token_present ? "yes (HF_TOKEN set ✓)" : "yes — set HF_TOKEN in .env ⚠"]);
  return rows.map(([k, v]) => `<span>${escapeHtml(k)}</span><span>${escapeHtml(v)}</span>`).join("");
}

async function onAnalyze() {
  const hf = $("#hf-id").value.trim();
  const out = $("#analyze-result");
  if (!hf) return;
  const btn = $("#analyze-btn");
  btn.classList.add("is-busy");
  out.hidden = false; out.className = "analyze-result"; out.innerHTML = "Analyzing…";
  try {
    const a = await api("/api/analyze", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ hf_id: hf }),
    });
    if (!a.exists) { out.className = "analyze-result bad"; out.innerHTML =
      `<div class="a-title">Can't use this model</div>${escapeHtml(a.error || "not found")}`; return; }
    const cls = a.fits ? "ok" : "warn";
    const verdict = a.fits ? "Fits — ready to run" :
      (a.need_gb ? `Might not fit (~${a.need_gb} GB needed, ${a.available_gb} GB free)` : "Size unknown");
    out.className = "analyze-result " + cls;
    const needTok = a.gated && !a.hf_token_present;
    out.innerHTML =
      `<div class="a-title">${escapeHtml(a.hf_id)}<span class="type-pill">${a.fmt === "gguf" ? "gguf · llama.cpp" : (a.multimodal ? "vision" : "text")}</span></div>` +
      `<div class="a-grid">${fmtAnalysis(a)}</div>` +
      (needTok ? `<p class="hint warn-text">This is a gated model — set a Hugging Face token (and request access on its page) before downloading.</p>` : "") +
      `<div class="a-actions">` +
        (needTok ? `<button id="analyze-settoken" class="btn btn-small">Set Hugging Face token</button>` : "") +
        `<button id="analyze-run" class="btn btn-small btn-primary">${a.fits ? "Download & Run" : "Run anyway"}</button>` +
      `</div>`;
    $("#analyze-run").addEventListener("click", () => doAdd(hf, !a.fits));
    if (needTok) $("#analyze-settoken").addEventListener("click", openTokenPanel);
  } catch (err) {
    out.className = "analyze-result bad"; out.textContent = err.message;
  } finally {
    btn.classList.remove("is-busy");
  }
}

let addInFlight = false;
async function doAdd(hf, force) {
  if (addInFlight) return;        // guard against double-submit (one launch per click)
  addInFlight = true;
  const errEl = $("#add-error"); errEl.hidden = true;
  const adv = readAdvanced();
  const btn = $("#run-btn"); btn.classList.add("is-busy");
  try {
    await api("/api/models", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ hf_id: hf, run: true, force: !!force, ...adv }),
    });
    $("#hf-id").value = ""; $("#analyze-result").hidden = true;
    toast("Launching — watch the logs for download progress.", "ok");
    await loadModels();
  } catch (err) {
    errEl.hidden = false;
    if (err.status === 409) {  // doesn't fit — offer an explicit override
      errEl.innerHTML = escapeHtml(err.message) +
        ` <button id="force-run" class="btn btn-small">Run anyway</button>`;
      const f = $("#force-run");
      if (f) f.addEventListener("click", () => doAdd(hf, true));
    } else {
      errEl.textContent = err.message;
    }
  } finally {
    addInFlight = false;
    btn.classList.remove("is-busy");
  }
}

async function onAdd(e) {
  e.preventDefault();
  const hf = $("#hf-id").value.trim();
  if (hf) doAdd(hf, false);
}

// --------------------------------------------------------------------- model cards
const STATUS_LABEL = {
  stopped: "Stopped", queued: "Queued", pulling: "Pulling image",
  starting: "Starting", downloading: "Downloading", loading: "Loading",
  ready: "Ready", error: "Error",
};

// Show either the network/public base URL or the localhost one, per st.showLocal.
// Copy reads .base-url's text, so it always copies whichever is currently shown.
function applyUrlView(st) {
  const be = $(".base-url", st.node);
  const useLocal = st.showLocal && be.dataset.local;
  be.textContent = useLocal ? be.dataset.local : be.dataset.network;
  const tgl = $(".act-url-toggle", st.node);
  if (tgl) tgl.textContent = useLocal ? "Use network URL" : "Use localhost";
}

function renderCard(m) {
  let st = cards.get(m.id);
  if (!st) {
    const node = $("#model-card-tpl").content.firstElementChild.cloneNode(true);
    st = { node, es: null, file: null, expanded: false, showLocal: false };
    cards.set(m.id, st);
    wireCard(st, m);
    $("#models-list").appendChild(node);
  }
  const n = st.node;
  $(".model-id", n).textContent = m.served_name;
  const srcPill = $(".source-pill", n);
  srcPill.hidden = m.source !== "preset";
  srcPill.textContent = "preset";
  $(".model-sub", n).textContent = `${m.hf_id} · :${m.port}`
    + (m.fmt === "gguf" ? " · GGUF (llama.cpp)" : "");

  const pill = $(".status-pill", n);
  pill.className = "status-pill status-" + m.status;
  $(".status-text", n).textContent = STATUS_LABEL[m.status] || m.status;

  const active = m.status !== "stopped" && m.status !== "error";
  $(".act-run", n).hidden = active;
  $(".act-run", n).textContent = m.status === "error" ? "Retry" : "Run";
  $(".act-stop", n).hidden = !active;

  // base_url + test panel only meaningful when ready. Stash both the network/public
  // URL and the localhost one so the toggle can swap between them (see applyUrlView).
  const be = $(".base-url", n);
  be.dataset.network = m.base_url;
  be.dataset.local = (m.local_url || "").replace("127.0.0.1", "localhost");
  applyUrlView(st);
  const alt = $(".alt-base-url", n);
  if (alt) alt.textContent = m.alt_base_url || "";
  $(".register-block", n).hidden = !m.ready;
  // "+ Attach image" only for multimodal models, and only until image is shown
  if (!st.imgShown) $(".act-addimg", n).hidden = !m.multimodal;
  $(".test-block", n).hidden = !m.ready;

  const errEl = $(".model-error", n);
  if (m.status === "error" && (m.error || m.detail)) {
    const raw = m.error || m.detail;
    if (m.needs_hf_token) {
      gatedModels.add(m.id);
      errEl.innerHTML =
        `<strong>Gated model — a Hugging Face token is required to download it.</strong> ` +
        `Request access on the model's HF page, then paste a token to continue.` +
        `<div class="err-actions"><button class="btn btn-small btn-primary act-settoken" type="button">Set Hugging Face token</button></div>` +
        `<details class="err-detail"><summary>Show full error</summary><pre>${escapeHtml(raw)}</pre></details>`;
      $(".act-settoken", n).addEventListener("click", openTokenPanel);
      if (st.lastStatus !== "error") { setExpanded(st, true); openTokenPanel(); }  // surface the input once
    } else {
      gatedModels.delete(m.id);
      errEl.textContent = raw;
      if (st.lastStatus !== "error") setExpanded(st, true);  // auto-open on a new error
    }
    errEl.hidden = false;
  } else {
    gatedModels.delete(m.id);
    errEl.hidden = true;
  }
  $(".model-body", n).hidden = !st.expanded;
  n.classList.toggle("expanded", !!st.expanded);
  st.lastStatus = m.status;
  st.model = m;
}

// Expand/collapse a card's body (logs + register + test). Collapsing also stops
// the log stream so we don't keep an SSE open for a hidden card.
function setExpanded(st, val) {
  st.expanded = val;
  $(".model-body", st.node).hidden = !val;
  st.node.classList.toggle("expanded", val);
  if (!val) { $(".logs-block", st.node).hidden = true; closeLogs(st); }
}

function wireCard(st, m) {
  const n = st.node;
  const id = m.id;
  const body = $(".model-body", n);

  $(".act-run", n).addEventListener("click", () => act(id, "start", $(".act-run", n)));
  $(".act-stop", n).addEventListener("click", () => act(id, "stop", $(".act-stop", n)));
  $(".act-remove", n).addEventListener("click", async () => {
    if (!confirm(`Remove ${id}? (stops it; keeps downloaded weights)`)) return;
    closeLogs(st);
    try { await api(`/api/models/${encodeURIComponent(id)}`, { method: "DELETE" }); }
    catch (e) { toast(e.message, "error"); return; }
    cards.delete(id); n.remove(); toast("Removed " + id, "ok");
  });

  // Click the card header (anywhere but a button) toggles the whole body.
  $(".model-card-top", n).addEventListener("click", (e) => {
    if (e.target.closest("button")) return;
    setExpanded(st, !st.expanded);
  });

  $(".act-logs", n).addEventListener("click", () => {
    const lb = $(".logs-block", n);
    if (lb.hidden) { setExpanded(st, true); lb.hidden = false; openLogs(st, id); }
    else { lb.hidden = true; closeLogs(st); }
  });

  $(".act-url-toggle", n).addEventListener("click", () => {
    st.showLocal = !st.showLocal;
    applyUrlView(st);
  });

  $(".act-copy", n).addEventListener("click", () => {
    const url = $(".base-url", n).textContent;
    copyText(url).then(
      () => toast("Copied base URL", "ok"),
      () => toast("Copy failed — select the URL and copy manually", "error"));
  });

  // quick test — text by default; image optional (multimodal only)
  $(".act-addimg", n).addEventListener("click", async () => {
    st.imgShown = true;
    $(".act-addimg", n).hidden = true;
    $(".test-img-wrap", n).hidden = false;
    $(".test-img", n).src = "/api/test-image";
    try {  // load the default test image so a click-run actually sends it
      const blob = await (await fetch("/api/test-image")).blob();
      st.file = new File([blob], "test.png", { type: "image/png" });
    } catch (e) {}
  });
  $(".act-pick", n).addEventListener("click", () => $(".test-file", n).click());
  $(".test-file", n).addEventListener("change", (ev) => {
    const f = ev.target.files[0]; if (!f) return;
    st.file = f; $(".test-img", n).src = URL.createObjectURL(f);
  });
  $(".act-test", n).addEventListener("click", () => runTest(st, id));
}

async function act(id, verb, btn) {
  btn.classList.add("is-busy");
  try { await api(`/api/models/${encodeURIComponent(id)}/${verb}`, { method: "POST" }); await loadModels(); }
  catch (e) { toast(e.message, "error"); }
  finally { btn.classList.remove("is-busy"); }
}

async function runTest(st, id) {
  const n = st.node;
  const btn = $(".act-test", n);
  const ansEl = $(".test-answer", n);
  btn.classList.add("is-busy");
  ansEl.hidden = false; ansEl.textContent = "Running…";
  try {
    const fd = new FormData();
    fd.append("prompt", $(".test-prompt", n).value);
    if (st.file) fd.append("image", st.file);
    const r = await api(`/api/models/${encodeURIComponent(id)}/test`, { method: "POST", body: fd });
    ansEl.innerHTML = escapeHtml(r.answer || "(empty response)") +
      `<div class="meta">${r.latency_ms} ms${r.usage ? " · " + (r.usage.total_tokens || "?") + " tokens" : ""}</div>`;
  } catch (e) {
    ansEl.textContent = "Test failed: " + e.message;
  } finally {
    btn.classList.remove("is-busy");
  }
}

// --------------------------------------------------------------------- logs SSE
function openLogs(st, id) {
  closeLogs(st);
  const view = $(".logs-view", st.node);
  view.textContent = "";
  const es = new EventSource(`/api/models/${encodeURIComponent(id)}/logs`);
  st.es = es;
  es.onmessage = (ev) => {
    let d; try { d = JSON.parse(ev.data); } catch (e) { return; }
    if (d.line !== undefined) {
      const atBottom = view.scrollTop + view.clientHeight >= view.scrollHeight - 30;
      view.textContent += d.line + "\n";
      if (view.textContent.length > 200000) view.textContent = view.textContent.slice(-150000);
      if (atBottom) view.scrollTop = view.scrollHeight;
    }
    if (d.status !== undefined && st.model) {
      // reflect live status on the pill without a full reload
      const pill = $(".status-pill", st.node);
      pill.className = "status-pill status-" + d.status;
      $(".status-text", st.node).textContent = STATUS_LABEL[d.status] || d.status;
      if (d.ready) { st.tokenPrompted = false; loadModels(); }  // reveal register/test blocks once serving
      // Gated-repo failure detected mid-download → surface the token input at once.
      else if (d.needs_hf_token && !st.tokenPrompted) {
        st.tokenPrompted = true;
        loadModels();      // re-render this card with the "Set token" button + error
        openTokenPanel();  // and pop the paste box
      }
    }
    if (d.eof) closeLogs(st);
  };
  es.onerror = () => { /* keep the partial logs; browser auto-retries */ };
}
function closeLogs(st) {
  if (st.es) { st.es.close(); st.es = null; }
}

// --------------------------------------------------------------------- cache library
async function loadCache() {
  let data;
  try { data = await api("/api/cache"); }
  catch (e) { return; }
  const models = data.models || [];
  const list = $("#cache-list");
  list.innerHTML = "";
  $("#cache-count").textContent = models.length
    ? `· ${models.length} on disk (${data.total_gb} GB)` : "";
  $("#cache-empty").hidden = models.length > 0;
  for (const m of models) {
    const li = document.createElement("li");
    li.className = "cache-row";
    li.innerHTML =
      `<span class="c-id">${escapeHtml(m.hf_id)}` +
      (m.multimodal ? `<span class="type-pill">vision</span>` : "") + `</span>` +
      `<span class="c-size">${m.size_gb} GB</span>` +
      `<span class="c-actions">` +
      `<button class="btn btn-small btn-primary c-run">Run</button>` +
      `<button class="btn btn-small btn-danger c-free">Free</button></span>`;
    $(".c-run", li).addEventListener("click", () => {
      toast("Launching " + m.hf_id + " from cache…", "ok");
      doAdd(m.hf_id, false);
    });
    $(".c-free", li).addEventListener("click", async () => {
      if (!confirm(`Delete ${m.hf_id} weights from disk (${m.size_gb} GB)? Re-running it later will re-download.`)) return;
      try { await api("/api/cache/delete", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ hf_id: m.hf_id }) });
        toast("Freed " + m.hf_id, "ok"); loadCache();
      } catch (e) { toast(e.message, "error"); }
    });
    list.appendChild(li);
  }
}

// --------------------------------------------------------------------- list + poll
async function loadModels() {
  let data;
  try { data = await api("/api/models"); }
  catch (e) { toast("Could not load models: " + e.message, "error"); return; }
  const models = data.models || [];
  const seen = new Set();
  for (const m of models) { renderCard(m); seen.add(m.id); }
  // drop removed
  for (const [id, st] of cards) {
    if (!seen.has(id)) { closeLogs(st); st.node.remove(); cards.delete(id); }
  }
  $("#models-empty").hidden = models.length > 0;
}

// --------------------------------------------------------------------- boot
function init() {
  initTheme();
  $("#add-form").addEventListener("submit", onAdd);
  $("#analyze-btn").addEventListener("click", onAnalyze);
  $("#refresh-btn").addEventListener("click", () => { loadBanner(); loadModels(); });
  $("#cache-refresh").addEventListener("click", loadCache);
  // hugging face token controls
  $("#hf-token-toggle").addEventListener("click", () => {
    const p = $("#hf-token-panel");
    p.hidden = !p.hidden;
    if (!p.hidden) $("#hf-token-input").focus();
  });
  $("#hf-token-save").addEventListener("click", saveHFToken);
  $("#hf-token-input").addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); saveHFToken(); } });
  $("#hf-token-clear").addEventListener("click", clearHFToken);
  loadBanner();
  loadHFToken();
  loadModels();
  loadCache();
  // periodic refresh keeps status pills fresh for collapsed cards
  refreshTimer = setInterval(loadModels, 5000);
  setInterval(loadBanner, 15000);
}
document.addEventListener("DOMContentLoaded", init);
