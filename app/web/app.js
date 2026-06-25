/* Meeting Assistant frontend runtime (task T014; US1–US4 + edge cases).
 * Calls window.pywebview.api.* (the Api bridge); receives pushes via
 * window.onBackendEvent(evt). SpeakerView color is a deterministic function of
 * speaker-label arrival order, mirroring app/dto.py SPEAKER_COLORS. */

"use strict";

const SPEAKER_COLORS = ["#1A7F64", "#2D7FF9", "#B4515C", "#9B59B6"];
const SCREENS = ["setup", "ready", "session", "review"];

const state = {
  status: "setting_up",
  speakerOrder: [],     // labels in arrival order → color index
  rendered: new Map(),  // segment_id → <li> (dedupe + stable position)
  langHint: null,
};

const $ = (id) => document.getElementById(id);
const api = () => window.pywebview?.api;

/* ---------- SpeakerView: deterministic color by arrival order ---------- */
function colorFor(label) {
  let idx = state.speakerOrder.indexOf(label);
  if (idx === -1) { state.speakerOrder.push(label); idx = state.speakerOrder.length - 1; }
  return SPEAKER_COLORS[idx % SPEAKER_COLORS.length];
}

/* ---------- screen routing ---------- */
function showScreen(name) { SCREENS.forEach((s) => $(s).classList.toggle("hidden", s !== name)); }

function setStatus(status) {
  state.status = status;
  const badge = $("statusBadge");
  badge.className = "badge";
  if (status === "recording" || status === "processing") { badge.classList.add("badge--rec"); badge.textContent = status === "recording" ? "recording" : "processing…"; }
  else if (status === "starting") { badge.classList.add("badge--rec"); badge.textContent = "starting…"; }
  else if (status === "ready") { badge.classList.add("badge--ok"); badge.textContent = "ready"; }
  else if (status === "stopped") { badge.textContent = "stopped"; }
  else if (status === "error") { badge.textContent = "error"; }
  else if (status === "stopping") { badge.textContent = "stopping…"; }
  else { badge.classList.add("badge--muted"); badge.textContent = status; }
}

/* ---------- transcript rendering ---------- */
function fmtTime(t) {
  const s = Math.max(0, Math.floor(t));
  const m = Math.floor(s / 60), r = s % 60;
  const pad = (n) => String(n).padStart(2, "0");
  return `${pad(m)}:${pad(r)}`;
}

function segmentLine(seg, into) {
  let li = state.rendered.get(seg.segment_id);
  if (!li) {
    li = document.createElement("li");
    state.rendered.set(seg.segment_id, li);
    into.appendChild(li); // appended in arrival order; backend emits chronological
  }
  const color = colorFor(seg.speaker_label);
  const lang = seg.language ? seg.language : "auto";
  const low = seg.confidence_band === "low" || seg.confidence_band === "unknown";
  // Provisional (still-transcribing) lines are flagged so they can be styled as live.
  li.classList.toggle("provisional", seg.is_final === false);
  li.innerHTML = "";
  const rail = document.createElement("span"); rail.className = "rail"; rail.style.background = color; li.appendChild(rail);
  const meta = document.createElement("span"); meta.className = "meta"; meta.textContent = fmtTime(seg.start); li.appendChild(meta);
  const body = document.createElement("span"); body.className = "text" + (low ? " low" : "");
  const who = document.createElement("span"); who.className = "who"; who.style.color = color; who.textContent = seg.speaker_label;
  const lg = document.createElement("span"); lg.className = "lang"; lg.textContent = ` (${lang})`;
  body.append(who, lg, document.createTextNode(": " + seg.text));
  li.appendChild(body);
  return li;
}

function renderSegmentsInto(segments, into) {
  // Sort defensively by start (FR-019); backend already emits in order.
  segments.slice().sort((a, b) => a.start - b.start || a.end - b.end).forEach((s) => segmentLine(s, into));
}

/* ---------- event dispatcher (Python → JS) ---------- */
window.onBackendEvent = function (evt) {
  try {
    switch (evt.type) {
      case "status":
        setStatus(evt.status);
        if (evt.status === "stopped" || evt.status === "error") showReview();
        break;
      case "segment": {
        $("emptyHint")?.classList.add("hidden");
        segmentLine(evt.segment, $("transcript"));
        // Edge (T050): >4 simultaneous speakers exceeds the diarizer capacity —
        // flag once; labels remain best-effort (color wraps past the 4-color palette).
        if (state.speakerOrder.length > 4 && !state._overCapacityWarned) {
          state._overCapacityWarned = true;
          showError({ code: "diar.capacity", message: "more than 4 speakers detected", hint: "Beyond 4 simultaneous speakers, labels are best-effort (low confidence)." });
        }
        break;
      }
      case "progress": {
        const f = Math.min(1, Math.max(0, evt.fraction || 0));
        $("fileProgress").classList.remove("hidden");
        $("fileFill").style.width = (f * 100).toFixed(1) + "%";
        break;
      }
      case "prepare_progress": {
        $("prepareProgress").classList.remove("hidden");
        $("prepareFill").style.width = (evt.fraction * 100).toFixed(1) + "%";
        $("prepareLabel").textContent = `${evt.asset} — ${Math.round(evt.fraction * 100)}%`;
        break;
      }
      case "prepare_done":
        $("prepareProgress").classList.add("hidden");
        if (evt.readiness?.ready) showReady(); else renderReadiness(evt.readiness);
        break;
      case "error":
        showError(evt.error);
        if (evt.error?.code === "not_ready") showSetup();
        break;
    }
  } catch (e) { console.error("event handler error (isolated from backend)", e); }
};

/* ---------- setup / readiness ---------- */
async function refreshReadiness() {
  const r = await api().get_readiness();
  if (r.ready) showReady(); else renderReadiness(r);
}
function renderReadiness(r) {
  showSetup();
  const ul = $("readinessList"); ul.innerHTML = "";
  if (!r || !r.missing || r.missing.length === 0) {
    const li = document.createElement("li"); li.textContent = "Ready."; ul.appendChild(li);
    return;
  }
  r.missing.forEach((m) => {
    const li = document.createElement("li");
    li.innerHTML = `<span class="req-state req--missing">⌀ missing</span><span>${escapeHtml(m)}</span>`;
    ul.appendChild(li);
  });
}
function showSetup() { setStatus("setting_up"); showScreen("setup"); }

/* ---------- ready / live ---------- */
function showReady() {
  setStatus("ready"); state.rendered.clear(); $("transcript").innerHTML = "";
  showScreen("ready");
}
async function startLive() {
  const sources = [];
  if ($("srcMic").checked) sources.push("microphone");
  if ($("srcSys").checked) sources.push("system");
  if (sources.length === 0) { showError({ code: "sources.invalid", message: "pick at least one source", hint: "Select Microphone or System audio." }); return; }
  const hint = ($("langHint").value || "").trim(); state.langHint = hint || null;
  const res = await api().start_live(sources, state.langHint);
  if (res.error) { showError(res.error); if (res.error.code === "not_ready") showSetup(); return; }
  beginSession("live");
}
async function importFile() {
  const picked = await api().pick_audio_file();
  if (!picked || !picked.path) return; // cancelled
  const hint = ($("langHint").value || "").trim();
  const res = await api().transcribe_file(picked.path, hint || null);
  if (res.error) { showError(res.error); if (res.error.code === "not_ready") showSetup(); return; }
  beginSession("file");
  $("fileProgress").classList.remove("hidden"); $("fileFill").style.width = "0%";
}
function beginSession(mode) {
  $("sessionMode").textContent = mode === "file" ? "file import" : "live";
  if (mode === "live") $("fileProgress").classList.add("hidden");
  $("emptyHint").classList.toggle("hidden", false);
  // Live: model loading runs on a worker — show "starting" until the backend emits "recording".
  setStatus(mode === "file" ? "processing" : "starting");
  showScreen("session");
}

/* ---------- stop / review ---------- */
async function stopSession() { setStatus("stopping"); await api().stop_session(); }
async function showReview() {
  // Re-render the full snapshot into the review pane for a stable read.
  const snap = await api().get_transcript();
  state.rendered.clear();
  const into = $("reviewTranscript"); into.innerHTML = "";
  renderSegmentsInto(snap.segments, into);
  $("reviewTitle").textContent = snap.segments.length ? `Transcript · ${snap.segments.length} lines` : "Transcript";
  setStatus(state.status === "error" ? "error" : "stopped");
  showScreen("review");
}

/* ---------- export ---------- */
async function exportAs(format) {
  const picked = await api().pick_export_path(format);
  if (!picked || !picked.path) return;
  const res = await api().export_transcript(picked.path, format);
  if (res.error) showError(res.error);
}

/* ---------- error surface ---------- */
function showError(err) {
  if (!err) return;
  $("errorToast").classList.remove("hidden");
  $("errTitle").textContent = err.code || "Error";
  $("errMsg").textContent = err.message || "";
  $("errHint").textContent = err.hint || "";
}

/* ---------- helpers ---------- */
function escapeHtml(s) { return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }

/* ---------- bootstrap ---------- */
function bind() {
  $("refreshReadiness").onclick = refreshReadiness;
  $("prepareBtn").onclick = async () => { $("prepareProgress").classList.remove("hidden"); $("prepareFill").style.width = "0%"; await api().prepare(); };
  $("startBtn").onclick = startLive;
  $("importBtn").onclick = importFile;
  $("stopBtn").onclick = stopSession;
  $("exportMd").onclick = () => exportAs("markdown");
  $("exportJson").onclick = () => exportAs("json");
  $("newSession").onclick = showReady;
  $("errClose").onclick = () => $("errorToast").classList.add("hidden");
}

let _booted = false;

// pywebview injects `window.pywebview.api` as an EMPTY object first (truthy!),
// then populates the real methods just before firing `pywebviewready`. Gating on
// `window.pywebview?.api` alone fires bootstrap too early → `api().get_readiness`
// is undefined → throws → a spurious "backend unavailable" toast. Require a real
// method so we only call the bridge once it is genuinely wired.
function bridgeReady() {
  return typeof window.pywebview?.api?.get_readiness === "function";
}

async function bootstrap() {
  if (_booted) return; // pywebviewready + the fallback poll may both fire — boot once.
  _booted = true;
  bind();
  try { await refreshReadiness(); }
  catch (e) {
    // Backend unavailable (rare): show a recovery path rather than a blank screen.
    showSetup();
    showError({ code: "backend.unavailable", message: "could not reach the local backend", hint: "Relaunch the app; if it persists, re-run setup." });
  }
}

// Primary: the bridge is fully wired when this event fires.
window.addEventListener("pywebviewready", bootstrap);
// Fallback poll for environments that don't fire the event promptly — but only
// once the API methods actually exist (not just the empty placeholder object).
(function wait() {
  if (_booted) return;
  if (bridgeReady()) bootstrap();
  else setTimeout(wait, 60);
})();
