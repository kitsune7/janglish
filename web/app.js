// Contributor recording flow.
//
// Parses ?t=<token> from the URL, fetches the assignment from the Worker,
// steps the user through their sentences one at a time, and uploads each
// recorded clip as a 16 kHz mono WAV. Progress (which IDs are done) is
// persisted in localStorage, keyed by the token — closing the tab and
// returning to the same link resumes where the contributor left off.
//
// The audio path mirrors web/wav-encoder.js + web/downsampler-worklet.js
// from Phase 3a.

import { encodeWav } from "./wav-encoder.js";

const OUT_RATE = 16000;
const KANJI_RE = /[\u3400-\u4DBF\u4E00-\u9FFF々]/u;

// API base: ?api= override → <meta name="janglish-api"> → same origin.
const params = new URLSearchParams(location.search);
const API = (
  params.get("api") ||
  document.querySelector('meta[name="janglish-api"]')?.content ||
  location.origin
).replace(/\/$/, "");

const token = params.get("t") ?? "";

const $ = (id) => document.getElementById(id);
const elements = {
  app: $("app"),
  fatal: $("fatal"),
  greeting: $("greeting"),
  progress: $("progress"),
  sentence: $("sentence"),
  furiganaStatus: $("furigana-status"),
  record: $("record"),
  stop: $("stop"),
  play: $("play"),
  upload: $("upload"),
  prev: $("prev"),
  next: $("next"),
  player: $("player"),
  status: $("status"),
  completedBanner: $("completed-banner"),
};

// --- state ---
let assignment = null; // {name, sentences: [{id, text}]}
let storageKey = null; // "janglish/progress/<token>"
let completed = {}; // {id: true} — uploaded ids
let index = 0;
let currentBlob = null; // WAV blob for the sentence currently on screen
let isRecording = false;
let isUploading = false;
let recorder = null;
const furiganaCache = new Map(); // sentence text -> escaped HTML with ruby tags
const furiganaState = new Map(); // sentence text -> "ready" | "fallback"
const furiganaPending = new Set(); // sentence text currently queued in the worker
const furiganaRequests = new Map(); // request id -> sentence text
let furiganaWorker = null;
let furiganaRequestId = 0;

// --- bootstrap ---
(async function main() {
  if (!token) return fatal("Missing token. Open the share link you were sent.");
  try {
    const response = await fetch(`${API}/assignment?t=${encodeURIComponent(token)}`);
    if (response.status === 401) return fatal("This link is invalid or has expired. Ask for a new one.");
    if (response.status === 404) return fatal("No assignment found for this link. Ask for a new one.");
    if (!response.ok) return fatal(`Server error ${response.status} while loading your assignment.`);
    assignment = await response.json();
  } catch (err) {
    return fatal(`Can't reach the server: ${err.message}`);
  }
  if (!assignment?.sentences?.length) return fatal("Your assignment is empty. Ask for a new one.");

  storageKey = `janglish/progress/${token}`;
  loadProgress();
  elements.greeting.textContent = `Hi ${assignment.name} — you have ${assignment.sentences.length} sentence(s) to record.`;
  elements.app.hidden = false;
  index = firstUnfinishedIndex();
  render();
  wireControls();
})();

// --- progress persistence ---
function loadProgress() {
  try {
    const raw = localStorage.getItem(storageKey);
    if (!raw) return;
    const saved = JSON.parse(raw);
    if (saved && typeof saved === "object") completed = saved.completed ?? {};
  } catch {
    completed = {};
  }
}
function saveProgress() {
  try {
    localStorage.setItem(storageKey, JSON.stringify({ completed }));
  } catch {
    /* ignore — private mode, quota, etc. */
  }
}

function firstUnfinishedIndex() {
  const i = assignment.sentences.findIndex((sentence) => !completed[sentence.id]);
  return i === -1 ? 0 : i;
}

// --- rendering ---
function render() {
  const total = assignment.sentences.length;
  const currentSentence = assignment.sentences[index];
  const doneCount = assignment.sentences.filter((sentence) => completed[sentence.id]).length;
  const marker = completed[currentSentence.id] ? "✓" : "•";

  elements.progress.textContent = `Sentence ${index + 1} of ${total}   (${doneCount}/${total} uploaded)`;
  elements.sentence.innerHTML = `<span class="done-marker ${completed[currentSentence.id] ? "ok" : ""}">${marker}</span> ${renderSentenceText(currentSentence.text)}`;
  renderFuriganaStatus(currentSentence.text);

  elements.prev.disabled = index === 0;
  elements.next.disabled = index === total - 1;

  const canInteract = !isRecording && !isUploading;
  elements.record.disabled = !canInteract;
  elements.stop.disabled = !isRecording;
  elements.play.disabled = !currentBlob || isRecording || isUploading;
  elements.upload.disabled = !currentBlob || isRecording || isUploading;

  elements.completedBanner.hidden = doneCount < total;
}

// --- controls ---
function wireControls() {
  elements.record.addEventListener("click", startRecording);
  elements.stop.addEventListener("click", stopRecording);
  elements.play.addEventListener("click", () => elements.player.play());
  elements.upload.addEventListener("click", uploadCurrent);
  elements.prev.addEventListener("click", () => move(-1));
  elements.next.addEventListener("click", () => move(+1));

  window.addEventListener("keydown", (event) => {
    if (event.target !== document.body) return;
    if (event.key === "r" || event.key === "R") { isRecording ? stopRecording() : startRecording(); event.preventDefault(); }
    else if (event.key === " ") { if (!elements.play.disabled) elements.player.play(); event.preventDefault(); }
    else if (event.key === "ArrowLeft") move(-1);
    else if (event.key === "ArrowRight") move(+1);
    else if (event.key === "Enter") { if (!elements.upload.disabled) uploadCurrent(); }
  });
}

function move(delta) {
  const target = index + delta;
  if (target < 0 || target >= assignment.sentences.length) return;
  discardCurrentBlob();
  index = target;
  render();
}

function discardCurrentBlob() {
  if (elements.player.src) {
    URL.revokeObjectURL(elements.player.src);
    elements.player.removeAttribute("src");
    elements.player.load();
  }
  currentBlob = null;
}

// --- recording ---
async function startRecording() {
  if (isRecording || isUploading) return;
  discardCurrentBlob();
  setStatus("requesting mic…");
  try {
    recorder = await createRecorder();
  } catch (err) {
    setStatus(`mic error: ${err.message}`, "error");
    return;
  }
  isRecording = true;
  render();
  setStatus(`recording (ctx ${recorder.ctxRate} Hz → ${OUT_RATE} Hz)…`);
}

async function stopRecording() {
  if (!isRecording || !recorder) return;
  isRecording = false;
  setStatus("finalizing…");
  const samples = await recorder.stop();
  recorder = null;
  if (!samples.length) {
    setStatus("no audio captured — try again.", "error");
    render();
    return;
  }
  currentBlob = encodeWav(samples, OUT_RATE);
  elements.player.src = URL.createObjectURL(currentBlob);
  const seconds = (samples.length / OUT_RATE).toFixed(1);
  setStatus(`captured ${seconds}s (${currentBlob.size} bytes). Play to review, Upload to submit.`);
  render();
}

async function createRecorder() {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      sampleRate: OUT_RATE,
      echoCancellation: false,
      noiseSuppression: false,
      autoGainControl: false,
    },
  });
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  await ctx.audioWorklet.addModule("./downsampler-worklet.js");

  const source = ctx.createMediaStreamSource(stream);
  const node = new AudioWorkletNode(ctx, "downsampler", {
    numberOfInputs: 1,
    numberOfOutputs: 0,
    processorOptions: { outRate: OUT_RATE },
  });

  const chunks = [];
  node.port.onmessage = (evt) => chunks.push(evt.data);
  source.connect(node);

  return {
    ctxRate: ctx.sampleRate,
    async stop() {
      try {
        source.disconnect();
        node.disconnect();
        stream.getTracks().forEach((t) => t.stop());
        await ctx.close();
      } catch (err) {
        console.warn("teardown:", err);
      }
      return concatFloat32(chunks);
    },
  };
}

function concatFloat32(parts) {
  let total = 0;
  for (const p of parts) total += p.length;
  const out = new Float32Array(total);
  let off = 0;
  for (const p of parts) { out.set(p, off); off += p.length; }
  return out;
}

// --- upload ---
async function uploadCurrent() {
  if (!currentBlob || isUploading) return;
  const currentSentence = assignment.sentences[index];
  isUploading = true;
  render();
  setStatus(`uploading sentence ${index + 1}…`);
  try {
    const r = await fetch(
      `${API}/upload?t=${encodeURIComponent(token)}&id=${encodeURIComponent(currentSentence.id)}`,
      { method: "PUT", headers: { "content-type": "audio/wav" }, body: currentBlob },
    );
    if (r.status === 401) throw new Error("token rejected — ask for a new link");
    if (!r.ok) throw new Error(`server ${r.status}: ${await r.text()}`);
    completed[currentSentence.id] = true;
    saveProgress();
    setStatus(`✓ uploaded sentence ${index + 1}.`, "ok");
    discardCurrentBlob();
    // Auto-advance to the next unfinished sentence, if any.
    const nextIdx = nextUnfinishedAfter(index);
    if (nextIdx !== null) index = nextIdx;
  } catch (err) {
    setStatus(`upload failed: ${err.message}. Your recording is still here — try Upload again.`, "error");
  } finally {
    isUploading = false;
    render();
  }
}

function nextUnfinishedAfter(start) {
  const n = assignment.sentences.length;
  for (let step = 1; step <= n; step++) {
    const i = (start + step) % n;
    if (!completed[assignment.sentences[i].id]) return i;
  }
  return null;
}

// --- helpers ---
function setStatus(msg, kind) {
  elements.status.textContent = msg;
  elements.status.className = kind ?? "";
}

function fatal(msg) {
  elements.fatal.textContent = msg;
  elements.fatal.hidden = false;
  elements.app.hidden = true;
  elements.greeting.textContent = "";
}

function renderSentenceText(text) {
  if (!hasKanji(text)) return escapeHtml(text);
  const cached = furiganaCache.get(text);
  if (cached) return cached;

  queueFurigana(text);
  return escapeHtml(text);
}

function renderFuriganaStatus(text) {
  if (!hasKanji(text)) {
    elements.furiganaStatus.hidden = true;
    elements.furiganaStatus.textContent = "";
    elements.furiganaStatus.className = "";
    return;
  }

  elements.furiganaStatus.hidden = false;
  if (furiganaPending.has(text)) {
    elements.furiganaStatus.textContent = "Loading furigana...";
    elements.furiganaStatus.className = "loading";
    return;
  }

  if (furiganaState.get(text) === "fallback") {
    elements.furiganaStatus.textContent = "Furigana unavailable; showing plain text.";
    elements.furiganaStatus.className = "error";
    return;
  }

  if (furiganaState.get(text) === "ready") {
    elements.furiganaStatus.textContent = "Furigana ready.";
    elements.furiganaStatus.className = "ok";
    return;
  }

  elements.furiganaStatus.textContent = "Preparing furigana...";
  elements.furiganaStatus.className = "loading";
}

function queueFurigana(text) {
  if (furiganaPending.has(text)) return;

  const worker = getFuriganaWorker();
  if (!worker) {
    furiganaCache.set(text, escapeHtml(text));
    furiganaState.set(text, "fallback");
    return;
  }

  const id = ++furiganaRequestId;
  furiganaPending.add(text);
  furiganaRequests.set(id, text);
  worker.postMessage({ id, text });
}

function getFuriganaWorker() {
  if (!("Worker" in window)) return null;
  if (furiganaWorker) return furiganaWorker;

  try {
    furiganaWorker = new Worker("./furigana-worker.js");
  } catch (err) {
    console.warn("furigana worker:", err);
    return null;
  }
  furiganaWorker.addEventListener("message", ({ data }) => {
    const text = furiganaRequests.get(data?.id);
    if (!text) return;

    furiganaRequests.delete(data.id);
    furiganaPending.delete(text);
    if (data.error) {
      console.warn("furigana:", data.error);
      furiganaCache.set(text, escapeHtml(text));
      furiganaState.set(text, "fallback");
    } else {
      furiganaCache.set(text, renderFuriganaMap(data.result, text));
      furiganaState.set(text, "ready");
    }

    if (assignment?.sentences?.[index]?.text === text) render();
  });
  furiganaWorker.addEventListener("error", (event) => {
    console.warn("furigana worker:", event.message);
    furiganaWorker?.terminate();
    furiganaWorker = null;
    for (const text of furiganaPending) {
      furiganaCache.set(text, escapeHtml(text));
      furiganaState.set(text, "fallback");
    }
    furiganaPending.clear();
    furiganaRequests.clear();
    render();
  });

  return furiganaWorker;
}

function renderFuriganaMap(result, originalText) {
  const text = typeof result?.text === "string" ? result.text : originalText;
  const ruby = Array.isArray(result?.ruby) ? result.ruby : [];
  let html = "";
  let pos = 0;

  for (const span of ruby) {
    const start = Number.isInteger(span.s) ? span.s : -1;
    const end = Number.isInteger(span.e) ? span.e : -1;
    if (start < pos || end <= start || end > text.length || typeof span.rt !== "string") continue;

    html += escapeHtml(text.slice(pos, start));
    const base = text.slice(start, end);
    html += hasKanji(base) ? renderRuby(base, span.rt) : escapeHtml(base);
    pos = end;
  }

  return html + escapeHtml(text.slice(pos));
}

function renderRuby(text, reading) {
  return `<ruby>${escapeHtml(text)}<rp>(</rp><rt>${escapeHtml(reading)}</rt><rp>)</rp></ruby>`;
}

function hasKanji(text) {
  return KANJI_RE.test(text);
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
