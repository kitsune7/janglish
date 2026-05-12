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

// API base: ?api= override → <meta name="janglish-api"> → same origin.
const params = new URLSearchParams(location.search);
const API = (
  params.get("api") ||
  document.querySelector('meta[name="janglish-api"]')?.content ||
  location.origin
).replace(/\/$/, "");

const token = params.get("t") ?? "";

const $ = (id) => document.getElementById(id);
const els = {
  app: $("app"),
  fatal: $("fatal"),
  greeting: $("greeting"),
  progress: $("progress"),
  sentence: $("sentence"),
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
let idx = 0;
let currentBlob = null; // WAV blob for the sentence currently on screen
let isRecording = false;
let isUploading = false;
let recorder = null;

// --- bootstrap ---
(async function main() {
  if (!token) return fatal("Missing token. Open the share link you were sent.");
  try {
    const r = await fetch(`${API}/assignment?t=${encodeURIComponent(token)}`);
    if (r.status === 401) return fatal("This link is invalid or has expired. Ask for a new one.");
    if (r.status === 404) return fatal("No assignment found for this link. Ask for a new one.");
    if (!r.ok) return fatal(`Server error ${r.status} while loading your assignment.`);
    assignment = await r.json();
  } catch (err) {
    return fatal(`Can't reach the server: ${err.message}`);
  }
  if (!assignment?.sentences?.length) return fatal("Your assignment is empty. Ask for a new one.");

  storageKey = `janglish/progress/${token}`;
  loadProgress();
  els.greeting.textContent = `Hi ${assignment.name} — you have ${assignment.sentences.length} sentence(s) to record.`;
  els.app.hidden = false;
  idx = firstUnfinishedIndex();
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
  const i = assignment.sentences.findIndex((s) => !completed[s.id]);
  return i === -1 ? 0 : i;
}

// --- rendering ---
function render() {
  const total = assignment.sentences.length;
  const s = assignment.sentences[idx];
  const doneCount = assignment.sentences.filter((x) => completed[x.id]).length;
  const marker = completed[s.id] ? "✓" : "•";

  els.progress.textContent = `Sentence ${idx + 1} of ${total}   (${doneCount}/${total} uploaded)`;
  els.sentence.innerHTML = `<span class="done-marker ${completed[s.id] ? "ok" : ""}">${marker}</span> ${escapeHtml(s.text)}`;

  els.prev.disabled = idx === 0;
  els.next.disabled = idx === total - 1;

  const canInteract = !isRecording && !isUploading;
  els.record.disabled = !canInteract;
  els.stop.disabled = !isRecording;
  els.play.disabled = !currentBlob || isRecording || isUploading;
  els.upload.disabled = !currentBlob || isRecording || isUploading;

  els.completedBanner.hidden = doneCount < total;
}

// --- controls ---
function wireControls() {
  els.record.addEventListener("click", startRecording);
  els.stop.addEventListener("click", stopRecording);
  els.play.addEventListener("click", () => els.player.play());
  els.upload.addEventListener("click", uploadCurrent);
  els.prev.addEventListener("click", () => move(-1));
  els.next.addEventListener("click", () => move(+1));

  window.addEventListener("keydown", (e) => {
    if (e.target !== document.body) return;
    if (e.key === "r" || e.key === "R") { isRecording ? stopRecording() : startRecording(); e.preventDefault(); }
    else if (e.key === " ") { if (!els.play.disabled) els.player.play(); e.preventDefault(); }
    else if (e.key === "ArrowLeft") move(-1);
    else if (e.key === "ArrowRight") move(+1);
    else if (e.key === "Enter") { if (!els.upload.disabled) uploadCurrent(); }
  });
}

function move(delta) {
  const target = idx + delta;
  if (target < 0 || target >= assignment.sentences.length) return;
  discardCurrentBlob();
  idx = target;
  render();
}

function discardCurrentBlob() {
  if (els.player.src) {
    URL.revokeObjectURL(els.player.src);
    els.player.removeAttribute("src");
    els.player.load();
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
  els.player.src = URL.createObjectURL(currentBlob);
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
  const s = assignment.sentences[idx];
  isUploading = true;
  render();
  setStatus(`uploading sentence ${idx + 1}…`);
  try {
    const r = await fetch(
      `${API}/upload?t=${encodeURIComponent(token)}&id=${encodeURIComponent(s.id)}`,
      { method: "PUT", headers: { "content-type": "audio/wav" }, body: currentBlob },
    );
    if (r.status === 401) throw new Error("token rejected — ask for a new link");
    if (!r.ok) throw new Error(`server ${r.status}: ${await r.text()}`);
    completed[s.id] = true;
    saveProgress();
    setStatus(`✓ uploaded sentence ${idx + 1}.`, "ok");
    discardCurrentBlob();
    // Auto-advance to the next unfinished sentence, if any.
    const nextIdx = nextUnfinishedAfter(idx);
    if (nextIdx !== null) idx = nextIdx;
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
  els.status.textContent = msg;
  els.status.className = kind ?? "";
}

function fatal(msg) {
  els.fatal.textContent = msg;
  els.fatal.hidden = false;
  els.app.hidden = true;
  els.greeting.textContent = "";
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
