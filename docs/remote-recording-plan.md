# Remote voice-sample collection

Plan for letting non-technical contributors record sentences in the browser and have them land in `data/manual/<name>/*.flac` with matching rows in `data/data-pairs.csv` — without Chris doing per-contributor cleanup.

## Context

`just record` (Go + malgo + ffmpeg) produces 16 kHz mono FLAC into `data/manual/chris/sN.flac`, paired with sentences in `data/data-pairs.csv`. All existing manual recordings are Chris's voice, which risks overfitting a speech model to one speaker. The goal is contributions from 5–20 people × 50–100 sentences each, in the same `16 kHz mono FLAC` format as the existing dataset.

## Note on building

Whenever you need to build a binary for `cmd/contrib`, make sure it gets output to `bin/` where it's safely gitignored rather than in the root of the repo.

### Design decisions (locked in)

- **Hosting:** Static frontend on GitHub Pages + Cloudflare Worker + R2 (all free tier).
- **Encoding:** Browser records WAV at 16 kHz mono; Chris transcodes to FLAC locally on pull.
- **Auth:** Signed-link-per-contributor. No passwords. Token encodes the contributor's name.
- **Trimming:** None — keep whatever the browser captured.
- **Assignment:** Per-contributor custom lists (Chris picks each person's sentences).
- **Re-records:** Overwrite previous takes.

### Architecture

```
 Contributor browser                Cloudflare Worker            R2 bucket
 ┌──────────────────┐   HTTPS      ┌────────────────────┐    ┌─────────────┐
 │ Static site on   │ ──GET──────► │ /assignment?t=…    │    │             │
 │ GitHub Pages     │ ◄─JSON──────│  validates token   │    │ manual/     │
 │                  │              │  returns sentences │    │  alice/     │
 │  getUserMedia    │              │                    │    │   sent42.wav│
 │  → WAV 16 kHz    │ ─PUT audio─► │ /upload?t=…&id=42  │───►│             │
 │  → upload        │              │  validates token,  │    │             │
 │                  │              │  writes to R2      │    │             │
 └──────────────────┘              └────────────────────┘    └─────────────┘
                                                                   │
                                                       rclone sync │
                                                                   ▼
                                                         Chris's laptop
                                                    `just pull-recordings`
                                                    → ffmpeg WAV→FLAC
                                                    → data/manual/<name>/
                                                    → append to data-pairs.csv
```

---

## Phase 1 — Local tooling (pull + gen-assignment) ✅ COMPLETE

Build the local CLI and CSV append logic in isolation, before involving the network, so we know the transcode + dedupe work.

- [x] Add `data/wanted-sentences.csv` (pipe-delimited `ID|Sentence`, header row, seeded with ~15 sentences)
- [x] Create `cmd/contrib/main.go` with subcommands:
  - [x] `pull` — scans `work/uploads/manual/<name>/<id>.wav`, transcodes each to `data/manual/<name>/<id>.flac` at 16 kHz mono s16 via ffmpeg, deletes the WAV, appends new rows to `data-pairs.csv` (dedupe by path, never rewrites existing rows)
  - [x] `gen-assignment <name> <id>...` — writes `work/uploads/assignments/<name>.json` with `{name, sentences: [{id, text}]}`, prints a placeholder share URL
- [x] Add `justfile` recipes: `pull-recordings`, `gen-assignment name +ids`
- [x] Add `work/` to `.gitignore`
- [x] `go build ./cmd/contrib` and `go vet ./cmd/contrib` clean
- [x] End-to-end verification:
  - [x] `just gen-assignment test-user wanted-001 wanted-002` → JSON correct, URL printed
  - [x] Staged a WAV under `work/uploads/manual/test-user/` → `just pull-recordings` → `ffprobe` reports `flac / 16000 Hz / mono / s16`, matching `data/manual/chris/s1.flac`
  - [x] Idempotent re-run with no new WAVs is a no-op
  - [x] Re-record overwrites FLAC in place; CSV still has exactly one row
  - [x] `data-pairs.csv` byte-identical to pre-test snapshot after cleanup

---

## Phase 2 — Cloudflare Worker + R2 backend

Stand up the network tier so contributors can fetch assignments and upload audio. Still no frontend — verify with `curl`.

### Step 2a — Worker scaffold + HMAC only ✅ COMPLETE

No R2, no deploy — just the auth spine end-to-end between Go and the Worker.

- [x] Create `worker/` (TypeScript) with `wrangler.jsonc`, `tsconfig.json`, `package.json` (wrangler as local dev dep)
- [x] Token scheme locked in: `base64url(payload).base64url(hmacSHA256(secret, payload_b64))` where payload is compact JSON `{"name","exp"}`. Implementation in `internal/token` (Go) and `worker/src/index.ts` (TS) — shared secret in `JANGLISH_HMAC_SECRET`
- [x] Routes implemented with dev stubs (no R2 yet):
  - [x] `GET /assignment?t=<token>` — verify HMAC, return `{name, sentences: []}` stub
  - [x] `PUT /upload?t=<token>&id=<id>` — verify HMAC, require `audio/wav`, cap 10 MB, return `{bytes}`
  - [x] `GET /health` — 200 `ok`
- [x] CORS: preflight + `ALLOWED_ORIGIN` var (default `*` for dev)
- [x] `cmd/contrib gen-token <name> [ttl]` + matching `just gen-token` recipe
- [x] Tiny `.env` loader in `internal/token/env.go` so `cmd/contrib` reads `JANGLISH_HMAC_SECRET` from repo-root `.env`
- [x] `.dev.vars.example` checked in; `.dev.vars` gitignored
- [x] End-to-end verification: `wrangler dev` + `go run ./cmd/contrib gen-token` — good token → 200, bad/missing → 401, wrong content-type → 415, missing id → 400, OPTIONS → 204

### Step 2b — R2 wiring ✅ COMPLETE

- [x] Create R2 bucket in Cloudflare dashboard; add `r2_buckets` binding to `wrangler.jsonc` (binding `RECORDINGS`)
- [x] `GET /assignment` fetches `assignments/<name>.json` from R2 (falls back to 404 if missing)
- [x] `PUT /upload` writes body to `manual/<name>/<id>.wav` (R2 overwrites by default — matches re-record behavior). Also guards `id` against `^[A-Za-z0-9_-]+$` and `name` against `^[a-z0-9][a-z0-9-]*$`.
- [x] `contrib gen-assignment` uploads the JSON to R2, signs a real token, and prints `<JANGLISH_SITE_BASE>/?t=<token>`
- [x] `contrib list-progress [name]` — reads R2 to print `assigned / staged / done` per contributor
- [x] Rewrite `contrib pull` to `rclone copy` R2:manual/ → `work/uploads/manual/` and `rclone deletefile` each WAV from R2 after successful transcode (staging area drains end-to-end)
- [x] Configure `rclone` against R2; document the one-time setup in `docs/rclone-setup.md`. Required config: `no_check_bucket = true` (bucket-scoped tokens cannot call `CreateBucket`).
- [x] Add `list-progress` just recipe
- [x] End-to-end verification (against `wrangler dev --remote` bound to production R2):
  - [x] `just gen-assignment alice wanted-001 wanted-002` wrote `assignments/alice.json` to R2 and printed a signed share URL
  - [x] `curl -X PUT --data-binary @sample.wav "<url>&id=wanted-001"` landed `manual/alice/wanted-001.wav` in R2 (200, plus 400/401/415 from the negative cases)
  - [x] `just pull-recordings` produced `data/manual/alice/wanted-001.flac` (`ffprobe`: flac / 16000 Hz / mono / s16), appended one CSV row, drained R2; idempotent re-run was a no-op

Deployed URL: https://janglish-recordings.janglish.workers.dev

### Step 2c — Deploy ✅ COMPLETE

- [x] `wrangler secret put JANGLISH_HMAC_SECRET` against production (value matches repo-root `.env` so Go-signed tokens verify)
- [x] `wrangler deploy` the worker; end-to-end verification against the live URL:
  - [x] `GET /health` → 200, bad token → 401, `OPTIONS` → 204
  - [x] `gen-assignment prod-check …` uploaded JSON to R2; `GET /assignment?t=…` returned the sentences
  - [x] `PUT /upload` 200 for valid WAV; negative cases 415 / 400 / 401
  - [x] `list-progress` correctly reported `assigned=2, staged=1` then `done=1` after pull
  - [x] `just pull-recordings` drained R2, produced `data/manual/prod-check/wanted-001.flac` (flac / 16000 Hz / mono / s16), appended one CSV row; idempotent re-run was a no-op
  - [x] Test artifacts cleaned up (CSV restored from snapshot, local + R2 assignment and manual/ prefixes removed)
- [ ] Confirm R2 free-tier usage is zero-cost for expected volume

---

## Phase 3 — Static frontend (GitHub Pages)

Build the contributor-facing UI. Plain HTML/JS — no framework.

- [ ] Create `web/` with `index.html`, `app.js`, `wav-encoder.js`
- [ ] On load: parse `?t=<token>` from URL, `GET /assignment?t=…` → store `{name, sentences}`
- [ ] Record UI: one sentence at a time, with buttons for **Record / Stop / Play back / Re-record / Next / Previous** and an `X of N` progress indicator
- [ ] Audio capture via `getUserMedia({audio: {channelCount: 1, sampleRate: 16000, echoCancellation: false, noiseSuppression: false}})`; downsample in an `AudioWorklet` if the browser doesn't honor the 16 kHz hint
- [ ] Encode captured PCM to 16-bit WAV in-browser (~100-line encoder, no library)
- [ ] On **Next**: `PUT /upload?t=…&id=<sentence-id>` with the WAV bytes. Show ✓ on 200. Keep the blob locally until confirmed so failed uploads are retryable.
- [ ] Persist progress in `localStorage` keyed by token so closing the tab doesn't lose state
- [ ] Add `.github/workflows/pages.yml` to deploy `web/` to GitHub Pages
- [ ] End-to-end verification:
  - [ ] Local: `python -m http.server` + `wrangler dev`, full record-and-upload round-trip
  - [ ] Deployed: test on desktop Chrome, Firefox, Safari
  - [ ] Deployed: test on mobile (iOS Safari + Android Chrome) — `getUserMedia` permissions + upload success
  - [ ] After recording several sentences, `just pull-recordings` produces correct FLACs and CSV rows

---

## Phase 4 — Dogfood and contributor onboarding

- [ ] Write a short `README.md` aimed at contributors (what they'll be asked to do, time estimate, privacy note)
- [ ] Run Chris through the full flow as a "contributor" on a second device; fix anything awkward
- [ ] Invite one real contributor; watch for problems
- [ ] Iterate on sentence set / UI copy based on first-contributor feedback
- [ ] Scale to the remaining contributors

---

## File inventory

**Created in Phase 1:**
- `cmd/contrib/main.go`
- `data/wanted-sentences.csv`
- `justfile` (extended)
- `.gitignore` (extended to ignore `work/`)

**Created in Phase 2a:**
- `worker/src/index.ts`, `worker/wrangler.jsonc`, `worker/package.json`, `worker/tsconfig.json`, `worker/.dev.vars.example`
- `internal/token/token.go` (+ test), `internal/token/env.go`
- `cmd/contrib/main.go` gains `gen-token`
- `justfile` gains `gen-token`

**Created in Phase 2b:**
- `cmd/contrib/r2.go` (rclone shell-out helpers)
- `cmd/contrib/main.go` gained `list-progress`, rclone-backed `pull`, R2-uploading `gen-assignment`
- `justfile` gained `list-progress`
- `docs/rclone-setup.md`
- `worker/wrangler.jsonc` gained the `RECORDINGS` R2 binding

**Created in Phase 2c:**
- Production secret `JANGLISH_HMAC_SECRET` (set via `wrangler secret put`, matches repo-root `.env`)
- Deployed Worker at https://janglish-recordings.janglish.workers.dev (version `c618bff9-0f35-4ee5-89ce-a617590c4067`)

**To create in Phase 3:**
- `web/index.html`, `web/app.js`, `web/wav-encoder.js`
- `.github/workflows/pages.yml`

**To create in Phase 4:**
- `README.md` (contributor-facing)

---

## Constraints and cross-cutting rules

- All new audio must end up as **16 kHz mono FLAC s16** to match `data/manual/chris/*.flac`. Verify with `ffprobe` after any pipeline change.
- `data/data-pairs.csv` is pipe-delimited with backslash-escaped double quotes. Never rewrite existing rows — open-append only. New rows may use Go's default CSV quoting; that's fine for the current sentence set.
- Idempotency is a hard requirement for `pull-recordings` — re-runs must never produce duplicate CSV rows or re-transcode files that are already done.
- Everything in `work/` is local staging only and is gitignored.
