# BrickFlow – Project Context (handoff document)

Read this first at the start of every session, then verify against `git status` / `git log`.

## Objective
Local-first app that counts bricks in recorded video (truck loading/unloading, conveyors): YOLO11n detection → BoT-SORT tracking → count once per tracked brick crossing a configurable virtual line → annotated video + CSV report. Streamlit UI. Portfolio/demonstration project (Ultralytics AGPL-3.0 is acceptable for this use).

## Current milestone
**Milestone 1 – Foundation: complete.** Next: **Milestone 2 – Detection and tracking pipeline.**

## Completed implementation (M1)
- Validated YAML config (`src/config.py`, `config/settings.yaml`, `config/botsort.yaml`).
- Detector interface + Ultralytics implementation with operator-friendly `ModelLoadError` (`src/detection.py`).
- Utilities: safe filenames/job IDs, path sandboxing, display paths, verified device selection (`src/utils.py`).
- Streamlit shell (`app.py`, `ui/theme.py`, `.streamlit/config.toml`): home screen with real config/device/model status; other screens labelled "not implemented yet".
- Privacy: Streamlit telemetry off, server bound to localhost, `YOLO_OFFLINE=1`.

## Architecture
See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Core `src/` is framework-independent; `ui/` + `app.py` are presentation only. Counting logic (M3) must not import detector/tracker/UI code.

## Important files
| File | Purpose |
| --- | --- |
| `app.py` | Streamlit entry point |
| `src/config.py` | Config dataclasses + validation |
| `src/detection.py` | `Detector` protocol, `UltralyticsDetector`, `check_weights_file` |
| `src/utils.py` | Paths, sanitisation, `select_device` |
| `config/settings.yaml` | All app settings |
| `config/botsort.yaml` | Tracker parameters |

## Model and tracker configuration
- Weights: `models/brick_yolo11n.pt` (not supplied; user has no custom weights yet). conf 0.35, IoU 0.50, imgsz 640, device auto.
- BoT-SORT: high 0.25, low 0.10, new 0.30, buffer 45, match 0.8, GMC sparseOptFlow, ReID off.
- Counting defaults: anchor bottom_center, direction forward (right-hand side of the point1→point2 direction), hysteresis 6 px, min track frames 3, track timeout 90 frames, line (0.10,0.55)→(0.90,0.55).

## Technical decisions
- Python **3.13** venv at `.venv` (3.12 registration on dev machine is stale/broken; 3.14 avoided for CV libs).
- CPU PyTorch 2.15.0. Dev GPU: GTX 1050 3 GB, driver 457.49 — too old for current CUDA PyTorch; GPU support optional later after driver update.
- ffmpeg not installed system-wide → `imageio-ffmpeg` (pip) planned for H.264 transcoding in M5.
- Line placement UI: `streamlit-image-coordinates` (clicks) + sliders; `streamlit-drawable-canvas` avoided (unmaintained).
- Background processing (M4): worker thread + thread-safe progress state, UI polls with `st.fragment(run_every=...)`.
- Counting algorithm (M3): per-track confirmed side with hysteresis band; count on side flip only if the segment between confirmed points intersects the line segment; direction filter; min track age; one count per ID; state pruned after timeout to handle ID reuse; objects first seen inside the band must settle on one side first.

## Tests actually performed
- `pytest`: **50 passed** (config validation, utilities/path safety, model-load error paths, headless Streamlit AppTest smoke tests).
- Manual: `streamlit run app.py` started; `/_stcore/health` returned `ok`.
- **Not yet verified:** loading a valid model and running `track()` end-to-end (no weights available; downloading generic weights was blocked in the dev session). To be done in M2.
- UI not yet inspected visually in a browser (verified only via AppTest).

## Known bugs / limitations
- No brick weights → detection cannot be demonstrated on bricks yet. User will supply a real brick-loading video at the end of the project; custom weights still required.
- `UltralyticsDetector.reset()` behaviour across videos not yet verified (M2).

## Pending tasks
- M2: `tracking.py` (Track records), `video_io.py` (validation/probe/reader), frame pipeline, annotations, EOF/invalid-input handling; verify positive model-load path.
- M3: `line_crossing.py`, `counting.py`, 10 synthetic scenarios.
- M4: UI upload/preview/line config/live dashboard. M5: export + CSV. M6: testing, perf, security review, `pip-audit`.

## Commands
```powershell
.venv\Scripts\activate
streamlit run app.py
.venv\Scripts\python -m pytest
```

## Git
- Branch `main`, remote `origin` = github.com/ah0890/brickflow-counter.
- Commit/push status: see `git log` — M1 commit pending user confirmation of author identity at time of writing.
- Policy: commits under the user's identity only; no AI attribution in commits, PRs or repository files.
