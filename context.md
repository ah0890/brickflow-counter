# BrickFlow – Project Context (handoff document)

Read this first at the start of every session, then verify against `git status` / `git log`.

## Objective

Local-first app that counts bricks in recorded video (truck loading/unloading, conveyors): YOLO11n detection → BoT-SORT tracking → count once per tracked brick crossing a configurable virtual line → annotated video + CSV report. Streamlit UI. Portfolio/demonstration project (Ultralytics AGPL-3.0 is acceptable for this use).

## Current milestone

**Milestones 1–4 complete.** Next: **Milestone 5 – annotated video export + CSV reports** (`src/video_export.make_sink_factory`, `src/report_generator.write_job_reports` / `list_recent_jobs` — the UI already imports these lazily and lights up when they exist). User instructed (2026-11-04) to continue through all milestones without waiting for approval.

## Completed implementation

- **M1** – validated YAML config (`src/config.py`), Detector interface (`src/detection.py`), utilities (`src/utils.py`), Streamlit shell (`app.py`, `ui/theme.py`), telemetry off, localhost-only.
- **M3** – line geometry (`src/line_crossing.py`) and `CountingEngine` (`src/counting.py`): hysteresis, finite-segment intersection, direction filter, one count per ID, pending confirmation after min age, teleport guard, ID handover, pruning, `CountingStats`. Algorithm + limitations in docs/ARCHITECTURE.md.
- **M4** – four-step Streamlit UI (`ui/pages/home|configure|process|results.py`), `ui/state.py` (session state, cached resources), `src/job.py` (`ProcessingJob` background thread + `JobSnapshot`, cancel, `sink_factory`/`finalizer` hooks). Click-to-place line via `streamlit-image-coordinates`. Temp uploads in `outputs/.tmp` deleted on replacement and after `temp_max_age_hours`.
- **M2** – video validation/probing/reading (`src/video_io.py`), `Track` records (`src/tracking.py`), overlays (`src/annotator.py`), frame pipeline with counter/sink hooks, cancel and failure handling (`src/video_processor.py`). `Detector.track()` returns `list[Track]`.

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Core `src/` is framework-independent; `ui/` + `app.py` are presentation only. The counting engine (M3) must implement `FrameCounter` (`update(tracks, frame_index) -> events`, `total`, `counted_ids`) from `src/video_processor.py` and must not import detector/tracker/UI code.

## Important files

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit entry point (home screen only so far) |
| `src/config.py` | Config dataclasses + validation |
| `src/detection.py` | `Detector` protocol, `UltralyticsDetector`, `check_weights_file` |
| `src/tracking.py` | `Track`, `tracks_from_ultralytics` |
| `src/video_io.py` | `validate_video_file`, `probe_video`, `iter_frames`, `read_frame` |
| `src/annotator.py` | Drawing functions for tracks, line, count panel, info bar |
| `src/video_processor.py` | `process_video`, `ProcessingOptions`, `FrameCounter`, `ProcessingResult` |
| `config/settings.yaml` | All app settings |
| `config/botsort.yaml` | Tracker parameters |
| `src/line_crossing.py` | `Line`, `segments_intersect`, `intersection_point`, direction helpers |
| `src/counting.py` | `CountingEngine`, `CountingParams`, `CountEvent`, `build_counter` |
| `src/job.py` | `ProcessingJob`, `JobSpec`, `JobSnapshot`, `JobStatus` |
| `ui/state.py`, `ui/pages/*.py` | Streamlit session state and the four screens |
| `scripts/ui_screenshots.py` | Playwright (installed Edge) UI walk-through + screenshots |
| `tests/conftest.py` | Synthetic video writer + `FakeDetector` |

## Model and tracker configuration

- Configured weights: `models/brick_yolo11n.pt` — **does not exist yet** (no brick model). conf 0.35, IoU 0.50, imgsz 640, device auto.
- Present locally (gitignored): `models/yolo11n.pt` (generic COCO, supplied by user for pipeline verification only).
- BoT-SORT: high 0.25, low 0.10, new 0.30, buffer 45, match 0.8, GMC sparseOptFlow, ReID off. Ultralytics builds trackers with frame_rate=30, so `track_buffer` is in 30-fps frames.
- Counting defaults: anchor bottom_center, direction forward (right-hand side of the point1→point2 direction, normal (-dy, dx)), hysteresis 6 px, min track frames 3, track timeout 90 frames, max_step_ratio 1.5, handover gap 15 frames / distance 0.75 diag, line (0.10,0.55)→(0.90,0.55).

## Model situation (decision pending with user)

User has no brick weights and only one video. Options presented: (a) open-vocabulary detector (e.g. YOLO-World prompted with "brick") — no training, rough; (b) extract ~150–300 frames, pre-label with (a), user corrects in a local labelling tool, train YOLO11n on CPU (hours). Train/test must use separate time ranges of the video.

## Sample footage

`samples/Real_video.mp4` (gitignored, user-supplied): 716×1274 portrait, 30 fps, 909 frames (30.3 s), H.264, 9.5 MB. Handheld/moving camera, several edited scenes (truck unloading 0–5 s and 23–30 s, stacks/loose pile ~8–20 s, wheelbarrows ~15 s), bricks carried in stacks of ~8–10, many static stacked bricks, burned-in advertising text with a third-party phone number in the lower third. **Not suitable for a meaningful count** (camera motion makes static bricks cross a screen-fixed line; scene cuts break tracks). User chose to proceed with it anyway: use for pipeline demonstration and training-data extraction; do not present its count as an accurate result. Get permission before publishing the footage.

## Tests actually performed

- M4: `pytest` **162 passed**. Full browser walk-through (headless Edge via `scripts/ui_screenshots.py --video Real_video.mp4 --process`): select sample → configure → process (6.5 fps CPU, live preview/count/ETA updating) → results (count 3, generic-model warning shown, "no output files" as expected before M5).

- `pytest`: **148 passed** (M3 added 67 geometry/counting tests covering all 10 synthetic scenarios; scenario 10 = video validation tests).
- Real counting run (generic model tracking people, direction both, CPU): 909 frames, 5.9 fps (slower than the 9.8 fps M2 run; machine load varies), 3 counts, 5 repeat crossings blocked (already counted), 4 handovers, 0 jumps. One count was a stationary person swept across the line by camera panning — confirms the camera-motion limitation. Mechanics check only, not brick counting.
- Earlier: `pytest` **81 passed** at M2 (config, utils/path safety, model-load errors, video validation incl. corrupt/renamed/truncated, track conversion, pipeline mechanics with fake detector, headless Streamlit smoke tests, 3 real-model integration tests that skip if weights/sample absent).
- Real run, generic `yolo11n.pt` tracking **people** (class 0) on the sample video, CPU: 909/909 frames, 92.6 s, **9.8 fps average**, 22 unique track IDs; main workers held one ID for 130–230 frames with few gaps; tracker reset verified (IDs restart at 1, deterministic). Model load 4.6 s. This verifies pipeline mechanics only — not brick detection.
- Pipeline without model on sample: 909/909 frames at ~156 fps (decode + annotate overhead).
- Overlay visually inspected on sample frames: legible on portrait footage.
- Not yet: GPU (CPU PyTorch installed; GTX 1050 / driver 457.49 too old), UI in a real browser.

## Known bugs / limitations

- No brick model → no brick detection yet.
- Info bar (top-right) can overlap labels of boxes touching the top edge (cosmetic).
- Generic model missed some partially occluded workers (detector limitation, expected).

## Pending tasks

- Brick model: user did not choose; proceeding without one. Provide frame-extraction script + training guide (docs/TRAINING.md) so a model can be trained later.
- Scene-cut detection was evaluated on the sample (histogram correlation and grey-level difference): only 1 of ~4 cuts separable from handheld shake → not implemented; teleport guard used instead.
- M5: export (`frame_sink` + imageio-ffmpeg H.264) + CSV. M6: testing, perf, security review, `pip-audit`.

## Commands

```powershell
.venv\Scripts\activate
streamlit run app.py
.venv\Scripts\python -m pytest
```

## Git

- Branch `main`, remote `origin` = github.com/ah0890/brickflow-counter. Author identity: user's configured git identity (do not modify).
- M1 `6ccb513`, M2 `4cdd0c9`, M3 `44fbc02` (pushed). M4 onward: see `git log`.
- Policy: commits under the user's identity only; no AI attribution in commits, PRs or repository files.
