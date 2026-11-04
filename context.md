# BrickFlow – Project Context (handoff document)

Read this first at the start of every session, then verify against `git status` / `git log`.

## Objective

Local-first app that counts bricks in recorded video (truck loading/unloading, conveyors): YOLO11n detection → BoT-SORT tracking → count once per tracked brick crossing a configurable virtual line → annotated video + CSV report. Streamlit UI. Portfolio/demonstration project (Ultralytics AGPL-3.0 is acceptable for this use).

## Current milestone

**Milestones 1–2 complete.** Next: **Milestone 3 – line-crossing geometry and counting engine.** In parallel, a brick-detection model is needed (see "Model situation").

## Completed implementation

- **M1** – validated YAML config (`src/config.py`), Detector interface (`src/detection.py`), utilities (`src/utils.py`), Streamlit shell (`app.py`, `ui/theme.py`), telemetry off, localhost-only.
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
| `tests/conftest.py` | Synthetic video writer + `FakeDetector` |

## Model and tracker configuration

- Configured weights: `models/brick_yolo11n.pt` — **does not exist yet** (no brick model). conf 0.35, IoU 0.50, imgsz 640, device auto.
- Present locally (gitignored): `models/yolo11n.pt` (generic COCO, supplied by user for pipeline verification only).
- BoT-SORT: high 0.25, low 0.10, new 0.30, buffer 45, match 0.8, GMC sparseOptFlow, ReID off. Ultralytics builds trackers with frame_rate=30, so `track_buffer` is in 30-fps frames.
- Counting defaults: anchor bottom_center, direction forward (right-hand side of the point1→point2 direction, normal (-dy, dx)), hysteresis 6 px, min track frames 3, track timeout 90 frames, line (0.10,0.55)→(0.90,0.55).

## Model situation (decision pending with user)

User has no brick weights and only one video. Options presented: (a) open-vocabulary detector (e.g. YOLO-World prompted with "brick") — no training, rough; (b) extract ~150–300 frames, pre-label with (a), user corrects in a local labelling tool, train YOLO11n on CPU (hours). Train/test must use separate time ranges of the video.

## Sample footage

`samples/Real_video.mp4` (gitignored, user-supplied): 716×1274 portrait, 30 fps, 909 frames (30.3 s), H.264, 9.5 MB. Handheld/moving camera, several edited scenes (truck unloading 0–5 s and 23–30 s, stacks/loose pile ~8–20 s, wheelbarrows ~15 s), bricks carried in stacks of ~8–10, many static stacked bricks, burned-in advertising text with a third-party phone number in the lower third. **Not suitable for a meaningful count** (camera motion makes static bricks cross a screen-fixed line; scene cuts break tracks). User chose to proceed with it anyway: use for pipeline demonstration and training-data extraction; do not present its count as an accurate result. Get permission before publishing the footage.

## Tests actually performed

- `pytest`: **81 passed** (config, utils/path safety, model-load errors, video validation incl. corrupt/renamed/truncated, track conversion, pipeline mechanics with fake detector, headless Streamlit smoke tests, 3 real-model integration tests that skip if weights/sample absent).
- Real run, generic `yolo11n.pt` tracking **people** (class 0) on the sample video, CPU: 909/909 frames, 92.6 s, **9.8 fps average**, 22 unique track IDs; main workers held one ID for 130–230 frames with few gaps; tracker reset verified (IDs restart at 1, deterministic). Model load 4.6 s. This verifies pipeline mechanics only — not brick detection.
- Pipeline without model on sample: 909/909 frames at ~156 fps (decode + annotate overhead).
- Overlay visually inspected on sample frames: legible on portrait footage.
- Not yet: GPU (CPU PyTorch installed; GTX 1050 / driver 457.49 too old), UI in a real browser.

## Known bugs / limitations

- No brick model → no brick detection yet.
- Info bar (top-right) can overlap labels of boxes touching the top edge — revisit in M4 UI polish.
- Generic model missed some partially occluded workers (detector limitation, expected).

## Pending tasks

- M3: `line_crossing.py`, `counting.py` (implements `FrameCounter`), 10 synthetic scenarios; consider scene-cut detection (reset tracks on hard cuts) given the sample footage.
- Brick model: user decision between open-vocabulary bootstrap and labelling + training.
- M4: UI upload/preview/line config/live dashboard (background thread + `st.fragment(run_every)`). M5: export (`frame_sink` + imageio-ffmpeg H.264) + CSV. M6: testing, perf, security review, `pip-audit`.

## Commands

```powershell
.venv\Scripts\activate
streamlit run app.py
.venv\Scripts\python -m pytest
```

## Git

- Branch `main`, remote `origin` = github.com/ah0890/brickflow-counter. Author identity: user's configured git identity (do not modify).
- M1 committed and pushed as `6ccb513`. M2 commit: see `git log`.
- Policy: commits under the user's identity only; no AI attribution in commits, PRs or repository files.
