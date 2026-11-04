# Changelog

All notable changes to BrickFlow are recorded here, newest first.

## [0.3.0] – 2026-11-04 – Milestone 3: Virtual line and counting engine

### Added

- `src/line_crossing.py`: line geometry with an explicit side convention, signed distance, hysteresis side test, robust segment intersection (collinear/endpoint cases), intersection point, direction helpers.
- `src/counting.py`: `CountingEngine` implementing the pipeline's `FrameCounter` — confirmed-side tracking with hysteresis, finite-segment crossing test, direction filter, one count per ID per job, pending confirmation after minimum track age, teleport guard, identity handover for tracker ID switches, state pruning, rejection statistics, and `CountEvent` records.
- Settings `max_step_ratio`, `handover_max_gap_frames`, `handover_max_distance`.
- 67 new tests: geometry edge cases and all synthetic counting scenarios (single, sequential, near-line jitter, reverse, temporary disappearance, repeated crossing, simultaneous, ID changes, appearing on the line, duplicate frames, track age, segment limits, jumps, vertical/diagonal lines). 148 tests total.
- Counting algorithm and limitations documented in `docs/ARCHITECTURE.md`.

## [0.2.0] – 2026-11-04 – Milestone 2: Detection and tracking pipeline

### Added

- `src/video_io.py`: layered video validation (existence, extension, size limit, container signature vs extension, first-frame decode, duration limit), `VideoInfo` probe, safe frame iteration and single-frame reads.
- `src/tracking.py`: framework-independent `Track` record with bottom-centre/centre anchors; conversion from Ultralytics results that drops detections without a tracker ID.
- `src/annotator.py`: scaled overlays — boxes, track IDs, confidence, counting line with direction arrow(s), prominent green count panel, frame/time/active-track readout.
- `src/video_processor.py`: frame pipeline with tracker reset per video, progress callbacks, cancellation, `FrameCounter` hook for the counting engine, `frame_sink` hook for export, truncation warning, and failure results instead of exceptions.
- Tests: video validation (corrupt, renamed, empty, truncated, oversized, too long), track conversion, pipeline mechanics with a fake detector, and real-model integration tests that skip when weights/footage are absent. 81 tests total.

### Changed

- `Detector.track()` now returns `list[Track]` instead of an Ultralytics `Results` object, keeping downstream code framework-independent.

## [0.1.0] – 2026-11-04 – Milestone 1: Foundation

### Added

- Project structure: `src/` (framework-independent core), `ui/` (Streamlit presentation), `config/`, `models/`, `outputs/`, `tests/`, `docs/`.
- `config/settings.yaml` with model, tracker, counting, video-limit and output settings; `config/botsort.yaml` BoT-SORT tracker configuration.
- `src/config.py`: typed, validated configuration loader with operator-readable errors.
- `src/detection.py`: `Detector` interface and `UltralyticsDetector` (YOLO11); pre-flight weights check; `ModelLoadError` for missing, unsupported, corrupt or non-detection weights; warning flag for generic COCO models.
- `src/utils.py`: safe filename/job-ID generation, path-traversal protection, path display without local absolute paths, compute-device selection with a verified CUDA test and CPU fallback.
- `app.py` Streamlit shell with dark industrial theme and system-status home screen; unbuilt screens are labelled as not implemented.
- `.streamlit/config.toml`: localhost binding, upload limit, telemetry disabled.
- Ultralytics offline mode (`YOLO_OFFLINE=1`) to prevent analytics and automatic downloads.
- 50 automated tests (configuration, utilities, model loading errors, headless UI smoke tests).
- README, context.md, CHANGELOG, architecture and troubleshooting docs; `requirements.txt` with verified versions; `.gitignore` excluding weights, videos, outputs and virtual environments.
