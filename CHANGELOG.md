# Changelog

All notable changes to BrickFlow are recorded here, newest first.

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
