# BrickFlow Architecture

## Layers

```text
app.py + ui/            Streamlit presentation only (no counting logic)
        │
        ▼
src/                    Framework-independent core
  config.py             YAML → validated, typed AppConfig
  detection.py          Detector protocol + UltralyticsDetector (YOLO11, BoT-SORT via model.track) → list[Track]
  tracking.py           Plain Track records; conversion from Ultralytics results (untracked boxes dropped)
  video_io.py           Video validation / probing / frame reading
  annotator.py          Boxes, IDs, line + direction arrow, count panel, frame info
  video_processor.py    Frame loop: reset → track → counter hook → annotate → sink; callbacks, cancel
  utils.py              Paths, safe filenames, sandboxing, device selection
  (planned)
  line_crossing.py      Pure geometry: side-of-line, segment intersection, hysteresis
  counting.py           CountingEngine (implements FrameCounter): per-track state, direction, dedup, events
  video_export.py       Video writer + H.264 transcode (plugs into frame_sink)
  report_generator.py   Summary and event CSVs
```

Per-frame data flow:

```text
iter_frames ─► Detector.track(frame) ─► list[Track] ─► FrameCounter.update ─► events
                                             │                                   │
                                             └──────► annotator ◄────────────────┘
                                                          │
                                         on_frame (UI)  ◄─┴─►  frame_sink (writer)
```

`process_video` never raises for per-frame failures: detector errors, writer errors and truncated files end the run with `status="failed"` (or a warning) and keep the partial results.

Dependency rule: `counting.py` and `line_crossing.py` depend only on the standard library and NumPy, so the counting logic is tested with synthetic trajectories and is independent of the detector, tracker and UI.

## Key decisions

| Decision | Reason |
| --- | --- |
| Python 3.13 venv | 3.12 install on the dev machine was broken; 3.13 has stable wheels for Torch, OpenCV, Ultralytics. 3.14 avoided for CV-library compatibility. |
| CPU PyTorch by default | Dev GPU (GTX 1050, driver 457.49) cannot run current CUDA PyTorch builds. `select_device` verifies CUDA with a test kernel before using it. |
| Detector behind a Protocol | Ultralytics is AGPL-3.0; allows swapping for a permissively licensed detector without touching counting. |
| `YOLO_OFFLINE=1` | Disables Ultralytics analytics and silent downloads (local-first requirement). |
| Weights restricted to `models/` in the UI (planned) | `.pt` files are pickles; limiting selectable locations reduces risk. |
| Counting line stored as fractions of frame size | Resolution-independent; converted to pixels per video. |
| Counting direction relative to line orientation | "forward" = movement to the right-hand side of the point1→point2 direction in image coordinates. |
| ReID disabled in BoT-SORT | Bricks are visually near-identical; appearance features add cost with little identity benefit. |
