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

## Counting algorithm (`src/line_crossing.py`, `src/counting.py`)

Geometry convention (image coordinates, y down): the line runs from point 1 to point 2; the **positive side** is the right-hand side of that direction (normal `(-dy, dx)`). A **forward** crossing moves from the negative to the positive side — for a line drawn left-to-right, that is top-to-bottom.

Per tracked ID the engine keeps a *confirmed side*, the last anchor seen on that side, age, last position and a counted flag. On every frame:

1. **Anchor** = bottom-centre of the box (configurable: `center`).
2. **Hysteresis band**: anchors within `hysteresis_px` of the line do not change the confirmed side.
3. **Side flip** (confirmed side A → clearly on side B) is a *candidate crossing*. It becomes a pending count only if
   - the segment from the last confirmed anchor to the current anchor intersects the **finite** line segment (orientation test, collinear/endpoint cases handled),
   - the direction is allowed (`forward` / `reverse` / `both`),
   - the ID has not already been counted.
4. **Minimum age**: a pending crossing is registered once the track has existed `min_track_frames` frames; if it flips back first, it is cancelled. Registered event records the *crossing* frame and the *confirmation* frame.
5. **One count per ID per job**. Counted IDs are remembered permanently, even after the track's state is pruned.
6. **First seen on the line** → no side until it leaves the band, so appearing on or past the line is never a crossing.
7. **Teleport guard**: a move greater than `max_step_ratio` × box diagonal × elapsed frames resets the side (scene cuts, identity swaps).
8. **Identity handover**: a *new* ID appearing within `handover_max_distance` × diagonal of an ID lost within `handover_max_gap_frames` inherits its side, pending crossing and counted flag; each lost ID can be inherited once.
9. Lost tracks' state is pruned after `track_timeout_frames`.

`CountingStats` records why candidate crossings were rejected (outside segment, wrong direction, already counted, cancelled, jumps, handovers) for diagnostics.

### Known limitations of tracking-based counting

- **Camera motion**: the line is fixed in image coordinates. With a handheld or panning camera, *stationary* objects move across the image and can be counted. A fixed camera is required for meaningful counts.
- **Missed detections**: an object never detected (occluded inside a carried stack, motion blur) cannot be counted.
- **ID switches** handover cannot resolve (new ID appears far from the lost one or after the gap window) can cause a miss; a switch to an ID that was on the far side before crossing is not double-counted, but other pathological swaps can be.
- **Handover between touching objects**: if one object's track is lost and a *different*, adjacent object receives a new ID within the handover distance, the second object can inherit the first one's counted flag and be missed. Lower `handover_max_distance` for dense scenes.
- **Very fast objects**: moves larger than `max_step_ratio` diagonals per frame are rejected as jumps.
- Accuracy must be measured on labelled footage from the target site; no accuracy figure is claimed.
