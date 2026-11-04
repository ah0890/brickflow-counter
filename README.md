# BrickFlow – Automated Brick Counting

Automated brick counting system using YOLO11 object detection, BoT-SORT multi-object tracking, and virtual line-crossing detection to count bricks in videos, generate annotated footage, and maintain digital counting records.

> **Status:** All six milestones complete (v1.0.0). The pipeline, counting engine, UI and exports are tested. **No brick model is included**: train one with [docs/TRAINING.md](docs/TRAINING.md). See [CHANGELOG.md](CHANGELOG.md).

## Why

Manual brick counting during truck loading, unloading and conveyor operations is slow, error-prone and hard to verify in quantity disputes. BrickFlow processes recorded footage locally and produces a count, an annotated video and a CSV record of every counting event.

## How it works

1. **Detect** – a custom-trained YOLO11n model finds bricks in each frame.
2. **Track** – BoT-SORT assigns each brick a persistent ID across frames.
3. **Count** – each tracked brick is counted once when its trajectory crosses a virtual line, in the configured direction.
4. **Record** – annotated video and CSV report are saved locally.

## Features

| Feature | Status |
| --- | --- |
| Validated YAML configuration | Done (M1) |
| Model-loading interface with clear errors for missing/invalid weights | Done (M1) |
| CPU processing, GPU auto-selection with verified CPU fallback | Done (M1) |
| Streamlit shell with system status | Done (M1) |
| Video validation (type, size, signature, decodability, duration) | Done (M2) |
| YOLO11 + BoT-SORT frame pipeline with boxes, IDs, line and count overlay | Done (M2) |
| Trajectory-based line-crossing counting engine (direction filter, jitter band, duplicate/ID-switch safeguards) | Done (M3) |
| Video upload or local selection, click-to-place counting line, model/threshold/direction settings | Done (M4) |
| Live processing dashboard (annotated preview, count, progress, FPS, ETA, cancel) and results screen | Done (M4) |
| Annotated video export (browser-playable H.264), summary and event CSV reports, recent-jobs list | Done (M5) |

## Technology stack

Python 3.13 · Ultralytics YOLO11 · BoT-SORT (via Ultralytics) · OpenCV · NumPy · Pandas · Streamlit · PyYAML · imageio-ffmpeg · Pytest

## Installation (Windows)

```powershell
# From the repository root
py -3.13 -m venv .venv
.venv\Scripts\activate

# 1. PyTorch for your hardware (CPU shown; see pytorch.org for CUDA builds)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 2. Everything else
pip install -r requirements.txt
```

macOS/Linux: use `python3 -m venv .venv` and `source .venv/bin/activate`.

## Model weights

BrickFlow **does not ship a brick detection model**. A generic pretrained YOLO model has no "brick" class and will not count bricks reliably.

1. Train (or obtain) a YOLO11 detection model on images of bricks from your site/camera angle.
2. Copy the weights file (`.pt` or `.onnx`) into `models/`.
3. Set `model.weights` in [config/settings.yaml](config/settings.yaml) (default `models/brick_yolo11n.pt`).

No model yet? [docs/TRAINING.md](docs/TRAINING.md) covers frame extraction (`scripts/extract_frames.py`), labelling, training and measuring counting accuracy.

Only load weights from sources you trust — `.pt` files can execute code when loaded. Without weights, the app starts and shows a "Model weights required" notice.

## Run

```powershell
.venv\Scripts\activate
streamlit run app.py
```

Opens at `http://localhost:8501`. The server binds to `localhost` only and Streamlit usage statistics are disabled.

## Using the app

1. **Choose video** – upload a file, or pick one from `samples/`. The video is validated before use.
2. **Configure** – click two points on a frame to place the counting line (the arrow shows the counted direction), pick the model, classes, thresholds and direction.
3. **Process** – watch the annotated preview, live count, progress and speed; cancel at any time (partial results are kept).
4. **Results** – final count, annotated video, processing summary, counting-event table and downloads.

## Outputs

Each run creates `outputs/<job_id>/`:

| File | Contents |
| --- | --- |
| `<job_id>_annotated.mp4` | Video with boxes, track IDs, counting line and running count (H.264) |
| `<job_id>_summary.csv` | Video, model, thresholds, line, direction, status, total count, timing, diagnostics |
| `<job_id>_events.csv` | One row per counted crossing: count, track ID, frame, timestamp, direction, crossing point |
| `job.json` | Metadata for the "Recent jobs" list |

## Tests

```powershell
.venv\Scripts\python -m pytest
```

## Configuration

- [config/settings.yaml](config/settings.yaml) – model, thresholds, device, counting line, direction, file limits.
- [config/botsort.yaml](config/botsort.yaml) – BoT-SORT tracker parameters.

Invalid values are rejected at startup with a message naming the setting.

## Performance

Measured on a CPU-only Windows laptop (PyTorch CPU build), 716×1274 video, YOLO11n at 640 px: **5–10 fps** processing, i.e. about 3–6× slower than real time for 30 fps footage. Decoding and annotation alone run at ~150 fps, so detection dominates. To speed it up: use a CUDA GPU, lower `model.imgsz` (e.g. 480), or use a trimmed clip.

## Privacy

All processing is local. No uploads, cloud storage, analytics or telemetry: Streamlit usage stats are disabled and Ultralytics runs with `YOLO_OFFLINE=1` (no analytics, no automatic downloads). See [docs/SECURITY.md](docs/SECURITY.md) for the security review.

## Limitations

- Counting accuracy depends on the detection model and on tracking quality; no accuracy figure is claimed without a labelled evaluation.
- Tracking-based counting can miss or double-count when bricks are heavily occluded, overlap, or when the tracker switches IDs as a brick crosses the line.
- **A fixed camera is required.** The counting line is fixed in the image, so with a handheld or panning camera, stationary bricks move across the image and can be counted.
- See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#known-limitations-of-tracking-based-counting) for the full list.
- CPU processing is considerably slower than real time.

## Licensing

Ultralytics YOLO is licensed under **AGPL-3.0**. This project is a portfolio/demonstration project and is compatible with that licence. Commercial closed-source or hosted use would require an Ultralytics Enterprise licence or replacing the detector (the detector sits behind an interface for this reason).
