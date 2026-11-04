# BrickFlow – Automated Brick Counting

Automated brick counting system using YOLO11 object detection, BoT-SORT multi-object tracking, and virtual line-crossing detection to count bricks in videos, generate annotated footage, and maintain digital counting records.

> **Status:** Milestone 1 of 6 (foundation) complete. Detection/tracking pipeline, counting engine, full UI and exports are in progress — see [CHANGELOG.md](CHANGELOG.md) and [context.md](context.md).

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
| YOLO11 + BoT-SORT video pipeline | Planned (M2) |
| Trajectory-based line-crossing counting engine | Planned (M3) |
| Video upload, line placement, live counting dashboard | Planned (M4) |
| Annotated video export and CSV reports | Planned (M5) |

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

Only load weights from sources you trust — `.pt` files can execute code when loaded. Without weights, the app starts and shows a "Model weights required" notice.

## Run

```powershell
.venv\Scripts\activate
streamlit run app.py
```

Opens at `http://localhost:8501`. The server binds to `localhost` only and Streamlit usage statistics are disabled.

## Tests

```powershell
.venv\Scripts\python -m pytest
```

## Configuration

- [config/settings.yaml](config/settings.yaml) – model, thresholds, device, counting line, direction, file limits.
- [config/botsort.yaml](config/botsort.yaml) – BoT-SORT tracker parameters.

Invalid values are rejected at startup with a message naming the setting.

## Privacy

All processing is local. No uploads, cloud storage, analytics or telemetry: Streamlit usage stats are disabled and Ultralytics runs with `YOLO_OFFLINE=1` (no analytics, no automatic downloads).

## Limitations

- Counting accuracy depends on the detection model and on tracking quality; no accuracy figure is claimed without a labelled evaluation.
- Tracking-based counting can miss or double-count when bricks are heavily occluded, overlap, or when the tracker switches IDs as a brick crosses the line.
- CPU processing is considerably slower than real time.

## Licensing

Ultralytics YOLO is licensed under **AGPL-3.0**. This project is a portfolio/demonstration project and is compatible with that licence. Commercial closed-source or hosted use would require an Ultralytics Enterprise licence or replacing the detector (the detector sits behind an interface for this reason).
