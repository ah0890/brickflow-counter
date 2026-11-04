# Training a brick detection model

BrickFlow needs a YOLO11 detection model trained on bricks as they appear at your site. Generic pretrained models (COCO) have no "brick" class. This guide goes from your own footage to `models/brick_yolo11n.pt`.

## 1. Choose footage

- Use a **fixed camera** with the same angle, distance and lighting you will use for counting.
- Include the situations you need to count: single bricks, bricks carried in stacks, partial occlusion by hands/bodies, motion blur, different times of day.
- Keep some footage aside that you **never** train on. You will use it to measure counting accuracy at the end.

## 2. Extract frames

```powershell
.venv\Scripts\python scripts\extract_frames.py --video samples\site.mp4 --out datasets\bricks --every 0.5 --val-fraction 0.2
```

- `--every` – seconds between frames. 0.5–1 s avoids near-duplicate images.
- `--start` / `--end` – limit extraction to a time range (seconds).
- The **last 20 % of the time range** becomes the validation set. Neighbouring frames are almost identical, so a random split would put copies of training images in validation and overstate accuracy.

Aim for **150–300 labelled images** for a first model, more for varied scenes. The `datasets/` folder is not committed to git.

## 3. Label bricks

Draw one box per **visible brick** (class `0: brick`) and save labels in **YOLO format**, one `.txt` per image with the same file name, in `labels/train` and `labels/val`.

Local labelling tools that export YOLO format: [Label Studio](https://labelstud.io/) (run locally), [CVAT](https://www.cvat.ai/) (self-hosted), [labelImg](https://github.com/HumanSignal/labelImg).

Be consistent:
- Label every brick whose outline is clearly visible, including bricks in a carried stack (each brick separately), so the model and the line count agree.
- Skip bricks that are mostly hidden, rather than guessing their outline.
- Decide whether static stacked bricks (pallets, walls) should be labelled. If they never cross the counting line, labelling them is fine. If they do cross it, counting rules need care.

## 4. Train

```powershell
.venv\Scripts\yolo detect train model=yolo11n.pt data=datasets\bricks\data.yaml epochs=100 imgsz=640 batch=8 device=cpu
```

- `yolo11n.pt` is the pretrained starting point. Copy it into the working directory first, because BrickFlow runs Ultralytics in offline mode.
- On CPU, expect roughly 1–3 hours for 200 images × 100 epochs. With a supported NVIDIA GPU, use `device=0`.
- Results are written to `runs/detect/train*/`. Check `results.png` and the validation **mAP50**. Low precision means false boxes; low recall means missed bricks.

## 5. Use the model

```powershell
copy runs\detect\train\weights\best.pt models\brick_yolo11n.pt
```

`models/brick_yolo11n.pt` is the default in `config/settings.yaml`. Any `.pt`/`.onnx` in `models/` can also be selected on the Configure screen. Tune the **confidence threshold** there: raise it if non-bricks are counted, lower it if bricks are missed.

## 6. Measure counting accuracy

Detection mAP is not the same as counting accuracy. For each held-out video:

1. Count the bricks crossing the line by hand (twice, ideally by two people).
2. Process the video in BrickFlow with the final line position and direction.
3. Compare the totals and inspect `<job_id>_events.csv` and the annotated video for misses and double counts.

Report accuracy as total counted vs. manual count over all held-out videos, and state the conditions (camera, lighting, stacking).
