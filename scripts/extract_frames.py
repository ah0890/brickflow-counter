"""Extract frames from a video into a YOLO dataset skeleton for labelling bricks.

Frames are sampled at a fixed time interval. The validation set is taken from a
*separate time range* (the end of the video by default) rather than random
frames, because neighbouring frames are near-duplicates and a random split
would overstate accuracy.

Usage:
    python scripts/extract_frames.py --video samples/site.mp4 --out datasets/bricks \
        --every 0.5 --val-fraction 0.2 [--start 0 --end 0]

Creates:
    <out>/images/train/*.jpg, <out>/images/val/*.jpg
    <out>/labels/train/, <out>/labels/val/      (empty: fill with YOLO .txt labels)
    <out>/data.yaml                             (single class: brick)
See docs/TRAINING.md for the labelling and training steps.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import safe_stem  # noqa: E402


def plan_frames(fps: float, frame_count: int, every_s: float, start_s: float = 0.0,
                end_s: float = 0.0, val_fraction: float = 0.2) -> tuple[list[int], list[int]]:
    """Return (train, val) frame indices. Validation = the last ``val_fraction`` of the time range."""
    if fps <= 0 or frame_count <= 0:
        raise ValueError("Video reports no frames or no frame rate.")
    if every_s <= 0:
        raise ValueError("--every must be positive.")
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("--val-fraction must be in [0, 1).")
    first = max(int(round(start_s * fps)), 0)
    last = frame_count - 1 if end_s <= 0 else min(int(round(end_s * fps)), frame_count - 1)
    if last <= first:
        raise ValueError("The selected time range is empty.")
    step = max(int(round(every_s * fps)), 1)
    indices = list(range(first, last + 1, step))
    if val_fraction == 0:
        return indices, []
    split_at = first + (last - first) * (1.0 - val_fraction)
    return [i for i in indices if i < split_at], [i for i in indices if i >= split_at]


def write_data_yaml(out: Path) -> Path:
    path = out / "data.yaml"
    path.write_text(
        f"path: {out.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: brick\n",
        encoding="utf-8",
    )
    return path


def extract(video: Path, out: Path, every_s: float, start_s: float, end_s: float,
            val_fraction: float, quality: int = 95) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {video}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        train, val = plan_frames(fps, count, every_s, start_s, end_s, val_fraction)
        wanted = {i: "train" for i in train} | {i: "val" for i in val}
        for split in ("train", "val"):
            (out / "images" / split).mkdir(parents=True, exist_ok=True)
            (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        stem = safe_stem(video.name, 40)
        written = {"train": 0, "val": 0}
        index, last = 0, max(wanted) if wanted else -1
        while index <= last:  # sequential read: seeking is unreliable in many codecs
            ok, frame = cap.read()
            if not ok:
                break
            split = wanted.get(index)
            if split:
                name = out / "images" / split / f"{stem}_f{index:06d}.jpg"
                cv2.imwrite(str(name), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
                written[split] += 1
            index += 1
    finally:
        cap.release()
    write_data_yaml(out)
    return written["train"], written["val"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--out", default=Path("datasets/bricks"), type=Path)
    ap.add_argument("--every", type=float, default=0.5, help="Seconds between extracted frames (default 0.5)")
    ap.add_argument("--start", type=float, default=0.0, help="Start time in seconds")
    ap.add_argument("--end", type=float, default=0.0, help="End time in seconds (0 = end of video)")
    ap.add_argument("--val-fraction", type=float, default=0.2,
                    help="Fraction of the time range, at its end, used for validation (default 0.2)")
    args = ap.parse_args()
    try:
        n_train, n_val = extract(args.video, args.out, args.every, args.start, args.end, args.val_fraction)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Wrote {n_train} training and {n_val} validation frames to {args.out}")
    print("Next: label bricks (YOLO format) and train - see docs/TRAINING.md")


if __name__ == "__main__":
    main()
