"""CSV reports and job metadata.

Each job directory ``outputs/<job_id>/`` receives:

* ``<job_id>_summary.csv`` – one ``field,value`` row per setting/result.
* ``<job_id>_events.csv``  – one row per counted crossing.
* ``job.json``             – small metadata record used for the recent-jobs list.

CSV cells that a spreadsheet would interpret as a formula (leading ``=``,
``+``, ``-``, ``@``, tab or carriage return) are prefixed with an apostrophe,
because file names are user-controlled. Files are UTF-8 with BOM so Excel
detects the encoding.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.job import JobSnapshot, JobSpec
    from src.video_processor import ProcessingResult

METADATA_FILE = "job.json"
_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")

SUMMARY_FILE = "{job_id}_summary.csv"
EVENTS_FILE = "{job_id}_events.csv"
EVENT_COLUMNS = [
    "count", "track_id", "crossing_frame", "crossing_time_s", "crossing_timestamp",
    "confirmed_frame", "direction", "crossing_x_px", "crossing_y_px", "inherited_from_track",
]


def csv_safe(value: Any) -> Any:
    """Neutralise spreadsheet formula injection for string cells."""
    if isinstance(value, str) and value.startswith(_FORMULA_PREFIXES):
        return "'" + value
    return value


def fmt_timestamp(seconds: float) -> str:
    seconds = max(seconds, 0.0)
    ms = int(round(seconds * 1000))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> Path:
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for row in rows:
            w.writerow([csv_safe(v) for v in row])
    partial.replace(path)
    return path


def summary_rows(spec: JobSpec, snap: JobSnapshot, result: ProcessingResult) -> list[list[Any]]:
    v = spec.video
    p1, p2 = spec.counting.line.to_pixels(v.width, v.height)
    ln = spec.counting.line
    mi = snap.model_info
    classes = "all"
    if spec.model.class_ids is not None:
        names = mi.class_names if mi else {}
        classes = "; ".join(f"{i}:{names.get(i, '?')}" for i in spec.model.class_ids)
    stats = snap.counting_stats or {}
    rows: list[list[Any]] = [
        ["job_id", spec.job_id],
        ["report_generated_at", datetime.now().isoformat(timespec="seconds")],
        ["status", result.status],
        ["total_count", result.count if result.count is not None else snap.count],
        ["counting_events", len(snap.events)],
        ["video_file", spec.original_filename],
        ["video_resolution", f"{v.width}x{v.height}"],
        ["video_fps", round(v.fps, 3)],
        ["video_duration_s", round(v.duration_s, 3)],
        ["frames_processed", result.frames_processed],
        ["frames_expected", result.expected_frames],
        ["model_weights", spec.model.weights.name],
        ["model_is_generic_coco", bool(mi.looks_generic_coco) if mi else ""],
        ["device", mi.device.label if mi else ""],
        ["confidence_threshold", spec.model.confidence],
        ["iou_threshold", spec.model.iou],
        ["classes_counted", classes],
        ["tracker_config", spec.tracker_config.name],
        ["count_direction", spec.counting.direction],
        ["anchor_point", spec.counting.anchor],
        ["line_start_px", f"{p1[0]};{p1[1]}"],
        ["line_end_px", f"{p2[0]};{p2[1]}"],
        ["line_normalised", f"{ln.x1:.4f};{ln.y1:.4f};{ln.x2:.4f};{ln.y2:.4f}"],
        ["processing_time_s", round(result.elapsed_s, 2)],
        ["average_processing_fps", round(result.average_fps, 2)],
        ["distinct_track_ids_not_a_count", result.unique_track_ids],
    ]
    rows += [[f"diagnostic_{k}", val] for k, val in stats.items()]
    rows += [
        ["warnings", " | ".join(snap.warnings + [w for w in result.warnings if w not in snap.warnings])],
        ["error", result.error or ""],
    ]
    return rows


def event_rows(spec: JobSpec, snap: JobSnapshot) -> list[list[Any]]:
    fps = spec.video.fps or 0.0
    rows = []
    for e in snap.events:
        t = e.crossing_frame / fps if fps else 0.0
        rows.append([
            e.count, e.track_id, e.crossing_frame + 1, round(t, 3), fmt_timestamp(t),
            e.confirmed_frame + 1, e.direction, round(float(e.position[0]), 1), round(float(e.position[1]), 1),
            e.inherited_from if e.inherited_from is not None else "",
        ])
    return rows


def write_job_reports(spec: JobSpec, snap: JobSnapshot, result: ProcessingResult) -> dict[str, Path]:
    """Write summary/events CSVs and job metadata. Frame numbers in CSVs are 1-based."""
    out = spec.output_dir
    out.mkdir(parents=True, exist_ok=True)
    summary = _write_csv(out / SUMMARY_FILE.format(job_id=spec.job_id), ["field", "value"],
                         summary_rows(spec, snap, result))
    events = _write_csv(out / EVENTS_FILE.format(job_id=spec.job_id), EVENT_COLUMNS, event_rows(spec, snap))
    meta = {
        "job_id": spec.job_id,
        "created_at": datetime.fromtimestamp(snap.started_at).isoformat(timespec="seconds")
        if snap.started_at else datetime.now().isoformat(timespec="seconds"),
        "video": spec.original_filename,
        "status": result.status,
        "count": result.count if result.count is not None else snap.count,
        "model": spec.model.weights.name,
        "generic_model": bool(snap.model_info.looks_generic_coco) if snap.model_info else None,
        "direction": spec.counting.direction,
        "frames_processed": result.frames_processed,
    }
    (out / METADATA_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"summary_csv": summary, "events_csv": events}


def list_recent_jobs(outputs_dir: Path, limit: int = 10) -> list[dict[str, Any]]:
    """Read job metadata from ``outputs_dir/*/job.json``, newest first. Unreadable entries are skipped."""
    outputs_dir = Path(outputs_dir)
    if not outputs_dir.is_dir():
        return []
    jobs = []
    for d in outputs_dir.iterdir():
        if not d.is_dir() or d.name.startswith(".") or d.is_symlink():
            continue
        try:
            meta = json.loads((d / METADATA_FILE).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(meta, dict):
            continue
        jobs.append(meta)
    jobs.sort(key=lambda m: str(m.get("created_at", "")), reverse=True)
    return [{
        "Date": str(m.get("created_at", "")).replace("T", " "),
        "Video": str(m.get("video", "")),
        "Count": m.get("count"),
        "Status": str(m.get("status", "")),
        "Model": str(m.get("model", "")) + (" (generic)" if m.get("generic_model") else ""),
        "Job": str(m.get("job_id", "")),
    } for m in jobs[:limit]]
