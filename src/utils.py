"""Shared helpers: project paths, safe filenames, path sandboxing, device selection."""

from __future__ import annotations

import re
import secrets
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")
_WINDOWS_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


class UnsafePathError(ValueError):
    """Raised when a path would escape its permitted base directory."""


def safe_stem(name: str, max_len: int = 60) -> str:
    """Reduce an arbitrary (user-supplied) filename to a safe stem.

    Directory components are discarded, unsafe characters replaced, and
    Windows reserved device names avoided. Never returns an empty string.
    """
    base = re.split(r"[\\/]", name or "")[-1]
    stem = base.rsplit(".", 1)[0] if "." in base.strip(".") else base
    stem = _UNSAFE_CHARS.sub("_", stem).strip("._-")[:max_len]
    if not stem or stem.upper() in _WINDOWS_RESERVED:
        stem = f"video_{stem}" if stem else "video"
    return stem


def generate_job_id(original_name: str) -> str:
    """Unique, filesystem-safe job identifier: <timestamp>_<stem>_<random>."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{safe_stem(original_name, 40)}_{secrets.token_hex(3)}"


def resolve_within(base: Path, candidate: Path | str) -> Path:
    """Resolve ``candidate`` and ensure it lies inside ``base``.

    Relative candidates are interpreted relative to ``base``. Raises
    UnsafePathError on traversal (``..``), absolute paths elsewhere, or
    symlinks that point outside ``base``.
    """
    base_resolved = Path(base).resolve()
    cand = Path(candidate)
    target = (cand if cand.is_absolute() else base_resolved / cand).resolve()
    if target != base_resolved and base_resolved not in target.parents:
        raise UnsafePathError("Path is outside the permitted directory.")
    return target


def display_path(path: Path) -> str:
    """Path relative to the project root for display, avoiding full local paths."""
    try:
        return Path(path).resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return Path(path).name


@dataclass(frozen=True)
class DeviceInfo:
    device: str           # Value to pass to Ultralytics ("cpu", "cuda:0", ...)
    label: str            # Human-readable description.
    reason: str = ""      # Why a fallback occurred, if any.


def select_device(requested: str = "auto") -> DeviceInfo:
    """Choose a compute device, verifying CUDA actually works before using it.

    ``torch.cuda.is_available()`` can be True on GPUs the installed PyTorch
    build no longer supports; a small test kernel catches that case so we
    fall back to CPU instead of failing mid-video.
    """
    requested = (requested or "auto").lower()
    if requested == "cpu":
        return DeviceInfo("cpu", "CPU", "")

    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dependency
        return DeviceInfo("cpu", "CPU", "PyTorch is not installed.")

    if not torch.cuda.is_available():
        reason = (
            "Installed PyTorch build has no CUDA support."
            if torch.version.cuda is None
            else "No usable CUDA GPU detected."
        )
        return DeviceInfo("cpu", "CPU", reason)

    device = "cuda:0" if requested in ("auto", "cuda") else requested
    try:
        idx = int(device.split(":")[1])
        name = torch.cuda.get_device_name(idx)
        (torch.ones(8, device=device) * 2).sum().item()  # verify kernels run
        return DeviceInfo(device, f"GPU: {name}", "")
    except Exception as exc:  # noqa: BLE001 - any CUDA failure means fallback
        return DeviceInfo("cpu", "CPU", f"CUDA present but unusable ({type(exc).__name__}).")
