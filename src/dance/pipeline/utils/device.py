"""Device selection for torch-backed stages.

A tiny shared helper so Demucs (separate) and CLAP (embed) — plus any future
torch-based stage — pick the best available device the same way.
"""

from __future__ import annotations


def pick_device(preferred: str = "auto") -> str:
    """Return the best available torch device.

    Args:
        preferred: One of ``"auto"``, ``"cuda"``, ``"mps"``, ``"cpu"``.
            If anything other than ``"auto"``, returned as-is.

    Returns:
        ``"cuda"`` if available, else ``"mps"`` on Apple Silicon, else ``"cpu"``.
    """
    if preferred != "auto":
        return preferred

    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
