"""Process-wide GPU resource gate.

Both the parallel dispatcher (audio embeds + Demucs separates) and the
text-search endpoint (CLAP text encode) share one MPS queue. Without
coordination they fight each other and small fast operations (a text
encode) starve behind big slow ones (Demucs chunks).

This module exposes a single ``threading.Semaphore`` so every GPU-bound
caller acquires politely. Default is 1 (serialize all GPU work). Set
``DANCE_GPU_CONCURRENCY=2`` (or higher) to allow that many concurrent
GPU operations — on M2 Pro 16 GB, 2 is roughly the safe ceiling.
"""

from __future__ import annotations

import os
import threading


_DEFAULT_GPU_CONCURRENCY = int(os.environ.get("DANCE_GPU_CONCURRENCY", "1"))
GPU_SEMAPHORE = threading.Semaphore(_DEFAULT_GPU_CONCURRENCY)


__all__ = ["GPU_SEMAPHORE"]
