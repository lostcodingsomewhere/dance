"""
Binary serialization helpers for numpy arrays stored in the database.

Two distinct codecs are exposed:

* :func:`encode_curve` / :func:`decode_curve` — gzip-compressed float32 arrays
  for time-series curves like RMS energy. Curves are highly repetitive so
  gzip pays off.
* :func:`encode_embedding` / :func:`decode_embedding` — raw float32 bytes for
  high-dimensional embeddings (e.g. CLAP). Embedding values are essentially
  noise-like floats, so compression doesn't help and just costs CPU.
"""

from __future__ import annotations

import gzip

import numpy as np


# ---------------------------------------------------------------------------
# RMS / energy curves
# ---------------------------------------------------------------------------


def encode_curve(arr: np.ndarray) -> bytes:
    """Encode a 1-D float32 numpy array as gzipped raw bytes."""

    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    arr = np.ascontiguousarray(arr)
    return gzip.compress(arr.tobytes())


def decode_curve(blob: bytes, expected_length: int | None = None) -> np.ndarray:
    """Decode bytes produced by :func:`encode_curve` back into a float32 array.

    If ``expected_length`` is provided, raise ``ValueError`` when the decoded
    array doesn't match — useful as a sanity check against torn writes.
    """

    raw = gzip.decompress(blob)
    arr = np.frombuffer(raw, dtype=np.float32)
    if expected_length is not None and arr.shape[0] != expected_length:
        raise ValueError(
            f"decoded curve length {arr.shape[0]} does not match expected "
            f"{expected_length}"
        )
    # Return a writable copy so callers can mutate freely.
    return arr.copy()


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


def encode_embedding(arr: np.ndarray) -> bytes:
    """Encode a 1-D float32 numpy array as raw bytes (no compression)."""

    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    arr = np.ascontiguousarray(arr)
    return arr.tobytes()


def decode_embedding(blob: bytes, dim: int) -> np.ndarray:
    """Decode raw bytes produced by :func:`encode_embedding`.

    ``dim`` is required because we don't store any framing — the caller is
    expected to know the embedding dimension (it's stored on the row).
    """

    arr = np.frombuffer(blob, dtype=np.float32)
    if arr.shape[0] != dim:
        raise ValueError(
            f"decoded embedding length {arr.shape[0]} does not match dim {dim}"
        )
    return arr.copy()
