"""Pure pyarrow encode / decode of every dora payload shape (14-dora §3).

No ``dora`` import here — this module is unit-tested without a daemon. Shapes:

- images: flat ``UInt8[H*W*3]`` (rgb8) / ``UInt16[H*W]`` (mono16 depth) with
  ``width`` / ``height`` / ``encoding`` metadata (dora-hub convention);
- vectors: flat ``Float32`` / ``Float64``;
- actions: flat ``Float32[K*D]`` row-major (K chunk rows x D dims);
- structured data: one ``Utf8`` scalar holding the JSON of a core model.

Metadata values may only be ``bool`` / ``int`` / ``float`` / ``str`` /
``list[int|float|str]`` — dora silently stringifies anything else (dicts, numpy
scalars, ``None``) — so :func:`clean_metadata` normalises numpy scalars and
drops ``None`` values before a dict is handed to ``send_output``.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pyarrow as pa
from pydantic import BaseModel

ENCODING_RGB8 = "rgb8"
ENCODING_MONO16 = "mono16"


# -- metadata hygiene ------------------------------------------------------------------------
def _scalar(v: Any) -> Any:
    if isinstance(v, np.generic):
        return v.item()
    return v


def clean_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """Make ``meta`` safe for ``send_output``: numpy scalars -> python, arrays ->
    lists of python numbers, ``None`` dropped, nested dicts JSON-encoded (they are
    not representable), everything else passed through."""
    out: dict[str, Any] = {}
    for k, v in meta.items():
        if v is None:
            continue
        if isinstance(v, np.ndarray):
            out[k] = [_scalar(x) for x in v.reshape(-1).tolist()]
        elif isinstance(v, (list, tuple)):
            out[k] = [_scalar(x) for x in v]
        elif isinstance(v, dict):
            out[k] = json.dumps(v, sort_keys=True)
        else:
            out[k] = _scalar(v)
    return out


# -- images ----------------------------------------------------------------------------------
def _flat_zero_copy(arr: np.ndarray, dtype: pa.DataType) -> pa.Array:
    """Flat Arrow view over a contiguous ndarray (no copy: dora copies once into SHM).
    The caller keeps the frame alive until ``send_output`` returns (the bus thread does)."""
    flat = arr.reshape(-1)
    return pa.Array.from_buffers(dtype, flat.shape[0], [None, pa.py_buffer(flat)])


def encode_rgb8(rgb: np.ndarray) -> tuple[pa.Array, dict[str, Any]]:
    """(H, W, 3) uint8 -> flat UInt8 + {width, height, encoding, primitive}."""
    arr = np.ascontiguousarray(rgb)
    if arr.ndim != 3 or arr.shape[2] != 3 or arr.dtype != np.uint8:
        raise ValueError(f"rgb8 image must be (H, W, 3) uint8, got {arr.shape} {arr.dtype}")
    h, w = arr.shape[:2]
    return _flat_zero_copy(arr, pa.uint8()), {
        "width": int(w),
        "height": int(h),
        "encoding": ENCODING_RGB8,
        "primitive": "image",
    }


def encode_mono16(depth: np.ndarray, depth_scale_m: float = 0.001) -> tuple[pa.Array, dict]:
    """(H, W) uint16 -> flat UInt16 + {width, height, encoding, depth_scale_m, aligned_to}."""
    arr = np.ascontiguousarray(depth)
    if arr.ndim != 2 or arr.dtype != np.uint16:
        raise ValueError(f"mono16 image must be (H, W) uint16, got {arr.shape} {arr.dtype}")
    h, w = arr.shape
    return _flat_zero_copy(arr, pa.uint16()), {
        "width": int(w),
        "height": int(h),
        "encoding": ENCODING_MONO16,
        "primitive": "image",
        "depth_scale_m": float(depth_scale_m),
        "aligned_to": "color",
    }


def decode_image(value: pa.Array, meta: dict[str, Any]) -> np.ndarray:
    """Flat array + metadata -> (H, W, 3) uint8 or (H, W) uint16 view (zero-copy
    when the buffer allows it)."""
    h, w = int(meta["height"]), int(meta["width"])
    enc = str(meta.get("encoding", ENCODING_RGB8))
    flat = value.to_numpy(zero_copy_only=False)
    if enc == ENCODING_RGB8:
        return flat.reshape(h, w, 3)
    if enc == ENCODING_MONO16:
        return flat.reshape(h, w)
    raise ValueError(f"unknown image encoding {enc!r}")


# -- vectors -----------------------------------------------------------------------------------
def encode_f64(vec: np.ndarray | list[float]) -> pa.Array:
    return pa.array(np.asarray(vec, dtype=np.float64).reshape(-1), type=pa.float64())


def encode_f32(vec: np.ndarray | list[float]) -> pa.Array:
    return pa.array(np.asarray(vec, dtype=np.float32).reshape(-1), type=pa.float32())


def encode_i64(values: list[int] | int) -> pa.Array:
    if isinstance(values, int):
        values = [values]
    return pa.array([int(v) for v in values], type=pa.int64())


def decode_vector(value: pa.Array, dtype: Any = np.float32) -> np.ndarray:
    arr = value.to_numpy(zero_copy_only=False)
    return np.asarray(arr, dtype=dtype).reshape(-1)


# -- actions (14-dora §5): Float32[K*D] row-major -----------------------------------------------
def encode_action(actions: np.ndarray) -> tuple[pa.Array, dict[str, Any]]:
    """(D,) or (K, D) float32 -> flat Float32[K*D] + {chunk_len, action_dim, finite}."""
    arr = np.asarray(actions, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] == 0:
        raise ValueError(f"actions must be (D,) or (K, D), got {arr.shape}")
    k, d = arr.shape
    return pa.array(np.ascontiguousarray(arr).reshape(-1), type=pa.float32()), {
        "chunk_len": int(k),
        "action_dim": int(d),
        "finite": bool(np.all(np.isfinite(arr))),
    }


def decode_action(value: pa.Array, meta: dict[str, Any]) -> np.ndarray:
    """Flat Float32 + {chunk_len, action_dim} -> (K, D) float32; raises ValueError on
    a shape mismatch (the caller drops the message and counts it)."""
    k = int(meta["chunk_len"])
    d = int(meta["action_dim"])
    if k < 1 or d < 1:
        raise ValueError(f"chunk_len / action_dim must be >= 1, got {k} / {d}")
    flat = decode_vector(value, np.float32)
    if flat.shape[0] != k * d:
        raise ValueError(
            f"payload has {flat.shape[0]} values, expected chunk_len*action_dim={k * d}"
        )
    return flat.reshape(k, d)


# -- JSON models ---------------------------------------------------------------------------------
def warm_up() -> None:
    """Exercise every encoder once on tiny inputs. pyarrow / numpy set-up costs land on the
    FIRST call of each path (the first real ``mic_*`` build measured 312 ms, 2026-09-08) and
    the bus thread holds the GIL meanwhile - a stall the 100 Hz control thread would eat if a
    session were running; the bridge pays it here, before it reports ``attached``."""
    encode_rgb8(np.zeros((2, 2, 3), dtype=np.uint8))
    encode_mono16(np.zeros((2, 2), dtype=np.uint16))
    encode_f64(np.zeros(4))
    encode_f32(np.zeros(4, dtype=np.float32))
    encode_i64([1, 2])
    encode_action(np.zeros((1, 6)))
    encode_json({"warm": True})
    pa.array(np.zeros(4, dtype=np.float32)).to_numpy()
    pa.array([0.0]).to_numpy()


def encode_json(model: BaseModel | dict[str, Any] | str) -> pa.Array:
    """One Utf8 scalar holding the JSON text of a pydantic model / dict / str."""
    if isinstance(model, BaseModel):
        text = model.model_dump_json()
    elif isinstance(model, dict):
        text = json.dumps(model, sort_keys=True)
    else:
        text = str(model)
    return pa.array([text], type=pa.string())


def decode_json_text(value: Any) -> str:
    """The Utf8 scalar (or a bytes payload) of a JSON message -> text."""
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8")
    if isinstance(value, str):
        return value
    if isinstance(value, pa.Array):
        if len(value) == 0:
            raise ValueError("empty JSON payload")
        if pa.types.is_string(value.type) or pa.types.is_large_string(value.type):
            return str(value[0].as_py())
        if pa.types.is_binary(value.type) or pa.types.is_large_binary(value.type):
            return bytes(value[0].as_py()).decode("utf-8")
        if pa.types.is_uint8(value.type):  # send_output_raw bytes land as UInt8
            return bytes(value.to_numpy(zero_copy_only=False).tobytes()).decode("utf-8")
    raise ValueError(f"not a JSON payload: {type(value).__name__} {getattr(value, 'type', '')}")


def decode_json(value: Any, model: type[BaseModel]) -> BaseModel:
    return model.model_validate_json(decode_json_text(value))


__all__ = [
    "ENCODING_MONO16",
    "ENCODING_RGB8",
    "clean_metadata",
    "decode_action",
    "decode_image",
    "decode_json",
    "decode_json_text",
    "decode_vector",
    "encode_action",
    "encode_f32",
    "encode_f64",
    "encode_i64",
    "encode_json",
    "encode_mono16",
    "encode_rgb8",
]
