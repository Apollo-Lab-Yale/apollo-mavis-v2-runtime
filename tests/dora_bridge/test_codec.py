"""codec round trips (14-dora §3, §10 tier 1): pyarrow only, no daemon."""

from __future__ import annotations

import json

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")

from apollo_mavis_v2_core.protocol.external import PolicySpecAnnounce, PolicySpecModel  # noqa: E402

from apollo_mavis_v2_runtime.dora_bridge import codec  # noqa: E402


def test_rgb8_round_trip_is_flat_uint8_with_shape_metadata():
    rgb = (np.arange(4 * 6 * 3) % 251).astype(np.uint8).reshape(4, 6, 3)
    arr, meta = codec.encode_rgb8(rgb)
    assert arr.type == pa.uint8() and len(arr) == 4 * 6 * 3
    assert meta == {"width": 6, "height": 4, "encoding": "rgb8", "primitive": "image"}
    back = codec.decode_image(arr, meta)
    assert back.shape == (4, 6, 3) and np.array_equal(back, rgb)
    with pytest.raises(ValueError):
        codec.encode_rgb8(np.zeros((4, 6), dtype=np.uint8))


def test_mono16_depth_round_trip():
    depth = np.full((3, 5), 1234, dtype=np.uint16)
    depth[0, 0] = 65535
    arr, meta = codec.encode_mono16(depth, 0.001)
    assert arr.type == pa.uint16()
    assert meta["encoding"] == "mono16" and meta["depth_scale_m"] == 0.001
    assert meta["aligned_to"] == "color"
    assert np.array_equal(codec.decode_image(arr, meta), depth)
    with pytest.raises(ValueError):
        codec.encode_mono16(depth.astype(np.float32))


def test_vectors_and_actions():
    v = np.linspace(-1, 1, 32)
    assert codec.decode_vector(codec.encode_f64(v), np.float64).tolist() == v.tolist()
    f = codec.decode_vector(codec.encode_f32(v), np.float32)
    assert f.dtype == np.float32 and f.shape == (32,)
    assert codec.encode_i64(7).to_pylist() == [7]
    rows = np.arange(24, dtype=np.float32).reshape(3, 8)
    arr, meta = codec.encode_action(rows)
    assert meta == {"chunk_len": 3, "action_dim": 8, "finite": True}
    assert np.array_equal(codec.decode_action(arr, meta), rows)  # row-major K x D
    one, m1 = codec.encode_action(np.ones(8, dtype=np.float32))
    assert m1["chunk_len"] == 1 and codec.decode_action(one, m1).shape == (1, 8)
    nan_rows = rows.copy()
    nan_rows[1, 2] = np.nan
    _, mn = codec.encode_action(nan_rows)
    assert mn["finite"] is False  # NaN rows PASS THROUGH; the executor's 3-strike is the guard
    with pytest.raises(ValueError):
        codec.decode_action(arr, {"chunk_len": 2, "action_dim": 8})  # 16 != 24 values
    with pytest.raises(ValueError):
        codec.decode_action(arr, {"chunk_len": 0, "action_dim": 8})


def test_json_models_ride_one_utf8_scalar():
    ann = PolicySpecAnnounce(
        policy_id="p",
        policy_version=3,
        node_version="0.1",
        rate_hz=15.0,
        spec=PolicySpecModel(
            action_space="delta_ee",
            action_frame="arm_base:grip",
            action_names=["grip_ee.dx"],
            state_names=["grip_joint1.pos"],
        ),
    )
    arr = codec.encode_json(ann)
    assert arr.type == pa.string() and len(arr) == 1
    back = codec.decode_json(arr, PolicySpecAnnounce)
    assert back == ann
    assert json.loads(codec.decode_json_text(codec.encode_json({"b": 1, "a": 2})))["a"] == 2
    assert codec.decode_json_text(codec.encode_json("plain")) == "plain"
    assert codec.decode_json_text(b'{"x": 1}') == '{"x": 1}'
    raw = pa.array(np.frombuffer(b'{"x": 2}', dtype=np.uint8), type=pa.uint8())
    assert json.loads(codec.decode_json_text(raw))["x"] == 2
    with pytest.raises(ValueError):
        codec.decode_json_text(pa.array([1.0]))


def test_clean_metadata_only_emits_dora_safe_scalar_types():
    meta = codec.clean_metadata(
        {
            "i": np.int64(3),
            "f": np.float32(1.5),
            "arr": np.arange(3),
            "l": (np.int32(1), 2.0),
            "none": None,
            "nested": {"a": 1},
            "s": "x",
            "b": True,
        }
    )
    assert meta == {
        "i": 3,
        "f": 1.5,
        "arr": [0, 1, 2],
        "l": [1, 2.0],
        "nested": '{"a": 1}',
        "s": "x",
        "b": True,
    }
    assert all(isinstance(v, (bool, int, float, str, list)) for v in meta.values())
    assert isinstance(meta["i"], int) and not isinstance(meta["i"], np.generic)
