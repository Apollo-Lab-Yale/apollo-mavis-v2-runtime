"""Per-episode feature statistics and their aggregation, lerobot semantics
(``compute_stats.compute_episode_stats`` / ``aggregate_stats``) in plain numpy.

Torch-free by design: ``lerobot.datasets.__init__`` drags torch in, so neither
the recorder's save path nor the export job may import ``compute_stats``. The
shapes follow lerobot's validator (``_assert_type_and_shape``): every stat is an
``np.ndarray`` with ``ndim >= 1``, ``count`` is ``(1,)``, image stats are
``(3, 1, 1)`` in ``[0, 1]``. Quantiles here are exact (``np.quantile``) where
lerobot's ``RunningQuantileStats`` approximates them from histograms.
"""

from __future__ import annotations

from typing import Any

import numpy as np

QUANTILES: dict[str, float] = {"q01": 0.01, "q10": 0.10, "q50": 0.50, "q90": 0.90, "q99": 0.99}
STAT_KEYS: tuple[str, ...] = ("min", "max", "mean", "std", "count", *QUANTILES)


def feature_stats(values: np.ndarray) -> dict[str, np.ndarray]:
    """Stats of one non-video feature over an episode.

    ``values`` is ``(N, D)`` (vector features) or ``(N,)`` (``shape == (1,)``
    features, which lerobot stores as scalars and reduces with ``keepdims`` so
    every stat is ``(1,)``). Booleans / ints are computed in float64 like
    lerobot's running stats do.
    """
    arr = np.asarray(values)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    arr = arr.astype(np.float64, copy=False)
    n = int(arr.shape[0])
    out: dict[str, np.ndarray] = {
        "min": np.min(arr, axis=0),
        "max": np.max(arr, axis=0),
        "mean": np.mean(arr, axis=0),
        "std": np.std(arr, axis=0),
        "count": np.array([n], dtype=np.int64),
    }
    for key, q in QUANTILES.items():
        out[key] = np.quantile(arr, q, axis=0) if n >= 2 else out["mean"].copy()
    return out


def video_stats_from_encoder(
    raw: dict[str, np.ndarray], channels: int = 3
) -> dict[str, np.ndarray]:
    """lerobot's writer transform for the streaming encoder's ``(C,)`` 0..255 stats:
    every key but ``count`` -> ``squeeze(v.reshape(1, -1, 1, 1) / 255, axis=0)``
    = ``(C, 1, 1)`` in ``[0, 1]``; ``count`` stays ``(1,)``."""
    out: dict[str, np.ndarray] = {}
    for key, value in raw.items():
        v = np.asarray(value)
        if key == "count":
            out[key] = v.reshape(1).astype(np.int64)
        else:
            out[key] = np.squeeze(v.astype(np.float64).reshape(1, -1, 1, 1) / 255.0, axis=0)
            if out[key].shape[0] != channels:  # a (1,) mean etc. -> broadcast per channel
                out[key] = np.broadcast_to(out[key], (channels, 1, 1)).copy()
    return out


def to_jsonable(stats: dict[str, dict[str, np.ndarray]]) -> dict[str, dict[str, Any]]:
    return {
        feature: {k: np.asarray(v).tolist() for k, v in per.items()}
        for feature, per in stats.items()
    }


def from_jsonable(stats: dict[str, dict[str, Any]]) -> dict[str, dict[str, np.ndarray]]:
    """lerobot ``cast_stats_to_numpy``: every leaf -> ``np.atleast_1d(np.array(v))``."""
    return {
        feature: {k: np.atleast_1d(np.asarray(v)) for k, v in per.items()}
        for feature, per in stats.items()
    }


def _validate(stats_list: list[dict[str, dict[str, np.ndarray]]]) -> None:
    for stats in stats_list:
        for feature, per in stats.items():
            for key, value in per.items():
                if not isinstance(value, np.ndarray) or value.ndim == 0:
                    raise ValueError(f"stat {key!r} of {feature!r} must be a >=1-d ndarray")
                if key == "count" and value.shape != (1,):
                    raise ValueError(f"count of {feature!r} must be (1,), is {value.shape}")
                image_shapes = ((3, 1, 1), (1, 1, 1))
                if "image" in feature and key != "count" and value.shape not in image_shapes:
                    raise ValueError(
                        f"image stat {key!r} of {feature!r} must be (3,1,1) or (1,1,1), "
                        f"is {value.shape}"
                    )


def aggregate_feature_stats(stats_ft_list: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """lerobot ``aggregate_feature_stats`` verbatim: count-weighted mean, parallel
    variance, min/max envelope, count-weighted quantiles."""
    means = np.stack([s["mean"] for s in stats_ft_list])
    variances = np.stack([s["std"] ** 2 for s in stats_ft_list])
    counts = np.stack([s["count"] for s in stats_ft_list])
    total_count = counts.sum(axis=0)
    while counts.ndim < means.ndim:
        counts = np.expand_dims(counts, axis=-1)
    total_mean = (means * counts).sum(axis=0) / total_count
    delta_means = means - total_mean
    total_variance = ((variances + delta_means**2) * counts).sum(axis=0) / total_count
    aggregated: dict[str, np.ndarray] = {
        "min": np.min(np.stack([s["min"] for s in stats_ft_list]), axis=0),
        "max": np.max(np.stack([s["max"] for s in stats_ft_list]), axis=0),
        "mean": total_mean,
        "std": np.sqrt(total_variance),
        "count": total_count,
    }
    quantile_keys = [k for k in stats_ft_list[0] if k.startswith("q") and k[1:].isdigit()]
    for q_key in quantile_keys:
        if all(q_key in s for s in stats_ft_list):
            quantile_values = np.stack([s[q_key] for s in stats_ft_list])
            aggregated[q_key] = (quantile_values * counts).sum(axis=0) / total_count
    return aggregated


def aggregate_stats(
    stats_list: list[dict[str, dict[str, np.ndarray]]],
) -> dict[str, dict[str, np.ndarray]]:
    """lerobot ``aggregate_stats``: the union of feature keys, each aggregated."""
    _validate(stats_list)
    keys = {key for stats in stats_list for key in stats}
    return {
        key: aggregate_feature_stats([s[key] for s in stats_list if key in s]) for key in keys
    }


__all__ = [
    "QUANTILES",
    "STAT_KEYS",
    "aggregate_feature_stats",
    "aggregate_stats",
    "feature_stats",
    "from_jsonable",
    "to_jsonable",
    "video_stats_from_encoder",
]
