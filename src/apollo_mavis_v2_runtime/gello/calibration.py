"""GELLO calibration (16-gello §4 "Calibration", D10): the file and the math.

``var/gello_calibration.json`` (``GelloConfig.calibration_path``) holds the per-joint
offsets and the two gripper endpoints written by the session-less
``POST /api/gello/calibrate`` ops:

* ``match_arm`` — ``offset_j = round((raw_j - sign_j * q_arm_j) / (pi/2)) * pi/2`` from the
  leader's raw joints and the Manipulation Arm's CURRENT joints (the GELLO convention: the
  leader's servo horns are mounted at multiples of a quarter turn from the arm's zero, so
  the offset is snapped to the nearest one and a slightly mis-posed calibration still lands
  on the right multiple) — :func:`match_arm_offsets`;
* ``gripper_open`` / ``gripper_closed`` — the raw gripper reading at the two endpoints;
  ``gripper_frac = clip((raw - closed) / (open - closed), 0, 1)`` — :func:`gripper_frac`;
* ``clear`` — deletes the file.

``joint_signs`` are operator-owned config and never stored here. A ``joint_offsets_rad``
config value overrides the file (the reader decides; this module only stores). Pure and
thread-free: the reader and the REST op call it from their own threads, writes are
atomic (temp file + ``os.replace``).
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

QUARTER_TURN_RAD = math.pi / 2.0  # the GELLO offset quantum


@dataclass(frozen=True)
class GelloCalibration:
    """What the file holds. ``joint_offsets_rad`` None = uncalibrated joints; either
    gripper endpoint None = no gripper fraction yet. ``saved_at`` is the ISO-8601 UTC
    stamp of the last write (None for the empty / missing file)."""

    joint_offsets_rad: tuple[float, ...] | None = None
    gripper_open_rad: float | None = None
    gripper_closed_rad: float | None = None
    saved_at: str | None = None

    @property
    def calibrated(self) -> bool:
        return self.joint_offsets_rad is not None

    @property
    def gripper_calibrated(self) -> bool:
        return (
            self.gripper_open_rad is not None
            and self.gripper_closed_rad is not None
            and self.gripper_open_rad != self.gripper_closed_rad
        )

    def to_json(self) -> dict:
        return {
            "joint_offsets_rad": (
                [float(x) for x in self.joint_offsets_rad]
                if self.joint_offsets_rad is not None
                else None
            ),
            "gripper_open_rad": self.gripper_open_rad,
            "gripper_closed_rad": self.gripper_closed_rad,
            "saved_at": self.saved_at,
        }

    @classmethod
    def from_json(cls, data: object) -> GelloCalibration:
        if not isinstance(data, dict):
            raise ValueError("calibration file must hold a JSON object")
        offsets = data.get("joint_offsets_rad")
        if offsets is not None:
            if not isinstance(offsets, list) or len(offsets) != 7:
                raise ValueError("joint_offsets_rad must be a list of 7 floats")
            offsets = tuple(float(x) for x in offsets)
            if not all(math.isfinite(x) for x in offsets):
                raise ValueError("joint_offsets_rad must be finite")

        def _opt_float(key: str) -> float | None:
            v = data.get(key)
            if v is None:
                return None
            f = float(v)
            if not math.isfinite(f):
                raise ValueError(f"{key} must be finite")
            return f

        saved = data.get("saved_at")
        return cls(
            joint_offsets_rad=offsets,
            gripper_open_rad=_opt_float("gripper_open_rad"),
            gripper_closed_rad=_opt_float("gripper_closed_rad"),
            saved_at=str(saved) if saved is not None else None,
        )


EMPTY_CALIBRATION = GelloCalibration()


class GelloCalibrationStore:
    """``var/gello_calibration.json`` (16-gello §4). :meth:`load` returns the empty
    calibration for a missing file and — with a WARNING, kept in ``last_error`` — for an
    unreadable one (a corrupt file must never take the runtime down; the operator sees
    ``calibrated: false`` and re-runs ``match_arm``). :meth:`save` / :meth:`update` write
    atomically and stamp ``saved_at``; :meth:`clear` deletes the file (idempotent)."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self.last_error: str = ""

    @property
    def path(self) -> Path:
        return self._path

    def exists(self) -> bool:
        return self._path.is_file()

    def load(self) -> GelloCalibration:
        try:
            text = self._path.read_text(encoding="utf-8")
        except FileNotFoundError:
            self.last_error = ""
            return EMPTY_CALIBRATION
        except OSError as e:
            self.last_error = f"cannot read {self._path}: {e}"
            logger.warning("gello calibration: %s", self.last_error)
            return EMPTY_CALIBRATION
        try:
            cal = GelloCalibration.from_json(json.loads(text))
        except (ValueError, TypeError) as e:
            self.last_error = f"{self._path} is unreadable ({e}); treating as uncalibrated"
            logger.warning("gello calibration: %s", self.last_error)
            return EMPTY_CALIBRATION
        self.last_error = ""
        return cal

    def save(self, cal: GelloCalibration) -> GelloCalibration:
        stamped = replace(cal, saved_at=datetime.now(UTC).isoformat(timespec="seconds"))
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_name(self._path.name + ".tmp")
        tmp.write_text(json.dumps(stamped.to_json(), indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, self._path)
        self.last_error = ""
        return stamped

    def update(self, **fields) -> GelloCalibration:
        """Load, replace the given fields and save (one REST op = one field group)."""
        return self.save(replace(self.load(), **fields))

    def clear(self) -> bool:
        """Delete the file; True when one existed."""
        try:
            self._path.unlink()
        except FileNotFoundError:
            return False
        self.last_error = ""
        return True


def match_arm_offsets(raw, q_arm, signs) -> np.ndarray:
    """``offset_j = round((raw_j - sign_j * q_arm_j) / (pi/2)) * pi/2`` (16-gello §4).

    ``raw`` = the leader's raw joints (rad, 7), ``q_arm`` = the Manipulation Arm's current
    joints (rad, 7), ``signs`` = ``joint_signs``. With this offset the mapped reading
    ``sign * (raw - offset)`` equals ``q_arm`` up to the residual below a quarter turn —
    i.e. up to how well the operator posed GELLO like the arm.
    """
    raw = np.asarray(raw, dtype=np.float64).reshape(7)
    q_arm = np.asarray(q_arm, dtype=np.float64).reshape(7)
    signs = np.asarray(signs, dtype=np.float64).reshape(7)
    if not (np.all(np.isfinite(raw)) and np.all(np.isfinite(q_arm))):
        raise ValueError("match_arm needs finite leader and arm joints")
    return np.round((raw - signs * q_arm) / QUARTER_TURN_RAD) * QUARTER_TURN_RAD


def gripper_frac(
    raw: float | None, open_rad: float | None, closed_rad: float | None
) -> float | None:
    """``clip((raw - closed) / (open - closed), 0, 1)``; None until both endpoints exist
    (and differ) or without a gripper reading."""
    if raw is None or open_rad is None or closed_rad is None or open_rad == closed_rad:
        return None
    return float(np.clip((float(raw) - closed_rad) / (open_rad - closed_rad), 0.0, 1.0))


__all__ = [
    "QUARTER_TURN_RAD",
    "EMPTY_CALIBRATION",
    "GelloCalibration",
    "GelloCalibrationStore",
    "match_arm_offsets",
    "gripper_frac",
]
