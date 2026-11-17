"""Native waypoint decoding."""

from __future__ import annotations

import numpy as np


def decode(
    prediction: dict,
    slot: int,
    anchor: np.ndarray,
    current: np.ndarray,
    target: np.ndarray,
) -> dict:
    angles = np.asarray(prediction["angle"], dtype=np.float64)
    anchor = np.asarray(anchor, dtype=np.float64)
    current = np.asarray(current, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if (
        angles.shape != (5,)
        or not isinstance(slot, (int, np.integer))
        or isinstance(slot, (bool, np.bool_))
        or not 0 <= slot < len(angles)
        or anchor.shape != (3,)
        or current.shape != (3,)
        or target.shape != (2,)
    ):
        raise ValueError(
            "expected a native waypoint slot, planar poses, and a local xz target"
        )
    if not np.isfinite(anchor).all() or not np.isfinite(current).all():
        raise ValueError("cached command poses must be finite")
    theta = float(angles[slot]) - (current[2] - anchor[2])
    if not np.isfinite(theta) or not np.isfinite(target).all():
        raise ValueError("nonfinite cached command")
    return {
        "action": "GO_TOWARD_POINT",
        "action_args": {
            "r": float(np.linalg.norm(target)),
            "theta": float((theta + np.pi) % (2 * np.pi) - np.pi),
        },
    }
