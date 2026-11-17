"""Native waypoint decoding and command diagnostics."""

from __future__ import annotations

import numpy as np

from .odometry import (
    action_for_local_target,
    advance_odometry,
    rebase_targets_xz,
)


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


def command_metrics(action: dict, current: np.ndarray, target: np.ndarray) -> dict:
    next_pose = advance_odometry(current, action)
    return {
        "translation_m": float(action["action_args"]["r"]),
        "rotation_deg": float(np.rad2deg(action["action_args"]["theta"])),
        "remaining_target_distance_m": float(np.linalg.norm(target - current[:2])),
        "command_target_error_m": float(np.linalg.norm(target - next_pose[:2])),
    }


def absolute_target_check() -> None:
    pose = np.asarray([0.1, -0.2, 0.7])
    target = np.asarray([0.4, -0.8])
    action = action_for_local_target(rebase_targets_xz(target[None], pose)[0])
    assert command_metrics(action, pose, target)["command_target_error_m"] < 1e-12
    assert command_metrics(action, pose, target)["translation_m"] < np.linalg.norm(
        target
    )
