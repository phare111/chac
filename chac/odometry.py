"""Planar command odometry and waypoint target transforms."""

from __future__ import annotations

import numpy as np

TURN_STEP_DEG = 10.0
MOVE_STEP_M = 0.25


def advance_odometry(estimate: np.ndarray, action: str | dict) -> np.ndarray:
    """Integrate commanded planar motion; simulator pose is audit-only."""
    estimate = np.asarray(estimate, dtype=np.float64)
    if estimate.shape != (3,) or not np.isfinite(estimate).all():
        raise ValueError("command odometry requires a finite planar pose")
    x, z, yaw = map(float, estimate)
    name = str(action if isinstance(action, str) else action.get("action", ""))
    if name == "TURN_LEFT":
        yaw += np.deg2rad(TURN_STEP_DEG)
    elif name == "TURN_RIGHT":
        yaw -= np.deg2rad(TURN_STEP_DEG)
    elif name == "MOVE_FORWARD":
        x -= MOVE_STEP_M * np.sin(yaw)
        z -= MOVE_STEP_M * np.cos(yaw)
    elif name == "GO_TOWARD_POINT":
        args = action.get("action_args", {})
        radius, theta = float(args["r"]), float(args["theta"])
        if not np.isfinite((radius, theta)).all() or radius < 0:
            raise ValueError(
                "GO_TOWARD_POINT odometry requires a finite nonnegative target"
            )
        yaw += theta
        x -= radius * np.sin(yaw)
        z -= radius * np.cos(yaw)
    elif name != "STOP":
        raise ValueError(f"unsupported odometry action: {name}")
    return np.asarray([x, z, (yaw + np.pi) % (2 * np.pi) - np.pi], dtype=np.float64)


def rebase_targets_xz(targets: np.ndarray, receiver_pose: np.ndarray) -> np.ndarray:
    targets = np.asarray(targets, dtype=np.float64)
    receiver = np.asarray(receiver_pose, dtype=np.float64)
    if targets.ndim != 2 or targets.shape[1] != 2 or receiver.shape != (3,):
        raise ValueError("expected world xz targets and planar receiver odometry")
    if not np.isfinite(targets).all() or not np.isfinite(receiver).all():
        raise ValueError("waypoint targets and receiver odometry must be finite")
    dx, dz = targets[:, 0] - receiver[0], targets[:, 1] - receiver[1]
    c, s = np.cos(receiver[2]), np.sin(receiver[2])
    return np.stack((c * dx - s * dz, s * dx + c * dz), axis=-1)
