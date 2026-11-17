"""Use local visited poses to propose one native movement before STOP."""

from copy import deepcopy

import numpy as np

from .history import native_history_from_full
from .session import _env_action, pose


def native_w1_bound(norm, scale):
    lo, hi = (
        np.asarray(norm["min"], dtype=float)[0, :2],
        np.asarray(norm["max"], dtype=float)[0, :2],
    )
    if not np.isfinite([*lo, *hi, scale]).all() or np.any(lo > hi) or scale <= 0:
        raise ValueError("invalid frozen native normalization")
    return float(scale * np.linalg.norm(np.maximum(np.abs(lo), np.abs(hi))))


def motor_step_proposal(visited, current, bound, remaining, step_size):
    from scipy.spatial.transform import Rotation

    visited = np.asarray(visited, dtype=float)
    current = np.asarray(current, dtype=float)
    if (
        current.shape != (7,)
        or visited.ndim != 2
        or visited.shape[1] != 3
        or not len(visited)
    ):
        raise ValueError("Invalid visited positions or current pose")
    if not np.array_equal(visited[-1], current[:3]):
        raise ValueError("Latest position differs from current pose")
    if not np.isfinite(current).all() or not np.isfinite(bound) or bound <= 0:
        raise ValueError("Invalid pose or native bound")
    if type(remaining) is not int or remaining < 1:
        raise ValueError("Invalid action budget")
    if (
        isinstance(step_size, (bool, np.bool_))
        or not np.isfinite(step_size)
        or step_size <= 0
    ):
        raise ValueError("Invalid native motor step")
    # Each native history image votes once. The three current views share a pose.
    indices = native_history_from_full(list(range(len(visited))), 20)
    positions = np.vstack((visited[indices], [current[:3]] * 3))
    if not np.isfinite(positions).all():
        raise ValueError("Invalid history positions")
    if remaining < 2:
        return {"eligible": False, "action": None}
    displacements = positions - current[:3]
    weights = (np.linalg.norm(displacements, axis=1) <= 5 * bound).astype(float)
    offset = (displacements * weights[:, None]).sum(axis=0) / weights.sum()
    center = current[:3] + offset
    delta = Rotation.from_quat(current[3:]).inv().apply(offset)
    radius = float(np.linalg.norm(delta[[0, 2]]))
    eligible = radius > 0
    return {
        "eligible": eligible,
        "center": center.tolist(),
        "image_weights": weights.tolist(),
        "local_image_count": int(weights.sum()),
        "radius_unclamped": radius,
        "action": {
            "action": "GO_TOWARD_POINT",
            "action_args": {
                "r": min(radius, float(bound), float(step_size)),
                "theta": float(np.arctan2(-delta[0], -delta[2])),
            },
        }
        if eligible
        else None,
    }


class TerminalStep:
    def __init__(self, env, contact, bound, step_size):
        self.env = env
        self.contact = contact
        self.bound = bound
        self.step_size = step_size
        self.visited = []
        self.execution = []
        self.proposal = None

    def execute(self, action, source):

        before = pose(self.env).copy()
        result = self.contact.step(action)
        after = pose(self.env).copy()
        receipt = getattr(self.contact, "records", [])
        self.execution.append(
            {
                "action": deepcopy(action),
                "source": source,
                "pose_before": before.tolist(),
                "pose_after": after.tolist(),
                "contact": deepcopy(receipt[-1]) if receipt else None,
            }
        )
        return result

    def step(self, action):

        if not self.visited:
            self.visited.append(pose(self.env)[:3].copy())
        if action["action"].lower() == "stop":
            self.proposal = motor_step_proposal(
                self.visited,
                pose(self.env),
                self.bound,
                500 - len(self.execution),
                self.step_size,
            )
            if self.proposal["eligible"]:
                projected = self.contact.preview(self.proposal["action"])
                if projected is None:
                    raise RuntimeError("unsupported terminal controller")
                command = np.asarray(projected["command_delta"])
                actual = np.asarray(projected["delta"])
                self.proposal["motor_valid"] = bool(
                    command @ command > 0 and command @ actual > 0
                )
                if self.proposal["motor_valid"]:
                    self.execute(_env_action(self.proposal["action"]), "terminal")
            return self.execute(action, "original_stop")
        result = self.execute(action, "main")
        self.visited.append(pose(self.env)[:3].copy())
        return result
