import numpy as np


def memory_proposal(
    support, positions, current, bound, remaining, require_history_support=False
):
    """Attribution-weighted visited-pose consensus; never receives task labels or GT."""
    from scipy.spatial.transform import Rotation

    positions = np.asarray(positions, dtype=float)
    current = np.asarray(current, dtype=float)
    if (
        current.shape != (7,)
        or positions.shape != (len(support), 3)
        or len(support) < 3
        or not np.isfinite([*positions.ravel(), *current, bound]).all()
        or bound <= 0
    ):
        raise ValueError("invalid visited poses or native bound")
    if isinstance(remaining, bool) or not isinstance(remaining, int) or remaining < 1:
        raise ValueError("invalid action budget")
    if not np.array_equal(positions[-3:], np.tile(current[:3], (3, 1))):
        raise ValueError("current-view poses changed")
    weights = []
    for grid in support:
        grid = np.asarray(grid, dtype=float)
        if grid.ndim != 2 or not grid.size or not np.isfinite(grid).all():
            raise ValueError("invalid image attribution")
        weights.append(float(np.maximum(grid, 0).mean()))
    weights = np.asarray(weights)
    local = np.linalg.norm(positions - current[:3], axis=1) <= 5 * bound
    weights = np.where(local, weights, 0)
    total = float(weights.sum())
    past_mean = float(weights[:-3].mean()) if len(weights) > 3 else 0.0
    current_max = float(weights[-3:].max())
    if require_history_support and past_mean <= current_max:
        return dict(
            eligible=False,
            action=None,
            reason="current_or_insufficient_history_support",
            past_mean=past_mean,
            current_max=current_max,
        )
    if total == 0 or remaining < 2:
        return dict(eligible=False, action=None)
    offset = ((positions - current[:3]) * weights[:, None]).sum(axis=0) / total
    center = current[:3] + offset
    delta = Rotation.from_quat(current[3:]).inv().apply(offset)
    radius = float(np.linalg.norm(delta[[0, 2]]))
    eligible = radius > 0
    return dict(
        eligible=eligible,
        center=center.tolist(),
        image_weights=weights.tolist(),
        local_image_count=int(local.sum()),
        radius_unclamped=radius,
        action=dict(
            action="GO_TOWARD_POINT",
            action_args=dict(
                r=min(radius, float(bound)),
                theta=float(np.arctan2(-delta[0], -delta[2])),
            ),
        )
        if eligible
        else None,
    )


def uniform_proposal(
    visited, current, bound, remaining, medoid=False, terminal_history=False
):
    """v10.31 ablation: native history selection and equal image weights; no model."""
    from .native_history import native_history_from_full

    visited = np.asarray(visited, dtype=float)
    if visited.ndim != 2 or visited.shape[1] != 3 or not len(visited):
        raise ValueError("invalid visited positions")
    if not np.array_equal(visited[-1], np.asarray(current)[:3]):
        raise ValueError("latest visited position differs from current pose")
    if medoid and terminal_history:
        raise ValueError("one spatial-reference change per candidate")
    indices = (
        list(range(max(0, len(visited) - 6), len(visited) - 1))
        if terminal_history
        else native_history_from_full(list(range(len(visited))), 20)
    )
    positions = [visited[i] for i in indices] + [np.asarray(current)[:3]] * 3
    result = memory_proposal(
        [[[1.0]]] * len(positions), positions, current, bound, remaining
    )
    if medoid:
        positions = np.asarray(positions)
        local = np.flatnonzero(
            np.linalg.norm(positions - np.asarray(current)[:3], axis=1) <= 5 * bound
        )
        points = positions[local]
        costs = np.linalg.norm(points[:, None] - points[None, :], axis=2).sum(axis=1)
        # Prefer the smallest displacement when actual visited medoids tie.
        chosen = local[
            np.lexsort(
                (np.linalg.norm(points - np.asarray(current)[:3], axis=1), costs)
            )[0]
        ]
        result = memory_proposal(
            [[[float(i == chosen)]] for i in range(len(positions))],
            positions,
            current,
            bound,
            remaining,
        )
    return result


def motor_step_proposal(visited, current, bound, remaining, step_size):
    """One simulator Motor unit, not the model's largest possible W1 displacement."""
    if (
        isinstance(step_size, (bool, np.bool_))
        or not np.isfinite(step_size)
        or step_size <= 0
    ):
        raise ValueError("invalid native Motor step")
    result = uniform_proposal(visited, current, bound, remaining)
    if result["eligible"]:
        result["action"]["action_args"]["r"] = min(
            result["action"]["action_args"]["r"], float(step_size)
        )
    return result
