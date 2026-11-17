from pathlib import Path

import numpy as np

PLAN_LENGTH = 5


def pose(env) -> np.ndarray:
    state = env._sim.get_agent_state()
    return np.asarray(
        [
            *state.position,
            state.rotation.x,
            state.rotation.y,
            state.rotation.z,
            state.rotation.w,
        ],
        dtype=np.float64,
    )


def reset_exact_episode(env, route: dict) -> dict:
    episode = next(
        (
            value
            for value in env.episodes
            if str(value.episode_id) == str(route["episode_id"])
            and Path(value.scene_id)
            .name.removesuffix(".basis.glb")
            .removesuffix(".glb")
            == route["scene"]
        ),
        None,
    )
    if episode is None:
        raise RuntimeError(
            "frozen episode and scene are absent from the configured split"
        )
    if (
        route.get("category")
        and getattr(episode, "object_category", None) != route["category"]
    ):
        raise RuntimeError(
            "frozen ObjectNav category does not match the dataset episode"
        )
    previous = getattr(env, "_current_episode", None)
    if previous is not None:
        previous._shortest_path_cache = None
    env.current_episode = episode
    return env.reset()


def _env_action(action: str | dict) -> dict:
    if isinstance(action, dict):
        return (
            {**action, "action": "stop"} if action.get("action") == "STOP" else action
        )
    return {"action": "stop" if action == "STOP" else action}


def waypoint_targets_xz(prediction: dict, anchor_pose: np.ndarray) -> np.ndarray:
    trunk = np.asarray(prediction["trunk"], dtype=np.float64)
    anchor = np.asarray(anchor_pose, dtype=np.float64)
    if trunk.shape != (PLAN_LENGTH, 2) or anchor.shape != (3,):
        raise ValueError("native five-waypoint targets are unavailable")
    if not np.isfinite(trunk).all() or not np.isfinite(anchor).all():
        raise ValueError("native waypoint plan must be finite")
    # ponytail: reuse the predicted cumulative point directly; separate heading
    # control needs a richer action API than GO_TOWARD_POINT provides.
    x, forward = trunk.T
    c, s = np.cos(anchor[2]), np.sin(anchor[2])
    return np.stack(
        (anchor[0] + c * x - s * forward, anchor[1] - s * x - c * forward), axis=-1
    )


def native_action(prediction: dict) -> dict:
    if prediction["stop"]:
        return {"action": "STOP"}
    radius = float(np.linalg.norm(np.asarray(prediction["trunk"], dtype=np.float64)[0]))
    return {
        "action": "GO_TOWARD_POINT",
        "action_args": {
            "r": radius,
            "theta": float(np.asarray(prediction["angle"])[0]),
        },
    }
