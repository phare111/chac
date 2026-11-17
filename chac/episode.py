"""Execute consecutive native waypoints until a boundary requests inference."""

import hashlib
import json
import random
import time
from collections import Counter

import numpy as np

from .commitment import Commitment
from .evidence import gray32, motor_event, visual_event
from .history import native_history_from_full
from .odometry import advance_odometry, rebase_targets_xz
from .registration import json_value
from .runtime import current_views, front_view_index, ovon_target_text
from .session import (
    _env_action,
    native_action,
    pose,
    reset_exact_episode,
    waypoint_targets_xz,
)
from .waypoints import decode


def run_episode(env, agent, row, terminal, stream):
    import torch

    reset_exact_episode(env, row)
    seed = row["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    boundary = Commitment("combined")
    contact = terminal.contact
    sensors = env._sim.habitat_config.agents.main_agent.sim_sensors
    offsets = np.array(
        [sensors[key].orientation[1] for key in ("left", "front", "right")]
    )
    fov = np.deg2rad(float(sensors.front.hfov))
    if any(sensors[key].hfov != sensors.front.hfov for key in ("left", "right")):
        raise ValueError("Expected equal horizontal fields of view")
    front = front_view_index("OVON-v1")
    history = [np.asarray(current_views(env, "OVON-v1")[front]).copy()]
    history_yaws = [0.0]
    r2r = "instruction" in row
    instruction = (
        row["instruction"]
        if r2r
        else ovon_target_text(env.current_episode.object_category)
    )
    odometry = np.zeros(3)
    cursor = 5
    calls = 0
    inference_s = 0.0
    prediction = anchor = targets = visual_anchor = old_centers = plan_id = None
    chosen = Counter()
    horizons = []
    reason = "initial"
    start = time.perf_counter()
    for step in range(500):
        if env.episode_over:
            break
        current = current_views(env, "OVON-v1")
        reuse = cursor < 5
        gate = commitment = None
        if reuse:
            action = decode(
                prediction,
                cursor,
                anchor,
                odometry,
                rebase_targets_xz(targets, odometry)[cursor],
            )
            gate = visual_event(
                np.stack([gray32(v) for v in current]),
                visual_anchor,
                odometry[2] + action["action_args"]["theta"],
                odometry[2] + offsets,
                old_centers,
                fov,
            )
            reuse = not gate["event"]
            reason = "continue_native_chunk" if reuse else "view_frontier"
            if reuse:
                commitment = boundary.check(
                    prediction, cursor, action, contact, step, calls, plan_id
                )
                if commitment["rethink"]:
                    reuse = False
                    reason = commitment["reason"]
        if not reuse:
            prediction = agent.forward(
                [current[0], current[2], current[1]] if r2r else current,
                native_history_from_full(history, 20),
                instruction,
                dataset_type="VLN-CE-v1" if r2r else "OVON-v1",
            )
            calls += 1
            horizons.append(0)
            chosen[1 if prediction["stop"] else 5] += 1
            inference_s += prediction["forward_ms"] / 1000
            visual_anchor = np.stack([gray32(v) for v in current])
            old_centers = list(odometry[2] + offsets) + native_history_from_full(
                history_yaws, 20
            )
            anchor = odometry.copy()
            targets = waypoint_targets_xz(prediction, anchor)
            plan_id = hashlib.sha256(
                np.r_[prediction["trunk"].ravel(), prediction["angle"]].tobytes()
            ).hexdigest()
            cursor = 0
            action = native_action(prediction)
        before = pose(env)
        observation = terminal.step(_env_action(action))
        stream.write(
            json.dumps(
                json_value(
                    dict(
                        step=step,
                        source="REUSE" if reuse else "THINK",
                        slot=cursor,
                        plan_id=plan_id,
                        action=action,
                        reason=reason,
                        visual=gate,
                        commitment=commitment,
                        pose_before=before,
                        pose_after=pose(env),
                    )
                )
            )
            + "\n"
        )
        horizons[-1] += 1
        if action["action"] == "STOP":
            break
        frame = np.asarray(observation["front"], dtype=np.uint8)[:, :, :3].copy()
        history.append(frame)
        odometry = advance_odometry(odometry, action)
        history_yaws.append(float(odometry[2]))
        cursor += 1
        reason = "chunk_exhausted"
        filters = contact.records[-1]["filters"]
        collision = bool(getattr(env._sim, "previous_step_collided", False)) or any(
            f["blocked"] for f in filters
        )
        if contact.receipt is None:
            raise RuntimeError("Missing native movement receipt")
        for record in filters:
            origin = np.asarray(record["start"])[[0, 2]]
            event = motor_event(
                np.asarray(record["requested"])[[0, 2]] - origin,
                contact.receipt[[0, 2]] - origin,
                collision,
            )
            if event["requires_replan"]:
                collision = True
        if collision:
            cursor = 5
            reason = "execution"
    actual_horizons = horizons.copy()
    actual_horizons[-1] += len(terminal.execution) - sum(horizons)
    metrics = env.get_metrics()
    return dict(
        success=metrics["success"],
        spl=metrics["spl"],
        distance_to_goal_m=metrics["distance_to_goal"],
        policy_calls=calls,
        inference_s=inference_s,
        episode_s=time.perf_counter() - start,
        environment_actions=len(terminal.execution),
        chosen_horizon_distribution=dict(chosen),
        native_horizon_distribution=dict(Counter(horizons)),
        executed_horizon_distribution=dict(Counter(actual_horizons)),
        mean_executed_horizon=np.mean(actual_horizons),
        terminal_proposal=terminal.proposal,
    )
