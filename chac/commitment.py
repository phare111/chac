"""Two native commitment events; no candidate, sampling, transport or fitted threshold."""

import hashlib, pickle, random, sys
import numpy as np


def rng_digest():
    state = [random.getstate(), np.random.get_state()]
    torch = sys.modules.get("torch")
    if torch is not None:
        state.extend(
            [
                torch.get_rng_state().numpy().tobytes(),
                [x.cpu().numpy().tobytes() for x in torch.cuda.get_rng_state_all()]
                if torch.cuda.is_initialized()
                else [],
            ]
        )
    return hashlib.sha256(pickle.dumps(state)).hexdigest()


def goal_event(prediction, cursor, action):
    a = np.asarray(prediction["arrive"], dtype=float)
    if (
        a.shape != (5,)
        or not np.isfinite(a).all()
        or not isinstance(cursor, (int, np.integer))
        or isinstance(cursor, (bool, np.bool_))
        or not 1 <= cursor < 5
    ):
        raise ValueError("expected native forecast and completed prefix 1..4")
    return bool(a[cursor - 1] > 0.5 and action["action"] == "GO_TOWARD_POINT")


def motor_event(command, preview):
    c, p = np.asarray(command, dtype=float), np.asarray(preview, dtype=float)
    if c.shape != (2,) or p.shape != (2,) or not np.isfinite([c, p]).all():
        raise ValueError("invalid native controller projection")
    return bool(c @ c > 0 and p @ c <= 0)


class Commitment:
    def __init__(self, mode):
        if mode not in ("task", "execution", "combined"):
            raise ValueError("unknown commitment mode")
        self.mode = mode
        self.records = []

    def check(self, prediction, cursor, action, contact, step, chunk, plan):
        before = rng_digest()
        try:
            task = goal_event(prediction, cursor, action)
            goal = self.mode != "execution" and task
            motor = False
            projection = None
            preview_available = None
            # A necessary normal rethink needs no additional geometry check.
            if (
                not goal
                and self.mode != "task"
                and action["action"] == "GO_TOWARD_POINT"
                and action["action_args"]["r"] > 0
            ):
                preview = getattr(contact, "preview", None)
                projection = preview(action) if preview is not None else None
                preview_available = projection is not None
                if preview_available:
                    if (
                        not isinstance(projection, dict)
                        or projection.get("step") != step
                        or projection.get("action") != action
                    ):
                        raise ValueError(
                            "controller preview must match the current step and action"
                        )
                    motor = motor_event(
                        projection["command_delta"], projection["delta"]
                    )
        finally:
            if rng_digest() != before:
                raise RuntimeError("read-only commitment check consumed randomness")
        record = dict(
            readonly_rng_exact=True,
            step=step,
            chunk=chunk,
            completed_index=cursor,
            next_index=cursor + 1,
            old_plan=plan,
            completed_arrive=float(prediction["arrive"][cursor - 1]),
            goal=bool(goal),
            motor=bool(motor),
            projection=projection,
            preview_available=preview_available,
            rethink=bool(goal or motor),
            reason="commitment_goal" if goal else "commitment_motor" if motor else None,
        )
        self.records.append(record)
        return record
