"""Read the native controller's final submitted position, after snap_point."""

from contextlib import ExitStack
from copy import deepcopy
from unittest.mock import patch

import numpy as np

from .contact import StalledContact


class MotorContact(StalledContact):
    def __init__(self, env):
        self.executing = False
        self.receipt = None
        self.projections = []
        try:
            super().__init__(env)
            self.original_observe = env._sim.get_observations_at
            self.observe_patch = patch.object(
                env._sim, "get_observations_at", self.observe
            )
            self.observe_patch.start()
        except BaseException:
            self.close()
            raise

    def observe(self, *args, **kwargs):
        result = self.original_observe(*args, **kwargs)
        if (
            self.executing
            and kwargs.get("keep_agent_at_new_pose")
            and "position" in kwargs
        ):
            p = np.asarray(kwargs["position"], dtype=float)
            if p.shape != (3,) or not np.isfinite(p).all():
                raise ValueError("invalid native execution receipt")
            self.receipt = p.copy()
        return result

    def step(self, action):
        self.executing = True
        self.receipt = None
        try:
            result = super().step(action)
            step = len(self.records) - 1
            # Previews are appended in step order; older steps cannot validate this receipt.
            for projection in reversed(self.projections):
                if projection["step"] < step:
                    break
                if projection["step"] == step and projection["action"] == action:
                    if self.receipt is None or not self.filters:
                        raise RuntimeError(
                            "native movement did not submit an execution receipt"
                        )
                    actual = (self.receipt - np.asarray(self.filters[-1]["start"]))[
                        [0, 2]
                    ]
                    if not np.allclose(actual, projection["delta"], rtol=0, atol=1e-5):
                        raise RuntimeError(
                            "native projection differs from executed receipt"
                        )
            return result
        finally:
            self.executing = False

    def preview(self, action):
        """Return an audited read-only projection, or None for an unsupported controller."""
        if (
            type(self.env.task.actions.get("GO_TOWARD_POINT")).__name__
            != "GoTowardPoint"
        ):
            return None
        self.project(action)
        return self.projections[-1]

    def project(self, action):
        """Run the audited native geometry, intercepting before any pose update or rendering."""
        if self.executing or action["action"] != "GO_TOWARD_POINT":
            raise ValueError("projection requires an idle native MOVE")
        controller = self.env.task.actions["GO_TOWARD_POINT"]
        if type(controller).__name__ != "GoTowardPoint":
            raise TypeError("unaudited native controller")
        before = self.env._sim.get_agent_state()
        start = np.asarray(before.position).copy()
        rotation = deepcopy(before.rotation)
        commands = []

        def filtered(a, b):
            commands.append(
                (np.asarray(b, dtype=float) - np.asarray(a, dtype=float))[[0, 2]]
            )
            return self.original_filter(a, b)

        def endpoint(*args, **kwargs):
            if args or not kwargs.get("keep_agent_at_new_pose"):
                raise ValueError("unexpected native projection submission")
            return np.asarray(kwargs["position"], dtype=float).copy()

        # The native step only reads pose/filter/snap until this submission; no env.step or future observations.
        try:
            with (
                patch.object(self.env._sim, "step_filter", filtered),
                patch.object(self.env._sim, "get_observations_at", endpoint),
            ):
                end = controller.step(**action["action_args"])
        finally:
            after = self.env._sim.get_agent_state()
            if not np.array_equal(start, after.position) or not np.array_equal(
                rotation, after.rotation
            ):
                raise RuntimeError("controller projection mutated pose")
        if end.shape != (3,) or not np.isfinite(end).all():
            raise ValueError("invalid projected controller endpoint")
        delta = (end - start)[[0, 2]]
        if len(commands) != 1:
            raise RuntimeError("expected one audited native movement request")
        self.projections.append(
            dict(
                step=len(self.records),
                action=deepcopy(action),
                delta=delta.tolist(),
                command_delta=commands[0].tolist(),
            )
        )
        return delta

    def close(self):
        with ExitStack() as cleanup:
            for name in ("filter_patch", "step_patch", "observe_patch"):
                patcher = getattr(self, name, None)
                if patcher is not None:
                    cleanup.callback(patcher.stop)
