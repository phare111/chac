"""Read native filter events; preserve the frozen C3 invalidation rule."""

from unittest.mock import patch

import numpy as np


class Contact:
    """Expose the action's own navmesh rejection as contact, without changing it."""

    def __init__(self, env):
        self.env = env
        self.filters = []
        self.records = []
        self.original_filter = env._sim.step_filter
        self.original_step = env.step
        self.filter_patch = patch.object(env._sim, "step_filter", self.filter)
        self.step_patch = patch.object(env, "step", self.step)
        self.filter_patch.start()
        self.step_patch.start()

    def filter(self, start, target):
        result = self.original_filter(start, target)
        a, b = np.asarray(target), np.asarray(result)
        assert (
            a.shape == b.shape == (3,) and np.isfinite(a).all() and np.isfinite(b).all()
        )
        # Numerical equality tolerance, not an authorization-risk threshold.
        self.filters.append(
            dict(
                requested=a.tolist(),
                filtered=b.tolist(),
                planar_error_m=float(np.linalg.norm((a - b)[[0, 2]])),
                blocked=not np.allclose(a[[0, 2]], b[[0, 2]], atol=1e-5, rtol=0),
            )
        )
        return result

    def step(self, action):
        self.filters = []
        result = self.original_step(action)
        self.records.append(
            dict(
                action=action,
                filters=self.filters.copy(),
                contact=any(r["blocked"] for r in self.filters),
                legacy_collision=bool(
                    getattr(self.env._sim, "previous_step_collided", False)
                ),
            )
        )
        return result

    def close(self):
        self.step_patch.stop()
        self.filter_patch.stop()


class StalledContact(Contact):
    def filter(self, start, target):
        result = super().filter(start, target)
        event = self.filters[-1]
        event["navmesh_filtered"] = event["blocked"]
        event["start"] = np.asarray(start).tolist()
        event["blocked"] = event["blocked"] and np.allclose(
            np.asarray(start)[[0, 2]], np.asarray(result)[[0, 2]], atol=1e-5, rtol=0
        )
        return result
