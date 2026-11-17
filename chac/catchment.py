"""Online Cognitive Horizon: native execution protection and one visual event."""

import numpy as np
from PIL import Image


def gray32(rgb):
    array = np.asarray(rgb)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("expected uint8 RGB")
    return (
        np.asarray(
            Image.fromarray(array).convert("L").resize((32, 32), Image.Resampling.BOX),
            dtype=np.float64,
        )
        / 255
    )


class Catchment:
    def __init__(self, mode="visual"):
        if mode not in (
            "fixed",
            "local",
            "visual",
            "motor",
            "evidence",
            "content",
            "sparse",
        ):
            raise ValueError("unknown horizon mode")
        self.mode = mode
        self.reset_episode()

    def reset_episode(self):
        self.previous = None

    def on_policy_inference(self, rgb):
        self.previous = gray32(rgb)
        self.total = 0.0
        self.executed = 0
        self.exited = False

    def on_action_executed(self, rgb, blocked=False):
        if self.previous is None or self.exited or self.executed >= 5:
            raise ValueError("no active contiguous chunk")
        current = gray32(rgb)
        d = float(np.mean(np.abs(current - self.previous)))
        self.executed += 1
        visual = self.mode == "visual" and self.executed >= 2 and d > self.total
        execution = self.mode != "fixed" and bool(blocked)
        reason = "execution" if execution else ("visual" if visual else None)
        result = dict(
            index=self.executed,
            delta=d,
            previous_total=self.total,
            visual_event=bool(visual),
            execution_event=execution,
            requires_replan=reason is not None,
            reason=reason,
        )
        self.total += d
        self.previous = current
        self.exited = reason is not None
        return result
