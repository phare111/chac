import unittest

import numpy as np

from chac.commitment import goal_event, motor_event
from chac.native_history import native_history_from_full
from chac.terminal import motor_step_proposal


class ExecutionTests(unittest.TestCase):
    def test_native_history(self):
        self.assertEqual(
            native_history_from_full(list(range(22)), 20), [0, *range(2, 21)]
        )

    def test_commitment(self):
        action = {"action": "GO_TOWARD_POINT"}
        prediction = {"arrive": [0.5, 0.51, 0, 0, 0]}
        self.assertFalse(goal_event(prediction, 1, action))
        self.assertTrue(goal_event(prediction, 2, action))
        self.assertTrue(motor_event([1, 0], [0, 1]))
        self.assertFalse(motor_event([0, 0], [0, 0]))

    def test_terminal_budget_and_native_cap(self):
        pose = np.array([0, 0, 0, 0, 0, 0, 1])
        visited = [[0, 0, -2], [0, 0, 0]]
        result = motor_step_proposal(visited, pose, 1.0, 2, 0.25)
        self.assertEqual(result["action"]["action_args"]["r"], 0.25)
        self.assertFalse(motor_step_proposal(visited, pose, 1.0, 1, 0.25)["eligible"])
