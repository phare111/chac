from copy import deepcopy
import numpy as np
from .terminal import uniform_proposal, motor_step_proposal


class TerminalStep:
    def __init__(self, env, contact, bound, step_size=None):
        self.env = env
        self.contact = contact
        self.bound = bound
        self.step_size = step_size
        self.visited = []
        self.execution = []
        self.proposal = None

    def execute(self, action, source):
        from .session import pose

        before = pose(self.env).copy()
        result = self.contact.step(action)
        after = pose(self.env).copy()
        receipt = getattr(self.contact, "records", [])
        self.execution.append(
            dict(
                action=deepcopy(action),
                source=source,
                pose_before=before.tolist(),
                pose_after=after.tolist(),
                contact=deepcopy(receipt[-1]) if receipt else None,
            )
        )
        return result

    def step(self, action):
        from .session import pose, _env_action

        if not self.visited:
            self.visited.append(pose(self.env)[:3].copy())
        if action["action"].lower() == "stop":
            self.proposal = (
                uniform_proposal(
                    self.visited, pose(self.env), self.bound, 500 - len(self.execution)
                )
                if self.step_size is None
                else motor_step_proposal(
                    self.visited,
                    pose(self.env),
                    self.bound,
                    500 - len(self.execution),
                    self.step_size,
                )
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
