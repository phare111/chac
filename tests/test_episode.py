"""Exercise the public loop with deterministic predictions and a simulated receiver."""

import io
import json
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch
import numpy as np

try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None, "episode adapter requires PyTorch")
class EpisodeTests(unittest.TestCase):
    def test_five_moves_then_stop(self):
        from chac import episode
        from chac.terminal_step import TerminalStep

        camera = NS(orientation=[0, 0, 0], hfov=110)

        class Cameras(dict):
            front = camera

        sensors = Cameras.fromkeys(("front", "left", "right"), camera)
        sim = NS(habitat_config=NS(agents=NS(main_agent=NS(sim_sensors=sensors))))
        state = NS(position=np.zeros(3), rotation=NS(x=0, y=0, z=0, w=1))
        sim.get_agent_state = lambda: state
        env = NS(
            _sim=sim,
            current_episode=NS(object_category="chair"),
            episode_over=False,
            get_metrics=lambda: dict(success=0, spl=0, distance_to_goal=1.0),
        )
        frame = np.zeros((32, 32, 3), np.uint8)
        contact = NS(records=[], receipt=state.position, preview=lambda action: None)

        def step(action):
            contact.records.append({"filters": []})
            return {"front": frame}

        contact.step = step
        terminal = TerminalStep(env, contact, 1.0, 0.25)
        chunk = dict(
            trunk=np.zeros((5, 2)),
            angle=np.zeros(5),
            arrive=np.zeros(5),
            forward_ms=1.0,
        )
        agent = Mock()
        agent.forward.side_effect = [
            dict(**chunk, stop=False),
            dict(**chunk, stop=True),
        ]
        stream = io.StringIO()
        with (
            patch.object(episode, "reset_exact_episode"),
            patch.object(episode, "current_views", return_value=[frame] * 3),
        ):
            result = episode.run_episode(env, agent, {"seed": 42}, terminal, stream)
        self.assertEqual(result["chosen_horizon_distribution"], {5: 1, 1: 1})
        self.assertEqual(result["environment_actions"], 6)
        self.assertEqual(result["policy_calls"], 2)
        self.assertEqual(
            [json.loads(line)["slot"] for line in stream.getvalue().splitlines()],
            [0, 1, 2, 3, 4, 0],
        )
