"""Evaluate one explicitly selected HM3D-OVON or R2R-CE episode."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("omninav", "model", "dataset", "scenes", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--benchmark", choices=("ovon", "r2r"), default="ovon")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument(
        "--split",
        default="val_seen",
        choices=("val_seen", "val_seen_synonyms", "val_unseen"),
    )
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    from .registration import load_ovon_inference_modules, json_value

    root = args.omninav.resolve()
    load_ovon_inference_modules(root / "infer_ovon")
    sys.path.insert(0, str(root / "infer_r2r_rxr"))
    from .runtime import OmniNav, FLOW_MATCH_NORM, PREDICT_SCALE
    from .bounds import native_w1_bound
    from .episode import run_episode
    from .feedback import MotorContact
    from .terminal_step import TerminalStep

    args.output.mkdir(parents=True, exist_ok=False)
    agent = OmniNav(str(args.model.resolve()))
    if (
        not agent.model.cross_attention_flow_match
        or agent.model.action_head.num_inference_timesteps != 10
    ):
        raise ValueError(
            "Expected the frozen OmniNav ten-step Flow Matching checkpoint"
        )
    from .environments import environment

    with ExitStack() as stack:
        env, row = environment(args)
        stack.callback(env.close)
        env.seed(row["seed"])
        contact = MotorContact(env)
        stack.callback(contact.close)
        terminal = TerminalStep(
            env,
            contact,
            native_w1_bound(FLOW_MATCH_NORM[0], PREDICT_SCALE),
            float(env._sim.habitat_config.forward_step_size),
        )
        counts = {"vlm_forwards": 0, "flow_head_forwards": 0}

        def counter(name):
            def increment(*_):
                counts[name] += 1

            return increment

        for name, module in [
            ("vlm_forwards", agent.model),
            ("flow_head_forwards", agent.model.action_head.action_decoder),
        ]:
            stack.callback(module.register_forward_pre_hook(counter(name)).remove)
        with (args.output / "steps.jsonl").open("x") as stream:
            result = run_episode(env, agent, row, terminal, stream)
        for name, data in [
            ("execution", terminal.execution),
            (
                "result",
                dict(
                    **row,
                    benchmark=args.benchmark,
                    split=args.split,
                    **result,
                    **counts,
                ),
            ),
        ]:
            (args.output / (name + ".json")).write_text(
                json.dumps(json_value(data), indent=2, allow_nan=False) + "\n"
            )


if __name__ == "__main__":
    main()
