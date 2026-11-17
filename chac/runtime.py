#!/usr/bin/env python3
"""Frozen OmniNav inference adapter and ObjectNav environment construction."""

from __future__ import annotations

import copy
import time

import numpy as np
import torch
from PIL import Image

from .offset import apply_representation_offset


NUM_ACTION_TRUNK = 5
NUM_CURRENT_IMAGE = 3
INPUT_SIZE = (640, 569)
PREDICT_SCALE = 0.3
FLOW_MATCH_NORM = [
    {
        "min": [
            [
                -0.49142804741859436,
                -0.018926994875073433,
                -0.5000011853675626,
                0.8660246981163404,
                0.0,
            ],
            [
                -0.8506758809089646,
                -0.11684392392635345,
                -0.5176391471000088,
                -0.36602701582911296,
                0.0,
            ],
            [
                -0.9391180276870728,
                -0.262770414352417,
                -0.5176390363234377,
                -0.5000015591363245,
                0.0,
            ],
            [
                -0.9319084137678146,
                -0.5872985124588013,
                -0.5176390363234377,
                -0.5176391890893195,
                0.0,
            ],
            [
                -0.9333658218383789,
                -0.8579317331314087,
                -0.5176390363233605,
                -0.5176391431200632,
                0.0,
            ],
        ],
        "max": [
            [0.8222980499267578, 1.1485368013381958, 0.5000012222510074, 1.0, 1.0],
            [
                0.8579317331314087,
                1.0390557050704985,
                0.5176391335634103,
                0.13397477820902748,
                1.0,
            ],
            [
                0.9584183096885622,
                0.9541159868240356,
                0.5176391335632885,
                0.3660255949191993,
                1.0,
            ],
            [
                0.9442337155342072,
                0.9441415071487427,
                0.5176391335631186,
                0.5000004173778672,
                1.0,
            ],
            [
                0.9610724449157715,
                0.9491362571716309,
                0.5176391335630393,
                0.5176390878671062,
                1.0,
            ],
        ],
    }
]
OBJECTNAV_CATEGORIES = frozenset(
    {
        "chair",
        "table",
        "picture",
        "cabinet",
        "cushion",
        "sofa",
        "bed",
        "chest_of_drawers",
        "plant",
        "sink",
        "toilet",
        "stool",
        "towel",
        "tv_monitor",
        "shower",
        "bathtub",
        "counter",
        "fireplace",
        "gym_equipment",
        "seating",
        "clothes",
    }
)


def as_image(value: np.ndarray) -> Image.Image:
    return Image.fromarray(np.asarray(value).astype(np.uint8))


def objectnav_target_text(category: str) -> str:
    if category not in OBJECTNAV_CATEGORIES:
        raise ValueError(f"unknown ObjectNav category: {category}")
    return f"Find a {category.replace('_', ' ')}."


def ovon_target_text(category: str) -> str:
    category = str(category).strip()
    if not category or "\n" in category:
        raise ValueError("OVON category must be a nonempty single line")
    return f"Find a {category.replace('_', ' ')}."


def _view_order(dataset_type: str | None) -> tuple[str, str, str]:
    if dataset_type == "OVON-v1":
        return "left", "front", "right"
    if dataset_type in (None, "ObjectNav-v1", "VLN-CE-v1"):
        return "left", "right", "front"
    raise ValueError(f"unsupported navigation dataset: {dataset_type}")


def front_view_index(dataset_type: str | None = None) -> int:
    return _view_order(dataset_type).index("front")


def _current_view_labels(dataset_type: str | None) -> str:
    return ", ".join(f"{name}side: " for name in _view_order(dataset_type))


def native_text_prompt(
    instruction: str, dataset_type: str | None = "ObjectNav-v1"
) -> str:
    return (
        "You are an autonomous navigation robot. You will get a task with historical pictures and current pictures you see.\n"
        f"            Based on these information, you need to decide your next {NUM_ACTION_TRUNK} actions, which could involve "
        "<|left|>,<|right|>,<|forward|>. If you finish your mission, output <|stop|>. Here are some examples: "
        "<|left|><|forward|><|forward|><|stop|>, <|forward|><|forward|><|forward|><|left|><|forward|> or <|stop|>\n"
        "            # Your historical pictures are: \n"
        f"            # Your current observations is {_current_view_labels(dataset_type)}\n"
        f"            # Your mission is: {instruction}<|NAV|>"
    )


def model_goal_input(
    episode, dataset_type: str | None = None, instruction_override: str | None = None
) -> tuple[None, str]:
    category = str(episode.object_category)
    if dataset_type is None:
        dataset_type = "ObjectNav-v1" if category in OBJECTNAV_CATEGORIES else "OVON-v1"
    if dataset_type == "OVON-v1":
        if instruction_override is not None:
            if (
                not isinstance(instruction_override, str)
                or not instruction_override.strip()
            ):
                raise ValueError("OVON instruction override must be a nonempty string")
            return None, instruction_override
        return None, ovon_target_text(category)
    if dataset_type != "ObjectNav-v1":
        raise ValueError(f"unsupported navigation dataset: {dataset_type}")
    return None, objectnav_target_text(category)


def current_views(env, dataset_type: str | None = None) -> list[np.ndarray]:
    """Render current views in the native checkpoint order for the dataset."""
    from scipy.spatial.transform import Rotation

    state = env._sim.get_agent_state()
    position = np.asarray(state.position).tolist()
    quaternion = [
        state.rotation.x,
        state.rotation.y,
        state.rotation.z,
        state.rotation.w,
    ]
    base_rotation = Rotation.from_quat(quaternion)
    observations = env._sim.get_observations_at(position, quaternion)
    order = _view_order(dataset_type)
    if "rgb" not in observations:
        keys = order
        return [np.asarray(observations[key], dtype=np.uint8) for key in keys]
    angles = {"left": 90, "right": -90, "front": 0}
    return [
        np.asarray(
            env._sim.get_observations_at(
                position,
                (Rotation.from_euler("y", angles[name], degrees=True) * base_rotation)
                .as_quat()
                .tolist(),
            )["rgb"],
            dtype=np.uint8,
        )
        for name in order
    ]


class OmniNav:
    def __init__(self, model_path: str):
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.process_vision_info = process_vision_info
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
            attn_implementation="sdpa",
        ).eval()
        self.nav_inner_ids = tuple(
            self.processor.tokenizer("|NAV|", add_special_tokens=False).input_ids
        )
        self.last = {}
        self._representation_offset = None
        query = getattr(self.model, "query_multihead_attn", None)
        if query is None:
            query = getattr(self.model, "query_multihead_multi_attn")
        query.register_forward_hook(self._query_hook)
        self.model.wp_predictor.register_forward_pre_hook(self._wp_hook)
        self.model.model.layers[-1].register_forward_hook(self._decoder_hook)

    def _query_hook(self, _module, _inputs, output):
        value = output[0] if isinstance(output, tuple) else output
        representation_offset = self._representation_offset
        if representation_offset is not None:
            self._representation_offset = None
            self.last["r2_policy_context_unshifted"] = value.detach().float().cpu()
            value = apply_representation_offset(value, representation_offset)
            output = (value, *output[1:]) if isinstance(output, tuple) else value
            self.last["r2_offset_applied"] = True
        self.last["r2_policy_context"] = value.detach().float().cpu()
        return output if representation_offset is not None else None

    def _wp_hook(self, _module, inputs):
        self.last["r3_pre_action"] = inputs[0].detach().float().cpu()

    def _decoder_hook(self, _module, _inputs, output):
        value = output[0] if isinstance(output, tuple) else output
        self.last["r1_decoder_hidden"] = value.detach().float().cpu()

    def _capture_nav_hidden(self, input_ids, dataset_type):
        hidden = self.last["r1_decoder_hidden"]
        ids = input_ids[0].detach().cpu().tolist()
        nav_index = None
        width = len(self.nav_inner_ids)
        for start in range(len(ids) - width + 1):
            if tuple(ids[start : start + width]) == self.nav_inner_ids:
                # The following token carries the closing '>' (often fused with
                # the following newline), so its hidden state is the final
                # marker state rather than the last '|' piece.
                nav_index = start + width
        if not width or nav_index is None or nav_index >= len(ids):
            if dataset_type == "VLN-CE-v1":
                for end in range(max(1, len(ids) - 32), len(ids) + 1):
                    if "<|NAV|>" in self.processor.tokenizer.decode(
                        ids[max(0, end - 12) : end]
                    ):
                        self.last["r1_nav_hidden"] = hidden[:, end - 1 : end]
                        self.last["r1_nav_token_index"] = end - 1
                        return
            raise RuntimeError(
                "complete <|NAV|> marker was not found in model input_ids"
            )
        self.last["r1_nav_hidden"] = hidden[:, nav_index : nav_index + 1]
        self.last["r1_nav_token_index"] = nav_index

    @staticmethod
    def representation_metadata(representations: dict) -> dict:
        consumers = {
            "r1_nav_hidden": "decoder output; no direct action-head consumer",
            "r2_policy_context": "wp_predictor, wp_predictor_angle, and arrive_predictor via action_feature",
            "r3_pre_action": "wp_predictor linear head",
        }
        result = {}
        for name, value in representations.items():
            if not isinstance(value, torch.Tensor):
                continue
            result[name] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype).replace("torch.", ""),
                "device": str(value.device),
                "requires_grad": bool(value.requires_grad),
                "next_consumer": consumers.get(name),
            }
        return result

    def _messages(
        self,
        current: np.ndarray | list[np.ndarray],
        history: list[np.ndarray],
        goal: np.ndarray | None,
        instruction: str,
        dataset_type: str | None = None,
    ):
        current_views = (
            list(current)
            if isinstance(current, (list, tuple))
            else [current] * NUM_CURRENT_IMAGE
        )
        if len(current_views) != NUM_CURRENT_IMAGE:
            raise ValueError(
                f"expected {NUM_CURRENT_IMAGE} current views, got {len(current_views)}"
            )
        images = ([as_image(goal)] if goal is not None else []) + [
            as_image(x) for x in history
        ]
        images += [as_image(x) for x in current_views]
        content = []
        for index, image in enumerate(images):
            scale = 1 if index >= len(images) - NUM_CURRENT_IMAGE else 0.25
            content.append(
                {
                    "type": "image",
                    "image": image,
                    "resized_height": int(INPUT_SIZE[1] * scale),
                    "resized_width": int(INPUT_SIZE[0] * scale),
                }
            )
        if goal is None:
            prompt = native_text_prompt(instruction, dataset_type)
            if dataset_type == "VLN-CE-v1":
                prompt = prompt.replace("            ", "")
        else:
            task = "The first image is the visual goal. Navigate to the location shown in the visual goal."
            prompt = (
                "You are an autonomous navigation robot. Based on the historical and current pictures, "
                f"decide the next {NUM_ACTION_TRUNK} waypoint actions.\n"
                "Your historical pictures are: "
                f"Your current observations are {_current_view_labels(dataset_type)}.\n"
                f"Your mission is: {task}<|NAV|>\nOutput the waypoint"
            )
        content.append(
            {
                "type": "text",
                "text": prompt,
            }
        )
        return [{"role": "user", "content": content}]

    def forward(
        self,
        current: np.ndarray | list[np.ndarray],
        history: list[np.ndarray],
        goal: np.ndarray | None,
        instruction: str,
        representation_offset: np.ndarray | None = None,
        capture_flow_trace: bool = False,
        capture_policy_context: bool = False,
        dataset_type: str | None = None,
    ) -> dict:
        messages = self._messages(current, history, goal, instruction, dataset_type)
        text = (
            self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            + "<|im_end|>"
        )
        text = text.replace("<|vision_start|><|image_pad|><|vision_end|>", "")
        num_history = len(messages[0]["content"]) - 1 - NUM_CURRENT_IMAGE
        pads = "<|vision_start|><|image_pad|><|vision_end|>"
        marker = "Your historical pictures are: "
        text = text.replace(marker, marker + pads * num_history)
        text = text.replace("leftside: ", "leftside: " + pads)
        text = text.replace("rightside: ", "rightside: " + pads)
        text = text.replace("frontside: ", "frontside: " + pads)
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")
        self.last = {}
        started = time.perf_counter()
        if (
            representation_offset is not None
            and not np.isfinite(representation_offset).all()
        ):
            raise ValueError("representation_offset must be finite")
        flow_enabled = getattr(self.model, "cross_attention_flow_match", False) is True
        flow_trace = None
        flow_hooks = []
        flow_context = []
        if capture_policy_context and flow_enabled:
            norm = getattr(getattr(self.model, "model", None), "norm", None)
            if norm is None:
                raise RuntimeError(
                    "Flow checkpoint does not expose its final hidden-state norm"
                )
            flow_hooks.append(
                norm.register_forward_hook(
                    lambda _module, _inputs, output: flow_context.append(output)
                )
            )
        if capture_flow_trace:
            action_head = getattr(self.model, "action_head", None)
            if not flow_enabled or action_head is None:
                raise RuntimeError(
                    "Flow trace requested from a checkpoint without a Flow Matching action head"
                )
            steps = int(action_head.num_inference_timesteps)
            if steps <= 0:
                raise RuntimeError("Flow Matching checkpoint has no inference steps")
            flow_trace = []

            def capture_state(_module, inputs):
                if (
                    len(flow_trace) >= steps
                    or not inputs
                    or not isinstance(inputs[0], torch.Tensor)
                ):
                    raise RuntimeError("invalid Flow Matching state trace")
                flow_trace.append(
                    {
                        "x_t": inputs[0].detach().float().cpu(),
                        "t": len(flow_trace) / steps,
                    }
                )

            def capture_velocity(_module, _inputs, output):
                value = output[0] if isinstance(output, tuple) else output
                if (
                    not flow_trace
                    or "v_t" in flow_trace[-1]
                    or not isinstance(value, torch.Tensor)
                ):
                    raise RuntimeError("invalid Flow Matching velocity trace")
                flow_trace[-1]["v_t"] = value.detach().float().cpu()

            flow_hooks.extend(
                (
                    action_head.action_encoder.register_forward_pre_hook(capture_state),
                    action_head.action_decoder.register_forward_hook(capture_velocity),
                )
            )
        self._representation_offset = representation_offset
        try:
            with torch.inference_mode():
                model_inputs = {
                    **inputs,
                    "action_former": True,
                    "gt_waypoints": 0,
                    "train": False,
                    "train_branch": ["continue"],
                    "return_dict": False,
                }
                if flow_enabled:
                    model_inputs["norm"] = FLOW_MATCH_NORM
                outputs = self.model(**model_inputs)
        finally:
            self._representation_offset = None
            for hook in flow_hooks:
                hook.remove()
        if representation_offset is not None and not self.last.get("r2_offset_applied"):
            raise RuntimeError(
                "representation offset was not consumed by query_multihead_attn"
            )
        self._capture_nav_hidden(inputs["input_ids"], dataset_type)
        if (
            capture_policy_context
            and flow_enabled
            and "r2_policy_context" not in self.last
        ):
            if len(flow_context) != 1:
                raise RuntimeError(
                    f"expected one Flow context tensor, got {len(flow_context)}"
                )
            context = (
                flow_context[0][0]
                if isinstance(flow_context[0], tuple)
                else flow_context[0]
            )
            query = self.model.query_action.expand(context.shape[0], -1, -1)
            # Flow uses the token sequence directly; OdrNav consumes the checkpoint's learned pooling.
            with torch.inference_mode():
                if self.model.query_action_layer == 1:
                    feature, _ = self.model.query_multihead_attn(
                        query, context, context
                    )
                else:
                    feature = self.model.query_multihead_multi_attn(
                        query, context, context
                    )
            self.last["r2_policy_context"] = feature.detach().float().cpu()
        elapsed_ms = (time.perf_counter() - started) * 1000
        if not isinstance(outputs, tuple) or len(outputs) != 4:
            raise RuntimeError(
                "frozen OmniNav requires waypoint, arrive, sine, and cosine outputs"
            )
        waypoint, arrive, sin_angle, cos_angle = outputs
        expected_shapes = ((1, NUM_ACTION_TRUNK, 2),) + ((1, NUM_ACTION_TRUNK),) * 3
        for name, value, shape in zip(
            ("waypoint", "arrive", "sin_angle", "cos_angle"), outputs, expected_shapes
        ):
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
                raise RuntimeError(f"invalid native {name} output; expected {shape}")
            if not torch.isfinite(value).all():
                raise RuntimeError(f"nonfinite native {name} output")
        waypoint = waypoint.detach().float().cpu().numpy()[0] * PREDICT_SCALE
        arrive_values = arrive.detach().float().cpu().numpy().reshape(-1)
        angle = (
            torch.atan2(sin_angle, cos_angle).detach().float().cpu().numpy().reshape(-1)
        )
        stopped = (
            bool(np.all(arrive_values > 0.5))
            if flow_enabled
            else bool(np.all(arrive_values >= 0))
        )
        result = {
            "trunk": waypoint,
            "angle": angle,
            "arrive": arrive_values,
            "stop": stopped,
            "angle_source": "native_flow_cumulative_sin_cos"
            if flow_enabled
            else "native_predict_angle_head",
            "forward_ms": elapsed_ms,
            "representations": copy.deepcopy(self.last),
            "representation_meta": self.representation_metadata(self.last),
        }
        if flow_trace is not None:
            if len(flow_trace) != steps or any(
                "v_t" not in step for step in flow_trace
            ):
                raise RuntimeError(
                    f"incomplete Flow Matching trace: expected {steps} steps, got {len(flow_trace)}"
                )
            result["flow_trace"] = flow_trace
        return result
