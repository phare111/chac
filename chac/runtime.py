"""Frozen OmniNav inference adapter."""

from __future__ import annotations

import time

import numpy as np
import torch
from PIL import Image

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


def as_image(value: np.ndarray) -> Image.Image:
    return Image.fromarray(np.asarray(value).astype(np.uint8))


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

    def _messages(
        self,
        current: np.ndarray | list[np.ndarray],
        history: list[np.ndarray],
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
        images = [as_image(x) for x in history]
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
        prompt = native_text_prompt(instruction, dataset_type)
        if dataset_type == "VLN-CE-v1":
            prompt = prompt.replace("            ", "")
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
        instruction: str,
        dataset_type: str | None = None,
    ) -> dict:
        messages = self._messages(current, history, instruction, dataset_type)
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
        started = time.perf_counter()
        with torch.inference_mode():
            outputs = self.model(
                **inputs,
                action_former=True,
                gt_waypoints=0,
                train=False,
                train_branch=["continue"],
                return_dict=False,
                norm=FLOW_MATCH_NORM,
            )
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
        return {
            "trunk": waypoint,
            "angle": angle,
            "arrive": arrive_values,
            "stop": bool(np.all(arrive_values > 0.5)),
            "forward_ms": (time.perf_counter() - started) * 1000,
        }
