"""Frozen OVON import and JSON adapters; no training or evaluation CLI."""

import importlib
import math
import os
import sys
import types
from pathlib import Path

import numpy as np
import torch


def json_value(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def load_ovon_inference_modules(infer_root: Path) -> None:
    habitat_root = os.environ.get("CHAC_HABITAT_LAB_ROOT")
    habitat_deps = os.environ.get("CHAC_HABITAT_DEPS")
    if habitat_root:
        habitat_root = Path(habitat_root).resolve()
        lab_path = habitat_root / "habitat-lab"
        baselines_path = habitat_root / "habitat-baselines"
        if not (
            (lab_path / "habitat").is_dir()
            and (baselines_path / "habitat_baselines").is_dir()
        ):
            raise FileNotFoundError(
                f"expected Habitat-Lab 0.2.3 source under {habitat_root}"
            )
        sys.path[:0] = [str(lab_path), str(baselines_path)]
        if "habitat_baselines" not in sys.modules:
            # ponytail: bypass trainer imports; OVON only needs baseline config dataclasses.
            package = types.ModuleType("habitat_baselines")
            package.__path__ = [str(baselines_path / "habitat_baselines")]
            package.__package__ = "habitat_baselines"
            sys.modules["habitat_baselines"] = package
    if habitat_deps:
        habitat_deps = Path(habitat_deps).resolve()
        if not habitat_deps.is_dir():
            raise FileNotFoundError(
                f"Habitat OVON dependencies not found: {habitat_deps}"
            )
        sys.path.insert(0, str(habitat_deps))

    # ponytail: skip OVON's training-only top-level imports; install training extras to restore them.
    package = types.ModuleType("ovon")
    package.__path__ = [str(infer_root / "ovon")]
    package.__package__ = "ovon"
    sys.modules["ovon"] = package
    for name in (
        "config",
        "dataset",
        "measurements.nav",
        "measurements.imagenav",
        "task.rewards",
        "task.sensors",
        "task.simulator",
    ):
        importlib.import_module(f"ovon.{name}")

    native_root = infer_root.parent / "infer_r2r_rxr" / "VLN_CE"
    for name, path in (
        ("VLN_CE", native_root),
        ("VLN_CE.habitat_extensions", native_root / "habitat_extensions"),
    ):
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module
    importlib.import_module("VLN_CE.habitat_extensions.actions")
