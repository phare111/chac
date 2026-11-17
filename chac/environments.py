"""Construct a one-episode Habitat task from the public dataset."""

import gzip
import json
from pathlib import Path

OVON_SPLIT_FILES = {
    "val_seen": "val_seen.json.gz",
    "val_seen_synonyms": "val_unseen_easy.json.gz",
    "val_unseen": "val_unseen_hard.json.gz",
}


def ovon_split_data_path(data_root: Path, split: str) -> Path:
    try:
        filename = OVON_SPLIT_FILES[split]
    except KeyError as exc:
        raise ValueError(
            f"OVON split must be one of {sorted(OVON_SPLIT_FILES)}"
        ) from exc
    return data_root / split / filename


def environment(args):
    from habitat import Env, get_config
    from habitat.core.dataset import Dataset
    from habitat.datasets import make_dataset
    from habitat.tasks.nav.nav import NavigationEpisode, NavigationGoal
    from omegaconf import open_dict, read_write

    cfg = get_config(str(Path(__file__).with_name("ovon.yaml").resolve())).habitat
    row = {"scene": args.scene, "episode_id": args.episode_id}
    with read_write(cfg):
        with open_dict(cfg.task.actions.GO_TOWARD_POINT):
            cfg.task.actions.GO_TOWARD_POINT.rotate_agent = True
        if args.benchmark == "ovon":
            cfg.dataset.split = args.split
            cfg.dataset.data_path = str(
                ovon_split_data_path(args.dataset, args.split).resolve()
            )
            cfg.dataset.scenes_dir = str(args.scenes.resolve())
            cfg.dataset.content_scenes = [args.scene]
            dataset = make_dataset(cfg.dataset.type, config=cfg.dataset)
            dataset.episodes = [
                e
                for e in dataset.episodes
                if str(e.episode_id) == args.episode_id
                and Path(e.scene_id).name.removesuffix(".basis.glb") == args.scene
            ]
            seed = 20261021
        else:
            cfg.environment.iterator_options.group_by_scene = False
            cfg.task.type = "Nav-v0"
            cfg.task.measurements.distance_to_goal.distance_to = "POINT"
            cfg.task.measurements.success.success_distance = 3.0
            with open_dict(cfg.task.measurements):
                del cfg.task.measurements["topdown_map_obj"]
            cfg.task.lab_sensors = {}
            cfg.simulator.agents.main_agent.height = 1.25
            for sensor in cfg.simulator.agents.main_agent.sim_sensors.values():
                sensor.position = [0, 1.25, 0]
            with gzip.open(
                args.dataset / args.split / (args.split + ".json.gz"), "rt"
            ) as stream:
                episodes = json.load(stream)["episodes"]
            dataset = Dataset()
            dataset.episodes = []
            for index, e in enumerate(episodes):
                if (
                    str(e["episode_id"]) != args.episode_id
                    or Path(e["scene_id"]).stem != args.scene
                ):
                    continue
                scene = (args.scenes / e["scene_id"]).resolve()
                if (
                    not scene.is_relative_to(args.scenes.resolve())
                    or not scene.is_file()
                ):
                    raise ValueError("Invalid dataset scene path")
                row["instruction"] = e["instruction"]["instruction_text"]
                if (
                    not isinstance(row["instruction"], str)
                    or not row["instruction"].strip()
                ):
                    raise ValueError("Empty R2R instruction")
                dataset.episodes.append(
                    NavigationEpisode(
                        episode_id=args.episode_id,
                        scene_id=str(scene),
                        start_position=e["start_position"],
                        start_rotation=e["start_rotation"],
                        goals=[
                            NavigationGoal(
                                position=g["position"], radius=g.get("radius")
                            )
                            for g in e["goals"]
                        ],
                        info=e.get("info", {}),
                    )
                )
                seed = 42 + index
    if len(dataset.episodes) != 1:
        raise ValueError("Expected one episode matching the selected scene and ID")
    row["seed"] = seed if args.seed is None else args.seed
    return Env(cfg, dataset), row
