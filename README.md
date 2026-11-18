# CHAC

CHAC adaptively executes the five waypoints predicted by a frozen OmniNav Flow policy. It reuses the current plan until a change in the view, the plan’s arrival prediction, or controller feedback calls for replanning. Before STOP, it can take one corrective step using previously visited positions.

## Setup

Use Python 3.10+ in the OmniNav CUDA inference environment, with OmniNav’s Transformers fork and Habitat-Lab/Habitat-Sim 0.2.3. Install CHAC from this directory and download the Flow checkpoint separately. [OmniNav setup](https://github.com/amap-cvlab/OmniNav) · [Flow checkpoint](https://www.modelscope.ai/models/chongchongjj/OmniNav_Flow)

```bash
python -m pip install --no-deps -e .
```

## Run one episode

Set the variables below to the OmniNav checkout, checkpoint directory, OVON dataset root, and scene root containing `hm3d/`. Choose a fresh output directory for each run.

```bash
python -m chac --benchmark ovon --split val_seen \
  --omninav "$OMNINAV" --model "$FLOW_MODEL" \
  --dataset "$OVON_DATA" --scenes "$SCENES" \
  --scene 5cdEh9F2hJL --episode-id 1756 --seed 20261021 \
  --output outputs/ovon-1756
```

For R2R-CE, use `--benchmark r2r` with the preprocessed VLN-CE dataset, a scene root containing `mp3d/`, and an episode from the selected split. Each run writes episode metrics to `result.json`, policy decisions to `steps.jsonl`, and executed actions to `execution.json`.
