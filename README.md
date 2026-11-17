# CHAC

CHAC controls how long a robot executes a five-waypoint plan from a frozen OmniNav Flow policy. A policy call produces W1–W5. Before reusing the next waypoint, CHAC checks whether the view has changed in the intended direction, whether the plan predicted arrival at the preceding waypoint, and whether the controller can make progress. After movement, it checks the actual controller receipt. A boundary starts a new policy call.

At a native STOP, CHAC uses the robot’s own visited positions to propose at most one corrective movement. The displacement is capped by the native W1 envelope and the simulator’s forward step size. Execution requires a positive controller preview and enough budget for the original STOP.

This repository packages **full CHAC v10.36** from the research branch `method/chac-v1036-ablation`, commit `0e5bf8a2fa67cc7680cb30514fbc7f1593cbb7be`. The branch also contains a later A4 experiment with the visual boundary disabled. The default here uses the visual boundary.

## Installation

For CPU checks, use Python 3.10 or later from the repository root.

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python -m chac --help
```

For navigation, start with the upstream OmniNav Flow inference environment on a CUDA machine. It must provide PyTorch, qwen-vl-utils, the **OmniNav fork of Transformers** from `train_code/transformers-main`, Habitat-Lab and Habitat-Sim 0.2.3, and the runtime dependencies for the upstream OVON task. Install CHAC into that environment with `python -m pip install --no-deps -e .` to preserve its dependency versions. Keep both `infer_ovon/` and `infer_r2r_rxr/` under the OmniNav checkout. If using a source checkout of Habitat-Lab, set `CHAC_HABITAT_LAB_ROOT` to its top-level directory. [OmniNav source and environment setup](https://github.com/amap-cvlab/OmniNav)

Supply the Flow checkpoint separately. The reference checkpoint revision is `f73a8d094f53d0a8c038335cd0092eef8c1e5737`; its config SHA-256 is `107d23c1b8d7368f6f7fa144b74e24641556d6be1d32ff2f0a94c7d71bc9d4be`. [OmniNav Flow weights](https://www.modelscope.ai/models/chongchongjj/OmniNav_Flow)

## Run a development episode

Set the four shell variables below to your own paths. `OMNINAV` is the source checkout, `FLOW_MODEL` the checkpoint directory, `OVON_DATA` the directory containing the split folders, and `SCENES` the scene root containing `hm3d/`. The example uses the original development smoke episode.

```bash
python -m chac --benchmark ovon --split val_seen \
  --omninav "$OMNINAV" --model "$FLOW_MODEL" \
  --dataset "$OVON_DATA" --scenes "$SCENES" \
  --scene 5cdEh9F2hJL --episode-id 1756 --seed 20261021 \
  --output outputs/ovon-1756
```

For R2R-CE, pass `--benchmark r2r`, the preprocessed VLN-CE dataset root, a scene root containing `mp3d/`, and the scene stem and episode ID from the chosen split. The loader reads the official instruction. The policy receives left/right/front views while the visual boundary operates on left/front/right views. The default R2R seed is 42 plus the episode’s index in its split file. HM3D-OVON uses 20261021 by default. Pass `--seed` to reproduce a particular manifest row.

Both tasks use a 500-action budget. The success distance is 0.25 m for HM3D-OVON and 3 m for R2R-CE. Each invocation evaluates exactly one episode and requires a fresh output directory.

## Read the outputs

| File | Contents |
| --- | --- |
| `steps.jsonl` | One row per policy step, including plan ID, waypoint slot, reuse decision, visual and commitment checks, action, and pose |
| `execution.json` | Every environment action, including any terminal correction and the subsequent STOP, with controller feedback |
| `result.json` | Episode success, SPL, final distance, policy calls, measured VLM and Flow forwards, inference and loop times, action count, and horizon distributions |

The chosen horizon is five for a movement plan and one for a STOP plan. The executed horizon counts the actions actually taken per policy call, including any terminal correction. The native horizon excludes the extra terminal move. Inference timing starts after processor inputs are transferred to the device and ends after the decoded outputs reach the CPU. This release omits historical hidden-state diagnostics, so measure timings again for performance comparisons. Episode timing covers the execution loop and per-step trace writing, excluding model loading and environment setup.

## Code map

| Start here | Responsibility |
| --- | --- |
| [episode.py](chac/episode.py) | Follow one plan from inference through reuse, early refresh, or STOP |
| [evidence.py](chac/evidence.py), [commitment.py](chac/commitment.py) | Check visual, task, and motor boundaries |
| [feedback.py](chac/feedback.py) | Preview the native controller and verify its movement receipt |
| [terminal.py](chac/terminal.py) | Propose, bound, and execute the final correction |
| [runtime.py](chac/runtime.py), [waypoints.py](chac/waypoints.py) | Construct model inputs, run frozen inference, and decode waypoints |
| [environments.py](chac/environments.py), [ovon.yaml](chac/ovon.yaml) | Select the requested episode and configure its task, sensors, and success distance |

The public entry replaces the research branch’s paired-trace verifier, cluster scheduler, and result dashboard. Validation of this packaging covers CPU tests. End-to-end CUDA/Habitat equivalence against the original branch remains unverified.
