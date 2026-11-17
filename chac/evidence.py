"""Parameter-free motor and action-conditioned view evidence; no success guarantee."""

import math
import numpy as np
from PIL import Image


def motor_event(command, actual, rejected=False):
    command = np.asarray(command, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if (
        command.shape != (2,)
        or actual.shape != (2,)
        or not np.isfinite([command, actual]).all()
    ):
        raise ValueError("expected finite planar controller displacements")
    norm = float(command @ command)
    progress = float(actual @ command)
    return dict(
        error=float(np.linalg.norm(actual - command)),
        progress=progress,
        requires_replan=bool(rejected or (norm > 0 and progress <= 0)),
    )


def covered(direction, centers, hfov):
    if (
        not 0 < hfov < math.pi
        or not math.isfinite(direction)
        or not np.isfinite(centers).all()
    ):
        raise ValueError("invalid camera coverage")
    return any(
        abs((direction - c + math.pi) % (2 * math.pi) - math.pi) <= hfov / 2
        for c in centers
    )


def regions(gray):
    g = np.asarray(gray, dtype=np.float32)
    if g.shape != (32, 32) or not np.isfinite(g).all() or np.any((g < 0) | (g > 1)):
        raise ValueError("expected normalized gray32")
    # Equal-width comparisons: resample the thumbnail to 30 columns, then three 10-column regions.
    return np.stack(
        np.split(
            np.asarray(Image.fromarray(g).resize((30, 32), Image.Resampling.BOX)),
            3,
            axis=1,
        )
    )


def visual_event(current, anchor, direction, current_centers, old_centers, hfov):
    current = np.asarray(current)
    anchor = np.asarray(anchor)
    if (
        current.shape != (3, 32, 32)
        or anchor.shape != (3, 32, 32)
        or len(current_centers) != 3
    ):
        raise ValueError("expected the three actually processed camera thumbnails")
    new_view = not covered(direction, old_centers, hfov)
    relative = np.asarray(
        [(direction - c + math.pi) % (2 * math.pi) - math.pi for c in current_centers]
    )
    camera = int(np.argmin(np.abs(relative)))
    visible = abs(relative[camera]) <= hfov / 2
    if not visible:
        return dict(
            new_route_view=new_view,
            route_visible=False,
            route_novelty=False,
            event=False,
        )
    projected = -math.tan(float(relative[camera])) / math.tan(hfov / 2)
    region = min(2, max(0, int((projected + 1) * 1.5)))
    old = np.concatenate([regions(g) for g in anchor])
    now = regions(current[camera])
    novelty = np.asarray([min(float(np.mean(np.abs(p - q))) for q in old) for p in now])
    route_novelty = bool(novelty[region] > max(np.delete(novelty, region)))
    return dict(
        new_route_view=new_view,
        route_visible=True,
        camera=camera,
        region=region,
        novelty=novelty.tolist(),
        route_novelty=route_novelty,
        event=bool(new_view and route_novelty),
    )
