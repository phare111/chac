"""Small operations on the frozen goal-conditioned representation."""

from __future__ import annotations

import numpy as np
import torch


def apply_representation_offset(
    value: torch.Tensor, offset: np.ndarray | torch.Tensor
) -> torch.Tensor:
    if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
        raise ValueError("representation must be a finite floating-point tensor")
    offset = torch.as_tensor(offset, device=value.device)
    if offset.is_complex():
        raise ValueError("representation offset must be real")
    offset = offset.to(torch.float32)
    if offset.shape != value.shape or not bool(torch.isfinite(offset).all()):
        raise ValueError(
            f"invalid representation offset {tuple(offset.shape)} for {tuple(value.shape)}"
        )
    result = (value.float() + offset).to(value.dtype)
    if not bool(torch.isfinite(result).all()):
        raise ValueError("representation offset overflowed the output dtype")
    return result
