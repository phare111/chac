import numpy as np


def native_w1_bound(norm, scale):
    lo, hi = (
        np.asarray(norm["min"], dtype=float)[0, :2],
        np.asarray(norm["max"], dtype=float)[0, :2],
    )
    if not np.isfinite([*lo, *hi, scale]).all() or np.any(lo > hi) or scale <= 0:
        raise ValueError("invalid frozen native normalization")
    return float(scale * np.linalg.norm(np.maximum(np.abs(lo), np.abs(hi))))
