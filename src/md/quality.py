"""Quality-control helpers for MD-derived polymer targets."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def block_means(values: Iterable[float], n_blocks: int = 4) -> np.ndarray:
    """Return block means for finite values split into approximately equal blocks."""

    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if n_blocks <= 0:
        raise ValueError("n_blocks must be positive")
    if array.size < n_blocks:
        return np.array([], dtype=float)
    return np.array([block.mean() for block in np.array_split(array, n_blocks) if block.size], dtype=float)


def relative_block_error(values: Iterable[float], n_blocks: int = 4) -> float:
    """Estimate relative uncertainty from block means."""

    means = block_means(values, n_blocks=n_blocks)
    if means.size < 2:
        return math.nan
    center = float(np.mean(means))
    if center == 0:
        return math.inf
    standard_error = float(np.std(means, ddof=1) / math.sqrt(means.size))
    return abs(standard_error / center)


def fraction_converged(estimates: dict[str, float], tolerance: float = 0.1) -> bool:
    """Check whether partial-trajectory estimates agree with full estimate."""

    full = estimates.get("full")
    if full is None or not np.isfinite(full) or full == 0:
        return False
    required = ["first_25", "first_50", "first_75"]
    for key in required:
        value = estimates.get(key)
        if value is None or not np.isfinite(value):
            return False
        if abs((value - full) / full) > tolerance:
            return False
    return True


def correlation_time_supported(tau: float, trajectory_time: float | None, max_fraction: float = 0.2) -> bool:
    """Reject correlation times longer than a conservative trajectory fraction."""

    if trajectory_time is None or not np.isfinite(tau) or not np.isfinite(trajectory_time):
        return False
    if tau <= 0 or trajectory_time <= 0:
        return False
    return tau <= max_fraction * trajectory_time


def positive_log_target(value: float) -> float:
    """Transform a non-negative heavy-tailed target using log1p."""

    if value is None or not np.isfinite(value) or value < 0:
        return math.nan
    return float(np.log1p(value))

