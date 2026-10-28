"""PSNR and friends - the whole scoreboard.

PSNR compares peak signal to mean squared error, so on 8-bit greyscale it is
``10 * log10(255**2 / mse)``. Identical images give infinite dB; that is not a special case in the
maths, it is a division by zero, so it is reported as ``inf`` on purpose.
"""

from __future__ import annotations

import math

from .image import Gray
from .masks import Mask

PEAK = 255.0


class MetricError(ValueError):
    """Raised when two rasters cannot be compared."""


def _check_pair(a: Gray, b: Gray) -> None:
    if (a.width, a.height) != (b.width, b.height):
        raise MetricError(
            f"size mismatch: {a.width}x{a.height} vs {b.width}x{b.height}"
        )


def mse(a: Gray, b: Gray) -> float:
    _check_pair(a, b)
    total = 0
    for left, right in zip(a.pixels, b.pixels, strict=True):
        difference = left - right
        total += difference * difference
    return total / a.area


def rmse(a: Gray, b: Gray) -> float:
    return math.sqrt(mse(a, b))


def psnr(a: Gray, b: Gray) -> float:
    error = mse(a, b)
    if error == 0.0:
        return float("inf")
    return 10.0 * math.log10(PEAK * PEAK / error)


def masked_rmse(a: Gray, b: Gray, mask: Mask) -> float:
    """RMSE over the holes only - the number that says whether the fill is any good."""
    _check_pair(a, b)
    if (a.width, a.height) != (mask.width, mask.height):
        raise MetricError(
            f"mask is {mask.width}x{mask.height} but images are {a.width}x{a.height}"
        )
    if not mask.holes:
        raise MetricError("mask has no holes to score")
    total = 0
    for x, y in mask.holes:
        difference = a.pixels[y * a.width + x] - b.pixels[y * a.width + x]
        total += difference * difference
    return math.sqrt(total / len(mask.holes))
