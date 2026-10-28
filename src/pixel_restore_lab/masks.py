"""Holes to paint into: a scratch, a rectangle, or scattered damage.

Every mask is a set of pixel coordinates plus the coverage it implies, and every generator is a
pure function of ``(size, ratio, seed)`` so the benchmark tables never move between runs.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterator
from dataclasses import dataclass

from .image import Gray

Pixel = tuple[int, int]


class MaskError(ValueError):
    """Raised for impossible masks or mask parameters."""


@dataclass(frozen=True)
class Mask:
    width: int
    height: int
    holes: frozenset[Pixel]

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise MaskError("mask dimensions must be positive")
        for x, y in self.holes:
            if not (0 <= x < self.width and 0 <= y < self.height):
                raise MaskError(
                    f"hole ({x}, {y}) is outside a {self.width}x{self.height} mask"
                )

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def coverage(self) -> float:
        """Share of the image that has to be invented."""
        return len(self.holes) / self.area

    def contains(self, x: int, y: int) -> bool:
        return (x, y) in self.holes


MaskFactory = Callable[[int, float, int], Mask]


def _target(size: int, ratio: float) -> int:
    if size < 8:
        raise MaskError("size must be at least 8")
    if not 0.0 < ratio < 1.0:
        raise MaskError("ratio must be strictly between 0 and 1")
    return max(1, round(size * size * ratio))


def _wobble(size: int, seed: int) -> list[int]:
    """A deterministic random walk, so the scratch is not a perfect straight line."""
    rng = random.Random(seed)
    drift = 0
    offsets = []
    for _ in range(size):
        drift = max(-2, min(2, drift + rng.choice((-1, 0, 0, 1))))
        offsets.append(drift)
    return offsets


def _diagonal_candidates(size: int, seed: int) -> Iterator[Pixel]:
    """Pixels ordered as widening diagonal bands, so any prefix is still a scratch."""
    wobble = _wobble(size, seed)
    for spread in range(size):
        for sign in ((1,) if spread == 0 else (1, -1)):
            for step in range(size):
                y = step + spread * sign + wobble[step]
                if 0 <= y < size:
                    yield (step, y)


def scratch_mask(size: int = 128, ratio: float = 0.1, seed: int = 3) -> Mask:
    """A diagonal scratch through the whole frame, plus a sprinkling of dust."""
    target = _target(size, ratio)
    holes: set[Pixel] = set()
    band_budget = max(1, round(target * 0.85))
    for pixel in _diagonal_candidates(size, seed):
        holes.add(pixel)
        if len(holes) >= band_budget:
            break

    rng = random.Random(seed + 1)
    attempts = 0
    limit = 8 * target + 64
    while len(holes) < target and attempts < limit:
        holes.add((rng.randrange(size), rng.randrange(size)))
        attempts += 1
    if len(holes) < target:
        for pixel in _diagonal_candidates(size, seed):
            holes.add(pixel)
            if len(holes) >= target:
                break
    return Mask(size, size, frozenset(holes))


def block_mask(size: int = 128, ratio: float = 0.1, seed: int = 5) -> Mask:
    """One or more rectangular holes - what a dropped tile or a sticker looks like."""
    target = _target(size, ratio)
    rng = random.Random(seed)
    blocks = 1 if target < 256 else 2 if target < 1600 else 3
    side = max(2, min(size, math.ceil(math.sqrt(target / blocks))))
    rectangles = []
    for _ in range(blocks):
        x0 = rng.randrange(0, size - side + 1)
        y0 = rng.randrange(0, size - side + 1)
        rectangles.append((x0, y0))

    holes: set[Pixel] = set()
    for x0, y0 in rectangles:
        for y in range(y0, min(size, y0 + side)):
            for x in range(x0, min(size, x0 + side)):
                holes.add((x, y))
                if len(holes) >= target:
                    return Mask(size, size, frozenset(holes))
    for y in range(size):
        for x in range(size):
            holes.add((x, y))
            if len(holes) >= target:
                return Mask(size, size, frozenset(holes))
    return Mask(size, size, frozenset(holes))


def scatter_mask(size: int = 128, ratio: float = 0.1, seed: int = 7) -> Mask:
    """Dropped pixels and 3x3 blobs, spread evenly - sensor damage rather than a gouge."""
    target = _target(size, ratio)
    rng = random.Random(seed)
    order = [(x, y) for y in range(size) for x in range(size)]
    rng.shuffle(order)
    holes: set[Pixel] = set()
    for index, (x, y) in enumerate(order):
        holes.add((x, y))
        if index % 3 == 2:
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < size and 0 <= ny < size:
                        holes.add((nx, ny))
        if len(holes) >= target:
            break
    if len(holes) > target:
        for pixel in sorted(holes)[target:]:
            holes.discard(pixel)
    return Mask(size, size, frozenset(holes))


MASK_KINDS: dict[str, MaskFactory] = {
    "scratch": scratch_mask,
    "block": block_mask,
    "scatter": scatter_mask,
}


def build_mask(kind: str, size: int = 128, ratio: float = 0.1, seed: int = 3) -> Mask:
    try:
        factory = MASK_KINDS[kind]
    except KeyError:
        choices = ", ".join(sorted(MASK_KINDS))
        raise MaskError(f"unknown mask {kind!r}; choose from {choices}") from None
    return factory(size, ratio, seed)


def damage(image: Gray, mask: Mask, value: int = 0) -> Gray:
    """Paint the holes out.

    This is what the restorer is given: the mask, and an image whose hole pixels hold nothing but
    a flat gouge. Nothing downstream may look at the original values under the holes.
    """
    if (image.width, image.height) != (mask.width, mask.height):
        raise MaskError(
            f"image is {image.width}x{image.height} but mask is {mask.width}x{mask.height}"
        )
    if not 0 <= value <= 255:
        raise MaskError("damage value must be between 0 and 255")
    pixels = bytearray(image.pixels)
    for x, y in mask.holes:
        pixels[y * image.width + x] = value
    return Gray(image.width, image.height, bytes(pixels))
