"""Three classical fills, none of which is allowed to look at the pixels it replaces.

All three are driven only by the damaged image and the mask. The damaged image holds a flat gouge
inside the holes, so every value that ends up under a hole was produced from *known* neighbours:
which is exactly the constraint that separates inpainting from cheating.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable

from .image import Gray
from .masks import Mask

NEIGHBOURS = ((-1, 0), (1, 0), (0, -1), (0, 1))
METHOD_NAMES: tuple[str, ...] = ("nearest", "diffusion", "median")
MIN_MEDIAN_SAMPLES = 4
MEDIAN_RADIUS = 2
MID_GREY = 128


class RestoreError(ValueError):
    """Raised when a restoration cannot be attempted."""


def _check(image: Gray, mask: Mask, iterations: int) -> None:
    if (image.width, image.height) != (mask.width, mask.height):
        raise RestoreError(
            f"image is {image.width}x{image.height} but mask is {mask.width}x{mask.height}"
        )
    if isinstance(iterations, bool) or not isinstance(iterations, int):
        raise RestoreError("iterations must be an integer")
    if iterations < 0:
        raise RestoreError("iterations must not be negative")


def _known_mean(image: Gray, mask: Mask) -> int:
    total = 0
    count = 0
    width, height = image.width, image.height
    for y in range(height):
        for x in range(width):
            if (x, y) not in mask.holes:
                total += image.pixels[y * width + x]
                count += 1
    return total // count if count else MID_GREY


def _nearest_buffer(image: Gray, mask: Mask) -> bytearray:
    """Multi-source BFS from the hole boundary: O(pixels), one pass per pixel.

    Every hole pixel touching known pixels becomes a source holding the mean of those known
    neighbours; the wavefront then carries each source's value inward, so a hole pixel ends up
    with the value of the nearest known edge.
    """
    width, height = image.width, image.height
    buffer = bytearray(image.pixels)
    filled = bytearray(len(buffer))
    queue: deque[tuple[int, int]] = deque()
    for x, y in sorted(mask.holes):
        total = 0
        count = 0
        for dx, dy in NEIGHBOURS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in mask.holes:
                total += image.pixels[ny * width + nx]
                count += 1
        if count:
            buffer[y * width + x] = (total + count // 2) // count
            filled[y * width + x] = 1
            queue.append((x, y))

    while queue:
        x, y = queue.popleft()
        value = buffer[y * width + x]
        for dx, dy in NEIGHBOURS:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            index = ny * width + nx
            if (nx, ny) in mask.holes and not filled[index]:
                buffer[index] = value
                filled[index] = 1
                queue.append((nx, ny))

    for x, y in mask.holes:
        index = y * width + x
        if not filled[index]:
            buffer[index] = _known_mean(image, mask)
    return buffer


def _diffusion_buffer(image: Gray, mask: Mask, iterations: int) -> bytearray:
    """Jacobi relaxation on the hole pixels: each becomes the mean of its four neighbours."""
    current = _nearest_buffer(image, mask)
    if not mask.holes or iterations <= 0:
        return current
    width, height = image.width, image.height
    holes = sorted(mask.holes)
    for _ in range(iterations):
        updated = bytearray(current)
        for x, y in holes:
            total = 0
            count = 0
            for dx, dy in NEIGHBOURS:
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    total += current[ny * width + nx]
                    count += 1
            updated[y * width + x] = (total + count // 2) // count
        current = updated
    return current


def _median_buffer(image: Gray, mask: Mask, iterations: int) -> bytearray:
    """Median of the already-restored pixels in a 5x5 window, working inward from the rim."""
    fallback = _diffusion_buffer(image, mask, iterations)
    width, height = image.width, image.height
    buffer = bytearray(image.pixels)
    known = bytearray(len(buffer))
    for y in range(height):
        for x in range(width):
            if (x, y) not in mask.holes:
                known[y * width + x] = 1

    pending = sorted(mask.holes)
    while pending:
        remaining: list[tuple[int, int]] = []
        for x, y in pending:
            values = []
            for dy in range(-MEDIAN_RADIUS, MEDIAN_RADIUS + 1):
                for dx in range(-MEDIAN_RADIUS, MEDIAN_RADIUS + 1):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height and known[ny * width + nx]:
                        values.append(buffer[ny * width + nx])
            if len(values) >= MIN_MEDIAN_SAMPLES:
                values.sort()
                buffer[y * width + x] = values[len(values) // 2]
                known[y * width + x] = 1
            else:
                remaining.append((x, y))
        if len(remaining) == len(pending):
            break
        pending = remaining

    for x, y in pending:
        buffer[y * width + x] = fallback[y * width + x]
    return buffer


def nearest(image: Gray, mask: Mask, iterations: int = 0) -> Gray:
    """Copy from the nearest known pixel. Sharp, striped, and a hard upper bound on cheapness."""
    _check(image, mask, iterations)
    return Gray(image.width, image.height, bytes(_nearest_buffer(image, mask)))


def diffusion(image: Gray, mask: Mask, iterations: int = 30) -> Gray:
    """Smooth the nearest fill towards a harmonic function of the known boundary."""
    _check(image, mask, iterations)
    return Gray(image.width, image.height, bytes(_diffusion_buffer(image, mask, iterations)))


def median(image: Gray, mask: Mask, iterations: int = 30) -> Gray:
    """Median fill: keeps an edge crossed by the hole instead of blurring it away."""
    _check(image, mask, iterations)
    return Gray(image.width, image.height, bytes(_median_buffer(image, mask, iterations)))


METHODS: dict[str, Callable[[Gray, Mask, int], Gray]] = {
    "nearest": nearest,
    "diffusion": diffusion,
    "median": median,
}


def restore(
    image: Gray, mask: Mask, method: str = "diffusion", iterations: int = 30
) -> Gray:
    """Dispatch on ``method``; ``iterations`` only matters for diffusion and median."""
    if method not in METHODS:
        choices = ", ".join(METHOD_NAMES)
        raise RestoreError(f"unknown method {method!r}; choose from {choices}")
    _check(image, mask, iterations)
    return METHODS[method](image, mask, iterations)
