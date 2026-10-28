"""A greyscale raster that lives in one ``bytes`` object, plus a PNG writer with no dependencies.

Everything this lab needs from an image file is a header and a zlib stream, so the encoder is a few
dozen lines of ``struct`` and ``zlib``. That keeps the runtime dependency-free and makes the
committed PNGs reproducible byte for byte.
"""

from __future__ import annotations

import random
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
BIT_DEPTH = 8
COLOUR_TYPE_GREY = 0


class ImageError(ValueError):
    """Raised when a raster or a PNG file is malformed."""


@dataclass(frozen=True)
class Gray:
    """An 8-bit greyscale image: one byte per pixel, row-major, top-left origin."""

    width: int
    height: int
    pixels: bytes

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ImageError("image dimensions must be positive")
        if len(self.pixels) != self.width * self.height:
            raise ImageError(
                f"expected {self.width * self.height} pixels, got {len(self.pixels)}"
            )

    @property
    def area(self) -> int:
        return self.width * self.height

    def get(self, x: int, y: int) -> int:
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise ImageError(f"pixel ({x}, {y}) is outside a {self.width}x{self.height} image")
        return self.pixels[y * self.width + x]

    def with_pixel(self, x: int, y: int, value: int) -> Gray:
        if not 0 <= value <= 255:
            raise ImageError("pixel values must be between 0 and 255")
        index = y * self.width + x
        self.get(x, y)
        return Gray(
            self.width,
            self.height,
            self.pixels[:index] + bytes((value,)) + self.pixels[index + 1 :],
        )

    def as_rows(self) -> list[bytes]:
        return [
            self.pixels[row * self.width : (row + 1) * self.width] for row in range(self.height)
        ]


def energy(image: Gray) -> list[int]:
    """Per-pixel gradient magnitude (|dx| + |dy|).

    This is the structure an inpainter can exploit: a smooth region can be interpolated almost
    exactly, an edge cannot. The synthetic scene is built to have both.
    """
    width, height = image.width, image.height
    pixels = image.pixels
    magnitudes = [0] * (width * height)
    for y in range(height):
        row = y * width
        for x in range(width):
            centre = pixels[row + x]
            left = pixels[row + x - 1] if x > 0 else centre
            right = pixels[row + x + 1] if x + 1 < width else centre
            up = pixels[row - width + x] if y > 0 else centre
            down = pixels[row + width + x] if y + 1 < height else centre
            magnitudes[row + x] = abs(right - left) + abs(down - up)
    return magnitudes


def upscale(image: Gray, scale: int) -> Gray:
    """Nearest-neighbour blow-up, so 128-pixel panels are legible in a README."""
    if scale < 1:
        raise ImageError("scale must be at least 1")
    if scale == 1:
        return image
    rows: list[bytes] = []
    for row in image.as_rows():
        wide = bytes(value for value in row for _ in range(scale))
        rows.extend([wide] * scale)
    return Gray(image.width * scale, image.height * scale, b"".join(rows))


def _chunk(tag: bytes, payload: bytes) -> bytes:
    crc = zlib.crc32(tag + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", crc)


def _ihdr(width: int, height: int) -> bytes:
    payload = struct.pack(">IIBBBBB", width, height, BIT_DEPTH, COLOUR_TYPE_GREY, 0, 0, 0)
    return _chunk(b"IHDR", payload)


def encode_png(image: Gray) -> bytes:
    """8-bit greyscale PNG: signature, IHDR, one IDAT of filter-0 scanlines, IEND."""
    scanlines = b"".join(b"\x00" + row for row in image.as_rows())
    return (
        PNG_SIGNATURE
        + _ihdr(image.width, image.height)
        + _chunk(b"IDAT", zlib.compress(scanlines, 9))
        + _chunk(b"IEND", b"")
    )


def write_png(image: Gray, path: str | Path, scale: int = 4) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(encode_png(upscale(image, scale)))


def read_png_size(path: str | Path) -> tuple[int, int]:
    """Parse the IHDR chunk and return ``(width, height)``.

    Proving the dimensions came out of a real IHDR (signature, length, CRC and all) is the
    dependency-free version of "Pillow can open it".
    """
    data = Path(path).read_bytes()
    if not data.startswith(PNG_SIGNATURE):
        raise ImageError("not a PNG: signature mismatch")
    if len(data) < 33:
        raise ImageError("not a PNG: file ends before IHDR is complete")
    (length, tag) = struct.unpack(">I4s", data[8:16])
    if tag != b"IHDR":
        raise ImageError(f"not a PNG: first chunk is {tag!r}, expected IHDR")
    if length != 13:
        raise ImageError("corrupt PNG: IHDR must be 13 bytes")
    width, height, depth, colour, compression, filter_method, interlace = struct.unpack(
        ">IIBBBBB", data[16:29]
    )
    (crc,) = struct.unpack(">I", data[29:33])
    if crc != zlib.crc32(data[12:29]) & 0xFFFFFFFF:
        raise ImageError("corrupt PNG: IHDR checksum mismatch")
    if depth != BIT_DEPTH or colour != COLOUR_TYPE_GREY:
        raise ImageError("only 8-bit greyscale PNGs are supported")
    if compression or filter_method or interlace:
        raise ImageError("PNG compression, filter and interlace flags must be zero")
    if width <= 0 or height <= 0:
        raise ImageError("PNG dimensions must be positive")
    return width, height


def synthetic(size: int = 128, seed: int = 11) -> Gray:
    """A deterministic little scene: sky gradient, sun disc, a wall, a road, light noise.

    Inpainting needs structure to exploit, so this is deliberately not random noise: there are
    smooth regions (where any fill interpolates well) and hard edges (where it does not).
    """
    if size < 16:
        raise ImageError("synthetic scenes need at least 16 pixels per side")
    noise = random.Random(seed)
    pixels = bytearray(size * size)
    horizon = int(size * 0.62)
    disc_cx, disc_cy = int(size * 0.72), int(size * 0.22)
    disc_radius = max(3, size // 10)
    wall_x0, wall_y0 = int(size * 0.10), int(size * 0.30)
    wall_x1, wall_y1 = int(size * 0.40), int(size * 0.74)
    band_half = max(1.0, size * 0.06)
    for y in range(size):
        for x in range(size):
            if y < horizon:
                value = 58.0 + 128.0 * (horizon - y) / horizon
            else:
                value = 52.0 + 46.0 * (size - y) / max(1, size - horizon)
            offset = y - (size * 0.80 + 0.45 * (x - size * 0.5))
            if abs(offset) <= band_half:
                value = 196.0 - 34.0 * abs(offset) / band_half
            if wall_x0 <= x < wall_x1 and wall_y0 <= y < wall_y1:
                value = 64.0 + 26.0 * (x - wall_x0) / (wall_x1 - wall_x0)
                if x - wall_x0 < 3:
                    value += 92.0
            if (x - disc_cx) ** 2 + (y - disc_cy) ** 2 <= disc_radius**2:
                value = 232.0
            value += noise.randint(-4, 4)
            pixels[y * size + x] = max(0, min(255, int(round(value))))
    return Gray(size, size, bytes(pixels))
