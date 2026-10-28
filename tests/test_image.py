import struct
import zlib

import pytest

from pixel_restore_lab.image import (
    PNG_SIGNATURE,
    Gray,
    ImageError,
    energy,
    read_png_size,
    synthetic,
    upscale,
    write_png,
)


def png_chunks(data: bytes) -> list[tuple[bytes, bytes]]:
    """Split a PNG into chunks and verify every CRC on the way through."""
    assert data.startswith(PNG_SIGNATURE)
    chunks = []
    offset = len(PNG_SIGNATURE)
    while offset < len(data):
        (length,) = struct.unpack(">I", data[offset : offset + 4])
        tag = data[offset + 4 : offset + 8]
        payload = data[offset + 8 : offset + 8 + length]
        (crc,) = struct.unpack(">I", data[offset + 8 + length : offset + 12 + length])
        assert crc == zlib.crc32(tag + payload) & 0xFFFFFFFF
        chunks.append((tag, payload))
        offset += 12 + length
    return chunks


def test_gray_reads_pixels_row_major():
    image = Gray(3, 2, bytes([1, 2, 3, 4, 5, 6]))
    assert image.area == 6
    assert [image.get(x, 0) for x in range(3)] == [1, 2, 3]
    assert [image.get(x, 1) for x in range(3)] == [4, 5, 6]
    assert image.as_rows() == [bytes([1, 2, 3]), bytes([4, 5, 6])]


def test_gray_rejects_a_pixel_count_mismatch():
    with pytest.raises(ImageError, match="expected 6 pixels"):
        Gray(3, 2, bytes(5))


@pytest.mark.parametrize("width, height", [(0, 4), (4, -1)])
def test_gray_rejects_non_positive_dimensions(width, height):
    with pytest.raises(ImageError, match="dimensions"):
        Gray(width, height, b"")


def test_get_rejects_out_of_bounds_pixels():
    image = Gray(2, 2, bytes(4))
    with pytest.raises(ImageError, match="outside"):
        image.get(2, 0)
    with pytest.raises(ImageError, match="outside"):
        image.get(0, -1)


def test_with_pixel_returns_a_copy_and_rejects_bad_values():
    image = Gray(2, 2, bytes(4))
    changed = image.with_pixel(1, 1, 200)
    assert changed.get(1, 1) == 200
    assert image.get(1, 1) == 0
    with pytest.raises(ImageError, match="between 0 and 255"):
        image.with_pixel(0, 0, 256)


def test_synthetic_is_deterministic():
    assert synthetic(32) == synthetic(32)
    assert synthetic(32, seed=11) == synthetic(32, seed=11)


def test_synthetic_depends_on_the_seed():
    assert synthetic(32, seed=5) != synthetic(32, seed=6)


def test_synthetic_has_smooth_regions_and_hard_edges():
    image = synthetic(64)
    gradients = energy(image)
    average = sum(gradients) / len(gradients)
    assert len(set(image.pixels)) > 30
    assert max(gradients) > 40
    assert 10 < average < 30


def test_synthetic_rejects_tiny_frames():
    with pytest.raises(ImageError, match="at least 16"):
        synthetic(8)


def test_energy_matches_a_hand_computed_gradient():
    assert energy(Gray(3, 1, bytes([10, 20, 40]))) == [10, 30, 20]
    assert set(energy(Gray(4, 4, bytes(16)))) == {0}


def test_png_has_signature_ihdr_scanlines_and_iend(tmp_path):
    target = tmp_path / "flat.png"
    write_png(Gray(5, 3, bytes(range(15))), target, scale=1)
    chunks = png_chunks(target.read_bytes())
    assert [tag for tag, _payload in chunks] == [b"IHDR", b"IDAT", b"IEND"]
    (payload,) = [body for tag, body in chunks if tag == b"IDAT"]
    raw = zlib.decompress(payload)
    assert len(raw) == 3 * (5 + 1)
    assert [raw[row * 6] for row in range(3)] == [0, 0, 0]
    assert raw[1:6] == bytes(range(5))


@pytest.mark.parametrize("width, height", [(3, 2), (64, 48)])
def test_png_ihdr_reports_the_dimensions(tmp_path, width, height):
    target = tmp_path / "sized.png"
    write_png(Gray(width, height, bytes(width * height)), target, scale=1)
    assert read_png_size(target) == (width, height)
    header = png_chunks(target.read_bytes())[0][1]
    numbers = struct.unpack(">IIBBBBB", header)
    assert numbers[:2] == (width, height)
    assert numbers[2:4] == (8, 0)
    assert numbers[4:] == (0, 0, 0)


def test_write_png_upscales_by_nearest_neighbour(tmp_path):
    target = tmp_path / "big.png"
    write_png(Gray(2, 2, bytes([0, 100, 200, 255])), target, scale=3)
    assert read_png_size(target) == (6, 6)
    (payload,) = [body for tag, body in png_chunks(target.read_bytes()) if tag == b"IDAT"]
    raw = zlib.decompress(payload)
    first_row = raw[1:7]
    assert first_row == bytes([0, 0, 0, 100, 100, 100])
    assert raw[8:14] == first_row


def test_write_png_creates_missing_directories(tmp_path):
    target = tmp_path / "nested" / "deeper" / "image.png"
    write_png(Gray(4, 4, bytes(16)), target, scale=2)
    assert target.exists()


def test_upscale_rejects_zero_scale_but_passes_one_through():
    image = Gray(2, 2, bytes([1, 2, 3, 4]))
    assert upscale(image, 1) is image
    with pytest.raises(ImageError, match="scale"):
        upscale(image, 0)


def test_read_png_size_rejects_foreign_and_truncated_files(tmp_path):
    foreign = tmp_path / "not-a-png.bin"
    foreign.write_bytes(b"GIF89a" + bytes(64))
    with pytest.raises(ImageError, match="signature"):
        read_png_size(foreign)

    short = tmp_path / "short.png"
    short.write_bytes(PNG_SIGNATURE + bytes(4))
    with pytest.raises(ImageError, match="ends before IHDR"):
        read_png_size(short)
