import math

import pytest

from pixel_restore_lab.bench import prepare, run_bench
from pixel_restore_lab.image import Gray, synthetic
from pixel_restore_lab.masks import (
    MASK_KINDS,
    Mask,
    MaskError,
    block_mask,
    build_mask,
    damage,
    scatter_mask,
    scratch_mask,
)
from pixel_restore_lab.metrics import masked_rmse, mse, psnr, rmse
from pixel_restore_lab.restore import METHOD_NAMES, RestoreError, restore

SIZE = 40
RATIO = 0.1
KINDS = sorted(MASK_KINDS)


def test_mask_coverage_is_close_to_the_request():
    for kind in KINDS:
        for ratio in (0.05, 0.1, 0.25):
            mask = build_mask(kind, SIZE, ratio)
            assert abs(mask.coverage - ratio) / ratio <= 0.10, (kind, ratio, mask.coverage)


def test_masks_are_deterministic():
    for kind in KINDS:
        first = build_mask(kind, SIZE, RATIO, seed=21)
        second = build_mask(kind, SIZE, RATIO, seed=21)
        other = build_mask(kind, SIZE, RATIO, seed=22)
        assert first.holes == second.holes
        assert first.holes != other.holes


def test_masks_stay_inside_the_frame():
    for kind in KINDS:
        mask = build_mask(kind, SIZE, 0.3)
        assert all(0 <= x < SIZE and 0 <= y < SIZE for x, y in mask.holes)
        assert (mask.width, mask.height) == (SIZE, SIZE)


def test_mask_rejects_holes_outside_the_frame():
    with pytest.raises(MaskError, match="outside"):
        Mask(4, 4, frozenset({(4, 0)}))


def test_mask_coverage_is_the_share_of_holes():
    mask = Mask(4, 4, frozenset({(0, 0), (1, 1)}))
    assert mask.coverage == pytest.approx(2 / 16)
    assert mask.contains(0, 0)
    assert not mask.contains(2, 2)


def test_mask_generators_reject_impossible_parameters():
    for ratio in (0.0, 1.5):
        with pytest.raises(MaskError, match="ratio"):
            scatter_mask(SIZE, ratio)
    with pytest.raises(MaskError, match="at least 8"):
        block_mask(4, RATIO)
    with pytest.raises(MaskError, match="unknown mask"):
        build_mask("scribble", SIZE, RATIO)


def test_damage_paints_the_holes_and_leaves_the_rest():
    original = synthetic(SIZE)
    mask = scratch_mask(SIZE, RATIO)
    damaged = damage(original, mask)
    assert all(damaged.pixels[y * SIZE + x] == 0 for x, y in mask.holes)
    assert all(
        damaged.pixels[y * SIZE + x] == original.pixels[y * SIZE + x]
        for y in range(SIZE)
        for x in range(SIZE)
        if (x, y) not in mask.holes
    )


def test_damage_rejects_a_size_mismatch():
    with pytest.raises(MaskError, match="mask is"):
        damage(synthetic(SIZE), scratch_mask(SIZE + 8, RATIO))


def test_nearest_reduces_the_hole_error():
    for kind in KINDS:
        original, mask, damaged = prepare(SIZE, RATIO, kind)
        restored = restore(damaged, mask, method="nearest")
        assert masked_rmse(restored, original, mask) < masked_rmse(damaged, original, mask)


def test_diffusion_reduces_the_hole_error():
    for kind in KINDS:
        original, mask, damaged = prepare(SIZE, RATIO, kind)
        restored = restore(damaged, mask, method="diffusion", iterations=30)
        assert masked_rmse(restored, original, mask) < masked_rmse(damaged, original, mask)


def test_median_reduces_the_hole_error():
    for kind in KINDS:
        original, mask, damaged = prepare(SIZE, RATIO, kind)
        restored = restore(damaged, mask, method="median", iterations=30)
        assert masked_rmse(restored, original, mask) < masked_rmse(damaged, original, mask)


def test_restore_dispatches_every_method():
    original, mask, damaged = prepare(SIZE, RATIO, "scratch")
    results = {
        method: restore(damaged, mask, method=method, iterations=5) for method in METHOD_NAMES
    }
    assert len({image.pixels for image in results.values()}) == 3
    assert all(image.pixels != original.pixels for image in results.values())


def test_restore_rejects_an_unknown_method():
    _original, mask, damaged = prepare(SIZE, RATIO, "scratch")
    with pytest.raises(RestoreError, match="unknown method"):
        restore(damaged, mask, method="telepathy")


def test_restore_rejects_bad_iteration_counts():
    _original, mask, damaged = prepare(SIZE, RATIO, "scratch")
    for iterations in (-1, -30):
        with pytest.raises(RestoreError, match="negative"):
            restore(damaged, mask, iterations=iterations)
    for iterations in (1.5, "10", None, True):
        with pytest.raises(RestoreError, match="integer"):
            restore(damaged, mask, iterations=iterations)


def test_restore_rejects_a_mask_of_another_size():
    _original, mask, damaged = prepare(SIZE, RATIO, "scratch")
    with pytest.raises(RestoreError, match="mask is"):
        restore(damaged, block_mask(SIZE + 8, RATIO), iterations=2)


def test_restore_ignores_the_pixels_it_replaces():
    original, mask, _damaged = prepare(SIZE, RATIO, "scratch")
    black = damage(original, mask, value=0)
    white = damage(original, mask, value=255)
    for method in METHOD_NAMES:
        assert (
            restore(black, mask, method=method, iterations=10).pixels
            == restore(white, mask, method=method, iterations=10).pixels
        )


def test_zero_diffusion_passes_equal_the_nearest_fill():
    _original, mask, damaged = prepare(SIZE, RATIO, "block")
    assert (
        restore(damaged, mask, method="diffusion", iterations=0).pixels
        == restore(damaged, mask, method="nearest").pixels
    )


def test_more_diffusion_passes_help_at_first_and_then_over_smooth():
    original, mask, damaged = prepare(128, RATIO, "scratch")
    one = psnr(restore(damaged, mask, method="diffusion", iterations=1), original)
    best = psnr(restore(damaged, mask, method="diffusion", iterations=12), original)
    many = psnr(restore(damaged, mask, method="diffusion", iterations=120), original)
    assert one < best
    assert many < best


def test_psnr_of_identical_images_is_infinite():
    image = synthetic(16)
    assert psnr(image, image) == float("inf")
    assert math.isinf(psnr(image, image))


def test_psnr_and_rmse_match_the_formula():
    left = Gray(4, 4, bytes(16))
    for wrong in (1, 4):
        right = Gray(4, 4, bytes([1] * wrong + [0] * (16 - wrong)))
        assert psnr(left, right) == pytest.approx(10 * math.log10(255**2 / (wrong / 16)))
        assert rmse(left, right) == pytest.approx(math.sqrt(wrong / 16))
    assert mse(Gray(2, 2, bytes(4)), Gray(2, 2, bytes([3, 0, 0, 0]))) == pytest.approx(9 / 4)


def test_psnr_rejects_images_of_different_sizes():
    with pytest.raises(ValueError, match="size mismatch"):
        psnr(Gray(2, 2, bytes(4)), Gray(4, 2, bytes(8)))


def test_masked_rmse_only_scores_the_holes():
    original = synthetic(SIZE)
    mask = Mask(SIZE, SIZE, frozenset({(2, 2), (3, 3)}))
    outside = original.with_pixel(10, 10, 255)
    inside = original.with_pixel(2, 2, 255)
    assert masked_rmse(outside, original, mask) == 0.0
    assert masked_rmse(inside, original, mask) == pytest.approx(
        abs(255 - original.get(2, 2)) / math.sqrt(2)
    )


def test_masked_rmse_rejects_an_empty_or_mismatched_mask():
    original = synthetic(16)
    with pytest.raises(ValueError, match="no holes"):
        masked_rmse(original, original, Mask(16, 16, frozenset()))
    with pytest.raises(ValueError, match="mask is"):
        masked_rmse(original, original, Mask(8, 8, frozenset({(0, 0)})))


def test_bench_numbers_are_reproducible():
    first = run_bench(size=32, ratio=RATIO)
    second = run_bench(size=32, ratio=RATIO)
    assert first.as_dict() == second.as_dict()
    assert [score.method for score in first.scores] == list(METHOD_NAMES)
    assert all(score.gain > 0 for score in first.scores)
    assert 0 < first.coverage <= 1
