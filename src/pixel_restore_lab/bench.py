"""Runs: one restoration, a ratio sweep, an iteration curve, and the README panels."""

from __future__ import annotations

from dataclasses import dataclass

from .image import Gray, synthetic
from .masks import Mask, build_mask, damage
from .metrics import masked_rmse, psnr
from .restore import METHOD_NAMES, restore

IMAGE_SEED = 11
MASK_SEED = 3
RATIOS: tuple[float, ...] = (0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
ITERATIONS: tuple[int, ...] = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 80, 100, 120)


@dataclass(frozen=True)
class MethodScore:
    method: str
    psnr: float
    hole_rmse: float
    gain: float

    def as_dict(self) -> dict:
        return {
            "method": self.method,
            "psnr": round(self.psnr, 3),
            "hole_rmse": round(self.hole_rmse, 3),
            "gain": round(self.gain, 3),
        }


@dataclass(frozen=True)
class BenchResult:
    size: int
    mask_kind: str
    ratio: float
    coverage: float
    iterations: int
    damaged_psnr: float
    damaged_hole_rmse: float
    scores: tuple[MethodScore, ...]

    @property
    def holes(self) -> int:
        return round(self.coverage * self.size * self.size)

    def as_dict(self) -> dict:
        return {
            "size": self.size,
            "mask": self.mask_kind,
            "ratio": self.ratio,
            "coverage": round(self.coverage, 4),
            "iterations": self.iterations,
            "damaged": {
                "psnr": round(self.damaged_psnr, 3),
                "hole_rmse": round(self.damaged_hole_rmse, 3),
            },
            "methods": [score.as_dict() for score in self.scores],
        }


def prepare(
    size: int = 128,
    ratio: float = 0.1,
    kind: str = "scratch",
    seed: int = IMAGE_SEED,
) -> tuple[Gray, Mask, Gray]:
    """Original scene, the mask, and the damaged image the restorer is handed."""
    original = synthetic(size, seed=seed)
    mask = build_mask(kind, size, ratio, seed=MASK_SEED)
    return original, mask, damage(original, mask)


def run_bench(
    size: int = 128,
    ratio: float = 0.1,
    kind: str = "scratch",
    iterations: int = 30,
    seed: int = IMAGE_SEED,
) -> BenchResult:
    original, mask, damaged = prepare(size, ratio, kind, seed)
    baseline = psnr(damaged, original)
    scores = []
    for method in METHOD_NAMES:
        restored = restore(damaged, mask, method=method, iterations=iterations)
        value = psnr(restored, original)
        scores.append(
            MethodScore(
                method=method,
                psnr=value,
                hole_rmse=masked_rmse(restored, original, mask),
                gain=value - baseline,
            )
        )
    return BenchResult(
        size=size,
        mask_kind=kind,
        ratio=ratio,
        coverage=mask.coverage,
        iterations=iterations,
        damaged_psnr=baseline,
        damaged_hole_rmse=masked_rmse(damaged, original, mask),
        scores=tuple(scores),
    )


def sweep_rows(
    size: int = 128, iterations: int = 30, kind: str = "scratch", seed: int = IMAGE_SEED
) -> list[dict]:
    """PSNR per method as the holes grow from 2% to 30% of the frame."""
    rows = []
    for ratio in RATIOS:
        original, mask, damaged = prepare(size, ratio, kind, seed)
        row = {
            "ratio": round(ratio, 3),
            "coverage": round(mask.coverage, 4),
            "damaged": round(psnr(damaged, original), 3),
        }
        for method in METHOD_NAMES:
            restored = restore(damaged, mask, method=method, iterations=iterations)
            row[method] = round(psnr(restored, original), 3)
        rows.append(row)
    return rows


def curve_rows(
    size: int = 128, ratio: float = 0.1, kind: str = "scratch", seed: int = IMAGE_SEED
) -> list[dict]:
    """PSNR against the number of Jacobi passes."""
    original, mask, damaged = prepare(size, ratio, kind, seed)
    baseline = psnr(damaged, original)
    nearest_psnr = psnr(restore(damaged, mask, method="nearest"), original)
    rows = []
    for iterations in ITERATIONS:
        row = {
            "iterations": iterations,
            "damaged": round(baseline, 3),
            "nearest": round(nearest_psnr, 3),
        }
        for method in ("diffusion", "median"):
            restored = restore(damaged, mask, method=method, iterations=iterations)
            row[method] = round(psnr(restored, original), 3)
        rows.append(row)
    return rows


def panels(
    size: int = 128,
    ratio: float = 0.1,
    kind: str = "scratch",
    iterations: int = 30,
    seed: int = IMAGE_SEED,
) -> dict[str, Gray]:
    """The four images the README shows, all produced by the real restorers."""
    original, mask, damaged = prepare(size, ratio, kind, seed)
    images = {"original": original, "damaged": damaged}
    for method in METHOD_NAMES:
        images[method] = restore(damaged, mask, method=method, iterations=iterations)
    return images
