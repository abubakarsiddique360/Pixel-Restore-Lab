"""Text reports for the restoration benchmarks."""

from __future__ import annotations

import json
import math

from .bench import BenchResult


def _db(value: float) -> str:
    if math.isinf(value):
        return "inf"
    return f"{value:.2f}"


def render_table(rows: list[dict], title: str = "Method comparison") -> str:
    """Aligned comparison of ``method``, ``psnr``, ``hole_rmse`` and ``gain``."""
    header = f"{'method':<12}{'psnr dB':>10}{'hole rmse':>12}{'gain dB':>10}"
    lines = [title, "=" * len(header), header]
    for row in rows:
        lines.append(
            f"{str(row['method'])[:11]:<12}{_db(row['psnr']):>10}"
            f"{row['hole_rmse']:>12.2f}{row['gain']:>+10.2f}"
        )
    lines.append("")
    lines.append("gain = dB over the damaged image; hole rmse ignores pixels outside the mask")
    return "\n".join(lines)


def render_bench(result: BenchResult) -> str:
    lines = [
        f"Restoring a {result.size}x{result.size} synthetic scene",
        "=" * 55,
        f"mask          {result.mask_kind}, {result.ratio * 100:.0f}% requested",
        f"coverage      {result.coverage * 100:.2f}% ({result.holes:,} hole pixels)",
        f"iterations    {result.iterations}",
        f"damaged       {_db(result.damaged_psnr)} dB  "
        f"(hole rmse {result.damaged_hole_rmse:.2f})",
        "",
        render_table([score.as_dict() for score in result.scores]),
    ]
    return "\n".join(lines)


def render_sweep(rows: list[dict]) -> str:
    header = (
        f"{'ratio':>7}{'coverage':>10}{'damaged':>10}{'nearest':>10}{'diffusion':>11}{'median':>9}"
    )
    lines = ["PSNR against how much of the frame is missing", "=" * len(header), header]
    for row in rows:
        lines.append(
            f"{row['ratio'] * 100:>6.0f}%{row['coverage'] * 100:>9.2f}%"
            f"{_db(row['damaged']):>10}{_db(row['nearest']):>10}"
            f"{_db(row['diffusion']):>11}{_db(row['median']):>9}"
        )
    lines.append("")
    lines.append("all values in dB, measured against the undamaged scene")
    return "\n".join(lines)


def render_curve(rows: list[dict]) -> str:
    header = f"{'passes':>8}{'damaged':>10}{'nearest':>10}{'diffusion':>11}{'median':>9}"
    lines = ["Diffusion passes against PSNR", "=" * len(header), header]
    for row in rows:
        lines.append(
            f"{row['iterations']:>8,}{_db(row['damaged']):>10}{_db(row['nearest']):>10}"
            f"{_db(row['diffusion']):>11}{_db(row['median']):>9}"
        )
    return "\n".join(lines)


def render_json(payload) -> str:
    return json.dumps(payload, indent=2)
