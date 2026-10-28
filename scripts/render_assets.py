"""Regenerate README assets from real runs.

Usage:  python scripts/render_assets.py
Writes: docs/results.json, docs/images/bench.png, docs/images/restore-panels.png,
        docs/images/psnr-curves.png and the individual PNGs in docs/images/panels/

Pillow and matplotlib are development-only. The library itself is standard library only, and every
PNG under docs/images/panels/ was written by pixel_restore_lab.image.write_png.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pixel_restore_lab.bench import curve_rows, panels, run_bench, sweep_rows  # noqa: E402
from pixel_restore_lab.image import read_png_size, write_png  # noqa: E402
from pixel_restore_lab.report import (  # noqa: E402
    render_bench,
    render_curve,
    render_sweep,
)

BG = "#0d1117"
CHROME = "#161b22"
FG = "#e6edf3"
MUTED = "#8b949e"
ACCENT = "#79c0ff"
SIZE = 128
RATIO = 0.1
ITERATIONS = 30
SCALE = 3
KINDS = ("scratch", "block", "scatter")
COLOURS = {"nearest": "#cf222e", "diffusion": "#1a7f37", "median": "#8250df"}


def load_font(size: int) -> ImageFont.FreeTypeFont:
    for name in ("consola.ttf", "courier.ttf", "DejaVuSansMono.ttf"):
        candidate = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / name
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def render_terminal(text: str, out_path: Path, title: str) -> None:
    font = load_font(15)
    lines = text.expandtabs(4).splitlines() or [""]
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    char_width = probe.textlength("M", font=font)
    line_height = 21
    pad = 24
    width = int(max(len(line) for line in lines) * char_width) + pad * 2
    height = pad * 2 + len(lines) * line_height + 44
    image = Image.new("RGB", (max(width, 800), height), BG)
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, image.width, 34], fill=CHROME)
    for index, color in enumerate(("#ff5f56", "#ffbd2e", "#27c93f")):
        draw.ellipse([16 + index * 22, 10, 28 + index * 22, 22], fill=color)
    draw.text((92, 9), title, font=load_font(13), fill=MUTED)

    for row, line in enumerate(lines):
        color = FG
        stripped = line.strip()
        if line.startswith("$"):
            color = ACCENT
        elif stripped.startswith("nearest"):
            color = COLOURS["nearest"]
        elif stripped.startswith("diffusion"):
            color = COLOURS["diffusion"]
        elif stripped.startswith("median"):
            color = COLOURS["median"]
        elif stripped.startswith(("mask", "coverage", "iterations", "damaged", "Restoring")):
            color = FG
        elif stripped.startswith(("=", "-")) or "comparison" in stripped.lower():
            color = MUTED
        draw.text((pad, 50 + row * line_height), line, font=font, fill=color)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def render_panels(out_path: Path, panel_dir: Path, figures: dict, bench) -> None:
    scores = {score.method: score.psnr for score in bench.scores}
    labels = [
        ("original", "original"),
        ("damaged", f"damaged: {bench.ratio:.0%} of the frame"),
        ("nearest", f"nearest fill: {scores['nearest']:.2f} dB"),
        ("diffusion", f"diffusion x{bench.iterations}: {scores['diffusion']:.2f} dB"),
    ]
    figure, axes = plt.subplots(1, 4, figsize=(12.8, 3.6))
    figure.patch.set_facecolor("#ffffff")
    for axis, (key, label) in zip(axes, labels, strict=True):
        target = panel_dir / f"{key}.png"
        write_png(figures[key], target, scale=SCALE)
        assert read_png_size(target) == (SIZE * SCALE, SIZE * SCALE)
        with Image.open(target) as handle:
            axis.imshow(handle.convert("L"), cmap="gray", vmin=0, vmax=255)
        axis.set_title(label, fontsize=10)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
    figure.suptitle(
        "One scratch, three fills: the gouge is flat black, the repairs are invented",
        fontsize=11,
        x=0.01,
        ha="left",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    figure.savefig(out_path, dpi=150)
    plt.close(figure)


def render_curves(sweep: list[dict], curve: list[dict], out_path: Path) -> None:
    figure, (left, right) = plt.subplots(1, 2, figsize=(11.8, 4.1))
    figure.patch.set_facecolor("#ffffff")

    ratios = [row["ratio"] * 100 for row in sweep]
    left.plot(ratios, [row["damaged"] for row in sweep], linestyle="--", color=MUTED,
              label="damaged")
    for method, colour in COLOURS.items():
        left.plot(ratios, [row[method] for row in sweep], marker="o", color=colour,
                  label=method)
    left.set_xlabel("% of the frame masked")
    left.set_ylabel("PSNR against the original (dB)")
    left.set_title("Holes cost every method, and the gaps between them widen", fontsize=9,
                   loc="left")
    left.grid(linestyle=":", alpha=0.45)
    left.legend(fontsize=8, frameon=False)

    passes = [row["iterations"] for row in curve]
    right.axhline(curve[0]["nearest"], linestyle="--", color=COLOURS["nearest"],
                  label="nearest (no passes)")
    right.plot(passes, [row["diffusion"] for row in curve], marker="o", color=COLOURS["diffusion"],
               label="diffusion")
    right.plot(passes, [row["median"] for row in curve], color=COLOURS["median"], label="median")
    right.set_xscale("log", base=2)
    right.set_xticks(passes)
    right.set_xticklabels([str(value) for value in passes], fontsize=6)
    right.set_ylim(30.8, 36.4)
    right.set_xlabel("diffusion passes (log scale)")
    right.set_ylabel("PSNR (dB)")
    right.set_title("Diffusion peaks, then erases the texture it was meant to keep", fontsize=9,
                    loc="left")
    right.grid(linestyle=":", alpha=0.45)
    right.legend(fontsize=8, frameon=False, loc="upper left")

    for axis in (left, right):
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
    figure.tight_layout()
    figure.savefig(out_path, dpi=150)
    plt.close(figure)


def main() -> int:
    images = ROOT / "docs" / "images"
    panels_dir = images / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)

    bench = run_bench(size=SIZE, ratio=RATIO, kind="scratch", iterations=ITERATIONS)
    by_mask = {
        kind: run_bench(size=SIZE, ratio=RATIO, kind=kind, iterations=ITERATIONS)
        for kind in KINDS
    }
    sweep = sweep_rows(size=SIZE, iterations=ITERATIONS)
    curve = curve_rows(size=SIZE, ratio=RATIO)

    payload = {
        "size": SIZE,
        "ratio": RATIO,
        "iterations": ITERATIONS,
        "bench": bench.as_dict(),
        "bench_by_mask": {kind: result.as_dict() for kind, result in by_mask.items()},
        "sweep": sweep,
        "curve": curve,
    }
    (ROOT / "docs" / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    terminal = (
        f"$ pixel-restore bench --size {SIZE} --mask-ratio {RATIO:.2f}\n\n"
        + render_bench(bench)
        + f"\n\n$ pixel-restore sweep --size {SIZE}\n\n"
        + render_sweep(sweep)
        + f"\n\n$ pixel-restore curve --size {SIZE}\n\n"
        + render_curve(curve[:8])
    )
    render_terminal(terminal, images / "bench.png", "pixel-restore  -  classical inpainting")
    render_panels(images / "restore-panels.png", panels_dir,
                  panels(size=SIZE, ratio=RATIO, kind="scratch", iterations=ITERATIONS), bench)
    render_curves(sweep, curve, images / "psnr-curves.png")

    scores = {score.method: score.psnr for score in bench.scores}
    print(
        f"scratch {RATIO:.0%}: damaged {bench.damaged_psnr:.2f} dB | "
        f"nearest {scores['nearest']:.2f} | diffusion {scores['diffusion']:.2f} | "
        f"median {scores['median']:.2f}"
    )
    print(
        "block: "
        + ", ".join(f"{score.method} {score.psnr:.2f}" for score in by_mask["block"].scores)
    )
    print(
        "scatter: "
        + ", ".join(f"{score.method} {score.psnr:.2f}" for score in by_mask["scatter"].scores)
    )
    print(f"peak diffusion at {max(curve, key=lambda row: row['diffusion'])['iterations']} passes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
