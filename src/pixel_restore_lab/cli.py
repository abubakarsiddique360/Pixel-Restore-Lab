"""Command line entry point for pixel-restore."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .bench import curve_rows, panels, run_bench, sweep_rows
from .image import ImageError, write_png
from .masks import MASK_KINDS, MaskError
from .report import render_bench, render_curve, render_json, render_sweep
from .restore import METHOD_NAMES, RestoreError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pixel-restore",
        description="Restore damaged grayscale images with classical inpainting and measure PSNR.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    bench = sub.add_parser("bench", help="damaged PSNR against every method, once")
    bench.add_argument("--size", type=int, default=128)
    bench.add_argument("--mask-ratio", type=float, default=0.1)
    bench.add_argument("--mask", choices=tuple(MASK_KINDS), default="scratch")
    bench.add_argument("--iterations", type=int, default=30)
    bench.add_argument("--format", choices=("text", "json"), default="text")

    sweep = sub.add_parser("sweep", help="PSNR against mask ratio")
    sweep.add_argument("--size", type=int, default=128)
    sweep.add_argument("--mask", choices=tuple(MASK_KINDS), default="scratch")
    sweep.add_argument("--iterations", type=int, default=30)
    sweep.add_argument("--format", choices=("text", "json"), default="text")

    curve = sub.add_parser("curve", help="PSNR against diffusion iterations")
    curve.add_argument("--size", type=int, default=128)
    curve.add_argument("--mask", choices=tuple(MASK_KINDS), default="scratch")
    curve.add_argument("--mask-ratio", type=float, default=0.1)
    curve.add_argument("--format", choices=("text", "json"), default="text")

    images = sub.add_parser("images", help="write the damaged and restored PNGs")
    images.add_argument("--out", required=True)
    images.add_argument("--size", type=int, default=128)
    images.add_argument("--mask-ratio", type=float, default=0.1)
    images.add_argument("--mask", choices=tuple(MASK_KINDS), default="scratch")
    images.add_argument("--iterations", type=int, default=30)
    images.add_argument("--scale", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "bench":
            result = run_bench(
                size=args.size,
                ratio=args.mask_ratio,
                kind=args.mask,
                iterations=args.iterations,
            )
            if args.format == "json":
                print(render_json(result.as_dict()))
            else:
                print(render_bench(result))
            return 0

        if args.command == "sweep":
            rows = sweep_rows(size=args.size, iterations=args.iterations, kind=args.mask)
            if args.format == "json":
                print(render_json(rows))
            else:
                print(render_sweep(rows))
            return 0

        if args.command == "curve":
            rows = curve_rows(size=args.size, ratio=args.mask_ratio, kind=args.mask)
            if args.format == "json":
                print(render_json(rows))
            else:
                print(render_curve(rows))
            return 0

        if args.command == "images":
            out = Path(args.out)
            figures = panels(
                size=args.size,
                ratio=args.mask_ratio,
                kind=args.mask,
                iterations=args.iterations,
            )
            for name in ("original", "damaged", *METHOD_NAMES):
                target = out / f"{name}.png"
                write_png(figures[name], target, scale=args.scale)
                print(
                    f"wrote {target}  {args.size}x{args.size} "
                    f"-> {args.size * args.scale}x{args.size * args.scale}"
                )
            return 0
    except (ImageError, MaskError, RestoreError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
