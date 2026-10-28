import json

import pytest

from pixel_restore_lab import __version__
from pixel_restore_lab.bench import curve_rows, run_bench, sweep_rows
from pixel_restore_lab.cli import build_parser, main
from pixel_restore_lab.image import read_png_size
from pixel_restore_lab.report import (
    render_bench,
    render_curve,
    render_json,
    render_sweep,
    render_table,
)


@pytest.mark.parametrize("kind", ["scratch", "block", "scatter"])
def test_bench_prints_a_table_for_every_mask(kind, capsys):
    assert main(["bench", "--size", "32", "--mask", kind]) == 0
    out = capsys.readouterr().out
    assert "Method comparison" in out
    assert "nearest" in out and "diffusion" in out and "median" in out


def test_bench_json_carries_the_measurements(capsys):
    assert main(["bench", "--size", "32", "--format", "json", "--iterations", "5"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["size"] == 32
    assert payload["iterations"] == 5
    assert [entry["method"] for entry in payload["methods"]] == ["nearest", "diffusion", "median"]
    assert payload["damaged"]["psnr"] < payload["methods"][0]["psnr"]


def test_sweep_prints_the_ratio_curve(capsys):
    assert main(["sweep", "--size", "24"]) == 0
    out = capsys.readouterr().out
    assert "PSNR against how much of the frame is missing" in out
    assert out.count("%") >= 7


def test_curve_prints_the_iteration_curve(capsys):
    assert main(["curve", "--size", "48", "--format", "json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert [row["iterations"] for row in rows] == [
        1,
        2,
        3,
        4,
        6,
        8,
        12,
        16,
        24,
        32,
        48,
        64,
        80,
        100,
        120,
    ]
    assert rows[0]["diffusion"] < rows[6]["diffusion"]


def test_images_writes_damaged_and_restored_pngs(tmp_path, capsys):
    out = tmp_path / "panels"
    assert main(["images", "--out", str(out), "--size", "32", "--scale", "2"]) == 0
    written = sorted(path.name for path in out.iterdir())
    assert written == [
        "damaged.png",
        "diffusion.png",
        "median.png",
        "nearest.png",
        "original.png",
    ]
    for name in written:
        assert read_png_size(out / name) == (64, 64)
    assert capsys.readouterr().out.count("wrote") == 5


def test_images_rejects_a_zero_scale(tmp_path, capsys):
    assert main(["images", "--out", str(tmp_path), "--size", "24", "--scale", "0"]) == 2
    assert capsys.readouterr().err.startswith("error:")


def test_cli_reports_a_bad_ratio(capsys):
    assert main(["bench", "--size", "24", "--mask-ratio", "1.5"]) == 2
    assert "error:" in capsys.readouterr().err


def test_cli_reports_a_size_it_cannot_build(capsys):
    assert main(["bench", "--size", "8"]) == 2
    captured = capsys.readouterr()
    assert captured.err.startswith("error:")
    assert "at least 16" in captured.err


def test_cli_rejects_an_unknown_mask_kind():
    with pytest.raises(SystemExit) as excinfo:
        main(["bench", "--mask", "scribble"])
    assert excinfo.value.code == 2


def test_version_flag_exits_cleanly(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_parser_exposes_the_documented_subcommands():
    parser = build_parser()
    assert parser.prog == "pixel-restore"
    choices = [action for action in parser._actions if action.dest == "command"]
    assert choices and set(choices[0].choices) == {"bench", "sweep", "curve", "images"}


def test_reports_render_every_section():
    result = run_bench(size=24, iterations=4)
    rows = sweep_rows(size=24, iterations=4)
    curve = curve_rows(size=24)
    assert "Restoring a 24x24 synthetic scene" in render_bench(result)
    assert "gain dB" in render_table([score.as_dict() for score in result.scores])
    assert "nearest" in render_sweep(rows)
    assert "passes" in render_curve(curve)
    assert json.loads(render_json(result.as_dict()))["mask"] == "scratch"
