"""grid CLI：假选项删除、单位统一（I-168）。

此前 `grid validate --threshold-area / --fix-duplicates / --fix-normals`、`grid parse --streaming` 接受参数却什么都不做；
`generate-volume`、`parse`、`validate`、`info`、`import-volume` 没有 --units（米制面网格被静默缩小 1000 倍）；面网格与
体网格解析各写一份单位判定；`grid info` 打印按 cell*44 + node*24 字节估出的"内存占用"。
"""

import json

import pytest
from click.testing import CliRunner
from loguru import logger

from autoflowcfd.cli.main import cli
from autoflowcfd.grid.nas_io.parser_core import NASParser
from autoflowcfd.grid.nas_io.units import resolve_units

_SURFACE = "GRID,1,,0.0,0.0,0.0\nGRID,2,,2.0,0.0,0.0\nGRID,3,,0.0,3.0,0.0\nCTRIA3,1,1,1,2,3\n"


@pytest.mark.parametrize("command, removed", [
    ("parse", ["--streaming"]),
    ("validate", ["--threshold-area", "--fix-duplicates", "--fix-normals"]),
])
def test_fake_options_are_gone(command, removed):
    help_text = CliRunner().invoke(cli, ["grid", command, "--help"]).output
    for option in removed:
        assert option not in help_text


@pytest.mark.parametrize("command", ["parse", "validate", "info", "generate-volume", "import-volume", "convert"])
def test_every_grid_command_takes_units(command):
    assert "--units" in CliRunner().invoke(cli, ["grid", command, "--help"]).output


def test_info_reports_the_bounding_box_in_meters_not_a_made_up_memory_figure(tmp_path):
    path = tmp_path / "s.nas"
    path.write_text(_SURFACE)
    result = CliRunner().invoke(cli, ["grid", "info", str(path), "--units", "m", "--json"])
    payload = json.loads(result.output[result.output.index("{"):])
    assert "estimated_memory_mb" not in payload
    assert payload["bounding_box_m"] == {"min": [0.0, 0.0, 0.0], "max": [2.0, 3.0, 0.0]}
    result = CliRunner().invoke(cli, ["grid", "info", str(path), "--json"])          # 缺省毫米
    assert json.loads(result.output[result.output.index("{"):])["bounding_box_m"]["max"] == [0.002, 0.003, 0.0]


def test_auto_units_is_resolved_once_for_both_parsers(tmp_path):
    path = tmp_path / "s.nas"
    path.write_text(_SURFACE)
    parser = NASParser(str(path), units="auto")
    parser.parse()
    assert parser.resolved_units == "m"
    assert resolve_units("auto", 4000.0) == ("mm", 1e-3)


def test_implausible_extent_after_conversion_warns():
    messages = []
    sink = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        resolve_units("mm", 5.0, source="x.nas")              # 5 mm 的计算域
    finally:
        logger.remove(sink)
    assert any("请检查 --units" in m for m in messages)
