"""命令行帮助只写面向用户的现行语义（I-193）：不出现开发史（日期、"此前"、评审编号、ProjectFiles 引用）与开发者用的
Args / Returns / Raises 段；示例按原样换行。顶层帮助与实际流程一致（体网格生成导出 .nas，求解可直接读它）。"""

import re

import click
import pytest
from click.testing import CliRunner

from autoflowcfd.cli.main import cli

_HISTORY = re.compile(r"20\d\d-\d\d|此前|以前|ProjectFiles|Part\d+|I-\d+\b|CL-\d+|评审")


def _all_commands():
    for gname, group in cli.commands.items():
        if isinstance(group, click.Group):
            for cname in group.commands:
                yield [gname, cname]


@pytest.mark.parametrize("path", list(_all_commands()), ids=lambda p: " ".join(p))
def test_help_has_no_development_history_or_developer_sections(path):
    text = CliRunner().invoke(cli, path + ["--help"]).output
    assert not _HISTORY.search(text), (path, _HISTORY.search(text).group(0))
    assert not re.search(r"^\s+(Args|Returns|Raises):", text, re.M), path


def test_top_level_help_matches_the_workflow():
    text = CliRunner().invoke(cli, ["--help"]).output
    assert "generate-volume" in text and "--surface-mesh" in text
    assert "必须为 .pkl" not in text and "漫流" not in text


def test_examples_keep_their_line_breaks():
    text = CliRunner().invoke(cli, ["post", "export-vtk", "--help"]).output
    lines = [ln.strip() for ln in text.splitlines()]
    assert "$ autoflowcfd post export-vtk --case results -o flow_ho.vtu --high-order" in lines
    assert not any("--variables velocity pressure" in ln for ln in lines)    # multiple 选项须重复给出


def test_groups_help_names_the_actual_default():
    text = CliRunner().invoke(cli, ["post", "coefficients", "--help"]).output
    assert "物理无滑移固壁" in text and "全部 WALL 组" not in text
