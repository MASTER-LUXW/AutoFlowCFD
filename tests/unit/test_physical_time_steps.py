"""瞬态总步数的唯一换算 `physical_time_steps`。

此前 CLI、配置层、API 各写一份 `int(total_time / dt)`：`int(0.3 / 0.1) == 2`、`int(0.7 / 0.1) == 6`，
常见输入少走一步、到不了要求的物理时刻。
"""

import math
import pathlib
import re

import pytest

from autoflowcfd.config.solver_config import TransientConfig
from autoflowcfd.core.time_integration.base import physical_time_steps


@pytest.mark.parametrize("total_time, dt, expected", [
    (0.3, 0.1, 3), (0.7, 0.1, 7), (1.0, 0.1, 10), (0.06, 0.002, 30), (0.1, 1e-4, 1000), (2.9, 0.1, 29),
])
def test_integer_multiples_are_not_truncated(total_time, dt, expected):
    assert physical_time_steps(total_time, dt) == expected


def test_non_multiple_reaches_requested_time():
    n = physical_time_steps(0.1, 3e-5)
    assert n == math.ceil(0.1 / 3e-5)
    assert n * 3e-5 >= 0.1 and (n - 1) * 3e-5 < 0.1


@pytest.mark.parametrize("total_time, dt", [(0.0, 0.1), (-1.0, 0.1), (1.0, 0.0), (1.0, -0.1)])
def test_non_positive_inputs_raise(total_time, dt):
    with pytest.raises(ValueError):
        physical_time_steps(total_time, dt)


def test_transient_config_uses_shared_conversion():
    assert TransientConfig(total_time=0.3, dt=0.1).total_steps == 3


def test_no_truncating_conversions_left_in_source():
    """源码里不得再出现 `int(<时间> / dt)` 这类截断换算（统一走 physical_time_steps）。"""
    root = pathlib.Path(__file__).resolve().parents[2] / "src" / "autoflowcfd"
    pattern = re.compile(r"int\(\s*(?:float\()?[\w.]*time[\w.]*\)?\s*/\s*[\w.]*dt\s*\)")
    hits = [f"{p.relative_to(root)}:{i}" for p in root.rglob("*.py")
            for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1) if pattern.search(line)]
    assert not hits, hits
