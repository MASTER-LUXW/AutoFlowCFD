# -*- coding: utf-8 -*-
"""mach_ref 的唯一来源：`core/fr_solver/mach_ref.py::resolve_mach_ref`。

2026-09-25 之前有 9 份手写：`FRSolver` 按 AUSM+up 预处理档分派下限
（默认档 0.05），单机 GPU / 多 GPU 构造函数硬编码 0.1，CLI 里 6 处构造
"完全分布式加载"求解包用 legacy 档别名 `_MACH_REF_FLOOR`（也是 0.1）——
同一算例（plate_demo，M=0.0882）在单机 CPU 上 mach_ref=0.0882、在其余路径上
0.1，AUSM+up 预处理与 CFL 都不同，解的是不同的离散问题。
"""

import pathlib
import re

import pytest

from autoflowcfd.core.fr_solver.mach_ref import (
    _MACH_REF_FLOOR_LEGACY,
    _MACH_REF_FLOOR_PHYSICAL,
    resolve_mach_ref,
)
from autoflowcfd.core.fr_operators.kernels import PRECOND_LEGACY, PRECOND_PHYSICAL


def test_plate_demo_value_depends_on_mode_only_through_the_floor():
    m_true = 30.0 / (1.4 * 101325.0 / 1.225) ** 0.5
    assert resolve_mach_ref(1.225, 30.0, 101325.0, PRECOND_PHYSICAL) == pytest.approx(m_true)
    assert resolve_mach_ref(1.225, 30.0, 101325.0, PRECOND_LEGACY) == pytest.approx(_MACH_REF_FLOOR_LEGACY)
    assert resolve_mach_ref(1.225, 1.0, 101325.0, PRECOND_PHYSICAL) == pytest.approx(_MACH_REF_FLOOR_PHYSICAL)


def test_no_hand_rolled_mach_formula_or_legacy_alias_elsewhere():
    root = pathlib.Path(__file__).resolve().parents[2] / "src" / "autoflowcfd"
    pat = re.compile(r"1\.4\s*\*\s*p_inf\s*/\s*max\(rho_inf|_MACH_REF_FLOOR\b")
    offenders = []
    for p in root.rglob("*.py"):
        rel = p.relative_to(root).as_posix()
        if rel.endswith("fr_solver/mach_ref.py"):
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if pat.search(line.split("#", 1)[0]):
                offenders.append(f"{rel}:{i}")
    assert not offenders, f"mach_ref 只能经 resolve_mach_ref 计算，发现手写：{offenders}"
