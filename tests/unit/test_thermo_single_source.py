"""热力学常数只在 `core/thermo.py` 定义（I-12，2026-10-19）。

此前 GAMMA 在 10 个模块各定义一次、R_AIR 3 次，残差 / Jacobian / CFL / 参考马赫数 / 状态换算与 CUDA 源码里另有直接
写的 1.4、0.4——改工质或换比热比做验证时任何一处漏改都会让各项之间不自洽，且不会报错。这里扫描源码守住"只此一处"。
"""

import ast
from pathlib import Path

import autoflowcfd
from autoflowcfd.core import thermo

SRC = Path(autoflowcfd.__file__).parent
NAMES = {"GAMMA", "R_AIR", "CP_AIR", "_GAMMA_GAS", "GAMMA_AIR"}
#: 与比热比无关的 1.4（tetgen 质量参数）
UNRELATED_1_4 = {"grid/mesh_gen/tetgen/mesh_tetgen_core.py"}


def _modules():
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        yield rel, ast.parse(path.read_text(encoding="utf-8"))


def test_constants_are_defined_only_in_thermo():
    offenders = []
    for rel, tree in _modules():
        if rel == "core/thermo.py":
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                names = {t.id for t in targets if isinstance(t, ast.Name)}
                if names & NAMES and isinstance(node.value, ast.Constant):
                    offenders.append(f"{rel}:{node.lineno}")
    assert offenders == []


def test_no_gamma_literal_in_code_or_cuda_sources():
    offenders = []
    for rel, tree in _modules():
        if rel == "core/thermo.py" or rel in UNRELATED_1_4:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, float) and node.value == 1.4:
                offenders.append(f"{rel}:{node.lineno}")
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and "__device__" in node.value \
                    and "1.4" in node.value:
                offenders.append(f"{rel}:{node.lineno} (CUDA)")
    assert offenders == []


def test_values():
    assert thermo.GAMMA == 1.4 and thermo.R_AIR == 287.0
    assert thermo.CP_AIR == thermo.GAMMA * thermo.R_AIR / (thermo.GAMMA - 1.0)
