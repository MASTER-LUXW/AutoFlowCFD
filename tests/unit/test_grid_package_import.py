"""`autoflowcfd.grid` 包导入不加载子模块（I-123）：求解器启动只需数据结构，不应付 tetgen 网格生成、.nas 解析、
质量校验的导入代价。2026-10-22 以前包 `__init__` 一次性导入它们。"""

import subprocess
import sys

_HEAVY = (
    "autoflowcfd.grid.mesh_gen.tetgen.volume_mesh_generator",
    "autoflowcfd.grid.nas_io.parser_core",
    "autoflowcfd.grid.validation.quality_validator",
    "autoflowcfd.grid.high_order.high_order_mesh",
)


def test_importing_the_grid_schema_does_not_load_heavy_submodules():
    code = ("import sys, autoflowcfd.grid, autoflowcfd.grid.schema\n"
            f"print([m for m in {_HEAVY!r} if m in sys.modules])")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip().splitlines()[-1] == "[]"
