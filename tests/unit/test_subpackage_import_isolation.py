"""每个模块在**全新进程**里作为第一个被导入的模块都能成功（并行，约 40 秒）。

同一进程里的测试共享已导入的模块，导入顺序恰好可行时循环导入不会暴露。2026-10-22（I-123）顶层包改为延迟导出 API
之后暴露了三条一直被"先导入整个 API"的加载顺序掩盖的环：`boundary.fr_ghost_state` -> `core/__init__` /
`core/utils/__init__` 的包级预导入 -> 求解器 -> `fr_residual.inviscid` -> `boundary.fr_ghost_state`；
`implicit/newton/mean_flow_step` -> `fr_solver.state`（触发 `fr_solver/__init__` 预导入 FRSolver）-> `solve_loop`
-> `mean_flow_step`。
"""

import pkgutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import autoflowcfd


def _import_first(module):
    res = subprocess.run([sys.executable, "-c", f"import {module}"], capture_output=True, text=True,
                         encoding="utf-8", errors="replace")
    return module, res.returncode, res.stderr.strip().splitlines()[-1] if res.returncode else ""


def test_every_module_imports_first_in_a_fresh_process():
    modules = sorted(m.name for m in pkgutil.walk_packages(autoflowcfd.__path__, "autoflowcfd."))
    with ThreadPoolExecutor(8) as pool:
        failures = [f"{m}: {err}" for m, code, err in pool.map(_import_first, modules) if code]
    assert not failures, "\n".join(failures)
