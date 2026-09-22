"""各 `AFCFD_FILTER_MODE` 档对**壁面剪应力**的代价（C1，2026-09-20）。

## 这份测试回答的决策问题

`FILTER_MODE` 的默认值必须是 `off`/`sensor` 而不是 `legacy`/`project`，
理由此前只有"legacy 每阶损失一整阶"这个**矩阵秩**层面的论证（项目记忆
`modal_filter_annihilates_one_order`）。秩论证不等于物理代价，而唯一有
精确解的粘性算例（Blasius 平板）此前量不出来 —— 从均匀初场出发那 400 步
处在增长的暂态里，两条棱柱基给出的 tau 都是沿全板近似恒定的非物理值
（见 `_blasius_case.set_blasius_exact_state` 文档的完整数据）。

**解析初场把这个前提去掉了**：从真解出发，任何档位造成的 cf 偏离都只能
来自该档位自己的离散动作。实测（`le_offset=0.5`、nx=16、CFL 0.1、
原生棱柱基、`TROUBLED_SENSOR=bounds`）：

    档位                      150 步 cf/cf_exact 中位   400 步
    off                       0.9890                    0.9805
    sensor + bounds           0.9890（与 off 相同）      0.9805（与 off 相同）
    legacy（全局每 stage）     **0.0000**                **0.0000**
    project（全局每 stage）    **0.0000**                **0.0000**

两条结论，都写成可执行判据：

1. **门控（`sensor`）对壁面剪应力零代价** —— 与 `off` 的 cf 中位、最差点
   完全相同。历史对比：最早 **-93%**、2026-09-18 修到 **-6.3%**、现在
   （顶点邻居模板 + 顶模态 0.99 有界衰减 + 解析初场量法）**0.0%**。
2. **全局每 stage 施加的滤波在 P1 上把壁面剪应力清零**。P1 的顶模态就是
   全部胞内梯度，而壁面剪应力恰好就是胞内梯度。

## 门控"零代价"只在这个窗口内成立（2026-09-20 补测，必须写在这里）

同一算例跑到 4000 步，`off` 与 `sensor` 会分叉：

    步数        500      1000     1500     2500     4000
    off        0.9763   0.9448   0.8832   0.4993   -3.2388
    sensor     0.9763   0.6070   -0.0739  -0.1241  -0.2186

* 到 500 步两档**逐位相同**（门控还没开始触发），这就是本文件 150 步
  判据成立的原因；
* 1000 步之后门控开始触发，它**把壁面剪应力打坏得更快**（0.945 -> 0.607
  -> -0.074）—— 与项目记忆 `bj_gate_destroyed_wall_shear` 记的机制一致：
  被标记单元局部降阶，而壁面剪应力恰好就是那部分胞内梯度；
* 但它同时**把失控截住**（4000 步 -0.22 vs 不门控的 -3.24）。

所以正确的表述是"**门控在解仍然光滑的窗口内零代价，一旦开始触发就有
实质代价**"，不是"门控没有代价"。本文件的 150 步判据落在前一个区间里，
仍然有效；这段数据是为了不让它被读成后者。

## 这是**短窗口**对照，长程漂移是另一回事

同一算例跑 4000 步会看到解从真解系统性走开（cf 中位 0.993 -> -3.24，
第 3000 步附近 tau 穿零）。本文件比较的是**同一窗口内不同档位之间**的
差别，那个差别（off/sensor 0.989 vs legacy/project 0.000）比漂移量级大
得多、且在窗口的每一步都成立，所以对照本身有效。长程漂移是一条独立的
开放问题，见 `test_blasius.py::TestWallShearPreservesTheAnalyticSolution`
的同名说明。

## 为什么必须子进程

`AFCFD_FILTER_MODE` 在 `fr/modal_filter.py` 的**模块导入期**读取并据此
定 FILTER_ALPHA 与滤波矩阵，同一个进程里改环境变量不会重建它们 ——
与 `test_bounds_gate_accuracy.py` 同一个理由、同一个做法。
"""

import os
import subprocess
import sys

import pytest

#: 150 步（而不是 400）：`off` 档实测 0.9890，与 400 步的 0.9805 同结论，
#: 而两次子进程运行的总时间只有一半。判据留了余量。
_N_STEPS = 150

_SRC = r"""
import os, sys
sys.path.insert(0, {src!r})
sys.path.insert(0, {tests!r})
os.environ["AFCFD_PRISM_BASIS"] = "native"
os.environ["AFCFD_TROUBLED_SENSOR"] = "bounds"
import numpy as np
from loguru import logger; logger.remove()
from validation._blasius_case import (
    build_blasius_solver, wall_shear_profile, set_blasius_exact_state,
    blasius_cf, L_PLATE)

solver, meta = build_blasius_solver(nx=16, cells_in_delta=4.0, cfl=0.10,
                                    le_offset=0.5)
set_blasius_exact_state(solver, meta)
x0 = meta["x_virtual_origin"]
for _ in range({nstep}):
    r = float(solver.step(1e-4))
    if not np.isfinite(r):
        print("RESULT nan"); raise SystemExit(0)
x, cf, tau, _ = wall_shear_profile(solver, solver.mesh)
cf_ex = blasius_cf(x + x0, meta["nu"])
m = x > 0.2 * L_PLATE
ratio = cf[m] / cf_ex[m]
print("RESULT %.6f %.6f %d" % (float(np.median(ratio)),
                               float(np.abs(ratio - 1.0).max()),
                               int(np.all(tau[m] > 0))))
"""


def _measure(filter_mode):
    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    src = _SRC.format(src=os.path.join(root, "src"),
                      tests=os.path.join(root, "tests"), nstep=_N_STEPS)
    env = dict(os.environ)
    env["AFCFD_FILTER_MODE"] = filter_mode
    env["PYTHONIOENCODING"] = "utf-8"
    out = subprocess.run([sys.executable, "-c", src], capture_output=True,
                         text=True, env=env, timeout=3600)
    assert out.returncode == 0, out.stderr[-3000:]
    line = [ln for ln in out.stdout.splitlines()
            if ln.startswith("RESULT")][-1].split()
    assert line[1] != "nan", f"{filter_mode} 档从解析初场出发就发散了"
    return float(line[1]), float(line[2]), bool(int(line[3]))


@pytest.mark.parametrize("mode", ["off", "sensor"])
def test_gate_and_off_preserve_the_wall_shear(mode):
    """`off` 与 `sensor` 都必须把 cf 保持在解析值的 3% 以内。"""
    median, worst, all_positive = _measure(mode)
    assert all_positive, f"{mode} 档出现非正壁面剪应力"
    assert abs(median - 1.0) < 0.03, (
        f"{mode} 档 cf 中位比值 {median:.4f} 偏离解析值超过 3%"
        f"（实测 150 步 0.9890）")
    assert worst < 0.05, (
        f"{mode} 档 cf 最差点偏离 {worst:.4f} 超过 5%（实测 0.0196）")


@pytest.mark.parametrize("mode", ["legacy", "project"])
def test_global_every_stage_filters_destroy_the_wall_shear(mode):
    """**负控制**：全局每 stage 施加的滤波在 P1 上把壁面剪应力清零。

    这条是 `FILTER_MODE` 默认值不能是 `legacy`/`project` 的定量依据。
    若哪天它不再成立（例如滤波改成只作用在真正的高阶内容上），应当来
    更新这里的记录值并重新评估默认值，而不是删掉本条。
    """
    median, _worst, all_positive = _measure(mode)
    assert median < 0.5, (
        f"{mode} 档 cf 中位比值 {median:.4f} 不再是清零量级（实测 0.0000）"
        f"—— 若这是真实修复，请更新本条与 fr/modal_filter.py 的默认值依据")
    assert not all_positive, (
        f"{mode} 档的壁面剪应力竟然处处为正（实测被清零后不成立）")
