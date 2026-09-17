"""
bounds 门控对**光滑精确解**的精度代价 —— 以及模态滤波必须是幂等投影。

## 这份测试要回答的决策问题

`AFCFD_TROUBLED_SENSOR=bounds` 把被标记单元局部降一阶。真实网格上它把
"iter 112 发散"变成"400 步残差单调降 3.94 倍"，但真实网格没有精确解，
量不出**精度代价**。对一个数值格式，决定性的问题是限制器类方法的标准
检验：

  **网格加密时，门控引入的误差是否按设计阶数收敛？**

Barth-Jespersen 判据在常数场/线性场上恒不激活。光滑解在加密时局部越来
越接近线性，所以若判据实现正确，标记比例应随加密趋于 0、门控渐近失活。

算例用等熵涡（`_isentropic_vortex.py`）：二维可压缩 Euler 的**解析解**，
全场光滑，涡核峰值切向速度约 270 m/s，是真正非平凡的弯曲流动。

## 结论（2026-09-17 实测，本文件把它钉住）

    order  标记比例(nx=8 -> 48)     增量误差收敛阶      设计阶   判定
    P1     17.19% -> 0.74%          2.16 / 2.18          2       通过
    P2     32.81% -> 17.58%(平台)   1.71 / 2.10 / 2.14   3       不通过

**P1 通过**：标记比例 6 倍加密降 23 倍，门控渐近失活，收敛阶不低于设计阶。
**P2 不通过**：标记比例平台在 ~17.6%，收敛阶只到 ~2.1（= 降一阶后的 P1
精度），比设计阶 3 低整一阶。

成因是判据本身**与阶数有偏**：它把"单元内解点极值"（随阶数增大伸得更远）
与"邻居单元均值区间"（与阶数无关）比。所以默认只在 P1 上开启。

## 顺带查出并修复的独立缺陷：模态滤波在 P2/P3 不幂等

门控要表达的操作是"把这个单元降一阶"，而它**每个 RK stage 都施加一次**
——那在数学上要求算子是**幂等投影**。实测指数型 sigma 谱：

    P1  sigma=[1, 2.22e-16]                      |F@F-F| = 2.2e-16  投影
    P2  sigma=[1, 0.868667, 2.22e-16]            |F@F-F| = 6.3e-02  非投影
    P3  sigma=[1, 0.994521, 0.245032, 2.22e-16]  |F@F-F| = 3.5e-01  非投影

P2/P3 的**中间**模态 sigma 严格介于 0 和 1，于是每步乘三次、逐步累积：
P2 中间模态每步残留 0.6555（约 100 步后 ~1e-18，实际退化到 P0 而不是
P1）；P3 的 eta=2/3 模态每步残留 0.0147（一步内基本清零，两阶）。

这一条对**默认档 `legacy`** 同样成立，而它是全局每 stage 施加的——所以
项目记忆 `modal_filter_annihilates_one_order` 记的"每阶恰好损失一整阶"
（用单次施加的矩阵秩验证）对长程运行**不成立**，P2/P3 损失的更多。

修复：新增 `AFCFD_FILTER_MODE=project`（sigma 严格取 {0,1}），且
`sensor` 档改用同一套投影矩阵。实测把 P2 的门控收敛阶从 ~1.35 提到
~2.1。P1 上两者**逐位相同**（legacy 的 P1 sigma 本来就是 {1, 2.2e-16}）。
"""

import os
import subprocess
import sys

import numpy as np
import pytest


# --- 期望值（2026-09-17 实测，留余量）---
#
# 两档矩阵都钉住：改善幅度本身就是"为什么需要 project 档"的证据。
# P1 上两档**逐位相同**（legacy 的 P1 sigma 本来就是 {1, 2.2e-16}）。
#
#: (filter_mode, order) -> ([(nx, 标记比例上界, 增量误差上界), ...], 渐近阶下界)
_EXPECT = {
    ("legacy", 1): ([(8, 0.20, 3.0e-1), (16, 0.14, 1.3e-1),
                     (32, 0.035, 3.0e-2), (48, 0.013, 1.3e-2)], 2.0),
    ("project", 1): ([(8, 0.20, 3.0e-1), (16, 0.14, 1.3e-1),
                      (32, 0.035, 3.0e-2), (48, 0.013, 1.3e-2)], 2.0),
    # legacy 在 P2 非幂等 -> 误差更大、阶更低（实测 1.39/1.44/1.32）
    ("legacy", 2): ([(8, 0.38, 6.5e-2), (16, 0.30, 2.5e-2),
                     (32, 0.25, 9.0e-3), (48, 0.22, 5.5e-3)], 1.1),
    # project 是精确投影 -> 阶提到 ~2.1（= 降一阶后的 P1 精度）
    ("project", 2): ([(8, 0.38, 5.0e-2), (16, 0.30, 1.6e-2),
                      (32, 0.25, 3.8e-3), (48, 0.22, 1.6e-3)], 1.9),
}

LX, H, LZ = 10.0, 10.0, 0.5
RHO_INF, P_INF, U_INF = 1.225, 101325.0, 100.0

#: 在子进程里跑测量：`AFCFD_FILTER_MODE` 在 `fr/modal_filter.py` 的**模块
#: 导入期**读取并据此定 FILTER_ALPHA，同一个进程里改环境变量不会重建矩阵。
#: 第一版直接在测试进程里测，于是无论参数化成什么档，量到的都是 pytest
#: 进程自己的默认档（legacy）——P2 的 project 期望值当场对不上，被这条
#: 断言抓到。
_MEASURE_SRC = r"""
import sys, json
sys.path.insert(0, {src!r})
sys.path.insert(0, {tests!r})
import numpy as np
from loguru import logger; logger.remove()
from autoflowcfd.core.fr_operators.bounds_sensor import compute_bounds_violation_mask
from autoflowcfd.core.fr_solver.residual_diagnostics import _reference_scales
from autoflowcfd.fr.operators import generate_fr_operators
from validation._isentropic_vortex import (
    build_vortex_mesh, primitive_to_conservative, vortex_primitive_field)

LX, H, LZ = 10.0, 10.0, 0.5
RHO_INF, P_INF, U_INF = 1.225, 101325.0, 100.0
order = {order}
ref = _reference_scales({{"rho_inf": RHO_INF, "vel_inf": U_INF, "p_inf": P_INF}}, 5)
res = []
for n in {levels!r}:
    mesh = build_vortex_mesh(order, n, n, 1, LX, H, LZ)
    ops = generate_fr_operators(order)
    xyz = np.asarray(mesh.sps_coords).reshape(mesh.n_cells, mesh.n_sps_per_cell, 3)
    prim = vortex_primitive_field(xyz[..., 0], xyz[..., 1], 0.0,
                                  LX / 2, H / 2, LX, RHO_INF, P_INF, U_INF)
    U = np.stack(primitive_to_conservative(*prim), axis=-1)[..., :5]
    fc = mesh.face_connectivity
    mask = compute_bounds_violation_mask(
        U, np.asarray(fc.owner_cell), np.asarray(fc.neighbor_cell),
        np.asarray(fc.is_boundary, dtype=bool), ref_scales=ref)
    Ug = U.copy()
    for lo, hi, F in ((0, mesh.n_prism_cells, np.asarray(ops.filter_prism)),
                      (mesh.n_prism_cells, mesh.n_cells, np.asarray(ops.filter_tet))):
        if hi <= lo:
            continue
        sel = np.arange(lo, hi)[mask[lo:hi]]
        if sel.size:
            Ug[sel] = np.einsum('ij,cjv->civ', F, U[sel])
    d = (Ug - U).reshape(-1, 5)
    res.append([float(mask.mean()),
                float(np.max(np.sqrt(np.mean(d ** 2, axis=0)) / ref))])
print("RESULT" + json.dumps(res))
"""


def _measure_all(mode, order, levels):
    """在子进程里按给定 FILTER_MODE 测量各分辨率的 (标记比例, 增量误差)。"""
    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    src = _MEASURE_SRC.format(
        src=os.path.join(root, "src"), tests=os.path.join(root, "tests"),
        order=order, levels=list(levels))
    env = dict(os.environ)
    env["AFCFD_FILTER_MODE"] = mode
    env["PYTHONIOENCODING"] = "utf-8"
    out = subprocess.run([sys.executable, "-c", src],
                         capture_output=True, text=True, env=env, timeout=3600)
    assert out.returncode == 0, out.stderr[-3000:]
    line = [ln for ln in out.stdout.splitlines() if ln.startswith("RESULT")][-1]
    import json
    return json.loads(line[len("RESULT"):])


@pytest.mark.parametrize("mode,order", sorted(_EXPECT))
def test_gate_accuracy_on_exact_smooth_solution(mode, order):
    """标记比例单调下降 + 增量误差渐近收敛阶达到该档的实测下界。

    P2 上两档的阶下界刻意不同（legacy 1.1 / project 1.9）——那个差距
    就是"滤波必须是幂等投影"这条修复的定量收益。
    """
    levels, order_floor = _EXPECT[(mode, order)]
    got = _measure_all(mode, order, [n for n, _, _ in levels])
    fracs, errs, hs = [], [], []
    for (n, frac_max, err_max), (frac, err) in zip(levels, got):
        assert frac <= frac_max, (
            f"{mode} P{order} nx={n}: 标记比例 {100*frac:.3f}% 超出上界 "
            f"{100*frac_max:.1f}%")
        assert err <= err_max, (
            f"{mode} P{order} nx={n}: 门控引入误差 {err:.3e} 超出上界 "
            f"{err_max:.1e}")
        fracs.append(frac)
        errs.append(err)
        hs.append(LX / n)

    assert all(a > b for a, b in zip(fracs, fracs[1:])), (
        f"{mode} P{order}: 标记比例未随加密单调下降："
        f"{[f'{100*f:.2f}%' for f in fracs]}")
    # 渐近阶用最细两档（最粗那档 h=1.25 而涡核半径 Rc=1.0，解根本没被解析）
    p = float(np.log(errs[-2] / errs[-1]) / np.log(hs[-2] / hs[-1]))
    assert p >= order_floor, (
        f"{mode} P{order}: 最细两档增量误差收敛阶 {p:.2f} 低于下界 "
        f"{order_floor}")


def test_p1_gate_is_asymptotically_inactive():
    """P1 上门控必须**渐近失活**：最细一档标记比例 < 1.5%。

    这是"P1 上开启门控没有精度代价"的核心依据——收敛阶达标只说明"误差
    按阶收敛"，标记比例趋零才说明"加密后它根本不再动手"。
    """
    frac, _ = _measure_all("project", 1, [48])[0]
    assert frac < 0.015, f"P1 nx=48 标记比例 {100*frac:.3f}% 未趋于零"


def test_p2_gate_does_not_become_inactive():
    """反向钉桩：P2 上门控**不会**渐近失活（平台在 ~17.6%）。

    方向是刻意的：它把"为什么默认只在 P1 上开启"的依据钉住。若哪天判据
    改成与阶数无偏、这条会失败，那时应当连同默认值一起重新评估，而不是
    悄悄放开。
    """
    frac, _ = _measure_all("project", 2, [48])[0]
    assert frac > 0.10, (
        f"P2 nx=48 标记比例 {100*frac:.3f}% 已降到 10% 以下 —— 判据似乎不再"
        f"与阶数有偏，请重新评估 order>=2 的默认值并更新本测试")


class TestFilterIsIdempotentProjection:
    """`project` / `sensor` 档的滤波矩阵必须是**精确幂等投影**。

    子进程运行：`AFCFD_FILTER_MODE` 在 `fr/modal_filter.py` 的**模块导入期**
    读取并据此定 FILTER_ALPHA，同一个进程里改环境变量不会重建矩阵。
    """

    _SNIPPET = (
        "import sys; sys.path.insert(0, %r)\n"
        "import numpy as np\n"
        "from loguru import logger; logger.remove()\n"
        "from autoflowcfd.fr.operators import generate_fr_operators\n"
        "out = []\n"
        "for order in (1, 2, 3):\n"
        "    ops = generate_fr_operators(order)\n"
        "    for nm in ('filter_prism', 'filter_tet'):\n"
        "        F = np.asarray(getattr(ops, nm))\n"
        "        out.append('%%s %%d %%.3e %%d' %% (\n"
        "            nm, order, np.abs(F @ F - F).max(),\n"
        "            np.linalg.matrix_rank(F, tol=1e-10)))\n"
        "print('|'.join(out))\n"
    )

    def _run(self, mode):
        src_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)))), "src")
        env = dict(os.environ)
        env["AFCFD_FILTER_MODE"] = mode
        env["PYTHONIOENCODING"] = "utf-8"
        out = subprocess.run(
            [sys.executable, "-c", self._SNIPPET % src_dir],
            capture_output=True, text=True, env=env, timeout=900)
        assert out.returncode == 0, out.stderr[-2000:]
        rows = {}
        for item in out.stdout.strip().splitlines()[-1].split("|"):
            nm, order, idem, rank = item.split()
            rows[(nm, int(order))] = (float(idem), int(rank))
        return rows

    @pytest.mark.parametrize("mode", ["project", "sensor"])
    def test_idempotent_at_all_orders(self, mode):
        rows = self._run(mode)
        for (nm, order), (idem, _rank) in rows.items():
            assert idem < 1e-12, (
                f"{mode} 档 {nm} P{order} 不幂等：|F@F-F| = {idem:.3e}。"
                f"门控每个 RK stage 施加一次，非幂等会随步数累积、"
                f"把被标记单元一路削到常数"
            )

    @pytest.mark.parametrize("mode", ["project", "sensor"])
    def test_rank_equals_next_lower_order_dimension(self, mode):
        """秩必须恰好等于低一阶的维数（棱柱 order^3）——"恰好削掉最高阶"。"""
        rows = self._run(mode)
        for order in (1, 2, 3):
            idem, rank = rows[("filter_prism", order)]
            assert rank == order ** 3, (
                f"{mode} 档 P{order} 棱柱滤波秩 {rank} != 低一阶维数 {order**3}"
            )

    def test_legacy_is_not_idempotent_at_p2_p3(self):
        """反向对照：`legacy` 档在 P2/P3 上确实不幂等。

        这条钉住的是"为什么需要 project 档"。若哪天 legacy 也变成投影了，
        本测试会失败，那时应当连同 `modal_filter_annihilates_one_order`
        这条记忆一起更新——那条记忆记的"每阶恰好损失一整阶"是用**单次
        施加**的秩验证的，对非幂等算子的长程行为并不成立。
        """
        rows = self._run("legacy")
        assert rows[("filter_prism", 1)][0] < 1e-12, "legacy 在 P1 上本该是投影"
        assert rows[("filter_prism", 2)][0] > 1e-3, (
            "legacy 在 P2 上变成幂等了？请重新评估本文件与相关记忆"
        )
        assert rows[("filter_prism", 3)][0] > 1e-3, (
            "legacy 在 P3 上变成幂等了？请重新评估本文件与相关记忆"
        )
