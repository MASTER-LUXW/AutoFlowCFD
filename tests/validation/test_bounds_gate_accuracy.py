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

## 结论（2026-09-20 重测：邻域模板换成顶点邻居之后，P2 也过了）

生产单机路径自 2026-09-19（C2 修复）起用的是**顶点邻居模板**，而本文件
此前一直在测**面邻居模板** —— 测的不是生产在跑的那条判据。换过来重测：

    棱柱基   order  标记比例(nx=8 -> 48)   project 档收敛阶     设计阶  判定
    坍缩     P1     17.19% -> 0.087%       1.60 / 2.85 / 2.27    2      通过
    坍缩     P2     18.75% -> 0.260%       2.00 / 2.97 / 3.63    3      通过
    原生     P1     18.75% -> 0.521%       1.13 / 2.14 / 2.30    2      通过
    原生     P2     18.75% -> 0.608%       1.94 / 2.73 / 3.12    3      通过

**四个组合全部通过**：标记比例 6 倍加密降 30~200 倍（门控渐近失活），
project 档的增量误差收敛阶不低于设计阶。

### 与旧结论的差别，以及它推翻了什么

面模板下的旧结论是"P2 不通过：标记比例平台在 ~17.6%、收敛阶只到 ~2.1，
成因是判据**与阶数有偏**（胞内极值随阶数伸得更远，邻居单元均值区间与
阶数无关），所以默认只在 P1 上开启"。

顶点模板下 P2 的标记比例降到 0.26%/0.61%、阶到 2.9~3.6。所以那条"与阶数
有偏"**是面模板的性质，不是判据本身的性质**：面邻居模板的均值包络太窄，
单元角点/高阶解点的值本来就可能越出所有**面**邻居的均值区间，而顶点邻居
模板把共享该顶点的全部单元都纳入包络之后这条越界消失。这与 C2 那轮
（真实 TGV 网格上面模板 100% 不收敛、顶点模板 100%→25%→6.18%）是同一个
机制的两份独立证据。

**它直接推翻"默认只在 P1 上开启门控"这条依据**（该依据的唯一支撑就是
P2 那个平台）。默认值本身不在本文件改动范围内，但这条结论必须记在这里，
不能让一个已经被自己数据推翻的理由继续挂在默认值上。

### 两条棱柱基的差别

原生档的标记比例约为坍缩档的 2 倍（P1 nx=48：0.521% vs 0.087%）。成因
是解点位置：原生棱柱的三角形节点是 Warp&Blend 节点、**含单元角点**，
而坍缩棱柱是张量积 Gauss 点、全在单元内部。BJ 这类"胞内极值 vs 邻域
均值包络"的判据在角点上天然更容易越界。两档都收敛，所以这是常数差别，
不是收敛性差别。

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

from autoflowcfd.fr.native_prism.mode import resolve_prism_basis_mode


# --- 期望值（2026-09-20 实测，顶点模板 + 屏蔽零填充，留余量）---
#
# **判据路径已与生产对齐**：此前这张表是用**面邻居模板**测的，而单机
# CPU/GPU 的生产路径自 2026-09-19（C2 修复）起用的是**顶点邻居模板**。
# 换过来之后结论有两处实质变化，都记在这里：
#
#   1. P2 也渐近失活了。面模板下 P2 的标记比例平台在 ~17.6%、增量误差
#      收敛阶只到 ~2.1（比设计阶 3 低一整阶），那是"默认只在 P1 上开启
#      门控"的依据；顶点模板下 P2 降到 0.26%（坍缩）/0.61%（原生）、
#      project 档的阶到 2.00/2.97/3.63（坍缩）——**设计阶达标**。
#   2. 两条棱柱基的数字不同但结论相同（原生的标记比例约高 2 倍：它的
#      解点含单元角点，BJ 这类"胞内极值 vs 邻域均值包络"判据在角点上
#      天然更容易越界；顶点模板把这条差距从"不收敛"压回"收敛"）。
#
# `legacy` 与 `project` 在 P1 上**逐位相同**（legacy 的 P1 sigma 本来就是
# {1, 2.2e-16}）；P2 上两档的阶差就是"滤波必须是幂等投影"那条修复的收益。
#
#: (棱柱基, filter_mode, order) -> ([(nx, 标记比例上界, 增量误差上界), ...],
#:                                   渐近阶下界)
_EXPECT = {
    # 坍缩档：标记 17.188/5.859/0.391/0.087%
    ("collapsed", "legacy", 1): ([(8, 0.20, 3.0e-1), (16, 0.075, 1.0e-1),
                                  (32, 0.006, 1.5e-2), (48, 0.002, 6.0e-3)],
                                 2.0),
    ("collapsed", "project", 1): ([(8, 0.20, 3.0e-1), (16, 0.075, 1.0e-1),
                                   (32, 0.006, 1.5e-2), (48, 0.002, 6.0e-3)],
                                  2.0),
    # 坍缩档 P2：标记 18.750/6.250/0.977/0.260%
    ("collapsed", "legacy", 2): ([(8, 0.22, 6.5e-2), (16, 0.08, 2.0e-2),
                                  (32, 0.015, 3.5e-3), (48, 0.005, 1.3e-3)],
                                 2.4),
    ("collapsed", "project", 2): ([(8, 0.22, 5.5e-2), (16, 0.08, 1.4e-2),
                                   (32, 0.015, 1.8e-3), (48, 0.005, 4.2e-4)],
                                  2.9),
    # 原生档 P1：标记 18.750/8.203/1.562/0.521%
    ("native", "legacy", 1): ([(8, 0.22, 3.5e-1), (16, 0.11, 1.6e-1),
                               (32, 0.022, 4.0e-2), (48, 0.008, 1.6e-2)],
                              2.0),
    ("native", "project", 1): ([(8, 0.22, 3.5e-1), (16, 0.11, 1.6e-1),
                                (32, 0.022, 4.0e-2), (48, 0.008, 1.6e-2)],
                               2.0),
    # 原生档 P2：标记 18.750/7.812/1.562/0.608%
    ("native", "legacy", 2): ([(8, 0.22, 8.0e-2), (16, 0.10, 2.6e-2),
                               (32, 0.022, 5.0e-3), (48, 0.009, 1.8e-3)],
                              2.4),
    ("native", "project", 2): ([(8, 0.22, 7.5e-2), (16, 0.10, 2.1e-2),
                                (32, 0.022, 3.4e-3), (48, 0.009, 9.5e-4)],
                               2.6),
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
from autoflowcfd.core.fr_operators.vertex_stencil import build_vertex_stencil
from autoflowcfd.fr.native_padding import real_sps_per_cell
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
    # **与生产路径一致（2026-09-20）**：单机 CPU/GPU 的门控用的是
    # `vertex_stencil`（顶点邻居模板，2026-09-19 的 C2 修复），并且按
    # `real_sps_per_cell` 屏蔽零填充槽位。此前这里用的是面邻居模板、
    # 也没屏蔽填充 —— 测的不是生产在跑的那条判据。
    rip = np.zeros(mesh.n_cells, dtype=bool)
    rip[:mesh.n_prism_cells] = True
    n_rp, n_rt = real_sps_per_cell(order)
    mask = compute_bounds_violation_mask(
        U, np.asarray(fc.owner_cell), np.asarray(fc.neighbor_cell),
        np.asarray(fc.is_boundary, dtype=bool), ref_scales=ref,
        row_is_prism=rip, n_real_prism=n_rp, n_real_tet=n_rt,
        vertex_stencil=build_vertex_stencil(mesh))
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


@pytest.mark.parametrize("mode,order", [("legacy", 1), ("legacy", 2),
                                        ("project", 1), ("project", 2)])
def test_gate_accuracy_on_exact_smooth_solution(mode, order):
    """标记比例单调下降 + 增量误差渐近收敛阶达到该档的实测下界。

    P2 上两档的阶下界刻意不同（legacy 2.4 / project 2.9）——那个差距
    就是"滤波必须是幂等投影"这条修复的定量收益。

    期望值按**当前棱柱基**取（两条基的解点位置不同，标记比例差约 2 倍，
    见 `_EXPECT` 上方说明）。
    """
    basis = resolve_prism_basis_mode()
    levels, order_floor = _EXPECT[(basis, mode, order)]
    got = _measure_all(mode, order, [n for n, _, _ in levels])
    fracs, errs, hs = [], [], []
    for (n, frac_max, err_max), (frac, err) in zip(levels, got):
        assert frac <= frac_max, (
            f"{basis} {mode} P{order} nx={n}: 标记比例 {100*frac:.3f}% "
            f"超出上界 {100*frac_max:.1f}%")
        assert err <= err_max, (
            f"{basis} {mode} P{order} nx={n}: 门控引入误差 {err:.3e} "
            f"超出上界 {err_max:.1e}")
        fracs.append(frac)
        errs.append(err)
        hs.append(LX / n)

    assert all(a > b for a, b in zip(fracs, fracs[1:])), (
        f"{mode} P{order}: 标记比例未随加密单调下降："
        f"{[f'{100*f:.2f}%' for f in fracs]}")
    # 渐近阶用最细两档（最粗那档 h=1.25 而涡核半径 Rc=1.0，解根本没被解析）
    p = float(np.log(errs[-2] / errs[-1]) / np.log(hs[-2] / hs[-1]))
    assert p >= order_floor, (
        f"{basis} {mode} P{order}: 最细两档增量误差收敛阶 {p:.2f} 低于"
        f"下界 {order_floor}")


def test_p1_gate_is_asymptotically_inactive():
    """P1 上门控必须**渐近失活**：最细一档标记比例 < 1%。

    这是"P1 上开启门控没有精度代价"的核心依据——收敛阶达标只说明"误差
    按阶收敛"，标记比例趋零才说明"加密后它根本不再动手"。

    实测（nx=48，顶点模板）：坍缩 0.087%、原生 0.521%。阈值取 1% 对两条
    基都留了余量。
    """
    frac, _ = _measure_all("project", 1, [48])[0]
    assert frac < 0.01, f"P1 nx=48 标记比例 {100*frac:.3f}% 未趋于零"


def test_p2_gate_is_now_also_asymptotically_inactive():
    """**方向已翻转（2026-09-20）**：P2 上门控现在同样渐近失活。

    本条原先是反向钉桩（"P2 不会失活，平台在 ~17.6%"），并写明"若哪天
    判据改成与阶数无偏、这条会失败，那时应当连同默认值一起重新评估"。
    那一刻到了：把判据的邻域模板从**面邻居**换成**顶点邻居**（2026-09-19
    的 C2 修复，生产单机路径早已在用，只是本文件还在测旧模板）之后，
    nx=48 的标记比例从 17.578% 降到 **0.260%（坍缩）/ 0.608%（原生）**，
    project 档的增量误差收敛阶到 **2.00/2.97/3.63（坍缩）**、
    **1.94/2.73/3.12（原生）** —— 设计阶 3 达标。

    也就是说"P2 的 BJ 判据与阶数有偏"这条结论**是面模板的性质，不是判据
    本身的性质**。它直接影响"order>=2 是否可以开启门控"这个默认值决定，
    相关评估见 `fr/modal_filter.py` 与 `core/fr_solver/filter.py` 的默认档
    说明。
    """
    frac, _ = _measure_all("project", 2, [48])[0]
    assert frac < 0.01, (
        f"P2 nx=48 标记比例 {100*frac:.3f}% 未趋于零 —— 若这是真实回归，"
        f"先查顶点模板是否真的被用上（面模板下这个数是 17.578%）")


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

    def test_project_is_idempotent_at_all_orders(self):
        """`project` 档必须幂等：门控每个 RK stage 施加一次，非幂等会随
        步数累积、把被标记单元一路削到常数。"""
        rows = self._run("project")
        for (nm, order), (idem, _rank) in rows.items():
            assert idem < 1e-12, (
                f"project 档 {nm} P{order} 不幂等：|F@F-F| = {idem:.3e}")

    def test_project_rank_equals_next_lower_order_dimension(self):
        """秩必须恰好等于"低一阶"的维数 ——"恰好削掉最高阶"。

        两条棱柱基的这个数不同（坍缩 `order^3`；原生是"少一阶的模态数
        + 零填充槽位的单位行"），公式与实测值见
        `tests/unit/test_modal_filter_order_loss.py::_expected_legacy_rank`
        —— 那里是同一个量的唯一事实来源，这里直接复用，不抄第二份。
        """
        from tests.unit.test_modal_filter_order_loss import (
            _expected_legacy_rank,
        )

        rows = self._run("project")
        for order in (1, 2, 3):
            _idem, rank = rows[("filter_prism", order)]
            expect = _expected_legacy_rank(order)
            assert rank == expect, (
                f"project 档 P{order} 棱柱滤波秩 {rank} != 低一阶维数 "
                f"{expect}")

    def test_sensor_is_bounded_damping_not_projection_and_why_that_matters(self):
        """`sensor` 档**刻意不是**投影，而是顶模态 0.99 的有界衰减 ——
        并且这条设计有已实测的代价，一起钉在这里。

        ## 两难（2026-09-19 用两个有精确解的算例量清）

        * 幂等投影（`project`）在 **P1** 上等于把被标记单元**拍平成 P0**
          （P1 的顶模态就是全部非常数内容）：Blasius 平板的壁面剪应力
          实测塌 **-87.69%**（cf 中位偏差），不可用；
        * mild 型有界衰减（`sensor`，2026-09-18 起）非幂等，每 stage 施加
          一次会累积：P2 四面体 TGV 上 BJ 判据对那个**欠分辨光滑场 100%
          标记**，于是退化成"全局施加"，450 次之后动能**净增长 +5.14%**
          —— 非物理（滤波只可能耗散）。而同一算例 `off` 单调衰减、
          `dK/dt` 是解析耗散率的 0.31~0.54 倍。
        * `sensor` 在 Blasius 上与 `off` **逐位相同**（cf 中位都是 +9.52%）
          —— 也就是说它在那个算例上实质无操作。

        所以两种动作各自破坏一个物理量，默认值因此在 2026-09-19 改成
        `off`（完整依据见 `fr/modal_filter.py` 里"默认值 2026-09-19 改为
        off"那一节）。`sensor`/`project` 都保留为合法档，本测试把各自的
        矩阵契约钉住，避免将来有人以为 `sensor` 是投影型。
        """
        rows = self._run("sensor")
        for (nm, order), (idem, rank) in rows.items():
            assert idem > 1e-6, (
                f"sensor 档 {nm} P{order} 变成了幂等（|F@F-F| = {idem:.3e}）"
                f"—— 若这是刻意改动，请同时更新本测试与 `fr/modal_filter.py`"
                f"里那节依据，并重新在 Blasius（cf）与 TGV（动能）两个算例"
                f"上判定")
        for order in (1, 2, 3):
            _idem, rank = rows[("filter_prism", order)]
            assert rank == (order + 1) ** 3, (
                f"sensor 档 P{order} 棱柱滤波秩 {rank} != 满秩 "
                f"{(order + 1) ** 3} —— 有界衰减不该丢秩")

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
