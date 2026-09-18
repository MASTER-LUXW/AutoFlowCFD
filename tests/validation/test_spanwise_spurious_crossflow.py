"""准二维算例里的**伪横流**：病根是"该方向是被三角化的坍缩轴"。

## 这份测试钉的是什么

2026-09-18 决定性确证：棱柱网格上，**被三角化的那两个参考轴方向**会出现
伪横流速度，而**挤出（张量）轴方向**不会。

    展向 = 被三角化的轴（生产 Blasius 设置）  750 步 max|展向速度| = 5.1e-01
    展向 = 挤出轴（本文件的对照设置）         750 步 max|展向速度| = 3.2e-12
                                              （相差 11 个数量级）

两者其余一切相同（同求解器、同 BC 类型、同 CFL、同步数、同物理参数）。

## 机制

三角形的**斜边面**法向在两个三角化轴方向上都有分量。对与该方向无关的
压力场，闭合面上 `∮ p*n dS = 0` 是精确相消的；但界面用的是 AUSM+up
**数值**通量，其上风耗散项依赖左右态之差，绕单元闭合面**不相消**，于是
留下净的伪动量源，且对同一矩形的两个三角形反号（实测
`corr(w_a, w_b) = -0.7986`，成对差是成对和的 2 倍）。

它**在单元均值层面**：`legacy` 档每个 stage 全局清零顶模态（把 P1 压成
P0）仍留 2.82% 的幅值，所以滤波器原理上救不了。

它**不收敛**（固定累计伪时间 tau=3e-3、各方向一起加密）：

    nx=8/ny=18   1.70% of U    L2(w_cell)=1.69e-2
    nx=16/ny=36  3.71%         L2=2.33e-2
    nx=32/ny=72  4.31%         L2=1.60e-2

max 值不降反升、L2 无趋势，所以是**离散伪模态**而不是截断误差 ——
加密网格救不了，必须在格式层面修。
（只加密流向会测出 1.17/1.37 的"收敛阶"，那是把 `ny` 固定造成的假象；
单方向加密测出来的阶不能当结论。）

## 为什么这对生产很重要

真实网格（plate_demo / cube_demo）的边界层棱柱沿**壁面法向**挤出，于是
**两个切向都是被三角化的坍缩轴** —— 任何切向流动分量都受这条误差影响。
plate_demo 363k 的 P2 重测在第 68 步发散，而 P2 阶段各分量归一化残差的
增长恰好是横流分量主导（rho_v 3.94x、rho_w 7.13x，rho 只有 1.18x）。

## 判据取法

本文件只钉**好的那一半**（挤出轴方向必须是机器零）为硬判据 —— 那是一条
离散对称性，任何正确的实现都必须满足，将来的修复也不会破坏它。
三角化轴那一侧只做**特征化**记录（宽松上下界），成功修复之后应当把
`_TRIANGULATED_AXIS_MAX` 调小，而不是删掉这条测试。
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from validation._blasius_case import (  # noqa: E402
    L_PLATE, P_INF, RE_L_DEFAULT, RHO_INF, U_INF,
    blasius_delta99, build_blasius_solver, nu_for,
)
from validation._channel_mesh import (  # noqa: E402
    build_channel_mesh_prism, build_face_exact_ghost_provider,
)

#: 步数取 750：实测到 250 步伪横流就已经到 1% of U，750 步到 1.7%，
#: 而挤出轴那一侧始终在 1e-12。差别有 10 个数量级以上，短窗口足够判定，
#: 不需要长跑（这一点与 cf 那类"必须跑到 tau/T~1 才能比"的判据不同：
#: 那里比的是收敛后的物理量，这里比的是一条**离散对称性**）。
_STEPS = 750
_NX = 16
_CFL = 0.1

#: 挤出轴方向的伪横流上限（相对来流）。实测 3.2e-12/30 = 1.05e-13，
#: 留三个数量级余量。**这是硬判据**：它是离散对称性，不是精度指标。
_EXTRUSION_AXIS_MAX = 1e-10

#: 三角化轴方向的特征化区间（相对来流）。实测 1.7e-2。下界存在是为了
#: 防止"测试悄悄变成恒真"——若哪天它掉到下界以下，说明修复生效了，
#: 那时应当收紧本常数并更新上面的文档，而不是删掉这条测试。
_TRIANGULATED_AXIS_MIN = 1e-4
_TRIANGULATED_AXIS_MAX = 1e-1


def _advance(solver, n_steps):
    for _ in range(n_steps):
        solver.step(dt=1e-6)
    U = np.asarray(solver.state.U)
    rho = np.maximum(U[:, :, 0], 1e-30)
    return U, rho


def _build_span_extruded(nx, n_wall, n_span, cfl):
    """三角形落在 流向–壁法 平面、沿**展向**挤出。

    完全复用 `build_channel_mesh_prism`（它总是"(x,z) 三角化、沿 y
    挤出"），把 `H` 当展向宽度、`Lz` 当壁法高度传进去，再交换边界标签：

        builder 的 y 轴（挤出轴、ny 层、高度 H）  -> 本算例的**展向**
        builder 的 z 轴（三角化的第二轴、nz 层）  -> 本算例的**壁法向**

    **只用于判别机制，不是生产设置**：壁法方向不再是挤出轴，近壁剪切解
    就不再是挤出轴的精确多项式（理由见 `_channel_mesh.py` 的文档与项目
    记忆 `tet_collapsed_coord_anisotropy`），近壁精度会变差。
    """
    from autoflowcfd.core.fr_solver import FRSolver
    from autoflowcfd.core.time_integration import TimeIntegrationScheme

    nu = nu_for(RE_L_DEFAULT)
    mu = RHO_INF * nu
    d99 = float(blasius_delta99(np.array([L_PLATE]), nu)[0])
    h_wall = 6.0 * d99
    span = h_wall / 4.0
    mesh = build_channel_mesh_prism(1, nx=nx, ny=n_span, nz=n_wall,
                                    Lx=L_PLATE, H=span, Lz=h_wall)
    bc = {
        "wall_bottom": {"type": "SYMMETRY"},          # 展向两侧
        "wall_top": {"type": "SYMMETRY"},
        "z_min": {"type": "WALL", "is_no_slip": True,  # 平板
                  "wall_velocity": [0.0, 0.0, 0.0]},
        "z_max": {"type": "WALL", "is_no_slip": False},  # 外场
        "x_min": {"type": "INLET",
                  "Q_inlet": [RHO_INF, U_INF, 0.0, 0.0, P_INF]},
        "x_max": {"type": "OUTLET", "p_outlet": P_INF},
    }
    solver = FRSolver(
        mesh=mesh, order=1, turb_model_name="NONE", n_vars=5,
        time_scheme=TimeIntegrationScheme.SSP_RK3,
        rho_inf=RHO_INF, vel_inf=U_INF, p_inf=P_INF,
        mu_molecular=mu, bc_overrides=bc,
        adaptive_cfl=False, cfl_start=cfl, cfl_max=cfl, cfl_min=cfl,
    )
    solver.order_continuation_enabled = False
    solver.boundary_ghost_provider = build_face_exact_ghost_provider(
        mesh, L_PLATE, span, h_wall, bc)
    solver._reference_area = L_PLATE * span
    return solver


@pytest.fixture(scope="module")
def extrusion_axis_result():
    solver = _build_span_extruded(_NX, n_wall=36, n_span=1, cfl=_CFL)
    U, rho = _advance(solver, _STEPS)
    # 展向 = builder 的 y 轴 -> 守恒变量的 rho_v
    span_vel = float(np.max(np.abs(U[:, :, 2] / rho)))
    wall_vel = float(np.max(np.abs(U[:, :, 3] / rho)))
    return span_vel, wall_vel


@pytest.fixture(scope="module")
def triangulated_axis_result():
    solver, _ = build_blasius_solver(order=1, nx=_NX, nz=1, cfl=_CFL)
    U, rho = _advance(solver, _STEPS)
    # 展向 = z -> 守恒变量的 rho_w
    span_vel = float(np.max(np.abs(U[:, :, 3] / rho)))
    wall_vel = float(np.max(np.abs(U[:, :, 2] / rho)))
    return span_vel, wall_vel


class TestExtrusionAxisPreservesSpanwiseZero:
    """**硬判据**：展向是挤出轴时，展向速度必须保持机器零。"""

    def test_spanwise_velocity_stays_at_machine_zero(self,
                                                    extrusion_axis_result):
        span_vel, _ = extrusion_axis_result
        rel = span_vel / U_INF
        assert rel < _EXTRUSION_AXIS_MAX, (
            f"展向（挤出轴）速度 {span_vel:.3e} = {100 * rel:.6f}% of U，"
            f"超出 {100 * _EXTRUSION_AXIS_MAX:.1e}%。这是一条**离散对称性**："
            f"该方向上不存在法向带该分量的斜边面，伪动量源本就不该产生。"
            f"它变大意味着引入了新的对称性破坏。")

    def test_wall_normal_velocity_is_physically_present(self,
                                                        extrusion_axis_result):
        """反向自检：壁法速度必须**不是**零。

        否则上一条可能只是因为整个流动没发展起来（那样测试恒真）。
        边界层里 v ~ 0.86*U/sqrt(Re_x)，量级是 U 的百分之几。
        """
        _, wall_vel = extrusion_axis_result
        assert wall_vel > 1e-3 * U_INF, (
            f"壁法速度只有 {wall_vel:.3e}，流动似乎没发展起来——"
            f"那样'展向是机器零'这条判据是空的")


class TestTriangulatedAxisShowsSpuriousCrossflow:
    """**特征化**：展向是三角化轴时，伪横流确实存在（当前约 1.7% of U）。

    方向是刻意的：它把"这个缺陷现在还在"钉住。补上坍缩棱柱的面通量
    反混叠之后这个值应当下降，那时**收紧** `_TRIANGULATED_AXIS_MAX`
    并更新模块文档，而不是删掉本类。
    """

    def test_spurious_spanwise_velocity_is_present_and_bounded(
            self, triangulated_axis_result):
        span_vel, _ = triangulated_axis_result
        rel = span_vel / U_INF
        assert rel > _TRIANGULATED_AXIS_MIN, (
            f"伪横流只有 {100 * rel:.6f}% of U —— 若这是真实改善，请收紧"
            f"_TRIANGULATED_AXIS_MAX 并更新本文件文档；若不是，请检查测试"
            f"是否变成了恒真")
        assert rel < _TRIANGULATED_AXIS_MAX, (
            f"伪横流 {100 * rel:.4f}% of U 超出已记录量级 "
            f"{100 * _TRIANGULATED_AXIS_MAX:.1f}%，说明比 2026-09-18 记录的"
            f"情况更糟，需要重新排查")

    def test_two_configurations_differ_by_orders_of_magnitude(
            self, extrusion_axis_result, triangulated_axis_result):
        """核心结论：两者相差若干个数量级。

        这一条是整份测试的要点——它把"病根是该方向是不是被三角化的轴"
        这个因果关系本身钉住，而不是只钉两个绝对量。
        """
        span_ext, _ = extrusion_axis_result
        span_tri, _ = triangulated_axis_result
        ratio = span_tri / max(span_ext, 1e-300)
        assert ratio > 1e6, (
            f"三角化轴/挤出轴 的伪横流之比只有 {ratio:.3e}（实测应为 ~1e11）。"
            f"两者相差若干数量级正是'病根是被三角化的坍缩轴'这个结论的依据。")


# ----------------------------------------------------------------------
# P2 发散的**最小复现**
# ----------------------------------------------------------------------

#: P2 在本算例上发散的步数（CFL 0.1 实测第 75 步出现非有限值）。留余量
#: 到 150：只要在 150 步内发散，缺陷就还在。
_P2_DIVERGES_WITHIN = 150
#: P1 必须干净跑过的步数（实测 22000 步不发散，|w| 饱和在 ~1.3）。
_P1_MUST_SURVIVE = 300
#: P1 的 |w| 上限（相对来流）。实测 22000 步饱和在 4.3%，300 步时远小于它。
_P1_W_MAX = 0.10


def _run_until_nonfinite(order, nx, cfl, max_steps):
    """推进到出现非有限值或用完步数；返回 (发散步数或 None, max|w|/U)。"""
    solver, _ = build_blasius_solver(order=order, nx=nx, nz=1, cfl=cfl)
    worst = 0.0
    for it in range(1, max_steps + 1):
        solver.step(dt=1e-6)
        U = np.asarray(solver.state.U)
        if not np.all(np.isfinite(U)):
            return it, float("inf")
        rho = np.maximum(np.abs(U[:, :, 0]), 1e-30)
        worst = max(worst, float(np.max(np.abs(U[:, :, 3] / rho))) / U_INF)
    return None, worst


class TestP2DivergesOnACleanPrismMesh:
    """**已知缺陷的最小复现**：P2 在干净结构化棱柱网格上发散。

    为什么要把"它现在会发散"写成测试：plate_demo 363k 的 P2 重测每步
    150 s、第 68 步发散，调试循环极慢；同一个失效在这里 1152 单元、
    **一分钟一轮**就能复现。任何针对 P2 发散的修复都应当先在这上面验证，
    而这条测试会在修复生效的那一刻**失败**，强制更新文档与判据 ——
    这正是想要的行为，不要通过放宽判据来"修"它。

    这张网格**全是规整棱柱、无退化单元、均匀分布**，算例是**层流**且有
    精确解 —— 所以它同时排除了两条常被怀疑的成因：网格质量、湍流模型。

    实测（2026-09-18）：

        P2, CFL 0.1    第 75 步 nan   |w| 0 -> 0.037(25) -> 22.57(50)
        P2, CFL 0.03   第 220 步 nan
        P2, CFL 0.01   400 步未发散但 |w| 涨到 0.632、残差 6e6 -> 1.4e9
        P1, CFL 0.1    22000 步不发散，|w| 饱和在 ~1.3（4.3% of U）

    降 CFL 只推迟不解决，是这类失稳的既有特征。P1 上同一模态**饱和**、
    P2 上**无界增长** —— 差别只在阶数。
    """

    def test_p2_still_diverges(self):
        step, _ = _run_until_nonfinite(2, _NX, 0.1, _P2_DIVERGES_WITHIN)
        assert step is not None, (
            f"P2 在 {_P2_DIVERGES_WITHIN} 步内**没有**发散 —— 若这是真实"
            f"修复，请把本类改成正向判据（P2 必须稳定）并更新模块文档；"
            f"不要靠放宽判据让它继续通过。")
        assert step <= _P2_DIVERGES_WITHIN

    def test_p1_on_the_same_mesh_stays_bounded(self):
        """同一张网格上 P1 必须稳定且 |w| 有界 —— 这是正向要求。

        没有这一条，上面那条可能只是因为整个算例设置有问题。
        """
        step, worst_w = _run_until_nonfinite(1, _NX, 0.1, _P1_MUST_SURVIVE)
        assert step is None, f"P1 在第 {step} 步就发散了 —— 算例本身有问题"
        assert worst_w < _P1_W_MAX, (
            f"P1 的伪横流 {100 * worst_w:.3f}% of U 超出已记录量级 "
            f"{100 * _P1_W_MAX:.0f}%，说明比 2026-09-18 记录的情况更糟")
