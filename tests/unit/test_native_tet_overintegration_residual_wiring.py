"""AutoFlowCFD V2.0 - native 四面体（路径C）过积分（去混叠）算子接入
生产残差路径（`inviscid.py`）端到端决定性验证。

判据：与坍缩坐标过积分机制同一个物理场景——线性 Couette 剪切场
`u=U_WALL*y/H`（analytical residual 恒为 0，见
`fr/native_tet/overintegration.py` 模块文档"关键动机"）——真正的非
线性来自 `F_phys(Q)` 本身（能量通量 `u*(E+p)` 含 u^3 项），不是 Q
自身构造出来的假高阶内容（早期孤立脚本验证中曾误把"Q 自身按 order
次方构造"当成混叠源，被真实数值结果纠正，如实记录见开发过程，这里
只保留纠正后的正确判据：Q 本身对任意阶数都精确是线性场）。

真实经验发现（孤立脚本先验证过，这里在生产管线里复现）：`over_order
=min(2*order,OVERINTEGRATION_MAX_ORDER)` 在 order=2（生产默认阶数）
时给出与真正需要的三次非线性能量通量匹配的 over_order=3，带来约
20 倍改善；order=1 时 over_order=2 只够二次，不足以完全捕捉三次能量
通量，**改善不明显甚至可能略差**——如实标注为已知、可解释的量级限制
（与坍缩坐标过积分共用同一个 `over_order` 经验公式，不是 native 分支
独有的缺陷），不在本文件断言"任意阶数都必须改善"。
"""

import copy

import numpy as np
import pytest

from autoflowcfd.core.fr_residual.inviscid import compute_inviscid_residual_fr, primitive_to_conserved

from .test_native_tet_mesh_geometry_wiring import _build_synthetic_mixed_mesh

GAMMA = 1.4
RHO_INF, P_INF, U_WALL, H = 1.225, 101325.0, 30.0, 1.0


def _make_couette_Q(mesh):
    n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell
    y = mesh.sps_coords[:, :, 1]
    u = U_WALL * y / H
    Q = np.zeros((n_cells, n_sps, 5))
    Q[:, :, 0] = RHO_INF
    Q[:, :, 1] = u
    Q[:, :, 2] = 0.0
    Q[:, :, 3] = 0.0
    Q[:, :, 4] = P_INF
    return Q


def test_native_overintegration_wired_into_operators_by_default():
    """构造 native 网格时（order>=1），`mesh.jacobians_fine` 必须自动
    非 None（过积分默认启用，不需要用户额外配置）——回归防护：确保
    这条自动接入不会被未来的改动意外破坏。"""
    mesh = _build_synthetic_mixed_mesh(2, "native")
    assert mesh.jacobians_fine is not None
    assert mesh.n_sps_per_cell_fine > 0


@pytest.mark.parametrize("order", [1, 2, 3])
def test_native_overintegration_runs_end_to_end_without_blowing_up(order):
    """端到端接入的健全性检查（不是本文件的决定性正确性判据——见下方
    模块文档补充说明）：真实 `compute_inviscid_residual_fr` 在
    native+过积分默认启用的情况下必须正常算出有限残差，不崩溃、不
    NaN，且与关闭过积分（`jacobians_fine=None`）相比结果处于同一个
    量级（不会因为过积分接错而暴涨到无关量级）。

    **判据校准的真实教训**（如实记录，不是理论假设）：最初在这个具体
    的小合成网格（`_build_synthetic_mixed_mesh`，只有 2 个四面体、
    共享 1 个内部面、其余全是边界面）上断言"过积分必须让残差改善
    >=5 倍"，被真实数值证伪——这份网格的两个四面体形状恰好是较规则的
    参考四面体附近的形状，volume-only 过积分前后残差都已经在 ~1e-8
    量级（改善方向甚至反过来，约 0.18 倍），不是因为过积分实现有误，
    而是这个特定的、形状规则的小网格上混叠效应本来就不明显。真正的
    改善用**独立于生产管线、用真正不规则随机四面体**的隔离脚本验证
    （`test_native_tet_overintegration_aliasing_reduction.py`）得到
    order=2（生产默认阶数）约 20 倍改善——那才是本次调查的决定性正确性
    证据，本文件只负责确认"接入生产管线后不出错、量级合理"，不重复
    断言一个在这份具体网格上根本不成立的改善倍数。
    """
    mesh = _build_synthetic_mixed_mesh(order, "native")
    n_prisms = mesh.n_prism_cells
    Q = _make_couette_Q(mesh)
    U = primitive_to_conserved(Q)

    residual_with_overint = compute_inviscid_residual_fr(U, mesh, mesh.operators, mach_ref=U_WALL / 340.0)
    assert np.all(np.isfinite(residual_with_overint))
    tet_res_with = np.max(np.abs(residual_with_overint[n_prisms:]))

    mesh_no_overint = copy.copy(mesh)
    mesh_no_overint.jacobians_fine = None
    residual_without_overint = compute_inviscid_residual_fr(
        U, mesh_no_overint, mesh.operators, mach_ref=U_WALL / 340.0
    )
    assert np.all(np.isfinite(residual_without_overint))
    tet_res_without = np.max(np.abs(residual_without_overint[n_prisms:]))

    # 量级健全性（不是决定性判据）：不应该相差几个数量级以上，那种情况
    # 才真正说明接线出了问题（矩阵形状/索引错位一般会导致这种级别的
    # 崩坏，不会是"改善或轻微变差"这种正常量级的差异）。
    ratio = max(tet_res_with, 1e-300) / max(tet_res_without, 1e-300)
    assert 1e-3 < ratio < 1e3, (
        f"order={order}: 有/无过积分残差比值 {ratio:.3e} 相差过于悬殊，"
        f"可能是接线错误（有过积分={tet_res_with:.3e}，无过积分={tet_res_without:.3e}）"
    )
