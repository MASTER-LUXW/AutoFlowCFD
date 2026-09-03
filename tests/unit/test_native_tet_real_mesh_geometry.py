"""AutoFlowCFD V2.0 - 四面体路径C（native basis）在真实网格连接关系上
的几何/体积残差验证（Part6 阶段1，体积项部分）。

区别于 `test_native_simplex_basis.py`（用手造的随机四面体样本）：本文件
复用 `test_fr_residual_inviscid.py::_build_synthetic_mixed_mesh` 构造的
**真实通过 `HighOrderMesh.load_from_volume_mesh` 生产管线加载**的网格，
提取其中真实的四面体连接关系（`mesh._fixed_tet_conn`/`_node_coords`），
在这些真实单元上验证路径C的体积几何/残差——比纯随机合成样本更接近
真实使用场景，但仍然只测体积项（不含界面/面通量，那是 Part6 阶段2
范围，尚未实现，见该文档）。
"""

import numpy as np

from autoflowcfd.fr.native_simplex_basis import (
    build_native_tet_operators,
    map_native_tet_to_physical,
    compute_native_tet_jacobian,
)
from autoflowcfd.grid.curved_mapping.curved_mapping import batched_det_inv_3x3
from autoflowcfd.core.fr_operators.flux_kernels import euler_physical_flux_batch

from tests.unit.test_fr_residual_inviscid import _build_synthetic_mixed_mesh

GAMMA = 1.4
RHO_INF, P_INF, U_WALL, H = 1.225, 101325.0, 30.0, 1.0


def _real_tet_cells(order: int):
    """从真实生产网格加载管线里提取真实的四面体单元顶点坐标列表。"""
    mesh = _build_synthetic_mixed_mesh(order)
    assert mesh._fixed_tet_conn is not None and len(mesh._fixed_tet_conn) > 0
    return [mesh._node_coords[conn] for conn in mesh._fixed_tet_conn]


def test_native_jacobian_constant_and_matches_direct_computation_on_real_cells():
    """真实网格四面体单元：`compute_native_tet_jacobian` 给出的常数
    Jacobian，必须与逐点用 `D_native@phys` 算出来的结果处处一致（确认
    "常数矩阵简化"这个优化没有引入误差，是数学恒等式，不是近似）。"""
    order = 2
    ref_rst, D_native = build_native_tet_operators(order)
    for cell_nodes in _real_tet_cells(order):
        phys = map_native_tet_to_physical(ref_rst, cell_nodes)
        det_const, adj_const = compute_native_tet_jacobian(cell_nodes)
        assert det_const > 0, "真实网格里的四面体单元体积应为正（load_from_volume_mesh 已修正朝向）"

        jacobians = np.stack([D_native[:, :, m] @ phys for m in range(3)], axis=1)
        jacobians = np.swapaxes(jacobians, 1, 2)
        det_jacs, inv_jacs = batched_det_inv_3x3(np.ascontiguousarray(jacobians))
        adj_jacs = det_jacs[:, None, None] * inv_jacs

        np.testing.assert_allclose(det_jacs, det_const, rtol=1e-8)
        np.testing.assert_allclose(adj_jacs, np.broadcast_to(adj_const, adj_jacs.shape), rtol=1e-8, atol=1e-8)


def test_native_volume_residual_zero_on_uniform_flow_real_cells():
    """真实网格四面体单元，均匀流场，体积项残差必须近机器精度为零
    （自由流场保持——路径C下这是平凡成立的恒等式，因为几何雅可比是
    常数、任意基都精确表示常数场，但仍然用真实单元数据端到端跑一遍
    确认没有实现疏漏）。"""
    order = 2
    ref_rst, D_native = build_native_tet_operators(order)
    n = ref_rst.shape[0]

    rho_inf, u_inf, v_inf, w_inf, p_inf = 1.225, 30.0, 5.0, -3.0, 101325.0
    for cell_nodes in _real_tet_cells(order):
        det_const, adj_const = compute_native_tet_jacobian(cell_nodes)
        Q = np.zeros((n, 5))
        Q[:, 0] = rho_inf
        Q[:, 1] = u_inf
        Q[:, 2] = v_inf
        Q[:, 3] = w_inf
        Q[:, 4] = p_inf
        F_phys = euler_physical_flux_batch(Q)
        adj_j = np.broadcast_to(adj_const, (n, 3, 3))
        F_tilde = np.matmul(adj_j, F_phys)
        div_comp = np.zeros((n, 5))
        for m in range(3):
            div_comp += D_native[:, :, m] @ F_tilde[:, m, :]
        residual = np.abs(-div_comp / det_const).max()
        assert residual / p_inf < 1e-10, f"自由流场保持失败: rel={residual/p_inf:.3e}"


def test_native_couette_residual_is_small_on_real_cells():
    """真实网格四面体单元（不是随机合成样本），Couette 剪切流解析解
    体积残差量级检查。2026-09-03 更正：本测试原先与坍缩坐标基
    （`ops.D_3d_tet`）的残差做倍数对比，量化 native 相对 collapsed 在
    真实单元上的改善幅度；坍缩坐标四面体基已删除（见 fr/operators.py
    模块文档），对照组不再存在，也没有独立意义（`ops.D_3d_tet` 现在
    只是 `D_native_tet_padded` 的别名，用它算"collapsed 残差"实际是
    退化 Jacobian，不是坍缩坐标基真实行为，见 test_native_simplex_
    basis.py 同一处修复的说明）。改为只验证 native 残差本身处于合理
    量级（真实单元数据，不是合成样本）。
    """
    order = 2
    ref_native, D_native = build_native_tet_operators(order)

    def residual_from_phys_and_D(phys, D):
        y_phys = phys[:, 1]
        jacobians = np.stack([D[:, :, m] @ phys for m in range(3)], axis=1)
        jacobians = np.swapaxes(jacobians, 1, 2)
        det_jacs, inv_jacs = batched_det_inv_3x3(np.ascontiguousarray(jacobians))
        if np.any(det_jacs <= 0):
            return None
        adj_j = det_jacs[:, None, None] * inv_jacs
        n_sps = phys.shape[0]
        Q = np.zeros((n_sps, 5))
        u = U_WALL * y_phys / H
        Q[:, 0] = RHO_INF
        Q[:, 1] = RHO_INF * u
        Q[:, 4] = P_INF / (GAMMA - 1.0) + 0.5 * RHO_INF * u**2
        F_phys = euler_physical_flux_batch(Q)
        F_tilde = np.matmul(adj_j, F_phys)
        div_comp = np.zeros((n_sps, 5))
        for m in range(3):
            div_comp += D[:, :, m] @ F_tilde[:, m, :]
        return np.abs(-div_comp / det_jacs[:, None]).max()

    res_native = []
    for cell_nodes in _real_tet_cells(order):
        phys_native = map_native_tet_to_physical(ref_native, cell_nodes)
        r2 = residual_from_phys_and_D(phys_native, D_native)
        if r2 is not None:
            res_native.append(r2)

    assert len(res_native) >= 1, "真实测试网格里没有找到可用的四面体单元"
    res_native = np.array(res_native)
    scale = RHO_INF * U_WALL
    # 真实观测值约 30（这份合成网格的两个四面体单元数值几乎一致）——
    # 这是裸体积项（不含界面/AUSM+up 修正）的强形式残差，量级由局部
    # 单元形状决定，不是近机器精度；用比观测值宽松一个数量级的阈值，
    # 既能捕捉真实回归，又不因几何细节变化过于脆弱。
    assert np.max(res_native) / scale < 300.0, (
        f"native 四面体真实单元 Couette 残差量级异常：max={np.max(res_native):.3e}"
    )
