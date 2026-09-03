"""GPU 粘性界面项（BR1 平均 + 边界 IP 罚项）与 CPU 参考实现的交叉验证。

2026-08-23 新增，配套 gpu_viscous.py::compute_viscous_residual_fr_gpu
的界面项移植（之前该函数只算体积散度项，完全没有界面校正，是一个
在 `solve steady --backend gpu` 下真实可达的生产级缺陷，见
gpu_viscous.py 模块文档）。与
test_fr_viscous_flux_kernel_crosscheck.py（CPU 新旧 kernel 交叉验证）
是同一个验证思路，换成 CPU 参考 vs GPU 实现。

本机没有 CuPy/CUDA（`pytest.importorskip("cupy")` 会让整个文件在模块
级别跳过），无法在这台机器上真实执行验证——这一点与
test_gpu_modules.py 模块文档记录的限制完全一致，如实说明，不能声称
在本机验证过 GPU 数值结果。在有真实 GPU 的环境上运行本文件才是这次
移植的最终验证。
"""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from autoflowcfd.core.fr_residual.inviscid import (
    conserved_to_primitive, primitive_to_conserved, DefaultGhostProvider,
)
from autoflowcfd.core.fr_residual.viscous_flux import compute_viscous_residual_fr
from autoflowcfd.core.gpu.residual.gpu_viscous import compute_viscous_residual_fr_gpu

from .test_fr_residual_inviscid import _build_synthetic_mixed_mesh

MU = 1.8e-5
PR = 0.72
PR_T = 0.9


@pytest.mark.parametrize("order,rel_tol", [(1, 1e-6), (2, 1e-5)])
def test_gpu_matches_cpu_uniform_flow(order, rel_tol):
    """均匀自由流场：BR1 跳跃恒为零，两侧实现都应给出接近零的残差，
    彼此的差异也应是浮点噪声量级（不是"都很小"这种弱检验）。"""
    mesh = _build_synthetic_mixed_mesh(order)
    rho_inf, u_inf, v_inf, w_inf, p_inf = 1.225, 30.0, 5.0, -3.0, 101325.0
    Q_inf = np.array([rho_inf, u_inf, v_inf, w_inf, p_inf])
    U_inf = primitive_to_conserved(Q_inf)
    U = np.tile(U_inf, (mesh.n_cells, mesh.n_sps_per_cell, 1))

    cpu_residual = compute_viscous_residual_fr(U, mesh, mesh.operators, MU, PR)
    gpu_residual = compute_viscous_residual_fr_gpu(U, mesh, mesh.operators, MU, PR)
    gpu_residual_np = cp.asnumpy(gpu_residual) if not isinstance(gpu_residual, np.ndarray) else gpu_residual

    max_diff = np.max(np.abs(cpu_residual - gpu_residual_np))
    rel_diff = max_diff / p_inf
    assert rel_diff < rel_tol, f"P={order}: max|cpu-gpu|={max_diff:.3e}, rel={rel_diff:.3e}"


@pytest.mark.parametrize("order", [1, 2])
def test_gpu_matches_cpu_nonuniform_perturbed_flow(order):
    """非均匀扰动流场（含湍流涡粘场），覆盖 BR1 平均、Interior Penalty
    边界罚项、mu_t/Pr_t 耦合路径——与 CPU 交叉验证同名测试覆盖同一批
    分支。"""
    mesh = _build_synthetic_mixed_mesh(order)
    rng = np.random.default_rng(order * 2000 + 3)

    rho_inf, u_inf, v_inf, w_inf, p_inf = 1.225, 30.0, 5.0, -3.0, 101325.0
    Q_inf = np.array([rho_inf, u_inf, v_inf, w_inf, p_inf])
    U_inf = primitive_to_conserved(Q_inf)
    U = np.tile(U_inf, (mesh.n_cells, mesh.n_sps_per_cell, 1))

    n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell
    Q = conserved_to_primitive(U)
    Q[..., 0] *= 1.0 + rng.uniform(-0.05, 0.05, size=(n_cells, n_sps))
    Q[..., 1] += rng.uniform(-5.0, 5.0, size=(n_cells, n_sps))
    Q[..., 2] += rng.uniform(-5.0, 5.0, size=(n_cells, n_sps))
    Q[..., 3] += rng.uniform(-5.0, 5.0, size=(n_cells, n_sps))
    Q[..., 4] *= 1.0 + rng.uniform(-0.05, 0.05, size=(n_cells, n_sps))
    U = primitive_to_conserved(Q)
    mu_t_field = rng.uniform(0.0, 5e-3, size=(n_cells, n_sps))

    cpu_residual = compute_viscous_residual_fr(U, mesh, mesh.operators, MU, PR, mu_t_field=mu_t_field)
    gpu_residual = compute_viscous_residual_fr_gpu(U, mesh, mesh.operators, MU, PR, mu_t_field=mu_t_field)
    gpu_residual_np = cp.asnumpy(gpu_residual) if not isinstance(gpu_residual, np.ndarray) else gpu_residual

    max_diff = np.max(np.abs(cpu_residual - gpu_residual_np))
    scale = max(np.max(np.abs(cpu_residual)), 1.0)
    assert max_diff < max(1e-6, scale * 1e-6), f"P={order}: max|cpu-gpu|={max_diff:.3e}, scale={scale:.3e}"


def test_gpu_wall_boundary_uses_real_ghost_state():
    """回归本次修复的核心问题：GPU 界面项必须真的调用
    boundary_ghost_provider（不能像 gpu_inviscid.py::
    _compute_boundary_ghost_states_gpu 那样用零梯度外插简化），否则
    WALL 无滑移边界的剪应力/IP 罚项不存在。用一个自定义 ghost_provider
    模拟 WALL 无滑移镜像（速度取反），验证 GPU 残差里动量分量确实
    对这个自定义边界条件有响应（不是恒等于自由流残差）。"""
    from autoflowcfd.core.fr_residual.inviscid import DefaultGhostProvider

    class WallMirrorGhostProvider:
        def __call__(self, face_idx, Q_owner, normal):
            Q_g = Q_owner.copy()
            Q_g[..., 1:4] = -Q_g[..., 1:4]
            return Q_g

    mesh = _build_synthetic_mixed_mesh(1)
    rho_inf, u_inf, v_inf, w_inf, p_inf = 1.225, 30.0, 5.0, -3.0, 101325.0
    Q_inf = np.array([rho_inf, u_inf, v_inf, w_inf, p_inf])
    U_inf = primitive_to_conserved(Q_inf)
    U = np.tile(U_inf, (mesh.n_cells, mesh.n_sps_per_cell, 1))

    wall_provider = WallMirrorGhostProvider()
    cpu_residual = compute_viscous_residual_fr(
        U, mesh, mesh.operators, MU, PR, boundary_ghost_provider=wall_provider,
    )
    gpu_residual = compute_viscous_residual_fr_gpu(
        U, mesh, mesh.operators, MU, PR, boundary_ghost_provider=wall_provider,
    )
    gpu_residual_np = cp.asnumpy(gpu_residual) if not isinstance(gpu_residual, np.ndarray) else gpu_residual

    assert np.max(np.abs(cpu_residual)) > 1.0, "WALL 镜像边界条件应产生非零粘性残差"
    max_diff = np.max(np.abs(cpu_residual - gpu_residual_np))
    rel_diff = max_diff / p_inf
    assert rel_diff < 1e-6, f"max|cpu-gpu|={max_diff:.3e}, rel={rel_diff:.3e}"


@pytest.mark.parametrize("order", [1, 2])
def test_gpu_matches_cpu_native_tet_basis_nonuniform_flow(order):
    """native 四面体基（路径C）GPU 移植（2026-09-02）crosscheck——非均匀
    扰动流场+湍流涡粘场，覆盖体积项 D_native_tet_padded 分派 + 界面项
    面校正分配的 lift_native 分派（owner-primary/neighbor-primary 各一
    处）。粘性 kernel 不需要 side_factor/true_normal 安全阀分支（与无粘
    不同，见 gpu_viscous.py 模块文档），Q_o/Q_n 状态外插沿用既有的
    owner_src0/neighbor_src0 机制（该机制本身与 tet_basis_mode 无关，
    已在 CPU 端验证过对 native 面同样正确）。"""
    mesh = _build_synthetic_mixed_mesh(order, tet_basis_mode="native")
    rng = np.random.default_rng(order * 4000 + 41)

    rho_inf, u_inf, v_inf, w_inf, p_inf = 1.225, 30.0, 5.0, -3.0, 101325.0
    Q_inf = np.array([rho_inf, u_inf, v_inf, w_inf, p_inf])
    U_inf = primitive_to_conserved(Q_inf)
    U = np.tile(U_inf, (mesh.n_cells, mesh.n_sps_per_cell, 1))

    n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell
    Q = conserved_to_primitive(U)
    Q[..., 0] *= 1.0 + rng.uniform(-0.05, 0.05, size=(n_cells, n_sps))
    Q[..., 1] += rng.uniform(-5.0, 5.0, size=(n_cells, n_sps))
    Q[..., 2] += rng.uniform(-5.0, 5.0, size=(n_cells, n_sps))
    Q[..., 3] += rng.uniform(-5.0, 5.0, size=(n_cells, n_sps))
    Q[..., 4] *= 1.0 + rng.uniform(-0.05, 0.05, size=(n_cells, n_sps))
    U = primitive_to_conserved(Q)
    mu_t_field = rng.uniform(0.0, 5e-3, size=(n_cells, n_sps))

    cpu_residual = compute_viscous_residual_fr(U, mesh, mesh.operators, MU, PR, mu_t_field=mu_t_field)
    gpu_residual = compute_viscous_residual_fr_gpu(U, mesh, mesh.operators, MU, PR, mu_t_field=mu_t_field)
    gpu_residual_np = cp.asnumpy(gpu_residual) if not isinstance(gpu_residual, np.ndarray) else gpu_residual

    max_diff = np.max(np.abs(cpu_residual - gpu_residual_np))
    scale = max(np.max(np.abs(cpu_residual)), 1.0)
    assert max_diff < max(1e-6, scale * 1e-6), (
        f"native P={order}: max|cpu-gpu|={max_diff:.3e}, scale={scale:.3e}"
    )
