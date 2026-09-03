"""GPU P>=1 无粘界面校正（gpu_inviscid.py::_compute_interface_correction_gpu）
与 CPU 参考实现的交叉验证。

2026-08-23 新增。全仓库此前没有任何 `compute_inviscid_residual_fr_gpu`
的 crosscheck 测试——这次修复 GPU 粘性界面项（gpu_viscous.py）时，
用户顺带要求排查 gpu_inviscid.py 现有无粘界面校正是否有同类
owner_is_primary/neighbor_is_primary 过滤缺失问题，排查确认：不仅缺
这层过滤，还复现了 inviscid_kernel.py 模块文档明确记录、已被真实网格
验证证伪的"owner/neighbor 两侧共享同一个 AUSM+up 通量"反模式（该文档
记录自由流场残差因此从 9e-5 恶化到 3.1e7），并且 scatter-add 时遗漏了
CPU 版 `correction[...] += -contrib/dj` 的负号。三处一并修复，逐字仿照
inviscid_kernel.py 的两段独立结构（owner-primary 过滤块 + neighbor-
primary 过滤块，各自独立调用一次 AUSM+up）。

本机没有 CuPy/CUDA（`pytest.importorskip("cupy")` 会让整个文件在模块
级别跳过），无法在这台机器上真实执行验证——如实说明，不能声称在本机
验证过 GPU 数值结果。在有真实 GPU 的环境上运行本文件才是这次修复的
最终验证。
"""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from autoflowcfd.core.fr_residual.inviscid import (
    compute_inviscid_residual_fr, primitive_to_conserved, DefaultGhostProvider,
)
from autoflowcfd.core.gpu.residual.gpu_inviscid import compute_inviscid_residual_fr_gpu

from .test_fr_residual_inviscid import _build_synthetic_mixed_mesh

MACH_REF = 0.2


@pytest.mark.parametrize("order,rel_tol", [(1, 1e-6), (2, 1e-5)])
def test_gpu_matches_cpu_uniform_flow(order, rel_tol):
    """均匀自由流场：F(U,U)=F(U) 相容性下两侧实现都应给出接近零的
    残差，彼此的差异也应是浮点噪声量级。"""
    mesh = _build_synthetic_mixed_mesh(order)
    rho_inf, u_inf, v_inf, w_inf, p_inf = 1.225, 30.0, 5.0, -3.0, 101325.0
    Q_inf = np.array([rho_inf, u_inf, v_inf, w_inf, p_inf])
    U_inf = primitive_to_conserved(Q_inf)
    U = np.tile(U_inf, (mesh.n_cells, mesh.n_sps_per_cell, 1))

    cpu_residual = compute_inviscid_residual_fr(U, mesh, mesh.operators, mach_ref=MACH_REF)
    gpu_residual = compute_inviscid_residual_fr_gpu(U, mesh, mesh.operators, mach_ref=MACH_REF)
    gpu_residual_np = cp.asnumpy(gpu_residual) if not isinstance(gpu_residual, np.ndarray) else gpu_residual

    max_diff = np.max(np.abs(cpu_residual - gpu_residual_np))
    rel_diff = max_diff / p_inf
    assert rel_diff < rel_tol, f"P={order}: max|cpu-gpu|={max_diff:.3e}, rel={rel_diff:.3e}"


@pytest.mark.parametrize("order", [1, 2])
def test_gpu_matches_cpu_nonuniform_perturbed_flow(order):
    """非均匀扰动流场：覆盖 owner/neighbor 两侧各自独立通量求值、
    alignment<0.5 回退路径。"""
    mesh = _build_synthetic_mixed_mesh(order)
    rng = np.random.default_rng(order * 5000 + 17)

    rho_inf, u_inf, v_inf, w_inf, p_inf = 1.225, 30.0, 5.0, -3.0, 101325.0
    Q_inf = np.array([rho_inf, u_inf, v_inf, w_inf, p_inf])
    U_inf = primitive_to_conserved(Q_inf)
    U = np.tile(U_inf, (mesh.n_cells, mesh.n_sps_per_cell, 1))

    n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell
    from autoflowcfd.core.fr_residual.inviscid import conserved_to_primitive
    Q = conserved_to_primitive(U)
    Q[..., 0] *= 1.0 + rng.uniform(-0.05, 0.05, size=(n_cells, n_sps))
    Q[..., 1] += rng.uniform(-5.0, 5.0, size=(n_cells, n_sps))
    Q[..., 2] += rng.uniform(-5.0, 5.0, size=(n_cells, n_sps))
    Q[..., 3] += rng.uniform(-5.0, 5.0, size=(n_cells, n_sps))
    Q[..., 4] *= 1.0 + rng.uniform(-0.05, 0.05, size=(n_cells, n_sps))
    U = primitive_to_conserved(Q)

    cpu_residual = compute_inviscid_residual_fr(U, mesh, mesh.operators, mach_ref=MACH_REF)
    gpu_residual = compute_inviscid_residual_fr_gpu(U, mesh, mesh.operators, mach_ref=MACH_REF)
    gpu_residual_np = cp.asnumpy(gpu_residual) if not isinstance(gpu_residual, np.ndarray) else gpu_residual

    max_diff = np.max(np.abs(cpu_residual - gpu_residual_np))
    scale = max(np.max(np.abs(cpu_residual)), 1.0)
    assert max_diff < max(1e-6, scale * 1e-6), f"P={order}: max|cpu-gpu|={max_diff:.3e}, scale={scale:.3e}"


def test_gpu_split_prism_quad_face_no_duplicate_counting():
    """回归本次修复的核心问题：棱柱四边形侧面被拆分成 2 条子面记录的
    场景（约 5% 的棱柱），GPU 与 CPU 必须给出一致的残差——此前 GPU 版
    对这批面会因为缺失 owner_is_primary 过滤而重复计入/用全零假邻居态
    算出错误通量。用 order=2 的混合网格（含棱柱，_build_synthetic_
    mixed_mesh 保证棱柱四边形侧面被拆分）覆盖。"""
    mesh = _build_synthetic_mixed_mesh(2)
    flat = mesh.face_connectivity
    from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry
    flat_geom = get_flat_face_geometry(mesh, mesh.operators)
    n_split = int(np.sum(~flat_geom.owner_is_primary) + np.sum(~flat_geom.neighbor_is_primary))
    assert n_split > 0, "测试网格未包含拆分的棱柱四边形侧面，无法覆盖本次修复的场景"

    rng = np.random.default_rng(999)
    rho_inf, u_inf, v_inf, w_inf, p_inf = 1.225, 30.0, 5.0, -3.0, 101325.0
    Q_inf = np.array([rho_inf, u_inf, v_inf, w_inf, p_inf])
    U_inf = primitive_to_conserved(Q_inf)
    U = np.tile(U_inf, (mesh.n_cells, mesh.n_sps_per_cell, 1))
    n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell
    from autoflowcfd.core.fr_residual.inviscid import conserved_to_primitive
    Q = conserved_to_primitive(U)
    Q[..., 1] += rng.uniform(-3.0, 3.0, size=(n_cells, n_sps))
    Q[..., 4] *= 1.0 + rng.uniform(-0.03, 0.03, size=(n_cells, n_sps))
    U = primitive_to_conserved(Q)

    cpu_residual = compute_inviscid_residual_fr(U, mesh, mesh.operators, mach_ref=MACH_REF)
    gpu_residual = compute_inviscid_residual_fr_gpu(U, mesh, mesh.operators, mach_ref=MACH_REF)
    gpu_residual_np = cp.asnumpy(gpu_residual) if not isinstance(gpu_residual, np.ndarray) else gpu_residual

    max_diff = np.max(np.abs(cpu_residual - gpu_residual_np))
    scale = max(np.max(np.abs(cpu_residual)), 1.0)
    assert max_diff < max(1e-6, scale * 1e-6), f"max|cpu-gpu|={max_diff:.3e}, scale={scale:.3e}"


@pytest.mark.parametrize("order,rel_tol", [(1, 1e-6), (2, 1e-5)])
def test_gpu_matches_cpu_native_tet_basis_uniform_flow(order, rel_tol):
    """native 四面体基（路径C）GPU 移植（2026-09-02）crosscheck——均匀
    自由流场，覆盖体积项 D_native_tet_padded 分派 + 界面项
    boundary_extrap_native/lift_native 分派全部三处（owner-primary/
    neighbor-primary 的自身面外插 + 面校正分配）。本机没有 CuPy/CUDA，
    这个测试和上面几个一样从未在本机实际执行过，需要在真实 GPU 环境
    运行确认。"""
    mesh = _build_synthetic_mixed_mesh(order, tet_basis_mode="native")
    rho_inf, u_inf, v_inf, w_inf, p_inf = 1.225, 30.0, 5.0, -3.0, 101325.0
    Q_inf = np.array([rho_inf, u_inf, v_inf, w_inf, p_inf])
    U_inf = primitive_to_conserved(Q_inf)
    U = np.tile(U_inf, (mesh.n_cells, mesh.n_sps_per_cell, 1))

    cpu_residual = compute_inviscid_residual_fr(U, mesh, mesh.operators, mach_ref=MACH_REF)
    gpu_residual = compute_inviscid_residual_fr_gpu(U, mesh, mesh.operators, mach_ref=MACH_REF)
    gpu_residual_np = cp.asnumpy(gpu_residual) if not isinstance(gpu_residual, np.ndarray) else gpu_residual

    max_diff = np.max(np.abs(cpu_residual - gpu_residual_np))
    rel_diff = max_diff / p_inf
    assert rel_diff < rel_tol, f"native P={order}: max|cpu-gpu|={max_diff:.3e}, rel={rel_diff:.3e}"


@pytest.mark.parametrize("order", [1, 2])
def test_gpu_matches_cpu_native_tet_basis_nonuniform_perturbed_flow(order):
    """native 四面体基 GPU 移植——非均匀扰动流场，覆盖 native 面的
    AUSM+up 非线性通量求值（side_factor 固定+1 分支）+ DG 提升算子
    分配。"""
    mesh = _build_synthetic_mixed_mesh(order, tet_basis_mode="native")
    rng = np.random.default_rng(order * 7000 + 31)

    rho_inf, u_inf, v_inf, w_inf, p_inf = 1.225, 30.0, 5.0, -3.0, 101325.0
    Q_inf = np.array([rho_inf, u_inf, v_inf, w_inf, p_inf])
    U_inf = primitive_to_conserved(Q_inf)
    U = np.tile(U_inf, (mesh.n_cells, mesh.n_sps_per_cell, 1))

    n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell
    from autoflowcfd.core.fr_residual.inviscid import conserved_to_primitive
    Q = conserved_to_primitive(U)
    Q[..., 0] *= 1.0 + rng.uniform(-0.05, 0.05, size=(n_cells, n_sps))
    Q[..., 1] += rng.uniform(-5.0, 5.0, size=(n_cells, n_sps))
    Q[..., 2] += rng.uniform(-5.0, 5.0, size=(n_cells, n_sps))
    Q[..., 3] += rng.uniform(-5.0, 5.0, size=(n_cells, n_sps))
    Q[..., 4] *= 1.0 + rng.uniform(-0.05, 0.05, size=(n_cells, n_sps))
    U = primitive_to_conserved(Q)

    cpu_residual = compute_inviscid_residual_fr(U, mesh, mesh.operators, mach_ref=MACH_REF)
    gpu_residual = compute_inviscid_residual_fr_gpu(U, mesh, mesh.operators, mach_ref=MACH_REF)
    gpu_residual_np = cp.asnumpy(gpu_residual) if not isinstance(gpu_residual, np.ndarray) else gpu_residual

    max_diff = np.max(np.abs(cpu_residual - gpu_residual_np))
    scale = max(np.max(np.abs(cpu_residual)), 1.0)
    assert max_diff < max(1e-6, scale * 1e-6), (
        f"native P={order}: max|cpu-gpu|={max_diff:.3e}, scale={scale:.3e}"
    )
