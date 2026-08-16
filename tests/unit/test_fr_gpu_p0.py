"""P0 无粘残差 GPU（CUDA）实现的正确性验证 (B-01 阶段1)。

无本地 GPU 硬件，用 numba 自带的 `NUMBA_ENABLE_CUDASIM=1` 纯 Python CUDA
语义模拟器核验——这个环境变量必须在 numba 第一次被导入*之前*设置才会
生效（numba 在导入时读取一次决定用真实 CUDA driver 还是模拟器），所以
本文件本身不在普通 `pytest tests/` 全量运行里自动生效，必须整个测试
进程启动前就设置好这个环境变量：

    NUMBA_ENABLE_CUDASIM=1 python -m pytest tests/unit/test_fr_gpu_p0.py

普通全量回归跑这个文件时，若没有设这个环境变量（且本机也没有真实 GPU），
会在模块级别整体跳过（不是失败）——GPU 加速路径是可选能力，缺硬件/没开
模拟器不代表代码本身有问题。
"""
import os

import numpy as np
import pytest

if os.environ.get("NUMBA_ENABLE_CUDASIM") != "1":
    from autoflowcfd.core.backend.fr_gpu_p0 import gpu_p0_available
    if not gpu_p0_available():
        pytest.skip(
            "requires a real CUDA device or NUMBA_ENABLE_CUDASIM=1 set before process start "
            "(see module docstring)",
            allow_module_level=True,
        )

from autoflowcfd.core.backend.fr_gpu_p0 import compute_inviscid_residual_p0_gpu
from autoflowcfd.core.fr_residual_inviscid import _compute_inviscid_residual_fv_p0, DefaultGhostProvider

from tests.validation._channel_mesh import build_channel_mesh_prism

RHO_INF, P_INF, U_INF = 1.225, 101325.0, 30.0
GAMMA = 1.4


def _build_p0_mesh():
    return build_channel_mesh_prism(order=0, nx=4, ny=3, nz=2, Lx=2.0, H=1.0, Lz=0.5)


def test_gpu_p0_matches_cpu_freestream():
    """均匀自由流场：CPU/GPU 两个实现都应给出机器精度量级的零残差，
    且彼此的差异也应是机器精度量级（不只是"都很小"，而是真的在算
    同一个东西）。
    """
    mesh = _build_p0_mesh()
    n_cells = mesh.n_cells
    U = np.zeros((n_cells, 1, 5))
    U[:, 0, 0] = RHO_INF
    U[:, 0, 1] = RHO_INF * U_INF
    U[:, 0, 4] = P_INF / (GAMMA - 1.0) + 0.5 * RHO_INF * U_INF**2

    ghost = DefaultGhostProvider()
    res_cpu = _compute_inviscid_residual_fv_p0(U, mesh, boundary_ghost_provider=ghost)
    res_gpu = compute_inviscid_residual_p0_gpu(U, mesh, boundary_ghost_provider=ghost)

    assert np.max(np.abs(res_cpu)) < 1e-6
    assert np.max(np.abs(res_gpu)) < 1e-6
    assert np.max(np.abs(res_cpu - res_gpu)) < 1e-9


def test_gpu_p0_matches_cpu_nonuniform_field():
    """非均匀场：验证 GPU kernel 与 CPU 参考实现在真正有非零残差的
    情况下逐单元数值一致（不是两边都恰好为零这种弱检验）。
    """
    mesh = _build_p0_mesh()
    n_cells = mesh.n_cells
    x = mesh.sps_coords[:, 0, 0]
    y = mesh.sps_coords[:, 0, 1]

    u_field = U_INF + 0.3 * U_INF * np.sin(x) * np.cos(y)
    rho = np.full(n_cells, RHO_INF)
    E = P_INF / (GAMMA - 1.0) + 0.5 * rho * u_field**2

    U = np.zeros((n_cells, 1, 5))
    U[:, 0, 0] = rho
    U[:, 0, 1] = rho * u_field
    U[:, 0, 4] = E

    ghost = DefaultGhostProvider()
    res_cpu = _compute_inviscid_residual_fv_p0(U, mesh, boundary_ghost_provider=ghost)
    res_gpu = compute_inviscid_residual_p0_gpu(U, mesh, boundary_ghost_provider=ghost)

    assert np.max(np.abs(res_cpu)) > 1.0, "test field should produce a genuinely non-trivial residual"
    max_abs_diff = np.max(np.abs(res_cpu - res_gpu))
    max_abs_ref = np.max(np.abs(res_cpu))
    assert max_abs_diff < 1e-9 * max(max_abs_ref, 1.0) + 1e-9, (
        f"GPU/CPU P0 residual mismatch: max_abs_diff={max_abs_diff:.6e}, ref scale={max_abs_ref:.6e}"
    )
