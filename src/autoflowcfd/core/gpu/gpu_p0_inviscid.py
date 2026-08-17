"""
AutoFlowCFD V2.0 - P0 无粘残差的 CuPy CUDA 实现

从 core/backend/fr_gpu_p0.py（numba.cuda 版本）迁移而来，统一使用 CuPy 框架。
算法完全一致：经典分片常数有限体积格式，每个面独立取真实几何法向/面积，
调用 AUSM+up 黎曼求解器，原子累加到 owner/neighbor 单元残差。

与 numba.cuda 版本的对应关系：
- @cuda.jit → cp.RawKernel（CUDA C 代码嵌入）
- cuda.atomic.add → atomicAdd（CUDA C 内置）
- cuda.to_device → cp.asarray
- d_residual.copy_to_host() → cp.asnumpy

正确性验证：与 CPU 版 _compute_inviscid_residual_fv_p0 数值对比，
见 tests/unit/test_gpu_p0_inviscid.py。
"""

import numpy as np
from typing import Callable, Optional
from loguru import logger

from autoflowcfd.core.gpu import gpu_available, get_cupy

GAMMA = 1.4

# ─── AUSM+up CUDA C 核心（嵌入 RawKernel）──────────────────────
# 逐面计算：每个 CUDA 线程处理一个面
# - 读取 owner/neighbor 原始变量
# - 计算 AUSM+up 公共通量
# - 原子累加到残差
_P0_INVISCID_CUDA_CODE = r"""
extern "C" __global__
void p0_inviscid_residual(
    const int* owner_cell,
    const int* neighbor_cell,
    const bool* is_boundary,
    const double* normal,      // (n_faces, 3)
    const double* area_w,      // (n_faces,)
    const double* Q_all,       // (n_cells, 5) 原始变量
    const double* Q_ghost,     // (n_faces, 5) 边界幽灵态
    const double* cell_volumes,// (n_cells,)
    double* residual,          // (n_cells, 5) 输出残差
    const int n_faces
) {
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= n_faces) return;

    int oc = owner_cell[f];
    double nx = normal[f * 3 + 0];
    double ny = normal[f * 3 + 1];
    double nz = normal[f * 3 + 2];
    double aw = area_w[f];

    // Owner 侧原始变量
    double rhoL = Q_all[oc * 5 + 0];
    double uL   = Q_all[oc * 5 + 1];
    double vL   = Q_all[oc * 5 + 2];
    double wL   = Q_all[oc * 5 + 3];
    double pL   = Q_all[oc * 5 + 4];

    // Neighbor 侧原始变量
    double rhoR, uR, vR, wR, pR;
    bool is_bnd = is_boundary[f];
    if (is_bnd) {
        rhoR = Q_ghost[f * 5 + 0];
        uR   = Q_ghost[f * 5 + 1];
        vR   = Q_ghost[f * 5 + 2];
        wR   = Q_ghost[f * 5 + 3];
        pR   = Q_ghost[f * 5 + 4];
    } else {
        int nc = neighbor_cell[f];
        rhoR = Q_all[nc * 5 + 0];
        uR   = Q_all[nc * 5 + 1];
        vR   = Q_all[nc * 5 + 2];
        wR   = Q_all[nc * 5 + 3];
        pR   = Q_all[nc * 5 + 4];
    }

    // ── AUSM+up 数值通量 ──
    // 与 core/fr_kernels.py::compute_ausm_up_flux 逐字对应
    double gamma = 1.4;
    double alpha = 0.1875;
    double beta_param = 0.5;

    double rhoL_s = fmax(rhoL, 1e-6);
    double rhoR_s = fmax(rhoR, 1e-6);
    double pL_s = fmax(pL, 10.0);
    double pR_s = fmax(pR, 10.0);

    double unL = uL * nx + vL * ny + wL * nz;
    double unR = uR * nx + vR * ny + wR * nz;

    double aL = sqrt(fmax(gamma * pL_s / rhoL_s, 1e-10));
    double aR = sqrt(fmax(gamma * pR_s / rhoR_s, 1e-10));

    double M_L = unL / fmax(aL, 1e-10);
    double M_R = unR / fmax(aR, 1e-10);

    double a_half = 0.5 * (aL + aR);
    double rho_half = 0.5 * (rhoL_s + rhoR_s);
    double Mbar2 = (unL * unL + unR * unR) / (2.0 * a_half * a_half);
    double Ma_ref = 0.1;
    double M0_sq = fmin(1.0, fmax(Mbar2, Ma_ref * Ma_ref));
    double sqrt_M0_sq = sqrt(M0_sq);
    double fa = sqrt_M0_sq * (2.0 - sqrt_M0_sq);
    fa = fmax(fa, 1e-6);

    // 质量通量分裂 M+ / M-
    double Mp_L, Mm_R;
    if (fabs(M_L) >= 1.0) {
        Mp_L = 0.5 * (M_L + fabs(M_L));
    } else {
        Mp_L = 0.25 * (M_L + 1.0) * (M_L + 1.0) + alpha * (M_L * M_L - 1.0) * (M_L * M_L - 1.0);
    }
    if (fabs(M_R) >= 1.0) {
        Mm_R = 0.5 * (M_R - fabs(M_R));
    } else {
        Mm_R = -0.25 * (M_R - 1.0) * (M_R - 1.0) - alpha * (M_R * M_R - 1.0) * (M_R * M_R - 1.0);
    }
    double M_half = Mp_L + Mm_R;

    // Mp 压力扩散项
    double Kp = 0.25;
    double sigma_p = 1.0;
    double one_minus_sigma = 1.0 - sigma_p * Mbar2;
    if (one_minus_sigma < 0.0) one_minus_sigma = 0.0;
    double Mp = -(Kp / fa) * one_minus_sigma * (pR_s - pL_s) / (rho_half * a_half * a_half);
    double mass_flux = 0.5 * (rhoL_s * aL + rhoR_s * aR) * (M_half + Mp);

    // 压力分裂 P+ / P-
    double Pp_L, Pm_R;
    if (fabs(M_L) >= 1.0) {
        double sign_ML = (M_L > 0.0) ? 1.0 : ((M_L < 0.0) ? -1.0 : 0.0);
        Pp_L = 0.5 * (1.0 + sign_ML);
    } else {
        Pp_L = 0.25 * ((M_L + 1.0) * (M_L + 1.0) * (2.0 - M_L)
               + beta_param * M_L * (M_L * M_L - 1.0) * (M_L * M_L - 1.0));
    }
    if (fabs(M_R) >= 1.0) {
        double sign_MR = (M_R > 0.0) ? 1.0 : ((M_R < 0.0) ? -1.0 : 0.0);
        Pm_R = 0.5 * (1.0 - sign_MR);
    } else {
        Pm_R = 0.25 * ((M_R - 1.0) * (M_R - 1.0) * (2.0 + M_R)
               - beta_param * M_R * (M_R * M_R - 1.0) * (M_R * M_R - 1.0));
    }

    // pu 速度扩散项
    double Ku = 0.75;
    double p_half = Pp_L * pL_s + Pm_R * pR_s
        - Ku * Pp_L * Pm_R * (rhoL_s + rhoR_s) * fa * a_half * (unR - unL);

    // 上风通量
    double flux[5];
    bool upwind_L = (mass_flux >= 0.0);
    flux[0] = mass_flux;
    flux[1] = mass_flux * (upwind_L ? uL : uR) + p_half * nx;
    flux[2] = mass_flux * (upwind_L ? vL : vR) + p_half * ny;
    flux[3] = mass_flux * (upwind_L ? wL : wR) + p_half * nz;

    double hL = gamma / (gamma - 1.0) * pL_s / rhoL_s + 0.5 * (uL * uL + vL * vL + wL * wL);
    double hR = gamma / (gamma - 1.0) * pR_s / rhoR_s + 0.5 * (uR * uR + vR * vR + wR * wR);
    flux[4] = mass_flux * (upwind_L ? hL : hR);

    // ── 原子累加到残差 ──
    double vol_o = cell_volumes[oc];
    for (int v = 0; v < 5; v++) {
        atomicAdd(&residual[oc * 5 + v], -flux[v] * aw / vol_o);
    }

    if (!is_bnd) {
        int nc = neighbor_cell[f];
        double vol_n = cell_volumes[nc];
        for (int v = 0; v < 5; v++) {
            atomicAdd(&residual[nc * 5 + v], flux[v] * aw / vol_n);
        }
    }
}
"""


def _get_p0_kernel():
    """获取编译好的 P0 无粘残差 CUDA kernel（懒加载 + 缓存）。"""
    cp = get_cupy()
    if cp is None:
        raise RuntimeError("CuPy is not available")
    kernel = cp.RawKernel(
        _P0_INVISCID_CUDA_CODE,
        'p0_inviscid_residual',
        options=('--std=c++11',),
    )
    return kernel


_p0_kernel_cache = None


def _get_cached_p0_kernel():
    """全局缓存的 P0 kernel 实例。"""
    global _p0_kernel_cache
    if _p0_kernel_cache is None:
        _p0_kernel_cache = _get_p0_kernel()
    return _p0_kernel_cache


def compute_inviscid_residual_p0_cupy(
    U: np.ndarray,
    mesh,
    boundary_ghost_provider: Optional[Callable] = None,
    device_id: int = 0,
) -> np.ndarray:
    """P0 无粘残差的 CuPy CUDA 实现。

    函数签名/返回值与 core/fr_residual_inviscid.py::_compute_inviscid_residual_fv_p0
    完全一致，可以互相替换。

    Args:
        U: 守恒变量 (n_cells, n_sps, n_vars)
        mesh: HighOrderMesh（n_points_1d == 1）
        boundary_ghost_provider: 边界幽灵态提供者
        device_id: GPU 设备 ID

    Returns:
        residual: (n_cells, 1, 5) 残差数组

    Raises:
        RuntimeError: CuPy 不可用或非 P0 网格
    """
    cp = get_cupy()
    if cp is None:
        raise RuntimeError("CuPy is not available")

    if mesh.n_points_1d != 1:
        raise RuntimeError(
            f"compute_inviscid_residual_p0_cupy only supports P0 meshes "
            f"(n_points_1d=1), got n_points_1d={mesh.n_points_1d}"
        )
    if mesh.cell_volumes is None:
        raise RuntimeError(
            "mesh.cell_volumes not available - required for P0 finite-volume residual"
        )

    from autoflowcfd.core.fr_residual.inviscid import (
        conserved_to_primitive, DefaultGhostProvider
    )

    n_cells = mesh.n_cells
    Q_all = conserved_to_primitive(U[..., :5])[:, 0, :].astype(np.float64)

    fc = mesh.face_connectivity
    ffp_list = mesh.face_flux_points
    n_faces = fc.n_faces
    ghost_provider = (
        boundary_ghost_provider
        if boundary_ghost_provider is not None
        else DefaultGhostProvider()
    )

    # ── 准备面几何数据（CPU 侧）──
    owner_cell = fc.owner_cell.astype(np.int32)
    neighbor_cell = np.where(fc.is_boundary, 0, fc.neighbor_cell).astype(np.int32)
    is_boundary = fc.is_boundary.astype(np.bool_)
    normal = np.empty((n_faces, 3), dtype=np.float64)
    area_w = np.empty((n_faces,), dtype=np.float64)
    Q_ghost = np.zeros((n_faces, 5), dtype=np.float64)

    for f in range(n_faces):
        ffp = ffp_list[f]
        normal[f, :] = ffp.true_normal[0]
        area_w[f] = ffp.true_area_weight[0]
        if is_boundary[f]:
            Q_owner_fp = Q_all[owner_cell[f]: owner_cell[f] + 1]
            Q_ghost[f, :] = ghost_provider(f, Q_owner_fp, ffp.true_normal)[0]

    cell_volumes = mesh.cell_volumes.astype(np.float64)

    # ── 传输到 GPU ──
    with cp.cuda.Device(device_id):
        d_owner = cp.asarray(owner_cell)
        d_neighbor = cp.asarray(neighbor_cell)
        d_is_boundary = cp.asarray(is_boundary)
        d_normal = cp.asarray(normal)
        d_area_w = cp.asarray(area_w)
        d_Q = cp.asarray(Q_all)
        d_Q_ghost = cp.asarray(Q_ghost)
        d_volumes = cp.asarray(cell_volumes)
        d_residual = cp.zeros((n_cells, 5), dtype=np.float64)

        # ── 启动 CUDA kernel ──
        threads_per_block = 128
        blocks_per_grid = (n_faces + threads_per_block - 1) // threads_per_block

        kernel = _get_cached_p0_kernel()
        kernel(
            (blocks_per_grid,), (threads_per_block,),
            (d_owner, d_neighbor, d_is_boundary, d_normal, d_area_w,
             d_Q, d_Q_ghost, d_volumes, d_residual, np.int32(n_faces))
        )
        cp.cuda.Stream.null.synchronize()

        # ── 取回结果 ──
        residual5 = cp.asnumpy(d_residual)

    return residual5[:, None, :]


def compute_inviscid_residual_p0_cupy_gpu_resident(
    Q_gpu,
    owner_cell_gpu, neighbor_cell_gpu, is_boundary_gpu,
    normal_gpu, area_w_gpu, cell_volumes_gpu,
    Q_ghost_gpu, n_cells: int, n_faces: int,
):
    """P0 无粘残差的 GPU 常驻版本（数据已在 GPU 上，无需传输）。

    用于 GPUFRSolver 内部，所有数据都是 CuPy 数组，避免 CPU↔GPU 传输。

    Args:
        Q_gpu: (n_cells, 5) 原始变量，CuPy 数组
        owner_cell_gpu, neighbor_cell_gpu, is_boundary_gpu: 面连接关系
        normal_gpu, area_w_gpu: 面法向和面积权重
        cell_volumes_gpu: 单元体积
        Q_ghost_gpu: 边界幽灵态
        n_cells: 单元数
        n_faces: 面数

    Returns:
        residual_gpu: (n_cells, 1, 5) CuPy 残差数组
    """
    cp = get_cupy()
    if cp is None:
        raise RuntimeError("CuPy is not available")

    d_residual = cp.zeros((n_cells, 5), dtype=np.float64)

    threads_per_block = 128
    blocks_per_grid = (n_faces + threads_per_block - 1) // threads_per_block

    kernel = _get_cached_p0_kernel()
    kernel(
        (blocks_per_grid,), (threads_per_block,),
        (owner_cell_gpu, neighbor_cell_gpu, is_boundary_gpu,
         normal_gpu, area_w_gpu, Q_gpu, Q_ghost_gpu,
         cell_volumes_gpu, d_residual, np.int32(n_faces))
    )
    cp.cuda.Stream.null.synchronize()

    return d_residual[:, None, :]
