"""
AutoFlowCFD V2.0 - P0 无粘残差的 CuPy CUDA 实现（`GPUFRSolver` 的 P0 阶段，数据常驻 GPU）

经典分片常数有限体积格式，每个面独立取真实几何法向/面积，调用 AUSM+up 黎曼求解器，原子累加到
owner/neighbor 单元残差（与 CPU 版 `_compute_inviscid_residual_fv_p0` 同一算法）。

2026-10-04 删除了两个旧入口：带主机<->设备传输的 `compute_inviscid_residual_p0_cupy`（只有
`FRSolver(backend='gpu')` 在用，该分支已随单 GPU 统一到 `GPUFRSolver` 删除）与它的前身
`core/backend/fr_gpu_p0.py`（numba.cuda 版，生产代码零调用）。
"""

import numpy as np
from typing import Optional

from autoflowcfd.core.gpu import get_cupy

GAMMA = 1.4

# ─── AUSM+up CUDA C 核心（嵌入 RawKernel）──────────────────────
# 逐面计算：每个 CUDA 线程处理一个面
# - 读取 owner/neighbor 原始变量
# - 计算 AUSM+up 公共通量（混合拆分面另用幽灵态解一次做面积加权，B-8）
# - 原子累加到残差
_P0_INVISCID_CUDA_CODE = r"""
// 注意：只有 __global__ 入口需要 extern "C"（RawKernel 按名查找）；
// __device__ 辅助函数加 extern "C" 有 nvcc 链接语义风险，不加。
__device__
void ausm_up_flux(
    double rhoL, double uL, double vL, double wL, double pL,
    double rhoR, double uR, double vR, double wR, double pR,
    double nx, double ny, double nz,
    double mach_ref,
    int precond_mode,   // 预处理声速作用域：0=physical(默认)/
                        // 1=pressure_physical/2=legacy，语义与
                        // kernels.py::compute_ausm_up_flux 逐字对应
    double* flux
) {
    // ── AUSM+up 数值通量 ──
    // 与 core/fr_kernels.py::compute_ausm_up_flux 逐字对应
    double gamma = 1.4;

    double rhoL_s = fmax(rhoL, 1e-6);
    double rhoR_s = fmax(rhoR, 1e-6);
    double pL_s = fmax(pL, 10.0);
    double pR_s = fmax(pR, 10.0);

    double unL = uL * nx + vL * ny + wL * nz;
    double unR = uR * nx + vR * ny + wR * nz;

    double aL = sqrt(fmax(gamma * pL_s / rhoL_s, 1e-10));
    double aR = sqrt(fmax(gamma * pR_s / rhoR_s, 1e-10));

    double a_half = 0.5 * (aL + aR);
    double rho_half = 0.5 * (rhoL_s + rhoR_s);
    double Mbar2 = (unL * unL + unR * unR) / (2.0 * a_half * a_half);
    double M0_sq = fmin(1.0, fmax(Mbar2, mach_ref * mach_ref));
    double sqrt_M0_sq = sqrt(M0_sq);
    double fa = sqrt_M0_sq * (2.0 - sqrt_M0_sq);
    fa = fmax(fa, 1e-6);

    // M4±/P5± 耗散系数——真实 bug 修复（V2.0 专家组盲审第四次评审，
    // 2026-08-28，#12），与 core/fr_kernels.py::compute_ausm_up_flux
    // 逐字对应，完整推导/文献交叉核实见该文件同名注释。
    double beta_mass = 1.0 / 8.0;
    // P5± 的 alpha 项不乘 1/4（Liou 2006 式 (24)；2026-09-25 修正，见
    // core/fr_operators/kernels.py::compute_ausm_up_flux 的 P5 注释）。
    double alpha_pressure = 3.0 / 16.0 * (-4.0 + 5.0 * fa * fa);

    // Weiss-Smith 预处理声速（与 kernels.py::compute_ausm_up_flux 的
    // _WEISS_SMITH_K=1.1 同一个安全裕度常数、同一套 beta2 公式）。
    double beta2 = fmin(1.0, fmax(fmax(Mbar2, 1.1 * mach_ref * mach_ref), 1e-10));
    double sqrt_beta2 = sqrt(beta2);

    // 预处理声速的作用域按 precond_mode 分派，与 kernels.py::
    // compute_ausm_up_flux 的 s_mass/s_pres 逐字对应。默认档
    // (precond_mode==0) 两个因子都是 1.0，本函数精确退化为标准
    // AUSM+up；为什么 legacy 档不能做默认见该文件模块级注释。
    double s_mass = sqrt_beta2;
    double s_pres = sqrt_beta2;
    if (precond_mode == 0) {         // PRECOND_PHYSICAL
        s_mass = 1.0;
        s_pres = 1.0;
    } else if (precond_mode == 1) {  // PRECOND_PRESSURE_PHYSICAL
        s_pres = 1.0;
    }

    double aL_m = s_mass * aL;
    double aR_m = s_mass * aR;
    double a_half_m = s_mass * a_half;
    double a_half_pr = s_pres * a_half;

    double M_L = unL / fmax(aL_m, 1e-10);
    double M_R = unR / fmax(aR_m, 1e-10);
    double M_L_pr = unL / fmax(s_pres * aL, 1e-10);
    double M_R_pr = unR / fmax(s_pres * aR, 1e-10);

    // 质量通量分裂 M+ / M-
    double Mp_L, Mm_R;
    if (fabs(M_L) >= 1.0) {
        Mp_L = 0.5 * (M_L + fabs(M_L));
    } else {
        Mp_L = 0.25 * (M_L + 1.0) * (M_L + 1.0) + beta_mass * (M_L * M_L - 1.0) * (M_L * M_L - 1.0);
    }
    if (fabs(M_R) >= 1.0) {
        Mm_R = 0.5 * (M_R - fabs(M_R));
    } else {
        Mm_R = -0.25 * (M_R - 1.0) * (M_R - 1.0) - beta_mass * (M_R * M_R - 1.0) * (M_R * M_R - 1.0);
    }
    double M_half = Mp_L + Mm_R;

    // Mp 压力扩散项
    double Kp = 0.25;
    double sigma_p = 1.0;
    double one_minus_sigma = 1.0 - sigma_p * Mbar2;
    if (one_minus_sigma < 0.0) one_minus_sigma = 0.0;
    double Mp = -(Kp / fa) * one_minus_sigma * (pR_s - pL_s) / (rho_half * a_half_m * a_half_m);
    double mass_flux = 0.5 * (rhoL_s * aL_m + rhoR_s * aR_m) * (M_half + Mp);

    // 压力分裂 P+ / P-
    double Pp_L, Pm_R;
    if (fabs(M_L_pr) >= 1.0) {
        double sign_ML = (M_L_pr > 0.0) ? 1.0 : ((M_L_pr < 0.0) ? -1.0 : 0.0);
        Pp_L = 0.5 * (1.0 + sign_ML);
    } else {
        Pp_L = 0.25 * (M_L_pr + 1.0) * (M_L_pr + 1.0) * (2.0 - M_L_pr)
               + alpha_pressure * M_L_pr * (M_L_pr * M_L_pr - 1.0)
                 * (M_L_pr * M_L_pr - 1.0);
    }
    if (fabs(M_R_pr) >= 1.0) {
        double sign_MR = (M_R_pr > 0.0) ? 1.0 : ((M_R_pr < 0.0) ? -1.0 : 0.0);
        Pm_R = 0.5 * (1.0 - sign_MR);
    } else {
        Pm_R = 0.25 * (M_R_pr - 1.0) * (M_R_pr - 1.0) * (2.0 + M_R_pr)
               - alpha_pressure * M_R_pr * (M_R_pr * M_R_pr - 1.0)
                 * (M_R_pr * M_R_pr - 1.0);
    }

    // pu 速度扩散项
    double Ku = 0.75;
    double p_half = Pp_L * pL_s + Pm_R * pR_s
        - Ku * Pp_L * Pm_R * (rhoL_s + rhoR_s) * fa * a_half_pr * (unR - unL);

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
}

extern "C" __global__
void p0_inviscid_residual(
    const int* owner_cell,
    const int* neighbor_cell,
    const bool* is_boundary,
    const double* normal,      // (n_faces, 3)
    const double* area_w,      // (n_faces,)
    const double* Q_all,       // (n_cells, 5) 原始变量
    const double* Q_ghost,     // (n_faces, 5) 边界幽灵态（混合面也读它，B-8）
    const double* cell_volumes,// (n_cells,)
    const double* mixed_bnd_frac, // (n_faces,) 混合面边界子面面积占比（B-8，非混合面为 0）
    double* residual,          // (n_cells, 5) 输出残差
    const int n_faces,
    const int precond_mode,    // AUSM+up 预处理声速作用域（见上）
    const double mach_ref      // AUSM+up Weiss-Smith 预处理参考马赫数，
                                // 见 kernels.py::compute_ausm_up_flux 文档
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

    double flux[5];
    ausm_up_flux(rhoL, uL, vL, wL, pL, rhoR, uR, vR, wR, pR,
                 nx, ny, nz, mach_ref, precond_mode, flux);

    // 混合拆分面（B-8，镜像 CPU inviscid_p0_kernel.py 同名分支）：整张四边形面的通量按子面面积占比混合，
    // 边界半区用同一 owner 单元的幽灵态另解一次黎曼问题。
    double bfrac = mixed_bnd_frac[f];
    if (bfrac > 0.0 && !is_bnd) {
        double rhoB = Q_ghost[f * 5 + 0];
        double uB   = Q_ghost[f * 5 + 1];
        double vB   = Q_ghost[f * 5 + 2];
        double wB   = Q_ghost[f * 5 + 3];
        double pB   = Q_ghost[f * 5 + 4];
        double flux_b[5];
        ausm_up_flux(rhoL, uL, vL, wL, pL, rhoB, uB, vB, wB, pB,
                     nx, ny, nz, mach_ref, precond_mode, flux_b);
        for (int v = 0; v < 5; v++) {
            flux[v] = (1.0 - bfrac) * flux[v] + bfrac * flux_b[v];
        }
    }

    // ── 原子累加到残差 ──
    double vol_o = cell_volumes[oc];
    for (int v = 0; v < 5; v++) {
        atomicAdd(&residual[oc * 5 + v], -flux[v] * aw / vol_o);
    }

    if (!is_bnd) {
        // 混合面（B-8）：只把内部半区份额计入 neighbor，边界半区份额属于边界条件。
        int nc = neighbor_cell[f];
        double vol_n = cell_volumes[nc];
        double nshare = 1.0 - bfrac;
        for (int v = 0; v < 5; v++) {
            atomicAdd(&residual[nc * 5 + v], flux[v] * aw / vol_n * nshare);
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


def _pm(precond_mode):
    """把 None 解析成实际的 precond_mode（与 CPU 端同一个解析器）。

    与 CPU 端同一个解析器，否则同一次运行的 CPU/GPU 可能取到不同档。
    """
    from autoflowcfd.core.fr_operators.kernels import resolve_ausm_precond_mode

    return int(resolve_ausm_precond_mode() if precond_mode is None else precond_mode)


def compute_inviscid_residual_p0_cupy_gpu_resident(
    Q_gpu,
    owner_cell_gpu, neighbor_cell_gpu, is_boundary_gpu,
    normal_gpu, area_w_gpu, cell_volumes_gpu,
    Q_ghost_gpu, mixed_bnd_frac_gpu,
    n_cells: int, n_faces: int,
    mach_ref: float = 0.1,
    precond_mode: Optional[int] = None,
):
    """P0 无粘残差的 GPU 常驻版本（数据已在 GPU 上，无需传输）。

    用于 GPUFRSolver 内部，所有数据都是 CuPy 数组，避免 CPU↔GPU 传输。

    Args:
        Q_gpu: (n_cells, 5) 原始变量，CuPy 数组
        owner_cell_gpu, neighbor_cell_gpu, is_boundary_gpu: 面连接关系
        normal_gpu, area_w_gpu: 面法向和面积权重
        cell_volumes_gpu: 单元体积
        Q_ghost_gpu: 边界幽灵态（含混合拆分面边界子面的幽灵态，B-8）
        mixed_bnd_frac_gpu: (n_faces,) 混合面边界子面面积占比（B-8，
            非混合面为 0）
        n_cells: 单元数
        n_faces: 面数
        mach_ref: AUSM+up Weiss-Smith 预处理参考马赫数，调用方
            （gpu_solver.py）必须显式传入 `solver.freestream["mach_ref"]`。
        precond_mode: AUSM+up 预处理声速作用域，None 表示按环境变量
            `AFCFD_AUSM_PRECOND_MODE` 解析（与 CPU 端同一个解析器），
            语义见 kernels.py::compute_ausm_up_flux 文档。

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
         cell_volumes_gpu, mixed_bnd_frac_gpu, d_residual,
         np.int32(n_faces), np.int32(_pm(precond_mode)), np.float64(mach_ref))
    )
    cp.cuda.Stream.null.synchronize()

    return d_residual[:, None, :]
