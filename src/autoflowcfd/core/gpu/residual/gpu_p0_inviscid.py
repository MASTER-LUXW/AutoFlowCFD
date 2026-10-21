"""
AutoFlowCFD V2.0 - P0 无粘残差的 CuPy CUDA 实现（单 GPU 与多 GPU 分布式的 P0 阶段，数据常驻 GPU）

经典分片常数有限体积格式，每个面独立取真实几何法向/面积，调用 AUSM+up 黎曼求解器，原子累加到
owner/neighbor 单元残差（与 CPU 版 `_compute_inviscid_residual_fv_p0` 同一算法）。

"""

import numpy as np

from autoflowcfd.core.gpu import get_cupy

from autoflowcfd.core.thermo import GAMMA

# ─── AUSM+up CUDA C 核心（嵌入 RawKernel）──────────────────────
# 逐面计算：每个 CUDA 线程处理一个面
# - 读取 owner/neighbor 原始变量
# - 计算 AUSM+up 公共通量（混合拆分面另用幽灵态解一次做面积加权，B-8）
# - 原子累加到残差
_P0_INVISCID_CUDA_CODE = f"#define AFCFD_GAMMA {GAMMA!r}\n" + r"""
// 注意：只有 __global__ 入口需要 extern "C"（RawKernel 按名查找）；
// __device__ 辅助函数加 extern "C" 有 nvcc 链接语义风险，不加。
__device__
void ausm_up_flux(
    double rhoL, double uL, double vL, double wL, double pL,
    double rhoR, double uR, double vR, double wR, double pR,
    double nx, double ny, double nz,
    double mach_ref,
    double* flux
) {
    // ── AUSM+up 数值通量 ──
    // 与 core/fr_operators/kernels.py::compute_ausm_up_flux 逐字对应
    double gamma = AFCFD_GAMMA;   // core/thermo.py::GAMMA

    // 两侧状态不钳制（与 CPU compute_ausm_up_flux 同一约定，I-18）
    double rhoL_s = rhoL;
    double rhoR_s = rhoR;
    double pL_s = pL;
    double pR_s = pR;

    double unL = uL * nx + vL * ny + wL * nz;
    double unR = uR * nx + vR * ny + wR * nz;

    double aL = sqrt(gamma * pL_s / rhoL_s);
    double aR = sqrt(gamma * pR_s / rhoR_s);

    double a_half = 0.5 * (aL + aR);
    double rho_half = 0.5 * (rhoL_s + rhoR_s);
    double Mbar2 = (unL * unL + unR * unR) / (2.0 * a_half * a_half);
    double M0_sq = fmin(1.0, fmax(Mbar2, mach_ref * mach_ref));
    double sqrt_M0_sq = sqrt(M0_sq);
    double fa = sqrt_M0_sq * (2.0 - sqrt_M0_sq);
    fa = fmax(fa, 1e-6);

    // M4±/P5± 耗散系数——真实 bug 修复（V2.0 专家组盲审第四次评审，
    // 2026-08-28，#12），与 core/fr_operators/kernels.py::compute_ausm_up_flux
    // 逐字对应，完整推导/文献交叉核实见该文件同名注释。
    double beta_mass = 1.0 / 8.0;
    // P5± 的 alpha 项不乘 1/4（Liou 2006 式 (24)；2026-09-25 修正，见
    // core/fr_operators/kernels.py::compute_ausm_up_flux 的 P5 注释）。
    double alpha_pressure = 3.0 / 16.0 * (-4.0 + 5.0 * fa * fa);

    // 两侧马赫数用界面公共声速归一、质量通量取迎风密度（标准 AUSM+up，与 CPU 同式；通量内部全部用物理声速）
    double M_L = unL / fmax(a_half, 1e-10);
    double M_R = unR / fmax(a_half, 1e-10);

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
    double Mp = -(Kp / fa) * one_minus_sigma * (pR_s - pL_s) / (rho_half * a_half * a_half);
    double M_tot = M_half + Mp;
    double mass_flux = a_half * (M_tot >= 0.0 ? M_tot * rhoL_s : M_tot * rhoR_s);

    // 压力分裂 P+ / P-
    double Pp_L, Pm_R;
    if (fabs(M_L) >= 1.0) {
        double sign_ML = (M_L > 0.0) ? 1.0 : ((M_L < 0.0) ? -1.0 : 0.0);
        Pp_L = 0.5 * (1.0 + sign_ML);
    } else {
        Pp_L = 0.25 * (M_L + 1.0) * (M_L + 1.0) * (2.0 - M_L)
               + alpha_pressure * M_L * (M_L * M_L - 1.0)
                 * (M_L * M_L - 1.0);
    }
    if (fabs(M_R) >= 1.0) {
        double sign_MR = (M_R > 0.0) ? 1.0 : ((M_R < 0.0) ? -1.0 : 0.0);
        Pm_R = 0.5 * (1.0 - sign_MR);
    } else {
        Pm_R = 0.25 * (M_R - 1.0) * (M_R - 1.0) * (2.0 + M_R)
               - alpha_pressure * M_R * (M_R * M_R - 1.0)
                 * (M_R * M_R - 1.0);
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
    double* residual,          // (n_cells, 5) 输出残差
    const int n_faces,
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
                 nx, ny, nz, mach_ref, flux);

    // 混合拆分面的两半在面几何里已经是两个独立的面（inviscid_p0.py::p0_face_geometry）

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


def compute_inviscid_residual_p0_cupy_gpu_resident(
    Q_gpu,
    owner_cell_gpu, neighbor_cell_gpu, is_boundary_gpu,
    normal_gpu, area_w_gpu, cell_volumes_gpu,
    Q_ghost_gpu,
    n_cells: int, n_faces: int,
    mach_ref: float = 0.1,
):
    """P0 无粘残差的 GPU 常驻版本（数据已在 GPU 上，无需传输）。

    用于 GPUFRSolver 内部，所有数据都是 CuPy 数组，避免 CPU↔GPU 传输。

    Args:
        Q_gpu: (n_cells, 5) 原始变量，CuPy 数组
        owner_cell_gpu, neighbor_cell_gpu, is_boundary_gpu: 面连接关系
        normal_gpu, area_w_gpu: 面法向和面积权重
        cell_volumes_gpu: 单元体积
        Q_ghost_gpu: 边界幽灵态（按面索引，含混合拆分面的边界半区）
        n_cells: 单元数
        n_faces: 面数
        mach_ref: AUSM+up Weiss-Smith 预处理参考马赫数，调用方
            （gpu_solver.py）必须显式传入 `solver.freestream["mach_ref"]`。

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
         cell_volumes_gpu, d_residual,
         np.int32(n_faces), np.float64(mach_ref))
    )
    cp.cuda.Stream.null.synchronize()

    return d_residual[:, None, :]


def p0_face_arrays_gpu(flat, unit_normals: np.ndarray, area_weights: np.ndarray, cell_volumes) -> dict:
    """P0 有限体积核的设备端面几何（单 GPU 的全局面表与多 GPU 的本 rank 面表共用）。

    Args:
        flat: 面几何（`owner_cell` / `neighbor_cell` / `is_boundary`，单元编号与残差的状态数组同一个排列）
        unit_normals, area_weights: 同一面表的 P0 法向 / 面积（`fr_residual/inviscid_p0.py::p0_face_arrays`）
        cell_volumes: 单元体积（与状态数组同一排列）

    内核的单元编号参数是 `int*`：这里统一转成 int32（分区面表的压缩单元编号不一定是 int32）。
    """
    cp = get_cupy()
    is_bnd = np.asarray(flat.is_boundary, dtype=bool)
    normal_host = np.ascontiguousarray(unit_normals, dtype=np.float64)
    return {
        "owner": cp.asarray(np.asarray(flat.owner_cell, dtype=np.int32)),
        "neighbor": cp.asarray(np.where(is_bnd, 0, flat.neighbor_cell).astype(np.int32)),
        "is_bnd": cp.asarray(is_bnd),
        "normal_host": normal_host,
        "normal": cp.asarray(normal_host),
        "area_w": cp.asarray(np.ascontiguousarray(area_weights, dtype=np.float64)),
        "volumes": cp.asarray(cell_volumes, dtype=np.float64),
    }


def compute_inviscid_residual_p0_gpu(U, flat, geo: dict, boundary_ghost_provider, mach_ref: float):
    """P0 有限体积无粘残差（设备端），`(n_cells, 1, 5)`。

    原始变量取自传入的状态 `U`（隐式 Jacobian-向量积与 RK 各级传试探态）；边界幽灵态按面在主机上组装
    （`_precompute_ghost_states`，与 CPU 同一份），`geo` 为 `p0_face_arrays_gpu` 的结果（与 `flat` 同一面表）。
    """
    from autoflowcfd.boundary.fr_ghost_state import require_ghost_provider
    from autoflowcfd.core.fr_residual.inviscid_p0 import _precompute_ghost_states
    from autoflowcfd.core.gpu.residual.gpu_flux import conserved_to_primitive_gpu

    cp = get_cupy()
    n_cells = int(U.shape[0])
    Q_flat = conserved_to_primitive_gpu(U[:, 0, :5])
    provider = require_ghost_provider(boundary_ghost_provider, "compute_inviscid_residual_p0_gpu")
    Q_ghost = cp.asarray(_precompute_ghost_states(flat, flat, provider, cp.asnumpy(Q_flat), flat.n_faces,
                                                  geo["normal_host"]))
    return compute_inviscid_residual_p0_cupy_gpu_resident(
        Q_flat, geo["owner"], geo["neighbor"], geo["is_bnd"], geo["normal"], geo["area_w"], geo["volumes"],
        Q_ghost, n_cells, flat.n_faces, mach_ref=mach_ref)
