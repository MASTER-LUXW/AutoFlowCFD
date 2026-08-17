"""AutoFlowCFD V2.0 - P0（分片常数有限体积）无粘残差的真实 CUDA 实现 (B-01 阶段1)

背景：V2.0 专家评审发现 `core/backend/` 下此前的"GPU 支持"全部是孤立、
未被 `FRSolver` 真正调用的占位/简化代码（`gpu_backend.py::_cuda_flux_kernel`
甚至连 `@cuda.jit` 装饰器都没有、用中心平均冒充 AUSM+up；
`cuda_fr_kernels.py` 是从未被引用的孤儿模块，且自己承认"简化残差"）。
本模块是第一阶段的真实替代：把已验证正确的 P0 有限体积无粘残差
（`core/fr_residual_inviscid.py::_compute_inviscid_residual_fv_p0`，
经典分片常数迎风格式：每个面独立取真实几何法向/面积、调用真实 AUSM+up
黎曼求解器、原子累加到 owner/neighbor 单元）逐字忠实移植到
`numba.cuda` kernel，不简化物理、不用占位公式。

P>=1（坍缩坐标度量张量外插 + 逐面记录字典键控分发）复杂度高得多
（见 fr_residual_inviscid.py::compute_inviscid_residual_fr 文档），
需要重新设计数据布局才能上 CUDA，是后续独立阶段（B-01 阶段2），本模块
不处理。

无本地 GPU 硬件：正确性通过 `NUMBA_ENABLE_CUDASIM=1`（numba 自带的纯
Python CUDA 语义模拟器，不需要真实显卡，逐线程真实执行核函数逻辑）
对照已验证的 CPU 版本数值核验，见 tests/unit/test_fr_gpu_p0.py。
"""
import math
from typing import Callable, Optional

import numba
import numpy as np
from loguru import logger

try:
    from numba import cuda
    _CUDA_IMPORT_OK = True
except Exception:  # pragma: no cover - numba 本身缺失的极端环境
    _CUDA_IMPORT_OK = False

from autoflowcfd.core.fr_residual.inviscid import conserved_to_primitive, DefaultGhostProvider

GAMMA = 1.4


def gpu_p0_available() -> bool:
    """真实 CUDA 设备（或 NUMBA_ENABLE_CUDASIM=1 模拟器）是否可用。"""
    if not _CUDA_IMPORT_OK:
        return False
    try:
        return bool(cuda.is_available())
    except Exception:
        return False


if _CUDA_IMPORT_OK:

    @cuda.jit(device=True, inline=True)
    def _ausm_up_flux_device(rhoL, uL, vL, wL, pL, rhoR, uR, vR, wR, pR, nx, ny, nz, flux):
        """AUSM+up 数值通量，逐字对照 core/fr_kernels.py::compute_ausm_up_flux
        移植（同一套物理/参数，只是把嵌套函数 M_plus/M_minus/P_plus/P_minus
        展开成内联分支——numba CUDA target 对函数内定义闭包函数的支持不如
        CPU target 稳定，展开是为了可移植性，不改变任何数值结果）。
        把结果写入长度为 5 的 `flux` 数组（CUDA device 函数里避免返回新分配
        数组，与 CPU 版返回值一致，只是调用约定不同）。
        """
        gamma = 1.4
        alpha = 0.1875
        beta = 0.5

        rhoL_s = max(rhoL, 1e-6)
        rhoR_s = max(rhoR, 1e-6)
        pL_s = max(pL, 10.0)
        pR_s = max(pR, 10.0)

        unL = uL * nx + vL * ny + wL * nz
        unR = uR * nx + vR * ny + wR * nz

        aL = math.sqrt(max(gamma * pL_s / rhoL_s, 1e-10))
        aR = math.sqrt(max(gamma * pR_s / rhoR_s, 1e-10))

        M_L = unL / max(aL, 1e-10)
        M_R = unR / max(aR, 1e-10)

        # 界面声速/低马赫标度函数 Mbar2, fa（与 core/fr_kernels.py 逐字一致）。
        a_half = 0.5 * (aL + aR)
        rho_half = 0.5 * (rhoL_s + rhoR_s)
        Mbar2 = (unL * unL + unR * unR) / (2.0 * a_half * a_half)
        Ma_ref = 0.1
        M0_sq = min(1.0, max(Mbar2, Ma_ref * Ma_ref))
        sqrt_M0_sq = math.sqrt(M0_sq)
        fa = sqrt_M0_sq * (2.0 - sqrt_M0_sq)
        fa = max(fa, 1e-6)

        if abs(M_L) >= 1.0:
            Mp_L = 0.5 * (M_L + abs(M_L))
        else:
            Mp_L = 0.25 * (M_L + 1.0) ** 2 + alpha * (M_L**2 - 1.0) ** 2

        if abs(M_R) >= 1.0:
            Mm_R = 0.5 * (M_R - abs(M_R))
        else:
            Mm_R = -0.25 * (M_R - 1.0) ** 2 - alpha * (M_R**2 - 1.0) ** 2

        M_half = Mp_L + Mm_R

        # Mp 压力扩散项 (Liou 2006 AUSM+up 式17)，取代旧的、破坏反对称性的
        # "熵修正"（|M_L-M_R| 在 (L,R,n)->(R,L,-n) 变换下不翻号，直接违反
        # F(A,B,n)=-F(B,A,-n)；详见 core/fr_kernels.py::compute_ausm_up_flux
        # 同一处的完整推导与数值验证）。
        Kp = 0.25
        sigma_p = 1.0
        one_minus_sigma_mbar2 = 1.0 - sigma_p * Mbar2
        if one_minus_sigma_mbar2 < 0.0:
            one_minus_sigma_mbar2 = 0.0
        Mp = -(Kp / fa) * one_minus_sigma_mbar2 * (pR_s - pL_s) / (rho_half * a_half * a_half)
        mass_flux = 0.5 * (rhoL_s * aL + rhoR_s * aR) * (M_half + Mp)

        if abs(M_L) >= 1.0:
            sign_ML = 1.0 if M_L > 0.0 else (-1.0 if M_L < 0.0 else 0.0)
            Pp_L = 0.5 * (1.0 + sign_ML)
        else:
            Pp_L = 0.25 * ((M_L + 1.0) ** 2 * (2.0 - M_L) + beta * M_L * (M_L**2 - 1.0) ** 2)

        if abs(M_R) >= 1.0:
            sign_MR = 1.0 if M_R > 0.0 else (-1.0 if M_R < 0.0 else 0.0)
            Pm_R = 0.5 * (1.0 - sign_MR)
        else:
            Pm_R = 0.25 * ((M_R - 1.0) ** 2 * (2.0 + M_R) - beta * M_R * (M_R**2 - 1.0) ** 2)

        # pu 速度扩散项 (Liou 2006 AUSM+up 式18)，与 Mp 项配套。
        Ku = 0.75
        p_half = Pp_L * pL_s + Pm_R * pR_s \
            - Ku * Pp_L * Pm_R * (rhoL_s + rhoR_s) * fa * a_half * (unR - unL)

        upwind_L = mass_flux >= 0.0
        flux[0] = mass_flux
        flux[1] = mass_flux * (uL if upwind_L else uR) + p_half * nx
        flux[2] = mass_flux * (vL if upwind_L else vR) + p_half * ny
        flux[3] = mass_flux * (wL if upwind_L else wR) + p_half * nz

        hL = gamma / (gamma - 1.0) * pL_s / rhoL_s + 0.5 * (uL * uL + vL * vL + wL * wL)
        hR = gamma / (gamma - 1.0) * pR_s / rhoR_s + 0.5 * (uR * uR + vR * vR + wR * wR)
        flux[4] = mass_flux * (hL if upwind_L else hR)

    _CUDA_FLUX_DTYPE = numba.float64

    @cuda.jit
    def _p0_inviscid_residual_kernel(
        owner_cell, neighbor_cell, is_boundary, normal, area_w,
        Q_all, Q_ghost, cell_volumes, residual_out,
    ):
        """一个 CUDA 线程处理一条面记录：计算该面的 AUSM+up 公共通量，
        原子累加到 owner（总是）与 neighbor（仅内部面）两侧的残差——
        与 CPU 版 `_compute_inviscid_residual_fv_p0` 逐面循环体逐字对应，
        用 `cuda.atomic.add` 代替 CPU 版的 `residual5[cell] +=`，因为不同
        线程（面）可能同时写同一个单元（该单元的所有邻接面）。
        """
        f = cuda.grid(1)
        if f >= owner_cell.shape[0]:
            return

        oc = owner_cell[f]
        nx = normal[f, 0]
        ny = normal[f, 1]
        nz = normal[f, 2]
        aw = area_w[f]

        rhoL = Q_all[oc, 0]
        uL = Q_all[oc, 1]
        vL = Q_all[oc, 2]
        wL = Q_all[oc, 3]
        pL = Q_all[oc, 4]

        if is_boundary[f]:
            rhoR = Q_ghost[f, 0]
            uR = Q_ghost[f, 1]
            vR = Q_ghost[f, 2]
            wR = Q_ghost[f, 3]
            pR = Q_ghost[f, 4]
        else:
            nc = neighbor_cell[f]
            rhoR = Q_all[nc, 0]
            uR = Q_all[nc, 1]
            vR = Q_all[nc, 2]
            wR = Q_all[nc, 3]
            pR = Q_all[nc, 4]

        flux = cuda.local.array(5, dtype=_CUDA_FLUX_DTYPE)
        _ausm_up_flux_device(rhoL, uL, vL, wL, pL, rhoR, uR, vR, wR, pR, nx, ny, nz, flux)

        vol_o = cell_volumes[oc]
        for v in range(5):
            cuda.atomic.add(residual_out, (oc, v), -flux[v] * aw / vol_o)

        if not is_boundary[f]:
            nc = neighbor_cell[f]
            vol_n = cell_volumes[nc]
            for v in range(5):
                cuda.atomic.add(residual_out, (nc, v), flux[v] * aw / vol_n)


def compute_inviscid_residual_p0_gpu(
    U: np.ndarray,
    mesh,
    boundary_ghost_provider: Optional[Callable[[int, np.ndarray, np.ndarray], np.ndarray]] = None,
) -> np.ndarray:
    """P0 无粘残差的 GPU（CUDA）实现，函数签名/返回值契约与
    `core/fr_residual_inviscid.py::_compute_inviscid_residual_fv_p0` 完全一致
    （同一个 P0 有限体积算法，只是把逐面循环搬到 CUDA 线程网格上并行执行），
    可以互相替换、用同一组测试数值对比验证。

    边界幽灵态：`boundary_ghost_provider` 是任意 Python 可调用对象（真实
    WALL/INLET/OUTLET/SYMMETRY/FARFIELD 分发逻辑，见
    boundary/fr_ghost_state.py），无法在 CUDA kernel 内部调用——按面在
    CPU 上预先算好全部边界面的幽灵态（O(n_boundary_faces)，比 O(n_faces)
    的通量计算 + 原子累加便宜得多，不是这个函数的性能瓶颈），再把结果
    数组传给 kernel，kernel 内部只做纯数值查表，不改变边界条件本身的
    物理含义。

    Raises:
        RuntimeError: mesh.n_points_1d != 1（不是 P0 网格）或 CUDA 不可用
            （真实设备或 NUMBA_ENABLE_CUDASIM 都没有）——不做静默 CPU 回退，
            调用方（FRSolver）负责在外层决定回退策略并如实记录日志。
    """
    if not _CUDA_IMPORT_OK or not gpu_p0_available():
        raise RuntimeError("CUDA is not available (no real device and NUMBA_ENABLE_CUDASIM not set)")
    if mesh.n_points_1d != 1:
        raise RuntimeError(
            f"compute_inviscid_residual_p0_gpu only supports P0 meshes (n_points_1d=1), "
            f"got n_points_1d={mesh.n_points_1d}"
        )
    if mesh.cell_volumes is None:
        raise RuntimeError("mesh.cell_volumes not available - required for the P0 finite-volume residual path")

    n_cells = mesh.n_cells
    Q_all = conserved_to_primitive(U[..., :5])[:, 0, :].astype(np.float64)  # (n_cells,5)

    fc = mesh.face_connectivity
    ffp_list = mesh.face_flux_points
    n_faces = fc.n_faces
    ghost_provider = boundary_ghost_provider if boundary_ghost_provider is not None else DefaultGhostProvider()

    owner_cell = fc.owner_cell.astype(np.int32)
    neighbor_cell = np.where(fc.is_boundary, 0, fc.neighbor_cell).astype(np.int32)  # 边界面此列不会被读取
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

    d_owner = cuda.to_device(owner_cell)
    d_neighbor = cuda.to_device(neighbor_cell)
    d_is_boundary = cuda.to_device(is_boundary)
    d_normal = cuda.to_device(normal)
    d_area_w = cuda.to_device(area_w)
    d_Q = cuda.to_device(Q_all)
    d_Q_ghost = cuda.to_device(Q_ghost)
    d_volumes = cuda.to_device(cell_volumes)
    d_residual = cuda.to_device(np.zeros((n_cells, 5), dtype=np.float64))

    threads_per_block = 128
    blocks_per_grid = (n_faces + threads_per_block - 1) // threads_per_block
    _p0_inviscid_residual_kernel[blocks_per_grid, threads_per_block](
        d_owner, d_neighbor, d_is_boundary, d_normal, d_area_w,
        d_Q, d_Q_ghost, d_volumes, d_residual,
    )
    cuda.synchronize()

    residual5 = d_residual.copy_to_host()
    return residual5[:, None, :]
