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

已被取代、不在生产调用路径上（2026-08-23 核实）：`FRSolver`/
`GPUFRSolver` 的 GPU P0 无粘残差分发（solver.py/gpu_solver.py）现在
统一走 `core/gpu/gpu_p0_inviscid.py`（CuPy `RawKernel` 版本）；全仓库
搜索确认 `core.backend.fr_gpu_p0` 只被本模块自己的 docstring 和
`tests/unit/test_fr_gpu_p0.py` 引用，从未被 `solver.py`/`gpu_solver.py`
import。棱柱四边形侧面重复计数缺陷已同步修复（2026-08-23）：面法向/
面积改为直接复用 `fr_residual/inviscid_p0.py::_extract_p0_face_geometry`
（CPU P0 路径已验证过的去重/multi-source 回退逻辑），不再逐面独立
读取未去重的 `true_normal`/`true_area_weight`。仍然是死代码这一点
不变——本机没有真实 GPU/CUDA，只能靠 `NUMBA_ENABLE_CUDASIM=1` 模拟器
跑 `tests/unit/test_fr_gpu_p0.py` 核验，无法在真实硬件上验证。
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
    def _ausm_up_flux_device(rhoL, uL, vL, wL, pL, rhoR, uR, vR, wR, pR,
                             nx, ny, nz, mach_ref, precond_mode, flux):
        """AUSM+up 数值通量（含 Weiss-Smith 低马赫数预处理），逐字对照
        core/fr_kernels.py::compute_ausm_up_flux 移植（同一套物理/参数，
        只是把嵌套函数 M_plus/M_minus/P_plus/P_minus 展开成内联分支——
        numba CUDA target 对函数内定义闭包函数的支持不如 CPU target 稳定，
        展开是为了可移植性，不改变任何数值结果）。
        把结果写入长度为 5 的 `flux` 数组（CUDA device 函数里避免返回新分配
        数组，与 CPU 版返回值一致，只是调用约定不同）。
        """
        gamma = 1.4

        rhoL_s = max(rhoL, 1e-6)
        rhoR_s = max(rhoR, 1e-6)
        pL_s = max(pL, 10.0)
        pR_s = max(pR, 10.0)

        unL = uL * nx + vL * ny + wL * nz
        unR = uR * nx + vR * ny + wR * nz

        aL = math.sqrt(max(gamma * pL_s / rhoL_s, 1e-10))
        aR = math.sqrt(max(gamma * pR_s / rhoR_s, 1e-10))

        # 界面声速/低马赫标度函数 Mbar2, fa（与 core/fr_kernels.py 逐字一致）。
        a_half = 0.5 * (aL + aR)
        rho_half = 0.5 * (rhoL_s + rhoR_s)
        Mbar2 = (unL * unL + unR * unR) / (2.0 * a_half * a_half)
        M0_sq = min(1.0, max(Mbar2, mach_ref * mach_ref))
        sqrt_M0_sq = math.sqrt(M0_sq)
        fa = sqrt_M0_sq * (2.0 - sqrt_M0_sq)
        fa = max(fa, 1e-6)

        # M4±/P5± 耗散系数——真实 bug 修复（V2.0 专家组盲审第四次评审，
        # 2026-08-28，#12），与 core/fr_kernels.py::compute_ausm_up_flux
        # 逐字对应，完整推导/文献交叉核实见该文件同名注释。
        beta_mass = 1.0 / 8.0
        alpha_pressure = 3.0 / 16.0 * (-4.0 + 5.0 * fa * fa)

        # Weiss-Smith 预处理声速（与 kernels.py::compute_ausm_up_flux 的
        # _WEISS_SMITH_K=1.1 同一个安全裕度常数、同一套 beta2 公式）。
        beta2 = min(1.0, max(max(Mbar2, 1.1 * mach_ref * mach_ref), 1e-10))
        sqrt_beta2 = math.sqrt(beta2)

        # 预处理声速的作用域按 precond_mode 分派（0=physical 默认 /
        # 1=pressure_physical / 2=legacy），与 kernels.py::
        # compute_ausm_up_flux 的 s_mass/s_pres 逐字对应。
        s_mass = sqrt_beta2
        s_pres = sqrt_beta2
        if precond_mode == 0:
            s_mass = 1.0
            s_pres = 1.0
        elif precond_mode == 1:
            s_pres = 1.0

        aL_m = s_mass * aL
        aR_m = s_mass * aR
        a_half_m = s_mass * a_half
        a_half_pr = s_pres * a_half

        M_L = unL / max(aL_m, 1e-10)
        M_R = unR / max(aR_m, 1e-10)
        M_L_pr = unL / max(s_pres * aL, 1e-10)
        M_R_pr = unR / max(s_pres * aR, 1e-10)

        if abs(M_L) >= 1.0:
            Mp_L = 0.5 * (M_L + abs(M_L))
        else:
            Mp_L = 0.25 * (M_L + 1.0) ** 2 + beta_mass * (M_L**2 - 1.0) ** 2

        if abs(M_R) >= 1.0:
            Mm_R = 0.5 * (M_R - abs(M_R))
        else:
            Mm_R = -0.25 * (M_R - 1.0) ** 2 - beta_mass * (M_R**2 - 1.0) ** 2

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
        Mp = -(Kp / fa) * one_minus_sigma_mbar2 * (pR_s - pL_s) / (rho_half * a_half_m * a_half_m)
        mass_flux = 0.5 * (rhoL_s * aL_m + rhoR_s * aR_m) * (M_half + Mp)

        if abs(M_L_pr) >= 1.0:
            sign_ML = 1.0 if M_L_pr > 0.0 else (-1.0 if M_L_pr < 0.0 else 0.0)
            Pp_L = 0.5 * (1.0 + sign_ML)
        else:
            Pp_L = 0.25 * ((M_L_pr + 1.0) ** 2 * (2.0 - M_L_pr)
                           + alpha_pressure * M_L_pr * (M_L_pr**2 - 1.0) ** 2)

        if abs(M_R_pr) >= 1.0:
            sign_MR = 1.0 if M_R_pr > 0.0 else (-1.0 if M_R_pr < 0.0 else 0.0)
            Pm_R = 0.5 * (1.0 - sign_MR)
        else:
            Pm_R = 0.25 * ((M_R_pr - 1.0) ** 2 * (2.0 + M_R_pr)
                           - alpha_pressure * M_R_pr * (M_R_pr**2 - 1.0) ** 2)

        # pu 速度扩散项 (Liou 2006 AUSM+up 式18)，与 Mp 项配套。
        Ku = 0.75
        p_half = Pp_L * pL_s + Pm_R * pR_s \
            - Ku * Pp_L * Pm_R * (rhoL_s + rhoR_s) * fa * a_half_pr * (unR - unL)

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
        Q_all, Q_ghost, cell_volumes, mixed_bnd_frac, residual_out, mach_ref,
        precond_mode,
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
        _ausm_up_flux_device(rhoL, uL, vL, wL, pL, rhoR, uR, vR, wR, pR,
                             nx, ny, nz, mach_ref, precond_mode, flux)

        # 混合拆分面（B-8，镜像 CPU inviscid_p0_kernel.py / gpu_p0_inviscid.py
        # 同名分支）：整张四边形面的通量按子面面积占比混合，边界半区用同一
        # owner 单元的幽灵态另解一次黎曼问题。
        bfrac = mixed_bnd_frac[f]
        if bfrac > 0.0 and not is_boundary[f]:
            flux_b = cuda.local.array(5, dtype=_CUDA_FLUX_DTYPE)
            _ausm_up_flux_device(
                rhoL, uL, vL, wL, pL,
                Q_ghost[f, 0], Q_ghost[f, 1], Q_ghost[f, 2], Q_ghost[f, 3], Q_ghost[f, 4],
                nx, ny, nz, mach_ref, precond_mode, flux_b,
            )
            for v in range(5):
                flux[v] = (1.0 - bfrac) * flux[v] + bfrac * flux_b[v]

        vol_o = cell_volumes[oc]
        for v in range(5):
            cuda.atomic.add(residual_out, (oc, v), -flux[v] * aw / vol_o)

        if not is_boundary[f]:
            nc = neighbor_cell[f]
            vol_n = cell_volumes[nc]
            # 混合面（B-8）：只把内部半区份额计入 neighbor，边界半区份额属于边界条件。
            nshare = 1.0 - bfrac
            for v in range(5):
                cuda.atomic.add(residual_out, (nc, v), flux[v] * aw / vol_n * nshare)


def compute_inviscid_residual_p0_gpu(
    U: np.ndarray,
    mesh,
    boundary_ghost_provider: Optional[Callable[[int, np.ndarray, np.ndarray], np.ndarray]] = None,
    mach_ref: float = 0.1,
    precond_mode: Optional[int] = None,
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
    if precond_mode is None:
        from autoflowcfd.core.fr_operators.kernels import resolve_ausm_precond_mode

        precond_mode = resolve_ausm_precond_mode()

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
    Q_ghost = np.zeros((n_faces, 5), dtype=np.float64)

    # 面法向/面积：复用 CPU P0 路径已经修好的去重/multi-source 回退逻辑
    # （2026-08-23，见 inviscid_p0.py::_extract_p0_face_geometry 文档）
    # ——本文件此前逐面 `ffp_list[f]` 直接读 true_normal/true_area_weight，
    # 不做 owner_is_primary 过滤，对棱柱四边形侧面的重复三角化子面记录
    # 会重复 scatter-add 两次，与 CPU P0 kernel 修复前的同一个 bug（本
    # 模块是死代码、不在生产调用路径上，但既然要保持数值正确性就应该
    # 复用同一份已验证逻辑，而不是留着一份已知有 bug 的独立实现）。
    from autoflowcfd.core.fr_residual.inviscid_p0 import _extract_p0_face_geometry
    normal, area_w = _extract_p0_face_geometry(ffp_list, fc, n_faces)

    # 幽灵态预计算范围（B-8）：真边界面之外，混合拆分面的边界子面记录也读
    # Q_ghost[f]（kernel 混合分支），与 gpu_p0_inviscid.py 的 CPU 侧入口一致。
    mixed_bnd_face = getattr(ffp_list, "mixed_bnd_face", None)
    if mixed_bnd_face is None:
        mixed_bnd_face = np.zeros(n_faces, dtype=np.bool_)
    mixed_bnd_frac = getattr(ffp_list, "mixed_p0_bnd_frac", None)
    if mixed_bnd_frac is None:
        mixed_bnd_frac = np.zeros(n_faces, dtype=np.float64)

    for f in np.nonzero(is_boundary | mixed_bnd_face)[0]:
        Q_owner_fp = Q_all[owner_cell[f]: owner_cell[f] + 1]
        Q_ghost[f, :] = ghost_provider(f, Q_owner_fp, normal[f:f + 1])[0]

    cell_volumes = mesh.cell_volumes.astype(np.float64)

    d_owner = cuda.to_device(owner_cell)
    d_neighbor = cuda.to_device(neighbor_cell)
    d_is_boundary = cuda.to_device(is_boundary)
    d_normal = cuda.to_device(normal)
    d_area_w = cuda.to_device(area_w)
    d_Q = cuda.to_device(Q_all)
    d_Q_ghost = cuda.to_device(Q_ghost)
    d_volumes = cuda.to_device(cell_volumes)
    d_mixed_frac = cuda.to_device(mixed_bnd_frac)
    d_residual = cuda.to_device(np.zeros((n_cells, 5), dtype=np.float64))

    threads_per_block = 128
    blocks_per_grid = (n_faces + threads_per_block - 1) // threads_per_block
    _p0_inviscid_residual_kernel[blocks_per_grid, threads_per_block](
        d_owner, d_neighbor, d_is_boundary, d_normal, d_area_w,
        d_Q, d_Q_ghost, d_volumes, d_mixed_frac, d_residual, np.float64(mach_ref),
        np.int32(precond_mode),
    )
    cuda.synchronize()

    residual5 = d_residual.copy_to_host()
    return residual5[:, None, :]
