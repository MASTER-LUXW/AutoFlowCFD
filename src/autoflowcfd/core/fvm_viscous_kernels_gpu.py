"""CUDA GPU kernel：粘性通量（应力张量 + 热传导 + 湍流扩散），未经真实 GPU 硬件验证。

⚠️ 重要说明：本文件是 fvm_viscous_kernels.py 里已经用真实数值结果验证过
（与 numpy 参考实现误差 4.4e-16，机器精度级）的 Numba CPU kernel 的逐行
CUDA 翻译。开发环境没有可用 GPU，这里的 kernel **从未在真实 GPU 硬件上
编译、运行、验证过**，只做了逐行审查确保和 CPU 版本公式一致。

CPU 版本 fvm_viscous_kernels.py 里记录过一次真实的 Numba `prange` 并行
race condition（每面临时缓冲区 `gv_face`/`tau_n` 被错误地分配在循环外，
被多线程共享/踩踏），这里的 CUDA 版本每个线程处理一个面、缓冲区都是线程
局部的标量/寄存器变量（不是共享内存），所以结构上不存在同类风险；但因为
完全没有在真实硬件上跑过，仍然只能定性保证、不能定量验证。

在任何生产环境依赖这条路径之前，必须先在有 GPU 的机器上用和
fvm_viscous_kernels.py 同样的方法做数值对比验证。
"""

import numpy as np

try:
    from numba import cuda
    CUDA_AVAILABLE = cuda.is_available()
except Exception:
    CUDA_AVAILABLE = False


if CUDA_AVAILABLE:
    @cuda.jit(cache=True)
    def _viscous_internal_flux_kernel_gpu(
        int_owner, int_neigh, int_areas, int_normals,
        e_ON, dist,
        vel,            # (n_cells, 3)
        grad_vel,       # (n_cells, 3, 3)
        mu_eff,         # (n_cells,)
        T,              # (n_cells,)
        grad_T,         # (n_cells, 3)
        mu_t,           # (n_cells,)
        k_field, omega_field,
        grad_k, grad_omega,   # (n_cells, 3)
        mu_lam, cp, prandtl_lam, prandtl_turb, sigma_k1, sigma_w1,
        fvisc,          # (n_faces, 7) output
    ):
        """一个线程处理一个内部面，逐项对应
        fvm_viscous_kernels._viscous_internal_flux_kernel 的循环体。每个
        线程只写自己那一行 fvisc[f, :]，互不干扰，不需要原子操作。"""
        f = cuda.grid(1)
        if f >= int_owner.shape[0]:
            return

        o = int_owner[f]
        nb = int_neigh[f]
        nx = int_normals[f, 0]; ny = int_normals[f, 1]; nz = int_normals[f, 2]
        ex = e_ON[f, 0]; ey = e_ON[f, 1]; ez = e_ON[f, 2]
        d = dist[f]

        # 逐分量展开的 (3,3) 面平均速度梯度 + over-relaxed 修正
        # （CUDA device 函数里用局部标量代替 CPU 版本里的 gv_face(3,3)
        # scratch 数组，效果等价，避免每线程分配小数组的开销）。
        g00 = 0.5 * (grad_vel[o, 0, 0] + grad_vel[nb, 0, 0])
        g01 = 0.5 * (grad_vel[o, 0, 1] + grad_vel[nb, 0, 1])
        g02 = 0.5 * (grad_vel[o, 0, 2] + grad_vel[nb, 0, 2])
        g10 = 0.5 * (grad_vel[o, 1, 0] + grad_vel[nb, 1, 0])
        g11 = 0.5 * (grad_vel[o, 1, 1] + grad_vel[nb, 1, 1])
        g12 = 0.5 * (grad_vel[o, 1, 2] + grad_vel[nb, 1, 2])
        g20 = 0.5 * (grad_vel[o, 2, 0] + grad_vel[nb, 2, 0])
        g21 = 0.5 * (grad_vel[o, 2, 1] + grad_vel[nb, 2, 1])
        g22 = 0.5 * (grad_vel[o, 2, 2] + grad_vel[nb, 2, 2])

        proj0 = g00 * ex + g01 * ey + g02 * ez
        proj1 = g10 * ex + g11 * ey + g12 * ez
        proj2 = g20 * ex + g21 * ey + g22 * ez

        dvel0 = vel[nb, 0] - vel[o, 0]
        dvel1 = vel[nb, 1] - vel[o, 1]
        dvel2 = vel[nb, 2] - vel[o, 2]
        corr0 = dvel0 / d - proj0
        corr1 = dvel1 / d - proj1
        corr2 = dvel2 / d - proj2
        g00 += corr0 * ex; g01 += corr0 * ey; g02 += corr0 * ez
        g10 += corr1 * ex; g11 += corr1 * ey; g12 += corr1 * ez
        g20 += corr2 * ex; g21 += corr2 * ey; g22 += corr2 * ez

        mu_f = 0.5 * (mu_eff[o] + mu_eff[nb])
        divu = g00 + g11 + g22
        two_thirds_mu_divu = (2.0 / 3.0) * mu_f * divu
        t00 = mu_f * (g00 + g00) - two_thirds_mu_divu
        t01 = mu_f * (g01 + g10)
        t02 = mu_f * (g02 + g20)
        t11 = mu_f * (g11 + g11) - two_thirds_mu_divu
        t12 = mu_f * (g12 + g21)
        t22 = mu_f * (g22 + g22) - two_thirds_mu_divu
        tau_n0 = t00 * nx + t01 * ny + t02 * nz
        tau_n1 = t01 * nx + t11 * ny + t12 * nz
        tau_n2 = t02 * nx + t12 * ny + t22 * nz

        gtx = 0.5 * (grad_T[o, 0] + grad_T[nb, 0])
        gty = 0.5 * (grad_T[o, 1] + grad_T[nb, 1])
        gtz = 0.5 * (grad_T[o, 2] + grad_T[nb, 2])
        proj_T = gtx * ex + gty * ey + gtz * ez
        dT = T[nb] - T[o]
        corr_T = dT / d - proj_T
        gtx += corr_T * ex; gty += corr_T * ey; gtz += corr_T * ez
        cond = cp * (mu_lam / prandtl_lam + 0.5 * (mu_t[o] + mu_t[nb]) / prandtl_turb)
        qn = cond * (gtx * nx + gty * ny + gtz * nz)

        velx_f = 0.5 * (vel[o, 0] + vel[nb, 0])
        vely_f = 0.5 * (vel[o, 1] + vel[nb, 1])
        velz_f = 0.5 * (vel[o, 2] + vel[nb, 2])
        work = tau_n0 * velx_f + tau_n1 * vely_f + tau_n2 * velz_f

        gkx = 0.5 * (grad_k[o, 0] + grad_k[nb, 0])
        gky = 0.5 * (grad_k[o, 1] + grad_k[nb, 1])
        gkz = 0.5 * (grad_k[o, 2] + grad_k[nb, 2])
        gwx = 0.5 * (grad_omega[o, 0] + grad_omega[nb, 0])
        gwy = 0.5 * (grad_omega[o, 1] + grad_omega[nb, 1])
        gwz = 0.5 * (grad_omega[o, 2] + grad_omega[nb, 2])
        mut_f = 0.5 * (mu_t[o] + mu_t[nb])
        diff_k = (mu_lam + sigma_k1 * mut_f) * (gkx * nx + gky * ny + gkz * nz)
        diff_w = (mu_lam + sigma_w1 * mut_f) * (gwx * nx + gwy * ny + gwz * nz)

        a = int_areas[f]
        fvisc[f, 0] = 0.0
        fvisc[f, 1] = tau_n0 * a
        fvisc[f, 2] = tau_n1 * a
        fvisc[f, 3] = tau_n2 * a
        fvisc[f, 4] = (work + qn) * a
        fvisc[f, 5] = diff_k * a
        fvisc[f, 6] = diff_w * a


def viscous_internal_flux_gpu(
    int_owner: np.ndarray, int_neigh: np.ndarray,
    int_areas: np.ndarray, int_normals: np.ndarray,
    e_ON: np.ndarray, dist: np.ndarray,
    vel: np.ndarray, grad_vel: np.ndarray, mu_eff: np.ndarray,
    T: np.ndarray, grad_T: np.ndarray, mu_t: np.ndarray,
    k_field: np.ndarray, omega_field: np.ndarray,
    grad_k: np.ndarray, grad_omega: np.ndarray,
    mu_lam: float, cp: float, prandtl_lam: float, prandtl_turb: float,
    sigma_k1: float, sigma_w1: float,
) -> np.ndarray:
    """Host-side launcher，签名与 fvm_viscous_kernels._viscous_internal_flux_kernel
    一致，方便调用方直接替换。⚠️ 未经真实 GPU 硬件验证，见本文件模块级文档字符串。
    """
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA GPU not available in this environment")
    n = int_owner.shape[0]
    d_fvisc = cuda.device_array((n, 7), dtype=np.float64)

    threads_per_block = 256
    blocks = (n + threads_per_block - 1) // threads_per_block
    _viscous_internal_flux_kernel_gpu[blocks, threads_per_block](
        cuda.to_device(np.ascontiguousarray(int_owner, dtype=np.int64)),
        cuda.to_device(np.ascontiguousarray(int_neigh, dtype=np.int64)),
        cuda.to_device(np.ascontiguousarray(int_areas, dtype=np.float64)),
        cuda.to_device(np.ascontiguousarray(int_normals, dtype=np.float64)),
        cuda.to_device(np.ascontiguousarray(e_ON, dtype=np.float64)),
        cuda.to_device(np.ascontiguousarray(dist, dtype=np.float64)),
        cuda.to_device(np.ascontiguousarray(vel, dtype=np.float64)),
        cuda.to_device(np.ascontiguousarray(grad_vel, dtype=np.float64)),
        cuda.to_device(np.ascontiguousarray(mu_eff, dtype=np.float64)),
        cuda.to_device(np.ascontiguousarray(T, dtype=np.float64)),
        cuda.to_device(np.ascontiguousarray(grad_T, dtype=np.float64)),
        cuda.to_device(np.ascontiguousarray(mu_t, dtype=np.float64)),
        cuda.to_device(np.ascontiguousarray(k_field, dtype=np.float64)),
        cuda.to_device(np.ascontiguousarray(omega_field, dtype=np.float64)),
        cuda.to_device(np.ascontiguousarray(grad_k, dtype=np.float64)),
        cuda.to_device(np.ascontiguousarray(grad_omega, dtype=np.float64)),
        mu_lam, cp, prandtl_lam, prandtl_turb, sigma_k1, sigma_w1,
        d_fvisc,
    )
    return d_fvisc.copy_to_host()
