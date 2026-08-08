"""Numba CPU kernel：粘性通量（应力张量 + 热传导 + 湍流扩散），逐面循环版。

对应 fvm_viscous_residual.py 的 ViscousRANSResidual._viscous_flux /
_stress_dot_normal 内部面部分——把向量化 numpy（face-averaged 梯度 +
over-relaxed 修正 + 应力张量点法向）逐项翻译成显式逐面循环。边界面部分
（涉及 wall function、ghost 温度/湍流量）仍留在 fvm_viscous_residual.py
里用 numpy 实现（边界面数量远小于内部面，不是热点，且逻辑和壁面函数
Newton 迭代交织在一起，保持 numpy 实现更清楚）。
"""

import numpy as np

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range


@njit(inline='always', cache=True)
def _stress_dot_normal_one(gv, nx, ny, nz, mu, out):
    """tau.n for one face, tau = mu(grad u + grad u^T - 2/3 div(u) I).

    gv: (3, 3) velocity gradient [component, direction]. out: (3,) written
    in place with tau_n.
    """
    divu = gv[0, 0] + gv[1, 1] + gv[2, 2]
    two_thirds_mu_divu = (2.0 / 3.0) * mu * divu
    # tau[i,j] = mu*(gv[i,j]+gv[j,i]) - (i==j)*two_thirds_mu_divu
    t00 = mu * (gv[0, 0] + gv[0, 0]) - two_thirds_mu_divu
    t01 = mu * (gv[0, 1] + gv[1, 0])
    t02 = mu * (gv[0, 2] + gv[2, 0])
    t11 = mu * (gv[1, 1] + gv[1, 1]) - two_thirds_mu_divu
    t12 = mu * (gv[1, 2] + gv[2, 1])
    t22 = mu * (gv[2, 2] + gv[2, 2]) - two_thirds_mu_divu
    out[0] = t00 * nx + t01 * ny + t02 * nz
    out[1] = t01 * nx + t11 * ny + t12 * nz
    out[2] = t02 * nx + t12 * ny + t22 * nz


@njit(parallel=True, cache=True)
def _viscous_internal_flux_kernel(
    int_owner: np.ndarray, int_neigh: np.ndarray,
    int_areas: np.ndarray, int_normals: np.ndarray,
    e_ON: np.ndarray, dist: np.ndarray,
    vel: np.ndarray,            # (n_cells, 3)
    grad_vel: np.ndarray,       # (n_cells, 3, 3)
    mu_eff: np.ndarray,         # (n_cells,)
    T: np.ndarray,              # (n_cells,)
    grad_T: np.ndarray,         # (n_cells, 3)
    mu_t: np.ndarray,           # (n_cells,)
    k_field: np.ndarray, omega_field: np.ndarray,
    grad_k: np.ndarray, grad_omega: np.ndarray,   # (n_cells, 3)
    mu_lam: float, cp: float,
    prandtl_lam: float, prandtl_turb: float,
    sigma_k1: float, sigma_w1: float,
) -> np.ndarray:
    """Per-internal-face viscous flux contribution (7 vars), NOT yet
    multiplied by area or scattered into cells - caller does both (same
    two steps ViscousRANSResidual._viscous_flux already does), since the
    scatter step writes to two cells per face and must stay a serial (or
    otherwise race-free) pass, unlike this purely per-face loop."""
    n = int_owner.shape[0]
    fvisc = np.zeros((n, 7), dtype=np.float64)

    for f in prange(n):
        # gv_face/tau_n are per-face scratch buffers - MUST be allocated
        # inside the prange body, not hoisted above the loop: prange
        # spreads iterations across threads, and a buffer shared across
        # iterations would let concurrent threads clobber each other's
        # in-progress values (a real, confirmed data race - caught by
        # comparing against the numpy reference on a real mesh, which
        # this exact structure produced silently wrong stress/energy
        # flux values for while leaving the (per-face, no shared state)
        # turbulent-diffusion columns unaffected).
        gv_face = np.empty((3, 3))
        tau_n = np.empty(3)

        o = int_owner[f]
        nb = int_neigh[f]
        nx = int_normals[f, 0]; ny = int_normals[f, 1]; nz = int_normals[f, 2]
        ex = e_ON[f, 0]; ey = e_ON[f, 1]; ez = e_ON[f, 2]
        d = dist[f]

        # Face-averaged velocity gradient with over-relaxed directional
        # correction along the owner->neighbour line.
        proj = np.zeros(3)
        for i in range(3):
            for j in range(3):
                gij = 0.5 * (grad_vel[o, i, j] + grad_vel[nb, i, j])
                gv_face[i, j] = gij
            proj[i] = gv_face[i, 0] * ex + gv_face[i, 1] * ey + gv_face[i, 2] * ez
        for i in range(3):
            dvel_i = vel[nb, i] - vel[o, i]
            corr_i = dvel_i / d - proj[i]
            gv_face[i, 0] += corr_i * ex
            gv_face[i, 1] += corr_i * ey
            gv_face[i, 2] += corr_i * ez

        mu_f = 0.5 * (mu_eff[o] + mu_eff[nb])
        _stress_dot_normal_one(gv_face, nx, ny, nz, mu_f, tau_n)

        # Temperature gradient, same face-average + correction.
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
        work = tau_n[0] * velx_f + tau_n[1] * vely_f + tau_n[2] * velz_f

        # Turbulent variable diffusion.
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
        fvisc[f, 1] = tau_n[0] * a
        fvisc[f, 2] = tau_n[1] * a
        fvisc[f, 3] = tau_n[2] * a
        fvisc[f, 4] = (work + qn) * a
        fvisc[f, 5] = diff_k * a
        fvisc[f, 6] = diff_w * a

    return fvisc
