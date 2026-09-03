"""
AutoFlowCFD V2.0 - GPU 版物理通量计算（欧拉 + 粘性）

与 core/fr_flux_kernels_pointwise.py 对应的 CuPy 向量化版本。
用 CuPy 的逐元素运算替代 numba @njit 的逐点循环，底层自动走 cuBLAS/cuDA。

包含：
- euler_physical_flux_gpu: 欧拉物理通量张量 F_i(Q)
- viscous_physical_flux_gpu: 粘性物理通量张量 G_i(Q, grad_vel, grad_T)
- conserved_to_primitive_gpu: 守恒变量 → 原始变量
- primitive_to_conserved_gpu: 原始变量 → 守恒变量
"""

import numpy as np
from autoflowcfd.core.gpu import get_cupy

GAMMA = 1.4
R_AIR = 287.0


def conserved_to_primitive_gpu(U):
    """GPU 版守恒变量→原始变量。U=(rho,rho*u,rho*v,rho*w,rho*E) -> Q=(rho,u,v,w,p)。

    Args:
        U: CuPy 数组 (..., 5+)

    Returns:
        Q: CuPy 数组 (..., 5)
    """
    cp = get_cupy()
    rho = cp.maximum(U[..., 0], 1e-10)
    u = U[..., 1] / rho
    v = U[..., 2] / rho
    w = U[..., 3] / rho
    E = U[..., 4] / rho
    ke = 0.5 * (u**2 + v**2 + w**2)
    p = (GAMMA - 1.0) * rho * (E - ke)
    p = cp.maximum(p, 10.0)
    return cp.stack([rho, u, v, w, p], axis=-1)


def primitive_to_conserved_gpu(Q):
    """GPU 版原始变量→守恒变量。Q=(rho,u,v,w,p) -> U=(rho,rho*u,rho*v,rho*w,rho*E)。

    Args:
        Q: CuPy 数组 (..., 5)

    Returns:
        U: CuPy 数组 (..., 5)
    """
    cp = get_cupy()
    rho, u, v, w, p = Q[..., 0], Q[..., 1], Q[..., 2], Q[..., 3], Q[..., 4]
    rho_safe = cp.maximum(rho, 1e-10)
    ke = 0.5 * (u**2 + v**2 + w**2)
    e_internal = p / ((GAMMA - 1.0) * rho_safe)
    E = e_internal + ke
    return cp.stack([rho, rho * u, rho * v, rho * w, rho * E], axis=-1)


def euler_physical_flux_gpu(Q):
    """GPU 版欧拉物理通量张量 F_i(Q)。

    与 core/fr_flux_kernels_pointwise.py::euler_physical_flux_batch 公式一致。

    Args:
        Q: CuPy 数组 (N, 5)，(rho, u, v, w, p)

    Returns:
        F: CuPy 数组 (N, 3, 5)
    """
    cp = get_cupy()
    rho = Q[..., 0]
    u = Q[..., 1]
    v = Q[..., 2]
    w = Q[..., 3]
    p = Q[..., 4]
    rho_safe = cp.maximum(rho, 1e-10)

    ke = 0.5 * (u**2 + v**2 + w**2)
    e_internal = p / ((GAMMA - 1.0) * rho_safe)
    rhoE = rho * (e_internal + ke)
    H = (rhoE + p) / rho_safe

    # 质量通量
    mf0 = rho * u
    mf1 = rho * v
    mf2 = rho * w

    N = Q.shape[0]
    F = cp.zeros((N, 3, 5), dtype=cp.float64)

    # x-direction
    F[:, 0, 0] = mf0
    F[:, 0, 1] = mf0 * u + p
    F[:, 0, 2] = mf0 * v
    F[:, 0, 3] = mf0 * w
    F[:, 0, 4] = rho * H * u

    # y-direction
    F[:, 1, 0] = mf1
    F[:, 1, 1] = mf1 * u
    F[:, 1, 2] = mf1 * v + p
    F[:, 1, 3] = mf1 * w
    F[:, 1, 4] = rho * H * v

    # z-direction
    F[:, 2, 0] = mf2
    F[:, 2, 1] = mf2 * u
    F[:, 2, 2] = mf2 * v
    F[:, 2, 3] = mf2 * w + p
    F[:, 2, 4] = rho * H * w

    return F


def viscous_physical_flux_gpu(Q, grad_vel, grad_T, mu, Pr, mu_t=None, Pr_t=0.9):
    """GPU 版粘性物理通量张量 G_i(Q, grad_vel, grad_T)。

    与 core/fr_flux_kernels_pointwise.py::viscous_physical_flux_batch 公式一致。
    Boussinesq 假设：mu_total = mu + mu_t。

    真实 bug 修复（问题清单 #5 排查附带发现，2026-09-02，用真实含多源/
    混合拆分面的合成网格 + numpy-as-cupy 替身 + 完整 `GPUFRSolver.
    step()` 首次真正调用到体积项分支才暴露——之前只有 `gpu_viscous.py`
    的两处界面校正调用点被验证过，那两处调用前都会先 `.reshape(n*n_fp,
    5)`/`.reshape(n*n_fp,3,3)` 显式压平成真正的 2D/3D 输入，恰好绕开了
    这个问题）：`Q`/`grad_T` 用 `[..., k]`（省略号，天然支持任意数量的
    前导批量维）读取，但 `grad_vel`/`G` 一直用 `[:, i, j]`（只认定
    *恰好一个*前导批量维）——`compute_viscous_residual_fr_gpu` 的体积项
    分支直接传入未压平的 `Q=(n_cells,n_sps,5)`/`grad_vel=(n_cells,n_sps,
    3,3)`（两个前导批量维），`grad_vel[:,0,0]` 会把第二个前导维
    （n_sps）误当成"速度分量 i"去索引，产出形状错误
    （`(n_cells,n_sps,3,3)[:,0,0]` → `(n_cells,3)`，把 n_sps 维和"速度
    分量 i"维搞混，第 3 维（真正的空间方向 j）被误当成结果的第二维）
    的中间量，最终在 `G[:,0,1]=tau_xx` 这行因为形状对不上而
    `ValueError`。修复：`grad_vel`/`G` 同样改用 `[..., i, j]`，与
    `Q`/`grad_T` 保持一致的、对前导批量维数量无感知的约定——对已经
    压平成 2D/3D 的两处界面校正调用点，`[...,i,j]` 与原来的
    `[:,i,j]` 结果完全相同（只有一个前导维时二者等价），不改变其行为；
    只有体积项这个此前从未被走到过的调用点会因此获得正确结果。

    Args:
        Q: CuPy 数组 (..., 5)，(rho, u, v, w, p)——`...` 可以是单个批量
            维（界面校正调用点压平后的 `(N,5)`）或多个（体积项调用点
            的 `(n_cells,n_sps,5)`）
        grad_vel: CuPy 数组 (..., 3, 3)，速度梯度 grad_vel[...,i,j] =
            du_i/dx_j，前导批量维数量与 `Q` 一致
        grad_T: CuPy 数组 (..., 3)，温度梯度
        mu: 分子动力粘度（标量）
        Pr: 分子普朗特数（标量）
        mu_t: CuPy 数组 (...,) 或标量，湍流涡粘度（默认 0）
        Pr_t: 湍流普朗特数（默认 0.9）

    Returns:
        G: CuPy 数组 (..., 3, 5)，前导批量维与 `Q` 一致
    """
    cp = get_cupy()
    batch_shape = Q.shape[:-1]

    rho = Q[..., 0]
    u = Q[..., 1]
    v = Q[..., 2]
    w = Q[..., 3]

    # 有效粘度（mu_t 标量/数组两种情况下 mu + mu_t 的计算完全相同，
    # 此前拆成三个分支但后两个分支代码逐字相同，合并为一个）
    mu_total = mu if mu_t is None else mu + mu_t

    # 热导率：k = mu * Cp / Pr（分子）+ mu_t * Cp / Pr_t（湍流）
    Cp = GAMMA * R_AIR / (GAMMA - 1.0)
    if np.isscalar(mu_total):
        k_eff = mu_total * Cp / Pr if np.isscalar(Pr) else mu_total * Cp / Pr
    else:
        # mu_total 是数组时，mu 部分用 Pr，mu_t 部分用 Pr_t
        k_molecular = mu * Cp / Pr
        k_turbulent = mu_t * Cp / Pr_t
        k_eff = k_molecular + k_turbulent

    # 速度梯度分量（`[..., i, j]`，见函数文档"真实 bug 修复"一节——不能
    # 用 `[:, i, j]`，那只认定恰好一个前导批量维）
    dudx = grad_vel[..., 0, 0]
    dudy = grad_vel[..., 0, 1]
    dudz = grad_vel[..., 0, 2]
    dvdx = grad_vel[..., 1, 0]
    dvdy = grad_vel[..., 1, 1]
    dvdz = grad_vel[..., 1, 2]
    dwdx = grad_vel[..., 2, 0]
    dwdy = grad_vel[..., 2, 1]
    dwdz = grad_vel[..., 2, 2]

    # 散度
    div_vel = dudx + dvdy + dwdz

    # 应力张量（Boussinesq 假设）
    # tau_ij = mu_total * (du_i/dx_j + du_j/dx_i) - 2/3 * mu_total * div(V) * delta_ij
    tau_xx = mu_total * (2.0 * dudx - (2.0/3.0) * div_vel)
    tau_yy = mu_total * (2.0 * dvdy - (2.0/3.0) * div_vel)
    tau_zz = mu_total * (2.0 * dwdz - (2.0/3.0) * div_vel)
    tau_xy = mu_total * (dudy + dvdx)
    tau_xz = mu_total * (dudz + dwdx)
    tau_yz = mu_total * (dvdz + dwdy)

    # 温度梯度分量（`[..., j]`，同上）
    dTdx = grad_T[..., 0]
    dTdy = grad_T[..., 1]
    dTdz = grad_T[..., 2]

    G = cp.zeros(batch_shape + (3, 5), dtype=cp.float64)

    # 真实 bug 修复（第二个独立发现，2026-09-03，用真实网格 CPU/GPU
    # 界面项交叉验证时发现——均匀流场/纯前面的 shape 修复无法捕捉这个
    # bug：均匀流场下 grad_T 恒为 0，`+k_eff*dTdx`/`-k_eff*dTdx` 结果
    # 无差别；本次改动前的"3D 批量 vs 手动展平 2D"不变量测试也无法
    # 捕捉，因为那只验证同一个（错误）公式在不同批量维数下自洽，不
    # 对照 CPU 参考实现）：热传导项符号搞反了。Fourier 定律
    # `q = -k*grad(T)`（热流方向与温度梯度相反——热量从高温流向
    # 低温），与 CPU 版 `flux_kernels.py::viscous_physical_flux_point`
    # 的 `qx=-k_cond*grad_T[0]`（及 qy/qz）逐字对应；此前这里写成
    # `+k_eff*dTdx`，方向反了，任何有非零温度梯度的真实流动（几乎
    # 全部流动，除非等温）粘性残差的能量分量都会算错——动量分量
    # （tau 相关项）本身不受影响，公式恰好一致，这也是此前"零梯度
    # 单元测试"能通过但真实非均匀流场交叉验证会暴露问题的原因。
    G[..., 0, 0] = 0.0
    G[..., 0, 1] = tau_xx
    G[..., 0, 2] = tau_xy
    G[..., 0, 3] = tau_xz
    G[..., 0, 4] = (u * tau_xx + v * tau_xy + w * tau_xz) - k_eff * dTdx

    # y-direction: G_1
    G[..., 1, 0] = 0.0
    G[..., 1, 1] = tau_xy
    G[..., 1, 2] = tau_yy
    G[..., 1, 3] = tau_yz
    G[..., 1, 4] = (u * tau_xy + v * tau_yy + w * tau_yz) - k_eff * dTdy

    # z-direction: G_2
    G[..., 2, 0] = 0.0
    G[..., 2, 1] = tau_xz
    G[..., 2, 2] = tau_yz
    G[..., 2, 3] = tau_zz
    G[..., 2, 4] = (u * tau_xz + v * tau_yz + w * tau_zz) - k_eff * dTdz

    return G
