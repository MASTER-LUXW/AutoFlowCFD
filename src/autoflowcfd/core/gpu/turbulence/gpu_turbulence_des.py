"""
AutoFlowCFD V2.0 - GPU 版 DDES/IDDES 混合长度尺度 (#7 第四次评审第四轮)

与 core/turbulence/des.py::DDESModel/IDDESModel 对应的 CuPy 版本。

DDES/IDDES 在整个求解流程里唯一的作用是算出一个逐点长度尺度
`des_length_scale`，替换掉 GPUTurbulenceSST 内部 k 方程耗散项
`D_k = rho*k^1.5/l` 里原本用的 RANS 长度尺度（1/(beta_star*omega)
对应的隐式尺度）——GPUTurbulenceSST.compute_source_terms_gpu 已经
原生支持 `self.des_length_scale` 这个可选字段（见 gpu_turbulence_sst.py），
所以本文件不需要、也不实现一个独立的"GPU DDES 湍流模型"类去平行
GPUTurbulenceSST，只需要提供"算长度尺度、写回 sst_model.des_length_scale"
这两个函数，用法与 CPU 版 DDESModel.apply_to_sst_model/
IDDESModel.apply_to_sst_model_iddes 完全一致。

DDES/IDDES 都不需要 GPU 版 k/omega 输运（gpu_scalar_transport.py）才能
工作——两者只改长度尺度，不改变输运方程本身的结构；反过来，输运项本身
（对流+扩散）与是否启用 DDES/IDDES 无关，两者是正交的两处独立修复。
"""

from typing import Optional

from autoflowcfd.core.gpu import gpu_available, get_cupy


class GPUDDESModel:
    """GPU 版 DDES 长度尺度计算，与 CPU 版 DDESModel 公式逐字对应。

    Attributes:
        c_des: DES 常数（SST-DDES 标定值 0.65，见 CPU 版类文档）
        c_w1: 延迟参数（SST-DDES 重新标定值 20.0，见 CPU 版类文档）
    """

    def __init__(self, c_des: float = 0.65, c_w1: float = 20.0):
        if not gpu_available:
            raise RuntimeError("CuPy required for GPU turbulence model")
        self.c_des = c_des
        self.c_w1 = c_w1

    def compute_grid_scale_gpu(self, cell_volumes):
        """Delta = V^(1/3)，与 CPU 版 compute_grid_scale(method='cube_root') 一致。"""
        cp = get_cupy()
        return cp.abs(cell_volumes) ** (1.0 / 3.0)

    def compute_strain_rate_magnitude_gpu(self, grad_u):
        cp = get_cupy()
        S_ij = 0.5 * (grad_u + cp.transpose(grad_u, (0, 1, 3, 2)))
        return cp.sqrt(2.0 * cp.sum(S_ij * S_ij, axis=(2, 3)))

    def compute_vorticity_magnitude_gpu(self, grad_u):
        cp = get_cupy()
        Omega_ij = 0.5 * (grad_u - cp.transpose(grad_u, (0, 1, 3, 2)))
        return cp.sqrt(2.0 * cp.sum(Omega_ij * Omega_ij, axis=(2, 3)))

    def compute_shielding_function_gpu(self, d_w, nu_t, omega, nu, grad_u, kappa: float = 0.41):
        """F_d = 1 - tanh[(c_w1*r_d)^3]，与 CPU 版 compute_shielding_function
        逐字对应（含 |S|/|Omega| 联合尺度、分子粘度项两处 2012 年修正）。
        """
        cp = get_cupy()
        d_w = cp.maximum(d_w, 1e-6)
        omega = cp.maximum(omega, 1e-6)
        nu_t = cp.maximum(nu_t, 1e-10)

        S_mag = self.compute_strain_rate_magnitude_gpu(grad_u)
        Omega_mag = self.compute_vorticity_magnitude_gpu(grad_u)
        S_Omega_mag = cp.sqrt(0.5 * (S_mag ** 2 + Omega_mag ** 2))
        S_Omega_mag = cp.maximum(S_Omega_mag, 1e-6)

        r_d = (nu_t + nu) / (kappa ** 2 * d_w ** 2 * S_Omega_mag)
        r_d = cp.minimum(r_d, 10.0)
        f_d = 1.0 - cp.tanh((self.c_w1 * r_d) ** 3)
        return f_d

    def compute_effective_length_scale_gpu(self, k, omega, beta_star, delta, f_d):
        """l_eff = l_rans - f_d*max(0, l_rans-l_les)，与 CPU 版
        compute_effective_length_scale 逐字对应。delta 已要求与 k 同形状
        （(n_cells, n_sps)），广播由调用方负责，不在这里做 1D->2D tile
        （CPU 版会 tile，这里假定调用方直接传入正确形状，避免额外分支）。
        """
        cp = get_cupy()
        omega_safe = cp.maximum(omega, 1e-10)
        k_safe = cp.maximum(k, 0.0)
        l_rans = cp.sqrt(k_safe) / (beta_star * omega_safe)
        l_les = self.c_des * delta
        l_eff = l_rans - f_d * cp.maximum(0.0, l_rans - l_les)
        return cp.maximum(l_eff, 1e-10)

    def apply_to_sst_model_gpu(self, sst_model_gpu, d_w, cell_volumes, nu, grad_u):
        """把 DDES 应用到 GPU SST 模型：写 sst_model_gpu.des_length_scale。

        与 CPU 版 DDESModel.apply_to_sst_model 逐字对应。
        """
        cp = get_cupy()
        delta = self.compute_grid_scale_gpu(cell_volumes)
        n_sps = sst_model_gpu.k_field.shape[1]
        if delta.ndim == 1:
            delta = cp.tile(delta[:, None], (1, n_sps))

        k = sst_model_gpu.k_field
        omega = sst_model_gpu.omega_field
        nu_t = sst_model_gpu.nu_t

        f_d = self.compute_shielding_function_gpu(d_w, nu_t, omega, nu, grad_u)
        l_eff = self.compute_effective_length_scale_gpu(k, omega, sst_model_gpu.beta_star, delta, f_d)
        sst_model_gpu.des_length_scale = l_eff


class GPUIDDESModel(GPUDDESModel):
    """GPU 版 IDDES 长度尺度计算，与 CPU 版 IDDESModel 公式逐字对应
    （Shur et al. 2008 + Gritskevich et al. 2012 SST kw 标定，置信度
    说明见 CPU 版类文档：c_t/c_l 为中等置信度，h_wn 的非结构化网格
    替代方案为项目场景下必要近似，两处均与 CPU 版共享同一份"证据链"，
    这里不重复列出，只逐字对应实现）。
    """

    def __init__(self, c_des: float = 0.78, c_w1: float = 20.0,
                 c_w: float = 0.15, c_t: float = 1.87, c_l: float = 5.0):
        super().__init__(c_des, c_w1)
        self.c_w = c_w
        self.c_t = c_t
        self.c_l = c_l

    def compute_grid_scale_iddes_gpu(self, d_w, h_max, h_wn):
        cp = get_cupy()
        inner = cp.maximum(cp.maximum(self.c_w * d_w, self.c_w * h_max), h_wn)
        return cp.minimum(inner, h_max)

    def compute_alpha_gpu(self, d_w, h_max):
        cp = get_cupy()
        return 0.25 - d_w / cp.maximum(h_max, 1e-12)

    def compute_f_b_gpu(self, alpha):
        cp = get_cupy()
        return cp.minimum(2.0 * cp.exp(-9.0 * alpha ** 2), 1.0)

    def compute_f_e1_gpu(self, alpha):
        cp = get_cupy()
        return cp.where(
            alpha >= 0.0,
            2.0 * cp.exp(-11.09 * alpha ** 2),
            2.0 * cp.exp(-9.0 * alpha ** 2),
        )

    def compute_f_e2_gpu(self, nu_t, nu, d_w, S_Omega_mag, kappa: float = 0.41):
        cp = get_cupy()
        d_w_safe = cp.maximum(d_w, 1e-6)
        S_Omega_safe = cp.maximum(S_Omega_mag, 1e-6)
        denom = kappa ** 2 * d_w_safe ** 2 * S_Omega_safe
        r_dt = nu_t / denom
        r_dl = nu / denom
        f_t = cp.tanh((self.c_t ** 2 * r_dt) ** 3)
        f_l = cp.tanh((self.c_l ** 2 * r_dl) ** 10)
        return 1.0 - cp.maximum(f_t, f_l)

    def compute_effective_length_scale_iddes_gpu(self, k, omega, beta_star, delta_iddes, f_b, f_e):
        cp = get_cupy()
        omega_safe = cp.maximum(omega, 1e-10)
        k_safe = cp.maximum(k, 0.0)
        l_rans = cp.sqrt(k_safe) / (beta_star * omega_safe)
        l_les = self.c_des * delta_iddes
        l_iddes = f_b * (1.0 + f_e) * l_rans + (1.0 - f_b) * l_les
        return cp.maximum(l_iddes, 1e-10)

    def apply_to_sst_model_iddes_gpu(self, sst_model_gpu, d_w, h_max, h_wn, nu, grad_u):
        """把 IDDES 应用到 GPU SST 模型，与 CPU 版
        IDDESModel.apply_to_sst_model_iddes 逐字对应。

        Args:
            d_w: (n_cells, n_sps) CuPy 数组
            h_max, h_wn: (n_cells,) CuPy 数组（逐单元几何量，调用方一次性
                算好并常驻显存，见 gpu_solver_init.py 里 IDDES 分支的缓存）
            nu, grad_u: 同 apply_to_sst_model_gpu
        """
        cp = get_cupy()
        n_sps = sst_model_gpu.k_field.shape[1]
        h_max_b = cp.tile(h_max[:, None], (1, n_sps))
        h_wn_b = cp.tile(h_wn[:, None], (1, n_sps))

        k = sst_model_gpu.k_field
        omega = sst_model_gpu.omega_field
        nu_t = sst_model_gpu.nu_t

        S_mag = self.compute_strain_rate_magnitude_gpu(grad_u)
        Omega_mag = self.compute_vorticity_magnitude_gpu(grad_u)
        S_Omega_mag = cp.maximum(cp.sqrt(0.5 * (S_mag ** 2 + Omega_mag ** 2)), 1e-6)

        alpha = self.compute_alpha_gpu(d_w, h_max_b)
        f_b = self.compute_f_b_gpu(alpha)
        f_e1 = self.compute_f_e1_gpu(alpha)
        f_e2 = self.compute_f_e2_gpu(nu_t, nu, d_w, S_Omega_mag)
        f_e = cp.maximum(f_e1 - 1.0, 0.0) * f_e2

        delta_iddes = self.compute_grid_scale_iddes_gpu(d_w, h_max_b, h_wn_b)
        l_iddes = self.compute_effective_length_scale_iddes_gpu(
            k, omega, sst_model_gpu.beta_star, delta_iddes, f_b, f_e
        )
        sst_model_gpu.des_length_scale = l_iddes
