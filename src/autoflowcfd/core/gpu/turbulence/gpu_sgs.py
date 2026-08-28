"""
AutoFlowCFD V2.0 - GPU 版 WALE 亚格子应力模型 (#7 第四次评审第四轮)

与 core/turbulence/sgs.py::WALEModel 对应的 CuPy 版本。只移植生产路径
真正调用的入口链（compute_eddy_viscosity + 其内部
compute_strain_and_rotation_tensors/compute_second_invariants/
compute_wale_invariant），不移植 compute_velocity_gradients/
compute_subgrid_stress——这两个方法经 grep 全仓库确认在 CPU 版自身
生产代码里也是死代码（唯一出现的地方就是 sgs.py 自己的定义处），没有
移植的必要。SmagorinskyModel 同理未移植：`init_turbulence_models`
（fr_solver/turbulence.py）只构造 WALEModel（WMLES/LES 两个分支都是
`solver.sgs_model = WALEModel()`），SmagorinskyModel 在生产路径里从未
被实例化。

公式与 CPU 版 WALEModel.compute_eddy_viscosity 逐字对应，含该类文档记录
的"WALE 不变量必须去迹"修复（Nicoud & Ducros 1999 原始定义）。
"""

from autoflowcfd.core.gpu import gpu_available, get_cupy


class GPUWALEModel:
    """GPU 版 WALE 亚格子涡粘模型。

    Attributes:
        c_wale: WALE 模型常数
        nu_t: 最近一次 compute_eddy_viscosity_gpu 算出的涡粘系数场（CuPy）
    """

    def __init__(self, c_wale: float = 0.325):
        if not gpu_available:
            raise RuntimeError("CuPy required for GPU turbulence model")
        self.c_wale = c_wale
        self.nu_t = None

    def compute_strain_and_rotation_tensors_gpu(self, grad_u):
        """S_ij = 0.5*(grad_u + grad_u^T)，Omega_ij = 0.5*(grad_u - grad_u^T)。

        grad_u: (n_cells, n_sps, 3, 3) CuPy 数组。
        """
        cp = get_cupy()
        S_ij = 0.5 * (grad_u + cp.transpose(grad_u, (0, 1, 3, 2)))
        Omega_ij = 0.5 * (grad_u - cp.transpose(grad_u, (0, 1, 3, 2)))
        return S_ij, Omega_ij

    def compute_second_invariants_gpu(self, S_ij, Omega_ij):
        """S_sq = S_ij*S_ij，Omega_sq = Omega_ij*Omega_ij，形状 (n_cells, n_sps)。"""
        cp = get_cupy()
        S_sq = cp.sum(S_ij * S_ij, axis=(2, 3))
        Omega_sq = cp.sum(Omega_ij * Omega_ij, axis=(2, 3))
        return S_sq, Omega_sq

    def compute_wale_invariant_gpu(self, S_ij, Omega_ij):
        """WALE 核心不变量（已去迹），与 CPU 版
        WALEModel.compute_wale_invariant 逐字对应，含该方法文档记录的
        "漏迹会系统性高估涡粘 3 倍"修复验证（轴对称纯应变反例）。

        L_ij = S_ik*S_kj + Omega_ik*Omega_kj
        L_ij^d = L_ij - (1/3)*trace(L)*delta_ij
        L_sq = L_ij^d * L_ij^d
        """
        cp = get_cupy()
        L_ij = (
            cp.matmul(S_ij, S_ij) + cp.matmul(Omega_ij, Omega_ij)
        )  # (n_cells, n_sps, 3, 3)
        trace_L = L_ij[..., 0, 0] + L_ij[..., 1, 1] + L_ij[..., 2, 2]
        eye3 = cp.eye(3, dtype=L_ij.dtype)
        L_ij_traceless = L_ij - (trace_L / 3.0)[..., None, None] * eye3
        L_sq = cp.sum(L_ij_traceless * L_ij_traceless, axis=(2, 3))
        return L_sq

    def compute_eddy_viscosity_gpu(self, grad_u, delta):
        """WALE 涡粘系数，与 CPU 版
        WALEModel.compute_eddy_viscosity 逐字对应：

        nu_t = (c_wale*delta)^2 * L_sq^1.5 / (S_sq^2.5 + L_sq^1.25)，
        钳制到 <= 1e-3（与 CPU 版同一上限，均为"防止数值不稳定"的
        经验安全阀，非物理推导值）。

        Args:
            grad_u: (n_cells, n_sps, 3, 3) CuPy 数组
            delta: (n_cells, n_sps) CuPy 数组，网格尺度

        Returns:
            nu_t: (n_cells, n_sps) CuPy 数组
        """
        cp = get_cupy()
        S_ij, Omega_ij = self.compute_strain_and_rotation_tensors_gpu(grad_u)
        S_sq, Omega_sq = self.compute_second_invariants_gpu(S_ij, Omega_ij)
        L_sq = self.compute_wale_invariant_gpu(S_ij, Omega_ij)

        S_sq = cp.maximum(S_sq, 1e-10)
        L_sq = cp.maximum(L_sq, 1e-10)
        delta = cp.maximum(delta, 1e-10)

        numerator = L_sq ** 1.5
        denominator = S_sq ** 2.5 + L_sq ** 1.25
        nu_t = (self.c_wale * delta) ** 2 * numerator / cp.maximum(denominator, 1e-10)
        nu_t = cp.minimum(nu_t, 1e-3)
        nu_t = cp.where(cp.isfinite(nu_t), nu_t, 0.0)

        self.nu_t = nu_t
        return nu_t

    def cleanup(self):
        """释放 GPU 资源。"""
        if self.nu_t is not None:
            del self.nu_t
            self.nu_t = None
