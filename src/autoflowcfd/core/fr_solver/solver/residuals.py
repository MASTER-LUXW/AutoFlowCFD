"""AutoFlowCFD V2.0 - `FRSolver` 的残差、湍流与边界委托方法（mixin，只含方法）。

从 `core/fr_solver/solver.py` 拆出（2026-09-25）。
"""

from typing import Any, Dict, Optional

import numpy as np

from autoflowcfd.core.fr_residual.inviscid import compute_inviscid_residual_fr
from autoflowcfd.core.fr_residual.viscous import compute_viscous_residual as compute_viscous_residual_ldg
from autoflowcfd.core.utils import solver_helpers

from .. import boundary as fr_solver_boundary
from .. import turbulence as fr_solver_turbulence


class _SolverResidualMixin:
    """残差组装、湍流模型与边界幽灵态的委托入口。"""

    def _init_turbulence_models(self, n_cells: int, n_sps: int):
        """初始化湍流模型（委托给 fr_solver_turbulence）。"""
        fr_solver_turbulence.init_turbulence_models(self, n_cells, n_sps)
    
    def _build_boundary_ghost_provider(self, bc_overrides: Dict[str, Dict[str, Any]]):
        """构建边界幽灵态提供者 (BD-01)，委托给 fr_solver_boundary。"""
        return fr_solver_boundary.build_boundary_ghost_provider(self, bc_overrides)

    def compute_turbulence_source(self, dt) -> Optional[tuple]:
        """计算湍流模型源项（委托给 fr_solver_turbulence）。dt 可以是标量
        （DUAL_TIME 物理步长）或逐 SP 数组（稳态加速模式的局部 CFL
        步长 dt_local）——见 fr_solver_turbulence.compute_turbulence_source
        文档。"""
        return fr_solver_turbulence.compute_turbulence_source(self, dt)

    def apply_turbulence_corrections(self):
        """应用湍流模型的修正（SGS 涡粘系数），委托给 fr_solver_turbulence。

        WMLES 壁面剪应力**不**在这里施加——它是一个真正的残差贡献项，
        必须在时间积分*之前*参与残差组装才能生效，见
        compute_viscous_residual() 里的调用与该方法文档（T-05 修复：
        此前在这里调用，而这里在 step() 中排在状态更新*之后*，对本步
        毫无影响，架构上不可能生效）。
        """
        fr_solver_turbulence.apply_turbulence_corrections(self)

    def compute_inviscid_residual(self):
        """
        计算无粘残差 (S-02/S-04)。

        真实的曲边/坍缩坐标 FR 离散：体积项用逆变通量 (contravariant flux)
        散度实现（度量项一致，满足自由流场保持性/离散GCL），界面项用基于
        真实单元-面连接关系的 AUSM+up 黎曼求解 + Radau/VCJH 校正函数投影，
        边界面通过 boundary_ghost_provider (BD-01) 构造物理正确的幽灵态。

        取代旧版本"用全场平均态+硬编码法向量冒充相邻单元"的伪校正项——
        详见 core/fr_residual_inviscid.py 模块文档与
        tests/unit/test_fr_residual_inviscid.py 的自由流场保持性验证。
        """
        self.state._update_primitives()

        if not self.state.U.flags['C_CONTIGUOUS']:
            self.state.U = np.ascontiguousarray(self.state.U)

        # GPU 分发 (B-01)：请求 GPU 后端时走 CuPy 加速路径。
        # P0（阶数延续热身阶段）使用 CuPy RawKernel（core/gpu/residual/gpu_p0_inviscid.py）；
        # P>=1 高阶 FR 使用 CuPy 向量化实现（core/gpu/residual/gpu_inviscid.py）。
        # GPU 不可用时自动回退 CPU。
        if self.backend_type == "gpu":
            if self.mesh.n_points_1d == 1:
                from ..gpu.residual.gpu_p0_inviscid import compute_inviscid_residual_p0_cupy
                res_euler = compute_inviscid_residual_p0_cupy(
                    self.state.U, self.mesh,
                    boundary_ghost_provider=self.boundary_ghost_provider,
                    mach_ref=self.freestream["mach_ref"],
                )
            else:
                # P>=1 高阶 FR GPU 路径
                from ..gpu.residual.gpu_inviscid import compute_inviscid_residual_fr_gpu
                res_euler = compute_inviscid_residual_fr_gpu(
                    self.state.U, self.mesh, self.ops,
                    boundary_ghost_provider=self.boundary_ghost_provider,
                    mach_ref=self.freestream["mach_ref"],
                )
        else:
            res_euler = compute_inviscid_residual_fr(
                self.state.U, self.mesh, self.ops,
                boundary_ghost_provider=self.boundary_ghost_provider,
                mach_ref=self.freestream["mach_ref"],
                entropy_stable_volume=self.entropy_stable_volume_enabled,
            )

        if self.state.n_vars > 5:
            # 湍流量 (k, omega) 的对流输运项当前仍由 compute_turbulence_source
            # 单独处理（局部源项积分，不含对流通量），此处只补零占位维度，
            # 不在这里静默引入未经验证的湍流对流项。
            res_full = np.zeros((res_euler.shape[0], res_euler.shape[1], self.state.n_vars))
            res_full[:, :, :5] = res_euler
            return res_full
        return res_euler

    def compute_viscous_residual(self, mu_t_turb=None, nu_av=None):
        """
        计算粘性残差 (S-03)。

        Args:
            mu_t_turb: 本步冻结的湍流动力涡粘 `rho*nu_t`（`step()` 在湍流更新
                之后按步前状态求一次，本步全部残差求值共用）。None 时按当前
                状态求。冻结是全部后端同一个算子分裂约定（单机 GPU、多 GPU、
                CPU 分布式都整步冻结）；2026-09-25 以前单机 CPU 在每次残差
                求值里按**试探态**的 rho 重算，与其余后端每步差 O(dt)，隐式步
                下直接让 Jacobian 不同（分布式 n_ranks=1 对照里块 Jacobi 差 1e-3）。
            nu_av: 本步冻结的问题单元人工扩散系数（运动粘度，
                `compute_artificial_diffusivity_field` 在步前状态上的结果）。None
                且启用人工粘性时按当前状态求。冻结的理由与 mu_t_turb 相同：系数
                是"本步参数"，隐式步的残差与 Jacobian 看到同一个系数，判据里的
                max 不进入 Newton 线性化。

        真实的 BR1 面耦合粘性离散（core/fr_viscous_flux.py），并把湍流模型
        算出的涡粘系数真正耦合进应力张量/热传导（T-01/T-04/T-06 修复：
        此前调用处从不传湍流粘度，粘性通量永远只用分子粘度 1.8e-5，
        SST/DDES/WALE 算出的 nu_t 场只在自身模型内部自用，从未进入
        动量/能量方程的扩散项）。

        Returns:
            viscous_res: 粘性残差（WMLES 激活时已叠加壁面剪应力修正，
                见 solver_helpers.compute_wmles_wall_stress_correction
                文档 T-05 修复说明——必须在这里（残差组装、时间积分之前）
                施加才能真正影响本步的解，而不是像此前那样在状态更新
                之后才计算）；启用人工粘性时已叠加 `div(nu grad U)`
        """
        res = compute_viscous_residual_ldg(
            self.state.U, self.state.Q, self.ops, self.mesh,
            mu=self.mu_molecular,
            mu_t_field=self._get_turbulent_viscosity_field(mu_t_turb),
            boundary_ghost_provider=self.boundary_ghost_provider,
        )

        if self.wmles_model is not None:
            wall_stress_correction = solver_helpers.compute_wmles_wall_stress_correction(self)
            if wall_stress_correction is not None:
                res = res + wall_stress_correction[..., : res.shape[-1]]

        # 问题单元人工粘性：全部守恒变量的拉普拉斯（施加形式与判据见
        # `fr_operators/artificial_viscosity/entropy_viscosity.py`）。不走 mu_t
        # 通道 —— 那样动量走速度梯度应力、能量走温度传导，再配一个单独的质量
        # 扩散，三者不自洽（实测造出冷点并扩散）。复用湍流输运已验证的标量扩散
        # 装配（边界齐次 Neumann，五个守恒量都严格守恒）。
        if nu_av is None:
            nu_av = self.compute_artificial_diffusivity_field()
        if nu_av is not None:
            res = res + self._artificial_diffusion_residual(nu_av)[..., : res.shape[-1]]

        return res

    def compute_artificial_diffusivity_field(self) -> Optional[np.ndarray]:
        """当前状态上的问题单元人工扩散系数 nu (n_cells, n_sps)（运动粘度）；未启用返回 None。"""
        if not getattr(self, "artificial_viscosity_enabled", False):
            return None
        from autoflowcfd.core.fr_operators.artificial_viscosity import (
            compute_artificial_diffusivity,
        )
        from autoflowcfd.core.fr_residual.viscous import compute_scalar_gradient

        return compute_artificial_diffusivity(
            self.state.U, int(getattr(self, "current_order", self.order)),
            self.mesh.cell_volumes,
            np.arange(self.mesh.n_cells) < int(self.mesh.n_prism_cells),
            lambda phi: compute_scalar_gradient(phi, self.ops, self.mesh),
            alpha_av=self.artificial_viscosity_alpha)

    def _artificial_diffusion_residual(self, nu_av: np.ndarray) -> np.ndarray:
        """`div(nu grad U_k)`，k = 0..4（dU/dt 约定），形状同 `state.U`。"""
        from autoflowcfd.core.fr_operators.artificial_viscosity import (
            artificial_diffusion_residual,
        )
        from autoflowcfd.core.turbulence.transport import (
            compute_scalar_diffusion_residual,
        )

        return artificial_diffusion_residual(
            self.state.U, nu_av,
            lambda phi, gamma: compute_scalar_diffusion_residual(phi, gamma, self.mesh, self.ops))

    def _get_turbulent_viscosity_field(self, mu_t_turb=None) -> Optional[np.ndarray]:
        """当前激活的湍流模型给出的动力涡粘度场 mu_t = rho * nu_t（委托给
        fr_solver_turbulence）；`mu_t_turb` 给定时原样返回（本步冻结值）。

        人工粘性**不**在这里：它不走应力/传导通道（见 `compute_viscous_residual`），
        壁面摩擦系数等后处理读这个方法时也不该把它算进去。粘性步长限制读的是
        `_get_cfl_viscosity_field`。
        """
        return (fr_solver_turbulence.get_turbulent_viscosity_field(self)
                if mu_t_turb is None else mu_t_turb)

    def _get_cfl_viscosity_field(self) -> Optional[np.ndarray]:
        """粘性步长限制用的附加动力粘度：湍流涡粘 + `rho * nu_av`（人工粘性）。

        TGV 教训（2026-08-29）：额外的扩散必须让步长控制看见，否则显式推进的
        粘性稳定条件被违反（P2 TGV 3 步发散）。人工扩散 `div(nu grad U)` 的谱半径
        与动力粘度 `rho*nu` 的粘性项同阶，所以按 `rho*nu` 计入。
        """
        mu_t = self._get_turbulent_viscosity_field()
        nu_av = self.compute_artificial_diffusivity_field()
        if nu_av is None:
            return mu_t
        mu_av = self.state.U[..., 0] * nu_av
        return mu_av if mu_t is None else mu_t + mu_av

    def _compute_gradients(self) -> np.ndarray:
        """
        计算守恒变量的梯度。

        Returns:
            grad_U: 梯度，形状 (n_cells, n_sps, n_vars, 3)
        """
        from autoflowcfd.core.fr_residual.viscous import compute_gradients
        return compute_gradients(self.state.U, self.ops, self.mesh)
    
