"""AutoFlowCFD V2.0 - `FRSolver` 的残差、湍流与边界委托方法（mixin，只含方法）。

从 `core/fr_solver/solver.py` 拆出（2026-09-25）。
"""

from typing import Any, Dict, Optional

import numpy as np

from autoflowcfd.core.fr_residual.inviscid import compute_inviscid_residual_fr, mean_flow_primitives
from autoflowcfd.core.fr_residual.viscous_flux import compute_viscous_residual_fr
from autoflowcfd.core.fr_residual.viscous_flux.constants import PRANDTL

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
        """湍流场的一次更新（委托给 fr_solver_turbulence.compute_turbulence_source，参数含义见该函数）。"""
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

    def compute_mean_flow_residual(self, mu_t_turb=None, nu_av=None):
        """平均流残差 `R = -(无粘 + 粘性)`（TimeIntegrator 约定 `dU/dt = -R`），未经低马赫预处理；参数见
        `compute_viscous_residual`。显式推进（`fr_solver/step.py`）与隐式耦合步
        （`fr_solver/turbulence/implicit.py`）共用这一处组装。

        两项共用同一份原始变量与边界幽灵态（`mean_flow_primitives`）。原地累加与取负：粘性结果是新数组
        （B-12 P2 OOM 修复第⑤级，2026-08-26：`total = inv + visc` 再 `-total` 会多分配两个全场数组）。
        """
        self._refresh_primitives()
        from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry

        primitives = mean_flow_primitives(self.state.U, get_flat_face_geometry(self.mesh, self.ops),
                                          self.boundary_ghost_provider)
        res = self.compute_viscous_residual(mu_t_turb=mu_t_turb, nu_av=nu_av, primitives=primitives)
        res += self._inviscid_residual(primitives)
        res *= -1
        return res

    def _refresh_primitives(self) -> None:
        """`state.Q` 与当前 `state.U` 同步（低马赫预处理与湍流读它），`U` 转连续（残差核要连续输入）。"""
        self.state._update_primitives()
        if not self.state.U.flags['C_CONTIGUOUS']:
            self.state.U = np.ascontiguousarray(self.state.U)

    def _inviscid_residual(self, primitives=None):
        return compute_inviscid_residual_fr(
            self.state.U, self.mesh, self.ops,
            boundary_ghost_provider=self.boundary_ghost_provider,
            mach_ref=self.freestream["mach_ref"],
            primitives=primitives,
        )

    def compute_inviscid_residual(self):
        """
        计算无粘残差 (S-02/S-04)。

        原生基（棱柱 / 四面体）上的 FR 离散：体积项为逆变通量散度（度量一致，满足自由流保持 / 离散 GCL），
        界面项为真实单元-面连接关系上的 AUSM+up 黎曼通量与 DG 提升，边界面经 boundary_ghost_provider 构造幽灵态。
        见 `core/fr_residual/inviscid.py` 模块文档。
        """
        self._refresh_primitives()
        return self._inviscid_residual()

    def compute_viscous_residual(self, mu_t_turb=None, nu_av=None, primitives=None):
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
            primitives: 与无粘项共用的原始变量与边界幽灵态（`compute_mean_flow_residual`）；None 时现算。

        真实的 BR1 面耦合粘性离散（core/fr_residual/viscous_flux/），并把湍流模型
        算出的涡粘系数真正耦合进应力张量/热传导（T-01/T-04/T-06 修复：
        此前调用处从不传湍流粘度，粘性通量永远只用分子粘度 1.8e-5，
        SST/DDES/WALE 算出的 nu_t 场只在自身模型内部自用，从未进入
        动量/能量方程的扩散项）。

        Returns:
            viscous_res: 粘性残差（WMLES 激活时已叠加壁面剪应力修正，`core/turbulence/wall_model/`；
                必须在残差组装里施加才影响本步的解）；启用人工粘性时已叠加 `div(nu grad U)`
        """
        res = compute_viscous_residual_fr(
            self.state.U, self.mesh, self.ops, self.mu_molecular, PRANDTL,
            mu_t_field=self._get_turbulent_viscosity_field(mu_t_turb),
            boundary_ghost_provider=self.boundary_ghost_provider,
            primitives=primitives,
        )

        if self.wall_model is not None:
            from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry

            wall_stress_correction = self.wall_model.residual_correction(
                np, self.mesh, get_flat_face_geometry(self.mesh, self.ops), self.boundary_ghost_provider,
                self.state.U, self.mu_molecular)
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
            lambda phi, gamma: compute_scalar_diffusion_residual(phi, gamma, self.mesh, self.ops, gamma_band=0.0))

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

    
