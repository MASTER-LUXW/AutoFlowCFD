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

    def compute_wall_distance_field(self, mesh_nodes: np.ndarray,
                                   wall_indices: np.ndarray,
                                   connectivity: Optional[np.ndarray] = None,
                                   use_eikonal: bool = False):
        """计算壁面距离场（用于 DDES/WMLES/SST），委托给 fr_solver_turbulence。

        Args:
            mesh_nodes: 全部网格节点坐标
            wall_indices: WALL 边界节点索引
            connectivity: 节点邻接表，use_eikonal=True 时必须提供 - 见
                fr_solver_turbulence.compute_wall_distance_field 自己的文档
            use_eikonal: 是否用 Eikonal 方程（而不是纯欧氏 KD-Tree）求解
        """
        fr_solver_turbulence.compute_wall_distance_field(
            self, mesh_nodes, wall_indices, connectivity=connectivity, use_eikonal=use_eikonal
        )

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

    def compute_viscous_residual(self, mu_t_turb=None):
        """
        计算粘性残差 (S-03)。

        Args:
            mu_t_turb: 本步冻结的湍流动力涡粘 `rho*nu_t`（`step()` 在湍流更新
                之后按步前状态求一次，本步全部残差求值共用）。None 时按当前
                状态求。冻结是全部后端同一个算子分裂约定（单机 GPU、多 GPU、
                CPU 分布式都整步冻结）；2026-09-25 以前单机 CPU 在每次残差
                求值里按**试探态**的 rho 重算，与其余后端每步差 O(dt)，隐式步
                下直接让 Jacobian 不同（分布式 n_ranks=1 对照里块 Jacobi 差 1e-3）。

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
                之后才计算）
        """
        mu_t_field = self._get_turbulent_viscosity_field(mu_t_turb)
        res = compute_viscous_residual_ldg(
            self.state.U, self.state.Q, self.ops, self.mesh,
            mu=self.mu_molecular,
            mu_t_field=mu_t_field,
            boundary_ghost_provider=self.boundary_ghost_provider,
        )

        if self.wmles_model is not None:
            wall_stress_correction = solver_helpers.compute_wmles_wall_stress_correction(self)
            if wall_stress_correction is not None:
                res = res + wall_stress_correction[..., : res.shape[-1]]

        # 人工粘性的**质量扩散通道**（2026-09-14 补齐）。
        #
        # 此前 `artificial_viscosity.py` 模块文档里如实记录了一条范围
        # 限制、并把它称作"许多实际 DG/FR 实现采用的简化"：Persson &
        # Peraire (2006) 原方法对**全部**守恒变量（含连续性方程）叠加
        # 人工扩散，而本实现只把 epsilon 叠进 `mu_t_field`，于是它只能
        # 通过动量/能量方程既有的粘性应力/热传导通道起作用，密度本身
        # 完全不被扩散（`viscous_physical_flux` 的质量分量 G[...,0]
        # 恒为 0）。用户明确指出本项目不接受简化，这里补上缺的那一项。
        #
        # 实现方式：不改粘性热路径。AV 默认关闭，没有理由为它给所有
        # 运行的 `viscous_physical_flux_batch` 增加参数与分支；而
        # `div(eps*grad(rho))` 正是一个标量扩散算子，直接复用湍流输运
        # 已经验证过的 BR1 面耦合标量扩散装配
        # （`turbulence/transport.py::compute_scalar_diffusion_residual`，
        # 它返回的就是 +div(Gamma*grad(phi))，与这里 dU/dt 的符号约定
        # 一致）。AV 关闭时这段完全不执行，零开销。
        #
        # 守恒性与自由流场保持性：`div(eps*grad(rho))` 是散度形式，
        # 因此严格守恒；均匀流场下 grad(rho)=0，这一项恒为 0，不破坏
        # 自由流场保持性（已用测试钉住，见
        # tests/unit/test_artificial_viscosity_mass_diffusion.py）。
        if getattr(self, "artificial_viscosity_enabled", False):
            res = res + self._artificial_mass_diffusion_residual()

        return res

    def _artificial_mass_diffusion_residual(self) -> np.ndarray:
        """Persson-Peraire 人工粘性作用在连续性方程上的那一项。

        返回形状与粘性残差相同的数组，只有质量分量（索引 0）非零，
        其值为 `+div(epsilon * grad(rho))`（dU/dt 约定）。
        完整动机见 `compute_viscous_residual` 里的调用点注释。
        """
        from autoflowcfd.core.fr_operators.artificial_viscosity import (
            compute_persson_peraire_artificial_viscosity,
        )
        from autoflowcfd.core.turbulence.transport import (
            compute_scalar_diffusion_residual,
        )

        epsilon_av = compute_persson_peraire_artificial_viscosity(
            self, alpha_av=self.artificial_viscosity_alpha
        )
        rho = self.state.U[..., 0]
        d_rho_dt = compute_scalar_diffusion_residual(
            np.ascontiguousarray(rho), np.ascontiguousarray(epsilon_av),
            self.mesh, self.ops,
        )
        out = np.zeros_like(self.state.U[..., : self.state.U.shape[-1]])
        out[..., 0] = d_rho_dt
        return out

    def _get_turbulent_viscosity_field(self, mu_t_turb=None) -> Optional[np.ndarray]:
        """汇总当前激活的湍流模型给出的动力涡粘度场 mu_t = rho * nu_t（委托给 fr_solver_turbulence），
        再叠加 Persson-Peraire 人工粘性（若启用）。

        真实 bug 修复（2026-08-29，TGV 真实复现）：人工粘性最初被直接
        加进 `compute_viscous_residual` 里临时拼出的 `mu_t_field`，
        `_compute_local_time_step`（cfl.py）单独调用这个方法算粘性
        CFL 步长时完全看不到这份额外粘度——时间步长仍按"只有分子
        粘度+湍流涡粘"来估算，而实际粘性残差里已经叠加了一份可能
        大出物理粘度一个数量级的人工扩散，显式格式的粘性稳定性条件
        `dt<=C*h^2/mu_eff` 被违反，真实复现：TGV（P2，Re=20 低雷诺数
        算例，物理 mu 已经刻意调得比空气分子粘度大三个数量级）3 步内
        发散。必须让 CFL 计算与粘性残差看到*同一个* `mu_t_field`——
        统一在这个唯一的读取入口叠加，而不是分别在两个消费点各自
        处理（同一类问题见项目记忆 hardcoded_molecular_viscosity_
        mismatch/low_mach_cfl_ausm_inconsistency：任何"物理量在多个
        消费点独立计算/获取"的模式都有两处失去同步的风险）。
        """
        mu_t_field = (fr_solver_turbulence.get_turbulent_viscosity_field(self)
                      if mu_t_turb is None else mu_t_turb)
        if getattr(self, "artificial_viscosity_enabled", False):
            from autoflowcfd.core.fr_operators.artificial_viscosity import (
                compute_persson_peraire_artificial_viscosity,
            )

            epsilon_av = compute_persson_peraire_artificial_viscosity(
                self, alpha_av=self.artificial_viscosity_alpha
            )
            mu_t_field = epsilon_av if mu_t_field is None else mu_t_field + epsilon_av
        return mu_t_field

    def _compute_gradients(self) -> np.ndarray:
        """
        计算守恒变量的梯度。

        Returns:
            grad_U: 梯度，形状 (n_cells, n_sps, n_vars, 3)
        """
        from autoflowcfd.core.fr_residual.viscous import compute_gradients
        return compute_gradients(self.state.U, self.ops, self.mesh)
    
