"""AutoFlowCFD V2.0 - 后处理：气动系数、VTK 导出、收敛历史

从 `src/autoflowcfd/api.py` 的 `AutoFlowCFDAPI` 拆出（2026-09-24，项目「单文件不超
500 行」规范）。mixin 是本仓库既有惯例（`_SolverGeometryMixin`、
`_GPUSolverInitMixin` 等），沿用它而不是另发明一套。

**只含方法，没有状态**：全部属性由 `AutoFlowCFDAPI` 的 `__init__` 建立，
这里通过 `self` 访问。
"""

from pathlib import Path
from typing import Dict, List, Optional
from loguru import logger


class _APIPostMixin:
    """后处理：气动系数、VTK 导出、收敛历史（都作用于 `self.solver` 的当前状态；`SolverResult` 不携带解场，
    2026-10-19 以前各方法保留的未使用 `result` 形参已删除）"""

    def calculate_coefficients(
        self,
        reference_area: Optional[float] = None,
        reference_length: float = 1.0,
        moment_center: Optional[List[float]] = None,
        groups: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """当前求解器状态的气动力/力矩系数（`postprocess/fr_coefficients.py`）。

        Args:
            reference_area: 参考面积 (m^2)；None 时沿来流方向的迎风投影自动估算（与 CLI 同一个函数
                `cli/solve/aero_coefficients.py::reference_area_for`）。2026-10-19 以前缺省为 1.0，系数量级随网格单位任意
            reference_length: 力矩参考长度 (m)
            moment_center: 力矩参考点 (3,)，缺省为原点
            groups: 参与积分的 WALL 组名，缺省为全部 WALL 组
        """
        if self.solver is None or not hasattr(self.solver, 'mesh'):
            raise ValueError("calculate_coefficients 需要先运行 run_steady/run_transient 或 resume_simulation")
        from autoflowcfd.cli.solve.aero_coefficients import reference_area_for
        from autoflowcfd.postprocess.fr_coefficients import compute_aerodynamic_coefficients_fr

        host = self.solver.host_view()
        area = reference_area_for(host.freestream, self.volume_mesh, reference_area)
        if area is None:
            raise ValueError("无法从体网格自动估算参考面积（体网格不带面网格数据），请给出 reference_area")
        return compute_aerodynamic_coefficients_fr(
            host, reference_area=area, reference_length=reference_length,
            moment_center=moment_center, groups=groups,
        ).to_dict()

    def export_vtk(self, filename: str, high_order: bool = False, fields: Optional[List[str]] = None) -> None:
        """导出当前求解器状态的 VTK 文件（低阶：.vtk/.vtu 按扩展名；高阶拉格朗日单元：只 .vtu）。

        Args:
            filename: 输出文件名
            high_order: True 时按 FR 解的分段多项式导出高阶拉格朗日单元（`postprocess/vtk_export_highorder`）
            fields: 导出的场（默认速度、压力；可含 k/omega/nu_tilde/nut/turbulence/q_criterion，高阶导出另可含 density）
        """
        from autoflowcfd.postprocess.vtk_export import VTKExporter
        from autoflowcfd.core.backend.base import SolutionVector

        if self.solver is None or self.volume_mesh is None:
            raise ValueError("export_vtk 需要先运行 run_steady/run_transient 或 resume_simulation")
        host = self.solver.host_view()

        if high_order:
            from autoflowcfd.postprocess.vtk_export_highorder import export_solver_highorder_vtk

            out_path = Path(filename)
            if out_path.suffix != '.vtu':
                out_path = out_path.with_suffix('.vtu')
            export_solver_highorder_vtk(host, out_path, fields=fields)
            logger.info(f"高阶 VTK 已导出：{out_path}")
            return

        from autoflowcfd.core.turbulence.output import turbulence_cell_means
        from autoflowcfd.fr.native_padding import cell_integration_weights, weighted_cell_average

        # 单元积分平均（求积权重 x det(J)），与 checkpoint 写出的守恒解单元平均同一个函数
        W = cell_integration_weights(host)
        U_cell_avg = weighted_cell_average(host.state.U, W)
        solution = SolutionVector(data=U_cell_avg, n_cells=U_cell_avg.shape[0], n_variables=U_cell_avg.shape[1])
        q_field = None
        if fields is not None and 'q_criterion' in fields:
            from autoflowcfd.postprocess.q_criterion import q_criterion_cell_field
            q_field = q_criterion_cell_field(host)
        exporter = VTKExporter(self.volume_mesh, solution,
                               turbulence=turbulence_cell_means(getattr(host, 'turb_model', None), W),
                               q_criterion=q_field)
        exporter.export(filename, fields=fields, format='xml' if filename.endswith('.vtu') else 'legacy')
        logger.info(f"VTK 已导出：{filename}")

    def get_convergence_history(self) -> Dict[str, list]:
        """获取收敛历史。

        真实 bug 修复（V2.0 专家组盲审发现，2026-08-27）：此前这里恒为
        硬编码占位符 `{"iterations": [], "residuals": []}`，不管
        run_steady/run_transient 是否已经成功跑完、收敛得多好，调用方
        拿到的永远是两个空列表，且没有任何警告提示这是未实现的占位符。
        现在读取 `self.solver.residual_history`（CPU FRSolver 与
        GPUFRSolver/MultiGPUDistributedSolver 都在各自的 solve 循环里
        逐迭代 append，同一个约定，见 fr_solver/solver.py::solve() 与
        core/utils/order_continuation.py::run_order_continuation）。

        Returns:
            包含 iterations（1-based 迭代序号）和 residuals 的字典；
            尚未运行过 run_steady/run_transient（self.solver 为 None）
            或求解器本身未记录历史时返回两个空列表。
        """
        residuals = getattr(self.solver, "residual_history", None) if self.solver is not None else None
        if not residuals:
            return {"iterations": [], "residuals": []}
        return {
            "iterations": list(range(1, len(residuals) + 1)),
            "residuals": list(residuals),
        }
