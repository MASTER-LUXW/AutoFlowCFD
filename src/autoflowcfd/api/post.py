"""AutoFlowCFD V2.0 - 后处理：气动系数、VTK 导出、收敛历史

从 `src/autoflowcfd/api.py` 的 `AutoFlowCFDAPI` 拆出（2026-09-24，项目「单文件不超
500 行」规范）。mixin 是本仓库既有惯例（`_SolverGeometryMixin`、
`_GPUSolverInitMixin` 等），沿用它而不是另发明一套。

**只含方法，没有状态**：全部属性由 `AutoFlowCFDAPI` 的 `__init__` 建立，
这里通过 `self` 访问。
"""

from pathlib import Path
from typing import Dict, Any
from loguru import logger


class _APIPostMixin:
    """后处理：气动系数、VTK 导出、收敛历史"""

    def calculate_coefficients(
        self,
        result: Any,
        reference_area: float = 1.0,
        reference_length: float = 1.0,
        density: float = 1.225,
        velocity: float = 33.33
    ) -> Dict[str, float]:
        """计算气动力系数。
        
        Args:
            result: 求解器结果
            reference_area: 参考面积
            reference_length: 参考长度
            density: 流体密度
            velocity: 参考速度
            
        Returns:
            气动力系数字典（使用大写键名Cd, Cl等）
        """
        # FR 原生积分路径（唯一真实积分实现，见 api_postprocess.py 同名委托函数文档）
        if self.solver is not None and hasattr(self.solver, 'mesh'):
            from autoflowcfd.postprocess.fr_coefficients import (
                compute_aerodynamic_coefficients_fr,
            )
            coeffs = compute_aerodynamic_coefficients_fr(
                self.solver,
                reference_area=reference_area,
                reference_length=reference_length,
            )
            return coeffs.to_dict()

        # 无求解器：返回诚实零值并警告，不回退到伪积分实现（旧版 V1
        # CoefficientCalculator 依赖不存在的 get_face_data()，已移除）
        logger.warning(
            "calculate_coefficients: 没有可用的 FR 求解器（需先调用 run_steady/"
            "run_transient 或 resume_simulation），返回零系数。"
        )
        return {
            'Cd': 0.0, 'Cl': 0.0, 'Cm': 0.0,
            'Cs': 0.0, 'Cy': 0.0, 'Cr': 0.0,
        }

    def export_vtk(self, result: Any = None, filename: str = None, high_order: bool = False) -> None:
        """导出 VTK 可视化文件。

        使用 VTKExporter 将流场数据导出为 VTK 格式，支持 legacy .vtk
        和 XML .vtu 两种格式（根据文件扩展名自动选择）。

        Args:
            result: 未使用，仅为向后兼容签名保留——真正的解场从
                self.solver.state.U 读取（见下方说明），不是从
                SolverResult 对象（它只有 converged/iterations/
                final_residual 三个字段，从不携带解场，见
                core/fr_solver/state.py）。
            filename: 输出文件名（.vtk 或 .vtu）
            high_order: True 时改走 postprocess/vtk_export_highorder.py
                （#5，2026-08-28 新增）：按 FR 解真实的分段多项式（而不是
                `U.mean(axis=1)` 拍扁的单元中心平均值）导出成 VTK
                VTK_LAGRANGE_TETRAHEDRON/WEDGE 高阶单元，能在 ParaView 里
                看到单元内部真实的多项式分布。只支持 order<=2（见该模块
                文档），且只输出 .vtu（VTK legacy 格式不支持任意阶
                Lagrange 单元）；filename 若没有 .vtu 后缀会被自动改写。

        此前这里用 `result.solution`（SolverResult 根本没有这个字段，
        `hasattr` 检查恒为 False，必然走进"抛异常"分支）和
        `self.grid_data`（run_steady/run_transient 从不写入的表面网格，
        即便写了，单元数也和体网格解场对不上）构造 VTKExporter——两个
        参数都是错的，从未被真正跑通过（V2.0 专家组评审逐行核实）。
        改为镜像 CLI `post export-vtk`（cli/post/export_commands.py）
        真正验证过的用法：VTKExporter 的 `grid_data` 参数只是鸭子类型
        地读取 `.metadata.node_count`/`.cell_count`，`self.volume_mesh`
        （generate_volume_mesh 的输出，run_steady/run_transient 求解的
        就是它）满足这个接口；解场用 `self.solver.state.U.mean(axis=1)`
        拍扁成单元中心平均值（与 CheckpointManager.save 写 checkpoint
        时的约定一致）包装成 SolutionVector。
        """
        from autoflowcfd.postprocess.vtk_export import VTKExporter
        from autoflowcfd.core.backend.base import SolutionVector

        if self.solver is None or self.volume_mesh is None:
            raise ValueError(
                "export_vtk 需要先成功运行 run_steady/run_transient "
                "（需要 self.solver 和 self.volume_mesh 均已设置）。"
            )
        if filename is None:
            raise ValueError("export_vtk 需要提供 filename。")

        if high_order:
            from autoflowcfd.postprocess.vtk_export_highorder import export_highorder_vtk

            out_path = Path(filename)
            if out_path.suffix != '.vtu':
                out_path = out_path.with_suffix('.vtu')
            export_highorder_vtk(self.solver.mesh, self.solver.state.U, out_path)
            logger.info(f"High-order VTK exported: {out_path}")
            return

        U_cell_avg = self.solver.state.U.mean(axis=1)  # (n_cells, n_vars)
        solution = SolutionVector(
            data=U_cell_avg, n_cells=U_cell_avg.shape[0], n_variables=U_cell_avg.shape[1],
        )

        # 湍流涡粘度（用于精确的 nut 导出），有则给，没有就让 VTKExporter
        # 自己退化成简化估计（它自身文档已说明这个 fallback）。
        mu_t = None
        get_mu_t = getattr(self.solver, '_get_turbulent_viscosity_field', None)
        if callable(get_mu_t):
            mu_t_field = get_mu_t()
            if mu_t_field is not None:
                mu_t = mu_t_field.mean(axis=1)

        exporter = VTKExporter(self.volume_mesh, solution, mu_t=mu_t)

        # 根据扩展名选择格式
        fmt = 'xml' if filename.endswith('.vtu') else 'legacy'
        exporter.export(filename, format=fmt)
        logger.info(f"VTK exported: {filename}")

    def get_convergence_history(self, result: Any = None) -> Dict[str, list]:
        """获取收敛历史。

        真实 bug 修复（V2.0 专家组盲审发现，2026-08-27）：此前这里恒为
        硬编码占位符 `{"iterations": [], "residuals": []}`，不管
        run_steady/run_transient 是否已经成功跑完、收敛得多好，调用方
        拿到的永远是两个空列表，且没有任何警告提示这是未实现的占位符。
        现在读取 `self.solver.residual_history`（CPU FRSolver 与
        GPUFRSolver/MultiGPUDistributedSolver 都在各自的 solve 循环里
        逐迭代 append，同一个约定，见 fr_solver/solver.py::solve() 与
        core/utils/order_continuation.py::run_order_continuation）。

        Args:
            result: 未使用，保留以兼容旧调用签名

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
