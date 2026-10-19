"""求解前的网格质量门与高阶网格构造（`solve steady/transient` 命令与 Python API 共用的唯一实现）。

2026-10-19 以前只有 CLI 的 `cli/solve/mesh_loader.py` 跑质量门；`AutoFlowCFDAPI.run_steady/run_transient` 直接
`HighOrderMesh.load_from_volume_mesh`，质量门 FAILED 的网格照样提交计算，API 另有一份只看体积标准差的"玩具"校验。
"""

from autoflowcfd.grid.curved_mapping.curved_mapping import MeshDistortionError
from autoflowcfd.grid.high_order.high_order_mesh import HighOrderMesh


class MeshQualityGateError(ValueError):
    """体网格未通过求解前质量门，或构造高阶几何时检测到畸变单元。"""


def enforce_solve_quality_gate(volume_data, report=None, skip: bool = False) -> None:
    """未通过 `MeshQualityValidator` 的体网格拒绝求解（`report` 为已有的质量报告时复用，不重算）。

    Raises:
        MeshQualityGateError: 质量门未通过且 `skip` 为 False
    """
    if skip:
        print("⚠️  已跳过求解前的网格质量门检查")
        return
    if report is None:
        from autoflowcfd.grid.validation.quality_validator import MeshQualityValidator

        print("Validating volume mesh quality before solving...")
        report = MeshQualityValidator().validate_volume_mesh(volume_data)
    if not report.passed:
        raise MeshQualityGateError(
            f"网格质量门检查未通过，拒绝求解（避免残差在退化单元处异常放大）：\n{report.summary()}\n"
            f"明确了解风险、只是临时诊断时可以跳过这道检查（命令行 --skip-quality-check / API skip_quality_check=True）。")
    print("✅ Volume mesh quality gate passed")


def high_order_mesh_for_solver(volume_data, order: int, report=None, skip_quality_check: bool = False) -> HighOrderMesh:
    """质量门 + 高阶网格构造。

    质量门用启发式指标筛查，不保证拦住所有退化/反转单元（尤其跳过质量门时）；曲边映射构造高阶几何时做严格的
    Jacobian 符号检查，这里把它的 `MeshDistortionError` 换成同一个异常类型、给出可读诊断。

    Raises:
        MeshQualityGateError: 质量门未通过，或检测到畸变单元
    """
    enforce_solve_quality_gate(volume_data, report=report, skip=skip_quality_check)
    mesh = HighOrderMesh(order=order)
    try:
        mesh.load_from_volume_mesh(volume_data)
    except MeshDistortionError as e:
        raise MeshQualityGateError(
            f"网格在构造高阶曲边几何时检测到畸变单元，拒绝求解：{e}\n"
            f"请在网格生成/修复阶段处理该单元（'autoflowcfd grid check'/'repair'），不要跳过质量门强行求解。") from e
    return mesh


__all__ = ["MeshQualityGateError", "enforce_solve_quality_gate", "high_order_mesh_for_solver"]
