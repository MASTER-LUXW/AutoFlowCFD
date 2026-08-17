"""网格质量报告数据结构。

MeshQualityReport 是 MeshQualityValidator（quality_validator.py）的输出
类型——拆成独立模块，避免 quality_validator.py（检查逻辑的实现）还要
内联携带这么大的一个 dataclass/摘要格式化器。GridValidator
（validator.py）是另一个更简单、返回值是普通 dict 的面网格检查器，与
这个类无关。
"""

import numpy as np
from typing import List, Optional
from dataclasses import dataclass, field


@dataclass
class MeshQualityReport:
    """全面的网格质量报告。

    Attributes:
        n_cells: 单元总数
        n_nodes: 节点总数
        passed: 整体质量检查结果
        negative_volumes: 负体积单元数
        min_volume: 最小单元体积
        max_volume: 最大单元体积
        mean_volume: 平均单元体积
        std_volume: 体积标准差
        volume_ratio: 全局最大/最小体积比——仅供参考（见模块文档
            字符串——BL 网格合法的近壁到远场范围会主导这个值并
            淹没任何真实的局部缺陷信号）；adjacent_volume_ratio_max
            才是真正的关卡指标
        min_aspect_ratio: 最小长宽比
        max_aspect_ratio: 最大长宽比（全网格）
        mean_aspect_ratio: 平均长宽比（全网格）
        bl_max_aspect_ratio: 仅 BL 区域单元的最大长宽比
            （除非 validate() 提供了 bl_cell_mask，否则为 None）
        core_max_aspect_ratio: 核心区域单元的最大长宽比
        max_skewness: 最大半径比偏斜度（0=正四面体, 1=碎片）
        mean_skewness: 平均偏斜度
        orthogonality_max: 所有内部面上面法向与质心连线夹角
            的最差值（度），0=理想
        orthogonality_mean: 上述的均值
        adjacent_volume_ratio_max: 最差的面邻居体积比
            （共享面的两个单元间 max(V)/min(V)）——
            真正控制 Green-Gauss 条件数的指标
        adjacent_volume_ratio_mean: 上述的均值
        n_overlapping_cells: 至少有一个面与不同的、不相邻单元的
            面物理重叠的单元数（见 mesh_gen/../validation/
            mesh_overlap_check.py）——与负/退化单元不同的缺陷类别：
            涉及的每个单元自身都可以有完美的正体积和合理形状，
            问题是它们占据了重叠的物理空间。除非 validate() 传入
            check_overlap=True 否则为 0（这是此处唯一一个成本随
            局部网格密度缩放而非纯粹按单元数的检查，因此是可选
            而非始终静默跳过的）。
        n_close_cell_pairs: 面尚未重叠但距离小于局部缩放阈值
            的单元对——仅供参考，本身不是缺陷（这正是
            mesh_tetgen_core.compute_local_thickness_limit 在生成时
            试图防止 BL 前沿交叉的方式；这是对那个启发式方法
            未能完全防止的事后可见性检查）。
        overlap_min_gap: 接近对中最小的非重叠面到面距离，
            米（若未找到/未检查则为 None）
        warnings: 质量警告列表
        recommendations: 改进建议列表
        repair_stages_applied: 生成此报告前采取的修复操作
            的人类可读日志（见 mesh_gen/mesh_repair.py）——
            对不带修复的裸 validate() 调用为空。
        initial_report: 任何修复前的报告，用于 summary() 中的
            前后对比——若此报告本身就是修复前基线或未运行修复
            则为 None。
    """
    n_cells: int = 0
    n_nodes: int = 0
    passed: bool = True

    # 体积指标
    negative_volumes: int = 0
    min_volume: float = float('inf')
    max_volume: float = 0.0
    mean_volume: float = 0.0
    std_volume: float = 0.0
    volume_ratio: float = 0.0

    # 长宽比指标
    min_aspect_ratio: float = float('inf')
    max_aspect_ratio: float = 0.0
    mean_aspect_ratio: float = 0.0
    bl_max_aspect_ratio: Optional[float] = None
    bl_mean_aspect_ratio: Optional[float] = None
    core_max_aspect_ratio: Optional[float] = None
    core_mean_aspect_ratio: Optional[float] = None

    # 偏斜度指标（基于半径比）
    max_skewness: float = 0.0
    mean_skewness: float = 0.0

    # 正交性指标
    orthogonality_max: float = 0.0
    orthogonality_mean: float = 0.0

    # 相邻单元（面邻居）体积比
    adjacent_volume_ratio_max: float = 0.0
    adjacent_volume_ratio_mean: float = 0.0

    # 单元重叠/接近接触面指标（见 mesh_overlap_check.py）
    n_overlapping_cells: int = 0
    n_close_cell_pairs: int = 0
    overlap_min_gap: Optional[float] = None
    # (n_overlapping_cells,) int64——供需要知道具体是哪些单元
    # （而非仅数量）的调用方（例如网格修复循环）使用；避免
    # 为恢复此信息而重新运行重叠检查。
    overlapping_cell_ids: Optional[np.ndarray] = None

    # 定性反馈
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    # 修复循环簿记
    repair_stages_applied: List[str] = field(default_factory=list)
    initial_report: Optional['MeshQualityReport'] = None

    def summary(self) -> str:
        """生成人类可读的质量摘要。"""
        lines = [
            "=" * 70,
            "MESH QUALITY REPORT",
            "=" * 70,
            f"Overall Status: {'PASSED ✓' if self.passed else 'FAILED ✗'}",
            "",
            "Grid Size:",
            f"  Cells: {self.n_cells:,}",
            f"  Nodes: {self.n_nodes:,}",
            "",
            "Volume Quality:",
            f"  Negative volumes: {self.negative_volumes}",
            f"  Volume range: [{self.min_volume:.6e}, {self.max_volume:.6e}]",
            f"  Mean ± Std: {self.mean_volume:.6e} ± {self.std_volume:.6e}",
            f"  Global max/min ratio: {self.volume_ratio:.2e} (informational - BL grading, not a defect by itself)",
            f"  Adjacent-cell (face-neighbour) ratio: max={self.adjacent_volume_ratio_max:.2f}, mean={self.adjacent_volume_ratio_mean:.2f}",
            "",
            "Aspect Ratio:",
            f"  Overall: [{self.min_aspect_ratio:.3f}, {self.max_aspect_ratio:.3f}], mean={self.mean_aspect_ratio:.3f}",
        ]
        if self.bl_max_aspect_ratio is not None:
            lines.append(f"  BL region:   max={self.bl_max_aspect_ratio:.3f}, mean={self.bl_mean_aspect_ratio:.3f}")
        if self.core_max_aspect_ratio is not None:
            lines.append(f"  Core region: max={self.core_max_aspect_ratio:.3f}, mean={self.core_mean_aspect_ratio:.3f}")
        lines += [
            "",
            "Skewness (radius-ratio, 0=regular tet .. 1=sliver):",
            f"  Max: {self.max_skewness:.4f}",
            f"  Mean: {self.mean_skewness:.4f}",
            "",
            "Orthogonality (face-normal vs. centroid-connector angle, 0deg=ideal):",
            f"  Max: {self.orthogonality_max:.2f} deg",
            f"  Mean: {self.orthogonality_mean:.2f} deg",
        ]

        if self.n_overlapping_cells > 0 or self.n_close_cell_pairs > 0:
            lines += [
                "",
                "Cell Overlap / Proximity:",
                f"  Overlapping cells: {self.n_overlapping_cells}",
                f"  Near-touching cell pairs: {self.n_close_cell_pairs}"
                + (f" (min gap {self.overlap_min_gap:.3e} m)" if self.overlap_min_gap is not None else ""),
            ]

        if self.initial_report is not None:
            ir = self.initial_report
            lines += [
                "",
                "Before/After Repair Comparison:",
                f"  Status:                {'PASSED' if ir.passed else 'FAILED'} -> {'PASSED' if self.passed else 'FAILED'}",
                f"  Max skewness:           {ir.max_skewness:.4f} -> {self.max_skewness:.4f}",
                f"  Max non-orthogonality:  {ir.orthogonality_max:.2f} deg -> {self.orthogonality_max:.2f} deg",
                f"  Max adjacent vol ratio: {ir.adjacent_volume_ratio_max:.2f} -> {self.adjacent_volume_ratio_max:.2f}",
                f"  Negative volumes:       {ir.negative_volumes} -> {self.negative_volumes}",
                f"  Overlapping cells:      {ir.n_overlapping_cells} -> {self.n_overlapping_cells}",
            ]

        if self.repair_stages_applied:
            lines.append("")
            lines.append("Repair Actions Applied:")
            for i, action in enumerate(self.repair_stages_applied, 1):
                lines.append(f"  {i}. {action}")

        if self.warnings:
            lines.append("")
            lines.append("Warnings:")
            for i, warning in enumerate(self.warnings, 1):
                lines.append(f"  {i}. {warning}")

        if self.recommendations:
            lines.append("")
            lines.append("Recommendations:")
            for i, rec in enumerate(self.recommendations, 1):
                lines.append(f"  {i}. {rec}")

        lines.append("=" * 70)

        return "\n".join(lines)


