"""mesh_overlap_check.py 的结果数据类 OverlapProximityReport。

从 mesh_overlap_check.py 中拆分出来（原文件超过 400 行的项目约定上限）：
这个 dataclass 只是 check_face_overlap_and_proximity 的返回值容器，本身
不依赖该函数体的任何内部状态，是最自然的独立单元——原样搬移，字段、
方法、文档字符串均未改动。mesh_overlap_check.py 里通过
`从 .mesh_overlap_report 导入 OverlapProximityReport` 重新导出，
任何 `从 autoflowcfd.grid.验证.mesh_overlap_check 导入
OverlapProximityReport` 的既有导入路径不受影响。
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class OverlapProximityReport:
    """check_face_overlap_and_proximity 的结果。

    Attributes:
        n_faces_checked: 考虑的独立三角形面总数
        n_candidate_pairs: 通过宽相位 + 节点共享过滤后实际接受
            精确几何测试的面片对——仅供参考，用于判断宽相位的选择性
        n_overlapping_pairs: 真实相交的面片对
        n_close_pairs: 不相交但距离小于 `proximity_threshold_used`
            的面片对（每对局部缩放的阈值，非单一全局常量——
            见检查函数自身的文档字符串）
        overlapping_cell_ids: 拥有真实重叠面的唯一单元索引
        close_cell_ids: 拥有接近接触（但尚未重叠）面片的唯一单元索引
        min_gap_found: 所有接近对中最小的非重叠面到面距离
            （若无则为 None）
        overlap_examples: 最多 `max_examples` 个 (cell_a, cell_b) 对，
            供人类可读报告——在有许多重叠的网格上不是穷举
        close_examples: 最多 `max_examples` 个 (cell_a, cell_b, distance) 元组
        elapsed_seconds: 此检查的挂钟时间——包含它是因为与此项目
            其他 O(n) 向量化质量检查不同，此检查的成本随局部网格
            密度缩放，而非纯粹按单元数（见模块文档字符串）
    """
    n_faces_checked: int = 0
    n_candidate_pairs: int = 0
    n_overlapping_pairs: int = 0
    n_close_pairs: int = 0
    overlapping_cell_ids: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    close_cell_ids: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    min_gap_found: Optional[float] = None
    overlap_examples: List[Tuple[int, int]] = field(default_factory=list)
    close_examples: List[Tuple[int, int, float]] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    @property
    def has_overlaps(self) -> bool:
        return self.n_overlapping_pairs > 0

    def bad_cell_mask(self, n_cells: int) -> np.ndarray:
        """布尔掩码，真实重叠涉及的单元为 True
        （不包括接近但未重叠的——见 mesh_repair.py 对此的使用：
        只有实际重叠才严重到需要修复；
        "接近"仅报告可见性，本身不是缺陷）。"""
        mask = np.zeros(n_cells, dtype=bool)
        if len(self.overlapping_cell_ids):
            mask[self.overlapping_cell_ids] = True
        return mask

    def summary(self) -> str:
        lines = [
            "Cell Overlap / Proximity Check:",
            f"  Faces checked: {self.n_faces_checked:,}  "
            f"Candidate pairs tested: {self.n_candidate_pairs:,}  "
            f"({self.elapsed_seconds:.2f}s)",
            f"  Overlapping face pairs: {self.n_overlapping_pairs} "
            f"({len(self.overlapping_cell_ids)} cells implicated)",
            f"  Near-touching (not yet overlapping) face pairs: {self.n_close_pairs} "
            f"({len(self.close_cell_ids)} cells implicated)"
            + (f", min gap {self.min_gap_found:.3e} m" if self.min_gap_found is not None else ""),
        ]
        if self.overlap_examples:
            preview = ", ".join(f"({a},{b})" for a, b in self.overlap_examples[:10])
            more = f" (+{len(self.overlap_examples) - 10} more)" if len(self.overlap_examples) > 10 else ""
            lines.append(f"  Overlapping cell pairs (owner ids): {preview}{more}")
        if self.close_examples:
            preview = ", ".join(f"({a},{b}, {d:.2e}m)" for a, b, d in self.close_examples[:10])
            more = f" (+{len(self.close_examples) - 10} more)" if len(self.close_examples) > 10 else ""
            lines.append(f"  Near-touching cell pairs: {preview}{more}")
        return "\n".join(lines)
