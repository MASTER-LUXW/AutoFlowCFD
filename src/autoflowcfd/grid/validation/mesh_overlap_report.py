"""mesh_overlap_check.py 的结果数据类 OverlapProximityReport。

从 mesh_overlap_check.py 中拆分出来（原文件超过 400 行的项目约定上限）：
这个 dataclass 只是 check_face_overlap_and_proximity 的返回值容器，本身
不依赖该函数体的任何内部状态，是最自然的独立单元——原样搬移，字段、
方法、文档字符串均未改动。mesh_overlap_check.py 里通过
`from .mesh_overlap_report import OverlapProximityReport` 重新导出，
任何 `from autoflowcfd.grid.validation.mesh_overlap_check import
OverlapProximityReport` 的既有导入路径不受影响。
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class OverlapProximityReport:
    """Result of check_face_overlap_and_proximity.

    Attributes:
        n_faces_checked: Total distinct triangular faces considered
        n_candidate_pairs: Face pairs that survived broad-phase + node-
            sharing filtering and were actually given the exact geometric
            test - informational, for judging how selective the broad
            phase was
        n_overlapping_pairs: Face pairs that genuinely intersect
        n_close_pairs: Face pairs that don't intersect but are closer than
            `proximity_threshold_used` (a per-pair, locally-scaled
            distance, not a single global constant - see the check
            function's own docstring)
        overlapping_cell_ids: Unique cell indices that own at least one
            face involved in a genuine overlap
        close_cell_ids: Unique cell indices that own at least one face
            involved in a near-touching (but not yet overlapping) pair
        min_gap_found: Smallest non-overlapping face-to-face distance
            found among all close pairs (None if none found)
        overlap_examples: Up to `max_examples` (cell_a, cell_b) pairs, for
            a human-readable report - not exhaustive on a mesh with many
            overlaps
        close_examples: Up to `max_examples` (cell_a, cell_b, distance)
            tuples
        elapsed_seconds: Wall-clock time this check took - included
            because, unlike this project's other O(n) vectorized quality
            checks, this one's cost scales with local mesh density, not
            purely cell count (see module docstring)
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
        """Boolean mask, True for any cell implicated in a genuine overlap
        (NOT close-but-not-overlapping - see mesh_repair.py's use of this:
        only an actual overlap is bad enough to warrant repair action;
        "close" is reported for visibility but is not itself a defect)."""
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
