"""求解器侧探针：每个时间步在给定位置求 FR 解多项式的值并写成时间序列（`recorder.py` 模块文档）。"""

from autoflowcfd.core.probes.locate import ProbeLocations, locate_probes
from autoflowcfd.core.probes.recorder import (
    PROBE_FILE,
    PROBE_LAYOUT_FILE,
    PROBE_POINTS_KEY,
    RECORDED,
    ProbeRecorder,
    attach_probes,
    attach_probes_from_checkpoint,
    read_probe_series,
)

__all__ = [
    "PROBE_FILE",
    "PROBE_LAYOUT_FILE",
    "PROBE_POINTS_KEY",
    "RECORDED",
    "ProbeLocations",
    "ProbeRecorder",
    "attach_probes",
    "attach_probes_from_checkpoint",
    "locate_probes",
    "read_probe_series",
]
