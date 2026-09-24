"""
AutoFlowCFD V2.0 - GPU 版粘性残差计算

与 core/fr_viscous_flux.py 对应的 CuPy 版本。
包含：
- 粘性物理通量（应力张量 + 热传导 + Boussinesq 假设）
- BR1 界面耦合（界面原始变量取平均，梯度取平均+镜像，边界面 Interior
  Penalty 罚项）——2026-08-23 之前这里完全没有实现，本模块自己的文档
  却一直写着"包含"，`compute_viscous_residual_fr_gpu` 实际上只算了
  体积散度项；`GPUFRSolver`（gpu_solver.py）不是死代码，`solve steady
  --backend gpu` 真实会走到这里，任何跟壁面/边界层相关的 GPU 粘性流动
  通量都缺一整项。修复对照 CPU 端 viscous_flux_kernel.py 的
  `compute_viscous_interface_correction_kernel` 逐字移植数学公式，
  按图着色分色 + owner_is_primary/neighbor_is_primary 分组去重（与
  gpu_inviscid.py::_compute_interface_correction_gpu 不同——那里没有
  这层过滤，本文件新增代码保留过滤是为了不重蹈这次 resume 调查里
  P0 端棱柱四边形侧面重复计数的同一类 bug，即便当前 FlatFaceGeometry
  的图着色分组是否真的需要它尚未确认，这层过滤本身零代价）。
- 体积项散度（张量收缩 + 度量项）

公式与 CPU 版完全一致，见 core/fr_viscous_flux.py 和
core/fr_residual/viscous_flux_kernel.py 模块文档。


## 文件分工(2026-09-24 拆包, 原 723 行)

    extrap.py            面通量点外插与 tilde 通量配对(GPU 粘性)
    volume.py            粘性体积项与残差入口(GPU)
    interface.py         粘性界面校正(GPU)

本 `__init__.py` re-export 全部既有名, 所以全仓库导入一字不改。
"""

from .constants import (  # noqa: F401
    GAMMA,
)
from .extrap import (  # noqa: F401
    _add_src1_to_fp,
    _extrap_side,
    _extrap_to_fp,
    _self_extrap_side,
    _viscous_tilde_flux_pair,
)
from .volume import (  # noqa: F401
    _viscous_volume_overintegrated_gpu,
    compute_temperature_gpu,
    compute_viscous_residual_fr_gpu,
)
from .interface import (  # noqa: F401
    _compute_viscous_interface_correction_gpu,
)

__all__ = [
    "GAMMA",
    "compute_temperature_gpu",
    "compute_viscous_residual_fr_gpu",
]
