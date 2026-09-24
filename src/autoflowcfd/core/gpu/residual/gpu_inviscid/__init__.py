"""
AutoFlowCFD V2.0 - P>=1 高阶 FR 无粘残差 GPU 实现

完整的高阶 FR 无粘残差 GPU 版本，对应 core/fr_residual_inviscid.py。
包含两部分：
1. 体积项：CuPy 向量化（物理通量 + 张量收缩 + 度量项）
2. 界面项：CuPy kernel（AUSM+up + 校正分配，按图着色逐色处理）

设计：
- 体积项完全用 CuPy 向量化操作（cp.matmul, cp.tensordot），底层走 cuBLAS
- 界面项使用 CuPy ElementwiseKernel 逐面计算 AUSM+up 通量
- 校正分配使用图着色保证无冲突写入（同色面无 owner_cell 冲突）
- 数据全部常驻 GPU，避免 CPU↔GPU 传输


## 文件分工(2026-09-24 拆包, 原 710 行)

    flux.py              AUSM+up 批量通量(GPU, 与 kernels.py 逐字对应)
    interface.py         无粘界面校正与其 gather/scatter 辅助(GPU)
    residual.py          体积项与残差入口(GPU)

本 `__init__.py` re-export 全部既有名, 所以全仓库导入一字不改。
"""

from .flux import (  # noqa: F401
    _ausm_up_flux_batch_gpu,
)
from .interface import (  # noqa: F401
    _add_q_src1_to_fp,
    _ausm_direction_with_fallback,
    _compute_interface_correction_gpu,
    _extrap_q_to_fp,
    _lift_native_contrib,
    _native_self_extrap,
    _scatter_add_to_correction,
)
from .residual import (  # noqa: F401
    _compute_boundary_ghost_states_gpu,
    _compute_volume_term_gpu,
    compute_inviscid_residual_fr_gpu,
)

__all__ = [
    "compute_inviscid_residual_fr_gpu",
]
