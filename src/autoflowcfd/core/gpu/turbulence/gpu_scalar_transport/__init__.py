"""
AutoFlowCFD V2.0 - GPU 版湍流标量（k/omega）输运残差 (#7 第四次评审第四轮)

与 core/turbulence/transport.py + transport_kernel.py 对应的 CuPy 版本，
补齐 GPU SST/DDES/IDDES 长期缺失的输运项（此前 GPUTurbulenceSST 只有
逐点源项 ODE，`update_fields` 的 `transport_k`/`transport_omega`
参数从未被调用方传入，见 gpu_turbulence_sst.py 模块文档）。

不需要图着色分组：CPU numba 版用图着色规避多线程写冲突，`cp.scatter_add`
本身正确处理重复索引累加，全部面一次性向量化即可。

与 CPU 版保持的几条约定（每一条都曾因"只改了一份"出过真实缺陷）：

* 面上只用原生算子：自身外插 `boundary_extrap_native[op]`、提升
  `lift_native[op]`（按面算子索引），四面体体积项用 `D_native_tet_padded`；
* 只有 `owner_is_primary` / `neighbor_is_primary` 的记录贡献对应一侧（B-8
  混合拆分面同一物理面有两条记录）；
* **两侧各在自己的通量点顺序里构造跳变量**、按统一符号约定提升，扩散为
  IIPG 内罚（2026-09-26，理由与实测见 CPU 版 `transport/face_frames.py`）。

与 CPU 侧 `core/turbulence/transport/` 的分工**逐一对应**(faces /
omega_wall / residual), 便于两侧对照 -- 本项目反复出过"同一语义两份实现、
只改了一份"的缺陷, 让两边的文件边界重合能让漏改更容易被看见。

## 文件分工(2026-09-24 拆包, 原 743 行)

    faces.py             两侧坐标系的面外插、质量通量与 DG 提升(GPU，对应 CPU 版 face_frames.py)
    omega_wall.py        omega 壁面目标值/Dirichlet 掩码(GPU)
    residual.py          对流/扩散残差与顶层编排(GPU)

本 `__init__.py` re-export 全部既有名, 所以全仓库导入一字不改。
"""

from .faces import (  # noqa: F401
    _extrapolate_scalar_pair_gpu,
    _extrapolate_scalar_to_faces_gpu,
    _face_mass_flux_gpu,
    _lift_side_jumps_gpu,
)
from .omega_wall import (  # noqa: F401
    compute_omega_wall_target_gpu,
    compute_turbulence_face_masks_gpu,
)
from .residual import (  # noqa: F401
    _scalar_volume_div_overintegrated_gpu,
    compute_scalar_convection_residual_gpu,
    compute_scalar_diffusion_residual_gpu,
    compute_turbulence_transport_residual_gpu,
    scalar_convection_volume_divergence_gpu,
)

__all__ = [
    "compute_omega_wall_target_gpu",
    "compute_scalar_convection_residual_gpu",
    "scalar_convection_volume_divergence_gpu",
    "compute_scalar_diffusion_residual_gpu",
    "compute_turbulence_transport_residual_gpu",
    "compute_turbulence_face_masks_gpu",
]
