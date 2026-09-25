"""
AutoFlowCFD V2.0 - GPU 版湍流标量（k/omega）输运残差 (#7 第四次评审第四轮)

与 core/turbulence/transport.py + transport_kernel.py 对应的 CuPy 版本，
补齐 GPU SST/DDES/IDDES 长期缺失的输运项（此前 GPUTurbulenceSST 只有
逐点源项 ODE，`update_fields_gpu` 的 `transport_k`/`transport_omega`
参数从未被调用方传入，见 gpu_turbulence_sst.py 模块文档）。

不需要图着色分组：CPU numba 版本用图着色规避多线程写冲突（per-thread
buffer 的替代方案），但 `cp.scatter_add` 本身就正确处理重复索引累加，
对全部面一次性向量化处理即可，颜色分组对 GPU 版本的正确性没有必要。

分配机制复用 `gpu_inviscid_volume.py::distribute_face_correction_to_sps`
（V2.0 专家组盲审第四轮修复的 gather 机制，与 CPU numba kernel
`_distribute_point_scalar` 完全一致，仅 collapsed 面使用），不是矩阵乘法。

真实 bug 修复 + native 四面体（路径C）完整补全（2026-09-03，"把 native
全部补充完整"排查——本机没有真实 CuPy，本模块此前从未被真正执行验证过，
与本次同一排查发现的 gpu_inviscid.py/gpu_viscous.py/gpu_gradients.py 系列
bug 同源）：

1. **owner_is_primary/neighbor_is_primary 过滤缺失**（不是"刻意保留的
   行为差异"——本模块旧文档曾这样声称，但去核对 CPU 参考
   `transport_kernel.py::distribute_corrections_to_cells_kernel` 发现
   CPU 早在 2026-09-02（该函数自己的文档"真实 bug 修复"一节）就已经
   补上了这个过滤，旧文档的说法是过时信息，不是事实）：B-8 混合拆分面
   场景下，同一个物理面会被拆成 2 条记录，只有 `owner_is_primary=True`/
   `neighbor_is_primary=True` 的那条记录才应该贡献对应侧——本模块
   `_distribute_scalar_correction_gpu` 此前对全部面（含非 primary 的
   重复记录）无条件累加，等价于把这批面的贡献重复计入。现在补齐过滤，
   与 CPU 版逐字对应。

2. **完全没有 native 分派**：`_extrapolate_scalar_to_faces_gpu` 自身
   外插、`_distribute_scalar_correction_gpu` 面校正分配，此前都无条件
   走 collapsed 路径（`boundary_extrap[celltype,axis,side_idx]` 查表 +
   `distribute_face_correction_to_sps` 1D 修正函数分布）——对 native
   四面体面，`owner_axis`/`neighbor_axis` 存的是复用的 excluded_vertex
   （0~3），既会在 `axis` 维度只有 3 的表上越界（`==3` 时崩溃），修复
   越界后也仍然是错误结果（native 单纯形基没有"坍缩计算方向"，1D 分布
   机制本身不适用，必须用 `boundary_extrap_native`/`lift_native` DG
   提升算子）。现在按 CPU 版 `_extrap_owner_scalar_to_faces`/
   `distribute_corrections_to_cells_kernel` 的 native 分支逐字补齐：
   - 自身外插复用 `gpu_inviscid.py::_native_self_extrap`（形状签名
     `(n,n_fp,n_sps)` 与变量个数无关，标量场直接复用不需要改写）。
   - 面校正分配新增标量版 `_native_or_collapsed_contrib_scalar`（对照
     `gpu_inviscid.py::_native_or_collapsed_contrib`，去掉 5 变量末轴，
     `jump`/`contrib` 都是 `(n,n_fp)`/`(n,n_sps)`）。与 CPU 版一致，
     加权方式按面类型分派（collapsed 面乘 `|adj_row|`，native 面乘
     `true_area_weight`），加权发生在分配阶段，不再像旧版那样在调用方
     （`compute_scalar_convection/diffusion_residual_gpu`）提前统一乘
     `|owner_adj_row_exact|`——旧版这个提前加权对 native 面是错误的
     （native 面应该用 `true_area_weight` 而不是 `|adj_row|`），必须
     像 CPU 版一样把**未加权**的 `raw_jump_fp` 一路传到分配阶段，加权
     方式才能按面类型正确分派。
   - `compute_scalar_convection_residual_gpu`/
     `compute_scalar_diffusion_residual_gpu` 的四面体段 divergence
     收缩此前无条件用 `ops_data['D_3d_tet']`（坍缩坐标微分算子），与
     `gpu_gradients.py::compute_physical_gradient_gpu` 同一类遗漏——
     改用 `gpu_gradients.py` 已确立的自描述判据
     `'D_native_tet_padded' in ops_data`。

Args/Returns 类型标注、`_distribute_scalar_correction_gpu` 的调用方签名
均已同步更新（`correction_fp` 参数改名 `raw_jump_fp`，语义从"已加权"变
"未加权"，与 CPU 版 `raw_jump_fp` 命名及语义完全对齐）。

与 CPU 侧 `core/turbulence/transport/` 的分工**逐一对应**(faces /
omega_wall / residual), 便于两侧对照 -- 本项目反复出过"同一语义两份实现、
只改了一份"的缺陷, 让两边的文件边界重合能让漏改更容易被看见。

## 文件分工(2026-09-24 拆包, 原 743 行)

    faces.py             标量场面外插与面校正分配(GPU)
    omega_wall.py        omega 壁面目标值/松弛/Dirichlet 掩码(GPU)
    residual.py          对流/扩散残差与顶层编排(GPU)

本 `__init__.py` re-export 全部既有名, 所以全仓库导入一字不改。
"""

from .faces import (  # noqa: F401
    _distribute_scalar_correction_gpu,
    _extrapolate_scalar_to_faces_gpu,
    _lift_native_contrib_scalar,
)
from .omega_wall import (  # noqa: F401
    compute_omega_wall_target_gpu,
    compute_turbulence_face_masks_gpu,
    enforce_omega_wall_relaxation_gpu,
    omega_wall_cell_targets_gpu,
)
from .residual import (  # noqa: F401
    _scalar_volume_div_overintegrated_gpu,
    compute_scalar_convection_residual_gpu,
    compute_scalar_diffusion_residual_gpu,
    compute_turbulence_transport_residual_gpu,
)

__all__ = [
    "compute_omega_wall_target_gpu",
    "compute_scalar_convection_residual_gpu",
    "compute_scalar_diffusion_residual_gpu",
    "compute_turbulence_transport_residual_gpu",
    "compute_turbulence_face_masks_gpu",
    "enforce_omega_wall_relaxation_gpu",
    "omega_wall_cell_targets_gpu",
]
