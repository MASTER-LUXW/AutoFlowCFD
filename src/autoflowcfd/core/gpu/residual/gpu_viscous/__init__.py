"""
AutoFlowCFD V2.0 - GPU 版粘性残差计算

与 core/fr_residual/viscous_flux/ 对应的 CuPy 版本。
包含：
- 粘性物理通量（应力张量 + 热传导 + Boussinesq 假设）
- BR1 界面耦合（界面原始变量取平均，梯度取平均+镜像，边界面 Interior
  Penalty 罚项）——2026-08-23 之前这里完全没有实现，本模块自己的文档
  却一直写着"包含"，`compute_viscous_residual_fr_gpu` 实际上只算了
  体积散度项；`GPUFRSolver`（gpu_solver/）不是死代码，`solve steady
  --backend gpu` 真实会走到这里，任何跟壁面/边界层相关的 GPU 粘性流动
  通量都缺一整项。修复对照 CPU 端 viscous_flux_kernel.py 的
  `compute_viscous_interface_correction_kernel` 逐字移植数学公式，
  按图着色分色 + owner_is_primary/neighbor_is_primary 分组去重（与
  gpu_inviscid/interface.py::_compute_interface_correction_gpu 同一结构：非 primary 记录的跨单元来源 src0 不写入、
  不过滤就会用全零假邻居态并把 owner 侧贡献计两次，见该函数文档）。与 CPU 的逐位交叉验证在 numpy 替身下运行
  （`tests/unit/test_gpu_viscous_residual_crosscheck.py`）。
- 体积项散度（张量收缩 + 度量项）

公式与 CPU 版完全一致，见 core/fr_residual/viscous_flux/ 和
core/fr_residual/viscous_flux_kernel.py 模块文档。

## 文件分工

    extrap.py            面通量点外插与 tilde 通量配对(GPU 粘性)
    volume.py            粘性体积项与残差入口(GPU)
    interface.py         粘性界面校正(GPU)

本 `__init__.py` re-export 各子模块的公开名。
"""

from .volume import compute_temperature_gpu, compute_viscous_residual_fr_gpu  # noqa: F401

__all__ = [
    "compute_temperature_gpu",
    "compute_viscous_residual_fr_gpu",
]
