"""AutoFlowCFD V2.0 - CPU MPI 分布式路径的问题单元人工粘性。

判据与施加形式见 `core/fr_operators/artificial_viscosity/entropy_viscosity.py`
（单机、单 GPU、多 GPU 共用同一个数组模块无关的实现）。这里只做分布式的索引
空间换算，约定与分布式湍流输运相同（`distributed_turbulence.py`）：

* 平均流 halo 交换后按 `dist_fc.perm` 重排到 compact 空间（棱柱在前、local+halo）；
* 系数 `nu` 只依赖单元自身的解（单元内梯度），所以 halo 单元用 halo 数据算出的
  值与拥有它的 rank 算出的**逐位相同**，不需要再交换一次系数；
* 拉普拉斯项在 compact 空间用标量扩散装配（`flat_face_override=dist_fc.base_flat`），
  local 单元的面都齐全；结果按 `inv_perm` 换回原生排列后取 local 段（halo 行邻域
  不完整，丢弃）。
"""

import numpy as np

from autoflowcfd.core.mpi.distributed_compute import DistributedMeshAdapter


def _compact_state(U_local, partition, halo_exchange, dist_fc, local_mesh, ops):
    U_compact = halo_exchange.exchange(U_local)[dist_fc.perm]
    return U_compact, DistributedMeshAdapter(partition, dist_fc, local_mesh, ops)


def distributed_artificial_diffusivity(U_local, partition, halo_exchange, dist_fc, local_mesh, ops,
                                       *, order: int, alpha_av: float) -> np.ndarray:
    """compact 排列的人工扩散系数 nu (n_compact, n_sps)（运动粘度）。"""
    from autoflowcfd.core.fr_operators.artificial_viscosity import compute_artificial_diffusivity
    from autoflowcfd.core.fr_residual.viscous import compute_scalar_gradient

    U_compact, adapter = _compact_state(U_local, partition, halo_exchange, dist_fc, local_mesh, ops)
    n_compact = U_compact.shape[0]
    return compute_artificial_diffusivity(
        U_compact[..., :5], int(order), adapter.cell_volumes,
        np.arange(n_compact) < int(dist_fc.base_flat.n_prism),
        lambda phi: compute_scalar_gradient(phi, ops, adapter), alpha_av=alpha_av)


def distributed_artificial_diffusion_dudt(U_local, nu_compact, partition, halo_exchange, dist_fc,
                                          local_mesh, ops) -> np.ndarray:
    """local 排列的 `div(nu grad U_k)`（k = 0..4，dU/dt 约定），形状同 `U_local`。"""
    from autoflowcfd.core.fr_operators.artificial_viscosity import artificial_diffusion_residual
    from autoflowcfd.core.turbulence.transport import compute_scalar_diffusion_residual

    U_compact, adapter = _compact_state(U_local, partition, halo_exchange, dist_fc, local_mesh, ops)
    res_compact = artificial_diffusion_residual(
        U_compact[..., :5], nu_compact,
        lambda phi, gamma: compute_scalar_diffusion_residual(
            phi, gamma, adapter, ops, flat_face_override=dist_fc.base_flat))
    out = np.zeros_like(U_local)
    out[..., :5] = res_compact[dist_fc.inv_perm][:partition.n_local_cells]
    return out


def local_from_compact(field_compact, dist_fc, n_local: int) -> np.ndarray:
    """compact 排列的逐单元场换回原生排列并取 local 段。"""
    return field_compact[dist_fc.inv_perm][:n_local]
