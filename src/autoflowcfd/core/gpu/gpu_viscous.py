"""
AutoFlowCFD V2.0 - GPU 版粘性残差计算

与 core/fr_viscous_flux.py 对应的 CuPy 版本。
包含：
- 粘性物理通量（应力张量 + 热传导 + Boussinesq 假设）
- BR1 界面耦合（界面原始变量取平均，梯度取平均+镜像）
- 体积项散度（张量收缩 + 度量项）

公式与 CPU 版完全一致，见 core/fr_viscous_flux.py 模块文档。
"""

import numpy as np
from typing import Optional
from loguru import logger

from autoflowcfd.core.gpu import get_cupy
from autoflowcfd.core.gpu.gpu_volume_contract import (
    gpu_contract_shared_operator_1axis,
    gpu_contract_shared_operator_2axis,
)
from autoflowcfd.core.gpu.gpu_flux import viscous_physical_flux_gpu, conserved_to_primitive_gpu
from autoflowcfd.core.gpu.gpu_gradients import compute_physical_gradient_gpu

GAMMA = 1.4
R_AIR = 287.0


def compute_temperature_gpu(Q):
    """GPU 版温度计算。T = p/(rho*R)。"""
    cp = get_cupy()
    rho = cp.maximum(Q[..., 0], 1e-10)
    return Q[..., 4] / (rho * R_AIR)


def compute_viscous_residual_fr_gpu(
    U,
    mesh,
    ops,
    mu=1.8e-5,
    Pr=0.72,
    mu_t_field=None,
    Pr_t=0.9,
    boundary_ghost_provider=None,
    mesh_data=None,
    ops_data=None,
    device_id=0,
):
    """GPU 版粘性残差计算。

    与 core/fr_viscous_flux.py::compute_viscous_residual_fr 公式一致。

    Args:
        U: CuPy 数组 (n_cells, n_sps, n_vars)
        mesh: HighOrderMesh
        ops: FROperators
        mu: 分子动力粘度
        Pr: 分子普朗特数
        mu_t_field: CuPy 数组 (n_cells, n_sps)，湍流涡粘度（可选）
        Pr_t: 湍流普朗特数
        boundary_ghost_provider: 边界幽灵态提供者
        mesh_data: 预上传的网格数据
        ops_data: 预上传的算子数据
        device_id: GPU 设备 ID

    Returns:
        viscous_residual: CuPy 数组 (n_cells, n_sps, 5)
    """
    cp = get_cupy()
    n_cells = mesh.n_cells
    n_sps = mesh.n_sps_per_cell
    n_prism = mesh.n_prism_cells

    # 准备网格数据
    if mesh_data is None:
        from autoflowcfd.core.gpu.gpu_inviscid import _prepare_mesh_data, _prepare_ops_data
        mesh_data = _prepare_mesh_data(cp, mesh, device_id)
        ops_data = _prepare_ops_data(cp, ops, device_id)

    det_jacs = mesh_data['det_jacs']

    # 1. 计算物理梯度
    Q = conserved_to_primitive_gpu(U[..., :5])
    grad_U = compute_physical_gradient_gpu(U[..., :5], mesh_data, ops_data)

    # 速度梯度和温度梯度
    grad_vel = grad_U[..., 1:4, :]  # (n_cells, n_sps, 3, 3)
    grad_T_scalar = compute_temperature_gpu(Q)
    grad_T = compute_physical_gradient_gpu(
        grad_T_scalar[..., None], mesh_data, ops_data
    )  # (n_cells, n_sps, 1, 3) → squeeze
    grad_T = grad_T[..., 0, :]  # (n_cells, n_sps, 3)

    # 2. 体积项：粘性物理通量 + 散度
    # mu_t_field 是调用方按 CPU 版约定（core/fr_residual/viscous_flux.py）
    # 传入的动力涡粘度 mu_t = rho * nu_t，与分子粘度 mu 量纲一致，直接相加
    # 即可得到有效动力粘度。此前这里误多做了一次"除以 rho"把 mu_t_field
    # 转换成运动粘度 nu_t 再与动力粘度 mu 相加，量纲不一致，等效于把湍流
    # 粘性应力贡献错误缩小了约 1/rho 倍。
    if mu_t_field is None:
        mu_eff = mu
    else:
        mu_eff = mu + mu_t_field

    G_phys = viscous_physical_flux_gpu(
        Q, grad_vel, grad_T, mu_eff, Pr, mu_t=0.0, Pr_t=Pr_t,
    )  # (n_cells, n_sps, 3, 5)

    # 逆变通量
    adj_j = mesh_data['adj_j']
    G_tilde = cp.matmul(adj_j, G_phys)  # (n_cells, n_sps, 3, 5)

    # 散度
    div_G = cp.zeros((n_cells, n_sps, 5), dtype=cp.float64)
    if n_prism > 0:
        div_G[:n_prism] = gpu_contract_shared_operator_2axis(
            ops_data['D_3d_prism'], G_tilde[:n_prism]
        )
    if n_cells > n_prism:
        div_G[n_prism:] = gpu_contract_shared_operator_2axis(
            ops_data['D_3d_tet'], G_tilde[n_prism:]
        )

    # 粘性残差 = +div(G) / det(J)（注意：粘性项是正号，与无粘的负号相反）
    viscous_residual = div_G / det_jacs[..., None]

    return viscous_residual
