"""多 GPU 分布式湍流测试（test_gpu_distributed_turbulence*.py）共用的 CuPy 替身与算例构造。"""

import types
import numpy as np
import pytest

from autoflowcfd.core.turbulence.des import DDESModel, IDDESModel
from autoflowcfd.core.fr_residual.inviscid import primitive_to_conserved
from autoflowcfd.core.gpu.turbulence.gpu_turbulence_des import GPUDDESModel, GPUIDDESModel
import autoflowcfd.core.gpu.distributed.gpu_distributed_init as gdi_mod
import autoflowcfd.core.gpu.distributed.gpu_distributed as gd_mod
import autoflowcfd.core.gpu.residual.gpu_gradients as gpu_gradients_mod
import autoflowcfd.core.gpu.residual.gpu_volume_contract as gpu_volume_contract_mod
import autoflowcfd.core.gpu.residual.gpu_flux as gpu_flux_mod
import autoflowcfd.core.gpu.turbulence.gpu_scalar_transport as gst_mod
import autoflowcfd.core.gpu.turbulence.gpu_turbulence_sst as gpu_turbulence_sst_mod
import autoflowcfd.core.gpu.turbulence.gpu_turbulence_des as gpu_turbulence_des_mod
import autoflowcfd.core.gpu.turbulence.gpu_sgs as gpu_sgs_mod
import autoflowcfd.core.gpu.gpu_modal_filter as gpu_modal_filter_mod

from tests.unit._gpu_cupy_shim import patch_module_get_cupy
from tests.unit._numpy_as_cupy import NumpyAsCupy


def _bind_turb_source(stub):
    """把 `_GPUDistributedTurbSourceMixin` 的求值件绑定到替身上（被测入口
    `_compute_turbulence_source_distributed` 通过 `self` 调用它们）。"""
    from autoflowcfd.core.gpu.distributed.gpu_distributed_init.turb_source import (
        _GPUDistributedTurbSourceMixin as M,
    )
    for name in ("_les_mu_t_compact", "_prepare_turbulence_view_distributed", "_turbulence_velocity_gradient_compact",
                 "_sync_turbulence_view", "_evaluate_turbulence_rates_distributed",
                 "_finalize_turbulence_update_distributed", "_write_back_turbulence_distributed"):
        setattr(stub, name, types.MethodType(getattr(M, name), stub))


@pytest.fixture(autouse=True)
def _patch_get_cupy(monkeypatch):
    shim = NumpyAsCupy()
    patch_module_get_cupy(monkeypatch, [
        gdi_mod, gd_mod, gpu_gradients_mod, gpu_volume_contract_mod, gpu_flux_mod,
        gst_mod, gpu_turbulence_sst_mod, gpu_turbulence_des_mod, gpu_sgs_mod,
        gpu_modal_filter_mod], shim)
    # `GPUTurbulenceSST.__init__`/`GPUDDESModel.__init__`/`GPUWALEModel.
    # __init__` 单独检查模块级 `gpu_available` 标志（与 `get_cupy()` 是
    # 否被换成替身无关），本机没有真实 CuPy 时恒为 False，需要一并
    # patch 掉才能在没有真实 CUDA 设备时构造这些类。
    monkeypatch.setattr(gpu_turbulence_sst_mod, "gpu_available", True)
    monkeypatch.setattr(gpu_turbulence_des_mod, "gpu_available", True)
    monkeypatch.setattr(gpu_sgs_mod, "gpu_available", True)


def _prepare_compact_mesh_data(mesh, ops, compact_global_ids):
    """构造 compact 索引空间的 mesh_data/ops_data（与 `_CompactMeshDataView`
    /`GPUArrayManager.upload_mesh_data` 语义一致，只是数组仍是 numpy）。"""
    n_sps = mesh.n_sps_per_cell
    det_jacs = mesh.jacobians['det_jacs'].reshape(mesh.n_cells, n_sps)[compact_global_ids]
    inv_jacs = mesh.jacobians['inv_jacs'].reshape(mesh.n_cells, n_sps, 3, 3)[compact_global_ids]
    adj_j = det_jacs[..., None, None] * inv_jacs
    mesh_data = {
        'det_jacs': det_jacs, 'inv_jacs': inv_jacs, 'adj_j': adj_j,
        'n_cells': len(compact_global_ids), 'n_prism': None,  # 调用方会覆盖 n_prism
        'D_3d_prism': ops.D_3d_prism, 'D_3d_tet': ops.D_3d_tet,
    }
    # 补齐生产 GPU 路径会上传、替身容易漏掉的键，见
    # _gpu_standin_helpers 模块文档。本文件算子与网格数据在同一个
    # dict，且细点度量要按 compact 索引空间切。
    from ._gpu_standin_helpers import complete_gpu_standin
    complete_gpu_standin(mesh, ops, mesh_data, mesh_data,
                         compact_ids=compact_global_ids)
    return mesh_data


def _nonuniform_state(mesh, rng):
    rho_inf, u_inf, v_inf, w_inf, p_inf = 1.225, 30.0, 5.0, -3.0, 101325.0
    n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell
    Q = np.zeros((n_cells, n_sps, 5))
    Q[..., 0] = rho_inf * (1.0 + rng.uniform(-0.02, 0.02, size=(n_cells, n_sps)))
    Q[..., 1] = u_inf + rng.uniform(-3.0, 3.0, size=(n_cells, n_sps))
    Q[..., 2] = v_inf + rng.uniform(-2.0, 2.0, size=(n_cells, n_sps))
    Q[..., 3] = w_inf + rng.uniform(-2.0, 2.0, size=(n_cells, n_sps))
    Q[..., 4] = p_inf * (1.0 + rng.uniform(-0.01, 0.01, size=(n_cells, n_sps)))
    U = np.stack(
        [primitive_to_conserved(Q[c, s]) for c in range(n_cells) for s in range(n_sps)]
    ).reshape(n_cells, n_sps, 5)
    k_field = 1.0 * (1.0 + rng.uniform(-0.3, 0.3, size=(n_cells, n_sps)))
    omega_field = 500.0 * (1.0 + rng.uniform(-0.3, 0.3, size=(n_cells, n_sps)))
    return U, k_field, omega_field


def _make_ddes_gpu(turb_model_name):
    if turb_model_name == "SST":
        return None
    if turb_model_name == "DDES":
        return GPUDDESModel()
    if turb_model_name == "IDDES":
        return GPUIDDESModel()
    raise ValueError(turb_model_name)


def _make_ddes_cpu(turb_model_name):
    if turb_model_name == "SST":
        return None
    if turb_model_name == "DDES":
        return DDESModel()
    if turb_model_name == "IDDES":
        return IDDESModel()
    raise ValueError(turb_model_name)
