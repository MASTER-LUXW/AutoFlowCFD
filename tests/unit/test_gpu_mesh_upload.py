"""GPU 网格数据上传（I-113）：设备上只存 det(J) 与 adj(J)，物理梯度按 adj/det 现算、与 CPU 一致；每次上传返回新
字典（换阶重传时不残留上一阶的键）；单 GPU 与多 GPU 同一个度量上传函数。"""

import numpy as np
import pytest

import autoflowcfd.core.gpu.array_manager as array_manager_mod
from autoflowcfd.core.fr_operators.gradients import compute_physical_gradient
from autoflowcfd.fr.operators import generate_fr_operators
from tests.unit._gpu_cupy_shim import patch_module_get_cupy
from tests.unit._numpy_as_cupy import NumpyAsCupy
from tests.unit.test_fr_residual_inviscid import _build_synthetic_mixed_mesh


@pytest.fixture()
def manager(monkeypatch):
    import autoflowcfd.core.gpu.residual.gpu_gradients as grad_mod
    import autoflowcfd.core.gpu.residual.gpu_volume_contract as contract_mod

    shim = NumpyAsCupy()
    monkeypatch.setattr(array_manager_mod, "gpu_available", True)
    patch_module_get_cupy(monkeypatch, [array_manager_mod, grad_mod, contract_mod], shim, required=False)
    return array_manager_mod.GPUArrayManager(device_id=0)


@pytest.mark.parametrize("order", [1, 2])
def test_only_det_and_adjugate_are_resident_and_gradient_matches_cpu(manager, order):
    from autoflowcfd.core.gpu.residual.gpu_gradients import compute_physical_gradient_gpu

    mesh = _build_synthetic_mixed_mesh(order)
    ops = generate_fr_operators(order)
    data = manager.upload_mesh_data(mesh, ops)
    assert "inv_jacs" not in data and {"det_jacs", "adj_j", "cell_volumes"} <= set(data)
    field = np.random.default_rng(order).standard_normal((mesh.n_cells, mesh.n_sps_per_cell, 4))
    np.testing.assert_allclose(compute_physical_gradient_gpu(field, data, data),
                               compute_physical_gradient(field, mesh, ops), rtol=1e-12, atol=1e-10)


def test_reupload_returns_a_fresh_dict_without_stale_keys(manager):
    """降阶到 P0 时没有过积分细点度量：此前 `self.mesh_data.update(...)` 累加，P1 的细点度量键残留。"""
    mesh = _build_synthetic_mixed_mesh(1)
    p1 = manager.upload_mesh_data(mesh, generate_fr_operators(1))
    assert any(key.startswith("adj_j_fine") for key in p1)
    mesh.set_order(0)
    p0 = manager.upload_mesh_data(mesh, generate_fr_operators(0))
    assert p0 is not p1 and not any(key.startswith("adj_j_fine") for key in p0)


def test_memory_usage_uses_the_device_free_memory(manager):
    """余量与 `system_memory.device_available_bytes` 同一口径（设备空闲 + 显存池空闲块）。"""
    from autoflowcfd.core.utils.system_memory import device_available_bytes

    usage = manager.get_memory_usage()
    assert usage["free_mb"] == pytest.approx(device_available_bytes(NumpyAsCupy()) / 1024 ** 2)
    manager.cleanup()
