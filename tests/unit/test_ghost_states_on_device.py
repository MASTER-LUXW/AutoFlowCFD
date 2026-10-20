"""边界幽灵态在设备上计算，CPU 与 GPU 同一份实现（I-119）。

此前 GPU 残差（无粘、粘性、源项梯度提升）每次求值把整份原始变量 Q 拷回主机、在主机上算幽灵态、再把全部面
大小的数组传回设备（隐式下每次 J v 两次全场 PCIe 往返）。现在幽灵态公式按 `array_module` 取数组模块，
`compute_boundary_ghost_states` 在设备端面几何上外插并按边界组批量求值，只有 SEM 入口组经主机。
本机没有 CuPy：用一个记录调用的数组模块替身验证公式走的是传入数组的模块，并核对源码里不再有整场回传。
"""

import re

import numpy as np

from autoflowcfd.boundary import fr_ghost_state as fgs
from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry
from autoflowcfd.core.fr_residual.inviscid_kernel import boundary_ghost_faces, compute_boundary_ghost_states
from tests.unit._module_source import module_source
from tests.unit.test_p0_mixed_split_faces import _mixed_mesh, _provider


def test_ghost_face_list_is_cached_on_the_face_geometry():
    m = _mixed_mesh(1)
    flat = get_flat_face_geometry(m, m.operators)
    faces = boundary_ghost_faces(flat)
    assert boundary_ghost_faces(flat) is faces
    expected = np.flatnonzero(flat.is_boundary & (flat.owner_is_primary | flat.mixed_bnd_face))
    np.testing.assert_array_equal(faces, expected)


def test_formulas_use_the_module_of_the_input_arrays(monkeypatch):
    """五种边界公式按输入数组的模块取函数（设备数组走 CuPy 的同名函数），结果与 numpy 逐位相同。"""
    calls = []

    class _Recording:
        def __getattr__(self, name):
            calls.append(name)
            return getattr(np, name)

    rec = _Recording()
    monkeypatch.setattr(fgs, "array_module", lambda *a: rec)
    rng = np.random.default_rng(2)
    Q = rng.uniform(0.5, 2.0, (6, 5))
    n = rng.standard_normal((6, 3))
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    cfgs = [{"type": "WALL"}, {"type": "WALL", "is_no_slip": False}, {"type": "FARFIELD", "Q_free": Q[0]},
            {"type": "INLET", "Q_inlet": Q[1]}, {"type": "OUTLET", "p_outlet": 1.0}, {"type": "SYMMETRY"}]
    got = [fgs.ghost_state_for_config(c, Q, n) for c in cfgs]
    assert {"zeros", "sum", "tile", "asarray", "minimum"} <= set(calls)
    monkeypatch.undo()
    for c, g in zip(cfgs, got):
        np.testing.assert_array_equal(g, fgs.ghost_state_for_config(c, Q, n))


def test_batched_provider_matches_per_face_reference():
    m = _mixed_mesh(1)
    flat = get_flat_face_geometry(m, m.operators)
    prov = _provider(m)
    Q = np.random.default_rng(4).uniform(0.5, 2.0, (m.n_cells, m.n_sps_per_cell, 5))
    Qg = compute_boundary_ghost_states(flat, Q, prov)
    for f in boundary_ghost_faces(flat):
        E = flat.boundary_extrap_native[flat.owner_face_op[f]]
        np.testing.assert_allclose(Qg[f], prov(int(f), E @ Q[flat.owner_cell[f]], flat.true_normal[f]), rtol=1e-14)


def test_gpu_residuals_compute_ghost_states_on_the_device_geometry():
    for mod in ("autoflowcfd.core.gpu.residual.gpu_inviscid.residual",
                "autoflowcfd.core.gpu.residual.gpu_viscous.volume",
                "autoflowcfd.core.gpu.residual.gpu_corrected_gradient"):
        src = module_source(mod)
        assert re.search(r"compute_boundary_ghost_states\(\s*flat_face_gpu", src), mod
        assert "cp.asnumpy(Q" not in src, mod
