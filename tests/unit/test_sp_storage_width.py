"""解点存储宽度 = 原生棱柱真实自由度（I-29）与旧布局 checkpoint 的迁移。

2026-10-19 之前每单元解点存储宽度是坍缩棱柱基留下的 `(p+1)^3`（P2 27、P3 64），原生基上线后两类单元都只用其中
一部分；现在宽度为 `fr/native_padding.py::sps_width`（P2 18、P3 40），棱柱用满、四面体零填充。旧 checkpoint 在
`load_checkpoint` 里截到新宽度（只去掉填充槽位）。
"""

from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from autoflowcfd.fr.native_padding import order_from_n_sps, real_sps_per_cell, sps_width

_HISTORY = {"iterations": [1], "residuals": {"rho": [1.0]}, "coefficients": {}, "cfl_history": [0.03]}


def test_width_is_native_prism_dof_count():
    assert [sps_width(p) for p in range(5)] == [1, 6, 18, 40, 75]
    for p in range(6):
        n_prism, n_tet = real_sps_per_cell(p)
        assert sps_width(p) == n_prism >= n_tet
        assert order_from_n_sps(sps_width(p)) == p


@pytest.mark.parametrize("legacy", [8, 27, 64])
def test_legacy_cube_widths_are_not_any_order(legacy):
    """旧宽度与任何新宽度都不相等：迁移判别无歧义，漏迁移的数组在第一次归约时报错而不是被静默重解释。"""
    with pytest.raises(ValueError, match="不是任何阶数的存储宽度"):
        order_from_n_sps(legacy)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_operators_use_the_width(order):
    from autoflowcfd.fr.operators import generate_fr_operators

    ops = generate_fr_operators(order)
    w = sps_width(order)
    n_tet = real_sps_per_cell(order)[1]
    assert ops.n_sps == w
    assert ops.D_native_prism_padded.shape == (w, w, 3)
    assert ops.D_native_tet_padded.shape == (w, w, 3)
    # 四面体块之外为零（零填充块对角）
    assert not ops.D_native_tet_padded[n_tet:].any()
    assert not ops.D_native_tet_padded[:, n_tet:].any()


def _write(tmp_path, fields, metadata):
    from autoflowcfd.core.utils.checkpoint import CheckpointManager

    mgr = CheckpointManager(config=SimpleNamespace(), output_dir=str(tmp_path), quiet=True)
    n_cells = next(iter(fields.values())).shape[0]
    path = mgr.save(solution=np.ones((n_cells, 5)), history=_HISTORY, iteration=7,
                    metadata=metadata, extra_fields=fields)
    return mgr, path


def _legacy_fields(order, n_cells=3, seed=0):
    rng = np.random.default_rng(seed)
    legacy = (order + 1) ** 3
    w = sps_width(order)
    U = rng.uniform(1.0, 2.0, (n_cells, legacy, 5))
    U[:, w:] = 1e30                                     # 填充槽位：迁移后不得残留
    k = rng.uniform(0.1, 0.2, (n_cells, legacy))
    k[:, w:] = 1e30
    return {
        "U_sps": U,
        "k_field": k,
        "U_prev_sps": U * 0.5,
        "tau_accum": rng.uniform(0.0, 1.0, n_cells),     # 逐单元，一维
        "turb_mean_k": rng.uniform(0.0, 1.0, n_cells),
    }


@pytest.mark.parametrize("order", [1, 2, 3])
def test_legacy_checkpoint_is_cut_to_new_width(tmp_path, order):
    from autoflowcfd.core.utils.checkpoint_load import load_checkpoint

    fields = _legacy_fields(order)
    legacy = (order + 1) ** 3
    w = sps_width(order)
    mgr, path = _write(tmp_path, fields, {"order": order, "n_sps_per_cell": legacy})
    with h5py.File(path, "r+") as f:                    # 2026-10-19 之前写的 checkpoint 没有这个属性
        del f["metadata"].attrs["sp_layout"]

    solution, _, iteration, meta = load_checkpoint(mgr, path)
    out = meta["fields"]
    assert iteration == 7 and solution.shape == (3, 5)
    np.testing.assert_array_equal(out["U_sps"], fields["U_sps"][:, :w])
    np.testing.assert_array_equal(out["k_field"], fields["k_field"][:, :w])
    np.testing.assert_array_equal(out["U_prev_sps"], fields["U_prev_sps"][:, :w])
    np.testing.assert_array_equal(out["tau_accum"], fields["tau_accum"])
    np.testing.assert_array_equal(out["turb_mean_k"], fields["turb_mean_k"])
    assert meta["n_sps_per_cell"] == w
    assert order_from_n_sps(out["U_sps"].shape[1]) == order


def test_new_checkpoint_records_layout_and_is_not_cut(tmp_path):
    from autoflowcfd.core.utils.checkpoint_load import SP_LAYOUT, load_checkpoint

    w = sps_width(2)
    U = np.random.default_rng(1).uniform(1.0, 2.0, (3, w, 5))
    mgr, path = _write(tmp_path, {"U_sps": U}, {"order": 2, "n_sps_per_cell": w})
    with h5py.File(path, "r") as f:
        saved = f["metadata"].attrs["sp_layout"]
    assert (saved.decode() if isinstance(saved, bytes) else saved) == SP_LAYOUT
    _, _, _, meta = load_checkpoint(mgr, path)
    np.testing.assert_array_equal(meta["fields"]["U_sps"], U)


def test_unknown_layout_is_refused(tmp_path):
    from autoflowcfd.core.utils.checkpoint_load import load_checkpoint

    mgr, path = _write(tmp_path, {"U_sps": np.ones((2, 18, 5))}, {"order": 2})
    with h5py.File(path, "r+") as f:
        f["metadata"].attrs["sp_layout"] = np.string_("future_layout")
    with pytest.raises(Exception, match="解点存储布局"):
        load_checkpoint(mgr, path)


def test_legacy_init_from_lifts_to_higher_order(tmp_path):
    """旧布局 P1 checkpoint 经迁移后照常延拓到 P2（`--init-from` 路径）：线性场在两个空间里都精确，延拓后不变。"""
    from autoflowcfd.core.utils.checkpoint_load import load_checkpoint
    from autoflowcfd.core.utils.order_continuation.initial_field import prolongate_checkpoint_fields
    from autoflowcfd.fr.native_prism.basis import build_native_prism_nodes

    nodes1 = build_native_prism_nodes(1)
    nodes2 = build_native_prism_nodes(2)
    lin = lambda x: 1.0 + 0.3 * x[:, 0] - 0.2 * x[:, 1] + 0.1 * x[:, 2]
    U = np.zeros((1, 8, 5))
    U[0, :6, :] = lin(nodes1)[:, None]
    U[0, 6:, :] = U[0, 0]                               # 旧约定：填充槽位复制真实 SP #0
    mgr, path = _write(tmp_path, {"U_sps": U}, {"order": 1, "n_sps_per_cell": 8})
    with h5py.File(path, "r+") as f:
        del f["metadata"].attrs["sp_layout"]
    _, _, _, meta = load_checkpoint(mgr, path)

    out, ckpt_order = prolongate_checkpoint_fields(meta["fields"], 2, np.array([True]), None)
    assert ckpt_order == 1
    assert out["U_sps"].shape == (1, sps_width(2), 5)
    np.testing.assert_allclose(out["U_sps"][0, :, 0], lin(nodes2), rtol=0, atol=1e-13)
