"""CPU MPI 分布式路径上的传感器门控（`AFCFD_FILTER_MODE=sensor`）。

`sensor` 是 2026-09-17 定下的默认滤波档，但当时只有单机 CPU 接线，其余
三条后端默认退回 `project`（全局逐 stage 施加精确投影，功能上等于
legacy：P1 退化成 P0、壁面剪应力恒为零）。2026-09-18 补齐 CPU MPI。

本文件的核心判据是**分区独立性**：BJ 越界判据要面邻居的单元均值，分区
边界上的邻居是 halo 单元。如果实现把分区边界面当边界面排除，掩码就会
随 rank 数变化——同一个算例换分区数得到不同的解，这对求解器不可接受。
`test_mask_is_partition_independent` 直接把这条性质钉成逐位相等。
"""

import numpy as np
import pytest

from autoflowcfd.core.fr_operators.bounds_sensor import (
    compute_bounds_violation_mask,
)
from autoflowcfd.core.fr_solver.filter import (
    build_sensor_gated_filter_func_arrays,
    resolve_filter_mode,
    _SENSOR_MODE_SUPPORTED_BACKENDS,
)

from tests.unit._sensor_gate_distributed_common import _FREESTREAM, _REF, _build_global_case, _rank_view


# ------------------------------------------------------------------- 判据

class TestPartitionIndependence:
    def test_mask_is_partition_independent(self):
        """两个 rank 各自算出的掩码，与单机全局掩码逐位相同。

        这是本次接线的唯一决定性判据：它同时否掉「把分区边界面当边界面
        排除」和「只用本地单元均值」两种偷懒实现——两者都会在分区边界
        附近给出不同的掩码。
        """
        field, owner, neigh, is_bnd, _ = _build_global_case()
        n_cells = field.shape[0]
        global_mask = compute_bounds_violation_mask(
            field, owner, neigh, is_bnd, ref_scales=_REF)
        assert 0 < global_mask.sum() < n_cells, (
            f"掩码必须既不空也不满才有区分力，实际 "
            f"{global_mask.sum()}/{n_cells}")

        cut = n_cells // 2
        rank_cells = [np.arange(0, cut), np.arange(cut, n_cells)]
        n_partition_faces = 0
        for local_ids in rank_cells:
            native_ids, n_local, o_r, n_r, b_r = _rank_view(
                owner, neigh, is_bnd, local_ids)
            assert len(native_ids) > n_local, "该分区应当有 halo 单元"
            n_partition_faces += int(
                ((o_r >= n_local) | ((~b_r) & (n_r >= n_local))).sum())
            field_ext = field[native_ids]
            mask_r = compute_bounds_violation_mask(
                field_ext, o_r, n_r, b_r, ref_scales=_REF)[:n_local]
            np.testing.assert_array_equal(
                mask_r, global_mask[local_ids],
                err_msg="分区掩码与全局掩码不一致：门控结果依赖了分区数")
        assert n_partition_faces > 0, "构造的算例没有分区边界面，测不到东西"

    def test_excluding_partition_faces_would_change_the_mask(self):
        """反证：如果把 halo 面当边界面排除，掩码**确实**会变。

        没有这条，上面那个「逐位相同」可能只是因为算例里分区边界无关紧要。
        """
        field, owner, neigh, is_bnd, _ = _build_global_case()
        n_cells = field.shape[0]
        global_mask = compute_bounds_violation_mask(
            field, owner, neigh, is_bnd, ref_scales=_REF)
        cut = n_cells // 2
        differed = False
        for local_ids in (np.arange(0, cut), np.arange(cut, n_cells)):
            native_ids, n_local, o_r, n_r, b_r = _rank_view(
                owner, neigh, is_bnd, local_ids)
            # 错误实现：把所有涉及 halo 的面标成边界面
            b_wrong = b_r | (o_r >= n_local) | (n_r >= n_local)
            mask_wrong = compute_bounds_violation_mask(
                field[native_ids], o_r, n_r, b_wrong,
                ref_scales=_REF)[:n_local]
            if not np.array_equal(mask_wrong, global_mask[local_ids]):
                differed = True
        assert differed, (
            "构造的算例区分不了正确/错误实现，上面那条逐位相等是空的")


class TestHaloExtendContract:
    def test_halo_extend_none_matches_single_machine(self):
        """n_halo == 0（单 rank）时，扩展路径与单机路径逐位相同。"""
        field, owner, neigh, is_bnd, _ = _build_global_case()
        n_cells, n_sps = field.shape[0], field.shape[1]
        kw = dict(sensor="bounds", owner_cell=owner, neighbor_cell=neigh,
                  is_boundary=is_bnd, freestream=_FREESTREAM)
        # 一个真正非平凡的投影矩阵（把单元内非常数内容抹掉）
        F = np.full((n_sps, n_sps), 1.0 / n_sps)
        cip = np.zeros(n_cells, dtype=bool)
        plain = build_sensor_gated_filter_func_arrays(
            n_cells, n_sps, 1, F, F, cell_is_prism=cip, **kw)
        extended = build_sensor_gated_filter_func_arrays(
            n_cells, n_sps, 1, F, F, cell_is_prism=cip,
            halo_extend=lambda U: U, **kw)
        flat = field.reshape(n_cells * n_sps, 5).copy()
        np.testing.assert_array_equal(plain(flat.copy()), extended(flat.copy()))

    def test_halo_extend_actually_changes_the_result(self):
        """halo 均值真的参与了包络：被施加滤波的单元集合应当恰好等于
        全局掩码在本 rank 上的限制。"""
        field, owner, neigh, is_bnd, _ = _build_global_case()
        n_cells, n_sps = field.shape[0], field.shape[1]
        cut = n_cells // 2
        local_ids = np.arange(0, cut)
        native_ids, n_local, o_r, n_r, b_r = _rank_view(
            owner, neigh, is_bnd, local_ids)
        F = np.full((n_sps, n_sps), 1.0 / n_sps)
        cip = np.zeros(n_local, dtype=bool)
        ff = build_sensor_gated_filter_func_arrays(
            n_local, n_sps, 1, F, F, cell_is_prism=cip, sensor="bounds",
            owner_cell=o_r, neighbor_cell=n_r, is_boundary=b_r,
            freestream=_FREESTREAM,
            halo_extend=lambda U: field[native_ids])
        flat = field[local_ids].reshape(n_local * n_sps, 5).copy()
        out = ff(flat.copy())
        global_mask = compute_bounds_violation_mask(
            field, owner, neigh, is_bnd, ref_scales=_REF)
        changed = np.any(out.reshape(n_local, n_sps, 5) != field[local_ids],
                         axis=(1, 2))
        np.testing.assert_array_equal(
            changed, global_mask[local_ids],
            err_msg="被施加滤波的单元集合应当恰好是全局掩码标记的那些")

    def test_halo_extend_rejected_for_persson(self):
        """persson 是纯单元局部判据，给它 halo_extend 必须报错而不是忽略。"""
        F = np.eye(4) * 0.5
        with pytest.raises(ValueError, match="halo_extend"):
            build_sensor_gated_filter_func_arrays(
                10, 4, 1, F, F, cell_is_prism=np.zeros(10, bool),
                sensor="persson", halo_extend=lambda U: U)


class TestBackendRegistration:
    def test_cpu_mpi_is_registered_as_wired(self):
        assert "cpu-mpi" in _SENSOR_MODE_SUPPORTED_BACKENDS

    def test_cpu_mpi_no_longer_falls_back(self, monkeypatch):
        monkeypatch.setenv("AFCFD_FILTER_MODE", "sensor")
        assert resolve_filter_mode("cpu-mpi") == "sensor"
        assert resolve_filter_mode("cpu-single") == "sensor"

    def test_all_four_backends_resolve_to_sensor(self, monkeypatch):
        """四条后端全部接线（2026-09-18），显式请求都能满足。"""
        monkeypatch.setenv("AFCFD_FILTER_MODE", "sensor")
        for backend in ("cpu-single", "cpu-mpi", "gpu-single", "gpu-mpi"):
            assert resolve_filter_mode(backend) == "sensor", backend

    def test_unknown_backend_still_raises(self, monkeypatch):
        """未知后端名仍必须报错——退档分支删除后这是唯一的行为。"""
        monkeypatch.setenv("AFCFD_FILTER_MODE", "sensor")
        with pytest.raises(NotImplementedError, match="gpu-rocm"):
            resolve_filter_mode("gpu-rocm")
