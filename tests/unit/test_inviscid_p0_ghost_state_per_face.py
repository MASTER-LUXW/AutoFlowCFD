"""AutoFlowCFD V2.0 - P0 幽灵态按面索引回归测试 (#12 复核过程中发现)。

真实 bug 修复（V2.0 专家组盲审第四次评审，2026-08-28）：
`inviscid_p0.py::_precompute_ghost_states` 此前把幽灵态存成按 **owner
单元** 索引的 (n_cells,5) 数组（`Q_ghost[oc] = ...`），`inviscid_p0_
kernel.py::_p0_inviscid_kernel` 也按 `Q_ghost[owner_cell[f]]` 读取——如果
同一个 P0 单元同时挨着 2 个以上边界面（角点单元很常见），后处理的边界
面会把先处理的边界面的幽灵态覆盖掉。用 `DefaultGhostProvider`（ghost=
owner 状态，覆盖谁都一样）时这个问题被完全掩盖，只有真实的 WALL/
INLET/OUTLET 等"幽灵态依赖各面自己法向/位置、因面而异"的场景才会暴露
——本文件用一个按法向区分返回值的合成 ghost_provider，直接命中这个
差异。

修复为按 **面** 索引的 (n_faces,5) 数组，与 GPU 版
`core/gpu/residual/gpu_p0_inviscid.py` 的既有正确做法保持一致。
"""

import numpy as np
import pytest

from autoflowcfd.core.fr_residual.inviscid import (
    _compute_inviscid_residual_fv_p0,
    conserved_to_primitive,
)
from autoflowcfd.core.fr_residual.inviscid_p0 import (
    _extract_p0_face_geometry,
    _precompute_ghost_states,
)
from tests.validation._channel_mesh import build_channel_mesh_prism

RHO_INF, P_INF, U_INF = 1.225, 101325.0, 30.0
GAMMA = 1.4


def _build_mesh():
    return build_channel_mesh_prism(order=0, nx=4, ny=3, nz=2, Lx=2.0, H=1.0, Lz=0.5)


class _NormalEncodingGhostProvider:
    """合成 ghost_provider：返回值只由该面自身法向唯一决定（用密度分量
    编码 round(nx)/round(ny)/round(nz)），不同法向的两个面必然得到不同
    的幽灵密度——足以区分"按面索引"（正确）与"按 owner 单元索引"
    （此前的 bug，后一个面会覆盖前一个面的结果）两种实现。"""

    def __call__(self, face_idx, Q_owner_fp, true_normal):
        n = true_normal[0]
        code = 1000.0 + round(n[0]) * 10.0 + round(n[1]) * 100.0 + round(n[2]) * 1000.0
        ghost = Q_owner_fp.copy()
        ghost[:, 0] = code
        return ghost


class TestGhostStateIndexedByFaceNotByCell:
    def test_corner_cell_has_multiple_distinct_boundary_faces(self):
        """先确认测试前提成立：这个网格里确实存在一个 P0 单元同时挨着
        2 个以上、法向互不相同的边界面（否则本测试无法区分 bug 与修复）。
        """
        mesh = _build_mesh()
        fc = mesh.face_connectivity
        boundary_idx = np.nonzero(fc.is_boundary)[0]
        owners = fc.owner_cell[boundary_idx]
        counts = np.bincount(owners)
        corner_cells = np.nonzero(counts >= 2)[0]
        assert len(corner_cells) > 0, "测试网格应至少有一个挨着 >=2 个边界面的角点单元"

    def test_ghost_state_per_face_matches_own_normal_not_overwritten(self):
        mesh = _build_mesh()
        fc = mesh.face_connectivity
        ffp_list = mesh.face_flux_points
        n_faces = fc.n_faces
        n_cells = mesh.n_cells

        boundary_idx = np.nonzero(fc.is_boundary)[0]
        owners = fc.owner_cell[boundary_idx]
        counts = np.bincount(owners, minlength=n_cells)
        corner_cell = int(np.argmax(counts))
        corner_faces = boundary_idx[owners == corner_cell]
        assert len(corner_faces) >= 2

        Q_all = np.tile(
            np.array([RHO_INF, U_INF, 0.0, 0.0, P_INF]), (n_cells, 1)
        )
        unit_normals, _ = _extract_p0_face_geometry(ffp_list, fc, n_faces)

        provider = _NormalEncodingGhostProvider()
        Q_ghost = _precompute_ghost_states(ffp_list, fc, provider, Q_all, n_faces, unit_normals)

        # 修复后：每个边界面自己的 Q_ghost[f,0] 必须等于按*该面自己*法向
        # 编码出的值，不能等于按角点单元其它某个边界面法向编码出的值。
        expected_per_face = {}
        for f in corner_faces:
            n = unit_normals[f]
            expected_per_face[f] = 1000.0 + round(n[0]) * 10.0 + round(n[1]) * 100.0 + round(n[2]) * 1000.0

        distinct_expected = set(expected_per_face.values())
        assert len(distinct_expected) >= 2, (
            "角点单元的这些边界面法向应互不相同，否则无法区分按面/按单元索引"
        )

        for f, expected in expected_per_face.items():
            assert Q_ghost[f, 0] == pytest.approx(expected), (
                f"面 {f}（owner={corner_cell}）的幽灵态密度={Q_ghost[f, 0]} 与"
                f"该面自身法向编码出的期望值 {expected} 不符——疑似被同一个"
                f"owner 单元的另一个边界面覆盖"
            )

    def test_negative_control_owner_indexed_array_would_collide(self):
        """反向对照：如果幽灵态仍按 owner 单元索引成 (n_cells,5)（修复前
        的 bug），角点单元的多个边界面会被迫共享同一个值——用最朴素的
        "最后写入的赢"复现旧行为，证明它确实与按面索引的正确结果不同。
        """
        mesh = _build_mesh()
        fc = mesh.face_connectivity
        ffp_list = mesh.face_flux_points
        n_faces = fc.n_faces
        n_cells = mesh.n_cells

        boundary_idx = np.nonzero(fc.is_boundary)[0]
        owners = fc.owner_cell[boundary_idx]
        counts = np.bincount(owners, minlength=n_cells)
        corner_cell = int(np.argmax(counts))
        corner_faces = boundary_idx[owners == corner_cell]

        Q_all = np.tile(
            np.array([RHO_INF, U_INF, 0.0, 0.0, P_INF]), (n_cells, 1)
        )
        unit_normals, _ = _extract_p0_face_geometry(ffp_list, fc, n_faces)
        provider = _NormalEncodingGhostProvider()

        Q_ghost_correct = _precompute_ghost_states(ffp_list, fc, provider, Q_all, n_faces, unit_normals)

        # 复现修复前的"按 owner 单元索引、后写覆盖先写"行为。
        Q_ghost_owner_indexed = np.zeros((n_cells, 5))
        for f in np.nonzero(fc.is_boundary)[0]:
            oc = int(fc.owner_cell[f])
            Q_owner_fp = Q_all[oc:oc + 1]
            Q_ghost_owner_indexed[oc] = provider(f, Q_owner_fp, unit_normals[f:f + 1])[0]

        mismatches = [
            f for f in corner_faces
            if Q_ghost_correct[f, 0] != pytest.approx(Q_ghost_owner_indexed[corner_cell, 0])
        ]
        assert len(mismatches) > 0, (
            "按面索引的正确结果应至少在一个面上与按 owner 单元索引（最后写入"
            "者获胜）的复现结果不同——否则这个反向对照没有真正区分两种实现"
        )


class TestFullResidualPipelineWithFaceVaryingGhostProvider:
    def test_residual_runs_end_to_end_with_non_default_ghost_provider(self):
        """端到端冒烟：完整 P0 残差路径（含 numba kernel）在真实（非
        Default）ghost_provider 下能正常跑完，且角点单元的残差确实
        感知到了每个边界面各自不同的幽灵态（不是退化成好像只有一个
        边界条件在生效）。"""
        mesh = _build_mesh()
        n_cells = mesh.n_cells
        U = np.zeros((n_cells, 1, 5))
        U[:, 0, 0] = RHO_INF
        U[:, 0, 1] = RHO_INF * U_INF
        U[:, 0, 4] = P_INF / (GAMMA - 1.0) + 0.5 * RHO_INF * U_INF**2

        provider = _NormalEncodingGhostProvider()
        residual = _compute_inviscid_residual_fv_p0(U, mesh, boundary_ghost_provider=provider)

        assert np.all(np.isfinite(residual))
        assert np.max(np.abs(residual)) > 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
