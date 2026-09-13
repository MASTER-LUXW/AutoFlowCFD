"""真实 bug 回归测试（2026-09-12）：`compute_scalar_convection_residual`
的界面上风校正此前对 owner/neighbor 两侧使用同一个相对 phi_owner_fp
算出的跳变量（`raw_jump_fp = mass_flux*(phi_upwind-phi_owner_fp)`），
只是分配时符号相反——但正确的 DG/FR 做法（见 core/fr_residual/
inviscid_kernel.py 里 AUSM+up 通量分别用 owner/neighbor 各自状态独立
算出 jump_owner/jump_neighbor 的既有正确实现）要求 neighbor 侧必须用
相对 phi_neighbor_fp 的独立跳变量。

两者只在 owner 恰好是该面下风侧（mass_flux<0，phi_upwind==phi_neighbor_fp）
时才可能凑巧一致；owner 是上风侧时（mass_flux>=0，phi_upwind==
phi_owner_fp）旧代码给 neighbor 的跳变量恒为 0——等价于下风侧的
neighbor（真实网格里占全部内部面的一半）完全收不到这个面本该有的
对流稀释/浓缩，是 cube_demo 791,492 单元真实网格 P0 阶段长程续算
（iter 2600->3100）k_max 从 0.99 复合增长到 2.05（omega_max 同步
11577->25626，约1.16~1.2x/100步）的一个真实、独立的根因。

修复：neighbor 侧改用独立的 `raw_jump_fp_neighbor = mass_flux*
(phi_upwind - phi_neighbor_fp)`。

诚实说明该修复的实际效果范围（2026-09-12 真实网格验证结果）：单独
这一处修复用真实生产 checkpoint（iter 3100）验证 500 步，k_max 复合
增长速率减半（约从 1.16~1.2x/100步 降到 ~1.08x/100步），但没有完全
停止——P0 阶段体积项架构上恒为零（1x1 零微分矩阵），"跳变量=上风值-
自身面值"这套 DG 差额公式因此始终缺一部分本该由 volume_term 提供的
贡献，这是本次排查发现但尚未修复的第二个、更深的缺口（曾尝试照搬
core/fr_residual/inviscid_p0.py 的直接有限体积公式，但用合成网格
决定性证伪：会对局部质量不守恒残差重新引入敏感性，已撤销，见
transport.py 里保留的完整撤销记录）。本文件只覆盖已经确认修复、
不会引入新问题的这一处。
"""
from types import SimpleNamespace

import numpy as np
import pytest

from autoflowcfd.grid.high_order.high_order_mesh import HighOrderMesh


class _MockNodes:
    def __init__(self, coords):
        self._coords = coords

    def get_coordinates(self):
        return self._coords


class _MockCells:
    def __init__(self, connectivity):
        self.connectivity = connectivity


def _build_synthetic_mixed_mesh(order: int):
    """2 个共享面的四面体 + 2 个共享侧面的棱柱，与 test_turbulence_
    transport.py::_build_synthetic_mixed_mesh 完全相同的几何构造。"""
    nodes = np.array(
        [
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1],
            [10, 0, 0], [11, 0, 0], [10, 1, 0], [10, 0, 1], [11, 0, 1], [10, 1, 1],
        ],
        dtype=float,
    )
    tet_conn = np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int32)
    nodes = np.vstack([nodes, [[9, -1, 0], [9, -1, 1]]])
    prism_conn = np.array(
        [
            [5, 6, 7, 8, 9, 10],
            [5, 7, 11, 8, 10, 12],
        ],
        dtype=np.int32,
    )
    mock_volume = SimpleNamespace(
        cell_count=len(tet_conn) + len(prism_conn),
        nodes=_MockNodes(nodes),
        cells=_MockCells(tet_conn),
        prism_cells=_MockCells(prism_conn),
    )
    mesh = HighOrderMesh(order=order)
    mesh.load_from_volume_mesh(mock_volume)
    return mesh


class TestScalarConvectionOwnerNeighborAsymmetry:
    """最小 2-四面体复现：donor(低值,upstream,owner) -> receiver(高值,
    downstream,neighbor)，纯对流、无生成项、无扩散（P0）。receiver 已经
    是局部最大值，一个满足极值原理的迎风格式绝不应该让它继续增长——
    正确行为是被上游低值稀释（conv_k < 0），不应该恒为 0。"""

    def _setup(self):
        from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry

        mesh = _build_synthetic_mixed_mesh(order=0)
        ops = mesh.operators
        n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell
        n_prism = mesh.n_prism_cells
        tet0, tet1 = n_prism, n_prism + 1  # donor, receiver

        flat = get_flat_face_geometry(mesh, ops)
        owner, neighbor = flat.owner_cell, flat.neighbor_cell
        shared = [f for f in range(flat.n_faces)
                  if {owner[f], neighbor[f]} == {tet0, tet1}]
        assert len(shared) == 1, "两个 tet 应该恰好共享 1 个内部面"
        f0 = shared[0]
        assert owner[f0] == tet0 and neighbor[f0] == tet1, (
            "本测试要求 tet0 是该面的 owner（donor/upstream），"
            "tet1 是 neighbor（receiver/downstream）——与真实复现场景"
            "（owner 恰好是上风侧）完全对应"
        )

        flow_dir = flat.true_normal[f0, 0]
        flow_dir = flow_dir / np.linalg.norm(flow_dir)

        rho = np.full((n_cells, n_sps), 1.2)
        k_field = np.full((n_cells, n_sps), 0.1)
        k_field[tet1] = 2.0  # receiver 已经是全场最大值
        velocity = np.zeros((n_cells, n_sps, 3))
        velocity[:, :, :] = flow_dir * 10.0  # 均匀速度场：donor(owner) -> receiver(neighbor)

        return mesh, ops, flat, tet0, tet1, rho, k_field, velocity

    def test_receiver_is_diluted_not_frozen_at_zero(self):
        from autoflowcfd.core.turbulence.transport import compute_scalar_convection_residual

        mesh, ops, flat, tet0, tet1, rho, k_field, velocity = self._setup()

        conv_k = compute_scalar_convection_residual(
            k_field, rho, velocity, mesh, ops, flat_face_override=flat,
        )

        # 真实 bug 的直接数值证据：修复前这里恒为 0.0（owner 是上风侧时
        # neighbor 收不到任何校正）。修复后必须是负值——receiver 被上游
        # 低值(0.1)稀释，不能继续停留在自己的高值(2.0)不变。
        assert conv_k[tet1, 0] < -1e-6, (
            f"receiver(tet1) 的对流残差应该 < 0（被上游低值稀释），"
            f"实际={conv_k[tet1, 0]}——如果恰好是 0.0，说明 owner/neighbor "
            f"跳变量不对称的 bug 又出现了"
        )
        # donor 自己没有其他真实入流（边界面是纯 Neumann），保持不变。
        assert conv_k[tet0, 0] == pytest.approx(0.0, abs=1e-10)

    def test_donor_referenced_and_receiver_referenced_jumps_differ(self):
        """直接验证两侧不能是同一个跳变量：受体应该相对自己的高值(2.0)
        计算跳变，不是相对施主的低值(0.1)——两者数值上必须不同。"""
        from autoflowcfd.core.turbulence.transport import compute_scalar_convection_residual

        mesh, ops, flat, tet0, tet1, rho, k_field, velocity = self._setup()

        conv_k = compute_scalar_convection_residual(
            k_field, rho, velocity, mesh, ops, flat_face_override=flat,
        )
        # 用体积（det_jac 正比于体积）反推：如果两侧用的是同一个 raw_jump
        # （旧 bug 行为），conv_k[tet1] 必然精确为 0；现在必须显著非零，
        # 且量级应该与 mass_flux*(phi_upwind-phi_neighbor) 一致（不是 0）。
        assert abs(conv_k[tet1, 0]) > 1.0, (
            "receiver 的对流残差量级过小，怀疑仍在用 owner 参照的跳变量"
        )
