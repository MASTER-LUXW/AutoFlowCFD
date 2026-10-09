"""2026-08-23 在 79.1 万单元 cube_demo 的真实续算排查中发现的真实缺陷的回归
测试：网格生成器把每个棱柱四边形侧面三角化成 2 条面连接记录
（face_connectivity.py）。在"逐通量点精确法向"修复
（face_flux_points/exact_normal.py）之前，`true_normal`/`true_area_weight`
是每条记录自己真实的三角化半面几何，两条记录自然加成整张四边形的面积/
法向。修复之后，`true_normal`/`true_area_weight` 变成
`(owner_cell, owner_axis, owner_side)` 的纯函数——一张四边形的 2 条记录现在
带着*相同的整面*值，而不是真实的半面。

`inviscid_p0.py` 的 P0 有限体积核不加过滤地处理每条面记录（有意为之，为了
在*旧*做法下保持封闭性）——在新做法下这会把约 100% 的棱柱四边形侧面的
通量/面积静默地算两遍（不只是约 5% 真正跨两个不同四面体邻居的多源面）。
真实网格上的直接测量：单元 17790（一个边界层棱柱）的封闭面法向和相对误差
6.76%（任何有效的封闭单元，面积加权的面法向之和应严格为零），单这一项就
定量解释了约 100% 的、此前无法解释的 1.6e7 量级的伪动量残差（把全部速度
清零后残差不变得到确认——它纯粹是绝对压力 * 未封闭法向的伪影，与低马赫/
CFL 刚性无关）。

修复（`inviscid_p0.py::_extract_p0_face_geometry`）：对每个
(owner_cell, owner_cube_face) / (neighbor_cell, neighbor_cube_face) 组里
有 >=2 条记录的情形——全部记录共享同一个真实邻居（或 owner）时，只保留
主记录精确的整面值，其余记录面积清零（重复、不是多源——两条都加会算两遍）。
记录指向真正不同的真实单元（真多源，约 5%）时，每条记录都退回
`face_connectivity` 原始的三角化半面 `normal`/`area`（P0 每面只有 1 个
通量点，无法像 P>=1 的核那样经 `nb_extra_mat`/`ow_extra_mat` 在通量点层面
混合多个真实邻居——退回真实的三角化半面是唯一几何上正确的做法，并且自然
恰好加成整面一次）。
"""

from types import SimpleNamespace

import numpy as np

from autoflowcfd.core.fr_residual.inviscid_p0 import _extract_p0_face_geometry
from autoflowcfd.fr.face_flux_points.data import _KernelFaceData


def _make_ffp(n_faces, true_normal, true_area_weight, owner_is_primary,
              neighbor_is_primary, owner_groups, neighbor_groups):
    return _KernelFaceData(
        n_faces=n_faces, n_fp=1,
        true_normal=true_normal, true_area_weight=true_area_weight,
        owner_is_primary=owner_is_primary, neighbor_is_primary=neighbor_is_primary,
        _owner_groups=owner_groups, _neighbor_groups=neighbor_groups,
    )


class TestDuplicateRecordsSameNeighborAreDeduplicated:
    def test_only_primary_record_keeps_area_rest_zeroed(self):
        # f0, f1：同一个 (owner=0, code) 组，**都**指向 neighbor=1（约 95% 的
        # "重复、不是多源"情形）——两条记录目前带着完全相同的整张四边形的
        # true_normal/true_area_weight（正是 Part 1 现在产生的），必须去重。
        n_faces = 2
        true_normal = np.tile(np.array([1.0, 0.0, 0.0]), (n_faces, 1, 1))
        true_area_weight = np.full((n_faces, 1), 5.0)
        owner_is_primary = np.array([True, False])
        neighbor_is_primary = np.array([True, True])
        owner_groups = {(0, 7): [0, 1]}
        neighbor_groups = {}

        ffp = _make_ffp(n_faces, true_normal, true_area_weight,
                         owner_is_primary, neighbor_is_primary,
                         owner_groups, neighbor_groups)
        fc = SimpleNamespace(
            normal=np.zeros((n_faces, 3)), area=np.zeros(n_faces),
            owner_cell=np.array([0, 0]), neighbor_cell=np.array([1, 1]),
            is_boundary=np.array([False, False]),
        )

        unit_normals, area_weights = _extract_p0_face_geometry(ffp, fc, n_faces)

        assert area_weights[0] == 5.0  # 主记录保留精确的整面值
        assert area_weights[1] == 0.0  # duplicate zeroed, not double-counted
        np.testing.assert_array_equal(unit_normals[0], [1.0, 0.0, 0.0])

    def test_boundary_duplicate_records_are_deduplicated(self):
        """边界面从不出现在 _owner_groups/_neighbor_groups 里（build_face_flux_points
        把它们单独分到 boundary_owner_groups，不存到 _KernelFaceData 上），但它们的
        owner_is_primary 仍然设置正确——去重必须靠 is_boundary & ~owner_is_primary
        这个掩码抓到这种情形，与是否在组字典里无关。
        """
        n_faces = 2
        true_normal = np.tile(np.array([0.0, 1.0, 0.0]), (n_faces, 1, 1))
        true_area_weight = np.full((n_faces, 1), 3.0)
        owner_is_primary = np.array([True, False])
        neighbor_is_primary = np.array([True, True])

        ffp = _make_ffp(n_faces, true_normal, true_area_weight,
                         owner_is_primary, neighbor_is_primary,
                         owner_groups={}, neighbor_groups={})
        fc = SimpleNamespace(
            normal=np.zeros((n_faces, 3)), area=np.zeros(n_faces),
            owner_cell=np.array([2, 2]), neighbor_cell=np.array([-1, -1]),
            is_boundary=np.array([True, True]),
        )

        _, area_weights = _extract_p0_face_geometry(ffp, fc, n_faces)

        assert area_weights[0] == 3.0
        assert area_weights[1] == 0.0


class TestGenuineMultiSourceFallsBackToTriangulatedHalfFace:
    def test_both_records_keep_area_using_old_triangulated_geometry(self):
        # f0, f1：同一个 (owner=0, code) 组，但指向**两个不同**的真实邻居 (2, 3)
        # ——真正的多源（约 5% 的情形）。两条记录都必须保持有效（不能丢掉任何一个
        # 真实邻居的贡献），但必须用 face_connectivity 原始的三角化半面法向/面积，
        # **不能**用新的、精确但两条相同的整张四边形 true_normal/true_area_weight
        # （两条都用整面值会像重复情形一样把面积算两遍，只是对着两个不同的邻居
        # 而不是同一个）。
        n_faces = 2
        true_normal = np.tile(np.array([1.0, 0.0, 0.0]), (n_faces, 1, 1))
        true_area_weight = np.full((n_faces, 1), 5.0)  # 整张四边形的值，不能原样使用
        owner_is_primary = np.array([True, False])
        neighbor_is_primary = np.array([True, True])
        owner_groups = {(0, 7): [0, 1]}

        ffp = _make_ffp(n_faces, true_normal, true_area_weight,
                         owner_is_primary, neighbor_is_primary,
                         owner_groups, neighbor_groups={})
        old_normal = np.array([[0.9, 0.1, 0.0], [0.8, -0.2, 0.0]])
        old_area = np.array([2.1, 2.4])  # 真实的三角化半面，加起来是整张四边形
        fc = SimpleNamespace(
            normal=old_normal, area=old_area,
            owner_cell=np.array([0, 0]), neighbor_cell=np.array([2, 3]),
            is_boundary=np.array([False, False]),
        )

        unit_normals, area_weights = _extract_p0_face_geometry(ffp, fc, n_faces)

        np.testing.assert_array_equal(area_weights, old_area)
        np.testing.assert_array_equal(unit_normals, old_normal)


class TestUngroupedFacesAreUntouched:
    def test_plain_tet_tet_face_keeps_exact_value(self):
        n_faces = 1
        true_normal = np.array([[[0.0, 0.0, 1.0]]])
        true_area_weight = np.array([[7.0]])
        owner_is_primary = np.array([True])
        neighbor_is_primary = np.array([True])

        ffp = _make_ffp(n_faces, true_normal, true_area_weight,
                         owner_is_primary, neighbor_is_primary,
                         owner_groups={}, neighbor_groups={})
        fc = SimpleNamespace(
            normal=np.zeros((n_faces, 3)), area=np.zeros(n_faces),
            owner_cell=np.array([9]), neighbor_cell=np.array([10]),
            is_boundary=np.array([False]),
        )

        unit_normals, area_weights = _extract_p0_face_geometry(ffp, fc, n_faces)

        assert area_weights[0] == 7.0
        np.testing.assert_array_equal(unit_normals[0], [0.0, 0.0, 1.0])
