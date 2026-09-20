"""AutoFlowCFD V2.0 - GPU 面校正分配 gather 机制回归测试。

真实 bug 修复（V2.0 专家组盲审第四轮，2026-08-28）：`gpu_inviscid.py::
_compute_interface_correction_gpu`/`gpu_viscous.py::
_compute_viscous_interface_correction_gpu` 此前用 `ff.g_left[idx_o]`/
`ff.g_right[idx_o]`（按面索引去索引 g_left/g_right）再 `cp.matmul` 分配
——但 g_left/g_right 真实形状是 `(n1d,)`（本文件下面用真实网格实测
确认，P2 阶数下是 (3,)），用可达 `n_faces-1`（真实网格上远超 n1d）的
面索引去索引这个长度仅 n1d 的数组，会在真实 CUDA 硬件上第一次执行时
直接 IndexError——本机没有 CuPy，这处此前从未被实际执行验证过。

修复：`gpu_inviscid_volume.py::distribute_face_correction_to_sps` 改用
`dist_fp_of_sp`/`dist_axis_coord_of_sp` gather，与 CPU numba kernel
`fr_residual/inviscid_kernel.py::_distribute_point` 完全同一套机制。

本测试不需要 CuPy/真实 GPU：`distribute_face_correction_to_sps` 的实现
只用了 `cp.where`/`cp.arange`/fancy indexing——numpy 对这几个操作的语义
与 CuPy 完全一致，直接把 `numpy` 模块作为 `cp` 参数传给这个*生产函数
本身*（不是重新实现一份），用真实网格的 `dist_fp_of_sp`/
`dist_axis_coord_of_sp`/`g_left`/`g_right` 数据，对照 CPU 已验证的
`_distribute_point` 做逐面循环的真值比较——数值必须逐位精确相等（同一套
gather 公式，不是近似），不是"都接近零"这种弱判据。
"""

import os

import numpy as np
import pytest

from autoflowcfd.fr.operators import generate_fr_operators
from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry
from autoflowcfd.core.fr_residual.inviscid_kernel import _distribute_point
from autoflowcfd.core.gpu.residual.gpu_inviscid_volume import distribute_face_correction_to_sps
from tests.unit.test_fr_residual_inviscid import _build_synthetic_mixed_mesh


@pytest.fixture(scope="module")
def flat_face_p2():
    """**坍缩棱柱档**下的合成网格面几何（2026-09-20）。

    本文件验证的 `distribute_face_correction_to_sps` 只服务坍缩棱柱面
    （native 面走 `lift_native` DG 提升算子，是另一条路径），所以网格
    必须在坍缩档下构造 —— 环境变量要在**建网格与建算子之前**设好，
    放在类级 fixture 里来不及（module 级 fixture 先于它求值）。
    """
    old = os.environ.get("AFCFD_PRISM_BASIS")
    os.environ["AFCFD_PRISM_BASIS"] = "collapsed"
    try:
        mesh = _build_synthetic_mixed_mesh(order=2)
        ops = generate_fr_operators(order=2)
        return get_flat_face_geometry(mesh, ops)
    finally:
        if old is None:
            os.environ.pop("AFCFD_PRISM_BASIS", None)
        else:
            os.environ["AFCFD_PRISM_BASIS"] = old


class TestDistributeFaceCorrectionMatchesCpuGroundTruth:
    def test_g_left_g_right_shape_is_n1d_not_per_face(self, flat_face_p2):
        """先确认真实数据形状：g_left/g_right 是 (n1d,)，n_faces 远大于
        n1d——这正是旧代码用面索引去索引它们会越界的根本原因。"""
        flat = flat_face_p2
        assert flat.g_left.shape == (3,)  # P2: n1d=3
        assert flat.n_faces > flat.g_left.shape[0]

    def test_5var_case_matches_cpu_distribute_point(self, flat_face_p2):
        """2026-09-03 更正：四面体坍缩坐标基已删除（见 fr/operators.py
        模块文档），合成测试网格默认即为 native——`distribute_face_
        correction_to_sps` 的 1D 修正函数分布本身只对 collapsed（棱柱）
        面有意义（native 面走完全不同的 `lift_native` DG 提升算子，见
        gpu_inviscid.py::_native_or_collapsed_contrib），`owner_axis`
        对 native 面存的是复用槽位的 excluded_vertex（可达 3），不能
        无条件拿去 gather 只有 3 个轴的表——只在棱柱面（
        `owner_cube_face<6`）上验证这个函数，与生产代码里这个函数
        实际只服务棱柱面的事实一致。"""
        flat = flat_face_p2
        n_fp, n_sps, n_vars = flat.n_fp, flat.n_sps, 5
        prism_mask = flat.owner_cube_face < 6
        face_ids = np.nonzero(prism_mask)[0]
        assert len(face_ids) > 0, "合成网格应该至少有棱柱面"

        rng = np.random.default_rng(0)
        jump_full = rng.standard_normal((flat.n_faces, n_fp, n_vars))
        axis = flat.owner_axis[face_ids]
        side = flat.owner_side[face_ids]
        jump = jump_full[face_ids]

        expected = np.zeros((len(face_ids), n_sps, n_vars))
        for i in range(len(face_ids)):
            ax = axis[i]
            g_prime = flat.g_right if side[i] > 0 else flat.g_left
            fp_of_sp = flat.dist_fp_of_sp[ax]
            axis_coord = flat.dist_axis_coord_of_sp[ax]
            expected[i] = _distribute_point(jump[i], fp_of_sp, axis_coord, g_prime)

        actual = distribute_face_correction_to_sps(
            np, jump, axis, side, flat.dist_fp_of_sp, flat.dist_axis_coord_of_sp,
            flat.g_left, flat.g_right,
        )
        np.testing.assert_array_equal(actual, expected)

    def test_scalar_case_matches_cpu_distribute_point_scalar(self, flat_face_p2):
        """标量（无 trailing V 轴）版本，供 gpu_scalar_transport.py 复用同一个
        函数时验证。2026-09-03 更正：同上，只在棱柱面上验证。"""
        flat = flat_face_p2
        n_fp, n_sps = flat.n_fp, flat.n_sps
        prism_mask = flat.owner_cube_face < 6
        face_ids = np.nonzero(prism_mask)[0]

        rng = np.random.default_rng(1)
        correction_fp_full = rng.standard_normal((flat.n_faces, n_fp))
        axis = flat.owner_axis[face_ids]
        side = flat.owner_side[face_ids]
        correction_fp = correction_fp_full[face_ids]

        expected = np.zeros((len(face_ids), n_sps))
        for i in range(len(face_ids)):
            ax = axis[i]
            g_prime = flat.g_right if side[i] > 0 else flat.g_left
            fp_of_sp = flat.dist_fp_of_sp[ax]
            axis_coord = flat.dist_axis_coord_of_sp[ax]
            for s in range(n_sps):
                expected[i, s] = g_prime[axis_coord[s]] * correction_fp[i, fp_of_sp[s]]

        actual = distribute_face_correction_to_sps(
            np, correction_fp, axis, side, flat.dist_fp_of_sp, flat.dist_axis_coord_of_sp,
            flat.g_left, flat.g_right,
        )
        np.testing.assert_array_equal(actual, expected)

    def test_negative_control_wrong_axis_choice_would_diverge(self, flat_face_p2):
        """反向对照：用 neighbor_axis（而不是各面自己真正应该用的 owner_axis）
        去分配，数值上应与正确结果不同——证明本测试真的在检验对应关系，
        不是即便传错 axis 也凑巧全部相等的退化情形。2026-09-03 更正：
        只在两侧都是棱柱面的面上验证（同上，native 面的 axis 槽位不是
        这个函数的合法输入）。"""
        flat = flat_face_p2
        rng = np.random.default_rng(2)
        n_fp = flat.n_fp
        both_prism = (flat.owner_cube_face < 6) & (flat.neighbor_cube_face < 6) & (flat.neighbor_cube_face >= 0)
        face_ids = np.nonzero(both_prism)[0]
        assert len(face_ids) > 0, "合成网格应该至少有一个棱柱-棱柱内部面"
        correction_fp = rng.standard_normal((len(face_ids), n_fp))

        correct = distribute_face_correction_to_sps(
            np, correction_fp, flat.owner_axis[face_ids], flat.owner_side[face_ids],
            flat.dist_fp_of_sp, flat.dist_axis_coord_of_sp, flat.g_left, flat.g_right,
        )
        wrong = distribute_face_correction_to_sps(
            np, correction_fp, flat.neighbor_axis[face_ids], flat.owner_side[face_ids],
            flat.dist_fp_of_sp, flat.dist_axis_coord_of_sp, flat.g_left, flat.g_right,
        )
        # 只要存在至少一个内部面（owner_axis != neighbor_axis 是常见但非
        # 保证的几何事实），两者就应该产生不同的结果；用 any 而不是要求
        # 逐位置都不同。
        assert not np.array_equal(correct, wrong)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
