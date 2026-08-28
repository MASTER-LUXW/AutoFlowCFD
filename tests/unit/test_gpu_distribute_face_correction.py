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

import numpy as np
import pytest

from autoflowcfd.fr.operators import generate_fr_operators
from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry
from autoflowcfd.core.fr_residual.inviscid_kernel import _distribute_point
from autoflowcfd.core.gpu.residual.gpu_inviscid_volume import distribute_face_correction_to_sps
from tests.unit.test_fr_residual_inviscid import _build_synthetic_mixed_mesh


@pytest.fixture(scope="module")
def flat_face_p2():
    mesh = _build_synthetic_mixed_mesh(order=2)
    ops = generate_fr_operators(order=2)
    return get_flat_face_geometry(mesh, ops)


class TestDistributeFaceCorrectionMatchesCpuGroundTruth:
    def test_g_left_g_right_shape_is_n1d_not_per_face(self, flat_face_p2):
        """先确认真实数据形状：g_left/g_right 是 (n1d,)，n_faces 远大于
        n1d——这正是旧代码用面索引去索引它们会越界的根本原因。"""
        flat = flat_face_p2
        assert flat.g_left.shape == (3,)  # P2: n1d=3
        assert flat.n_faces > flat.g_left.shape[0]

    def test_5var_case_matches_cpu_distribute_point(self, flat_face_p2):
        flat = flat_face_p2
        n_faces, n_fp, n_sps, n_vars = flat.n_faces, flat.n_fp, flat.n_sps, 5

        rng = np.random.default_rng(0)
        jump = rng.standard_normal((n_faces, n_fp, n_vars))
        axis = flat.owner_axis
        side = flat.owner_side

        expected = np.zeros((n_faces, n_sps, n_vars))
        for f in range(n_faces):
            ax = axis[f]
            g_prime = flat.g_right if side[f] > 0 else flat.g_left
            fp_of_sp = flat.dist_fp_of_sp[ax]
            axis_coord = flat.dist_axis_coord_of_sp[ax]
            expected[f] = _distribute_point(jump[f], fp_of_sp, axis_coord, g_prime)

        actual = distribute_face_correction_to_sps(
            np, jump, axis, side, flat.dist_fp_of_sp, flat.dist_axis_coord_of_sp,
            flat.g_left, flat.g_right,
        )
        np.testing.assert_array_equal(actual, expected)

    def test_scalar_case_matches_cpu_distribute_point_scalar(self, flat_face_p2):
        """标量（无 trailing V 轴）版本，供 gpu_scalar_transport.py 复用同一个
        函数时验证。"""
        flat = flat_face_p2
        n_faces, n_fp, n_sps = flat.n_faces, flat.n_fp, flat.n_sps

        rng = np.random.default_rng(1)
        correction_fp = rng.standard_normal((n_faces, n_fp))
        axis = flat.owner_axis
        side = flat.owner_side

        expected = np.zeros((n_faces, n_sps))
        for f in range(n_faces):
            ax = axis[f]
            g_prime = flat.g_right if side[f] > 0 else flat.g_left
            fp_of_sp = flat.dist_fp_of_sp[ax]
            axis_coord = flat.dist_axis_coord_of_sp[ax]
            for s in range(n_sps):
                expected[f, s] = g_prime[axis_coord[s]] * correction_fp[f, fp_of_sp[s]]

        actual = distribute_face_correction_to_sps(
            np, correction_fp, axis, side, flat.dist_fp_of_sp, flat.dist_axis_coord_of_sp,
            flat.g_left, flat.g_right,
        )
        np.testing.assert_array_equal(actual, expected)

    def test_negative_control_wrong_axis_choice_would_diverge(self, flat_face_p2):
        """反向对照：用 neighbor_axis（而不是各面自己真正应该用的 owner_axis）
        去分配，数值上应与正确结果不同——证明本测试真的在检验对应关系，
        不是即便传错 axis 也凑巧全部相等的退化情形。"""
        flat = flat_face_p2
        rng = np.random.default_rng(2)
        n_faces, n_fp = flat.n_faces, flat.n_fp
        correction_fp = rng.standard_normal((n_faces, n_fp))

        correct = distribute_face_correction_to_sps(
            np, correction_fp, flat.owner_axis, flat.owner_side,
            flat.dist_fp_of_sp, flat.dist_axis_coord_of_sp, flat.g_left, flat.g_right,
        )
        wrong = distribute_face_correction_to_sps(
            np, correction_fp, flat.neighbor_axis, flat.owner_side,
            flat.dist_fp_of_sp, flat.dist_axis_coord_of_sp, flat.g_left, flat.g_right,
        )
        # 只要存在至少一个内部面（owner_axis != neighbor_axis 是常见但非
        # 保证的几何事实），两者就应该产生不同的结果；用 any 而不是要求
        # 逐位置都不同。
        assert not np.array_equal(correct, wrong)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
