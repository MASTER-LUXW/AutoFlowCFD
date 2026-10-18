"""AutoFlowCFD V2.0 - native 四面体 + WMLES：坍缩坐标外插表不得复活。

2026-09-02 曾查出 WMLES 壁面剪应力修正对四面体恒用坍缩坐标专属外插表（原生面上的 owner_axis/owner_side
是哑值，查表或 KeyError、或取到语义不对的矩阵）。现行 WMLES（`core/turbulence/wall_model/`）只用原生面外插与
提升；四面体 owner 的匹配点与残差修正由 `test_wmles_wall_model.py::test_native_tet_wall_face` 覆盖。
"""

from autoflowcfd.fr.operators import generate_fr_operators


class TestNativeTetWmlesBoundaryExtrap:
    def test_the_collapsed_extrap_table_is_gone(self):
        """原「修复前的做法会出错」那条前提验证已随被测对象一并删除。

        它当年逐面证明了：对原生四面体 WALL 面沿用
        `ops.boundary_extrap_tet[(owner_axis, owner_side)]` 外插，两种
        暴露方式都会真的出现 —— 有的面伪键根本不在 6 个合法 (axis,side)
        组合里、直接 `KeyError`；有的面伪键凑巧撞上某个真实键、取到的
        矩阵语义不对，外插结果与正确的原生外插不同。那份证据保留在
        `ProjectFiles/V2.0/19_重大问题修复-湍流模型跨后端逐项排查与完全
        分布式加载补齐.md`。

        2026-09-24 起 `FROperators` 上已经没有 `boundary_extrap_tet` /
        `boundary_extrap_prism` 这两张坍缩外插表（随坍缩 1D 分布机制一并
        删除，见 `fr_residual/inviscid_kernel.py::
        compute_inviscid_interface_correction_kernel` 文档），那条错误做法
        在代码层面已经不可能被写出来。这里把「表确实没了」钉住，取代
        原来那条无法再运行的前提验证。
        """
        ops = generate_fr_operators(2)
        for gone in ("boundary_extrap_tet", "boundary_extrap_prism",
                     "g_left", "g_right", "L_interp"):
            assert not hasattr(ops, gone), (
                f"FROperators 又出现了 {gone} —— 那条坍缩路径已于 "
                f"2026-09-24 删除，重新加回它需要先恢复一条一维张量积"
                f"离散路径（见 ProjectFiles/V2.0/27_...md）")


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
