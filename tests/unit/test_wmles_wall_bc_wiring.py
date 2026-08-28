"""Unit tests for the #9 fix: WMLES wall boundary condition double-counting.

真实 bug（V2.0 专家组盲审第4轮，2026-08-28，"WMLES 假滑移边界"）：此前
WALL 边界的 ghost state 恒用严格无滑移镜像构造（is_no_slip=True），BR1/LDG
粘性通量因此已经从解析速度梯度算出一个虚假壁面剪应力，
`solver_helpers.compute_wmles_wall_stress_correction` 算出的 tau_w 又作为
额外动量源项叠加在其上，造成双重计权。按 WMLES 壁面模型文献标准做法
（tau_w 应该"取代"而非"叠加"解析梯度剪应力，见
`fr_solver/boundary.py::build_boundary_ghost_provider` 文档引用的
Kawai & Larsson 团队页面与 Kang et al. 2024 arXiv:2405.15899）修复：
WMLES 激活时 WALL 组的 ghost state 改用 is_no_slip=False。

这里只钉住 `build_boundary_ghost_provider` 的条件分支本身（是否正确按
`solver.wmles_model` 选择 is_no_slip 取值），不重新验证
`tag_boundary_groups_for_mesh` 的边界匹配几何逻辑（那部分有自己的测试），
因此用 mock 替换掉它，避免为这一处条件分支搭建一整套真实网格。
"""

from unittest.mock import patch

import numpy as np

from autoflowcfd.core.fr_solver.boundary import build_boundary_ghost_provider


class _FakeMesh:
    def __init__(self):
        self.face_connectivity = object()  # 只需非 None，几何细节由下面的 mock 绕过
        self.boundary_groups = {"wall_group": np.array([0])}
        self.boundary_bc_types = {"wall_group": "WALL"}
        self.boundary_surface_mesh = None


class _FakeSolver:
    def __init__(self, wmles_model):
        self.mesh = _FakeMesh()
        self.freestream = {"rho_inf": 1.225, "vel_inf": 33.33, "p_inf": 101325.0}
        self.turb_model_name = "WMLES" if wmles_model is not None else "SST"
        self.wmles_model = wmles_model


def _build_provider(wmles_model):
    solver = _FakeSolver(wmles_model)
    group_code = np.array([0, 0, -1, -1], dtype=np.int32)
    name_to_code = {"wall_group": 0}
    with patch(
        "autoflowcfd.core.fr_solver.boundary.tag_boundary_groups_for_mesh",
        return_value=(group_code, name_to_code),
    ):
        return build_boundary_ghost_provider(solver, bc_overrides={})


class TestWmlesWallBcNoDoubleCounting:
    def test_wall_is_no_slip_false_when_wmles_active(self):
        """WMLES 激活时：WALL 组必须改用 is_no_slip=False（与 tau_w 修正
        配套，避免叠加在虚假的无滑移解析剪应力上）。"""
        provider = _build_provider(wmles_model=object())
        config = provider.code_to_config[0]
        assert config["type"] == "WALL"
        assert config["is_no_slip"] is False

    def test_wall_is_no_slip_true_when_wmles_inactive(self):
        """非 WMLES 场景（SST/DDES/IDDES/LES/NONE）：WALL 组行为必须保持
        此前验证过的严格无滑移不变，这是一个纯粹的 WMLES 专属修复，不能
        影响其余湍流模型的壁面处理。"""
        provider = _build_provider(wmles_model=None)
        config = provider.code_to_config[0]
        assert config["type"] == "WALL"
        assert config["is_no_slip"] is True
