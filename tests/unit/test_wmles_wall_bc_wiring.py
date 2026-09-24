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
from tests.unit.test_fr_residual_inviscid import _build_synthetic_mixed_mesh


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
    # patch 目标必须是**真正持有调用点**的子模块：`boundary` 2026-09-24
    # 拆成子包，调用点在 `boundary/ghost.py` 里。patch 包属性不会改变子模块
    # 内部的调用 —— 那样 patch 静默失效而测试照样"通过"
    # （ProjectFiles/V2.0/29 三·五 (3)）。
    with patch(
        "autoflowcfd.core.fr_solver.boundary.ghost.tag_boundary_groups_for_mesh",
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


class TestRealFRSolverWmlesConstructionOrder:
    """收尾集成测试：真正构造 `FRSolver(..., turb_model_name="wmles")`，
    不像上面两个测试那样用手搭的 `_FakeSolver`（提前把 `wmles_model`
    设好再调用 `build_boundary_ghost_provider`）。

    真实 bug（2026-09-02，排查分布式 WMLES 支持时发现，与分布式本身
    无关）：`FRSolver.__init__` 此前先构造 `self.boundary_ghost_
    provider`（第 3 步），几步之后才真正给 `self.wmles_model` 赋值
    （第 5 步，`_init_turbulence_models`）——`build_boundary_ghost_
    provider` 用 `getattr(solver,"wmles_model",None) is None` 判断是否
    要切换 is_no_slip，但此时这个属性根本还不存在，`getattr` 安全返回
    None、不报错，`wall_is_no_slip` 因此恒为 True。上面两个测试只验证
    `build_boundary_ghost_provider` 这个纯函数本身的分支逻辑（调用时
    `wmles_model` 已经摆在那——测试自己保证的前提，不是真实构造流程
    保证的），从未捕捉到这个真实构造顺序问题——同一类"子函数测过、
    完整入口没测过"的模式本次会话已经出现多次（grad_U/grad_vel、
    DDES/IDDES 调用顺序）。"""

    def test_fr_solver_wmles_wall_is_no_slip_false(self):
        order = 2
        mesh = _build_synthetic_mixed_mesh(order)
        # 手动给一个真实边界面的 owner 单元打上 WALL 组标签（沿用
        # BoundaryMap.groups 的既定约定：name -> owner 单元全局索引数组，
        # 见 tag_boundary_groups 文档），不依赖 boundary_surface_mesh
        # （回退到单元级别匹配 `tag_boundary_groups`，足以验证本次修复）。
        fc = mesh.face_connectivity
        boundary_face = int(np.nonzero(fc.is_boundary)[0][0])
        wall_cell = int(fc.owner_cell[boundary_face])
        mesh.boundary_groups = {"wall_group": np.array([wall_cell], dtype=np.int64)}
        mesh.boundary_bc_types = {"wall_group": "WALL"}

        from autoflowcfd.core.fr_solver.solver import FRSolver
        solver = FRSolver(mesh, order=order, turb_model_name="wmles")

        assert solver.wmles_model is not None
        wall_configs = [
            cfg for cfg in solver.boundary_ghost_provider.code_to_config.values()
            if cfg.get("type") == "WALL"
        ]
        assert wall_configs, "test setup must produce at least one real WALL group"
        for cfg in wall_configs:
            assert cfg["is_no_slip"] is False

    def test_fr_solver_non_wmles_wall_is_no_slip_true(self):
        """反向对照：同一构造路径，非 WMLES 时必须保持 is_no_slip=True
        不变（排除"任何 FRSolver 构造都变成 False"这种更粗暴的误判）。"""
        order = 2
        mesh = _build_synthetic_mixed_mesh(order)
        fc = mesh.face_connectivity
        boundary_face = int(np.nonzero(fc.is_boundary)[0][0])
        wall_cell = int(fc.owner_cell[boundary_face])
        mesh.boundary_groups = {"wall_group": np.array([wall_cell], dtype=np.int64)}
        mesh.boundary_bc_types = {"wall_group": "WALL"}

        from autoflowcfd.core.fr_solver.solver import FRSolver
        solver = FRSolver(mesh, order=order, turb_model_name="sst")

        assert solver.wmles_model is None
        wall_configs = [
            cfg for cfg in solver.boundary_ghost_provider.code_to_config.values()
            if cfg.get("type") == "WALL"
        ]
        assert wall_configs
        for cfg in wall_configs:
            assert cfg["is_no_slip"] is True
