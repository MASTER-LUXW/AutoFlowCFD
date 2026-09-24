# -*- coding: utf-8 -*-
"""攻角/侧滑角必须到达**每一处**"用自由来流填均匀守恒场"的地方。

## 钉住的真实缺陷（2026-09-24 发现）

"用自由来流填一个均匀守恒场"此前在 9 处各写一遍，**全部把速度写死成
`(vel_inf, 0, 0)`**，而边界条件（`Q_free`，经 `direction_from_freestream`）
一直用的是正确方向。`--aoa` 非零时初场与边界不一致 —— `FRSolver.__init__`
的注释早就写明这会让第一步吸收一个量级为 `vel_inf*sin(aoa)` 的速度跳跃。

| 位置 | 触发 |
|---|---|
| `core/utils/order_continuation.py` | 单机 CPU，目标阶数 >= 2 的全新算例降 P0 重建（必经） |
| `mpi/distributed_order_continuation/rebuild.py` | CPU MPI 阶数切换 |
| `mpi/distributed_mesh_loader/fully_distributed.py` | CPU 完全分布式阶数切换重分发 |
| `gpu/solver/gpu_solver/core.py` | 单机 GPU **初场**（与阶数无关） |
| `gpu/solver/gpu_solver_order_continuation.py` | 单机 GPU 阶数切换 |
| `gpu/distributed/gpu_distributed.py` | 多 GPU 传统模式**初场** |
| `gpu/distributed/gpu_distributed_order_continuation.py` | 多 GPU 阶数切换 |
| `gpu_distributed_fully_distributed/build.py` | 多 GPU 完全分布式**初场** |
| `gpu_distributed_fully_distributed/redistribute.py` | 同上，阶数切换 |

单机 CPU 的初场本来是对的，但随即被 Order Continuation 的降 P0 重建覆盖成
零攻角，所以单机也躲不开。另外其中 4 处带着 `get('vel_inf', 33.33)` 这类
魔法兜底值（与构造函数默认值是两份事实来源）。

修法：统一到 `core/utils/flow_direction.py::freestream_conservative_state`，
能量公式也收敛到 `core/fr_solver/state.py::uniform_conservative` 一份。

## 判据分三层

1. 共享函数本身：`aoa=aos=0` 时与原 `FRState.initialize_uniform` **逐位
   相同**（两条黄金轨迹依赖这一点）；非零角度时动量方向正确、能量不变；
   缺字段直接 KeyError（不给兜底）。
2. 行为：单机降 P0 重建与单机 GPU 初场，给 `aoa=10` 时动量方向必须是
   `(cos10, 0, sin10)`。
3. 结构：全仓库 `src/` 不许再出现写死 +x 的填充写法 —— 9 处同类写法里
   有 7 处本机跑不到（MPI / CuPy），只有结构判据能在将来新增第十处时拦住。
"""

import ast
import io
import pathlib
from types import SimpleNamespace

import numpy as np
import pytest

from autoflowcfd.core.fr_solver.state import FRState, uniform_conservative
from autoflowcfd.core.utils.flow_direction import (
    freestream_conservative_state,
    freestream_direction,
)

_RHO, _VEL, _P = 1.225, 33.33, 101325.0


def _fs(aoa=0.0, aos=0.0):
    return {"rho_inf": _RHO, "vel_inf": _VEL, "p_inf": _P,
            "aoa_deg": aoa, "aos_deg": aos}


# ---------------------------------------------------------------------------
# 1. 共享函数
# ---------------------------------------------------------------------------


class TestFreestreamConservativeState:
    def test_zero_angle_is_bit_identical_to_frstate(self):
        """两条黄金轨迹都走 `FRState.initialize_uniform`，零攻角时必须逐位相同。"""
        s = FRState(2, 4, 7)
        s.initialize_uniform(rho=_RHO, u=_VEL, v=0.0, w=0.0, p=_P)
        assert np.array_equal(freestream_conservative_state(_fs(), 5),
                              s.U[0, 0, :5])

    def test_frstate_formula_unchanged(self):
        """`initialize_uniform` 改为调用 `uniform_conservative` 后公式逐字未变。"""
        g = 1.4
        e = _P / ((g - 1.0) * _RHO) + 0.5 * (_VEL ** 2 + 0.0 ** 2 + 0.0 ** 2)
        ref = np.array([_RHO, _RHO * _VEL, _RHO * 0.0, _RHO * 0.0, _RHO * e])
        assert np.array_equal(uniform_conservative(_RHO, _VEL, 0.0, 0.0, _P), ref)

    @pytest.mark.parametrize("aoa,aos", [(10.0, 0.0), (-4.0, 0.0), (5.0, 3.0)])
    def test_momentum_direction_follows_the_angles(self, aoa, aos):
        v = freestream_conservative_state(_fs(aoa, aos), 5)
        mom = v[1:4] / v[0]
        np.testing.assert_allclose(mom, _VEL * freestream_direction(aoa, aos),
                                   rtol=0, atol=1e-12)

    def test_energy_does_not_depend_on_direction(self):
        """|v| 不随攻角变，总能也不能变。"""
        e0 = freestream_conservative_state(_fs(), 5)[4]
        e1 = freestream_conservative_state(_fs(10.0, 3.0), 5)[4]
        assert e1 == pytest.approx(e0, rel=1e-15)

    def test_extra_components_are_zero(self):
        """与此前 `np.zeros` 后只填前 5 列的行为一致（湍流量不在这里给）。"""
        v = freestream_conservative_state(_fs(10.0), 7)
        assert v.shape == (7,) and np.all(v[5:] == 0.0)

    @pytest.mark.parametrize("missing", ["rho_inf", "vel_inf", "p_inf"])
    def test_missing_field_is_a_hard_error(self, missing):
        """不给兜底值：缺字段是真缺陷，不能静默用一个"看似合理"的来流。"""
        fs = _fs()
        fs.pop(missing)
        with pytest.raises(KeyError):
            freestream_conservative_state(fs)

    def test_missing_angles_fall_back_to_plus_x(self):
        """旧 checkpoint 没有 aoa/aos 字段：退化为 +x，与此前行为逐位相同。"""
        fs = {"rho_inf": _RHO, "vel_inf": _VEL, "p_inf": _P}
        assert np.array_equal(freestream_conservative_state(fs),
                              freestream_conservative_state(_fs()))


# ---------------------------------------------------------------------------
# 2. 行为：单机 CPU 降 P0 重建
# ---------------------------------------------------------------------------


def _fake_cpu_solver(aoa):
    n_cells, n_vars = 3, 7
    state = FRState(n_cells, 8, n_vars)       # 目标阶数的状态（将被重建到 P0）
    state.initialize_uniform(rho=_RHO, u=_VEL, v=0.0, w=0.0, p=_P)
    return SimpleNamespace(
        state=state, freestream=_fs(aoa), turb_model=None, sgs_model=None,
        wall_distance=None, current_order=1, ops=None,
        mesh=SimpleNamespace(set_order=lambda p: None),
    )


class TestSingleCpuP0ResetKeepsTheDirection:
    def test_p0_reset_uses_the_angle(self):
        """**必经路径**：目标阶数 >= 2 的全新算例都会走这次重建。"""
        from autoflowcfd.core.utils.order_continuation import _reset_state_to_p0

        solver = _fake_cpu_solver(aoa=10.0)
        _reset_state_to_p0(solver, 1)
        U = solver.state.U
        assert U.shape[1] == 1, "没有重建到 P0"
        mom = U[:, 0, 1:4] / U[:, 0, 0:1]
        np.testing.assert_allclose(
            mom, np.broadcast_to(_VEL * freestream_direction(10.0), mom.shape),
            rtol=0, atol=1e-12)
        assert solver.current_order == 0

    def test_zero_angle_reset_matches_frstate(self):
        from autoflowcfd.core.utils.order_continuation import _reset_state_to_p0

        solver = _fake_cpu_solver(aoa=0.0)
        _reset_state_to_p0(solver, 1)
        ref = FRState(3, 1, 7)
        ref.initialize_uniform(rho=_RHO, u=_VEL, v=0.0, w=0.0, p=_P)
        assert np.array_equal(solver.state.U, ref.U)


# ---------------------------------------------------------------------------
# 2'. 行为：单机 GPU 初场（numpy 替身，复用该文件的 fixture）
# ---------------------------------------------------------------------------

from tests.unit.test_gpu_solver_order_continuation import (  # noqa: E402,F401
    _patch_gpu_modules,
)


class TestSingleGpuInitialFieldKeepsTheDirection:
    def test_gpu_initial_field_uses_the_angle(self):
        """与阶数无关：单机 GPU 构造完成那一刻的初场就必须带攻角。"""
        from autoflowcfd.fr.operators import generate_fr_operators
        from autoflowcfd.core.gpu.solver.gpu_solver import GPUFRSolver
        from tests.unit.test_fr_residual_inviscid import _build_synthetic_mixed_mesh

        order = 1
        mesh = _build_synthetic_mixed_mesh(order)
        solver = GPUFRSolver(
            mesh=mesh, ops=generate_fr_operators(order), order=order,
            device_id=0, mu_molecular=1.8e-5,
            rho_inf=_RHO, vel_inf=_VEL, p_inf=_P, turb_model="none",
            aoa_deg=10.0, aos_deg=0.0,
        )
        U = np.asarray(solver.U_gpu)
        mom = U[:, :, 1:4] / U[:, :, 0:1]
        np.testing.assert_allclose(
            mom.reshape(-1, 3),
            np.broadcast_to(_VEL * freestream_direction(10.0), (mom.size // 3, 3)),
            rtol=0, atol=1e-12)


# ---------------------------------------------------------------------------
# 3. 结构：不许再出现写死 +x 的填充
# ---------------------------------------------------------------------------

_SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "autoflowcfd"

#: 构造函数 **API 默认值**（`solver_kwargs.get('vel_inf', 33.33)`）不在禁止
#: 之列：那是 `DistributedFRSolver` 用 kwargs 声明自己的默认值，与另外三个
#: 求解器类签名里的 `vel_inf=33.33` 一致；它们喂的是 `freestream_velocity`
#: （方向已正确），不是"来流丢失时的兜底"。
_ALLOWED_FILES = {
    "core/mpi/distributed_solver/core.py",
    "core/mpi/distributed_solver/from_package.py",
}


def _is_x_momentum_fill(node):
    """`X[:, :, 1] = <...> * vel_inf` —— 按分量写死 x 动量（y/z 留零）。"""
    if not isinstance(node, ast.Assign) or len(node.targets) != 1:
        return False
    t = node.targets[0]
    if not isinstance(t, ast.Subscript) or not isinstance(t.slice, ast.Tuple):
        return False
    el = t.slice.elts
    if not (len(el) == 3 and isinstance(el[0], ast.Slice) and isinstance(el[1], ast.Slice)
            and isinstance(el[2], ast.Constant) and el[2].value == 1):
        return False
    return any(isinstance(n, ast.Name) and n.id == "vel_inf" for n in ast.walk(node.value))


def _is_initialize_uniform_without_direction(node):
    """`initialize_uniform(..., v=0.0, w=0.0, ...)` —— 调用处写死 v=w=0。"""
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "initialize_uniform"):
        return False
    kw = {k.arg: k.value for k in node.keywords if k.arg}
    return all(isinstance(kw.get(c), ast.Constant) and kw[c].value == 0.0 for c in ("v", "w"))


def _is_vel_inf_magic_fallback(node):
    """`.get('vel_inf', 33.33)` —— 来流字段缺失时静默用一个魔法值。"""
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get" and len(node.args) == 2
            and isinstance(node.args[0], ast.Constant) and node.args[0].value == "vel_inf"
            # 只禁**正的物理值**兜底。`get("vel_inf", 0.0)` 后接
            # `if vel_inf <= 0: return None` 是哨兵用法（诊断不可用就跳过，
            # 见 FRSolver._pseudo_time_budget），不是拿一个值冒充来流。
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, (int, float)) and node.args[1].value > 0)


_RULES = [
    (_is_x_momentum_fill, "按分量写死 x 动量（y/z 动量留零）"),
    (_is_initialize_uniform_without_direction, "initialize_uniform 调用处写死 v=w=0"),
    (_is_vel_inf_magic_fallback, "vel_inf 的魔法兜底值"),
]


@pytest.mark.parametrize("rule,what", _RULES, ids=[w for _r, w in _RULES])
def test_no_hardcoded_plus_x_freestream_fill(rule, what):
    """用 AST 只看真实的赋值与调用 —— 注释、文档字符串、函数签名里的
    默认值都不算（第一版用正则，误报了 `initialize_uniform` 的签名与本
    修复自己写在文档里的缺陷清单）。"""
    hits = []
    for p in sorted(_SRC.rglob("*.py")):
        rel = p.relative_to(_SRC).as_posix()
        if rel in _ALLOWED_FILES:
            continue
        tree = ast.parse(io.open(p, encoding="utf-8").read())
        hits += [f"{rel}:{n.lineno}" for n in ast.walk(tree) if rule(n)]
    assert not hits, (
        f"又出现了「{what}」—— 攻角/侧滑角（或来流本身）会在这里被静默丢掉。"
        f"改用 core/utils/flow_direction.py::freestream_conservative_state，"
        f"或直接取 solver.freestream 的键：" + "；".join(hits))
