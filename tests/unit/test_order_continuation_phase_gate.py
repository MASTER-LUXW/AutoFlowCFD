"""Order Continuation 阶段判据 `PhaseGate`：湍流必须一起到位才升阶 / 判收敛，已收敛到
舍入误差也算到位（`core/utils/order_continuation/policy.py` 模块文档）。"""

from types import SimpleNamespace

from autoflowcfd.core.utils.order_continuation.policy import (
    ROUNDOFF_DROP, PhaseGate, production_ramp_complete,
)


def _solver(turb_res=None, ramp_done=True, with_ramp=True):
    model = SimpleNamespace(production_factor=1.0) if with_ramp else SimpleNamespace()
    st = None if turb_res is None else {"last_info": {"res_norm": turb_res}}
    return SimpleNamespace(turb_model=model, _turb_production_ramp_complete=ramp_done, _newton_turb_state=st)


def _turb(s, r):
    s._newton_turb_state["last_info"]["res_norm"] = r


def test_ramp_incomplete_blocks_advance_even_with_large_mean_drop():
    s = _solver(turb_res=1.0, ramp_done=False)
    gate = PhaseGate()
    gate.observe(s, 1e6)
    assert not production_ramp_complete(s)
    assert not gate.reached(s, res=1.0, baseline=1e6, required=1e3)


def test_turbulence_must_drop_from_post_ramp_baseline():
    s = _solver(turb_res=100.0, ramp_done=False)
    gate = PhaseGate()
    gate.observe(s, 1e3)               # 斜坡期间：湍流首值 100，但不作为斜坡后基准
    s._turb_production_ramp_complete = True
    _turb(s, 50.0)
    gate.observe(s, 1e3)               # 斜坡完成后的首个值 50 是湍流基准
    assert not gate.reached(s, res=0.5, baseline=1e3, required=1e3)
    _turb(s, 0.1)                       # 相对基准只降 500 倍
    assert not gate.reached(s, res=0.5, baseline=1e3, required=1e3)
    _turb(s, 0.04)                      # 降 1250 倍
    assert gate.reached(s, res=0.5, baseline=1e3, required=1e3)
    assert not gate.reached(s, res=10.0, baseline=1e3, required=1e3)   # 平均流没降够同样不行


def test_roundoff_level_counts_as_reached():
    """斜坡完成前已收敛：斜坡后重置的基准本身已很小，相对它降不够，但相对本阶段第一步
    已到舍入误差（plate_demo P0 实测形态）。"""
    s = _solver(turb_res=1e2)
    gate = PhaseGate()
    gate.observe(s, 4e8)                              # 本阶段第一步
    _turb(s, 1e-10)
    gate.observe(s, 2e-3)                             # 湍流基准也已经在舍入误差附近
    first_drop_res = 4e8 / (ROUNDOFF_DROP * 2)
    assert gate.reached(s, res=first_drop_res, baseline=2e-3, required=1e3)
    assert not gate.reached(s, res=1e-1, baseline=2e-3, required=1e3)


def test_without_implicit_turbulence_only_mean_flow_and_ramp_count():
    s = _solver(turb_res=None)
    gate = PhaseGate()
    gate.observe(s, 2e3)
    assert gate.reached(s, res=1.0, baseline=2e3, required=1e3)
    s2 = _solver(turb_res=None, ramp_done=False)
    assert not PhaseGate().reached(s2, res=1.0, baseline=2e3, required=1e3)
    laminar = SimpleNamespace(turb_model=None)
    assert PhaseGate().reached(laminar, res=1.0, baseline=2e3, required=1e3)
