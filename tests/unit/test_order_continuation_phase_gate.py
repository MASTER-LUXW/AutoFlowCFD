"""Order Continuation 阶段判据 `PhaseGate`：湍流必须一起到位才升阶 / 判收敛
（`core/utils/order_continuation/policy.py` 模块文档）。"""

from types import SimpleNamespace

from autoflowcfd.core.utils.order_continuation.policy import PhaseGate, production_ramp_complete


def _solver(turb_res=None, ramp_done=True, with_ramp=True):
    model = SimpleNamespace(production_factor=1.0) if with_ramp else SimpleNamespace()
    st = None if turb_res is None else {"last_info": {"res_norm": turb_res}}
    return SimpleNamespace(turb_model=model, _turb_production_ramp_complete=ramp_done, _newton_turb_state=st)


def test_ramp_incomplete_blocks_advance_even_with_large_mean_drop():
    s = _solver(turb_res=1.0, ramp_done=False)
    assert not production_ramp_complete(s)
    assert not PhaseGate().reached(s, mean_drop=1e6, required=1e3)


def test_turbulence_must_drop_from_post_ramp_baseline():
    gate = PhaseGate()
    s = _solver(turb_res=100.0)
    assert not gate.reached(s, mean_drop=1e4, required=1e3)      # 基准取斜坡完成后的首个值
    s._newton_turb_state["last_info"]["res_norm"] = 1.0           # 只降 100 倍
    assert not gate.reached(s, mean_drop=1e4, required=1e3)
    s._newton_turb_state["last_info"]["res_norm"] = 0.05          # 降 2000 倍
    assert gate.reached(s, mean_drop=1e4, required=1e3)
    assert not gate.reached(s, mean_drop=10.0, required=1e3)      # 平均流没降够同样不行


def test_without_implicit_turbulence_only_mean_flow_and_ramp_count():
    s = _solver(turb_res=None)
    assert PhaseGate().reached(s, mean_drop=2e3, required=1e3)
    assert not PhaseGate().reached(_solver(turb_res=None, ramp_done=False), mean_drop=2e3, required=1e3)
    laminar = SimpleNamespace(turb_model=None)
    assert PhaseGate().reached(laminar, mean_drop=2e3, required=1e3)
