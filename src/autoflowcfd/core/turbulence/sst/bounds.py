"""AutoFlowCFD V2.0 - SST 湍流量 k/omega 的可容许区间（唯一定义）。

两处消费者共用这一份（2026-09-25 收拢，此前 CPU `update.py` 与 GPU
`gpu_turbulence_sst.py` 各写一份限制器）：

* 正性限制器（`clip_to_bounds`）：显式路径每步、隐式路径 Newton 步之后把场
  夹回区间；
* 隐式 k-omega 的界约束（`fr_solver/turbulence/implicit.py`）：贴住界、且残差
  仍要把它推出界的解点，方程换成"等于界值"——否则离散稳态在那些点上
  无解，Newton 永远顶着界走不动。

区间：

* k：下界 `max(ABS_FLOOR, K_FLOOR_FRACTION * k_inf)`（来流下限，2026-09-11
  引入，理由见 `update.py::apply_positivity_limiter`），上界 `k_max`；
* omega：下界 `max(ABS_FLOOR, 逐点 realizability 下限 0.1*max(S, omega_inf))`
  （`source.py` 在源项求值时刷新），上界 `omega_max`。下界优先于上界
  （realizability 下限高于 omega_max 时取下限，与历来的限制器顺序一致）。
"""

#: 防止 0/负值进入 sqrt 与除法的绝对下限。
ABS_FLOOR = 1e-12

#: k 下限相对来流值的比例。
K_FLOOR_FRACTION = 1e-3


def positivity_bounds(model, xp):
    """返回 `(lb_k, lb_omega, ub_k, ub_omega)`，可与场广播（标量或逐点数组）。"""
    lb_k = ABS_FLOOR
    k_inf = getattr(model, "k_inf", None)
    if k_inf is not None and k_inf > 0:
        lb_k = max(ABS_FLOOR, K_FLOOR_FRACTION * float(k_inf))
    r_min = getattr(model, "_omega_realizability_min", None)
    lb_w = ABS_FLOOR if r_min is None else xp.maximum(r_min, ABS_FLOOR)
    return lb_k, lb_w, model.k_max, model.omega_max


def clip_to_bounds(model, xp) -> None:
    """把 `model.k_field/omega_field` 夹回可容许区间；非有限值先按绝对下限恢复
    （`xp.maximum(NaN, x)` 仍是 NaN，不先处理会穿透限制器永久污染场）。"""
    lb_k, lb_w, ub_k, ub_w = positivity_bounds(model, xp)
    k = xp.where(xp.isfinite(model.k_field), model.k_field, ABS_FLOOR)
    w = xp.where(xp.isfinite(model.omega_field), model.omega_field, ABS_FLOOR)
    model.k_field = xp.minimum(xp.maximum(k, lb_k), ub_k)
    model.omega_field = xp.maximum(xp.minimum(w, ub_w), lb_w)
