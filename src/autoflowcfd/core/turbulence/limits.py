"""AutoFlowCFD V2.0 - 各湍流模型共用的物理上限（唯一定义）。

湍流标量梯度不设有量纲绝对上限：近壁真实的 |grad ln omega| ~ 2/y，任何固定上限都会在足够贴壁的网格上截断
物理值。
"""

from loguru import logger

#: 湍流粘性比 `mu_t / mu` 的上限。SST（CPU/GPU）的涡粘与 SA-neg 的 `nu_tilde` 上界都取它。
#:
#: 量级依据：零压力梯度湍流边界层外层 `nu_t,max ~ 0.0168 U delta*`（Clauser 外层涡粘），`Re_delta* ~ 0.046 Re_L^0.8`，
#: 即 `nu_t,max / nu ~ 7.7e-4 Re_L^0.8`——汽车（Re_L ~ 1e7）约 3e2，全尺度飞行器（Re_L ~ 1e9）约 1.2e4；尾迹的峰值
#: 同量级。1e5 比全尺度高雷诺数外流的物理最大值高约一个数量级，只拦截数值失控（SST 在阶数切换的冷启动暂态里曾把
#: 个别解点的 k/omega 比推到粘性比 6.8e10，见 `sst/blending.py::compute_eddy_viscosity` 文档）。上限在物理解上
#: 激活时由 `report_viscosity_cap` 计数告警（2026-10-19，I-47：此前静默截断）。
TURBULENT_VISCOSITY_RATIO_MAX = 1.0e5


def report_viscosity_cap(model, n_active: int, what: str) -> None:
    """涡粘比上限激活的计数与告警：累计解点数记在 `model.viscosity_cap_activations`；首次激活告警，之后累计数
    每翻一倍再告警一次（不逐步刷屏）。

    Args:
        model: 湍流模型（构造时 `viscosity_cap_activations = 0`、`_viscosity_cap_reported = 0`）
        n_active: 本次截断的解点数
        what: 告警里的量名（"SST nu_t" / "SA nu_tilde"）
    """
    if n_active <= 0:
        return
    model.viscosity_cap_activations += int(n_active)
    if model.viscosity_cap_activations >= 2 * model._viscosity_cap_reported:
        model._viscosity_cap_reported = model.viscosity_cap_activations
        logger.warning(f"{what} 撞到涡粘比上限 {TURBULENT_VISCOSITY_RATIO_MAX:g}：本次 {int(n_active)} 个解点被截断"
                       f"（累计 {model.viscosity_cap_activations}）——全尺度物理解不应到达该上限，检查湍流场是否失控")
