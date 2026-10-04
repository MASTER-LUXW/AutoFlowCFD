"""AutoFlowCFD V2.0 - 各湍流模型共用的物理上限与数值保护（唯一定义）。"""

import numpy as np

#: 湍流粘性比 `mu_t / mu` 的物理上限。工程外流里该比值最高到 1e4 量级；上限只拦截数值
#: 失控（SST 在阶数切换的冷启动暂态里曾把个别解点的 k/omega 比推到粘性比 6.8e10，见
#: `sst/blending.py::compute_eddy_viscosity` 文档），不改变任何物理解。SST（CPU/GPU）的涡粘与
#: SA-neg 的 `nu_tilde` 上界都取它。
TURBULENT_VISCOSITY_RATIO_MAX = 1.0e5

#: 被输运湍流标量的梯度模长上限。退化单元上理论为常数的场求梯度，度量比值 adj(J)/det(J)
#: 把浮点噪声放大到 >1e150（2026-08-22 真实网格），模长超过上限的点等比缩到上限。
MAX_GRADIENT_MAGNITUDE = 1e6


def clip_gradient_magnitude(grad, xp):
    """`grad (..., 3)` 模长超过 `MAX_GRADIENT_MAGNITUDE` 的点等比缩到上限，返回新数组。

    各湍流模型的源项与输运（CPU、单机 GPU、多 GPU）共用这一份。分量平方溢出时模长为 inf、
    缩放为 0（该点梯度置零，不是 NaN；有限输入不会产生 NaN）。
    """
    if xp is np:
        with np.errstate(over="ignore", invalid="ignore"):
            mag = np.linalg.norm(grad, axis=-1)
            return grad * np.clip(MAX_GRADIENT_MAGNITUDE / np.maximum(mag, 1e-10), 0.0, 1.0)[..., None]
    mag = xp.linalg.norm(grad, axis=-1)
    return grad * xp.clip(MAX_GRADIENT_MAGNITUDE / xp.maximum(mag, 1e-10), 0.0, 1.0)[..., None]
