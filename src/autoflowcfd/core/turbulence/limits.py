"""AutoFlowCFD V2.0 - 各湍流模型共用的物理上限（唯一定义）。

2026-10-17 删除了湍流标量梯度的有量纲绝对上限 `MAX_GRADIENT_MAGNITUDE = 1e6`（I-42）：它截断近壁真实的
|grad ln omega| ~ 2/y（y < 2e-6 m），设立理由是掩盖坍缩基时代退化单元上的度量噪声；扩散项一直用的是未截断的
梯度，截断从未保护过非有限值。
"""

#: 湍流粘性比 `mu_t / mu` 的物理上限。工程外流里该比值最高到 1e4 量级；上限只拦截数值
#: 失控（SST 在阶数切换的冷启动暂态里曾把个别解点的 k/omega 比推到粘性比 6.8e10，见
#: `sst/blending.py::compute_eddy_viscosity` 文档），不改变任何物理解。SST（CPU/GPU）的涡粘与
#: SA-neg 的 `nu_tilde` 上界都取它。
TURBULENT_VISCOSITY_RATIO_MAX = 1.0e5
