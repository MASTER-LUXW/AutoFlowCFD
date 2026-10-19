"""AutoFlowCFD V2.0 - 量热完全气体（空气）的热力学常数，全仓库唯一定义。

状态方程 `p = (GAMMA - 1) (rho E - rho |u|^2 / 2)`、`p = rho R_AIR T`，定压比热 `CP_AIR = GAMMA R_AIR / (GAMMA - 1)`。
无粘 / 粘性残差、解析 Jacobian、CFL、参考马赫数、低马赫预处理、正性限制、边界幽灵态与 GPU 的 CUDA 源码都从这里取。

2026-10-19（I-12）以前：GAMMA 在 10 个模块各定义一次、R_AIR 3 次，另有残差 / Jacobian / CFL / 参考马赫数 / 状态换算
与 CUDA 源码里直接写的 1.4、0.4——改工质（或换比热比做验证）时任何一处漏改都会让各项之间不自洽，且不会报错。
（粘性的普朗特数是输运性质，在 `fr_residual/viscous_flux/constants.py`。）
"""

#: 比热比
GAMMA = 1.4

#: 空气比气体常数 J/(kg*K)
R_AIR = 287.0

#: 定压比热 J/(kg*K)
CP_AIR = GAMMA * R_AIR / (GAMMA - 1.0)
