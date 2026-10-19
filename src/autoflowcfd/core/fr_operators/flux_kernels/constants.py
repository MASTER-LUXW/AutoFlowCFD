"""AutoFlowCFD V2.0 - flux_kernels 的共享常量。

从 `src/autoflowcfd/core/fr_operators/flux_kernels.py` 拆出(2026-09-24)。**唯一事实来源** -- 子模块一律从这里导入,
绝不各自复制一份(那是本项目明令禁止的"同一语义两个事实来源")。
"""


# 热力学常数（GAMMA、R_AIR、CP_AIR）在 core/thermo.py

#: 粘性 Interior Penalty 罚项常数。标准 DG 惯例取 O(1)~O(10)，理论要求
#: `c > C_trace(p)`（3D P1 的 trace 常数约 2.7）。**边界面与内部面用同一个
#: 值**：两者是同一个罚项、同一套量纲推导，没有理由给两个数。
#:
#: 此前这个常数在 `fr_residual/viscous_flux_kernel.py`、
#: `fr_residual/viscous_p0_kernel.py`、`gpu/residual/gpu_viscous.py` **各有
#: 一份独立的 `= 4.0`**，GPU 那边还把罚项公式整个抄了一遍。本项目已多次因
#: "两份实现只改了一份"出真实缺陷（滤波档双解析器、CFL 三处硬编码兜底、
#: 配置层与 CLI 相差 20 倍），所以统一到这里、由各 kernel 导入。
#: **实测标定值（2026-09-23）**，见 `resolve_viscous_ip_constant` 文档里的
#: 标定表。取 2.0 的四条判据：
#:   ① 理论硬要求 `c_ip > C_trace = (p+1)(p+3)/3`（3D P1 = 2.667）——
#:      BASE=1.0 给出的 c_ip 恰好等于下界、零余量，作为稳定化参数不可取；
#:   ② 长窗口精度：Blasius 4000 步最差偏离 0.0536（BASE=4.0 是 0.0581）；
#:   ③ 能量块谱改善在 `BASE>=2` 已饱和（33/96，与 BASE=4.0 相同），
#:      再加大没有收益；
#:   ④ 刚性因子 17（BASE=4.0 是 33）—— 虽然真实网格上实测代价为零
#:      （粘性从来不是约束方），但没有理由白付。
VISCOUS_IP_C_BASE = 2.0
