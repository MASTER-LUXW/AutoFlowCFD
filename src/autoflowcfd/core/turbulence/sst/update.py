"""AutoFlowCFD V2.0 - 正性限幅与场更新

从 `src/autoflowcfd/core/turbulence/sst.py` 的 `SSTModelFR` 拆出（2026-09-24，项目「单文件不超
500 行」规范）。mixin 是本仓库既有惯例（`_SolverGeometryMixin`、
`_GPUSolverInitMixin` 等），沿用它而不是另发明一套。

**只含方法，没有状态**：全部属性由 `SSTModelFR` 的 `__init__` 建立，
这里通过 `self` 访问。
"""

import numpy as np

from .bounds import clip_to_bounds
from .log_omega import log_omega, omega_from_log


class _SSTUpdateMixin:
    """正性限幅与场更新"""

    def apply_positivity_limiter(self):
        """
        k/omega 的非有限值恢复与上界（`bounds.py::clip_to_bounds`）。

        **不裁剪下界**（2026-09-26）：被输运的 k/omega 可以越过下限甚至为负，
        realizability 只作用于模型项求值，理由见 `bounds.py` 模块文档。

        上界的物理依据：k_max = 0.5 vel_inf^2（湍动能不超过平均流动能）、omega_max
        为远大于工程壁面值的保守上界。不设上界时 SST 源项与输运的正反馈（P_k ∝ k，
        transport ∝ ∇k）会让 k/omega 指数增长到 1e260 量级（cube_demo 100 步内实测），
        而平均流因 nu_t 被 a1 限幅保持正常——"平均流正常但湍流场完全发散"。
        """
        clip_to_bounds(self, np)

    def update_fields(self, dt, Sk: np.ndarray, S_log_omega: np.ndarray,
                      transport_k: np.ndarray = None, transport_log_omega: np.ndarray = None):
        """执行一个时间步长的湍流场更新（`advance_k_log_omega`），再过正性/上界限制器。"""
        advance_k_log_omega(self, dt, Sk, S_log_omega, transport_k, transport_log_omega, np)
        self.apply_positivity_limiter()


def advance_k_log_omega(model, dt, Sk, S_log_omega, transport_k, transport_log_omega, xp):
    """k 与 `w = ln(omega)` 的一步显式更新（见 `log_omega.py`），CPU 与 GPU 模型共用
    （`xp` 为模型数组所在的模块）。不含正性限制器，调用方随后施加。

    Args:
        model: 湍流模型（`SSTModelFR` 或 `GPUTurbulenceSST`），就地更新 `k_field/omega_field`
        dt: 时间步长（标量或可与 Sk 广播的逐点数组——稳态加速模式为局部 CFL 步长
            dt_local，DUAL_TIME 模式为标量物理 dt）
        Sk: 湍动能源项（dk/dt 量纲，已除以 rho，P_k-D_k 合并后的净值）
        S_log_omega: w 方程源项 `S_omega / (rho omega)`（P_omega-D_omega+CD_omega）
        transport_k / transport_log_omega: k 与 w 的完整输运（对流+扩散，
            w 另含 `Gamma_w |grad w|^2`），可为 None
    """
    # 源项点隐式阻尼（point-implicit destruction，Blazek / Wilcox 对 k-omega 刚性耗散
    # 项的标准做法）：耗散项在新值上线性化、系数用旧值，
    #   phi_new = phi_old + dt*S / (1 + dt*c)
    #   k:  D_k/rho = beta* omega k      -> c = beta* omega
    #   w:  D_omega/(rho omega) = beta omega，对 w 线性化 d(beta omega)/dw = beta omega
    #       -> c = beta omega
    # cfl.py 的对流/粘性 CFL 覆盖的是空间算子的谱半径，不覆盖这种逐点反应项刚性
    # （合成 Couette+SST 升阶到 P2 时纯前向欧拉失稳）。只阻尼源项，不阻尼输运。
    beta_blend = getattr(model, "_last_beta_blend", None)
    if beta_blend is None:
        # 防御性回退（正常路径下源项求值总在更新之前，已刷新混合 beta）：用 beta2
        # （> beta1，阻尼更强而非更弱）
        beta_blend = model.beta2
    omega_old = model.omega_field
    with np.errstate(over='ignore', invalid='ignore'):
        dk = Sk / (1.0 + dt * model.beta_star * omega_old)
        dw = S_log_omega / (1.0 + dt * beta_blend * omega_old)
    if transport_k is not None:
        dk = dk + transport_k
    if transport_log_omega is not None:
        dw = dw + transport_log_omega
    # 非有限增量归零：退化网格上源项/输运可能出现 inf-inf
    dk = xp.where(xp.isfinite(dk), dk, 0.0)
    dw = xp.where(xp.isfinite(dw), dw, 0.0)

    model.k_field = model.k_field + dt * dk
    model.omega_field = omega_from_log(log_omega(omega_old, xp) + dt * dw, model.omega_max, xp)
