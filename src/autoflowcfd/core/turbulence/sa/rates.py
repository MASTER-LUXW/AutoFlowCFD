"""AutoFlowCFD V2.0 - SA-neg 的速率求值（全部后端共用的一份算法）。

后端注入两样东西：模型（`SAModel`，场所在的数组模块即后端的数组模块）与标量输运原语
（CPU `turbulence/transport/primitives.py::CpuScalarTransport`、GPU
`gpu/turbulence/gpu_scalar_transport/primitives.py::GpuScalarTransport`；分布式传紧凑空间上的
同一类对象）。返回 `TurbulenceRates`：

    raw        rho (P - D)
    source     (P - D)                                         显式路径对它做点隐式阻尼
    transport  [conv + diff + 逐点梯度项] / rho                   梯度项与 SST 的 Gamma_w |grad w|^2 同一归类

密度梯度用与 `nu_tilde` 同一个单元内梯度算子（P0 时为零，与 SST 对 k/omega 梯度的处理一致）。
壁面解点（`d == 0`）上三者都为零：那里施加强 Dirichlet `nu_tilde = 0`（`model.py` 模块文档）。
"""

from autoflowcfd.core.turbulence.nonfinite import replace_nonfinite
from autoflowcfd.core.turbulence.transported import TurbulenceRates
from autoflowcfd.core.utils.pointwise_threads import map_cell_chunks

from .pointwise import sa_diffusivity, sa_gradient_source


def evaluate_sa_rates(model, transport, Q, grad_vel, d_wall, mu) -> TurbulenceRates:
    """在 `model.nu_tilde_field` 上求 `d(nu_tilde)/dt` 的源项部分与输运部分。

    副作用：刷新模型上的 `nu_t` 与点隐式阻尼系数（`SAModel.compute_source_terms`）。

    Args:
        model: `SAModel`
        transport: 标量输运原语（`gradient`/`convection`/`diffusion`/`xp`）
        Q: `(n_cells, n_sps, 5)` 原始变量；grad_vel: `(n_cells, n_sps, 3, 3)` 速度梯度
        d_wall: `(n_cells, n_sps)` 壁面距离；mu: 分子动力粘度
    """
    xp = transport.xp
    nt = model.nu_tilde_field
    rho = xp.maximum(Q[:, :, 0], 1e-10)
    raw = model.compute_source_terms(Q, grad_vel, d_wall, mu)
    # 逐点函数按单元分块并行（numpy 下；同一份定义、结果逐位相同，`core/utils/pointwise_threads.py`）
    grad_nt = transport.gradient(nt)
    grad_rho = transport.gradient(Q[:, :, 0])
    gamma = map_cell_chunks(sa_diffusivity, (nt, rho), mu, xp, xp=xp)
    with _errstate(xp):
        conv = transport.convection(nt, model.nu_tilde_inf)
        diff = transport.diffusion(nt, gamma, grad=grad_nt)
        trans = (conv + diff + map_cell_chunks(_gradient_source, (nt, rho, grad_nt, grad_rho), mu, xp, xp=xp)) / rho
        src = raw / rho
    off_wall = d_wall != 0.0
    raw = xp.where(off_wall, raw, 0.0)
    src = xp.where(off_wall, src, 0.0)
    trans = xp.where(off_wall, replace_nonfinite(model, trans, "SA 输运", xp), 0.0)
    return TurbulenceRates((raw,), (src,), (trans,))


def _gradient_source(nu_tilde, rho, grad_nu_tilde, grad_rho, mu, xp):
    """`sa_gradient_source` 按"数组在前、标量在后"的参数顺序（`map_cell_chunks` 的约定）。"""
    return sa_gradient_source(nu_tilde, rho, mu, grad_nu_tilde, grad_rho, xp)


class _errstate:
    """numpy 下屏蔽退化单元上的溢出/无效运算警告（非有限值随后归零）；cupy 不发这类警告。"""

    __slots__ = ("ctx",)

    def __init__(self, xp):
        import numpy as np
        self.ctx = np.errstate(over="ignore", invalid="ignore") if xp is np else None

    def __enter__(self):
        if self.ctx is not None:
            self.ctx.__enter__()

    def __exit__(self, *exc):
        if self.ctx is not None:
            return self.ctx.__exit__(*exc)
        return False
