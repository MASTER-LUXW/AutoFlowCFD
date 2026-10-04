"""AutoFlowCFD V2.0 - SA-neg 的解析单元块 Jacobian 接入（逐点求值器与壁面 Dirichlet 规格）。

块装配本身与模型无关（`turbulence/jacobian`）；SA 只给出：

* `SAPointwise`：逐点 `(S, Gamma)`，`S = rho (P - D) + 逐点梯度项`、`Gamma = (mu + rho nu_tilde f_n)/sigma`
  ——与残差（`rates.py`）同一组函数（`pointwise.py`），不修改模型状态；
* `sa_dirichlet_spec`：`nu_tilde` 在无滑移壁面为 0；
* `sa_strong_rows`：壁面解点（`d == 0`）上的强 Dirichlet 行（`model.py` 模块文档）。
"""

import numpy as np

from autoflowcfd.core.turbulence.limits import clip_gradient_magnitude

from .pointwise import sa_diffusivity, sa_gradient_source, sa_source_terms, vorticity_magnitude


class SAPointwise:
    """`evaluate(u (n_cells, n_sps, 1), g (n_cells, n_sps, 1, 3)) -> (S, Gamma)`（主机数组，各
    `(n_cells, n_sps, 1)`）。冻结的平均流输入（涡量、壁距、密度梯度）在构造时算好。做成类而不是
    闭包（项目规范）。"""

    __slots__ = ("xp", "rho", "mu", "nu", "d_wall", "omega_mag", "grad_rho", "production_factor")

    def __init__(self, xp, model, Q, grad_vel, d_wall, mu, grad_rho):
        self.xp = xp
        self.rho = xp.maximum(Q[:, :, 0], 1e-10)
        self.mu = float(mu)
        self.nu = self.mu / self.rho
        self.d_wall = d_wall
        self.omega_mag = vorticity_magnitude(grad_vel, xp)
        self.grad_rho = grad_rho
        self.production_factor = float(model.production_factor)

    def __call__(self, u, g):
        xp = self.xp
        nt = xp.ascontiguousarray(xp.asarray(u[..., 0]))
        gnt = clip_gradient_magnitude(xp.ascontiguousarray(xp.asarray(g[..., 0, :])), xp)
        source, _ = sa_source_terms(nt, self.nu, self.d_wall, self.omega_mag, self.production_factor, xp)
        S = self.rho * source + sa_gradient_source(nt, self.rho, self.mu, gnt, self.grad_rho, xp)
        Gamma = sa_diffusivity(nt, self.rho, self.mu, xp)
        return _host(S[..., None]), _host(Gamma[..., None])


def sa_dirichlet_spec(wall_zero_face):
    """SA 唯一 Newton 未知量的壁面 Dirichlet 规格 `(faces, values)`：无滑移壁 `nu_tilde = 0`。"""
    return (np.asarray(wall_zero_face, dtype=bool),), (None,)


def sa_strong_rows(d_wall):
    """`(n_cells, n_sps, 1)` 布尔：壁面解点的行（Jacobian 置零）。"""
    return np.asarray(_host(d_wall) == 0.0)[..., None]


def _host(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)
