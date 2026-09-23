"""粘性离散算子的耗散性：逐块谱判据（A2，2026-09-23）。

## 判据为什么按"块"写

粘性算子对守恒变量的雅可比是**块下三角**的：`rho` 行恒为零（粘性通量的
质量分量恒为 0）；动量行依赖 `(rho, rho_u)`、不依赖 `rho_E`（`tau` 只含
速度梯度）；能量行依赖全部三者。所以

    sigma(J) = {0}(rho 块) ∪ sigma(A_uu)(动量块) ∪ sigma(A_EE)(能量块)

两个对角块可以、也应该分开判——它们的物理算子根本不同：动量的
`div(tau)` 是**弹性型算子**（矢量 Laplacian + 散度梯度，强耦合），能量的
`div(-k grad T)` 是**纯标量 Laplacian**。实测两者的耗散性差别极大。

## 实测标定（均匀基态、16 单元、原生 P1、中心差分雅可比）

扫 `flux_kernels.VISCOUS_IP_C_BASE`（`base=0` 即**没有内部面 IP 罚项**，
也就是 2026-09-23 之前的内部面处理）：

    base   c_ip    动量块 max Re   正实部     能量块 max Re   正实部
     0.0    0.00   +4.0927e-10     0/288     +5.7013e+01     78/96
     1.0    2.67   +0.0000e+00     0/288     +4.9894e+01     40/96
     2.0    5.33   +0.0000e+00     0/288     +4.8313e+01     33/96
     4.0   10.67   +0.0000e+00     0/288     +4.7045e+01     33/96

两条结论：

1. **动量块本来就严格耗散**，与罚项无关（`base=0` 时 max Re 已是 4.1e-10，
   即舍入量级；那个零特征值是常速度模态，本该不被扩散衰减）。所以内部面
   罚项**不是**为动量块加的。
2. **能量块此前有 81% 的模态在增长**（78/96）；补上内部面 IP 罚项后降到
   34%（33/96）、`max Re` 从 +57.0 降到 +47.0，并在 `base>=2` 饱和。

## 残余的能量块正实部：已定性、未消除，不是本项目 A2 那个缺口

补罚项只能改善、不能消除它。已被实测排除的四条成因：

* **罚项强度**：扫到 `c_ip = 341`（base=128）仍饱和在 `max Re = +4.38e+01`，
  正实部数不再下降 —— 这些模态的跨面跳跃近乎为零，罚项按定义作用不到；
* **过积分**：`AFCFD_VISC_OVERINT=on/off` 两组读数**逐位相同**；
* **边界类型**：六面 SYMMETRY（绝热、纯 Neumann）47.0/33 vs 六面 FARFIELD
  （透射）37.5/25 —— 只差 20%，主因在内部面而非边界；
* **"强形式二次微分 D^2 本质非耗散"**这条曾被提出、又被自己的数据否掉：
  参考空间 `sum_k D_k^2` 的谱基本全零（原生棱柱 P1 max Re = 8.8e-17、
  `max(a^2-b^2) = 3.0e-32`），因为多项式微分矩阵是**幂零**的
  （`D^(p+1) = 0` ⇒ 全部特征值恰好 0）。体积项贡献零而非正实部。

加密缩放（nx=ny=n、nz=2 固定）：

    n    h=H/n      max Re      min Re      正实部/维数   max/|min|
    1    1.00e-1   +3.6914e+01  -6.2535e+02   11/24       0.059
    2    5.00e-2   +4.7045e+01  -7.3079e+02   33/96       0.064
    3    3.33e-2   +6.4184e+01  -9.2812e+02   60/216      0.069

相对权重稳定在 6~7%、不随加密塌缩，所以这是**格式的性质**而不是网格伪影。
消除它需要把粘性项从 FR 强形式改写成对称弱形式（`-B^T W B` 型刚度矩阵），
那是对本项目核心离散形式的改动，不在本次范围内；**物理判据上它当前没有
造成可观代价**：Blasius 解析初场 4000 步 cf 中位 +1.0074（BASE=1.0）/
+1.0149（BASE=4.0），`tau>0` 全程成立。

## 本文件钉的是什么

* 动量块必须**严格耗散**（硬判据，与罚项无关，防止将来改动破坏速度扩散）；
* 能量块的增长模态数必须**不超过 40**（`base=0` 时是 78）—— 这条直接钉住
  内部面 IP 罚项的存在与有效性，去掉罚项就会失败；
* 能量块 `max Re` 必须不超过 `+6.0e+01`（`base=0` 时 +57.0，带余量）。

## 规模

16 单元小网格（native 480 自由度 / collapsed 640），中心差分雅可比需要
`2*N` 次粘性残差求值，秒级。不满足耗散性的是**格式**、不是某张网格，所以
不需要大网格。
"""

import sys
from pathlib import Path

import numpy as np
import pytest

_TESTS_DIR = str(Path(__file__).resolve().parents[1])
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from validation._channel_mesh import (  # noqa: E402
    build_channel_mesh_prism,
    build_face_exact_ghost_provider,
)

_RHO, _U, _P, _GAMMA = 1.225, 30.0, 101325.0, 1.4
_LX, _H, _LZ = 0.4, 0.1, 0.08
#: 分子粘度取得偏大，让粘性项主导、谱的量级明确（与判据本身无关，判据是
#: 符号而不是量级）。
_MU = 1.8e-3


def _uniform_solver(monkeypatch):
    monkeypatch.setenv("AFCFD_PRISM_BASIS", "native")
    from autoflowcfd.core.fr_solver import FRSolver
    from autoflowcfd.core.time_integration import TimeIntegrationScheme

    mesh = build_channel_mesh_prism(1, nx=2, ny=2, nz=2,
                                    Lx=_LX, H=_H, Lz=_LZ)
    # 六面全取 SYMMETRY：均匀流沿 +x 时 y/z 四个面的法向速度本就为零、
    # 镜像不改变任何分量；x 两个面的法向分量被镜像，这是真实的 BC 行为，
    # 谱判据不要求"边界不参与"，只要求整个算子耗散。热边界全是绝热
    # （Neumann），所以边界档不该有能量罚项 —— 与 `k_total=0.0` 一致。
    bc = {name: {"type": "SYMMETRY"}
          for name in ("wall_bottom", "wall_top", "x_min", "x_max",
                       "z_min", "z_max")}
    solver = FRSolver(
        mesh=mesh, order=1, turb_model_name="NONE", n_vars=5,
        time_scheme=TimeIntegrationScheme.SSP_RK3,
        rho_inf=_RHO, vel_inf=_U, p_inf=_P, mu_molecular=_MU,
        bc_overrides=bc, adaptive_cfl=False,
        cfl_start=0.05, cfl_max=0.05, cfl_min=0.05,
    )
    solver.order_continuation_enabled = False
    solver.boundary_ghost_provider = build_face_exact_ghost_provider(
        mesh, _LX, _H, _LZ, bc)
    U = np.zeros_like(np.asarray(solver.state.U))
    U[..., 0] = _RHO
    U[..., 1] = _RHO * _U
    U[..., 4] = _P / (_GAMMA - 1.0) + 0.5 * _RHO * _U ** 2
    solver.state.U = np.ascontiguousarray(U)
    solver.state._update_primitives()
    return solver, mesh


def _viscous_jacobian(solver, mesh, n_real, rel_step=1e-6):
    U0 = np.array(solver.state.U, dtype=float, copy=True)

    def rhs(U):
        solver.state.U = np.ascontiguousarray(U)
        solver.state._update_primitives()
        return np.asarray(solver.compute_viscous_residual(),
                          dtype=float)[:, :n_real, :5].ravel().copy()

    idx = [(c, s, v) for c in range(mesh.n_cells)
           for s in range(n_real) for v in range(5)]
    scale = np.maximum(np.array([np.abs(U0[:, :n_real, v]).max()
                                 for v in range(5)]), 1e-30)
    J = np.empty((len(idx), len(idx)))
    for col, (c, s, v) in enumerate(idx):
        eps = rel_step * scale[v]
        Up = U0.copy(); Up[c, s, v] += eps
        Um = U0.copy(); Um[c, s, v] -= eps
        J[:, col] = (rhs(Up) - rhs(Um)) / (2.0 * eps)
    solver.state.U = U0
    solver.state._update_primitives()
    return J


#: 能量块增长模态数上限。`base=0`（无内部面 IP 罚项）实测 78/96，
#: `base>=2` 实测 33/96；取 40 既钉住罚项有效性、又留了余量。
_MAX_GROWING_ENERGY_MODES = 40
#: 能量块 max Re 上限（`base=0` 实测 +5.70e+01，带余量）。
_MAX_ENERGY_GROWTH_RATE = 6.0e1


def _block_spectrum(J, n_cells, n_real, var_indices):
    """取 J 的某个守恒变量子块的特征值实部（块下三角，见模块文档）。"""
    idx = np.arange(J.shape[0]).reshape(n_cells, n_real, 5)
    sub = idx[:, :, var_indices].ravel()
    return np.linalg.eigvals(J[np.ix_(sub, sub)]).real


def test_momentum_diffusion_block_is_strictly_dissipative(monkeypatch):
    """动量块必须严格耗散：`max Re(A_uu) <= 0`（到舍入）。

    这是硬判据，与 IP 罚项无关（`base=0` 时实测 max Re 已是 4.1e-10）。
    它守的是速度扩散本身 —— 将来任何改动把它破坏了，这里会立刻失败。
    """
    from autoflowcfd.fr.native_padding import real_sps_per_cell

    solver, mesh = _uniform_solver(monkeypatch)
    n_real = real_sps_per_cell(mesh.order)[0]
    J = _viscous_jacobian(solver, mesh, n_real)
    re = _block_spectrum(J, mesh.n_cells, n_real, [1, 2, 3])
    span = max(float(np.abs(re).max()), 1e-300)
    n_pos = int((re > 1e-8 * span).sum())
    assert n_pos == 0, (
        f"动量扩散块有 {n_pos}/{re.size} 个特征值实部为正"
        f"（max Re = {re.max():+.6e}，谱跨度 {span:.3e}）—— 速度扩散必须"
        f"处处耗散，`base=0`（无内部面罚项）时实测都只有 4.1e-10")
    assert re.min() < 0.0, (
        f"动量块没有任何负实部（min Re = {re.min():+.6e}）——"
        f"粘性算子整体失效，测试装置或实现有问题")


def test_interior_ip_penalty_bounds_the_energy_block(monkeypatch):
    """内部面 IP 罚项必须把能量块的增长模态压住。

    **去掉内部面罚项这条就会失败**（`base=0` 实测 78/96、max Re +57.0；
    现在 33/96、+47.0）。完整标定表与"为什么消除不了"的四条排除依据见
    模块文档。
    """
    from autoflowcfd.fr.native_padding import real_sps_per_cell

    solver, mesh = _uniform_solver(monkeypatch)
    n_real = real_sps_per_cell(mesh.order)[0]
    J = _viscous_jacobian(solver, mesh, n_real)
    re = _block_spectrum(J, mesh.n_cells, n_real, [4])
    span = max(float(np.abs(re).max()), 1e-300)
    n_pos = int((re > 1e-8 * span).sum())
    assert n_pos <= _MAX_GROWING_ENERGY_MODES, (
        f"能量块增长模态数 {n_pos}/{re.size} 超过上限 "
        f"{_MAX_GROWING_ENERGY_MODES} —— 内部面 IP 罚项失效或被削弱了"
        f"（无罚项时实测 78/96，罚项生效时 33/96）")
    assert re.max() <= _MAX_ENERGY_GROWTH_RATE, (
        f"能量块 max Re = {re.max():+.6e} 超过上限 "
        f"{_MAX_ENERGY_GROWTH_RATE:.1e}（无罚项时实测 +5.70e+01）")
    assert re.min() < -1.0 * re.max(), (
        f"能量块负谱 {re.min():+.6e} 没有压过正谱 "
        f"{re.max():+.6e} —— 算子整体不再以耗散为主")
