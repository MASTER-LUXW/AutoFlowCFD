"""无滑移壁 IP 罚项的**符号**必须是耗散的（2026-09-22 真实缺陷回归）。

## 这条测试钉的是什么

无滑移在本项目里是**弱施加**的：边界面的速度梯度按既定策略镜像内部值，
于是 `G_common` 与 `G_own` 的动量分量逐位相等、`jump_owner` 恒为零 ——
真正把壁面剪应力立起来的只有 `flux_kernels.viscous_boundary_penalty_tilde`
这一个正比于状态跳跃的 IP 罚项（见该函数文档）。它的符号错了不会有任何
形状/维度报错，只会让壁面**注入**动量而不是滞止流体。

## 曾经错在哪

罚项乘的 side 因子此前一律取 `owner_side[f]`。两条基的 `adj_row` 定向
约定不同：

* 坍缩面：`compute_exact_adj_rows` 给的是**未定向**的原始余因子行，外法向
  是 `oside * adj_row/|adj_row|`，所以罚项乘 `oside * |adj_row|` 是对的；
* 原生面（cube face 编码 >= 6）：`native_prism_face_adj_rows` 与
  `_native_tet_adj_row_batched` 返回的行**已按 outward 定向**（两个函数的
  文档都明确写了"下游 `side_factor = 1.0` 的处理对原生面同样正确、不需要
  再乘 owner_side"），所以罚项的 side 因子必须是 **+1**。

`exact_normal.py::compute_exact_face_normals_and_weights` 早就是
`side_factor = np.where(owner_code >= 6, 1.0, owner_side)`；无粘/粘性界面项
的原生分支也根本不用 side（定向靠 `adj_row`、分配靠 `lift_native`）。
**只有 IP 罚项漏了这一层分派**，于是 `owner_side = -1` 的原生面上罚项反号：

    原生棱柱 f0（底三角形封盖 c=-1）  side=-1   <- 平板算例的壁面正是这个面
    原生棱柱 f3（r=-1 侧四边形）      side=-1
    原生棱柱 f4（s=-1 侧四边形）      side=-1
    原生棱柱 f1/f2                    side=+1   本来就对
    原生四面体 4 个面                 side=-1（哑值）  全部反号

## 为什么此前所有验证算例都探不到

罚项正比于 `Q_o - Q_ghost`，无滑移壁上就是 `2 * u_wall_extrap`：

* **Couette** 的精确解可精确表示、外插到壁面**恒为零** -> 罚项恒为零，
  符号对不对结果一样。这就是"Couette 精确保持到 8.4e-9"对这条路径毫无
  约束力的原因（它是本项目唯一有精确解的粘性算例）;
* 解析 Blasius 初场同样外插到约零 -> 从极小的种子开始指数增长。实测
  贴壁第一层 `u/U` 从 0.096（解析）涨到 **1.454**（4000 步，x/L=0.3，
  是来流的 1.45 倍），第二层反而降到 0.187，胞内梯度翻负 -> cf 中位
  -3.2388，而全场 `u<0` 的解点**恒为 0**（不是分离，是伪壁面射流）;
* 罚项强度 A/B 直接钉住因果：`c_ip` 由 4 改 16（x4）后同一算例第 1000 步
  cf 就是 -1.1495（基线要到第 3000 步才穿零）—— 罚项越强漂移越快，正是
  "该项反耗散"的判据（若是欠施加，加强必须改善）。

## 本文件的判据（与长程运行无关，一次残差求值即可）

取**均匀流**：梯度恒零 -> BR1 通量项在所有面上恒为零 -> 粘性残差里**只
剩罚项**。于是
  1. 贴壁单元的流向动量残差必须 **< 0**（罚项必须反抗滑移）；
  2. 其余单元必须恒为 **0**（对称面上均匀流的法向跳跃为零，罚项为零）。
判据不含任何容差调参，符号反了就必然失败。
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# `tests/validation` 的算例装置（棱柱通道网格 + 逐面精确幽灵态）在本项目里
# 是 unit 测试也复用的共享装置，沿用既有做法把 `tests/` 加进 sys.path
# （见 `tests/unit/test_fixed_cfl_is_honoured.py` 同一段）。
_TESTS_DIR = str(Path(__file__).resolve().parents[1])
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from validation._channel_mesh import (  # noqa: E402
    build_channel_mesh_prism,
    build_face_exact_ghost_provider,
)

_RHO, _U, _P = 1.225, 30.0, 101325.0
_GAMMA = 1.4
_LX, _H, _LZ = 0.4, 0.1, 0.08

#: 哪个平面取无滑移壁 -> 它在原生棱柱里对应的面（见模块文档的表）。
#: `wall_bottom`/`wall_top` 是挤出方向的两个三角形封盖（f0/f1），
#: `z_min` 是三角形平面内的侧四边形（f3/f4）—— 三档合起来覆盖
#: `side=-1`（曾经反号）与 `side=+1`（本来就对）两类。
_WALL_PLANES = {
    "wall_bottom": "原生棱柱 f0（底封盖，side=-1，曾反号）",
    "wall_top": "原生棱柱 f1（顶封盖，side=+1，本来就对）",
    "z_min": "原生棱柱 f3/f4（侧四边形，side=-1，曾反号）",
}


def _build(basis, wall_plane, order, monkeypatch):
    monkeypatch.setenv("AFCFD_PRISM_BASIS", basis)
    from autoflowcfd.core.fr_solver import FRSolver
    from autoflowcfd.core.time_integration import TimeIntegrationScheme

    mesh = build_channel_mesh_prism(order, nx=2, ny=3, nz=2,
                                    Lx=_LX, H=_H, Lz=_LZ)
    # 除了被测那一面，其余面都要**跳跃为零**，否则它们自己的罚项会污染
    # 判据（第一版把 x_min/x_max 也写成 SYMMETRY，结果均匀流沿 +x 正是这
    # 两面的法向、镜像幽灵态给出 `u_ghost = -U`、跳跃 2U，非贴壁单元残差
    # 量级 4.03e+02 —— 装置错误，不是代码错误）：
    #   * y/z 两个方向的面取 SYMMETRY：法向速度本来就是零，镜像不改变任何
    #     分量，跳跃逐位为零；
    #   * x 两个面取 FARFIELD 且 `Q_free` 就是这个均匀态：幽灵态与内部态
    #     逐位相同，跳跃同样为零。
    bc = {name: {"type": "SYMMETRY"}
          for name in ("wall_bottom", "wall_top", "z_min", "z_max")}
    for name in ("x_min", "x_max"):
        bc[name] = {"type": "FARFIELD", "Q_free": [_RHO, _U, 0.0, 0.0, _P]}
    bc[wall_plane] = {"type": "WALL", "is_no_slip": True,
                      "wall_velocity": [0.0, 0.0, 0.0]}
    solver = FRSolver(
        mesh=mesh, order=order, turb_model_name="NONE", n_vars=5,
        time_scheme=TimeIntegrationScheme.SSP_RK3,
        rho_inf=_RHO, vel_inf=_U, p_inf=_P, mu_molecular=1.8e-3,
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


def _wall_cells(mesh, wall_plane, n_real):
    """贴被测面的那一层单元下标。

    判据取"到被测平面的最小解点距离等于全场最小值"——与阶数、解点布局
    （P0 只有一个胞心点、P1 有两层 Gauss 点）、网格层数都无关。第一版用
    的是"距离小于层厚的一半"这种按 ny 推出来的阈值，在 P0 上选不到任何
    单元（胞心距壁面 H/6，比按解点跨度算出的阈值大），是装置本身的脆弱。
    """
    xyz = np.asarray(mesh.sps_coords)[:, :n_real, :]
    axis, target = {"wall_bottom": (1, 0.0), "wall_top": (1, _H),
                    "z_min": (2, 0.0)}[wall_plane]
    dist = np.abs(xyz[..., axis] - target).min(axis=1)
    return np.nonzero(dist <= dist.min() * (1.0 + 1e-9) + 1e-14)[0]


@pytest.mark.parametrize("basis", ["native", "collapsed"])
@pytest.mark.parametrize("wall_plane", sorted(_WALL_PLANES))
def test_no_slip_penalty_decelerates_uniform_flow(monkeypatch, basis,
                                                  wall_plane):
    """均匀流 + 单面无滑移壁：贴壁单元的流向动量残差必须为负。"""
    from autoflowcfd.fr.native_padding import real_sps_per_cell

    solver, mesh = _build(basis, wall_plane, 1, monkeypatch)
    n_real = (real_sps_per_cell(mesh.order)[0] if basis == "native"
              else mesh.n_sps_per_cell)
    R = np.asarray(solver.compute_viscous_residual())[:, :n_real, :]
    wall = _wall_cells(mesh, wall_plane, n_real)
    assert wall.size > 0, "取不到贴壁单元，测试装置本身失效"

    rx = R[..., 1]
    scale = max(float(np.abs(rx).max()), 1e-300)
    assert float(rx[wall].sum()) < 0.0, (
        f"{basis} / {wall_plane}（{_WALL_PLANES[wall_plane]}）：贴壁单元的"
        f"流向动量残差合计 {float(rx[wall].sum()):+.6e} 不为负 —— 无滑移壁的"
        f"IP 罚项在往壁面单元**注入**动量而不是滞止流体，符号反了")

    other = np.setdiff1d(np.arange(mesh.n_cells), wall)
    if other.size:
        assert float(np.abs(rx[other]).max()) <= 1e-12 * scale, (
            f"{basis} / {wall_plane}：非贴壁单元的流向动量残差不为零"
            f"（{float(np.abs(rx[other]).max()):.3e}）—— 均匀流梯度恒零、"
            f"对称面跳跃恒零，这里本应严格为 0")


@pytest.mark.parametrize("basis", ["native", "collapsed"])
def test_no_slip_penalty_sign_at_p0(monkeypatch, basis):
    """P0 同一判据（`viscous_p0_kernel` 是独立的一份实现，同样漏了分派）。

    P0 下局部梯度恒为零，所以粘性残差**整体**就是罚项本身，这条比 P1 那条
    更直接。
    """
    from autoflowcfd.fr.native_padding import real_sps_per_cell

    solver, mesh = _build(basis, "wall_bottom", 0, monkeypatch)
    n_real = (real_sps_per_cell(mesh.order)[0] if basis == "native"
              else mesh.n_sps_per_cell)
    R = np.asarray(solver.compute_viscous_residual())[:, :n_real, :]
    wall = _wall_cells(mesh, "wall_bottom", n_real)
    assert wall.size > 0
    assert float(R[wall, :, 1].sum()) < 0.0, (
        f"{basis} P0：贴壁单元流向动量残差合计 "
        f"{float(R[wall, :, 1].sum()):+.6e} 不为负，罚项符号反了")
