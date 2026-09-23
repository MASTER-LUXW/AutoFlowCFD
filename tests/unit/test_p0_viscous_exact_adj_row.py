"""P0 粘性界面 kernel 的度量伴随行必须取**逐 FP 精确值**（2026-09-24 回归）。

## 曾经错在哪

`core/fr_residual/viscous_p0_kernel.py` 取度量伴随行用的是

    adj_o_s0 = adj_j[oc, 0, oax]          # oax = owner_axis[f]

也就是坍缩坐标那条"取 SP 网格度量的第 oax 个轴行"的做法。两个独立后果：

1. **越界读**：`owner_axis` 对原生面是**复用槽位**（原生四面体存
   excluded_vertex 属于 {0..3}，原生棱柱存面序号 属于 {0..4}），而
   `adj_j` 的轴维只有 3。`excluded_vertex == 3` 的四面体面直接越界 ——
   numba nopython 不做边界检查，读到的是相邻单元的内存。
2. **即使不越界那个量本身也是错的**：与逐 FP 精确行
   `owner_adj_row_exact[f, i]` 相比，相对差在**每一个面**上都是 O(1)。

P>=1 的通用 kernel（`viscous_flux_kernel.py`）早在 2026-08-23 就改用了
`owner_adj_row_exact`/`neighbor_adj_row_exact`（见
`fr/face_flux_points/exact_normal.py` 模块文档：外插 SP 网格度量到 FP 对
本质上是有理函数的 `adj(J)` 有不可忽略的截断误差），**P0 这份拷贝被漏掉
了** —— 又一次"同一语义两份实现、只改了一份"。

影响面：P0 粘性运行，以及 **Order Continuation 的 P0 阶段**（真实长程
算例都从 P0 起步）。

## 本文件的判据

前三条刻意都不依赖"再写一份参考实现"：

1. `test_owner_axis_exceeds_the_adj_j_axis_dimension` —— 真实混合网格上
   `owner_axis` 确实取到 3，而 `adj_j` 轴维只有 3，所以旧表达式**可达
   越界**。这是几何/编码约定的事实，不是对 kernel 的断言。
2. `test_the_two_metric_sources_are_not_interchangeable` —— 两个量在每个
   面上都差 O(1)，所以"用哪个"不是风格问题。
3. `test_p0_kernel_no_longer_accepts_the_sp_grid_metric` —— kernel 签名里
   不再有 `adj_j`/`owner_axis`/`neighbor_axis`，旧写法在代码层面已不可能
   被写出来。

第四条是端到端的差分判据（`test_old_metric_source_changes_the_residual`）：
把旧做法构造出的伪 adj 行喂给**同一个生产 kernel**，残差必须显著不同 ——
证明这不是一个"反正都差不多"的改动，不需要恢复旧代码也不需要第二份实现。
"""

import inspect
import sys
from pathlib import Path

import numpy as np

_TESTS_DIR = str(Path(__file__).resolve().parents[1])
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from unit.test_fr_residual_inviscid import _build_synthetic_mixed_mesh  # noqa: E402


def _flat_and_adj(order):
    from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry
    from autoflowcfd.core.fr_operators.volume_contract import compute_adj_j
    from autoflowcfd.fr.operators import generate_fr_operators

    mesh = _build_synthetic_mixed_mesh(order=order)
    ops = generate_fr_operators(order)
    flat = get_flat_face_geometry(mesh, ops)
    n_sps = mesh.n_sps_per_cell
    det = mesh.jacobians["det_jacs"].reshape(mesh.n_cells, n_sps)
    inv = mesh.jacobians["inv_jacs"].reshape(mesh.n_cells, n_sps, 3, 3)
    return mesh, flat, compute_adj_j(det, inv)


def test_owner_axis_exceeds_the_adj_j_axis_dimension():
    """旧表达式 `adj_j[oc, 0, owner_axis[f]]` 可达越界（判据 1）。"""
    mesh, flat, adj_j = _flat_and_adj(0)
    assert adj_j.shape[-2:] == (3, 3), (
        f"adj_j 末两维应当是 (3,3)，实际 {adj_j.shape[-2:]}")

    oax = np.asarray(flat.owner_axis)
    codes = np.asarray(flat.owner_cube_face)
    # 只看原生**四面体**面（[6,10)）：那里 owner_axis 存的是
    # excluded_vertex，取值可达 3。
    tet_faces = np.nonzero((codes >= 6) & (codes < 10))[0]
    assert tet_faces.size > 0, "合成网格应当含原生四面体面"
    assert int(oax[tet_faces].max()) >= 3, (
        f"原生四面体面的 owner_axis 最大值只有 {int(oax[tet_faces].max())} ——"
        f" 若 excluded_vertex 的编码约定改了，本文件的越界论证需要重写")


def test_the_two_metric_sources_are_not_interchangeable():
    """两个量在每个面上都差 O(1)（判据 2）。"""
    mesh, flat, adj_j = _flat_and_adj(0)
    oax = np.asarray(flat.owner_axis)
    oc = np.asarray(flat.owner_cell)
    exact = np.asarray(flat.owner_adj_row_exact)      # (n_faces, n_fp, 3)
    primary = np.nonzero(np.asarray(flat.owner_is_primary))[0]
    assert primary.size > 0

    rels = []
    for f in primary:
        # 旧做法：越界的轴先 clip（否则 numpy 直接报错；numba 当年是静默
        # 读相邻内存，这里只为把"即使不越界也是错的"这半边量出来）。
        ax = min(int(oax[f]), 2)
        used = adj_j[int(oc[f]), 0, ax]               # (3,)
        ref = exact[f].mean(axis=0)                   # (3,)
        denom = max(float(np.abs(exact[f]).max()), 1e-300)
        rels.append(float(np.abs(used - ref).max()) / denom)

    rels = np.asarray(rels)
    assert rels.min() > 0.1, (
        f"最小相对差只有 {rels.min():.3e} —— 两个量若真的接近，本文件的"
        f"『必须用精确行』论证就不成立了，需要重新评估")
    assert float(np.median(rels)) > 0.5, (
        f"相对差中位数 {float(np.median(rels)):.3e} 低于实测的 O(1) 量级")


def test_p0_kernel_no_longer_accepts_the_sp_grid_metric():
    """kernel 签名里不再有旧做法需要的那几个参数（判据 3）。"""
    from autoflowcfd.core.fr_residual.viscous_p0_kernel import (
        compute_viscous_interface_correction_p0_kernel as k,
    )

    src = inspect.getsource(k.py_func if hasattr(k, "py_func") else k)
    sig = src[:src.index(") -> np.ndarray:")]
    for gone in ("adj_j", "owner_axis", "neighbor_axis",
                 "owner_side", "neighbor_side"):
        assert gone not in sig, (
            f"P0 粘性 kernel 的签名里又出现了 {gone} —— 度量伴随行的唯一"
            f"正确来源是 owner_adj_row_exact/neighbor_adj_row_exact，"
            f"`owner_axis` 对原生面是复用槽位哑值")
    for need in ("owner_adj_row_exact", "neighbor_adj_row_exact"):
        assert need in sig, f"P0 粘性 kernel 必须接收 {need}"


def test_old_metric_source_changes_the_residual():
    """把旧做法的伪 adj 行喂给同一个生产 kernel，残差必须显著不同（判据 4）。

    状态取"均匀流 + 小扰动"：纯均匀流下全部跳跃为零、罚项恒零，残差与用
    哪个度量行无关（实测机器零），判据会空转。
    """
    from autoflowcfd.core.fr_operators.flux_kernels import (
        resolve_viscous_ip_constant,
    )
    from autoflowcfd.core.fr_residual.viscous_p0_kernel import (
        compute_viscous_interface_correction_p0_kernel as kernel,
    )

    mesh, flat, adj_j = _flat_and_adj(0)
    n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell
    n_fp = flat.n_fp
    rng = np.random.default_rng(7)

    rho, u, p = 1.225, 30.0, 101325.0
    Q = np.zeros((n_cells, n_sps, 5))
    Q[..., 0] = rho
    Q[..., 1] = u * (1.0 + 0.05 * rng.standard_normal((n_cells, n_sps)))
    Q[..., 2] = 0.05 * u * rng.standard_normal((n_cells, n_sps))
    Q[..., 3] = 0.05 * u * rng.standard_normal((n_cells, n_sps))
    Q[..., 4] = p
    grad_vel = 0.1 * rng.standard_normal((n_cells, n_sps, 3, 3))
    grad_T = 0.1 * rng.standard_normal((n_cells, n_sps, 3))
    mu_t = np.zeros((n_cells, n_sps))
    det_jacs = mesh.jacobians["det_jacs"].reshape(n_cells, n_sps)
    Q_ghost = np.ascontiguousarray(
        np.repeat(Q[np.asarray(flat.owner_cell), :1, :], n_fp, axis=1))
    bnd_adiabatic = np.zeros(flat.n_faces, dtype=np.bool_)
    c_ip = resolve_viscous_ip_constant(0)
    # `n_threads` 必须是紧邻调用前取的 `numba.get_num_threads()`：kernel
    # 用 `get_thread_id()` 索引 `(n_threads, ...)` 的 per-thread buffer，
    # 传小了就是越界写（第一版传了 1，numba 实际 8 线程 -> Windows
    # access violation 直接崩进程，不是 Python 异常）。
    import numba
    n_threads = numba.get_num_threads()

    def run(owner_rows, neighbor_rows):
        return kernel(
            Q, grad_vel, grad_T, mu_t,
            det_jacs, 1.8e-3, 0.72, 0.9,
            flat.owner_cell, flat.neighbor_cell, flat.is_boundary,
            flat.owner_is_primary, flat.neighbor_is_primary,
            owner_rows, neighbor_rows,
            flat.neighbor_src0_cell, flat.neighbor_src0_mat,
            flat.neighbor_src1_idx, flat.neighbor_src1_cell,
            flat.neighbor_src1_mat,
            flat.owner_src0_cell, flat.owner_src0_mat,
            flat.owner_src1_idx, flat.owner_src1_cell, flat.owner_src1_mat,
            flat.mixed_nb_partner, flat.mixed_nb_mask,
            flat.mixed_ow_partner, flat.mixed_ow_mask,
            Q_ghost, bnd_adiabatic, n_threads,
            flat.owner_cube_face, flat.neighbor_cube_face,
            flat.ref_area_weight,
            flat.boundary_extrap_native, flat.lift_native,
            flat.face_area, flat.cell_volume, c_ip,
        )

    def old_style(cell_idx, axis_slots):
        """旧做法读到的量，逐 FP 广播成 (n_faces, n_fp, 3)。"""
        ax = np.clip(np.asarray(axis_slots), 0, 2)
        rows = adj_j[np.asarray(cell_idx), 0, ax]       # (n_faces, 3)
        return np.ascontiguousarray(
            np.repeat(rows[:, None, :], n_fp, axis=1))

    R_fixed = run(np.ascontiguousarray(flat.owner_adj_row_exact),
                  np.ascontiguousarray(flat.neighbor_adj_row_exact))
    nc_safe = np.maximum(np.asarray(flat.neighbor_cell), 0)
    R_old = run(old_style(flat.owner_cell, flat.owner_axis),
                old_style(nc_safe, flat.neighbor_axis))

    scale = max(float(np.abs(R_fixed).max()), 1e-300)
    assert scale > 1e-30, "装置失效：修复后的残差本身就是零，判据空转"
    diff = float(np.abs(R_fixed - R_old).max()) / scale
    assert diff > 0.1, (
        f"两种度量来源给出的 P0 粘性界面校正只差 {diff:.3e}（相对）—— "
        f"若它们真的等价，本文件的论证与 2026-09-24 那次修复都需要重新"
        f"评估；实测应当是 O(1) 量级的差异")
