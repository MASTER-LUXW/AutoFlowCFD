"""
AutoFlowCFD V2.0 - Solver Helper Functions

本模块包含 FRSolver 的辅助函数：WMLES 壁面剪应力建模的接入。目的是减少
fr_solver.py 的代码复杂度。

V2.0 二次评审修复记录（T-05，三个独立 bug 叠加，见
ProjectFiles/V2.0/6_整体专家组二次评审.md）：
1. 壁面 SP/FP 提取此前三条路径全部失效（`solver.boundary_manager`/
   `mesh.boundary_faces` 属性都不存在，几何自动探测直接返回空数组），
   `apply_wmles_wall_stress` 因此永远是 no-op。改用与
   `core/fr_solver_boundary.py`（BD-01 幽灵态构建）完全相同的机制——
   `grid.face_connectivity.tag_boundary_groups` + `mesh.boundary_bc_types`
   ——识别 WALL 类型边界面，这是本代码库真正可用、已验证的边界分组
   查询路径，不是重新发明一套。
2. `apply_wmles_wall_stress` 试图写 `solver.residual[...]`，但
   `FRSolver` 从未定义过这个属性，必然 `AttributeError`。改为返回一个
   `(n_cells,n_sps,5)` 修正数组，用与本代码库其余面校正项完全一致的
   机制（`ops.boundary_extrap_*` 外插到 FP、`_distribute_from_face`
   按校正函数导数投影回 SPs、除以 `det_jacs`）计算，而不是直接摸底层
   状态。
3. 施加时机：此前在 `apply_turbulence_corrections()` 里调用，而该函数
   在 `fr_solver.py::step()` 中排在 `self.state.U = U_new_flat...`
   **之后**——本该在这一步生效的壁面应力源项在时间推进完成后才计算，
   对本步毫无影响，架构上不可能生效。改为在 `compute_viscous_residual()`
   内部计算并叠加到返回的粘性残差数组上，随其余残差一起参与时间积分
   （见 fr_solver.py::compute_viscous_residual 调用处）。
"""

from typing import Any, Optional

import numpy as np
from loguru import logger


def compute_wmles_wall_stress_correction(solver: Any) -> Optional[np.ndarray]:
    """计算 WMLES 壁面剪应力对动量残差的修正贡献。

    对每个 WALL 类型边界面：用 owner 单元的坍缩坐标模态基外插（与
    fr_viscous_flux.py/fr_residual_inviscid.py 同一套算子）把速度场、
    壁面距离外插到该面的 Flux Points，换算切向速度，交给
    `WMLESModel.compute_wall_shear_stress` 算出 tau_w，再用校正函数
    导数 `_distribute_from_face` 投影回 owner 单元的 SPs——这是本代码库
    里所有"面上的物理量 -> 单元残差贡献"共用的标准机制（无粘/粘性残差
    都是同一套），不是另起一套简化路径。

    Returns:
        (n_cells, n_sps, 5) 的动量修正数组（只有分量 1:4 非零），
        无 WMLES 模型/无 WALL 边界/尚未构建面连接关系时返回 None。
    """
    if solver.wmles_model is None:
        return None
    mesh = solver.mesh
    fc = mesh.face_connectivity
    if fc is None or mesh.face_flux_points is None:
        return None
    if solver.wall_distance is None:
        logger.warning("WMLES requires wall distance field but none is available; skipping wall stress")
        return None

    from autoflowcfd.grid.face_connectivity import tag_boundary_groups
    from autoflowcfd.core.fr_residual_inviscid import _distribute_from_face

    group_code, name_to_code = tag_boundary_groups(fc, mesh.boundary_groups or {})
    bc_types = mesh.boundary_bc_types or {}
    wall_codes = {code for name, code in name_to_code.items() if bc_types.get(name, "") == "WALL"}
    if not wall_codes:
        return None
    is_wall_face = np.isin(group_code, list(wall_codes))
    if not np.any(is_wall_face):
        return None

    n_cells, n_sps, n_vars = solver.state.U.shape
    n_prism = mesh.n_prism_cells
    n1d = mesh.n_points_1d
    ops = solver.ops
    Q = solver.state.Q
    det_jacs = mesh.jacobians["det_jacs"].reshape(n_cells, n_sps)

    def extrap_to_face(cell: int, field: np.ndarray, axis: int, side: float) -> np.ndarray:
        E = ops.boundary_extrap_prism[(axis, side)] if cell < n_prism else ops.boundary_extrap_tet[(axis, side)]
        trailing = field.shape[1:]
        flat = E @ field.reshape(field.shape[0], -1)
        return flat.reshape((E.shape[0],) + trailing)

    correction = np.zeros((n_cells, n_sps, n_vars))
    n_wall_faces_applied = 0
    y_plus_samples = []

    for f in np.nonzero(is_wall_face)[0]:
        ffp = mesh.face_flux_points[f]
        if not ffp.owner_is_primary:
            continue
        owner_cell = int(fc.owner_cell[f])
        axis, side = ffp.owner_axis, ffp.owner_side

        Q_fp = extrap_to_face(owner_cell, Q[owner_cell], axis, side)  # (n_fp,5)
        wd_fp = extrap_to_face(owner_cell, solver.wall_distance[owner_cell][:, None], axis, side)[:, 0]
        wd_fp = np.maximum(wd_fp, 1e-8)

        rho_fp = Q_fp[:, 0]
        vel_fp = Q_fp[:, 1:4]
        normal = ffp.true_normal  # (n_fp,3) 指向域外
        vel_n = np.sum(vel_fp * normal, axis=1, keepdims=True)
        vel_tangent = vel_fp - vel_n * normal

        tau_w = solver.wmles_model.compute_wall_shear_stress(
            u_tangent=vel_tangent, y_dist=wd_fp, rho=rho_fp, method="iterative"
        )  # (n_fp,3)，方向与切向速度同向（即"流体感受到的阻力"方向相反）

        y_plus_samples.append(getattr(solver.wmles_model, "y_plus", np.array([])))

        # 剪应力对流体做负功（阻力），换算成动量源项：S_mom = -tau_w * area，
        # 用与其余面校正项完全一致的 g_prime 投影/除以 det_jacs 组装方式。
        momentum_fp = -tau_w * ffp.true_area_weight[:, None]
        g_prime = ops.g_left if side < 0 else ops.g_right
        contrib = _distribute_from_face(momentum_fp, n1d, axis, g_prime)  # (n_sps,3)
        correction[owner_cell, :, 1:4] += contrib / det_jacs[owner_cell][:, None]
        n_wall_faces_applied += 1

    if n_wall_faces_applied == 0:
        return None

    if y_plus_samples:
        all_yplus = np.concatenate([a for a in y_plus_samples if a.size > 0]) if any(
            a.size > 0 for a in y_plus_samples
        ) else np.array([])
        if all_yplus.size > 0:
            logger.debug(
                f"WMLES wall stress applied to {n_wall_faces_applied} wall faces: "
                f"y+ min={all_yplus.min():.1f}, max={all_yplus.max():.1f}, mean={all_yplus.mean():.1f}"
            )


def resolve_backend_type(backend: str) -> str:
    """解析 FRSolver 构造参数 backend 的实际生效后端类型 (B-01)。

    此前这里构造一个 CUDABackend 实例、调用一次 .initialize()，之后
    solver.backend 在全文件里再也不会被引用——backend="gpu" 对实际计算
    路径没有任何影响，只改变构造时打印哪几行日志（V2.0 专家评审报告
    B-01 项指出的问题）。真正的 GPU 加速路径见 core/backend/fr_gpu_p0.py：
    目前只对 P0（Order Continuation 最低阶/有限体积）无粘残差实现了真实
    CUDA kernel（忠实移植已验证正确的 CPU 版
    _compute_inviscid_residual_fv_p0，见该模块文档），P>=1（真正的高阶
    FR，坍缩坐标度量张量外插+逐面记录字典键控分发）尚未实现 GPU 版本，
    如实回退 CPU、如实记录日志，不再静默构造一个从未被使用的对象。是否
    真正走 GPU 由 compute_inviscid_residual() 在每次调用时按
    solver.backend_type 与当前网格阶数共同判断（阶数延续期间会在多个
    阶数之间切换）。

    Returns:
        实际生效的后端类型字符串（"cpu" 或 "gpu"）——gpu 请求但硬件/
        NUMBA_ENABLE_CUDASIM 不可用时会如实回退为 "cpu"。
    """
    backend_type = backend.lower()
    if backend_type == "gpu":
        from .backend.fr_gpu_p0 import gpu_p0_available
        if gpu_p0_available():
            logger.info(
                "GPU (CUDA) backend available - will accelerate the P0 finite-volume inviscid "
                "residual only (order continuation warm-up stage); P>=1 orders still run on CPU "
                "(see core/backend/fr_gpu_p0.py for scope)."
            )
        else:
            logger.warning(
                "GPU backend requested but no CUDA device found (and NUMBA_ENABLE_CUDASIM not set) "
                "- falling back to CPU entirely"
            )
            backend_type = "cpu"

    if backend_type == "cpu":
        logger.info("CPU Backend (Numba) initialized")

    return backend_type

    return correction
