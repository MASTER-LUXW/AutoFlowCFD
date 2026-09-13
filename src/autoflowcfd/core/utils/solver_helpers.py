"""
AutoFlowCFD V2.0 - Solver Helper Functions

本模块包含 FRSolver 的辅助函数：WMLES 壁面剪应力建模的接入。目的是减少
fr_solver.py 的代码复杂度。

native tet + WMLES 真实 bug 修复（2026-09-02，见
ProjectFiles/V2.0/19_重大问题修复-湍流模型跨后端逐项排查与完全分布式加载补齐.md
第6.5节）：`compute_wmles_wall_stress_correction`此前对四面体单元恒用
坍缩坐标专属的`ops.boundary_extrap_tet[(axis,side)]`/1D Radau/VCJH
修正函数导数（`_distribute_from_face`），未区分`tet_basis_mode`。
native 面的`owner_axis`/`owner_side`存的是复用的 excluded_vertex/
哑值（见 face_kernels.py::FlatFaceGeometry 字段文档），拿去查坍缩
坐标专用的矩阵字典在语义上是错的。现在改用`owner_cube_face`
（>=6 即 native，与 inviscid_kernel.py 界面项分派同一个判据）分派
到 native 专属算子：自身面外插改用`boundary_extrap_native_tet`
（按需 pad 到全局 n_sps 宽度），面修正项改用 DG 提升算子
`lift_native_tet_padded`替代`_distribute_from_face`。

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

真实 bug 修复（#9，V2.0 专家组盲审第4轮，2026-08-28，"WMLES 假滑移边界"）：
此前 WALL 边界的 ghost state 恒用严格无滑移镜像构造（is_no_slip=True），
BR1/LDG 粘性通量因此已经从（本项目近壁网格通常无法真正分辨粘性底层的）
解析速度梯度算出一个虚假壁面剪应力，本函数算出的 tau_w 又作为额外动量
源项叠加在其上——两者同时生效，是真正的双重计权，不是设计如此。按
WMLES 文献标准做法（wall model 的 tau_w 应"取代"而非"叠加"解析梯度剪
应力，见 Kawai & Larsson 团队 wmles.umd.edu 页面明确用词"replace"，以及
Kang et al. 2024 arXiv:2405.15899 在 DG 类弱式框架下的对应公式与"不会
双重计权"论证）修复：`fr_solver/boundary.py::build_boundary_ghost_provider`
现在在 WMLES 激活时把 WALL 组的 ghost state 构造改为 is_no_slip=False
（与 SLIP_WALL 同一套构造：切向速度与内部值无跳跃），让该面对 BR1/LDG
粘性通量的切向梯度贡献退化为零——本函数算出的 tau_w 现在是该面切向应力
的唯一来源，不再与一个虚假的解析梯度剪应力共存。
"""

from typing import Any, Dict, Optional

import numpy as np
from loguru import logger


def compute_wmles_wall_stress_correction(
    solver: Any, flat_face_override: Optional[Any] = None,
) -> Optional[np.ndarray]:
    """计算 WMLES 壁面剪应力对动量残差的修正贡献。

    对每个 WALL 类型边界面：用 owner 单元的坍缩坐标模态基外插（与
    fr_viscous_flux.py/fr_residual_inviscid.py 同一套算子）把速度场、
    壁面距离外插到该面的 Flux Points，换算切向速度，交给
    `WMLESModel.compute_wall_shear_stress` 算出 tau_w，再用校正函数
    导数 `_distribute_from_face` 投影回 owner 单元的 SPs——这是本代码库
    里所有"面上的物理量 -> 单元残差贡献"共用的标准机制（无粘/粘性残差
    都是同一套），不是另起一套简化路径。

    Args:
        solver: 需要 `.wmles_model`/`.wall_distance`/`.mesh`/`.ops`/
            `.state.U`/`.state.Q`/`.boundary_ghost_provider`。
        flat_face_override: 显式传入时优先使用（本代码库其余面残差
            函数同名参数同一个约定，见 `compute_scalar_convection_
            residual` 文档），不再调用`get_flat_face_geometry(solver.
            mesh, solver.ops)`——真实修复（2026-09-02，排查分布式 WMLES
            支持时发现）：此前本函数直接用 `mesh.face_flux_points[f]`
            这个全局对象列表逐面取值，分布式场景下 `solver.mesh` 是
            compact 索引空间的适配器，`f` 是 compact 面索引，拿去索引
            全局 `face_flux_points` 列表在语义上是错的（与
            `_compute_omega_wall_target` 此前的同一类 bug 完全一样）。
            现在改用 `FlatFaceGeometry` 的批量数组字段（`owner_cell`/
            `owner_axis`/`owner_side`/`owner_is_primary`/`true_normal`/
            `true_area_weight`），它们本来就与残差组装用的其余代码
            共享同一个索引空间（分布式场景下传 `dist_fc.base_flat`
            即可，与 `distributed_compute_viscous_residual` 已经在用
            的 `flat_face_override=dist_fc.base_flat` 同一个对象）。

    Returns:
        (n_cells, n_sps, 5) 的动量修正数组（只有分量 1:4 非零），
        无 WMLES 模型/无 WALL 边界/尚未构建面连接关系时返回 None。
    """
    if solver.wmles_model is None:
        return None
    if solver.wall_distance is None:
        logger.warning("WMLES requires wall distance field but none is available; skipping wall stress")
        return None

    # WALL 面识别：真实修复（2026-09-02）——此前这里用 `tag_boundary_
    # groups_for_mesh(mesh, fc)` + `mesh.boundary_bc_types` 独立重新
    # 推导一遍边界分组，只依赖 `solver.mesh` 是否携带完整全局边界几何
    # 信息，分布式场景下不成立。`solver.boundary_ghost_provider`（构建
    # 时调的正是同一个 `tag_boundary_groups_for_mesh`，见
    # fr_solver/boundary.py::build_boundary_ghost_provider）已经在单机/
    # 全部分布式后端里正确初始化好，且它的 `group_code` 天然就是与
    # 本函数需要的面索引空间一致（分布式场景下已按 rank 切好），直接
    # 复用而不是重新推导——与 `_compute_wall_dirichlet_face_mask`
    # （transport.py）识别 WALL 面的方式完全同一个模式。
    # 注（2026-09-12，排查 transport.py::_compute_wall_dirichlet_face_mask
    # 的滑移壁 omega 误处理 bug 时曾经尝试、随即撤销的改法，记录下来避免
    # 后续重蹈覆辙）：曾在这里同步加上"排除 is_no_slip=False 的 WALL 编码"，
    # 类比 SST omega 壁面处理的修复——但真实回归测试（`test_wmles_native_
    # tet.py`/`test_distributed_compute_residual.py` 的 WMLES 用例）决定性
    # 证伪：`fr_solver/boundary.py::build_boundary_ghost_provider` 对
    # WMLES **激活时**的 WALL 组本来就故意把 `is_no_slip` 设为 `wmles_model
    # is None`（即 WMLES 激活时恒为 False）——这是 2026-08-28（#9）修复
    # "WMLES 假滑移边界"时的既有设计：让真实固壁在 WMLES 模式下退化成
    # is_no_slip=False 的 ghost state 构造，使 tau_w 成为该面切向应力的
    # 唯一来源、避免与解析剪应力双重计权，不是"这面墙其实是滑移远场"的
    # 语义。本函数只在 `solver.wmles_model is not None`（上面已 return None
    # 排除）时才会执行到这里，此时 WALL 组永远是真实固壁、`is_no_slip`
    # 恒为 False 只是内部机制——加上这个排除会让 `wall_codes` 恒为空，
    # WMLES 壁面剪应力修正完全失效。omega 壁面处理与 WMLES 壁面应力处理
    # 面对的是同一个 `is_no_slip` 字段在两种不同上下文下的不同语义，不能
    # 共用同一条排除逻辑——本函数保留原始行为（任何 WALL 类型编码都计入）。
    provider = getattr(solver, "boundary_ghost_provider", None)
    group_code = getattr(provider, "group_code", None)
    code_to_config = getattr(provider, "code_to_config", None)
    if group_code is None or code_to_config is None:
        return None
    wall_codes = [code for code, cfg in code_to_config.items() if cfg.get("type") == "WALL"]
    if not wall_codes:
        return None
    is_wall_face = np.isin(group_code, wall_codes)
    if not np.any(is_wall_face):
        return None

    from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry
    from autoflowcfd.core.fr_residual.inviscid import _distribute_from_face

    mesh = solver.mesh
    ops = solver.ops
    flat = flat_face_override if flat_face_override is not None else get_flat_face_geometry(mesh, ops)

    n_cells, n_sps, n_vars = solver.state.U.shape
    n_prism = mesh.n_prism_cells
    n1d = mesh.n_points_1d
    Q = solver.state.Q
    det_jacs = mesh.jacobians["det_jacs"].reshape(n_cells, n_sps)

    # native 四面体（`tet_basis_mode="native"`）支持（真实 bug 修复，
    # 2026-09-02，见本函数模块文档"native tet + WMLES"一节）：此前
    # `extrap_to_face`/下方的 `_distribute_from_face` 对四面体单元恒用
    # 坍缩坐标专属算子（`ops.boundary_extrap_tet[(axis,side)]`/1D
    # Radau/VCJH 修正函数导数），而 native 面的 `owner_axis`/`owner_side`
    # 存的是复用的 excluded_vertex/哑值（见 face_kernels.py::
    # FlatFaceGeometry 字段文档），拿去查坍缩坐标专用的矩阵字典在语义上
    # 是错的（要么查到无意义的值，要么伪键不在 6 个合法 (axis,side)
    # 组合里直接 KeyError）。改用 `owner_cube_face`（>=6 即 native，
    # excluded_vertex=code-6，与 inviscid_kernel.py 界面项分派同一个
    # 判据）分派到 native 专属算子：自身面外插矩阵改用
    # `boundary_extrap_native_tet[excluded_vertex]`（按需 pad 到全局
    # n_sps 宽度，与 face_kernels.py 界面项 kernel 同一个 pad 约定）；
    # 面修正项改用 DG 提升算子 `lift_native_tet_padded[excluded_vertex]`
    # 替代 `_distribute_from_face`（native 单纯形基没有"坍缩计算方向"，
    # 1D 修正函数分布机制不适用，见 native_simplex_basis.py::
    # build_native_tet_lift"弱形式提升定义"）。棱柱（`cell < n_prism`）
    # 恒用坍缩坐标算子，不受 tet_basis_mode 影响（native 只针对四面体）。
    _padded_extrap_native_cache: Dict[int, np.ndarray] = {}

    def _get_padded_extrap_native(excluded_vertex: int) -> np.ndarray:
        if excluded_vertex not in _padded_extrap_native_cache:
            from autoflowcfd.fr.native_tet_padding import pad_native_tet_matrix_to_global
            _padded_extrap_native_cache[excluded_vertex] = pad_native_tet_matrix_to_global(
                ops.boundary_extrap_native_tet[excluded_vertex], n_sps, pad_axes=(1,)
            )
        return _padded_extrap_native_cache[excluded_vertex]

    def extrap_to_face(cell: int, field: np.ndarray, axis: int, side: float, cube_face: int) -> np.ndarray:
        if cube_face >= 6:
            E = _get_padded_extrap_native(cube_face - 6)
        elif cell < n_prism:
            E = ops.boundary_extrap_prism[(axis, side)]
        else:
            E = ops.boundary_extrap_tet[(axis, side)]
        trailing = field.shape[1:]
        flat_field = E @ field.reshape(field.shape[0], -1)
        return flat_field.reshape((E.shape[0],) + trailing)

    correction = np.zeros((n_cells, n_sps, n_vars))
    n_wall_faces_applied = 0
    y_plus_samples = []

    for f in np.nonzero(is_wall_face)[0]:
        if not flat.owner_is_primary[f]:
            continue
        owner_cell = int(flat.owner_cell[f])
        axis, side = int(flat.owner_axis[f]), float(flat.owner_side[f])
        cube_face = int(flat.owner_cube_face[f])

        Q_fp = extrap_to_face(owner_cell, Q[owner_cell], axis, side, cube_face)  # (n_fp,5)
        wd_fp = extrap_to_face(
            owner_cell, solver.wall_distance[owner_cell][:, None], axis, side, cube_face,
        )[:, 0]
        wd_fp = np.maximum(wd_fp, 1e-8)

        rho_fp = Q_fp[:, 0]
        vel_fp = Q_fp[:, 1:4]
        normal = flat.true_normal[f]  # (n_fp,3) 指向域外
        vel_n = np.sum(vel_fp * normal, axis=1, keepdims=True)
        vel_tangent = vel_fp - vel_n * normal

        tau_w = solver.wmles_model.compute_wall_shear_stress(
            u_tangent=vel_tangent, y_dist=wd_fp, rho=rho_fp, method="iterative"
        )  # (n_fp,3)，方向与切向速度同向（即"流体感受到的阻力"方向相反）

        y_plus_samples.append(getattr(solver.wmles_model, "y_plus", np.array([])))

        # 剪应力对流体做负功（阻力），换算成动量源项：S_mom = -tau_w * area，
        # 用与其余面校正项完全一致的 g_prime 投影/除以 det_jacs 组装方式
        # （native 面改用 DG 提升算子，见上方 extrap_to_face 同一处
        # native 分支说明）。
        momentum_fp = -tau_w * flat.true_area_weight[f][:, None]
        if cube_face >= 6:
            contrib = ops.lift_native_tet_padded[cube_face - 6] @ momentum_fp  # (n_sps,3)
        else:
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

    return correction


def resolve_backend_type(backend: str) -> str:
    """解析 FRSolver 构造参数 backend 的实际生效后端类型 (B-01)。

    GPU 检测统一使用 CuPy（替代此前的 numba.cuda）。真正的 GPU 加速路径
    见 core/gpu/ 模块：P0 无粘残差已实现 CuPy RawKernel（
    core/gpu/gpu_p0_inviscid.py），P>=1 高阶 FR 路径逐步 GPU 化中。
    是否真正走 GPU 由 compute_inviscid_residual() 在每次调用时按
    solver.backend_type 与当前网格阶数共同判断。

    Returns:
        实际生效的后端类型字符串（"cpu" 或 "gpu"）——gpu 请求但 CuPy
        不可用时会如实回退为 "cpu"。
    """
    backend_type = backend.lower()
    if backend_type == "gpu":
        from ..gpu import gpu_available
        if gpu_available:
            from ..gpu import get_device_info
            info = get_device_info()
            logger.info(
                f"GPU (CuPy) backend available - device: {info.get('name', 'unknown')}, "
                f"compute capability: {info.get('compute_capability', 'unknown')}. "
                f"P0 inviscid residual accelerated via CuPy RawKernel; "
                f"P>=1 high-order FR GPU path available (see core/gpu/ for scope)."
            )
        else:
            logger.warning(
                "GPU backend requested but CuPy/CUDA not available "
                "- falling back to CPU entirely. "
                "Install with: pip install cupy-cuda12x"
            )
            backend_type = "cpu"

    if backend_type == "cpu":
        logger.info("CPU Backend (Numba) initialized")

    return backend_type
