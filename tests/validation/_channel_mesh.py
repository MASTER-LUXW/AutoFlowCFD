"""构造结构化平面通道网格（Couette/Poiseuille 定量精度基准共用），以及
按面物理位置精确分类边界条件的 ghost state provider。

网格：结构化六面体网格，每个六面体沿主对角线（局部角点 (i,j,k)->(i+1,j+1,k+1)，
在结构化编号下恰好是该六面体全局节点编号最小->最大的角点）拆成 6 个四面体
——这个拆分规则保证相邻六面体共享面上用的对角线只取决于该四边形面自身 4 个
角点的全局编号（min->max），与 fr/collapsed_basis.py 等模块已经验证过的
"任意四边形面按全局编号 min->max 取对角线"规则完全一致，因此是全局自洽、
无缝拼接的四面体网格，不需要单独的六面体几何映射支持。
"""
from types import SimpleNamespace

import numpy as np

from autoflowcfd.grid.high_order.high_order_mesh import HighOrderMesh
from autoflowcfd.boundary.fr_ghost_state import BoundaryGhostStateProvider


def build_face_exact_ghost_provider(mesh, Lx, H, Lz, bc_by_plane, tol_scale=1e-6):
    """绕开 mesh.boundaries.groups（按 owner 单元索引分组）在角/棱单元上的
    固有歧义——真实发现：owner-cell 分组机制下，同一个单元只要同时拥有
    属于 2 个不同命名边界组的边界面（角落/棱上的单元，任何有限盒子网格
    都无法避免），tag_boundary_groups 会把该单元的**全部**边界面都打上
    最后一次匹配到的组的标签，与这些面各自真实所在的物理边界面无关。
    对本算例（nz=1 时每个单元同时贴着 z_min 和 z_max）这个歧义命中了
    108 个边界面里的 20 个（约18.5%），把部分 y=0/y=H 壁面误标成
    x 端的 OUTLET，导致求解在第 0 步就以 O(1e9) 量级发散——不是
    求解器本身的 bug，是这个手工测试网格绕过 BoundaryGhostStateProvider
    时必须自己保证的前提条件。这里直接按每个边界面自身的物理位置
    （face center 坐标）分类，是与 owner-cell 分组完全等价、但没有
    歧义的实现，因为 BoundaryGhostStateProvider 本身只需要一个
    per-face 的 group_code 数组，不要求这个数组必须来自
    tag_boundary_groups。
    """
    fc = mesh.face_connectivity
    tol = tol_scale * max(Lx, H, Lz, 1.0)
    group_code = np.full(fc.n_faces, -1, dtype=np.int32)
    code_to_config = {}
    bidx = fc.get_boundary_face_indices()
    centers = fc.center[bidx]

    def classify(c):
        if abs(c[1] - 0.0) < tol:
            return "wall_bottom"
        if abs(c[1] - H) < tol:
            return "wall_top"
        if abs(c[2] - 0.0) < tol:
            return "z_min"
        if abs(c[2] - Lz) < tol:
            return "z_max"
        if abs(c[0] - 0.0) < tol:
            return "x_min"
        if abs(c[0] - Lx) < tol:
            return "x_max"
        raise ValueError(f"boundary face center {c} not on any known domain plane")

    name_to_code = {}
    for f, c in zip(bidx, centers):
        name = classify(c)
        if name not in name_to_code:
            name_to_code[name] = len(name_to_code)
            code_to_config[name_to_code[name]] = bc_by_plane[name]
        group_code[f] = name_to_code[name]

    default_config = {"type": "FARFIELD", "Q_free": [1.225, 0.0, 0.0, 0.0, 101325.0]}
    return BoundaryGhostStateProvider(group_code, code_to_config, default_config)


class ProfileInletGhostProvider:
    """把某个边界组的 INLET 状态换成**逐通量点**的给定剖面。

    ## 为什么需要它

    `BoundaryGhostStateProvider` 的 INLET 配置是一个 `(5,)` 常量（整面
    统一），而无前缘奇点的 Blasius 验证算例要求入口携带**解析 Blasius
    剖面**（`u = U_inf f'(y / sqrt(nu x0 / U_inf))`）。底层
    `boundary/fr_ghost_state.py::inlet_ghost_state` 本来就支持
    `(n_fp, 5)` 的逐点形式（合成湍流入口 SEM 用的就是它），缺的只是一个
    按通量点物理坐标求值的粘合层。

    通量点的物理坐标复用生产代码里那一份唯一实现
    （`core/fr_solver/boundary.py::_compute_inlet_fp_positions`：用体积
    到面的外插矩阵作用在 `mesh.sps_coords` 上 —— 外插是线性的，对坐标
    分量和对流场分量是同一个矩阵运算），不在测试里另写一套参考坐标到
    物理坐标的映射。

    ## 委托而不是继承

    本类只截获 INLET 那一个组，其余面**原样委托**给包装的 provider。
    这样边界条件的全部既有语义（OUTLET/WALL/SYMMETRY、流出时的内部
    延拓、绝热壁掩码等）都不需要在这里重复一遍。
    """

    def __init__(self, inner, inlet_group_code, positions_by_face,
                 profile_fn, bc_type="INLET"):
        """
        Args:
            inner: 被包装的 `BoundaryGhostStateProvider`。
            inlet_group_code: 要替换的那个组的 `group_code` 值。
            positions_by_face: `{face_idx: (n_fp, 3) 物理坐标}`。
            profile_fn: `(n_fp, 3) -> (n_fp, 5)`，给出该批通量点处的
                原始变量状态。
            bc_type: `"INLET"`（默认，流出时退回内部延拓）或 `"FARFIELD"`
                （幽灵态就是给定态，流入/流出由黎曼求解器的特征分裂决定）。
                **上边界必须用 FARFIELD**：Blasius 在 `y=H` 处有向上的
                夹带速度，INLET 的"流出时用内部态"会把它退化成透射边界，
                拿不到"把精确解加在外边界上"这个效果。
        """
        self._inner = inner
        self._code = int(inlet_group_code)
        self._bc_type = str(bc_type).upper()
        if self._bc_type not in ("INLET", "FARFIELD"):
            raise ValueError(
                f"bc_type={bc_type!r} 只支持 INLET | FARFIELD")
        self._q_by_face = {
            int(f): np.ascontiguousarray(profile_fn(pos), dtype=np.float64)
            for f, pos in positions_by_face.items()
        }
        # 与被包装对象共享这两个属性：下游（绝热壁掩码构造、批量化路径）
        # 会直接读它们，缺了会静默走回逐面慢路径或错标绝热壁。
        self.group_code = inner.group_code
        self.code_to_config = inner.code_to_config

    def __call__(self, face_idx, Q_owner_fp, true_normal):
        from autoflowcfd.boundary.fr_ghost_state import inlet_ghost_state

        q_given = self._q_by_face.get(int(face_idx))
        if q_given is None:
            return self._inner(face_idx, Q_owner_fp, true_normal)
        if self._bc_type == "FARFIELD":
            # 远场幽灵态**就是**给定态（流入/流出由 AUSM+up 的特征分裂
            # 决定，见 `boundary/fr_ghost_state.farfield_ghost_state`）。
            # 不能走 `farfield_ghost_state`：那个函数把 `(5,)` 常量 tile
            # 成逐点，这里已经是逐点的 `(n_fp,5)`。
            return q_given
        return inlet_ghost_state(Q_owner_fp, q_given, true_normal)


def wrap_inlet_with_profile(provider, solver, inlet_plane_name,
                            bc_by_plane, profile_fn,
                            expect_types=("INLET",)):
    """把 `provider` 的 `inlet_plane_name` 组换成逐点剖面态。

    `inlet_plane_name` 在 `bc_by_plane` 里的类型必须落在 `expect_types`
    里；否则这是调用方的配置错误（把剖面挂到一个不消费它的面上不会报错、
    只会静默不生效），所以这里硬失败。

    `expect_types` 默认只接受 `INLET`（既有行为）。上边界那一档要显式传
    `("FARFIELD",)` —— 见 `ProfileInletGhostProvider` 的 `bc_type` 文档。
    """
    bc_type = bc_by_plane.get(inlet_plane_name, {}).get("type")
    if bc_type not in expect_types:
        raise ValueError(
            f"{inlet_plane_name} 在 bc_by_plane 里是 {bc_type!r}，"
            f"不在期望的 {expect_types} 里，给它挂剖面态不会生效")
    from autoflowcfd.core.fr_solver.boundary import (
        _compute_inlet_fp_positions,
    )

    target_cfg = bc_by_plane[inlet_plane_name]
    code = None
    for c, cfg in provider.code_to_config.items():
        if cfg is target_cfg:
            code = c
            break
    if code is None:
        raise ValueError(
            f"在 provider 里找不到 {inlet_plane_name} 对应的 group_code")
    is_target = np.asarray(provider.group_code) == code
    positions = _compute_inlet_fp_positions(
        solver, solver.mesh.face_connectivity, is_target)
    if not positions:
        raise ValueError(
            f"{inlet_plane_name} 组里没有 owner_is_primary 的边界面 —— "
            f"剖面入口不会生效")
    return ProfileInletGhostProvider(provider, code, positions, profile_fn,
                                     bc_type=bc_type)


class _MockNodes:
    def __init__(self, coords):
        self._coords = coords

    def get_coordinates(self):
        return self._coords


class _MockCells:
    def __init__(self, connectivity):
        self.connectivity = connectivity


class _MockBoundaries:
    def __init__(self, groups, bc_types):
        self.groups = groups
        self.bc_types = bc_types


def build_channel_mesh(order, nx, ny, nz, Lx, H, Lz):
    nx1, ny1, nz1 = nx + 1, ny + 1, nz + 1
    ii, jj, kk = np.meshgrid(np.arange(nx1), np.arange(ny1), np.arange(nz1), indexing="ij")
    node_id = (ii + nx1 * jj + nx1 * ny1 * kk).ravel()
    order_idx = np.argsort(node_id)  # 保证 node 数组下标本身就是全局编号
    xs = (ii.ravel() / nx * Lx)[order_idx]
    ys = (jj.ravel() / ny * H)[order_idx]
    zs = (kk.ravel() / nz * Lz)[order_idx]
    nodes = np.column_stack([xs, ys, zs])

    def gid(i, j, k):
        return i + nx1 * j + nx1 * ny1 * k

    tets = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                v0 = gid(i, j, k)
                v1 = gid(i + 1, j, k)
                v2 = gid(i + 1, j + 1, k)
                v3 = gid(i, j + 1, k)
                v4 = gid(i, j, k + 1)
                v5 = gid(i + 1, j, k + 1)
                v6 = gid(i + 1, j + 1, k + 1)
                v7 = gid(i, j + 1, k + 1)
                sub_tets = [
                    [v0, v1, v2, v6],
                    [v0, v2, v3, v6],
                    [v0, v3, v7, v6],
                    [v0, v7, v4, v6],
                    [v0, v4, v5, v6],
                    [v0, v5, v1, v6],
                ]
                tets.extend(sub_tets)

    tet_conn = np.array(tets, dtype=np.int32)
    n_tets = len(tet_conn)

    # 边界分组：按每个 tet 的 4 个节点是否有 >=3 个落在目标边界平面上
    # （即该 tet 恰有一个面完全位于边界平面），纯几何判据，不依赖对
    # 6-tet 分解组合规律的手工推导。
    tol = 1e-9 * max(Lx, H, Lz, 1.0)

    def face_on_plane(coord_vals, target):
        on_plane = np.abs(coord_vals - target) < tol
        return on_plane.sum(axis=1) >= 3

    tet_coords = nodes[tet_conn]  # (n_tets,4,3)
    groups = {
        "wall_bottom": np.flatnonzero(face_on_plane(tet_coords[:, :, 1], 0.0)),
        "wall_top": np.flatnonzero(face_on_plane(tet_coords[:, :, 1], H)),
        "z_min": np.flatnonzero(face_on_plane(tet_coords[:, :, 2], 0.0)),
        "z_max": np.flatnonzero(face_on_plane(tet_coords[:, :, 2], Lz)),
        "x_min": np.flatnonzero(face_on_plane(tet_coords[:, :, 0], 0.0)),
        "x_max": np.flatnonzero(face_on_plane(tet_coords[:, :, 0], Lx)),
    }
    bc_types = {name: "WALL" for name in groups}  # 占位，真正类型由 bc_overrides 决定

    mock_volume = SimpleNamespace(
        cell_count=n_tets,
        nodes=_MockNodes(nodes),
        cells=_MockCells(tet_conn),
        prism_cells=None,
        boundaries=_MockBoundaries(groups, bc_types),
    )

    mesh = HighOrderMesh(order=order)
    mesh.load_from_volume_mesh(mock_volume, build_faces=True)
    return mesh


def build_channel_mesh_prism(order, nx, ny, nz, Lx, H, Lz):
    """棱柱通道网格：(x,z) 平面 nx*nz 个矩形各拆 2 个三角形，沿 y 方向
    （壁面法向）整层挤出 ny 层，共 2*nx*nz*ny 个棱柱。

    背景（见项目记忆 tet_collapsed_coord_anisotropy）：四面体坍缩坐标
    P2 方案的参考轴 (a,b,c) 权重天然不对称，约 1/3 的四面体单元若主
    梯度方向压在单一参考轴上，残差会被放大 6-7 个数量级，与网格质量/
    尺度无关——这正是 AutoFlowCFD 网格架构本身要求近壁/高剪切区用棱柱
    （不用四面体）的原因：棱柱挤出方向 (c 轴) 用完全无权重的普通
    Legendre 基，且对直壁挤出物理 y 是 c 的精确线性函数，Couette/
    Poiseuille 这类沿壁面法向变化的解析解复合后是 c 的精确多项式，
    插值截断误差为零。验证近壁剪切物理必须用棱柱网格，不能用纯四面体。

    ## z 中面镜像对称性（nz 必须是偶数才精确，2026-09-18）

    挤出方向必须沿壁面法向 y（上一段的理由），所以**展向 z 只能是三角形
    方向**。原实现"所有矩形用同一条对角线方向拆分"因此让整张网格**对
    z 中面 z=Lz/2 不对称** —— 后果是 `w == 0` 不再是离散对称性：展向
    动量的残差在 w≡0 的场上也不为零，w 被持续受迫激发。

    **实测（Blasius 平板算例，nz=1）**：
      * 精确均匀场上 `max|R_w|` 相对来流量级 **3.01e-9**（各分量里最大），
        且与壁面 BC 无关——把全部幽灵态换成同一个自由来流态后逐位相同，
        所以是纯粹的自由流保持性/几何来源；
      * 22000 步后 `|w|max` 长到 0.64~0.79，即来流 30 m/s 的 2.1~2.6%，
        `u` 最小值甚至转负；
      * 同一算例的壁面摩阻因此从 400 步时的 -6.3% 掉到 22000 步的 -97%。

    改动：上半区（`2*k >= nz`）改用**反对角线**。镜像映射 z -> Lz-z 把
    矩形 (i,k) 映到 (i, nz-1-k)，并把主对角线映成反对角线，所以这样拆分
    之后整张网格对 z 中面精确镜像对称（实测 SP 点集对中面镜像的最大偏差
    从 nz=1 的 **6.1e-2** 降到 nz=2 的 **2.2e-16**）。

    ## **重要：这个改动并没有解决 w 的增长**（2026-09-18 实测，如实记录）

    做完镜像对称之后又量了两件事，都把"z 不对称是 w 的来源"这个假设
    **否掉了**：

    1. **自由流保持性底噪与对称性无关**。精确均匀场上 `max|R_w|` 相对
       来流量级：nz=1 是 3.01e-9，nz=2（已镜像对称）是 6.78e-9 —— 没有
       降到机器零，而且 `rho_u` 同样是 ~2e-9。所以那 ~1e-9 是**各分量
       共有的**底噪（量级约等于坍缩基算子 `V·diag·V^-1` 的条件数放大的
       舍入：cond(V)~1e7 × 1e-16），不是 z 方向特有的缺陷。
    2. **w 的增长几乎不变**。nx=16、CFL 0.1、2000 步：
           nz=1（不对称）  max|w| = 1.299  (4.33% of U)
           nz=2（已对称）  max|w| = 1.408  (4.69% of U)
       两者都在 250 步内就到 1%，随后饱和在 4~5%。

    所以 w 的放大机制**另有其因，尚未查明**。本改动保留的理由是它本身
    就是对的（准二维算例的网格本来就应当对展向中面精确对称，这是一条
    可验证的几何性质），但**不要**把它当成 w 问题的修复——那个问题仍然
    开放，`nz=1` 与 `nz=2` 都受影响。

    **nz 必须是偶数**：奇数时中间那一层矩形映到自身，而一条对角线不可能
    是自己的镜像，精确对称在原理上做不到。`nz=1` 因此**不是**"等效二维"
    的正确设置——要 w 恒为机器零请用 `nz=2`。本函数对奇数 `nz>1` 会打
    警告；`nz=1` 保留为合法取值（很多单元测试用它、且不关心展向对称性），
    但 Blasius 这类要求 w≡0 的验证算例必须用偶数。
    """
    if nz > 1 and nz % 2 == 1:
        import warnings

        warnings.warn(
            f"build_channel_mesh_prism: nz={nz} 是奇数，(x,z) 三角剖分"
            f"**无法**对 z 中面精确镜像对称（中间那层矩形映到自身，而一条"
            f"对角线不可能是自己的镜像）。后果是 w==0 不再是离散对称性，"
            f"展向动量会被持续受迫激发（实测均匀场上 max|R_w| 相对量级 "
            f"3e-9，22000 步后 |w| 长到来流的 2%+）。要 w 恒为机器零请用"
            f"偶数 nz。",
            RuntimeWarning, stacklevel=2,
        )
    nx1, ny1, nz1 = nx + 1, ny + 1, nz + 1

    def gid(i, j, k):
        return i + nx1 * j + nx1 * ny1 * k

    ii, jj, kk = np.meshgrid(np.arange(nx1), np.arange(ny1), np.arange(nz1), indexing="ij")
    node_id = (ii + nx1 * jj + nx1 * ny1 * kk).ravel()
    order_idx = np.argsort(node_id)
    xs = (ii.ravel() / nx * Lx)[order_idx]
    ys = (jj.ravel() / ny * H)[order_idx]
    zs = (kk.ravel() / nz * Lz)[order_idx]
    nodes = np.column_stack([xs, ys, zs])

    prisms = []
    for i in range(nx):
        for k in range(nz):
            # (x,z) 平面矩形 (i,k)-(i+1,k)-(i+1,k+1)-(i,k+1) 拆成两个三角形。
            #
            # **对角线方向按 z 中面镜像翻转（2026-09-18 真实缺陷修复）**：
            # 上半区（2*k >= nz）用反对角线。原来"所有矩形用同一条对角线"
            # 会让整张网格**对 z 中面不对称**，而准二维算例的网格本来
            # 就应当对展向中面精确对称。**注意这不是 w 增长问题的修复**
            # ——实测两者的 w 增长几乎相同，见本函数文档"这个改动并没有
            # 解决 w 的增长"一节。
            #
            # 翻转不破坏流形性：相邻矩形共享的都是完整的矩形边
            # （x=const 或 z=const），对角线是矩形内部的。
            if 2 * k < nz:
                tri_a = [(i, k), (i + 1, k), (i + 1, k + 1)]
                tri_b = [(i, k), (i + 1, k + 1), (i, k + 1)]
            else:
                # 反对角线：连 (i,k+1)-(i+1,k)
                tri_a = [(i, k), (i + 1, k), (i, k + 1)]
                tri_b = [(i + 1, k), (i + 1, k + 1), (i, k + 1)]
            for tri in (tri_a, tri_b):
                for j in range(ny):
                    v = [gid(ti, j, tk) for ti, tk in tri]
                    w = [gid(ti, j + 1, tk) for ti, tk in tri]
                    prisms.append(v + w)

    prism_conn = np.array(prisms, dtype=np.int32)
    n_prisms = len(prism_conn)

    tol = 1e-9 * max(Lx, H, Lz, 1.0)

    def face_on_plane(coord_vals, target):
        on_plane = np.abs(coord_vals - target) < tol
        return on_plane.sum(axis=1) >= 3

    prism_coords = nodes[prism_conn]  # (n_prisms,6,3)
    groups = {
        "wall_bottom": np.flatnonzero(face_on_plane(prism_coords[:, :, 1], 0.0)),
        "wall_top": np.flatnonzero(face_on_plane(prism_coords[:, :, 1], H)),
        "z_min": np.flatnonzero(face_on_plane(prism_coords[:, :, 2], 0.0)),
        "z_max": np.flatnonzero(face_on_plane(prism_coords[:, :, 2], Lz)),
        "x_min": np.flatnonzero(face_on_plane(prism_coords[:, :, 0], 0.0)),
        "x_max": np.flatnonzero(face_on_plane(prism_coords[:, :, 0], Lx)),
    }
    bc_types = {name: "WALL" for name in groups}

    empty_tets = np.zeros((0, 4), dtype=np.int32)
    mock_volume = SimpleNamespace(
        cell_count=n_prisms,
        nodes=_MockNodes(nodes),
        cells=_MockCells(empty_tets),
        prism_cells=_MockCells(prism_conn),
        boundaries=_MockBoundaries(groups, bc_types),
    )

    mesh = HighOrderMesh(order=order)
    mesh.load_from_volume_mesh(mock_volume, build_faces=True)
    return mesh
