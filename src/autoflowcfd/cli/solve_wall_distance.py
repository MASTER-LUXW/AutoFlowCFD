"""求解命令的壁面距离场计算辅助函数 —— 从 solve_helpers.py 拆出，控制单文件行数。

见 solve_helpers.py 文档说明整体拆分结构。
"""

import click


def wall_nodes_from_boundary_faces(volume_data, bm):
    """从 WALL 组的**边界面**取出真正位于壁面上的节点索引。

    **这是 2026-09-15 发现的一处一阶物理错误的修复。** 原实现是：

        indices = bm.get_node_indices(bc_name)
        if len(indices) > 0 and np.max(indices) >= n_nodes:
            ...把 indices 当单元索引，转成节点...
        else:
            valid = indices[indices < n_nodes]   # 把 indices 当**节点**索引
            wall_nodes.update(valid)

    `BoundaryMap.groups` 按其自身文档存的**恒定是单元索引**
    （`{boundary_name: cell_indices_array}`，两条生产路径——
    `mesh_boundary.map_surface_boundaries` 与 `map_boundaries_by_geometry`
    ——都如此声明）。那个 `max(indices) >= n_nodes` 判据是在猜一件本来
    已经确定的事，而且**恰好只在 WALL 组上猜错**：壁面的边界单元就是
    边界层棱柱，占据单元索引的低位区间 `[0, n_prism)`，而两张真实网格
    都满足 `n_prism < n_nodes`，于是判据为假、单元索引被当成节点索引：

        cube_demo : body  range=[0,136974]  n_nodes=187702  -> 猜错
        plate_demo: body  range=[0, 65235]  n_nodes= 88496  -> 猜错
        （tunnel/inlet/outlet 贴四面体、索引超过 n_nodes，反而猜对了）

    后果是壁距场变成"到 13048 个按编号散布在全域的任意节点的距离"。
    plate_demo 实测 max 壁距 0.976 m，而到平板的真实最远距离是 4.359 m。
    SST 的 F1/F2 混合、omega 壁面目标值、nu_t 限幅、DDES/IDDES 长度尺度
    与 WMLES 全部由壁距驱动，所以这不是精度问题而是物理错误。

    **为什么用边界面而不是"单元的全部节点"**：一个边界层棱柱有 3 个节点
    在壁面上、3 个在第一层之外。把整个单元的节点都算作壁面节点会让壁距
    在近壁虚胖一层（第一层外侧节点的壁距变成 0 而不是真实的第一层厚度
    ~1e-5 m），而近壁正是 SST 最敏感的区域。取边界面的节点是精确的。

    已知的边际情形：`map_boundaries_by_geometry` 逐**面**匹配之后把结果
    折叠成 `{owner_cell: group}`，所以同一个单元若同时拥有属于不同组的
    边界面（只可能发生在组与组的交界棱上），这里会把它的全部边界面都
    计入。相对于原缺陷这是可忽略的过包含，且只影响交界棱一圈。

    Returns:
        (wall_node_indices, n_wall_faces)：前者是去重排序后的节点索引
        （int64），后者是参与统计的边界面数（供调用方打印/校验）。
    """
    import numpy as np

    n_cells = volume_data.cell_count
    wall_cells = set()
    for bc_name, bc_type in bm.bc_types.items():
        if bc_type != 'WALL' or not bm.has_boundary(bc_name):
            continue
        idx = np.asarray(bm.get_cell_indices(bc_name))
        if idx.size == 0:
            continue
        # 不静默截断：越界索引说明 BoundaryMap 与这份体网格不是一对
        if int(idx.max()) >= n_cells:
            raise click.ClickException(
                f"边界组 '{bc_name}' 的单元索引最大值 {int(idx.max())} 超出体"
                f"网格单元数 {n_cells}——BoundaryMap 与体网格不匹配，拒绝"
                f"继续（壁面距离场会整体错位）。"
            )
        wall_cells.update(int(c) for c in idx)

    if not wall_cells:
        return np.empty(0, dtype=np.int64), 0

    faces = volume_data.ensure_faces_exist()
    if faces.node_connectivity is None:
        raise click.ClickException(
            "体网格的面数据缺少 node_connectivity，无法从边界面取壁面节点"
            "——不能退回'把单元全部节点当壁面'（那会让近壁壁距虚胖一层，"
            "见 wall_nodes_from_boundary_faces 文档）。"
        )
    bidx = faces.get_boundary_face_indices()
    if len(bidx) == 0:
        return np.empty(0, dtype=np.int64), 0
    owner = faces.connectivity[bidx, 0]
    keep = np.fromiter((int(o) in wall_cells for o in owner), dtype=bool,
                       count=len(owner))
    sel = bidx[keep]
    if len(sel) == 0:
        return np.empty(0, dtype=np.int64), 0
    nodes = faces.node_connectivity[sel].ravel()
    nodes = nodes[nodes >= 0]
    return np.unique(nodes).astype(np.int64), int(len(sel))


def compute_wall_distance_for_solver(solver, volume_data, use_eikonal=False):
    """
    为求解器计算壁面距离场。

    Args:
        solver: FRSolver实例
        volume_data: load_mesh_for_solver 已经加载好的 VolumeMeshData - 直接
            复用，不重新解析一遍输入文件（这里以前是重新按 input_file 路径
            读一遍 .pkl，且只认 .pkl，.nas 体网格输入会直接跳过整个壁面距离
            计算、静默退化成"简化估计" - 现在 load_mesh_for_solver 两条路径
            都已经把 volume_data 解析好，直接传进来即可，同时对 .pkl/.nas
            两种输入路径都正确）
        use_eikonal: 是否使用 Eikonal 方程求解
    """
    import numpy as np

    turb_model = getattr(solver, 'turb_model_name', '').lower()
    if turb_model not in ['sst', 'ddes', 'iddes', 'wmles', 'les']:
        print(f"   ℹ️  Turbulence model '{turb_model}' does not require wall distance")
        return

    try:
        if volume_data is not None and hasattr(volume_data, 'boundaries'):
            print("\n🔍 Computing wall distance field...")

            bm = volume_data.boundaries
            n_nodes = volume_data.node_count

            # 壁面节点取自 WALL 组的**边界面**（2026-09-15 修复一处一阶
            # 物理错误——原实现用 `max(indices) >= n_nodes` 猜 BoundaryMap
            # 存的是单元还是节点索引，而它按契约恒为单元索引，那个判据
            # 恰好只在 WALL 组上猜错。完整推导、实测数字与"为什么不能用
            # 单元的全部节点"见 `wall_nodes_from_boundary_faces` 文档）。
            wall_indices_arr, n_wall_faces = wall_nodes_from_boundary_faces(
                volume_data, bm)
            wall_nodes = wall_indices_arr
            for bc_name, bc_type in bm.bc_types.items():
                if bc_type == 'WALL' and bm.has_boundary(bc_name):
                    print(f"   - Boundary '{bc_name}': "
                          f"{len(bm.get_cell_indices(bc_name))} wall cells")
            if len(wall_nodes) > 0:
                print(f"   {n_wall_faces} wall boundary faces -> "
                      f"{len(wall_nodes)} unique wall nodes")

            if len(wall_nodes) > 0:
                wall_indices = wall_nodes
                mesh_nodes = volume_data.nodes.get_coordinates()

                print(f"   Total unique wall nodes: {len(wall_indices)}")

                if use_eikonal:
                    # 只在真的要用 Eikonal 时才构建邻接表 - 这是一份对大网格
                    # 有实打实开销的图结构，KD-Tree 路径完全不需要它，没有
                    # 理由在默认路径上白白多算一遍。
                    print(f"   Building node adjacency graph for Eikonal solver...")
                    from autoflowcfd.grid.connectivity.node_connectivity import build_node_adjacency

                    tet_conn = volume_data.cells.connectivity if volume_data.cells else None
                    prism_conn = volume_data.prism_cells.connectivity if volume_data.prism_cells else None
                    connectivity = build_node_adjacency(
                        n_nodes, tet_connectivity=tet_conn, prism_connectivity=prism_conn
                    )
                    print(f"   Computing distances using Eikonal (graph-Dijkstra approx) solver...")
                    solver.compute_wall_distance_field(
                        mesh_nodes, wall_indices, connectivity=connectivity, use_eikonal=True
                    )
                else:
                    print(f"   Computing distances using KD-Tree...")
                    solver.compute_wall_distance_field(mesh_nodes, wall_indices)

                print(f"   ✅ Wall distance field computed successfully!\n")
            else:
                raise click.ClickException(
                    f"湍流模型 '{turb_model}' 需要壁面距离场，但网格里没有任何 "
                    f"WALL 类型边界（'boundaries.bc_types' 中无 WALL 项）——不能"
                    f"静默退化为'简化估计'继续求解：SST/DDES 的屏蔽函数、"
                    f"WMLES 的壁面应力模型都会用到错误的 d_w，得到看似正常、"
                    f"实际物理错误的结果。请检查体网格的边界分组是否正确。"
                )
        else:
            raise click.ClickException(
                "无法访问体网格边界数据（volume_data 缺少 'boundaries' 属性），"
                f"无法为湍流模型 '{turb_model}' 计算壁面距离场。"
            )
    except click.ClickException:
        raise
    except Exception as e:
        # 此前这里是裸 except Exception：任何失败（含 Eikonal 求解器内部
        # bug）都打印一行 warning 后静默降级为"simplified estimate"继续
        # 求解——但 solver.wall_distance 实际仍是 None，SST/DDES 下游会在
        # fr_solver_turbulence.py 里因 wall_distance is None 抛
        # RuntimeError（等于这里的"降级"从未真正发生），LES/WMLES 下游则
        # 没有这道保护、会真的带着错误的湍流模型悄悄跑完。与
        # load_mesh_for_solver 的质量门"宁可报错也不静默放行"原则矛盾，
        # 统一改为向上抛出可读错误。
        raise click.ClickException(
            f"壁面距离场计算失败，无法为湍流模型 '{turb_model}' 提供有效的 "
            f"d_w：{e}\n如需临时绕过做诊断，请改用 --turbulence-model none。"
        ) from e
