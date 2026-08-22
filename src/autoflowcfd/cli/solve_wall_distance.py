"""求解命令的壁面距离场计算辅助函数 —— 从 solve_helpers.py 拆出，控制单文件行数。

见 solve_helpers.py 文档说明整体拆分结构。
"""

import click


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
    if turb_model not in ['sst', 'ddes', 'wmles', 'les']:
        print(f"   ℹ️  Turbulence model '{turb_model}' does not require wall distance")
        return

    try:
        if volume_data is not None and hasattr(volume_data, 'boundaries'):
            print("\n🔍 Computing wall distance field...")

            bm = volume_data.boundaries
            wall_nodes = set()
            n_nodes = volume_data.node_count

            # 获取体网格连接关系用于单元->节点转换
            all_connectivity = []
            if volume_data.prism_cells:
                all_connectivity.extend(volume_data.prism_cells.connectivity)
            if volume_data.cells:
                all_connectivity.extend(volume_data.cells.connectivity)

            # 识别所有 WALL 类型的边界
            for bc_name, bc_type in bm.bc_types.items():
                if bc_type == 'WALL' and bm.has_boundary(bc_name):
                    indices = bm.get_node_indices(bc_name)

                    # 检查是否为单元索引（如果最大索引 >= 节点数）
                    if len(indices) > 0 and np.max(indices) >= n_nodes:
                        print(f"   - Boundary '{bc_name}': Detected as cell indices, converting...")
                        node_indices_from_cells = set()
                        for cell_idx in indices:
                            if cell_idx < len(all_connectivity):
                                cell_nodes = all_connectivity[cell_idx]
                                valid_nodes = [n for n in cell_nodes if n != -1 and n < n_nodes]
                                node_indices_from_cells.update(valid_nodes)

                        if node_indices_from_cells:
                            wall_nodes.update(node_indices_from_cells)
                            print(f"     Converted {len(indices)} cells to {len(node_indices_from_cells)} nodes")
                    else:
                        valid_indices = indices[indices < n_nodes]
                        if len(valid_indices) > 0:
                            wall_nodes.update(valid_indices.tolist())
                            print(f"   - Boundary '{bc_name}': {len(valid_indices)} nodes")

            if wall_nodes:
                wall_indices = np.array(list(wall_nodes))
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
