"""solve 命令组的共享辅助函数。

从 solve_commands.py 拆出来（该文件保留 steady/transient/resume/status 四个
click 命令本体），控制单文件行数：网格加载、壁面距离场计算、结果保存这三个
辅助函数被 steady 和 transient 两个命令共用，不属于任何一个命令自己的定义。
"""

import logging
import os
import pickle
from typing import Optional, Tuple

import click

from autoflowcfd.grid.high_order_mesh import HighOrderMesh
from autoflowcfd.grid.schema.grid_data import VolumeMeshData
from autoflowcfd.grid.curved_mapping import MeshDistortionError

logger = logging.getLogger(__name__)


def load_mesh_for_solver(
    input_file: str,
    order: int,
    surface_mesh: Optional[str] = None,
    skip_quality_check: bool = False,
) -> Tuple['HighOrderMesh', VolumeMeshData]:
    """
    工业级网格加载器：接受已经生成好的体网格，.pkl（VolumeMeshData 序列化）
    或 .nas（GRID + CTETRA/CPENTA 卡片，本软件自己 `grid generate-volume`
    的导出，或 ANSA 等外部工具自己的体网格导出）皆可，加载后强制跑一遍质量
    门检查。

    刻意不接受**面网格**（.nas 里只有 CTRIA3、没有 CTETRA/CPENTA）就地自动
    转换成体网格 - 早期版本这里有一条"检测到面网格就用写死的默认参数
    （min_cell_size=0.01, max_cell_size=0.1, bl_layers=5）现场生成体网格"的
    捷径，参数与 `autoflowcfd grid generate-volume` 的默认值完全不一致，且
    直接把文件路径字符串传给 VolumeMeshGenerator.generate_from_surface（该
    方法签名要的是 surface_nodes/surface_faces/bounding_box 数组，不是路
    径），实际调用会类型错误失败 - 一条本来就不能工作、且就算能工作也会因
    为参数不透明而生成一份用户没有实际审查过的体网格的隐藏路径，不如干脆
    去掉：体网格生成是一个独立、可审查、可复用的步骤（`grid
    generate-volume`/`grid import-volume`），不应该在提交计算这一步隐式重
    新发生一次。**体网格**（.nas 里已经有 CTETRA/CPENTA）则没有这个顾虑 -
    这条路径直接复用 `grid import-volume` 内部的同一套解析/边界反推/质量
    检查逻辑（mesh_external_import.import_external_volume_mesh），不是另起
    一套。

    体网格 .nas 文件本身通常不带边界条件信息（ANSA 自身的体网格导出只标注
    材料分区，不带 inlet/outlet/wall 这类面边界条件；本软件自己
    `generate-volume` 导出的 .nas 虽然确实把边界信息写进了 PSHELL/CTRIA3
    卡片，但目前还没有对应的"自包含读回"解析器，读体网格 .nas 统一走外部
    体网格那一套边界反推逻辑），所以必须额外提供 `surface_mesh`（原始面网
    格，用于按几何最近质心反推边界分组）才能正确识别 WALL/INLET/OUTLET 等
    边界 - 没有它，湍流模型需要的壁面距离场、边界条件都无从谈起，宁可直接
    报错也不要静默返回一个边界信息全部丢失的网格。

    质量门检查是补上的一个真实缺口：`grid generate-volume`/`import-volume`
    自己的帮助文本和多处代码注释里反复写着"'autoflowcfd solve steady'/
    'transient' 会在正式迭代前强制这道质量门，除非传了 --skip-quality-check"
    ——但实际实现里这道检查此前完全不存在，任何一个网格（哪怕质量门早就报
    FAILED，例如本项目 cube_demo 这类残留大量退化单元的网格）都会被原样
    直接拿去求解，没有任何拦截或提示，与代码里到处引用的"这是最终强制点"
    的说法完全对不上。这里按照那些文档已经写明的意图把它实现出来。

    Args:
        input_file: 体网格文件路径，.pkl（`grid generate-volume`/`grid
            import-volume` 的输出）或 .nas（GRID+CTETRA/CPENTA）
        order: FR 多项式阶数，传给 HighOrderMesh
        surface_mesh: 原始面网格路径 - input_file 是 .nas 体网格时必填，用于
            反推边界分组；input_file 是 .pkl 时忽略（边界信息已经在 pkl 里）
        skip_quality_check: 跳过质量门检查，网格质量不合格也强行求解 - 仅用于
            明确知道风险、需要临时诊断的场景，默认关闭

    Returns:
        (mesh, volume_data): mesh 是给 FRSolver 用的 HighOrderMesh；
        volume_data 是加载出来的原始 VolumeMeshData（边界分组等信息仍在
        其中），供调用方后续计算壁面距离场等操作复用，避免重新解析一遍
        输入文件。

    Raises:
        click.ClickException: input_file 既不是 .pkl 也不是有效的体网格
            .nas（例如实际是只有 CTRIA3 的面网格）；input_file 是 .nas 但
            没有提供 surface_mesh；.pkl 反序列化出的对象不是
            VolumeMeshData；或质量门检查未通过且 skip_quality_check 为 False
    """
    ext = os.path.splitext(input_file)[1].lower()
    report = None

    if ext == '.pkl':
        print(f"Detected pickle format. Loading volume mesh...")
        with open(input_file, 'rb') as f:
            volume_data = pickle.load(f)

        if not isinstance(volume_data, VolumeMeshData):
            raise click.ClickException(
                f"'{input_file}' 反序列化后不是 VolumeMeshData 实例 - 不是本"
                f"软件自己生成/导入的体网格缓存文件。"
            )

    elif ext == '.nas':
        if not surface_mesh:
            raise click.ClickException(
                f"'{input_file}' 是 .nas 体网格，缺少 --surface-mesh/-s：需要"
                f"原始面网格来反推 WALL/INLET/OUTLET 等边界分组，否则湍流模型"
                f"和边界条件都无法正确设置。用法：--surface-mesh <原始面网格.nas>"
            )
        from autoflowcfd.grid.mesh_gen.mesh_external_import import import_external_volume_mesh
        print(f"Detected volume-mesh NAS format. Parsing and attributing boundaries...")
        try:
            volume_data, report = import_external_volume_mesh(input_file, surface_mesh)
        except ValueError as e:
            raise click.ClickException(
                f"无法把 '{input_file}' 当作体网格解析：{e}\n"
                f"如果这其实是一份面网格，请先用 'autoflowcfd grid "
                f"generate-volume {input_file} -o <输出.nas>' 生成体网格。"
            ) from e

    else:
        raise click.ClickException(
            f"求解命令只接受体网格 (.pkl 或 .nas 体网格)，收到的是 '{ext}'。"
            f"请先用 'autoflowcfd grid generate-volume <面网格.nas> -o "
            f"<输出.nas>'（本软件生成体网格）或 'autoflowcfd grid "
            f"import-volume <体网格.nas> -s <面网格.nas> -o <输出.pkl>'"
            f"（导入外部/ANSA 生成的体网格）产出体网格后再提交计算。"
        )

    print(f"Volume mesh loaded: {volume_data.nodes.count} nodes, {volume_data.cell_count} cells")

    if skip_quality_check:
        print("⚠️  --skip-quality-check 已启用，跳过求解前的网格质量门检查")
    else:
        if report is None:
            # .pkl 路径没有随身带质量报告（不像 .nas 路径复用了
            # import_external_volume_mesh 内部已经跑过的那一份），这里现场
            # 补一次 - 两条路径最终都必须经过同一道质量门检查，不能因为走的
            # 是哪条加载路径而有差别。
            from autoflowcfd.grid.validation.quality_validator import MeshQualityValidator
            print("Validating volume mesh quality before solving...")
            report = MeshQualityValidator().validate_volume_mesh(volume_data)
        if not report.passed:
            raise click.ClickException(
                f"网格质量门检查未通过，拒绝求解（避免残差在退化单元处异常"
                f"放大）：\n{report.summary()}\n"
                f"如果明确了解风险、只是想临时诊断，可以加 "
                f"--skip-quality-check 跳过这道检查强行求解。"
            )
        print("✅ Volume mesh quality gate passed")

    mesh = HighOrderMesh(order=order)
    try:
        mesh.load_from_volume_mesh(volume_data)
    except MeshDistortionError as e:
        # 质量门检查（上面）用启发式指标筛查网格质量，不保证能拦住所有
        # 退化/反转单元（尤其 --skip-quality-check 场景）；曲边映射在
        # 构造高阶几何时会做严格的 Jacobian 符号检查，兜底再次拦截并给出
        # 可读诊断，而不是让 UnboundLocalError/裸 ValueError 直接扎穿到
        # 用户面前（此前 tet/prism 分支的报错逻辑本身有 bug 导致必然抛
        # UnboundLocalError 而不是这里设计好的 MeshDistortionError，已在
        # curved_mapping.py::compute_jacobian 修复）。
        raise click.ClickException(
            f"网格在构造高阶曲边几何时检测到畸变单元，拒绝求解：{e}\n"
            f"请在网格生成/修复阶段处理该单元（'autoflowcfd grid check'/"
            f"'repair'），不要用 --skip-quality-check 强行跳过。"
        ) from e
    return mesh, volume_data


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
                    from autoflowcfd.grid.node_connectivity import build_node_adjacency

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


def save_results(solver, output_dir: str):
    """
    保存求解结果到指定目录。

    Args:
        solver: FRSolver实例
        output_dir: 输出目录路径
    """
    os.makedirs(output_dir, exist_ok=True)

    # 保存最终状态
    state_path = os.path.join(output_dir, "final_state.pkl")
    with open(state_path, 'wb') as f:
        pickle.dump({
            'U': solver.state.U,
            'Q': solver.state.Q,
            'n_cells': solver.state.n_cells,
            'n_sps': solver.state.n_sps,
            'n_vars': solver.state.n_vars
        }, f)

    print(f"✅ Results saved to: {output_dir}")
    print(f"   - Final state: {state_path}")


def restore_state_from_checkpoint(
    checkpoint_path: str,
    solver,
    metadata: dict,
) -> int:
    """从 checkpoint 恢复求解器状态，用于 `solve transient --init-from` 以稳态结果
    为初场启动瞬态仿真（典型工作流：先跑稳态 SST 收敛到平衡态，再用 DDES/LES
    从该流场启动瞬态计算——避免从均匀流场直接启动 DES 需要极长的瞬态发展时间）。

    处理 n_vars 不匹配的情况：稳态 SST 与瞬态 DDES 都是 7 变量（rho/rho_u/rho_v/
    rho_w/rho_e/rho_k/rho_omega），直接拷贝；稳态 `none`（5 变量）→ 瞬态 DDES
    （7 变量）时，前 5 个守恒变量直接拷贝，湍流量（rho_k/rho_omega）用自由来流
    默认值初始化（k=1e-6, omega=1e-2，与 FRState.initialize_uniform 的默认值
    一致）；反之稳态 SST（7 变量）→ 瞬态 LES（5 变量）时只取前 5 个。

    Args:
        checkpoint_path: checkpoint 文件路径
        solver: 已创建但尚未开始求解的 FRSolver 实例
        metadata: checkpoint 加载后返回的 metadata 字典（含 fields 键）

    Returns:
        checkpoint 记录的迭代数（供调用方打印日志）

    Raises:
        click.ClickException: checkpoint 缺少 U_sps 字段或形状不兼容
    """
    fields = metadata.get("fields", {})
    if "U_sps" not in fields:
        raise click.ClickException(
            f"Checkpoint '{checkpoint_path}' 缺少 'U_sps' 字段（完整的 (n_cells,n_sps,n_vars) "
            f"求解器状态）——不是本版本 solve_helpers.write_checkpoint 写出的 checkpoint，"
            f"无法精确恢复。"
        )

    U_ckpt = fields["U_sps"]
    n_vars_ckpt = U_ckpt.shape[2] if U_ckpt.ndim == 3 else 0
    n_vars_solver = solver.state.n_vars
    ckpt_iter = metadata.get("iteration", 0)

    # 形状校验：n_cells 和 n_sps 必须一致（网格/阶数不匹配）
    if U_ckpt.shape[0] != solver.state.n_cells or U_ckpt.shape[1] != solver.state.n_sps:
        raise click.ClickException(
            f"Checkpoint 状态形状 {U_ckpt.shape} 与重建求解器的状态形状 "
            f"{solver.state.U.shape} 不匹配（网格或阶数可能已变化），拒绝恢复。"
        )

    if n_vars_ckpt == n_vars_solver:
        # 变量数一致：直接整体拷贝（最常见路径：稳态 SST→瞬态 DDES 都是 7 vars）
        solver.state.U = U_ckpt.copy()
        print(f"   ✅ 从 checkpoint 恢复完整状态 ({n_vars_ckpt} vars)")
    elif n_vars_ckpt < n_vars_solver:
        # checkpoint 变量少于新求解器：拷贝流体部分，湍流量用自由来流默认值初始化
        solver.state.U[:, :, :n_vars_ckpt] = U_ckpt
        if n_vars_solver > 5:
            # 用自由来流条件初始化湍流量（与 FRState.initialize_uniform 默认值一致）
            rho_inf = solver.freestream.get("rho_inf", 1.225)
            solver.state.U[:, :, 5] = rho_inf * 1e-6   # rho*k
            solver.state.U[:, :, 6] = rho_inf * 1e-2   # rho*omega
        print(f"   ✅ 从 checkpoint 恢复流体场 ({n_vars_ckpt} vars)，"
              f"湍流量用自由来流默认值初始化（新求解器需要 {n_vars_solver} vars）")
    else:
        # checkpoint 变量多于新求解器：只取前 n_vars_solver 个（如稳态 SST→瞬态 LES）
        solver.state.U = U_ckpt[:, :, :n_vars_solver].copy()
        print(f"   ✅ 从 checkpoint 恢复前 {n_vars_solver} 个变量（checkpoint 有 "
              f"{n_vars_ckpt} vars，新求解器只需 {n_vars_solver}）")

    solver.state._update_primitives()
    return ckpt_iter


def write_checkpoint(
    solver,
    output_dir: str,
    iteration: int,
    input_file: str,
    order: int,
    turbulence_model: str,
    backend: str,
    history: Optional[dict] = None,
) -> Optional[str]:
    """把求解器状态写成 HDF5 checkpoint，供 `solve resume` 真正恢复求解
    （V2.0 二次评审 Tier 1 #13/#14：此前 `solve steady/transient` 从不
    写 checkpoint，`solve resume` 因此永远无事可做，`post` 命令组也找
    不到任何 checkpoint 文件）。

    `CheckpointManager.save()` 的 `solution` 参数是 V1 时代遗留的
    `(n_cells, n_vars)` 单元中心格式；FR 的真实解是
    `(n_cells, n_sps, n_vars)` 多点存储，直接拍扁成单元中心值会丢失
    高阶信息，不能作为恢复求解的依据。这里用 `solution` 存一份逐单元
    平均值（供不关心高阶细节、只看大致场分布的场景，如未来的
    `post` 命令），把完整的 `(n_cells,n_sps,n_vars)` 状态通过
    `extra_fields` 整个存下来（HDF5 支持任意形状数组，不需要拍平），
    `resume` 读回时用这份完整状态精确恢复，不是有损重启。

    metadata 里额外存重建 FRSolver 所需的全部构造参数（input_file 用于
    重新加载网格——mesh/face_connectivity 这类对象本身没有放进
    checkpoint，序列化+反序列化整个网格对象比重新跑一遍
    `load_mesh_for_solver` 更脆弱、更没必要）。

    Returns:
        checkpoint 文件路径；h5py 不可用等失败情形返回 None（不中止求解）
    """
    from types import SimpleNamespace
    from autoflowcfd.core.checkpoint import CheckpointManager, H5PY_AVAILABLE

    if not H5PY_AVAILABLE:
        print("   ⚠️  h5py not available, skipping checkpoint write (final_state.pkl is still saved)")
        return None

    config = SimpleNamespace(
        mode="steady" if history is None else "transient",
        backend=backend,
        order=order,
        turbulence=turbulence_model,
    )
    manager = CheckpointManager(config, output_dir=output_dir)

    solution_cell_avg = solver.state.U.mean(axis=1)  # (n_cells, n_vars)，供粗粒度消费方使用
    extra_fields = {"U_sps": solver.state.U, "Q_sps": solver.state.Q}

    metadata = {
        "input_file": input_file,
        "order": order,
        "turbulence_model": turbulence_model,
        "backend": backend,
        "n_sps_per_cell": solver.state.n_sps,
        "n_vars": solver.state.n_vars,
        "rho_inf": solver.freestream["rho_inf"],
        "vel_inf": solver.freestream["vel_inf"],
        "p_inf": solver.freestream["p_inf"],
    }

    path = manager.save(
        solution_cell_avg,
        history or {"iterations": [iteration]},
        iteration,
        metadata=metadata,
        extra_fields=extra_fields,
    )
    if path:
        print(f"   - Checkpoint: {path}")
    return path
