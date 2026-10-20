"""求解命令的体网格加载（.pkl / .nas 体网格 + 面网格反推边界分组 + 求解前质量门）。"""

import os
import pickle
from typing import Optional, Tuple

import click

from autoflowcfd.grid.high_order.high_order_mesh import HighOrderMesh
from autoflowcfd.grid.validation.solve_gate import MeshQualityGateError, high_order_mesh_for_solver
from autoflowcfd.grid.schema.grid_data import VolumeMeshData


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
        print("Detected pickle format. Loading volume mesh...")
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
        from autoflowcfd.grid.mesh_gen.utils.mesh_external_import import import_external_volume_mesh
        print("Detected volume-mesh NAS format. Parsing and attributing boundaries...")
        from autoflowcfd.grid.nas_io.boundary_types import BoundaryTypeError
        try:
            # repair=False：求解用的必须是用户给的网格。修复（Stage A 光顺会移动节点）只在 `grid import-volume` 里
            # 显式进行；质量不过时由下方的质量门拦截。2026-10-20 以前这里用默认 repair=True，质量门不过就静默移动
            # 用户网格的节点再求解
            volume_data, report = import_external_volume_mesh(input_file, surface_mesh, repair=False)
        except BoundaryTypeError as e:
            raise click.ClickException(
                f"{e}\n边界条件类型在建立体网格时确定：请用 'autoflowcfd grid import-volume {input_file} "
                f"-s {surface_mesh} -o <输出.pkl> --bc-type 组名=类型 ...' 生成 .pkl 后再提交计算。") from e
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

    # 质量门与高阶网格构造：与 Python API 共用 `grid/validation/solve_gate.py`。.nas 路径复用导入时已经跑过的质量
    # 报告，.pkl 路径在那里现场算一次——两条加载路径必须经过同一道质量门
    try:
        mesh = high_order_mesh_for_solver(volume_data, order, report=report, skip_quality_check=skip_quality_check)
    except MeshQualityGateError as e:
        raise click.ClickException(str(e)) from e
    return mesh, volume_data
