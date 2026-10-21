"""求解命令的体网格加载（.pkl / .nas 体网格 + 面网格反推边界分组 + 求解前质量门）。"""

import os
import pickle
from typing import Dict, Optional, Tuple

import click

from autoflowcfd.grid.high_order.high_order_mesh import HighOrderMesh
from autoflowcfd.core.mpi.comm import run_on_root
from autoflowcfd.grid.validation.solve_gate import MeshQualityGateError, high_order_mesh_for_solver
from autoflowcfd.grid.schema.grid_data import VolumeMeshData


#: 求解命令把 .nas 体网格输入转存成的米制体网格缓存（输出目录下；后处理优先读它）
VOLUME_MESH_CACHE_NAME = "volume_mesh.pkl"


def save_volume_mesh_cache(volume_mesh: VolumeMeshData, output_dir: str) -> str:
    """体网格写成 `output_dir/volume_mesh.pkl`（与 `grid generate-volume` 的 .pkl 同一格式，坐标为米），返回路径。
    checkpoint 记录它作为 `input_file`，续算与后处理据此重建（CLI 求解命令与 Python API 共用）。"""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, VOLUME_MESH_CACHE_NAME)
    with open(path, "wb") as f:
        pickle.dump(volume_mesh, f)
    return path


def _import_volume_nas(input_file: str, surface_mesh: Optional[str], units: str, validate: bool,
                       bc_types: Optional[Dict[str, str]] = None):
    """.nas 体网格 + 面网格 -> (VolumeMeshData, 质量报告或 None)，不修复（修复只在 `grid import-volume` 里显式做）。
    `bc_types`：`--bc-type` 给出的边界条件类型（覆盖按组名识别的结果）。"""
    if not surface_mesh:
        raise click.ClickException(
            f"'{input_file}' 是 .nas 体网格，缺少 --surface-mesh/-s：需要原始面网格来反推 WALL/INLET/OUTLET 等边界分组，"
            f"否则湍流模型和边界条件都无法正确设置。用法：--surface-mesh <原始面网格.nas>")
    from autoflowcfd.grid.mesh_gen.utils.mesh_external_import import import_external_volume_mesh
    from autoflowcfd.grid.nas_io.boundary_types import BoundaryTypeError

    try:
        # repair=False：求解用的必须是用户给的网格（Stage A 光顺会移动节点）；质量不过由求解前质量门拦截。
        # 2026-10-20 以前用默认 repair=True，质量门不过就静默移动用户网格的节点再求解
        return import_external_volume_mesh(input_file, surface_mesh, repair=False, units=units, validate=validate,
                                           bc_types=bc_types)
    except BoundaryTypeError as e:
        raise click.ClickException(f"{e}\n用 --bc-type 组名=类型（可重复）给出这些组的边界条件类型。") from e
    except ValueError as e:
        raise click.ClickException(
            f"无法把 '{input_file}' 当作体网格解析：{e}\n"
            f"如果这其实是一份面网格，请先用 'autoflowcfd grid generate-volume {input_file} -o <输出.nas>' 生成体网格。"
        ) from e


def _read_volume_pkl(path: str) -> VolumeMeshData:
    """读 .pkl 体网格；不是 VolumeMeshData 时报错。"""
    with open(path, 'rb') as f:
        volume_data = pickle.load(f)
    if not isinstance(volume_data, VolumeMeshData):
        raise click.ClickException(
            f"'{path}' 反序列化后不是 VolumeMeshData 实例 - 不是本软件自己生成/导入的体网格缓存文件。")
    return volume_data


def _apply_bc_types(volume_data: VolumeMeshData, bc_types: Optional[Dict[str, str]]) -> None:
    """把 `--bc-type` 写进体网格及其面网格的边界映射（组名必须存在、类型必须受支持）。"""
    if not bc_types:
        return
    from autoflowcfd.grid.nas_io.boundary_types import BoundaryTypeError, apply_bc_types

    try:
        apply_bc_types(volume_data.boundaries, bc_types)
        surface = volume_data.surface_mesh or {}
        if surface.get("boundaries") is not None and surface["boundaries"] is not volume_data.boundaries:
            apply_bc_types(surface["boundaries"], bc_types)
    except BoundaryTypeError as e:
        raise click.ClickException(str(e)) from e


def prepare_solver_input(input_file: str, surface_mesh: Optional[str], units: str, output_dir: str,
                         bc_types: Optional[Dict[str, str]] = None) -> Tuple[str, Optional[str]]:
    """求解命令入口：.nas 体网格按 `units` 导入（不修复、不在此做质量检查），存成 `output_dir/volume_mesh.pkl`，之后
    全程（各后端、checkpoint、续算、后处理）用这份米制 .pkl；.pkl 输入原样返回。集体操作（root 转换，全部 rank 调用）。

    2026-10-20 以前求解命令没有 --units，.nas 体网格一律按毫米读入；且续算 / 后处理按 checkpoint 记录的 .nas
    重新加载时同样只能按毫米。

    `bc_types`（`--bc-type`）：边界条件类型覆盖，写进转存的 .pkl；输入本身是 .pkl 时也把覆盖后的体网格转存到输出目录
    （用户的 .pkl 不改）。.nas 不记录类型，2026-10-21 以前求解命令没有这个选项：生成体网格时用 --bc-type 指定的类型
    在读回 .nas 时丢失，组名有歧义的算例只能先 `grid import-volume` 转成 .pkl。

    Returns:
        (input_file, surface_mesh)：之后应使用的体网格文件与面网格（.pkl 已含边界与面网格，返回 None）
    """
    is_nas = os.path.splitext(input_file)[1].lower() == ".nas"
    if not is_nas and not bc_types:
        return input_file, surface_mesh

    def convert():
        target = os.path.join(output_dir, VOLUME_MESH_CACHE_NAME)
        if is_nas:
            print(f"Importing volume-mesh NAS (units={units}) -> {target}")
            volume_data, _ = _import_volume_nas(input_file, surface_mesh, units, validate=False, bc_types=bc_types)
        else:
            print(f"Applying --bc-type to {input_file} -> {target}")
            volume_data = _read_volume_pkl(input_file)
            _apply_bc_types(volume_data, bc_types)
        return save_volume_mesh_cache(volume_data, output_dir)

    run_on_root(convert)
    return os.path.join(output_dir, VOLUME_MESH_CACHE_NAME), None


def load_mesh_for_solver(
    input_file: str,
    order: int,
    surface_mesh: Optional[str] = None,
    skip_quality_check: bool = False,
    units: str = "mm",
    bc_types: Optional[Dict[str, str]] = None,
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
        units: input_file 为 .nas 体网格时它与面网格的坐标单位。求解命令在入口就把 .nas 转存成 .pkl
            （`prepare_solver_input`），走到这里的 .nas 是续算 2026-10-20 以前的 checkpoint（当时一律按毫米读入）
            或 `utils benchmark`
        bc_types: 边界条件类型覆盖（`--bc-type`，`utils benchmark` 用；求解命令在 `prepare_solver_input` 里已应用）

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
        volume_data = _read_volume_pkl(input_file)
        _apply_bc_types(volume_data, bc_types)

    elif ext == '.nas':
        print("Detected volume-mesh NAS format. Parsing and attributing boundaries...")
        volume_data, report = _import_volume_nas(input_file, surface_mesh, units, validate=True, bc_types=bc_types)

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
