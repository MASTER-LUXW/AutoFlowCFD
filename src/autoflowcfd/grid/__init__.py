"""网格：.nas 解析、数据结构、体网格生成、质量校验、高阶网格。

包导入不加载任何子模块（2026-10-22（I-123）以前这里一次性导入解析器、tetgen 体网格生成、质量校验、高阶网格等，
求解器每次启动都付这份导入代价）；按需从子包导入：

    schema/          数据结构（GridData、VolumeMeshData、NodeArray、CellArray、BoundaryMap ...）
    nas_io/          .nas 解析（NASParser）与导出（export_volume_mesh_to_nas）
    validation/      面网格校验（GridValidator）、体网格质量（MeshQualityValidator）、求解前质量门
    mesh_gen/        边界层挤出、背景网格、tetgen 体网格生成（VolumeMeshGenerator）、面提取
    connectivity/    FR 面连接关系、边界组打标、周期面配对
    high_order/      高阶网格（HighOrderMesh）

示例:
    >>> from autoflowcfd.grid.nas_io.parser_core import NASParser
    >>> from autoflowcfd.grid.validation.validator import GridValidator
    >>> grid = NASParser("car_model.nas").parse()
    >>> results = GridValidator(grid).validate()
"""
