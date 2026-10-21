"""体网格生成管线端到端（边界组分类 -> 尖锐拐角拆分 -> BL 挤出 -> 棱柱 -> tetgen 核心填充 -> 混合修复 -> 质量报告），
在几秒内能生成的合成算例上核对可精确知道的量（I-205 / I-207）。

算例（`_box_in_box_surface.py`）：[-3, 5] x [-2, 2] x [-2, 2] m 远场盒里放边长 1 m 的立方体物面（WALL），物面每面
3 x 3 个四边形（108 个三角形）。只有无滑移壁面挤出边界层：远场或滑移壁上都不能有棱柱。
"""

import numpy as np
import pytest

from autoflowcfd.grid.mesh_gen.extrusion.mesh_layer_step import MITER_LIMIT
from autoflowcfd.grid.nas_io.parser_volume import generate_volume_mesh_from_surface
from autoflowcfd.grid.validation.quality_validator import MeshQualityValidator
from tests.unit._box_in_box_surface import box_in_box_surface

_BODY_TRIANGLES = 6 * 3 * 3 * 2
_FLUID_VOLUME = 8.0 * 4.0 * 4.0 - 1.0


@pytest.mark.parametrize("tunnel_type", ["FARFIELD", "SLIP_WALL"])
def test_box_in_box_volume_mesh(tunnel_type):
    """I-207：此前"不挤出类型"清单漏了 FARFIELD——远场被挤出边界层，整个外框单独成组时被当成封闭内嵌实体打 tetgen
    孔洞种子，计算域被整体挖空（tetgen "Found 0 domain"，生成失败）。"""
    surface = box_in_box_surface()
    surface.boundaries.bc_types["tunnel"] = tunnel_type
    layers, h1 = 2, 0.02
    vm = generate_volume_mesh_from_surface(
        surface, {"bl_layers": layers, "first_layer_height": h1, "max_cell_size": 1.0, "min_cell_size": 0.05})

    assert vm.prism_cells is not None and vm.prism_cells.count == _BODY_TRIANGLES * layers   # 只有物面挤出
    np.testing.assert_allclose(vm.total_volume, _FLUID_VOLUME, rtol=1e-9)                    # 流体域被完整填满

    # 第一层棱柱贴在物面上：面内节点高度为给定的第一层高度，棱边 / 角点节点带斜接补偿（不超过 MITER_LIMIT 倍；
    # 粗网格立方体棱的隐含半径大于 min_cell_size，按光滑曲面处理、不拆分，见 split_sharp_corners）
    nodes = np.column_stack([vm.nodes.x, vm.nodes.y, vm.nodes.z])
    prisms = vm.prism_cells.connectivity
    base = nodes[prisms[:, :3]]
    heights = np.linalg.norm(nodes[prisms[:, 3:]] - base, axis=2)
    on_body = np.all(np.abs(base).max(axis=2) <= 0.5 + 1e-12, axis=1)
    assert on_body.sum() == _BODY_TRIANGLES
    face_interior = (np.abs(np.abs(base) - 0.5) < 1e-12).sum(axis=2) == 1
    np.testing.assert_allclose(heights[on_body & face_interior.all(axis=1)], h1, rtol=1e-9)
    assert np.all((heights[on_body] >= h1 * (1 - 1e-9)) & (heights[on_body] <= MITER_LIMIT * h1 * (1 + 1e-9)))

    report = MeshQualityValidator().validate_volume_mesh(vm, check_overlap=False)
    assert report.negative_volumes == 0


@pytest.mark.parametrize("body_type, module", [
    ("WALL", "autoflowcfd.grid.mesh_gen.background.mesh_background_merge_with_bl"),
])
def test_max_cell_size_region_seed_lies_in_the_fluid(monkeypatch, body_type, module):
    """tetgen 最大单元尺寸约束的区域种子点必须在流体域内（I-208）。此前取 PLC / 面网格点均值：物面点远多于远场点时
    均值落在物体孔洞里，tetgen 判定种子点在域外（"region point lies in the exterior"），整个体积约束静默失效。"""
    import importlib

    mod = importlib.import_module(module)
    seeds = []
    original = mod.fill_core_volume

    def recording_fill(*args, regions=None, **kwargs):
        seeds.extend(np.asarray(r[0]) for r in regions or ())
        return original(*args, regions=regions, **kwargs)

    monkeypatch.setattr(mod, "fill_core_volume", recording_fill)
    surface = box_in_box_surface(body_n=8, tunnel_n=4)             # 物面点远多于远场点
    surface.boundaries.bc_types["body"] = body_type
    surface.boundaries.bc_types["tunnel"] = "FARFIELD"
    generate_volume_mesh_from_surface(
        surface, {"bl_layers": 2, "first_layer_height": 0.02, "max_cell_size": 0.6, "min_cell_size": 0.05})

    assert seeds
    for seed in seeds:
        assert np.abs(seed).max() > 0.5 + 0.05                      # 不在物体（含边界层）里
        assert np.all(seed > [-3.0, -2.0, -2.0]) and np.all(seed < [5.0, 2.0, 2.0])
