"""cli/utils_commands.py::benchmark 的网格构造与残差调用接线的单元测试。

这里抓到的真实缺陷（2026-08-22 修复）：`benchmark` 调用
`HighOrderMesh(grid_data, order=order)`——但 `HighOrderMesh.__init__` 只接受
`order`（根本没有位置数据参数），而这里的 `grid_data` 是*面网格* GridData
（来自 `NASParser.parse()`），本来也不是 `load_from_volume_mesh` 需要的
VolumeMeshData。每次调用都在到达真正的基准循环之前立刻以
`TypeError: HighOrderMesh.__init__() got multiple values for argument
'order'` 崩溃——这条命令从未成功运行过一次。它还调用
`compute_inviscid_residual_fr(mesh, U_init)`，参数顺序/个数都不对，该函数
真实的签名是 `(U, mesh, ops, boundary_ghost_provider=None)`。

网格生成本身（边界层挤出 + tetgen）对单元测试太重，所以这里给
`generate_volume_mesh_from_surface` 打补丁，让它返回一个很小的合成
VolumeMeshData，并断言*构造与调用方式*正确——同样的方式已在对
cube_demo.nas 的真实端到端 CLI 运行里另行多次验证。
"""

from unittest.mock import patch, MagicMock
from types import SimpleNamespace

import numpy as np
import pytest
from click.testing import CliRunner

from autoflowcfd.cli.main import cli
from tests.unit.test_fr_residual_inviscid import _build_synthetic_mixed_mesh


def _tiny_volume_mesh_data():
    """`_build_synthetic_mixed_mesh` 的节点/连接数据原样（2 个四面体 + 2 个
    棱柱，已在本测试套件别处多次验证——见 test_fr_residual_inviscid.py、
    test_troubled_cell.py），包成 VolumeMeshData 的替身。
    """
    nodes = np.array(
        [
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1],
            [10, 0, 0], [11, 0, 0], [10, 1, 0], [10, 0, 1], [11, 0, 1], [10, 1, 1],
            [9, -1, 0], [9, -1, 1],
        ],
        dtype=float,
    )
    tet_conn = np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int32)
    prism_conn = np.array(
        [
            [5, 6, 7, 8, 9, 10],
            [5, 7, 11, 8, 10, 12],
        ],
        dtype=np.int32,
    )
    return SimpleNamespace(
        cell_count=len(tet_conn) + len(prism_conn),
        nodes=SimpleNamespace(get_coordinates=lambda: nodes),
        cells=SimpleNamespace(connectivity=tet_conn),
        prism_cells=SimpleNamespace(connectivity=prism_conn),
        boundaries=None,
    )


class TestBenchmarkMeshConstruction:
    def test_benchmark_builds_and_solves_without_crashing(self):
        """经 CLI 命令的端到端（网格生成用替身）——必须走到残差循环并打印结果，
        而不是在 HighOrderMesh 构造或残差调用签名上崩溃。
        """
        with patch(
            "autoflowcfd.grid.nas_io.parser_core.NASParser.parse"
        ) as mock_parse, patch(
            "autoflowcfd.grid.nas_io.parser_core.NASParser.generate_volume_mesh_from_surface"
        ) as mock_gen:
            mock_parse.return_value = MagicMock()
            mock_gen.return_value = _tiny_volume_mesh_data()

            runner = CliRunner()
            # order=0（P0）让残差核对单元测试来说足够便宜。
            result = runner.invoke(
                cli,
                ["utils", "benchmark", __file__, "-n", "1", "-p", "1", "--json"],
            )

        assert result.exit_code == 0, result.output
        import json
        payload = json.loads(result.stdout)
        assert payload["status"] == "success"
        assert payload["n_cells"] == 4  # 2 tet + 2 prism
