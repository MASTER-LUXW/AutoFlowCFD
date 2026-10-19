"""`utils benchmark`：网格加载、边界条件与求解器构造与 `solve steady` 同一条路径，计时求解器自己的无粘残差。

历史缺陷：
* 2026-08-22：`HighOrderMesh(grid_data, order=order)` 与错误的残差调用签名，命令从未跑通过；
* 2026-10-19（I-5）：输入是面网格、命令内部先生成体网格，残差调用不传边界幽灵态提供者、由核心函数静默换成零梯度
  外插——计时的是"边界不可见"的残差。现在输入体网格（.pkl 或 .nas + 面网格），与求解同一个加载函数与求解器工厂。
"""

import json
import pickle

from click.testing import CliRunner

from autoflowcfd.cli.main import cli
from tests.unit.test_exterior_faces_by_group import _volume_mesh


def _run(tmp_path, *args):
    vm, _, _ = _volume_mesh()                      # 3x3x3 立方体四面体网格，6 个组（z 面 WALL，其余 SYMMETRY）
    path = tmp_path / "cube.pkl"
    with open(path, "wb") as f:
        pickle.dump(vm, f)
    return CliRunner().invoke(cli, ["utils", "benchmark", str(path), "-n", "1", "-p", "1", "--json",
                                    "--skip-quality-check", *args]), vm


def test_benchmark_times_the_solver_residual_with_real_boundary_conditions(tmp_path):
    result, vm = _run(tmp_path)
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout[result.stdout.index("{"):])
    assert payload["status"] == "success" and payload["backend"] == "cpu"
    assert payload["n_cells"] == vm.cell_count and payload["n_sps"] == 6        # P1：sps_width(1)
    assert payload["benchmark_time_s"] >= 0.0 and "solver_setup_time_s" in payload


def test_benchmark_refuses_a_surface_mesh(tmp_path):
    """面网格不再在命令里现场生成体网格：与 solve steady 相同，直接报错。"""
    surface = tmp_path / "surface.nas"
    surface.write_text("CEND\nBEGIN BULK\nGRID,1,,0.,0.,0.\nGRID,2,,1.,0.,0.\nGRID,3,,0.,1.,0.\n"
                       "CTRIA3,1,1,1,2,3\nENDDATA\n")
    result = CliRunner().invoke(cli, ["utils", "benchmark", str(surface), "-n", "1", "-p", "1"])
    assert result.exit_code != 0
