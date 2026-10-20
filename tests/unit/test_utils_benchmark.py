"""`utils benchmark`：网格加载、边界条件与求解器构造与 `solve steady` 同一条路径，计时完整求解步（I-177）。

历史缺陷：
* 2026-08-22：`HighOrderMesh(grid_data, order=order)` 与错误的残差调用签名，命令从未跑通过；
* 2026-10-19（I-5）：输入是面网格、命令内部先生成体网格，残差调用不传边界幽灵态提供者、由核心函数静默换成零梯度
  外插——计时的是"边界不可见"的残差。现在输入体网格（.pkl 或 .nas + 面网格），与求解同一个加载函数与求解器工厂；
* 2026-10-20（I-177）：只计时一次无粘残差求值却以"计算速率 iter/min"报告。现在计时 solve() 的完整求解步（所选时间
  格式 / 湍流模型），分项给出残差求值次数与耗时、隐式每步 GMRES 迭代与矩阵向量积次数。
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


def _payload(result):
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout[result.stdout.index("{"):])


def test_benchmark_times_full_explicit_steps(tmp_path):
    result, vm = _run(tmp_path)
    payload = _payload(result)
    assert payload["status"] == "success" and payload["backend"] == "cpu"
    assert payload["n_cells"] == vm.cell_count and payload["n_sps"] == 6        # P1：sps_width(1)
    assert payload["residual_evaluations_per_step"] == 3.0                       # SSP-RK3 三级
    assert payload["time_per_step_s"] >= payload["residual_time_per_step_s"] > 0.0
    assert payload["gmres_iterations_per_step"] == 0.0


def test_benchmark_reports_newton_krylov_linear_solve_counts(tmp_path):
    payload = _payload(_run(tmp_path, "-t", "newton-krylov")[0])
    assert payload["time_scheme"] == "newton-krylov"
    assert payload["gmres_iterations_per_step"] > 0 and payload["matvec_per_step"] > 0
    # JFNK 的矩阵向量积就是残差求值：每步残差求值不少于矩阵向量积次数 + 1（初始残差）
    assert payload["residual_evaluations_per_step"] >= payload["matvec_per_step"] + 1


def test_benchmark_refuses_a_surface_mesh(tmp_path):
    """面网格不再在命令里现场生成体网格：与 solve steady 相同，直接报错。"""
    surface = tmp_path / "surface.nas"
    surface.write_text("CEND\nBEGIN BULK\nGRID,1,,0.,0.,0.\nGRID,2,,1.,0.,0.\nGRID,3,,0.,1.,0.\n"
                       "CTRIA3,1,1,1,2,3\nENDDATA\n")
    result = CliRunner().invoke(cli, ["utils", "benchmark", str(surface), "-n", "1", "-p", "1"])
    assert result.exit_code != 0
