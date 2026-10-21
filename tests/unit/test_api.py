"""Python API（`autoflowcfd.api`）的单元测试。

2026-10-19 重构（I-167）后的约定：参数就是配置类字段（默认值与校验只在配置类里）；求解前跑与 CLI 同一道质量门；
续算的时间步长/容差与 CLI `solve resume` 同一套（不再写死 dt=1e-3、tol=1e-6）；网格校验用真实校验器；后处理方法
作用于 `self.solver`，没有未使用的 `result` 形参。
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from autoflowcfd import AutoFlowCFDAPI, create_api, get_version
from autoflowcfd.config.solver_config import SteadyConfig, TransientConfig, TurbulenceModel
from autoflowcfd.core.time_integration.base import TimeIntegrationScheme as S

_GATE = "autoflowcfd.grid.validation.solve_gate.high_order_mesh_for_solver"
_FACTORY = "autoflowcfd.cli.solve.solver_factory.build_single_node_solver"


_WRITERS = ("autoflowcfd.cli.solve.checkpoint_io.periodic_checkpoint_callback",
            "autoflowcfd.cli.solve.checkpoint_io.write_single_node_outputs",
            "autoflowcfd.cli.solve.checkpoint_io.resume_run.periodic_checkpoint_callback",
            "autoflowcfd.cli.solve.checkpoint_io.resume_run.write_single_node_outputs")


@pytest.fixture
def writers():
    """结果落盘函数的替身（替身求解器不能真的写 checkpoint）；返回 {名字: mock}。"""
    with patch(_WRITERS[0]) as cb, patch(_WRITERS[1]) as final, \
         patch(_WRITERS[2]) as rcb, patch(_WRITERS[3]) as rfinal, \
         patch("autoflowcfd.cli.solve.mesh_loader.save_volume_mesh_cache", return_value="out/volume_mesh.pkl") as save:
        yield {"callback": cb, "final": final, "resume_callback": rcb, "resume_final": rfinal, "save": save}


@pytest.fixture(autouse=True)
def _no_reference_area_estimate(request):
    """替身体网格估算不了参考面积；求解入口的参考面积解析单独由 TestPost 验证。"""
    if request.node.cls is TestPost:
        yield None
        return
    with patch("autoflowcfd.cli.solve.aero_coefficients.resolve_reference_area") as resolve:
        yield resolve


def _solver(iterations=7):
    solver = MagicMock()
    solver.solve.return_value = SimpleNamespace(converged=True, iterations=iterations, final_residual=1e-9)
    return solver


class TestBasics:
    def test_initialization_and_environment(self):
        api = AutoFlowCFDAPI(verbose=True)
        assert api.verbose is True and api.solver is None and api.volume_mesh is None
        env = api.check_environment()
        assert {"python_version", "platform", "autoflowcfd_version", "gpu_available"} <= set(env)
        assert api.get_version() == get_version()

    def test_create_api(self):
        api = create_api(verbose=True)
        assert isinstance(api, AutoFlowCFDAPI) and api.verbose is True


class TestConfig:
    def test_create_configs_use_config_field_names(self):
        api = create_api()
        steady = api.create_steady_config(backend="gpu", order=3, turbulence="sst", time_scheme="newton-krylov")
        assert steady.backend.value == "gpu" and steady.order == 3
        assert steady.turbulence is TurbulenceModel.SST_KW and steady.time_scheme is S.NEWTON_KRYLOV
        transient = api.create_transient_config(turbulence="ddes", dt=1e-4, total_time=0.3)
        assert transient.n_steps == 3000
        with pytest.raises(TypeError):
            api.create_steady_config(turbulence_model="sst")          # 旧参数名不再接受
        with pytest.raises(ValueError, match="湍流模型"):
            api.create_transient_config(turbulence="kepsilon")

    def test_load_config_returns_a_validated_object(self, tmp_path):
        path = tmp_path / "c.yaml"
        path.write_text("mode: steady\nbackend: cpu\norder: 3\nturbulence: sst\nmax_iter: 300\n", encoding="utf-8")
        cfg = create_api().load_config(path)
        assert isinstance(cfg, SteadyConfig) and cfg.order == 3 and cfg.max_iter == 300
        path.write_text("mode: steady\norder: 9\n", encoding="utf-8")
        with pytest.raises(ValueError, match="目标阶数"):
            create_api().load_config(path)


class TestGrid:
    def test_load_grid_runs_the_real_surface_validator(self):
        grid = MagicMock(node_count=10, cell_count=4)
        with patch("autoflowcfd.grid.nas_io.parser.NASParser") as parser, \
             patch("autoflowcfd.grid.validation.validator.GridValidator") as validator:
            parser.return_value.parse.return_value = grid
            validator.return_value.validate.return_value = {"passed": True, "summary": ""}
            api = create_api()
            assert api.load_grid("x.nas") is grid
        validator.assert_called_once_with(grid)
        assert api.grid_data is grid

    def test_validate_volume_mesh_is_the_solve_quality_gate(self):
        report = SimpleNamespace(passed=False)
        with patch("autoflowcfd.grid.validation.quality_validator.MeshQualityValidator") as validator:
            validator.return_value.validate_volume_mesh.return_value = report
            assert create_api().validate_volume_mesh(MagicMock()) is report


@pytest.mark.usefixtures("writers")
class TestRunSteady:
    def test_defaults_come_from_the_config_class(self):
        solver = _solver()
        with patch(_GATE) as gate, patch(_FACTORY, return_value=solver) as factory:
            api = create_api()
            vm = MagicMock()
            result = api.run_steady(vm)
        default = SteadyConfig()
        gate.assert_called_once_with(vm, default.order, skip_quality_check=False)
        args, kw = factory.call_args
        assert args[0] == default.backend.value
        assert kw["turb_model_name"] == "sst" and kw["time_scheme"] is default.time_scheme
        assert kw["low_mach_precond"] is True and kw["gpu_device"] is None
        solve_kw = solver.solve.call_args.kwargs
        assert solve_kw["max_iter"] == default.max_iter and solve_kw["tol"] == default.convergence_tol
        assert result.iterations == 7 and api.solver is solver and api.volume_mesh is vm

    def test_fields_override_the_config(self, _no_reference_area_estimate):
        solver = _solver()
        cfg = SteadyConfig(order=1, max_iter=50, turbulence="sa")
        vm = MagicMock()
        with patch(_GATE), patch(_FACTORY, return_value=solver) as factory:
            create_api().run_steady(vm, cfg, order=3, time_scheme="newton-krylov", aoa_deg=4.0, reference_area=1.5)
        _no_reference_area_estimate.assert_called_once_with(solver, vm, 1.5)
        kw = factory.call_args.kwargs
        assert kw["order"] == 3 and kw["turb_model_name"] == "sa" and kw["time_scheme"] is S.NEWTON_KRYLOV
        assert kw["aoa_deg"] == 4.0
        assert solver.solve.call_args.kwargs["max_iter"] == 50
        assert cfg.order == 1, "调用方的 config 对象不能被改动"

    def test_unknown_field_and_wrong_config_type_are_rejected(self):
        api = create_api()
        with pytest.raises(TypeError, match="没有字段"):
            api.run_steady(MagicMock(), tol=1e-8)                    # 字段名是 convergence_tol
        with pytest.raises(TypeError, match="SteadyConfig"):
            api.run_steady(MagicMock(), TransientConfig())

    def test_quality_gate_failure_stops_the_run(self):
        """修复前 API 直接 `HighOrderMesh.load_from_volume_mesh`，质量门 FAILED 的网格照样求解。"""
        from autoflowcfd.grid.validation.solve_gate import MeshQualityGateError

        bad = SimpleNamespace(passed=False, summary=lambda: "FAILED")
        with patch("autoflowcfd.grid.validation.quality_validator.MeshQualityValidator") as validator, \
             patch(_FACTORY) as factory:
            validator.return_value.validate_volume_mesh.return_value = bad
            with pytest.raises(MeshQualityGateError, match="质量门"):
                create_api().run_steady(MagicMock())
        factory.assert_not_called()


@pytest.mark.usefixtures("writers")
class TestRunTransient:
    def test_steps_from_total_time_or_max_iter(self):
        for fields, steps in ((dict(total_time=0.3, dt=1e-4), 3000), (dict(max_iter=40), 40)):
            solver = _solver()
            with patch(_GATE), patch(_FACTORY, return_value=solver) as factory:
                create_api().run_transient(MagicMock(), **fields)
            kw = solver.solve.call_args.kwargs
            assert kw["max_iter"] == steps and kw["tol"] == 0.0
            assert factory.call_args.kwargs["turb_model_name"] == "ddes"
            assert factory.call_args.kwargs["time_accurate"] is False          # 默认 dual_time

    def test_explicit_scheme_is_time_accurate_and_init_from_is_used(self):
        solver = _solver()
        with patch(_GATE), patch(_FACTORY, return_value=solver) as factory, \
             patch("autoflowcfd.cli.solve.checkpoint_io.start_single_node_from_checkpoint") as start:
            create_api().run_transient(MagicMock(), time_scheme="rk3", init_from_checkpoint="steady.h5")
        assert factory.call_args.kwargs["time_accurate"] is True
        start.assert_called_once_with(solver, "steady.h5")


class TestWritesResults:
    """修复前 API 从不落盘：配置的 output_dir / checkpoint_interval 被静默忽略，中断即丢失、结果不能续算。"""

    def test_run_writes_checkpoints_and_results_resumable_from_the_saved_mesh(self, writers):
        solver = _solver(iterations=12)
        vm = MagicMock()
        with patch(_GATE), patch(_FACTORY, return_value=solver):
            create_api().run_steady(vm, output_dir="case_a", checkpoint_interval=25, turbulence="sa")
        writers["save"].assert_called_once_with(vm, "case_a")
        cb_args = writers["callback"].call_args
        assert cb_args.args == (25, "case_a", "out/volume_mesh.pkl", "sa", "cpu")
        assert solver.solve.call_args.kwargs["checkpoint_callback"] is writers["callback"].return_value
        final = writers["final"].call_args
        assert final.args == (solver, "case_a", 12, "out/volume_mesh.pkl", "sa", "cpu")

    def test_saved_mesh_is_the_volume_mesh_pickle(self, tmp_path):
        import pickle

        from autoflowcfd.cli.solve.mesh_loader import save_volume_mesh_cache as _save_volume_mesh

        path = _save_volume_mesh({"cells": [1, 2, 3]}, str(tmp_path / "out"))
        with open(path, "rb") as f:
            assert pickle.load(f) == {"cells": [1, 2, 3]}
        assert path.endswith("volume_mesh.pkl")


class TestResume:
    def test_missing_checkpoint(self):
        with pytest.raises(FileNotFoundError, match="Checkpoint file not found"):
            create_api().resume_simulation("nonexistent.h5")

    _META = {"input_file": "case/volume_mesh.pkl", "turbulence_model": "sa", "backend": "cpu"}

    def _resume(self, tmp_path, scheme, metadata, **kw):
        metadata = {**self._META, **metadata}
        (tmp_path / "checkpoints").mkdir(exist_ok=True)
        ck = tmp_path / "checkpoints" / "c.h5"
        ck.write_bytes(b"")
        solver = _solver()
        solver.time_integrator = SimpleNamespace(scheme=scheme)
        vm = MagicMock()
        with patch("autoflowcfd.cli.solve.checkpoint_io.rebuild_from_checkpoint",
                   return_value=(solver, 150, metadata, vm)) as rebuild:
            api = create_api()
            api.resume_simulation(str(ck), **kw)
        return api, solver, rebuild, vm

    def test_dual_time_resume_keeps_the_checkpoint_time_step(self, tmp_path, writers):
        """修复前 API 续算写死 dt=1e-3、tol=1e-6：续算瞬态时静默改掉物理时间步、按残差提前停；且不落盘。"""
        api, solver, _, vm = self._resume(tmp_path, S.DUAL_TIME, {"dt": 2.5e-4}, max_iter=10)
        kw = solver.solve.call_args.kwargs
        assert kw["dt"] == 2.5e-4 and kw["tol"] == 0.0
        assert api.solver is solver and api.volume_mesh is vm      # 续算后可直接导出
        cb = writers["resume_callback"].call_args
        assert cb.args[1] == str(tmp_path) and cb.kwargs["iteration_offset"] == 150
        final = writers["resume_final"].call_args
        assert final.args[2] == 150 + 7

    def test_steady_resume_options_reach_the_rebuild(self, tmp_path, writers):
        _, solver, rebuild, _ = self._resume(tmp_path, S.NEWTON_KRYLOV, {}, max_iter=5, time_scheme="rk3",
                                             low_mach_precond=False, cfl_max=50.0)
        kw = rebuild.call_args.kwargs
        assert kw["time_scheme"] == "rk3" and kw["low_mach_precond"] is False and kw["cfl_max"] == 50.0
        assert solver.solve.call_args.kwargs["tol"] == SteadyConfig().convergence_tol

    def test_zero_steps_only_restores(self, tmp_path):
        _, solver, _, _ = self._resume(tmp_path, S.SSP_RK3, {})
        solver.solve.assert_not_called()


class TestPost:
    def test_reference_area_defaults_to_the_windward_projection(self):
        """修复前缺省 reference_area=1.0：系数量级随网格单位任意。现在与 CLI 同一个估算函数。"""
        api = create_api()
        api.solver = MagicMock()
        api.solver.host_view.return_value = SimpleNamespace(freestream={"vel_inf": 30.0}, bc_overrides={})
        api.volume_mesh = MagicMock()
        coeffs = SimpleNamespace(to_dict=lambda: {"Cd": 0.3})
        with patch("autoflowcfd.cli.solve.aero_coefficients.reference_area_for", return_value=2.2) as area, \
             patch("autoflowcfd.postprocess.fr_coefficients.compute_aerodynamic_coefficients_fr",
                   return_value=coeffs) as compute:
            assert api.calculate_coefficients() == {"Cd": 0.3}
        assert area.call_args.args[2] is None
        assert compute.call_args.kwargs["reference_area"] == 2.2

    def test_methods_need_a_solver(self):
        api = create_api()
        with pytest.raises(ValueError, match="run_steady"):
            api.calculate_coefficients(reference_area=2.0)
        with pytest.raises(ValueError, match="run_steady"):
            api.export_vtk("output.vtk")
        assert api.get_convergence_history() == {"iterations": [], "residuals": []}


class TestGetVersion:
    def test_version_matches_module(self):
        import autoflowcfd

        assert get_version() == autoflowcfd.__version__ and len(get_version()) > 0


class TestApiGenerateVolumeMesh:
    """`AutoFlowCFDAPI.generate_volume_mesh` 与 CLI `grid generate-volume` 走同一个入口（此前门面多传一个位置参数
    `method` 必崩，且 API 自己构造 VolumeMeshGenerator、生成的网格不带面网格数据）。"""

    def _surface(self):
        import numpy as np

        xyz = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        nodes = SimpleNamespace(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2])
        return SimpleNamespace(
            nodes=nodes, cells=SimpleNamespace(connectivity=np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])),
            boundaries=SimpleNamespace(groups={}, bc_types={}), metadata=SimpleNamespace(bounding_box=[0, 1, 0, 1, 0,
                                                                                                       1]))

    def test_forwards_parameters_and_keeps_surface_mesh(self, monkeypatch):
        import autoflowcfd.grid.mesh_gen.tetgen.volume_mesh_generator as vmg

        seen = {}

        class FakeGenerator:
            def __init__(self, **kwargs):
                seen.update(kwargs)

            def generate_from_surface(self, **kwargs):
                return SimpleNamespace(node_count=4, cell_count=1, total_volume=1.0 / 6.0)

        monkeypatch.setattr(vmg, "VolumeMeshGenerator", FakeGenerator)
        api = create_api()
        mesh = api.generate_volume_mesh(self._surface(), growth_rate=1.15, min_cell_size=2e-4, bl_layers=6)
        assert seen == {"growth_rate": 1.15, "min_cell_size": 2e-4, "bl_layers": 6}
        assert api.volume_mesh is mesh
        assert mesh.surface_mesh["faces"].shape == (4, 3)

    def test_unknown_parameter_is_rejected(self):
        with pytest.raises(TypeError, match="method"):
            create_api().generate_volume_mesh(self._surface(), method="tetrahedral")


def test_quickstart_style_example_uses_existing_names():
    """包文档与门面文档里的示例参数名必须真实存在（此前示例对面网格调 run_steady、传 turbulence/convergence_tol）。"""
    import inspect

    sig = inspect.signature(AutoFlowCFDAPI.calculate_coefficients)
    assert "result" not in sig.parameters and "reference_area" in sig.parameters
    assert "result" not in inspect.signature(AutoFlowCFDAPI.export_vtk).parameters
