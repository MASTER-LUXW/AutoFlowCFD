"""
自适应 CFL 三元组必须在**每一条入口**上可配、且默认值一致。

## 为什么需要这份测试

这个缺口在本项目已经复发了**四次**，每次都是"某一条入口漏传"：

    2026-09-07  `--cfl-start/--cfl-max` 加进 `solve steady`，但控制器
                构造点恒用硬编码默认值（cfl_start/cfl_max 从未接上）
    2026-09-15  `--cfl-min` 新增；同日发现它在**全部四条分布式路径**上
                被静默丢弃（提交 34d9f39）
    2026-09-15  配置层 `SteadyConfig.cfl_min` 新增 = 0.01，但 CLI 默认
                仍是 0.05（提交 5e4e18a 只改了配置层的一半）
    2026-09-17  `solve resume` 整条命令被漏掉（单机缺 --cfl-min、分布式
                三个全丢）；`solve transient` 一个选项都没有；
                `FRSolver.__init__` 的构造默认仍是 0.05

后果每次都一样且致命：控制器 `cfl_min` 默认 0.05 **高于**本项目在
cube_demo 与 plate_demo 两张真实 ANSA 网格上实测稳定的 CFL ~0.03，而
`adaptive_cfl/` 模块文档第 12 条记录的实测结论是"越界一次之后收缩救
不回来"。所以下限一旦高于实测稳定点，那个已验证可用的工作点**通过任何
入口都到不了**，必然发散——而用户在命令行传 `--cfl-start 0.03` 也无济于
事，因为下限压在上面。

单看某一次修复都"改完了"，缺的是一条**跨入口的一致性约束**。本文件就是
那条约束：它枚举全部入口，任何新入口漏掉这三个参数都会在这里失败。
"""

import inspect

import pytest

from autoflowcfd.cli.main import cli

#: 三处默认值必须一致的"真值"。0.01 的依据：0.05 会恰好等于
#: `cfl_init` 的默认 0.05，那样控制器**一步也收缩不了**；而且它高于两张
#: 真实网格实测稳定的 ~0.03。方向安全——下限只允许控制器收缩得更多，
#: 绝不会抬高 CFL，所以不可能把原本稳定的运行变成不稳定。
EXPECTED_CFL_MIN = 0.01

_CLI_COMMANDS_WITH_CFL = [
    ("steady", ("cfl_start", "cfl_max", "cfl_min")),
    ("transient", ("cfl_start", "cfl_max", "cfl_min")),
    ("resume", ("cfl_start", "cfl_max", "cfl_min")),
]


def _params(cmd_name):
    return {p.name: p for p in cli.commands["solve"].commands[cmd_name].params}


class TestCliOptionsExist:
    @pytest.mark.parametrize("cmd,names", _CLI_COMMANDS_WITH_CFL)
    def test_options_present(self, cmd, names):
        params = _params(cmd)
        for n in names:
            assert n in params, (
                f"`solve {cmd}` 缺 --{n.replace('_', '-')} 选项。"
                f"这三个参数必须在每条入口上都可配，理由见本文件模块文档"
            )

    @pytest.mark.parametrize("cmd,names", _CLI_COMMANDS_WITH_CFL)
    def test_cli_defaults_defer_to_the_control_law(self, cmd, names):
        """2026-09-25 起入口一律默认 None，由 `build_cfl_policy` 按时间格式
        取 CFL 律签名里的默认值——"各入口默认值一致"由**同一个来源**保证，
        而不是由各入口各写一份相同的数字保证（后者正是本文件模块文档里
        四次复发的根源）。引入隐式格式后同一组选项要服务两个默认值差两个
        数量级的 CFL 律，入口再写死任何一个都必然让另一个格式拿到错值。"""
        p = _params(cmd)
        for n in names:
            assert p[n].default is None, (
                f"`solve {cmd} --{n.replace('_', '-')}` 默认值 {p[n].default} 不是 None")


class TestSolverConstructorDefaults:
    def test_frsolver_defaults_defer_to_the_control_law(self):
        """`FRSolver.__init__` 同样默认 None（此前写死显式那一组，隐式格式
        不传 CFL 时会静默拿到 0.03）。"""
        from autoflowcfd.core.fr_solver.solver import FRSolver

        p = inspect.signature(FRSolver.__init__).parameters
        for n in ("cfl_start", "cfl_max", "cfl_min"):
            assert p[n].default is None, f"FRSolver.__init__ 的 {n} 默认值不是 None"

    def test_explicit_controller_defaults(self):
        """显式控制器签名是显式格式默认值的唯一来源：下限 0.01、三元组自洽。"""
        from autoflowcfd.core.time_integration.adaptive_cfl import AdaptiveCFLController

        c = AdaptiveCFLController()
        assert c.cfl_min == EXPECTED_CFL_MIN
        assert c.cfl_min <= c.cfl_start <= c.cfl_max

    def test_ser_defaults_are_self_consistent(self):
        from autoflowcfd.core.time_integration.adaptive_cfl import SERCFLController

        c = SERCFLController()
        assert c.cfl_min <= c.cfl_start <= c.cfl_max


class TestConfigLayer:
    """配置层（`SteadyConfig` / `TransientConfig`）必须有这三个字段，且与 CLI、`FRSolver.__init__` 一样默认 None。"""

    @pytest.mark.parametrize("cls_name", ["SteadyConfig", "TransientConfig"])
    def test_fields_exist_and_defer_to_the_control_law(self, cls_name):
        """2026-10-05 以前配置层写死显式那一组（0.03/0.06/0.01）：CLI 与求解器 2026-09-25 已改为 None，这份拷贝被
        漏掉，经 API 选隐式格式时隐式 CFL 律（默认上限 1e8）会被压在 0.06 下。"""
        from autoflowcfd.config import solver_config

        cfg = getattr(solver_config, cls_name)()
        for field in ("cfl_init", "cfl_max", "cfl_min"):
            assert hasattr(cfg, field), (
                f"{cls_name} 缺 {field} 字段 —— YAML/config 用户配置不出它，"
                f"而显式稳态路径上自适应控制器是激活的"
            )
            assert getattr(cfg, field) is None, f"{cls_name}.{field} 默认值不是 None"


def _factory_kwargs(fn_name, config):
    """用替身网格/求解器跑一次 API 入口，返回求解器工厂收到的关键字参数。"""
    from unittest.mock import MagicMock, patch

    from autoflowcfd.api import AutoFlowCFDAPI
    # 先导入：`mesh_loader` 在模块导入时按名字绑定 `solve_gate.high_order_mesh_for_solver`。若它在下面的补丁生效期间
    # 才第一次被导入（替身目标 `...mesh_loader.save_volume_mesh_cache` 就会触发），绑定到的是 MagicMock，补丁撤掉
    # 后仍留在模块里，之后同一进程里所有走 `mesh_loader` 的测试（如 `utils benchmark`）拿到的网格都是替身
    import autoflowcfd.cli.solve.mesh_loader  # noqa: F401

    solver = MagicMock()
    solver.solve.return_value = MagicMock(iterations=1)
    with patch("autoflowcfd.grid.validation.solve_gate.high_order_mesh_for_solver"), \
         patch("autoflowcfd.cli.solve.solver_factory.build_single_node_solver", return_value=solver) as factory, \
         patch("autoflowcfd.cli.solve.aero_coefficients.resolve_reference_area"), \
         patch("autoflowcfd.cli.solve.mesh_loader.save_volume_mesh_cache", return_value="vm.pkl"), \
         patch("autoflowcfd.cli.solve.checkpoint_io.periodic_checkpoint_callback"), \
         patch("autoflowcfd.cli.solve.checkpoint_io.write_single_node_outputs"):
        getattr(AutoFlowCFDAPI(), fn_name)(MagicMock(), config)
    return factory.call_args.kwargs


class TestApiForwardsConfigCfl:
    """`api.run_steady` / `api.run_transient` 必须把 config 的三个字段映射成求解器的构造参数
    （`cfl_init -> cfl_start`）——漏掉任何一个都会让该字段静默失效。用替身网格与求解器真实调用入口。"""

    @pytest.mark.parametrize("fn_name,cls_name", [("run_steady", "SteadyConfig"), ("run_transient", "TransientConfig")])
    def test_config_cfl_forwarded(self, fn_name, cls_name):
        from autoflowcfd.config import solver_config

        cfg = getattr(solver_config, cls_name)(cfl_min=0.01, cfl_init=0.02, cfl_max=0.05)
        kw = _factory_kwargs(fn_name, cfg)
        assert (kw["cfl_start"], kw["cfl_max"], kw["cfl_min"]) == (0.02, 0.05, 0.01)


class TestDistributedEntriesForwardCfl:
    """分布式入口（稳态/瞬态/两个 resume 重建）都必须接收并转发三个参数。

    历史上这些是最容易漏的一层：`solve steady` 的四条分布式构造点在
    2026-09-15 才补齐，`solve resume` 的四条与 `solve transient` 的两条
    到 2026-09-17 才补齐。
    """

    #: 按**模块名**而不是文件路径：本项目把超 500 行的模块陆续拆成子包，
    #: 硬编码 `.py` 路径在拆包后直接 FileNotFoundError。`module_source`
    #: 会把包的全部子模块拼进来，新增子模块也不用维护清单。
    _MODULES = [
        "autoflowcfd.cli.solve.run",
        "autoflowcfd.cli.solve.distributed_build",
        "autoflowcfd.cli.solve.distributed_checkpoint_io",
        "autoflowcfd.cli.solve.checkpoint_io",
    ]

    @pytest.mark.parametrize("rel", _MODULES)
    def test_file_passes_all_three(self, rel):
        from tests.unit._module_source import module_source

        src = module_source(rel)
        for name in ("cfl_start", "cfl_max", "cfl_min"):
            assert f"{name}=" in src, (
                f"{rel} 里没有出现 `{name}=` —— 该入口的求解器构造点很可能"
                f"漏传了这个参数，那会让它静默使用控制器默认值"
            )
