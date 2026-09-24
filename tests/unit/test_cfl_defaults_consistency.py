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
`adaptive_cfl.py` 模块文档第 12 条记录的实测结论是"越界一次之后收缩救
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

    @pytest.mark.parametrize("cmd,_names", _CLI_COMMANDS_WITH_CFL)
    def test_cfl_min_default_is_consistent(self, cmd, _names):
        got = _params(cmd)["cfl_min"].default
        assert got == EXPECTED_CFL_MIN, (
            f"`solve {cmd} --cfl-min` 默认值 {got} != {EXPECTED_CFL_MIN}。"
            f"入口之间默认值不一致会让\"同一组参数换条命令跑\"悄悄改变数值"
            f"行为——本项目已因此出过真实发散"
        )

    @pytest.mark.parametrize("cmd,_names", _CLI_COMMANDS_WITH_CFL)
    def test_min_le_start_le_max_by_default(self, cmd, _names):
        p = _params(cmd)
        lo, start, hi = (p["cfl_min"].default, p["cfl_start"].default,
                         p["cfl_max"].default)
        assert lo <= start <= hi, (
            f"`solve {cmd}` 的默认 CFL 三元组不自洽：{lo} / {start} / {hi}。"
            f"下限高于起始值时控制器会把 CFL 抬上去（adaptive_cfl.py 第 11 条）"
        )


class TestSolverConstructorDefaults:
    def test_frsolver_cfl_min_default(self):
        """`FRSolver.__init__` 的构造默认值也必须是 0.01。

        不传 CFL 的调用方吃的就是它：`solve transient` 的 rk3/imex 路径
        （在补齐 CLI 选项之前）、不带 config 的 `api.run_steady/
        run_transient`、以及任何直接构造 FRSolver 的脚本/测试。
        """
        from autoflowcfd.core.fr_solver.solver import FRSolver

        params = inspect.signature(FRSolver.__init__).parameters
        assert params["cfl_min"].default == EXPECTED_CFL_MIN, (
            f"FRSolver.__init__ 的 cfl_min 默认值 "
            f"{params['cfl_min'].default} != {EXPECTED_CFL_MIN}"
        )

    def test_frsolver_defaults_are_self_consistent(self):
        from autoflowcfd.core.fr_solver.solver import FRSolver

        p = inspect.signature(FRSolver.__init__).parameters
        lo, start, hi = (p["cfl_min"].default, p["cfl_start"].default,
                         p["cfl_max"].default)
        assert lo <= start <= hi, f"FRSolver 默认 CFL 三元组不自洽：{lo}/{start}/{hi}"


class TestConfigLayer:
    """配置层（`SteadyConfig` / `TransientConfig`）必须有这三个字段且默认一致。"""

    @pytest.mark.parametrize("cls_name", ["SteadyConfig", "TransientConfig"])
    def test_fields_exist_and_consistent(self, cls_name):
        from autoflowcfd.config import solver_config

        cls = getattr(solver_config, cls_name)
        cfg = cls()
        for field in ("cfl_init", "cfl_max", "cfl_min"):
            assert hasattr(cfg, field), (
                f"{cls_name} 缺 {field} 字段 —— YAML/config 用户配置不出它，"
                f"而 rk3/imex 路径上自适应控制器是激活的"
            )
        assert cfg.cfl_min == EXPECTED_CFL_MIN, (
            f"{cls_name}.cfl_min = {cfg.cfl_min} != {EXPECTED_CFL_MIN}"
        )
        assert cfg.cfl_min <= cfg.cfl_init <= cfg.cfl_max, (
            f"{cls_name} 默认 CFL 三元组不自洽："
            f"{cfg.cfl_min} / {cfg.cfl_init} / {cfg.cfl_max}"
        )


class TestApiForwardsConfigCfl:
    """`api.run_steady` / `api.run_transient` 必须把 config 的三个字段
    映射成 FRSolver 的构造参数（`cfl_init -> cfl_start`）。

    用源码文本检查而不是跑真实求解：这两个入口要真实网格与完整求解器，
    而"有没有接这三个字段"是结构性事实。三个字段名各自出现即足够——
    漏掉任何一个都会让该字段静默失效，那正是本文件要防的。
    """

    @pytest.mark.parametrize("fn_name", ["run_steady", "run_transient"])
    def test_config_cfl_forwarded(self, fn_name):
        from tests.unit._module_source import module_sources

        # api 2026-09-24 拆成子包，`api_mod.__file__` 只是 __init__.py。
        # 先定位到真正定义这个方法的子模块，再在它内部截取方法体（截取
        # 依赖"下一个同缩进 def"，拼接后没有意义）。
        src = next((s for _n, s in module_sources("autoflowcfd.api")
                    if f"def {fn_name}(" in s), None)
        assert src is not None, f"api 包里找不到 def {fn_name}("
        start = src.index(f"def {fn_name}(")
        # 截到下一个同缩进的 def，避免把相邻方法的代码算进来
        nxt = src.find("\n    def ", start + 1)
        body = src[start:nxt if nxt > 0 else len(src)]
        for pair in ('kwargs.setdefault("cfl_start", config.cfl_init)',
                     'kwargs.setdefault("cfl_max", config.cfl_max)',
                     'kwargs.setdefault("cfl_min", config.cfl_min)'):
            assert pair in body, f"api.{fn_name} 没有转发：{pair}"


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
        "autoflowcfd.cli.solve_steady_command",
        "autoflowcfd.cli.solve_transient_distributed",
        "autoflowcfd.cli.solve_distributed_checkpoint_io",
        "autoflowcfd.cli.solve_checkpoint_io",
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
