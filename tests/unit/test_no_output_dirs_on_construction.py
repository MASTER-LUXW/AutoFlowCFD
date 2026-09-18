"""构造配置/管理器对象**不许**在文件系统上留下任何目录。

## 为什么需要这条测试

用户**两次**明确要求（2026-09-03、2026-09-18）：项目文件夹里不允许出现
`checkpoints` / `results` / `transient_results` 目录。而 2026-09-18 排查
发现它们的真实来源不是某条忘了加 `--output` 的 CLI 命令，而是**构造对象
本身的副作用**：

  * `SolverConfig.__post_init__` 里有 `os.makedirs(self.output_dir)`，
    而 `output_dir` 默认是 `./results`（定常）/ `./transient_results`
    （瞬态）—— 只要有人构造 `SteadyConfig()`，当前工作目录下就凭空出现
    那个目录；
  * `CheckpointManager.__init__` 里有
    `self.checkpoint_dir.mkdir(parents=True)`，同样相对 CWD。

后果是**跑一遍 `pytest tests/unit` 就在仓库根目录留下三个空目录**
（实测 2026-09-18 03:05~03:07 的时间戳正是那次全量测试）。两处的
`makedirs` 还都是**冗余**的：真正写输出的地方各自都建目录。

所以这条测试钉的不是"某个命令要记得传 --output"，而是**更根本的那条：
值对象/管理器对象的构造必须无文件系统副作用**。

判据方式：在一个空的临时目录里切过去构造对象，然后断言该目录**仍然是
空的**。这比"检查仓库根目录有没有那三个名字"强得多——后者只能在污染
已经发生之后发现，而且换个目录名就漏。
"""

import os

import pytest


@pytest.fixture
def empty_cwd(tmp_path, monkeypatch):
    """切到一个空的临时目录，退出时切回。"""
    monkeypatch.chdir(tmp_path)
    assert not list(tmp_path.iterdir()), "临时目录应当是空的"
    return tmp_path


def _assert_still_empty(root, what):
    leftovers = sorted(p.name for p in root.iterdir())
    assert not leftovers, (
        f"构造 {what} 之后当前目录出现了 {leftovers} —— 值对象/管理器的"
        f"构造不允许有文件系统副作用（用户两次明确要求项目文件夹里不许"
        f"出现 checkpoints/results/transient_results）")


class TestConfigConstructionIsPure:
    def test_steady_config_creates_no_directory(self, empty_cwd):
        from autoflowcfd.config.solver_config import SteadyConfig

        cfg = SteadyConfig()
        assert cfg.output_dir, "配置仍然要有 output_dir 字段"
        _assert_still_empty(empty_cwd, "SteadyConfig()")

    def test_transient_config_creates_no_directory(self, empty_cwd):
        from autoflowcfd.config.solver_config import TransientConfig

        TransientConfig()
        _assert_still_empty(empty_cwd, "TransientConfig()")

    def test_explicit_ensure_output_dir_does_create_it(self, empty_cwd):
        """显式调用 `ensure_output_dir()` 才建目录——意图必须由调用方表达。"""
        from autoflowcfd.config.solver_config import SteadyConfig

        cfg = SteadyConfig()
        path = cfg.ensure_output_dir()
        assert os.path.isdir(path), f"{path} 应当已被创建"
        assert path == cfg.output_dir

    def test_custom_output_dir_still_not_created(self, empty_cwd):
        from autoflowcfd.config.solver_config import SteadyConfig

        SteadyConfig(output_dir=str(empty_cwd / "deep" / "nested"))
        _assert_still_empty(empty_cwd, "SteadyConfig(output_dir=...)")


class TestCheckpointManagerConstructionIsPure:
    def test_constructing_manager_creates_no_directory(self, empty_cwd):
        h5py = pytest.importorskip(
            "h5py", reason="CheckpointManager 需要 h5py")
        assert h5py is not None
        from autoflowcfd.core.utils.checkpoint import CheckpointManager
        from autoflowcfd.config.solver_config import SteadyConfig

        mgr = CheckpointManager(SteadyConfig(), output_dir=str(empty_cwd),
                                checkpoint_interval=10, quiet=True)
        assert mgr.checkpoint_dir.name == "checkpoints"
        _assert_still_empty(empty_cwd, "CheckpointManager(...)")

    def test_dir_is_created_on_demand(self, empty_cwd):
        pytest.importorskip("h5py", reason="CheckpointManager 需要 h5py")
        from autoflowcfd.core.utils.checkpoint import CheckpointManager
        from autoflowcfd.config.solver_config import SteadyConfig

        mgr = CheckpointManager(SteadyConfig(), output_dir=str(empty_cwd),
                                checkpoint_interval=10, quiet=True)
        mgr._ensure_checkpoint_dir()
        assert mgr.checkpoint_dir.is_dir(), "按需创建应当真的建出来"


class TestRepoRootStaysClean:
    """仓库根目录本身不许有那三个目录（第二道防线）。

    上面几条钉的是"不会再被污染"，这一条钉的是"现在是干净的"——两者
    都要，因为污染源不止一处（任何写死相对路径的新代码都可能重犯）。
    """

    @pytest.mark.parametrize("name", ["checkpoints", "results",
                                      "transient_results"])
    def test_repo_root_has_no_output_dir(self, name):
        import pathlib

        root = pathlib.Path(__file__).resolve().parents[2]
        p = root / name
        assert not p.exists(), (
            f"仓库根目录出现了 {name}/ —— 用户两次明确要求项目文件夹里"
            f"不许有它。先查是谁建的（构造期副作用？忘了传 --output 的"
            f"CLI 命令？），再删除，不要只删不查。")
