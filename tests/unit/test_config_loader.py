"""config/loader.py 模板 <-> 加载器往返的单元测试。

这里抓到的真实缺陷（2026-08-21 修复）：`config_commands.py` 里手写的瞬态模板
用的键是 `time_integration`，而 `TransientConfig` 的 dataclass 字段实际叫
`time_scheme`。`ConfigLoader._merge_defaults` 只认目标 dataclass 的字段——
不认识的键记一条"未知配置键"日志后被静默丢弃，于是用户在生成的模板里写在
`time_integration:` 下的任何非默认值都完全不起作用：加载出的配置总是静默
退回默认的 `time_scheme`，没有任何可见的错误（只有一条容易漏看的日志警告）。
"""

from autoflowcfd.config import ConfigLoader, TransientConfig
from autoflowcfd.config.solver_config import TimeIntegrationScheme


class TestConfigLoaderTemplateRoundTrip:
    def test_transient_time_scheme_key_is_not_silently_dropped(self, tmp_path):
        """YAML 里非默认的 `time_scheme` 必须真的到达加载出的 TransientConfig，
        不能被静默忽略而用 dataclass 默认值。
        """
        yaml_path = tmp_path / "transient.yaml"
        yaml_path.write_text(
            "mode: transient\ntime_scheme: rk3\ndt: 1.0e-4\ntotal_time: 0.3\n",
            encoding="utf-8",
        )

        config = ConfigLoader().load(str(yaml_path))

        assert isinstance(config, TransientConfig)
        assert config.time_scheme == TimeIntegrationScheme.SSP_RK3, (
            "YAML 里的 time_scheme 必须真的进到 TransientConfig。"
            "2026-09-18：配置层那个独立的同名枚举已删除，`rk3` 现在解析成"
            "核心层的 SSP_RK3（此前是配置层的 RK3，两者是不同的类）")

    def test_generated_templates_round_trip_to_the_config_defaults(self, tmp_path, caplog):
        """`config init` 的模板由配置类字段生成（2026-10-05 起，此前是手写模板：默认值过时、瞬态模板的
        `turbulence: des` 不是合法取值、加载即失败）：加载回来等于默认配置，且没有未知键警告。"""
        import dataclasses

        from click.testing import CliRunner

        from autoflowcfd.cli.main import cli
        from autoflowcfd.config.loader import load_config
        from autoflowcfd.config.solver_config import SteadyConfig, TransientConfig, TurbulenceModel

        for template, cls in (("steady", SteadyConfig), ("transient", TransientConfig)):
            out = tmp_path / f"{template}.yaml"
            result = CliRunner().invoke(cli, ["config", "init", "--template", template, "-o", str(out)])
            assert result.exit_code == 0, result.output
            text = out.read_text(encoding="utf-8")
            assert ", ".join(t.value for t in TurbulenceModel) in text
            loaded, default = load_config(out), cls()
            for f in dataclasses.fields(cls):
                if f.init:
                    assert getattr(loaded, f.name) == getattr(default, f.name), (template, f.name)
            assert "total_steps" not in text
            if template == "transient":
                assert "time_scheme:" in text and "time_integration:" not in text

    def test_mode_key_does_not_trigger_unknown_config_key_warning(self, tmp_path, caplog):
        """`mode` 是 `ConfigLoader.load()` 自己消费的合法顶层路由键——与 dataclass
        默认值合并时不能把它报成未识别的配置键。
        """
        yaml_path = tmp_path / "steady.yaml"
        yaml_path.write_text("mode: steady\n", encoding="utf-8")

        from loguru import logger
        import sys

        messages = []
        handler_id = logger.add(lambda msg: messages.append(msg.record["message"]), level="WARNING")
        try:
            ConfigLoader().load(str(yaml_path))
        finally:
            logger.remove(handler_id)

        assert not any("mode" in m and "未知的配置键" in m for m in messages)
