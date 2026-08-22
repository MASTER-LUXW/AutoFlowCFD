"""Unit tests for config/loader.py's template <-> loader round-trip.

Real bug caught here (fixed 2026-08-21): `config_commands.py`'s hardcoded
transient template text used the key `time_integration`, but
`TransientConfig`'s actual dataclass field is `time_scheme`. `ConfigLoader.
_merge_defaults` only recognizes keys that are fields of the target
dataclass - an unrecognized key is logged as an "unknown config key" and
silently dropped, so any non-default value a user wrote under
`time_integration:` in a generated template had zero effect: the loaded
config always silently fell back to the default `time_scheme`, without any
visible error (only a log warning easy to miss).
"""

from autoflowcfd.config import ConfigLoader, TransientConfig
from autoflowcfd.config.solver_config import TimeIntegrationScheme


class TestConfigLoaderTemplateRoundTrip:
    def test_transient_time_scheme_key_is_not_silently_dropped(self, tmp_path):
        """A non-default `time_scheme` value in the YAML must actually reach
        the loaded TransientConfig, not be silently ignored in favor of the
        dataclass default."""
        yaml_path = tmp_path / "transient.yaml"
        yaml_path.write_text(
            "mode: transient\ntime_scheme: rk3\ndt: 1.0e-4\ntotal_time: 0.3\n",
            encoding="utf-8",
        )

        config = ConfigLoader().load(str(yaml_path))

        assert isinstance(config, TransientConfig)
        assert config.time_scheme == TimeIntegrationScheme.RK3

    def test_generated_transient_template_uses_the_real_field_name(self):
        """The hardcoded template text in `cli/config_commands.py` must use
        `time_scheme:` (the real TransientConfig field), not a differently
        named key that would be silently dropped by the loader."""
        from autoflowcfd.cli.config_commands import init as config_init_cmd

        # Reach into the module source rather than invoking the CLI, so this
        # test pins the exact literal key name independent of Click's I/O.
        # `init` is a Click Command wrapping the real function - inspect its
        # `.callback`, not the Command object itself.
        import inspect

        source = inspect.getsource(config_init_cmd.callback)
        assert "time_scheme:" in source
        assert "time_integration:" not in source

    def test_mode_key_does_not_trigger_unknown_config_key_warning(self, tmp_path, caplog):
        """`mode` is a legitimate top-level routing key consumed by
        `ConfigLoader.load()` itself - it must not be reported as an
        unrecognized config key when merging with dataclass defaults."""
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
