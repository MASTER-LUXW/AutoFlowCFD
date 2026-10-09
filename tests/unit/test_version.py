"""包元数据与版本信息的单元测试。"""

import pytest
from autoflowcfd import __version__, get_version, get_info


class TestVersionInfo:
    """版本与包信息的测试。"""

    def test_version_format(self) -> None:
        """版本号符合 semver 格式。"""
        assert isinstance(__version__, str)
        parts = __version__.split(".")
        assert len(parts) == 3
        assert all(part.isdigit() for part in parts)

    def test_get_version_returns_string(self) -> None:
        """get_version() 返回字符串。"""
        version = get_version()
        assert isinstance(version, str)
        assert version == __version__

    def test_get_info_returns_dict(self) -> None:
        """get_info() 返回含必需键的字典。"""
        info = get_info()
        assert isinstance(info, dict)
        assert "name" in info
        assert "version" in info
        assert "author" in info
        assert "license" in info

    def test_get_info_version_matches(self) -> None:
        """info 里的版本号与 __version__ 一致。"""
        info = get_info()
        assert info["version"] == __version__

    def test_package_name(self) -> None:
        """包名正确。"""
        info = get_info()
        assert info["name"] == "AutoFlowCFD"

    def test_license_is_apache(self) -> None:
        """许可证是 Apache 2.0。"""
        info = get_info()
        assert info["license"] == "Apache-2.0"
