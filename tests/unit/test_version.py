"""Unit tests for package metadata and version info."""

import pytest
from autoflowcfd import __version__, get_version, get_info


class TestVersionInfo:
    """Test suite for version and package information."""

    def test_version_format(self) -> None:
        """Test that version follows semver format."""
        assert isinstance(__version__, str)
        parts = __version__.split(".")
        assert len(parts) == 3
        assert all(part.isdigit() for part in parts)

    def test_get_version_returns_string(self) -> None:
        """Test that get_version() returns a string."""
        version = get_version()
        assert isinstance(version, str)
        assert version == __version__

    def test_get_info_returns_dict(self) -> None:
        """Test that get_info() returns a dictionary with required keys."""
        info = get_info()
        assert isinstance(info, dict)
        assert "name" in info
        assert "version" in info
        assert "author" in info
        assert "license" in info

    def test_get_info_version_matches(self) -> None:
        """Test that version in info matches __version__."""
        info = get_info()
        assert info["version"] == __version__

    def test_package_name(self) -> None:
        """Test that package name is correct."""
        info = get_info()
        assert info["name"] == "AutoFlowCFD"

    def test_license_is_apache(self) -> None:
        """Test that license is Apache 2.0."""
        info = get_info()
        assert info["license"] == "Apache-2.0"
