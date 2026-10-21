"""包版本号的唯一来源：pyproject.toml 的 [project] version。

已安装（含 `pip install -e .`）时读包元数据；源码树里直接运行（未安装）时读同一份 pyproject.toml。
2026-10-20 以前 `__init__.py` 另写一份 "0.1.0"。
"""

import re
from importlib import metadata
from pathlib import Path


def read_version() -> str:
    """当前版本号。

    Raises:
        RuntimeError: 既没有安装包元数据，源码树里也找不到 pyproject.toml 的版本号
    """
    try:
        return metadata.version("autoflowcfd")
    except metadata.PackageNotFoundError:
        pass
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    if pyproject.is_file():
        match = re.search(r'(?ms)^\[project\]\s.*?^version\s*=\s*"([^"]+)"', pyproject.read_text(encoding="utf-8"))
        if match:
            return match.group(1)
    raise RuntimeError("找不到 autoflowcfd 的版本号：包未安装（pip install -e .），源码树里也没有 pyproject.toml")
