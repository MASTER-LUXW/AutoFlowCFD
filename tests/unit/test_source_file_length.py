"""单个源文件不超过 500 行（项目规范：超过即按功能拆分），覆盖 src 与 tests。"""

from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_LIMIT = 500


def _python_files(sub: str):
    return sorted(p for p in (_ROOT / sub).rglob("*.py") if "__pycache__" not in p.parts)


@pytest.mark.parametrize("sub", ["src", "tests"])
def test_no_python_file_exceeds_the_line_limit(sub):
    too_long = []
    for p in _python_files(sub):
        with open(p, encoding="utf-8") as f:
            n = sum(1 for _ in f)
        if n > _LIMIT:
            too_long.append(f"{p.relative_to(_ROOT)}: {n}")
    assert not too_long, f"超过 {_LIMIT} 行的文件需要按功能拆分：\n" + "\n".join(too_long)
