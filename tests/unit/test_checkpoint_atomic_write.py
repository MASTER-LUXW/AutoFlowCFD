"""checkpoint 原子写出与 NumPy 2 兼容的字符串属性（I-85）。

此前 `CheckpointManager.save` 直接以最终文件名截断写入：写到一半出错（崩溃、断电、磁盘满）即留下同名的损坏文件，
且同迭代号的旧 checkpoint 已被覆盖。现在先写同目录临时文件、写完 `os.replace`。字符串属性此前用 `np.string_`
（NumPy 2.0 已移除）。
"""

from pathlib import Path

import numpy as np
import pytest

import autoflowcfd
from autoflowcfd.core.utils.checkpoint import CheckpointManager
from autoflowcfd.core.utils.checkpoint_load import load_checkpoint


class _Unconvertible:
    """`np.asarray(..., dtype=float64)` 时抛异常：模拟写到一半出错。"""

    def __array__(self, dtype=None):
        raise RuntimeError("模拟写出中途失败")


def _save(manager, value, extra=None):
    return manager.save(np.full((4, 5), value), {"iterations": [7]}, iteration=7, metadata={"order": 1},
                        extra_fields=extra)


def test_failed_write_keeps_the_previous_checkpoint_and_leaves_no_temp_file(tmp_path):
    manager = CheckpointManager(output_dir=str(tmp_path), quiet=True)
    path = _save(manager, 1.0)
    with pytest.raises(RuntimeError, match="模拟写出中途失败"):
        _save(manager, 2.0, extra={"U_sps": _Unconvertible()})
    solution, _, iteration, _ = load_checkpoint(path)
    np.testing.assert_array_equal(solution, 1.0)
    assert iteration == 7
    assert sorted(p.name for p in Path(path).parent.iterdir()) == [Path(path).name]


def test_successful_write_replaces_the_file(tmp_path):
    manager = CheckpointManager(output_dir=str(tmp_path), quiet=True)
    _save(manager, 1.0)
    path = _save(manager, 3.0)
    solution, _, _, metadata = load_checkpoint(path)
    np.testing.assert_array_equal(solution, 3.0)
    assert sorted(p.name for p in Path(path).parent.iterdir()) == [Path(path).name]


def test_no_numpy_string_alias_in_sources():
    """`np.string_` 在 NumPy 2.0 已移除（`np.bytes_` 是同一个类型，1.x 与 2.x 都有）。"""
    src = Path(autoflowcfd.__file__).parent
    offenders = [str(p.relative_to(src)) for p in src.rglob("*.py") if "np.string_(" in p.read_text(encoding="utf-8")]
    assert offenders == []
