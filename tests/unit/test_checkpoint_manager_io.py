"""checkpoint 管理器的读写约定（I-86，2026-10-19）。

* 写出失败直接抛出：此前 `save` 吞掉一切异常只返回 None，求解照样往下跑而没有可续算的检查点；
* 读取失败保留原始异常类型：此前统一改包成 RuntimeError，布局不符的 ValueError 被抹掉；
* 管理器不再接收求解配置：配置哈希在读取端恒为默认配置的哈希，"配置不一致"警告每次续算都报、毫无意义；
  旧 checkpoint 里的 `config_hash` 属性读取时不再出现在 metadata 里，后端等量取写出端记录的真实值。
"""

import h5py
import numpy as np
import pytest

from autoflowcfd.core.utils.checkpoint import CheckpointManager
from autoflowcfd.core.utils.checkpoint_load import load_checkpoint

_HISTORY = {"iterations": [3], "residuals": {"rms": [0.5]}, "cfl_history": [10.0]}


def _write(tmp_path, metadata=None):
    return CheckpointManager(output_dir=str(tmp_path), quiet=True).save(
        np.ones((2, 5)), _HISTORY, 3, metadata=metadata, extra_fields={"U_sps": np.ones((2, 6, 5))})


def test_save_failure_raises(tmp_path):
    blocker = tmp_path / "case"
    blocker.write_text("不是目录")                     # checkpoints/ 建不出来
    with pytest.raises(OSError):
        CheckpointManager(output_dir=str(blocker), quiet=True).save(np.ones((2, 5)), _HISTORY, 1)


def test_load_keeps_the_original_exception_type(tmp_path):
    path = _write(tmp_path)
    with h5py.File(path, "r+") as f:
        f["metadata"].attrs["sp_layout"] = np.string_("future_layout")
    with pytest.raises(ValueError, match="解点存储布局"):
        load_checkpoint(path)
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "missing.h5")


def test_legacy_config_hash_is_not_surfaced_and_backend_is_the_writers(tmp_path):
    path = _write(tmp_path, metadata={"backend": "gpu", "order": 1})
    with h5py.File(path, "r+") as f:                     # 2026-10-19 之前写出端都带这个属性
        f["metadata"].attrs["config_hash"] = np.string_("0a83ce3dcf9365e1")
    solution, history, iteration, meta = CheckpointManager().load(path)
    assert "config_hash" not in meta and "original_backend" not in meta
    assert meta["backend"] == "gpu" and meta["order"] == 1 and meta["iteration"] == 3
    assert solution.shape == (2, 5) and iteration == 3
    assert history == {"iterations": [3], "residuals": {"rms": [0.5]}, "cfl_history": [10.0]}


def test_reading_needs_no_solver_configuration():
    with pytest.raises(TypeError):
        CheckpointManager(config=object())
