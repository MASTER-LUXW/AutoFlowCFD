"""tetgen 收到的质量参数就是调用方给的值（I-143）。

此前 fill_core_volume 内部把传入值悄悄放宽成 max(1.1, minratio-0.2) / max(5, mindihedral-10)：全部调用方传
CORE_TETGEN_MINRATIO=1.15 / CORE_TETGEN_MINDIHEDRAL=15，tetgen 实际收到 1.1 / 5——常量名与文档和实际不符。现在
常量直接写成实际生效值（生成结果逐位不变），不再有隐藏变换。
"""

import numpy as np
import pytest
import tetgen

from autoflowcfd.grid.mesh_gen.tetgen.mesh_tetgen_core import (
    CORE_TETGEN_MINDIHEDRAL, CORE_TETGEN_MINRATIO, fill_core_volume,
)

_PTS = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float)
_TRIS = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                  [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]], dtype=np.int32)


@pytest.mark.parametrize("kwargs, expected", [
    ({}, (CORE_TETGEN_MINRATIO, CORE_TETGEN_MINDIHEDRAL)),
    ({"minratio": 1.3, "mindihedral": 12.0}, (1.3, 12.0)),
])
def test_tetgen_receives_the_quality_parameters_verbatim(monkeypatch, kwargs, expected):
    seen = {}
    real = tetgen.TetGen.tetrahedralize

    def spy(self, *args, **kw):
        seen.update(minratio=kw["minratio"], mindihedral=kw["mindihedral"])
        return real(self, *args, **kw)

    monkeypatch.setattr(tetgen.TetGen, "tetrahedralize", spy)
    fill_core_volume(_PTS, _TRIS, verbose=False, **kwargs)
    assert (seen["minratio"], seen["mindihedral"]) == expected


def test_constants_are_the_values_production_always_ran_with():
    assert (CORE_TETGEN_MINRATIO, CORE_TETGEN_MINDIHEDRAL) == (1.1, 5.0)
