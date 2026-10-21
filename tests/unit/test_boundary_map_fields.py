"""BoundaryMap 不再携带已删除的边界 YAML 配置体系留下的 detection_mode / config_source（I-198）；
带着这两个属性写出的旧 .pkl 仍能读回并参与求解所需的全部查询。"""

import dataclasses
import pickle

import numpy as np

from autoflowcfd.grid.schema.grid_boundaries import BoundaryMap


def _bmap():
    return BoundaryMap(groups={"wall": np.array([0, 1], dtype=np.int32)}, bc_types={"wall": "WALL"},
                       property_ids={"wall": 3}, property_names={3: "WALL"})


def test_legacy_configuration_fields_are_gone():
    names = {f.name for f in dataclasses.fields(BoundaryMap)}
    assert not names & {"detection_mode", "config_source"}
    assert not hasattr(BoundaryMap, "get_summary")


def test_old_pickle_with_legacy_attributes_still_loads():
    old = _bmap()
    old.__dict__.update(detection_mode="auto", config_source=None)      # 旧版本写出的对象状态
    loaded = pickle.loads(pickle.dumps(old))
    assert loaded.get_boundary_type("wall") == "WALL"
    assert loaded.get_property_id("wall") == 3
    np.testing.assert_array_equal(loaded.get_cell_indices("wall"), [0, 1])
