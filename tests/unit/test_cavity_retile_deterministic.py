"""局部空腔重铺的截止条件是确定性的（I-140）：结果与机器速度、负载无关。

此前 `remesh_core_cavity` 另有 400 s 挂钟上限，超时剩余坏单元保持原样——同一输入在慢机器或高负载下得到不同网格。
这里把时钟改成每次读取跳 1000 s（相当于极慢的机器），结果必须与正常运行逐位相同。
"""

import itertools
import time

import numpy as np

from autoflowcfd.grid.mesh_gen.extraction.face_extractor import FaceExtractor
from autoflowcfd.grid.mesh_gen.repair.mesh_repair_cavity import remesh_core_cavity
from autoflowcfd.grid.mesh_gen.tetgen.mesh_prism_to_tet import orient_tetrahedra
from autoflowcfd.grid.schema.grid_nodes import NodeArray
from autoflowcfd.grid.validation.quality_validator import MeshQualityValidator
from tests.unit.test_cavity_weld_conformity import _cube_tets


def _distorted_cube():
    nodes, tets, _ = _cube_tets(4)
    nodes = nodes.copy()
    k = int(np.argmin(np.linalg.norm(nodes - 0.5, axis=1)))
    nodes[k] += [0.11, 0.1, 0.09]                               # 中心节点挪位，周围单元变差
    tets = orient_tetrahedra(nodes, tets.astype(np.int64)).astype(np.int32)
    return nodes, tets, np.isin(tets, k).any(axis=1)


def _run():
    nodes, tets, bad = _distorted_cube()
    faces = FaceExtractor.extract_faces(tets, NodeArray.from_array(nodes))
    return remesh_core_cavity(nodes, tets, faces, bad, MeshQualityValidator())


def test_slow_clock_gives_the_same_mesh(monkeypatch):
    reference = _run()
    assert len(reference[1]) != len(_distorted_cube()[1])      # 确有空腔被重铺
    ticks = itertools.count()
    monkeypatch.setattr(time, "perf_counter", lambda: 1000.0 * next(ticks))
    slow = _run()
    assert np.array_equal(slow[0], reference[0]) and np.array_equal(slow[1], reference[1])
    assert np.array_equal(slow[2], reference[2])
