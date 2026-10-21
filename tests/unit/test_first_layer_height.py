"""边界层第一层高度（I-137）：显式给定或由目标 y+ 推出，原样使用；不再有 0.1 mm 硬下限。

修复前第一层恒为 min(min_cell_size, 0.2% 域尺寸)，低于 0.1 mm 时被静默抬到 0.1 mm（只打 warning）——SA-neg/SST
的低雷诺数壁面处理要求 y+ ~ 1，车辆尺度约 10 um。
"""

import math

import numpy as np
import pytest

from autoflowcfd.grid.mesh_gen.extrusion.first_layer import first_layer_height_for_yplus


def test_flat_plate_correlation():
    """手算：U=40 m/s、L=4 m、空气 -> Re=1.089e7，Cf=0.026 Re^(-1/7)，u_tau=sqrt(Cf/2) U，y1=y+ nu/u_tau。"""
    rho, mu = 1.225, 1.8e-5
    re = rho * 40.0 * 4.0 / mu
    u_tau = math.sqrt(0.5 * 0.026 * re ** (-1.0 / 7.0)) * 40.0
    assert first_layer_height_for_yplus(1.0, 40.0, 4.0, rho, mu) == pytest.approx(mu / (rho * u_tau), rel=1e-14)
    assert first_layer_height_for_yplus(1.0, 40.0, 4.0, rho, mu) == pytest.approx(1.025e-5, rel=1e-3)
    assert first_layer_height_for_yplus(30.0, 40.0, 4.0, rho, mu) == pytest.approx(
        30.0 * first_layer_height_for_yplus(1.0, 40.0, 4.0, rho, mu), rel=1e-14)
    with pytest.raises(ValueError):
        first_layer_height_for_yplus(1.0, 0.0, 4.0, rho, mu)


@pytest.mark.parametrize("height", [1.0e-5, 3.0e-3])
def test_extrusion_uses_the_given_first_layer_height(height):
    """平面片沿法向挤出：第一层节点的偏移恰为给定高度（1e-5 m 低于此前的 0.1 mm 硬下限）。"""
    from autoflowcfd.grid.mesh_gen.extrusion.mesh_extrusion import extrude_layers
    from autoflowcfd.grid.mesh_gen.utils.mesh_utils import compute_face_normals

    nodes = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    bbox = {'min': np.array([-10., -10., -10.]), 'max': np.array([10., 10., 10.])}
    all_nodes, layers = extrude_layers(nodes, faces, compute_face_normals(nodes, faces), bbox, growth_rate=1.2,
                                       min_cell_size=0.005, bl_layers=3, first_layer_height=height)
    first = all_nodes[len(nodes):2 * len(nodes)]
    np.testing.assert_allclose(np.abs(first[:, 2]), height, rtol=1e-12)
    np.testing.assert_allclose(first[:, :2], nodes[:, :2], atol=1e-15)


def test_cli_resolution():
    import click

    from autoflowcfd.cli.grid.volume_commands import _resolve_first_layer_height as r

    assert r(None, None, None, None, 1.225, 1.8e-5) is None
    assert r(2e-5, None, None, None, 1.225, 1.8e-5) == 2e-5
    assert r(None, 1.0, 40.0, 4.0, 1.225, 1.8e-5) == pytest.approx(first_layer_height_for_yplus(1.0, 40.0, 4.0,
                                                                                                1.225, 1.8e-5))
    for args in ((2e-5, 1.0, 40.0, 4.0), (None, 1.0, None, 4.0), (None, None, 40.0, None)):
        with pytest.raises(click.UsageError):
            r(*args, 1.225, 1.8e-5)
