"""展平面几何要求网格带面表（I-104）：存在性检查在 `get_flat_face_geometry` 里，分布式路径传入
`flat_face_override`、不经过它；`PrecompactedMeshData` 不再挂字符串哨兵冒充 `face_flux_points`。"""

from types import SimpleNamespace

import pytest

from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry


@pytest.mark.parametrize("fc, ffp", [(None, None), (object(), None), (None, object())])
def test_mesh_without_face_table_is_rejected(fc, ffp):
    with pytest.raises(RuntimeError, match="面表"):
        get_flat_face_geometry(SimpleNamespace(face_connectivity=fc, face_flux_points=ffp), None)


def test_non_kernel_face_data_is_rejected():
    with pytest.raises(TypeError, match="_KernelFaceData"):
        get_flat_face_geometry(SimpleNamespace(face_connectivity=object(), face_flux_points="sentinel"), None)
