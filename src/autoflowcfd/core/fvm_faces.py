"""FVM 面数据的轻量持有者（data holder）。

FVMFaceExtractor 本身不再做面提取——真正的面提取统一走
grid.mesh_gen.face_extractor.FaceExtractor（经 VolumeMeshData.
ensure_faces_exist()），本类只是求解器内部各模块（bc_handler、
aero_calculator 等）共享读取的一个属性容器：由调用方把已经算好的
connectivity/normals/areas/centers/boundary_flags/cell_centroids
直接赋值进来。

朝向约定：每个面法向都保证从 owner 单元（connectivity 第 0 列）指向
neighbour 单元（第 1 列，内部面），边界面则指向域外——这个朝向约定是
残差累加方式（owner 减、neighbour 加）具有离散守恒性、气动力积分符号
正确的前提。
"""

import numpy as np
from typing import Dict


class FVMFaceExtractor:
    """FVM 求解器共用的面几何属性容器。"""

    def __init__(self):
        self.face_connectivity = None
        self.face_areas = None
        self.face_centers = None
        self.face_normals = None
        self.boundary_flags = None
        # 单元中心点既用于法向定向，也用于梯度重构（Green-Gauss / 最小二乘）。
        self.cell_centroids = None

    def get_face_data(self) -> Dict[str, np.ndarray]:
        """获取全部面数据。"""
        return {
            'connectivity': self.face_connectivity,
            'areas': self.face_areas,
            'centers': self.face_centers,
            'normals': self.face_normals,
            'boundary_flags': self.boundary_flags,
            'cell_centroids': self.cell_centroids,
        }
