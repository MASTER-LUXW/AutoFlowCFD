"""参考面积（frontal reference area）计算。

从 aero_coeffs.py 拆出来的 mixin：Cd/Cl 的无量纲化需要一个参考面积，这
部分逻辑（优先用面网格算迎风投影面积，体网格边界面兜底）和力/系数的
实际积分（AeroCoefficientCalculator.compute_coefficients）在概念上是
独立的一步，拆成单独文件只是为了控制单文件行数，不改变任何计算逻辑。
"""

import numpy as np
from loguru import logger


class ReferenceAreaMixin:
    """提供 `_compute_reference_area` 及其两种实现给 `AeroCoefficientCalculator`。

    依赖宿主类（`AeroCoefficientCalculator`）已有的
    `self.grid_data`/`self.face_extractor`/`self._cached_ref_area`/
    `self._ref_area_computed` 属性，不独立维护状态。
    """

    def _compute_reference_area(self, body_face_indices: np.ndarray) -> float:
        """用包围盒估计计算参考迎风面积。

        用车身边界的包围盒尺寸估计迎风投影面积，避免体网格边界层
        （附面层网格）带来的问题。

        Args:
            body_face_indices: 车身表面面的索引（本实现未直接使用）

        Returns:
            参考面积，单位 m^2
        """
        # 若已经算过，用缓存值
        if self._ref_area_computed and self._cached_ref_area is not None:
            return self._cached_ref_area

        try:
            # 从包围盒计算参考面积
            ref_area = self._compute_ref_area_from_surface_mesh()

            if ref_area > 0 and np.isfinite(ref_area):
                self._cached_ref_area = ref_area
                self._ref_area_computed = True
                return ref_area

            # 包围盒估计失败时回退到旧方法
            logger.warning("Bounding box estimation failed, using volume mesh fallback")
            ref_area = self._compute_ref_area_from_volume_mesh(body_face_indices)

            self._cached_ref_area = ref_area
            self._ref_area_computed = True

            return ref_area

        except Exception as e:
            logger.error(f"Failed to compute reference area: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return 1.0

    def _compute_ref_area_from_surface_mesh(self) -> float:
        """从原始面网格几何计算参考面积。

        直接使用 NAS 文件里的面网格三角形，避免体网格边界层延伸或边界
        污染带来的问题。

        Returns:
            参考面积，单位 m^2；计算失败则返回 0.0
        """
        try:
            # 检查面网格是否可用
            if not hasattr(self.grid_data, 'surface_mesh') or self.grid_data.surface_mesh is None:
                logger.warning("Surface mesh not available in grid_data, using fallback method")
                return 0.0

            surface_mesh = self.grid_data.surface_mesh
            surface_nodes = surface_mesh.get('nodes')  # shape=(n_nodes, 3)
            surface_faces = surface_mesh.get('faces')  # shape=(n_faces, 3)
            surface_boundaries = surface_mesh.get('boundaries')  # BoundaryMap

            if surface_nodes is None or surface_faces is None:
                logger.warning("Surface mesh nodes or faces not available")
                return 0.0

            # 从面网格取车身边界面索引
            if surface_boundaries is None:
                logger.warning("Surface mesh boundaries not available")
                return 0.0

            body_boundary_names = [
                name for name in surface_boundaries.boundary_names
                if 'BODY' in name.upper() or 'CAR' in name.upper()
            ]

            if not body_boundary_names:
                logger.warning("No body boundary found in surface mesh")
                return 0.0

            # 收集面网格里所有车身面索引
            body_face_indices = []
            for boundary_name in body_boundary_names:
                face_indices = surface_boundaries.get_cell_indices(boundary_name)
                body_face_indices.extend(face_indices)

            if len(body_face_indices) == 0:
                logger.warning("No body faces found in surface mesh")
                return 0.0

            body_face_indices = np.array(body_face_indices, dtype=np.int64)

            logger.info(f"Surface mesh body analysis:")
            logger.info(f"  Body faces: {len(body_face_indices)}")

            # 取车身面的节点坐标
            v0 = surface_nodes[surface_faces[body_face_indices, 0]]
            v1 = surface_nodes[surface_faces[body_face_indices, 1]]
            v2 = surface_nodes[surface_faces[body_face_indices, 2]]

            # 计算面法向量和面积
            e1 = v1 - v0
            e2 = v2 - v0
            normals = np.cross(e1, e2)
            areas = 0.5 * np.linalg.norm(normals, axis=1)

            # 把法向量归一化为单位向量
            norms = np.linalg.norm(normals, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)  # 避免除零
            unit_normals = normals / norms

            # 调试输出
            logger.info(f"  Total area: {areas.sum():.6f} m^2")
            logger.info(f"  Mean face area: {areas.mean():.6e} m^2")
            logger.info(f"  Min/Max face area: {areas.min():.6e} / {areas.max():.6e} m^2")

            # 计算 X 方向（自由来流方向）的投影面积
            x_component = unit_normals[:, 0]

            # 只统计迎风面（法向朝向来流的反方向，n_x < 0）
            upstream_mask = x_component < 0
            projected_areas = -x_component[upstream_mask] * areas[upstream_mask]
            ref_area = np.sum(projected_areas)

            # 校验
            if ref_area <= 0 or not np.isfinite(ref_area):
                logger.warning(f"Invalid reference area from surface mesh: {ref_area:.6e}")
                # 兜底：用绝对投影除以 2（适用于对称车身）
                projected_areas_all = np.abs(x_component) * areas
                ref_area_fallback = np.sum(projected_areas_all) / 2.0
                if ref_area_fallback > 0 and np.isfinite(ref_area_fallback):
                    logger.info(f"Fallback reference area (|n_x|/2): {ref_area_fallback:.6f} m^2")
                    return float(ref_area_fallback)
                return 0.0

            # Ahmed Body 的合理性检查
            if ref_area < 0.01 or ref_area > 1.0:
                logger.warning(f"Reference area {ref_area:.4f} m^2 outside expected range (0.1-0.3 m^2)")
                logger.warning(f"  Upstream-facing faces: {np.sum(upstream_mask)} / {len(body_face_indices)}")
                logger.warning(f"  Mean |n_x|: {np.mean(np.abs(x_component)):.4f}")

            logger.info(f"Reference area (from surface mesh): {ref_area:.6f} m^2")
            logger.info(f"  Upstream-facing ratio: {np.sum(upstream_mask) / len(body_face_indices) * 100:.1f}%")
            logger.info(f"  Mean projected area per upstream face: {ref_area / max(1, np.sum(upstream_mask)):.6e} m^2")

            return float(ref_area)

        except Exception as e:
            logger.error(f"Failed to compute reference area from surface mesh: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return 0.0

    def _compute_ref_area_from_volume_mesh(self, body_face_indices: np.ndarray) -> float:
        """兜底方案：从体网格边界面计算参考面积。

        当面网格不可用时使用此方法。

        Args:
            body_face_indices: 体网格中车身表面面的索引

        Returns:
            参考面积，单位 m^2
        """
        if len(body_face_indices) == 0:
            logger.warning("No body faces identified for reference area calculation")
            return 1.0

        face_normals = self.face_extractor.face_normals[body_face_indices]
        face_areas = self.face_extractor.face_areas[body_face_indices]

        # X 方向（自由来流方向）的投影面积
        x_component = face_normals[:, 0]

        # 只统计迎风面（更准确）
        upstream_mask = x_component < 0
        projected_areas = -x_component[upstream_mask] * face_areas[upstream_mask]
        ref_area = np.sum(projected_areas)

        # 校验
        if ref_area <= 0 or not np.isfinite(ref_area):
            logger.warning(f"Invalid reference area: {ref_area:.6e}, using fallback")
            projected_areas_all = np.abs(x_component) * face_areas
            ref_area_fallback = np.sum(projected_areas_all) / 2.0
            if ref_area_fallback > 0 and np.isfinite(ref_area_fallback):
                return float(ref_area_fallback)
            return 1.0

        return float(ref_area)
