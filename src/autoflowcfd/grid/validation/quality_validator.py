"""面向 CFD 网格的质量校验工具。

为四面体和三角形网格提供全面的质量指标，包括体积检查、长宽比分析、
扭曲度评估和正交性评估。

关键指标：
    - 体积质量（负体积）
    - 相邻单元体积比（关系到 Green-Gauss 梯度重构的条件数）
    - 长宽比（单元形状质量，BL 区域和 core 区域用不同阈值）
    - 扭曲度（基于半径比的形状度量）
    - 正交性（面法向与单元质心连线的夹角）

指标的选取是针对本项目具体求解器校准的，不是通用默认值——推导依据：
    - 梯度重构用的是 FR 微分算子（core/fr_operators.py），
      grad ~ D_ij * q_j——通过预计算的微分矩阵进行高阶求导。
      体积比相邻单元小几个数量级的单元，梯度会被同样倍数放大，
      与局部伪时间步长无关（局部时间步长保护的是该单元自身的*稳定性*，
      不是它交给相邻单元的量的*精度*）。这就是为什么这里检查相邻单元
      体积比和非正交性（两者都直接影响离散格式的数值条件数），而不只是
      一个全局最大/最小体积比——BL 网格从近壁到远场的全局体积范围本来
      就会跨越好几个数量级，这本身不是缺陷。
    - 网格只有四面体（没有六面体/棱柱），近壁是 BL 挤出棱柱拆分成的
      四面体，其余是 tetgen 核心填充——长宽比对两个区域分别检查，因为
      BL 单元预期比 core 单元拉伸得多。

参考文献：
    - Knupp, P. "Advances in grid 质量度量", 2000
    - Liao, D.A. "Qualitative measures for initial 网格生成", 1988
    - Verdict Geometric Quality Library (Sandia) - TetRadiusRatio metric
"""

import numpy as np
from typing import Dict, Optional, TYPE_CHECKING
from loguru import logger

from .quality_report import MeshQualityReport
from . import quality_metrics as _qm
from .quality_evaluation import evaluate_quality, generate_recommendations

if TYPE_CHECKING:
    from ..structures import FaceData, VolumeMeshData


class MeshQualityValidator:
    """验证 CFD 仿真的网格质量。

    计算各种质量指标以确保网格适合精确且稳定的 CFD 仿真。

    Attributes:
        thresholds: Quality metric thresholds for pass/fail criteria
    """

    def __init__(self):
        """初始化 validator，使用默认质量阈值。"""
        self.thresholds = {
            'max_negative_volumes': 0,       # No negative volumes allowed
            'max_volume_ratio': 1e6,         # 全局范围 - 仅供参考，见 MeshQualityReport 文档字符串
            'max_aspect_ratio': 100.0,       # fallback when no BL/core split is available
            'bl_max_aspect_ratio': 50.0,     # BL cells: expected to be stretched
            'core_max_aspect_ratio': 10.0,   # core-fill cells: should be close to isotropic
            'max_skewness': 0.95,            # radius-ratio based (Fluent-equivalent severity)
            'max_orthogonality_angle': 70.0, # 角度；与 OpenFOAM 对齐（Green-Gauss 对非正交性
                                              # 比表面法向修正格式更敏感，因此这里刻意比
                                              # Fluent 宽松的 orthogonal-quality 下限更严格）
            'max_adjacent_volume_ratio': 5.0,  # 与 STAR-CCM+ 的 "Volume Change" 指导对齐；
                                                # 这个才是真正控制 Green-Gauss 的 1/V
                                                # 梯度放大条件数的因素
            'max_overlapping_cells': 0,        # 任何物理重叠的单元对都不合格——
                                                # 见 mesh_overlap_check.py；“接近但未重叠”
                                                # 仅供参考，不作为关卡
        }

        logger.info("MeshQualityValidator initialized with default thresholds")

    def validate(
        self,
        nodes: np.ndarray,
        cells: np.ndarray,
        cell_type: str = "tetrahedron",
        faces: Optional['FaceData'] = None,
        bl_cell_mask: Optional[np.ndarray] = None,
        log_summary: bool = True,
        check_overlap: bool = True,
    ) -> MeshQualityReport:
        """执行全面的网格质量验证。

        Args:
            nodes: 节点坐标, shape=(n_nodes, 3)
            cells: 单元连接关系, shape=(n_cells, n_vertices)
            cell_type: 单元类型 ('tetrahedron' 或 'triangle')
            faces: 可选的预计算 FaceData（owner/neighbour
                连接关系 + 法向量）。正交性、相邻体积比和
                重叠/接近度检查需要面连接关系；如果未提供，
                则通过 FaceExtractor.extract_faces 内部推导
                （对大规模网格有实际但非平凡的成本——已经
                拥有它的调用者，例如网格生成/修复管线，
                应该传递它以避免冗余工作）。
                对 cell_type='triangle' 忽略。
            bl_cell_mask: 可选布尔数组, shape=(n_cells,), True
                表示 BL 区域单元——启用分离的 BL 区域/核心区域
                长宽比分析。未设置时回退到单一的全网格长宽比
                检查（之前的行为）。
            log_summary: 通过 logger.info 记录完整格式化报告。
                对于将自行打印更完整版本（例如附带前后对比）
                的调用者设为 False——避免同一报告文本连续出现两次。
            check_overlap: 运行单元重叠/接近度检查（见
                mesh_overlap_check.py）。与此处所有其他检查不同，
                其成本随局部网格密度缩放（宽相位空间搜索 +
                幸存者精确几何测试），而非纯粹按单元数——
                对需要最快周转且愿意接受重叠在下一次完整
                validate() 调用前不被检测到的调用者提供退出选项。

        Returns:
            MeshQualityReport 包含所有质量指标
        """
        logger.info(f"Validating mesh quality: {len(cells)} {cell_type}s...")

        report = MeshQualityReport(
            n_cells=len(cells),
            n_nodes=len(nodes)
        )

        # 计算所有质量指标
        self._check_volumes(report, nodes, cells, cell_type)
        self._check_aspect_ratios(report, nodes, cells, cell_type, bl_cell_mask)
        self._check_skewness(report, nodes, cells, cell_type)
        if cell_type == "tetrahedron":
            # 最多提取一次并共享给下面两个检查——
            # _check_orthogonality_and_adjacency 和
            # _check_overlap_and_proximity 否则会各自独立调用
            # self._extract_faces（当调用者未预提供 `faces` 时），
            # 无意义地为整个网格连续做两次完整面提取
            # （已确认：在真实 150 万单元的网格上，一次 validate()
            # 调用中连续出现两次 "Extracting faces from N
            # tetrahedral cells..." 日志，因为两个子检查各自
            # 内部提取的 FaceData 没有缓存回此处供另一个复用）。
            if faces is None:
                faces = self._extract_faces(nodes, cells)
            self._check_orthogonality_and_adjacency(report, nodes, cells, faces)
            if check_overlap:
                self._check_overlap_and_proximity(report, nodes, cells, faces)

        # 评估通过/失败标准
        evaluate_quality(report, self.thresholds)

        # 生成建议
        generate_recommendations(report, self.thresholds)

        # 日志摘要
        if log_summary:
            logger.info(f"\n{report.summary()}")

        return report

    def validate_volume_mesh(
        self,
        volume_mesh: 'VolumeMeshData',
        faces: Optional['FaceData'] = None,
        bl_cell_mask: Optional[np.ndarray] = None,
        check_overlap: bool = True,
    ) -> MeshQualityReport:
        """验证 VolumeMeshData 对象（便捷方法）。

        Args:
            volume_mesh: 包含四面体单元的 VolumeMeshData（以及可选的
                prism_cells——存在时转调到 validate_mixed()，见该方法）
            faces: 可选的预计算 FaceData——如果未提供且
                volume_mesh.faces 已填充（已调用 ensure_faces_exist），
                则复用它而不是重新提取。
            bl_cell_mask: 可选的 BL/核心区域拆分，见 validate()。
                当 volume_mesh.prism_cells 已设置时忽略——
                validate_mixed 自行推导（棱柱即 BL 区域，四面体即核心，
                按本项目的全局单元索引约定）。
            check_overlap: 见 validate()

        Returns:
            MeshQualityReport 包含所有质量指标
        """
        if faces is None:
            faces = volume_mesh.faces

        if volume_mesh.prism_cells is not None:
            return self.validate_mixed(volume_mesh, faces=faces, check_overlap=check_overlap)

        return self.validate(
            nodes=np.column_stack([
                volume_mesh.nodes.x,
                volume_mesh.nodes.y,
                volume_mesh.nodes.z
            ]),
            cells=volume_mesh.cells.connectivity,
            cell_type="tetrahedron",
            faces=faces,
            bl_cell_mask=bl_cell_mask,
            check_overlap=check_overlap,
        )

    def validate_mixed(
        self,
        volume_mesh: 'VolumeMeshData',
        faces: Optional['FaceData'] = None,
        log_summary: bool = True,
        check_overlap: bool = True,
    ) -> MeshQualityReport:
        """校验混合棱柱(BL) + 四面体(core) VolumeMeshData。

        结构与 validate() 相同，但每个 per-cell 指标按区域分别计算
        （棱柱单元用 quality_metrics 的棱柱函数，四面体单元用已有的
        四面体函数——两种形状需要完全不同的公式，见 quality_metrics.py），
        然后按所有其他棱柱感知代码使用的全局单元索引顺序拼接
        （棱柱 [0, n_prism)，四面体 [n_prism, n_prism+n_tet)——
        见 PrismCells / face_extractor.extract_faces_mixed）。
        正交性和相邻体积比（基于面，因此天然跨越 BL/core 界面）
        使用全局面图的一次合并遍历，通过 compute_face_diagnostics 的
        cell_centroids/cell_volumes 参数（专门为此添加，避免从混合网格
        没有的单一均匀连接关系数组重新推导 per-cell 质心/体积）。

        bl_cell_mask 不是这里的参数（与 validate() 不同）——按构造它
        恰好是 [True]*n_prism + [False]*n_tet，调用者无法有意义地覆盖。

        实现位于 quality_validator_mixed.py（按项目 >400 行文件拆分规则
        提取）——此处延迟导入以避免模块加载时的循环导入（该模块的类型
        提示引用了本文件的 MeshQualityValidator）。
        """
        from .quality_validator_mixed import validate_mixed_mesh

        return validate_mixed_mesh(
            self, volume_mesh, faces=faces, log_summary=log_summary, check_overlap=check_overlap
        )

    @staticmethod
    def _extract_faces(nodes: np.ndarray, cells: np.ndarray) -> 'FaceData':
        """当调用者尚未拥有面时推导面连接关系。
        延迟导入（mesh_gen -> validation 在本包中是单向依赖；
        此处仅在调用时反向导入，避免考虑导入顺序）。"""
        from ..mesh_gen.extraction.face_extractor import FaceExtractor
        from ..schema.grid_nodes import NodeArray

        node_arr = NodeArray.from_array(nodes)
        return FaceExtractor.extract_faces(cells.astype(np.int32), node_arr)

    def _check_volumes(
        self,
        report: MeshQualityReport,
        nodes: np.ndarray,
        cells: np.ndarray,
        cell_type: str
    ) -> None:
        """检查单元体积的有效性（向量化）。

        实现位于 quality_validator_metrics.py（按项目 >400 行文件拆分规则
        提取；函数体不使用 `self`，因此作为普通函数搬移）。
        """
        from .quality_validator_metrics import check_volumes

        check_volumes(report, nodes, cells, cell_type)

    def _compute_tetrahedron_volumes(self, nodes: np.ndarray, cells: np.ndarray) -> np.ndarray:
        """quality_metrics.compute_tetrahedron_volumes 的薄实例方法包装
        ——保留给外部调用者（例如 mesh_gen/mesh_repair.py 的 Stage A），
        它们直接访问此 validator 实例而不是自行导入度量函数。"""
        return _qm.compute_tetrahedron_volumes(nodes, cells)

    def _check_aspect_ratios(
        self,
        report: MeshQualityReport,
        nodes: np.ndarray,
        cells: np.ndarray,
        cell_type: str,
        bl_cell_mask: Optional[np.ndarray] = None,
    ) -> None:
        """检查单元长宽比（向量化），可选按 BL 区域与核心区域拆分
        （见 MeshQualityReport 文档字符串了解为何需要分离阈值）。

        实现位于 quality_validator_metrics.py（按项目 >400 行文件拆分规则
        提取；函数体不使用 `self`，因此作为普通函数搬移）。
        """
        from .quality_validator_metrics import check_aspect_ratios

        check_aspect_ratios(report, nodes, cells, cell_type, bl_cell_mask=bl_cell_mask)

    def _check_skewness(
        self,
        report: MeshQualityReport,
        nodes: np.ndarray,
        cells: np.ndarray,
        cell_type: str
    ) -> None:
        """检查单元偏斜度（向量化）。

        实现位于 quality_validator_metrics.py（按项目 >400 行文件拆分规则
        提取；函数体不使用 `self`，因此作为普通函数搬移）。
        """
        from .quality_validator_metrics import check_skewness

        check_skewness(report, nodes, cells, cell_type)

    def compute_cell_skewness(self, nodes: np.ndarray, cells: np.ndarray) -> np.ndarray:
        """公共 per-cell 半径比偏斜度数组, shape=(n_cells,)——
        max_skewness/mean_skewness 背后的原始值，供需要知道
        *哪些*单元有问题（而非仅聚合统计）的调用者使用
        （例如 mesh_gen/mesh_repair.py 中的网格修复循环）。"""
        from .quality_validator_metrics import compute_cell_skewness as _compute_cell_skewness

        return _compute_cell_skewness(nodes, cells)

    def compute_face_diagnostics(
        self,
        nodes: np.ndarray,
        cells: np.ndarray,
        faces: Optional['FaceData'] = None,
        cell_centroids: Optional[np.ndarray] = None,
        cell_volumes: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """公共 per-内部面诊断——orthogonality_max/adjacent_volume_ratio_max
        背后的原始数组，供需要知道哪些面/单元涉及（而非仅聚合值）的调用者使用。

        实现位于 quality_validator_metrics.py（按项目 >400 行文件拆分规则
        提取）——见该模块的 compute_face_diagnostics 了解完整的 Args/Returns
        文档。
        """
        from .quality_validator_metrics import compute_face_diagnostics as _compute_face_diagnostics

        return _compute_face_diagnostics(
            self, nodes, cells, faces=faces, cell_centroids=cell_centroids, cell_volumes=cell_volumes
        )

    def _check_orthogonality_and_adjacency(
        self,
        report: MeshQualityReport,
        nodes: np.ndarray,
        cells: np.ndarray,
        faces: Optional['FaceData'],
        cell_centroids: Optional[np.ndarray] = None,
        cell_volumes: Optional[np.ndarray] = None,
    ) -> None:
        """检查面非正交性和相邻单元（面邻居）体积比——这两个指标
        直接控制本项目求解器的 Green-Gauss 梯度条件数（见模块文档字符串）。
        两者都需要面 owner/neighbour 连接关系，因此共享单次面提取
        （compute_face_diagnostics）。

        cell_centroids/cell_volumes：见 compute_face_diagnostics——
        对混合棱柱+四面体网格传入，因为仅靠 `cells` 无法描述
        每个单元的形状。
        """
        diag = self.compute_face_diagnostics(
            nodes, cells, faces, cell_centroids=cell_centroids, cell_volumes=cell_volumes
        )
        if len(diag['angle_deg']) == 0:
            return

        report.orthogonality_max = float(np.max(diag['angle_deg']))
        report.orthogonality_mean = float(np.mean(diag['angle_deg']))
        report.adjacent_volume_ratio_max = float(np.max(diag['volume_ratio']))
        report.adjacent_volume_ratio_mean = float(np.mean(diag['volume_ratio']))

    def _check_overlap_and_proximity(
        self,
        report: MeshQualityReport,
        nodes: np.ndarray,
        cells: np.ndarray,
        faces: Optional['FaceData'],
    ) -> None:
        """检测面物理重叠不同且不相邻单元的面，或距离足够近
        以至于一个参数变化就可能与之重叠的单元——见
        mesh_overlap_check.py 了解精确几何测试以及为何这是与
        负/退化体积不同的缺陷类别。
        """
        from .mesh_overlap_check import check_face_overlap_and_proximity

        overlap_report = check_face_overlap_and_proximity(nodes, cells, faces=faces)
        report.n_overlapping_cells = len(overlap_report.overlapping_cell_ids)
        report.n_close_cell_pairs = overlap_report.n_close_pairs
        report.overlap_min_gap = overlap_report.min_gap_found
        report.overlapping_cell_ids = overlap_report.overlapping_cell_ids

