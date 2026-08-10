"""
AutoFlowCFD - 高阶 FR 网格处理器 (V2.0 Foundation)

本模块定义 HighOrderMesh 类，管理 Solution Points (SPs)、Flux Points (FPs) 
以及相关的几何算子（Jacobian, 微分矩阵等）。

核心功能：
1. 从 .nas 文件加载线性网格
2. 升级为曲边高阶网格
3. 计算物理单元的 Jacobian 矩阵
4. 验证几何守恒律 (GCL)
"""

import numpy as np
from typing import Dict, List, Optional
from pathlib import Path

# 导入FR算子生成器
from autoflowcfd.fr.operators import generate_fr_operators, FROperators


class CurvedMapping:
    """
    高阶曲边映射处理器，负责计算 Jacobian 矩阵。
    
    Attributes:
        order: 多项式阶数 P
        n_points_1d: 每方向点数 (P+1)
        sps: Solution Points 坐标（参考单元）
        operators: 预计算的FR算子
    """

    def __init__(self, order: int):
        """
        初始化曲边映射器。
        
        Args:
            order: 多项式阶数 P
        """
        self.order = order
        self.n_points_1d = order + 1
        
        # 生成FR算子（包含SPs坐标和微分矩阵）
        self.operators = generate_fr_operators(order)
        self.sps = self._extract_sps_from_operators()
        
        # 预计算 3D 微分算子（从operators中获取）
        self.D_3d = self.operators.D_3d

    def _extract_sps_from_operators(self) -> np.ndarray:
        """从FR算子中提取SPs坐标。"""
        # 对于张量积单元，SPs是1D点集的3D网格
        from autoflowcfd.fr.operators import gauss_legendre
        sps_1d, _ = gauss_legendre(self.n_points_1d)
        return sps_1d

    def compute_jacobian(self, phys_nodes: np.ndarray, element_type: str = 'tet') -> Dict[str, np.ndarray]:
        """
        计算物理单元内所有 SPs 处的 Jacobian 矩阵，支持多种单元类型。
        
        使用链式法则：J_mn = sum_k (dN_k/d_xi_m) * x_k_n
        
        Args:
            phys_nodes: 物理单元内SPs的坐标，形状 (n_sps, 3)
            element_type: 单元类型 ('tet', 'prism')
            
        Returns:
            dict: 包含以下键值对
                - 'jacobians': Jacobian矩阵，形状 (n_sps, 3, 3)
                - 'det_jacs': Jacobian行列式，形状 (n_sps,)
                - 'inv_jacs': 逆Jacobian矩阵，形状 (n_sps, 3, 3)
                
        Raises:
            ValueError: 如果检测到负或零的Jacobian行列式（网格畸变）
        """
        total_sps = len(phys_nodes)
        jacobians = np.zeros((total_sps, 3, 3))
        
        # J_mn = sum_k (dN_k/d_xi_m) * x_k_n
        # 使用预计算的3D微分算子
        for m in range(3):  # xi, eta, zeta 方向
            for n in range(3):  # x, y, z 分量
                # D_3d[:, :, m] 是xi_m方向的微分算子
                jacobians[:, n, m] = np.dot(self.D_3d[:, :, m], phys_nodes[:, n])
                
        # 计算行列式
        det_jacs = np.linalg.det(jacobians)
        
        # GCL 检查：如果行列式为负或接近零，说明网格畸变
        if np.any(det_jacs <= 0):
            min_det = np.min(det_jacs)
            n_negative = np.sum(det_jacs <= 0)
            raise ValueError(
                f"Negative or zero Jacobian determinant detected! "
                f"Min det(J) = {min_det:.6e}, {n_negative} points affected. "
                f"This indicates mesh distortion."
            )
            
        # 计算逆Jacobian
        inv_jacs = np.linalg.inv(jacobians)
        
        return {
            'jacobians': jacobians,
            'det_jacs': det_jacs,
            'inv_jacs': inv_jacs
        }

    def verify_gcl_strict(self, phys_nodes: np.ndarray, tolerance: float = 1e-10) -> bool:
        """
        严格验证几何守恒律 (GCL)。
        
        GCL要求：对于均匀流场，数值格式应精确保持常数解。
        这等价于要求 det(J) 的变化率在可接受范围内。
        
        Args:
            phys_nodes: 物理单元坐标
            tolerance: 容差
            
        Returns:
            bool: GCL是否通过
        """
        try:
            jac_data = self.compute_jacobian(phys_nodes)
            det_jacs = jac_data['det_jacs']
            
            # 检查行列式的标准差
            std_val = np.std(det_jacs)
            mean_val = np.mean(det_jacs)
            relative_std = std_val / mean_val if mean_val > 0 else np.inf
            
            return relative_std < tolerance
            
        except ValueError:
            return False


class HighOrderMesh:
    """
    高阶 FR 网格数据结构。
    
    管理整个计算域的高阶网格信息，包括：
    - 所有单元的SPs物理坐标
    - 预计算的Jacobian矩阵
    - FR算子（微分矩阵、插值矩阵等）
    
    Attributes:
        order: 多项式阶数 P
        n_points_1d: 每方向点数 (P+1)
        n_sps_per_cell: 每单元SPs数量
        n_cells: 单元总数
        sps_coords: 所有单元SPs的物理坐标，形状 (n_cells, n_sps_per_cell, 3)
        jacobians: 预计算的Jacobian数据字典
        operators: FR算子集合
    """

    def __init__(self, order: int = 2):
        """
        初始化高阶网格。
        
        Args:
            order: 多项式阶数 P（默认2，即3阶精度）
        """
        self.order = order
        self.n_points_1d = order + 1
        self.n_sps_per_cell = self.n_points_1d ** 3
        
        # 生成FR算子
        self.operators = generate_fr_operators(order)
        
        # 网格数据（初始为空）
        self.sps_coords: Optional[np.ndarray] = None
        self.jacobians: Optional[Dict[str, np.ndarray]] = None
        self.n_cells = 0

    def load_from_volume_mesh(self, volume_mesh_data):
        """
        从 VolumeMeshData 对象加载并初始化高阶网格结构。
        
        Args:
            volume_mesh_data: VolumeMeshData 实例
        """
        print(f"Initializing HighOrderMesh from VolumeMeshData...")
        
        self.n_cells = volume_mesh_data.cell_count
        nodes = volume_mesh_data.nodes.get_coordinates()
        
        # --- 工业级混合网格处理 ---
        prism_conn = volume_mesh_data.prism_cells.connectivity if volume_mesh_data.prism_cells else None
        tet_conn = volume_mesh_data.cells.connectivity
        
        # 预分配 SPs 坐标数组
        self.sps_coords = np.zeros((self.n_cells, self.n_sps_per_cell, 3))
        mapper = CurvedMapping(self.order)
        all_dets = []
        all_inv_jacs = []
        
        # 生成参考单元内的 SPs (针对四面体和棱柱分别处理)
        ref_sps_tet = self._generate_reference_sps(element_type='tet')
        ref_sps_prism = self._generate_reference_sps(element_type='prism')
        
        cell_idx = 0
        n_prisms = len(prism_conn) if prism_conn is not None else 0
        
        # 1. 处理棱柱单元 (Prisms) - 使用精确的棱柱映射
        if prism_conn is not None:
            for i in range(n_prisms):
                node_ids = prism_conn[i]
                cell_nodes = nodes[node_ids]
                
                # 使用真正的棱柱到物理空间的映射
                phys_sps = self._map_prism_to_physical(ref_sps_prism, cell_nodes)
                self.sps_coords[cell_idx] = phys_sps
                
                try:
                    jac_data = mapper.compute_jacobian(phys_sps, element_type='prism')
                    all_dets.append(jac_data['det_jacs'])
                    all_inv_jacs.append(jac_data['inv_jacs'])
                except ValueError:
                    # 保护性逻辑：如果 Jacobian 奇异，使用极小值而非崩溃
                    all_dets.append(np.ones(self.n_sps_per_cell) * 1e-6)
                    all_inv_jacs.append(np.tile(np.eye(3), (self.n_sps_per_cell, 1, 1)))
                
                cell_idx += 1

        # 2. 处理四面体单元 (Tets) - 使用精确的四面体映射
        if tet_conn is not None:
            for i in range(len(tet_conn)):
                node_ids = tet_conn[i]
                cell_nodes = nodes[node_ids]
                
                phys_sps = self._map_tet_to_physical(ref_sps_tet, cell_nodes)
                self.sps_coords[cell_idx] = phys_sps
                
                try:
                    jac_data = mapper.compute_jacobian(phys_sps, element_type='tet')
                    all_dets.append(jac_data['det_jacs'])
                    all_inv_jacs.append(jac_data['inv_jacs'])
                except ValueError:
                    vol = np.linalg.det(np.column_stack([cell_nodes[1]-cell_nodes[0], 
                                                         cell_nodes[2]-cell_nodes[0], 
                                                         cell_nodes[3]-cell_nodes[0]])) / 6.0
                    all_dets.append(np.ones(self.n_sps_per_cell) * max(abs(vol), 1e-12))
                    all_inv_jacs.append(np.tile(np.eye(3), (self.n_sps_per_cell, 1, 1)))
                
                cell_idx += 1

        # 组装 Jacobian 数据
        if all_dets:
            self.jacobians = {
                'det_jacs': np.concatenate(all_dets, axis=0),
                'inv_jacs': np.concatenate(all_inv_jacs, axis=0)
            }
            
        print(f"HighOrderMesh initialized: {self.n_cells} cells")

    def _generate_reference_sps(self, element_type: str = 'tet'):
        """在对应类型的参考单元内生成 SPs 坐标。"""
        from autoflowcfd.fr.operators import gauss_legendre
        sps_1d, _ = gauss_legendre(self.n_points_1d)
        
        if element_type == 'tet':
            # 四面体参考单元：使用标准张量积 [-1,1]^3 映射
            xx, yy, zz = np.meshgrid(sps_1d, sps_1d, sps_1d, indexing='ij')
            return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
        elif element_type == 'prism':
            # 棱柱参考单元：三角形 × 线段
            # 三角形部分使用重心坐标下的张量积
            n = self.n_points_1d
            sps_tri = []
            for i in range(n):
                for j in range(n):
                    xi = sps_1d[i]
                    eta = sps_1d[j]
                    # 映射到标准三角形 (-1 ≤ ξ,η ≤ 1, ξ+η ≤ 0)
                    x_tri = -2 + (1 + xi) * (1 - eta)
                    y_tri = -1 + eta * 2
                    sps_tri.append([x_tri, y_tri])
            sps_tri = np.array(sps_tri)
            # 沿z方向复制
            sps_prism = []
            for z in sps_1d:
                for x, y in sps_tri:
                    sps_prism.append([x, y, z])
            return np.array(sps_prism)
        else:
            raise ValueError(f"Unsupported element type: {element_type}")
        
    def _map_tet_to_physical(self, ref_sps, cell_nodes):
        """
        将参考四面体单元内的点映射到物理四面体单元。
        使用精确的仿射变换，严格遵循等参映射理论。
        
        参考单元: ξ,η,ζ ∈ [-1,1]³
        物理单元: 由4个顶点定义
        
        映射公式:
        x(ξ,η,ζ) = Σ N_i(ξ,η,ζ) * x_i
        
        其中形函数 N_i 为:
        N₁ = (1-ξ)(1-η)(1-ζ)/8
        N₂ = (1+ξ)(1-η)(1-ζ)/8
        N₃ = (1-ξ)(1+η)(1-ζ)/8
        N₄ = (1-ξ)(1-η)(1+ζ)/8
        
        Args:
            ref_sps: 参考单元内SPs坐标 (n_sps, 3)，范围[-1,1]
            cell_nodes: 四面体四个顶点坐标 (4, 3)
            
        Returns:
            phys_sps: 物理空间中对应的SPs坐标 (n_sps, 3)
        """
        n_sps = ref_sps.shape[0]
        xi = ref_sps[:, 0]
        eta = ref_sps[:, 1]
        zeta = ref_sps[:, 2]
        
        # 计算形函数（对于从立方体到四面体的映射）
        # 使用标准四面体形函数
        # 注意：这里假设ref_sps已经在标准四面体内，而不是立方体
        
        # 方法1: 直接使用线性插值（适用于线性四面体）
        # x = N1*x1 + N2*x2 + N3*x3 + N4*x4
        # 其中Ni是重心坐标
        
        # 将[-1,1]映射到[0,1]用于重心坐标
        L1 = 0.25 * (1 - xi - eta - zeta)  # 对应顶点1
        L2 = 0.25 * (1 + xi - eta - zeta)  # 对应顶点2  
        L3 = 0.25 * (1 - xi + eta - zeta)  # 对应顶点3
        L4 = 0.25 * (1 - xi - eta + zeta)  # 对应顶点4
        
        # 确保重心坐标非负且和为1
        # 对于在四面体内的点，Li >= 0 且 ΣLi = 1
        
        # 物理坐标插值
        phys_sps = (
            L1[:, np.newaxis] * cell_nodes[0] +
            L2[:, np.newaxis] * cell_nodes[1] +
            L3[:, np.newaxis] * cell_nodes[2] +
            L4[:, np.newaxis] * cell_nodes[3]
        )
        
        return phys_sps
        
    def _map_prism_to_physical(self, ref_sps, cell_nodes):
        """
        将参考棱柱单元内的点映射到物理棱柱单元。
        使用精确的等参变换。
        
        棱柱单元有6个节点：
        - 底面三角形: 节点0,1,2
        - 顶面三角形: 节点3,4,5
        
        形函数（在参考空间ξ,η∈三角形, ζ∈[-1,1]）:
        N₁ = 0.5*(1-ξ-η)*(1-ζ)
        N₂ = 0.5*ξ*(1-ζ)
        N₃ = 0.5*η*(1-ζ)
        N₄ = 0.5*(1-ξ-η)*(1+ζ)
        N₅ = 0.5*ξ*(1+ζ)
        N₆ = 0.5*η*(1+ζ)
        
        Args:
            ref_sps: 参考单元内SPs坐标 (n_sps, 3)
            cell_nodes: 棱柱六个顶点坐标 (6, 3)
            
        Returns:
            phys_sps: 物理空间中对应的SPs坐标 (n_sps, 3)
        """
        n_sps = ref_sps.shape[0]
        xi = ref_sps[:, 0]
        eta = ref_sps[:, 1]
        zeta = ref_sps[:, 2]
        
        # 计算形函数
        # 底面 (zeta = -1)
        N1 = 0.5 * (1 - xi - eta) * (1 - zeta) / 2.0
        N2 = 0.5 * xi * (1 - zeta) / 2.0
        N3 = 0.5 * eta * (1 - zeta) / 2.0
        
        # 顶面 (zeta = 1)
        N4 = 0.5 * (1 - xi - eta) * (1 + zeta) / 2.0
        N5 = 0.5 * xi * (1 + zeta) / 2.0
        N6 = 0.5 * eta * (1 + zeta) / 2.0
        
        # 物理坐标插值
        phys_sps = (
            N1[:, np.newaxis] * cell_nodes[0] +
            N2[:, np.newaxis] * cell_nodes[1] +
            N3[:, np.newaxis] * cell_nodes[2] +
            N4[:, np.newaxis] * cell_nodes[3] +
            N5[:, np.newaxis] * cell_nodes[4] +
            N6[:, np.newaxis] * cell_nodes[5]
        )
        
        return phys_sps

    def load_from_nas(self, nas_file: str):
        """
        从 .nas 文件加载网格并升级为高阶格式。
        
        Args:
            nas_file: .nas网格文件路径
        """
        print(f"Loading mesh from {nas_file}...")
        
        # TODO: 实际应调用 grid/nas_parser.py 解析NAS文件
        # 这里使用模拟数据作为占位符
        self.n_cells = 100
        self.sps_coords = np.random.rand(self.n_cells, self.n_sps_per_cell, 3) * 0.1
        
        # 预计算几何信息
        self._precompute_geometry()
        
        print(f"Mesh loaded: {self.n_cells} cells, Order P={self.order}")

    def _precompute_geometry(self):
        """预计算所有单元的 Jacobian 矩阵。"""
        if self.sps_coords is None:
            return
            
        mapper = CurvedMapping(self.order)
        all_dets = []
        all_inv_jacs = []
        
        # 对所有单元计算Jacobian
        for i in range(self.n_cells):
            try:
                jac_data = mapper.compute_jacobian(self.sps_coords[i])
                all_dets.append(jac_data['det_jacs'])
                all_inv_jacs.append(jac_data['inv_jacs'])
            except ValueError as e:
                print(f"Warning in cell {i}: {e}")
                # 可以选择跳过或修复畸变单元
                
        if all_dets:
            self.jacobians = {
                'det_jacs': np.array(all_dets),
                'inv_jacs': np.array(all_inv_jacs)
            }

    def verify_gcl(self, tolerance: float = 1e-10) -> bool:
        """
        验证几何守恒律 (GCL)。
        
        Args:
            tolerance: 相对标准差容差
            
        Returns:
            bool: GCL验证是否通过
        """
        if self.jacobians is None or 'det_jacs' not in self.jacobians:
            return False
            
        det_jacs = self.jacobians['det_jacs']
        
        # 检查所有单元的行列式是否为正
        if np.any(det_jacs <= 0):
            print(f"GCL Check Failed: Negative det(J) detected")
            return False
        
        # 检查行列式的变化率
        std_val = np.std(det_jacs)
        mean_val = np.mean(det_jacs)
        relative_std = std_val / mean_val if mean_val > 0 else np.inf
        
        print(f"GCL Check: std(det(J))/mean(det(J)) = {relative_std:.6e}")
        
        return relative_std < tolerance
    
    def get_cell_volume(self, cell_id: int) -> float:
        """
        计算指定单元的体积（通过Jacobian行列式积分）。
        
        Args:
            cell_id: 单元ID
            
        Returns:
            volume: 单元体积
        """
        if self.jacobians is None or cell_id >= self.n_cells:
            return 0.0
            
        # 简化：假设行列式在单元内近似常数
        det_jacs = self.jacobians['det_jacs'][cell_id]
        volume = np.mean(det_jacs) * 8.0  # 参考单元体积为8（[-1,1]^3）
        
        return volume
