"""网格边界条件映射结构。

提供 BoundaryMap 类，用于管理边界条件，支持自动检测、
手动配置和混合模式。
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class BoundaryMap:
    """边界条件映射表（v2.0扩展版）
    
    将边界名称映射到单元索引列表，支持基于Properties Name的自动识别。
    
    属性:
        groups: 边界组字典 {boundary_name: cell_indices_array}
               cell_indices_array: numpy int32数组，存储属于该边界的单元索引
        bc_types: 边界条件类型映射 {boundary_name: bc_type}
                 支持的类型: VELOCITY_INLET, PRESSURE_OUTLET, WALL, SYMMETRY, SLIP_WALL
        property_ids: Property ID映射 {boundary_name: PID}
                     记录每个边界对应的Property ID
        property_names: Property名称映射 {PID: property_name}
                       反向映射，用于调试和日志输出
        detection_mode: 检测模式 "auto" | "manual" | "hybrid"
        config_source: YAML配置文件路径（仅manual/hybrid模式）
        parameters: 边界参数 {boundary_name: param_dict}
                   存储每个边界的详细参数配置
    
    示例:
        >>> bmap = BoundaryMap(
        ...     groups={"inlet": np.array([0, 1, 2], dtype=np.int32)},
        ...     bc_types={"inlet": "VELOCITY_INLET"},
        ...     property_ids={"inlet": 10},
        ...     property_names={10: "INLET"},
        ...     detection_mode="auto",
        ...     parameters={"inlet": {"velocity": [33.33, 0.0, 0.0]}}
        ... )
        >>> print(bmap.get_boundary_type("inlet"))  # VELOCITY_INLET
        >>> print(bmap.get_property_id("inlet"))  # 10
    """
    
    groups: Dict[str, np.ndarray]
    bc_types: Dict[str, str]
    property_ids: Dict[str, int] = field(default_factory=dict)
    property_names: Dict[int, str] = field(default_factory=dict)
    detection_mode: str = "auto"
    config_source: Optional[str] = None
    parameters: Dict[str, Dict] = field(default_factory=dict)
    
    def __post_init__(self):
        """验证数据结构一致性
        
        抛出异常:
            ValueError: 如果groups、bc_types、property_ids键不一致
        """
        group_keys = set(self.groups.keys())
        bc_keys = set(self.bc_types.keys())
        pid_keys = set(self.property_ids.keys())
        
        # groups和bc_types必须完全一致
        if group_keys != bc_keys:
            raise ValueError(
                f"groups and bc_types keys mismatch:\n"
                f"  In groups but not bc_types: {group_keys - bc_keys}\n"
                f"  In bc_types but not groups: {bc_keys - group_keys}"
            )
        
        # property_ids应该是groups的子集（允许部分缺失）
        if not pid_keys.issubset(group_keys):
            raise ValueError(
                f"property_ids contains unknown boundaries: {pid_keys - group_keys}"
            )
        
        # 验证detection_mode合法性
        valid_modes = {"auto", "manual", "hybrid"}
        if self.detection_mode not in valid_modes:
            raise ValueError(
                f"Invalid detection_mode '{self.detection_mode}'. "
                f"Must be one of {valid_modes}"
            )
        
        # 确保所有数组为int32且连续
        for name, indices in self.groups.items():
            if indices.dtype != np.int32:
                self.groups[name] = indices.astype(np.int32)
            if not indices.flags['C_CONTIGUOUS']:
                self.groups[name] = np.ascontiguousarray(indices)
    
    def get_boundary_type(self, boundary_name: str) -> str:
        """获取边界条件类型
        
        Args:
            boundary_name: 边界名称
            
        Returns:
            str: 边界条件类型（如VELOCITY_INLET）
            
        抛出异常:
            KeyError: 边界名称不存在
        """
        if boundary_name not in self.bc_types:
            raise KeyError(f"Boundary '{boundary_name}' not found")
        return self.bc_types[boundary_name]
    
    def get_property_id(self, boundary_name: str) -> Optional[int]:
        """获取边界对应的Property ID
        
        Args:
            boundary_name: 边界名称
            
        Returns:
            Optional[int]: Property ID，若未记录则返回None
        """
        return self.property_ids.get(boundary_name)
    
    def get_property_name(self, boundary_name: str) -> Optional[str]:
        """获取边界对应的Property 名称
        
        Args:
            boundary_name: 边界名称
            
        Returns:
            Optional[str]: Property Name，若未记录则返回None
        """
        pid = self.get_property_id(boundary_name)
        if pid is not None:
            return self.property_names.get(pid)
        return None
    
    def get_parameters(self, boundary_name: str) -> Dict:
        """获取边界参数配置
        
        Args:
            boundary_name: 边界名称
            
        Returns:
            Dict: 参数字典，若无配置则返回空字典
        """
        return self.parameters.get(boundary_name, {})
    
    def update_parameters(self, boundary_name: str, **kwargs) -> None:
        """更新边界参数
        
        Args:
            boundary_name: 边界名称
            **kwargs: 要更新的参数键值对
            
        抛出异常:
            KeyError: 边界名称不存在
        """
        if boundary_name not in self.groups:
            raise KeyError(f"Boundary '{boundary_name}' not found")
        
        if boundary_name not in self.parameters:
            self.parameters[boundary_name] = {}
        
        self.parameters[boundary_name].update(kwargs)
    
    @property
    def boundary_names(self) -> List[str]:
        """获取所有边界名称"""
        return list(self.groups.keys())
    
    @property
    def boundary_count(self) -> int:
        """获取边界数量"""
        return len(self.groups)
    
    def has_boundary(self, boundary_name: str) -> bool:
        """检查边界是否存在
        
        Args:
            boundary_name: 边界名称
            
        Returns:
            bool: 是否存在
        """
        return boundary_name in self.groups
    
    def get_node_indices(self, boundary_name: str) -> np.ndarray:
        """获取边界节点索引数组（兼容旧接口）
        
        Args:
            boundary_name: 边界名称
            
        Returns:
            np.ndarray: 节点索引数组（int32）
            
        抛出异常:
            KeyError: 边界名称不存在
        """
        if boundary_name not in self.groups:
            raise KeyError(f"Boundary '{boundary_name}' not found")
        return self.groups[boundary_name]
    
    def get_cell_indices(self, boundary_name: str) -> np.ndarray:
        """获取边界单元索引数组
        
        Args:
            boundary_name: 边界名称
            
        Returns:
            np.ndarray: 单元索引数组（int32）
            
        抛出异常:
            KeyError: 边界名称不存在
        """
        if boundary_name not in self.groups:
            raise KeyError(f"Boundary '{boundary_name}' not found")
        return self.groups[boundary_name]
    
    def get_summary(self) -> Dict:
        """获取边界配置摘要
        
        Returns:
            Dict: 包含边界统计信息的字典
        """
        summary = {
            "total_boundaries": self.boundary_count,
            "detection_mode": self.detection_mode,
            "config_source": self.config_source,
            "boundaries": {}
        }
        
        for name in self.boundary_names:
            summary["boundaries"][name] = {
                "type": self.bc_types[name],
                "cell_count": len(self.groups[name]),
                "property_id": self.property_ids.get(name),
                "property_name": self.get_property_name(name),
                "parameters": self.parameters.get(name, {})
            }
        
        return summary
    
    def save_hdf5(self, group, prefix: str = "boundaries"):
        """保存边界映射到HDF5组
        
        Args:
            group: h5py Group对象
            prefix: HDF5数据集前缀
        """
        import json
        
        # 保存基本信息
        names = self.boundary_names
        group.create_dataset(f"{prefix}/names", data=np.array(names, dtype='S'))
        group.attrs['detection_mode'] = self.detection_mode
        if self.config_source:
            group.attrs['config_source'] = self.config_source
        
        # 保存每个边界的详细信息
        for i, name in enumerate(names):
            # 单元索引
            cells = self.groups[name]
            ds_cells = group.create_dataset(f"{prefix}/boundary_{i}/cells", data=cells)
            ds_cells.attrs['name'] = name
            
            # BC类型
            group.create_dataset(
                f"{prefix}/boundary_{i}/bc_type",
                data=np.string_(self.bc_types[name])
            )
            
            # 属性 ID
            pid = self.property_ids.get(name, -1)
            group.create_dataset(
                f"{prefix}/boundary_{i}/property_id",
                data=np.int32(pid)
            )
            
            # 参数（序列化为JSON）
            params = self.parameters.get(name, {})
            params_json = json.dumps(params)
            group.create_dataset(
                f"{prefix}/boundary_{i}/parameters",
                data=np.string_(params_json)
            )
    
    @classmethod
    def load_hdf5(cls, group, prefix: str = "boundaries") -> 'BoundaryMap':
        """从HDF5组加载边界映射
        
        Args:
            group: h5py Group对象
            prefix: HDF5数据集前缀
            
        Returns:
            BoundaryMap: 加载的边界映射对象
        """
        import json
        
        names = group[f"{prefix}/names"][:].astype(str)
        detection_mode = group.attrs.get('detection_mode', 'auto')
        config_source = group.attrs.get('config_source')
        
        groups = {}
        bc_types = {}
        property_ids = {}
        property_names = {}
        parameters = {}
        
        for i, name in enumerate(names):
            # 加载单元索引
            cells = group[f"{prefix}/boundary_{i}/cells"][:]
            groups[name] = cells
            
            # 加载BC类型
            bc_type = group[f"{prefix}/boundary_{i}/bc_type"][()].decode('utf-8')
            bc_types[name] = bc_type
            
            # 加载Property ID
            pid = int(group[f"{prefix}/boundary_{i}/property_id"][()])
            if pid >= 0:
                property_ids[name] = pid
            
            # 加载Parameters
            params_json = group[f"{prefix}/boundary_{i}/parameters"][()].decode('utf-8')
            if params_json:
                parameters[name] = json.loads(params_json)
        
        return cls(
            groups=groups,
            bc_types=bc_types,
            property_ids=property_ids,
            property_names=property_names,
            detection_mode=detection_mode,
            config_source=config_source,
            parameters=parameters
        )
