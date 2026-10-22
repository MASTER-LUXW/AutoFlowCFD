"""网格边界条件映射结构：BoundaryMap（边界组 -> 边界面 / 边界面所属单元、边界类型、Property ID 与参数）。"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class BoundaryMap:
    """边界条件映射表：边界名称 -> 边界面（体网格）/ 单元下标与边界类型。

    同一个类用在两处，`groups` 的含义随之不同：
      * 面网格（`GridData`）：`groups[名]` 是属于该组的三角形下标，`faces` 为 None；
      * 体网格（`VolumeMeshData`）：`faces[名]` 是该组的外部面——体网格节点编号的三角形 `(n, 3)`，棱柱侧面四边形按
        面提取的统一对角线拆成两个（`mesh_gen/utils/prism_split.py`），是边界组归属的唯一事实来源；`groups[名]` 是
        拥有这些面的单元（全局混合编号：棱柱 [0, n_prism)、四面体其后），由 `faces` 派生。

    逐面存储（2026-10-22，I-130）以前体网格只存单元下标：一个单元同时拥有两个组的外部面（角 / 棱单元）时无法判定
    每个面属于哪个组，壁面距离过包含（I-92）、角单元误打标签（I-125）、周期配对需要几何二次筛选，求解期逐面打
    标签只能回到原始面网格做几何匹配。

    边界类型由解析器按 Property 名称识别（`grid/nas_io/boundary_types.py`），可用 `--bc-type` 覆盖。

    属性:
        groups: 边界组字典 {boundary_name: indices}（int32；面网格为三角形下标，体网格为单元下标，见上）
        faces: 体网格的逐组外部面 {boundary_name: (n, 3) int64 节点编号}；面网格为 None
        bc_types: 边界条件类型映射 {boundary_name: bc_type}
                 支持的类型: VELOCITY_INLET, PRESSURE_OUTLET, WALL, SYMMETRY, SLIP_WALL
        property_ids: Property ID映射 {boundary_name: PID}
                     记录每个边界对应的Property ID
        parameters: 边界参数 {boundary_name: param_dict}
                   存储每个边界的详细参数配置

    示例:
        >>> bmap = BoundaryMap(
        ...     groups={"inlet": np.array([0, 1, 2], dtype=np.int32)},
        ...     bc_types={"inlet": "VELOCITY_INLET"},
        ...     property_ids={"inlet": 10},
        ...     parameters={"inlet": {"velocity": [33.33, 0.0, 0.0]}}
        ... )
        >>> print(bmap.get_boundary_type("inlet"))  # VELOCITY_INLET
        >>> print(bmap.get_property_id("inlet"))  # 10
    """

    groups: Dict[str, np.ndarray]
    bc_types: Dict[str, str]
    property_ids: Dict[str, int] = field(default_factory=dict)
    parameters: Dict[str, Dict] = field(default_factory=dict)
    faces: Optional[Dict[str, np.ndarray]] = None

    @classmethod
    def from_exterior_faces(cls, names: List[str], face_nodes: np.ndarray, owner: np.ndarray, group: np.ndarray,
                            bc_types: Dict[str, str]) -> 'BoundaryMap':
        """体网格的边界映射：外部面 `face_nodes (n, 3)`（体网格节点编号）按 `group (n,)`（`names` 的下标）逐面
        分组，`groups` 取拥有这些面的单元（`owner (n,)`）。没有面的组不出现；出现的组都必须在 `bc_types` 里。"""
        face_nodes = np.asarray(face_nodes, dtype=np.int64)
        groups: Dict[str, np.ndarray] = {}
        faces: Dict[str, np.ndarray] = {}
        types: Dict[str, str] = {}
        for gi, name in enumerate(names):
            sel = np.asarray(group) == gi
            if not sel.any():
                continue
            if name not in bc_types:
                raise ValueError(f"边界组 '{name}' 没有边界条件类型")
            faces[name] = np.ascontiguousarray(face_nodes[sel])
            groups[name] = np.unique(np.asarray(owner)[sel]).astype(np.int32)
            types[name] = bc_types[name]
        return cls(groups=groups, bc_types=types, faces=faces)

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

        if self.faces is not None:
            if set(self.faces) != group_keys:
                raise ValueError(f"faces 与 groups 的组名不一致：{sorted(set(self.faces) ^ group_keys)}")
            for name, tri in self.faces.items():
                tri = np.ascontiguousarray(tri, dtype=np.int64)
                if tri.ndim != 2 or tri.shape[1] != 3:
                    raise ValueError(f"边界组 '{name}' 的外部面应为 (n, 3) 节点编号，得到形状 {tri.shape}")
                self.faces[name] = tri

        # 确保所有数组为int32且连续
        for name, indices in self.groups.items():
            if indices.dtype != np.int32:
                self.groups[name] = indices.astype(np.int32)
            if not indices.flags['C_CONTIGUOUS']:
                self.groups[name] = np.ascontiguousarray(indices)

    def face_counts(self) -> Dict[str, int]:
        """体网格每个组的外部面数。

        Raises:
            ValueError: 不是体网格的边界映射（没有逐面边界组）
        """
        if self.faces is None:
            raise ValueError("这份边界映射没有逐面边界组（面网格的 BoundaryMap）")
        return {name: int(len(tri)) for name, tri in self.faces.items()}

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

    # `get_node_indices` 已于 2026-09-15 删除。它是 `get_cell_indices` 的
    # 错名"兼容旧接口"别名——返回的恒是**单元**索引（见本类 `groups`
    # 字段文档），但名字让调用方以为是节点索引。这个错名直接造成了一处
    # 一阶物理错误：`cli/solve/wall_distance.py` 据此用
    # `max(indices) >= n_nodes` 去猜索引含义，而该判据恰好只在 WALL 组上
    # 猜错（壁面边界单元是边界层棱柱、索引落在 [0, n_prism)，两张真实
    # 网格都满足 n_prism < n_nodes），使壁面距离场变成"到一堆按编号散布
    # 在全域的任意节点的距离"。完整推导见
    # `core/utils/wall_distance/surface.py` 的模块文档。
    # 需要单元索引用 `get_cell_indices`；需要壁面节点用那个函数。

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

        # 保存每个边界的详细信息
        for i, name in enumerate(names):
            # 单元索引
            cells = self.groups[name]
            ds_cells = group.create_dataset(f"{prefix}/boundary_{i}/cells", data=cells)
            ds_cells.attrs['name'] = name
            if self.faces is not None:
                group.create_dataset(f"{prefix}/boundary_{i}/faces", data=self.faces[name])

            # BC类型
            group.create_dataset(
                f"{prefix}/boundary_{i}/bc_type",
                data=np.bytes_(self.bc_types[name])
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
                data=np.bytes_(params_json)
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

        groups = {}
        bc_types = {}
        property_ids = {}
        parameters = {}
        faces = {}

        for i, name in enumerate(names):
            # 加载单元索引
            cells = group[f"{prefix}/boundary_{i}/cells"][:]
            groups[name] = cells
            if f"{prefix}/boundary_{i}/faces" in group:
                faces[name] = group[f"{prefix}/boundary_{i}/faces"][:]

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
            parameters=parameters,
            faces=faces if faces else None,
        )
