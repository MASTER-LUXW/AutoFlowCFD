"""边界条件管理器。

本模块提供 `BoundaryManager` 类，用于在 AutoFlowCFD 仿真中管理和登记
边界条件。

Key Components:
    - BoundaryManager: 边界条件管理器主类

⚠️ 现状说明：`apply_boundary()`/`apply_all()` 方法（把边界条件直接
应用到 solution 数组上）已删除——这条路径从未被生产求解路径调用过，
真正的边界处理由 `core/bc_handler.py` 的 `BoundaryConditionHandler`
独立完成。本类现在只承担边界条件的元数据登记角色（`add_bc`/`get_bc`/
`auto_configure`/`configure_from_yaml`/`hybrid_configure`/
`get_summary` 等），这部分**是**求解主流程实际使用的（见
`solver_steady.py`/`transient_solver_loop.py` 的 `_setup_boundary_conditions`
调用 `add_bc`），用于登记配置供查询/导出/校验，不用于计算边界值。详见
`conditions.py` 模块文档字符串里对这个决定的完整说明。

Example:
    >>> from autoflowcfd.boundary import BoundaryManager
    >>> from autoflowcfd.grid import BoundaryMap
    >>>
    >>> # 创建边界映射
    >>> bmap = BoundaryMap()
    >>> bmap.add_boundary("INLET", [0, 1, 2])
    >>> bmap.add_boundary("OUTLET", [3, 4, 5])
    >>> bmap.add_boundary("BODY", list(range(6, 100)))
    >>>
    >>> # 创建管理器
    >>> bc_manager = BoundaryManager(bmap)
    >>>
    >>> # 登记边界条件
    >>> bc_manager.add_bc("INLET", velocity_x=30.0)
    >>> bc_manager.add_bc("OUTLET", pressure=101325.0)
    >>> bc_manager.add_bc("BODY", wall_function='enhanced')
    >>>
    >>> # 查看登记摘要
    >>> print(bc_manager.get_summary())
"""

from typing import Dict, Any, List, Optional
from loguru import logger

from .conditions import (
    BaseBC,
    create_boundary_condition,
    get_boundary_condition_class,
)
from .config import YAMLConfigLoader, BoundaryTypeMapper, ParameterValidator


class BoundaryManager:
    """CFD 仿真的边界条件管理器 (v2.0)。

    管理边界条件实例，并把它们与网格给出的边界映射关联起来登记。支持
    三种配置模式：auto（自动）、manual（手动）、hybrid（混合）。

    Attributes:
        boundary_map: 来自网格数据的边界映射
        _bc_instances: 边界条件实例字典
        _config_loader: YAML 配置加载器
        _type_mapper: auto 模式用的边界类型映射器
        _param_validator: 参数校验器

    Example:
        >>> manager = BoundaryManager(boundary_map)
        >>> manager.auto_configure()  # Auto 模式
        >>> manager.configure_from_yaml("config.yaml")  # Manual 模式
        >>> manager.hybrid_configure("config.yaml")  # Hybrid 模式
    """

    def __init__(self, boundary_map: Any):
        """初始化边界管理器。

        Args:
            boundary_map: 来自网格数据的 BoundaryMap 对象

        Raises:
            ValueError: boundary_map 无效时
        """
        if not hasattr(boundary_map, 'boundary_names'):
            raise ValueError("boundary_map must have 'boundary_names' attribute")

        self.boundary_map = boundary_map
        self._bc_instances: Dict[str, BaseBC] = {}
        self._config_loader = YAMLConfigLoader()
        self._type_mapper = BoundaryTypeMapper()
        self._param_validator = ParameterValidator()

        logger.info(
            f"BoundaryManager initialized with "
            f"{len(boundary_map.boundary_names)} boundaries: "
            f"{boundary_map.boundary_names}"
        )

    def auto_configure(self) -> None:
        """自动配置边界条件（Auto 模式）。

        根据 Properties Name 关键词自动识别边界类型，并套用默认参数。

        Raises:
            BoundaryError: 解析失败或没有找到有效 Properties 时

        Example:
            >>> bc_manager = BoundaryManager(grid.boundaries)
            >>> bc_manager.auto_configure()
            >>> print(bc_manager.get_summary())
        """
        logger.info("Configuring boundaries in AUTO mode...")

        # 更新边界映射的识别模式
        self.boundary_map.detection_mode = "auto"

        # 为每个边界用默认参数创建 BC 实例
        for boundary_name in self.boundary_map.boundary_names:
            bc_type = self.boundary_map.get_boundary_type(boundary_name)

            # 按 BC 类型取默认参数
            default_params = self._get_default_parameters(bc_type)

            # 创建边界条件实例
            try:
                bc_instance = create_boundary_condition(bc_type, **default_params)
                bc_instance.validate()
                self._bc_instances[boundary_name] = bc_instance

                # 把参数存进边界映射
                self.boundary_map.parameters[boundary_name] = default_params

                logger.info(
                    f"Auto-configured {boundary_name} as {bc_type} "
                    f"with {len(self.boundary_map.get_cell_indices(boundary_name))} cells"
                )
            except Exception as e:
                logger.error(f"Failed to configure boundary '{boundary_name}': {e}")
                raise

        logger.success(
            f"Auto configuration completed: {len(self._bc_instances)} boundaries configured"
        )

    def configure_from_yaml(self, config_path: str) -> None:
        """从 YAML 配置文件配置边界条件（Manual 模式）。

        完全依赖 YAML 配置文件，YAML 中未配置的 Properties 会报错。

        Args:
            config_path: YAML 配置文件路径

        Raises:
            FileNotFoundError: 找不到配置文件
            ConfigurationError: YAML 语法错误或配置不完整

        Example:
            >>> bc_manager.configure_from_yaml("boundary_config.yaml")
        """
        logger.info(f"Configuring boundaries in MANUAL mode from {config_path}...")

        # 加载 YAML 配置
        config = self._config_loader.load(config_path)

        # 校验模式是否为 manual
        mode = self._config_loader.get_mode()
        if mode != 'manual':
            logger.warning(
                f"Configuration mode is '{mode}', but configure_from_yaml expects 'manual'. "
                f"Proceeding anyway."
            )

        # 更新边界映射
        self.boundary_map.detection_mode = "manual"
        self.boundary_map.config_source = config_path

        # 从配置里取 properties 映射
        properties_mapping = self._config_loader.get_properties_mapping()

        # 逐个从 YAML 配置 property
        configured_boundaries = set()

        for prop_name, prop_config in properties_mapping.items():
            bc_type = prop_config['type']
            params = prop_config.get('parameters', {})

            # 在 boundary_map 里找匹配的边界
            boundary_name = self._find_matching_boundary(prop_name)

            if boundary_name is None:
                logger.warning(
                    f"Property '{prop_name}' from YAML not found in boundary map. "
                    f"Skipping."
                )
                continue

            # 校验参数
            self._validate_parameters(params, bc_type, prop_name)

            # 创建边界条件实例
            try:
                bc_instance = create_boundary_condition(bc_type, **params)
                bc_instance.validate()
                self._bc_instances[boundary_name] = bc_instance

                # 把参数存进边界映射
                self.boundary_map.parameters[boundary_name] = params

                configured_boundaries.add(boundary_name)

                logger.info(
                    f"Configured {boundary_name} as {bc_type} "
                    f"(from Property '{prop_name}') with {len(self.boundary_map.get_cell_indices(boundary_name))} cells"
                )
            except Exception as e:
                logger.error(f"Failed to configure boundary '{boundary_name}': {e}")
                raise

        # 检查是否所有边界都配置了（manual 模式要求）
        unconfigured = set(self.boundary_map.boundary_names) - configured_boundaries
        if unconfigured:
            raise ValueError(
                f"Manual mode requires all boundaries to be configured. "
                f"Unconfigured boundaries: {unconfigured}"
            )

        logger.success(
            f"Manual configuration completed: {len(self._bc_instances)} boundaries configured"
        )

    def hybrid_configure(self, config_path: str) -> None:
        """混合模式配置边界条件。

        YAML 配置优先；YAML 中未配置的 Properties 用自动识别规则处理。

        Args:
            config_path: YAML 配置文件路径

        Raises:
            FileNotFoundError: 找不到配置文件
            ConfigurationError: YAML 语法错误

        Example:
            >>> bc_manager.hybrid_configure("partial_config.yaml")
        """
        logger.info(f"Configuring boundaries in HYBRID mode from {config_path}...")

        # 加载 YAML 配置
        config = self._config_loader.load(config_path)

        # 更新边界映射
        self.boundary_map.detection_mode = "hybrid"
        self.boundary_map.config_source = config_path

        # 从配置里取 properties 映射和默认值
        properties_mapping = self._config_loader.get_properties_mapping()
        defaults = self._config_loader.get_defaults()

        configured_boundaries = set()

        # 第一步：优先从 YAML 配置边界
        for prop_name, prop_config in properties_mapping.items():
            bc_type = prop_config['type']
            params = prop_config.get('parameters', {})

            # 如果有默认值则合并
            if bc_type.lower() in defaults:
                default_params = defaults[bc_type.lower()]
                merged_params = {**default_params, **params}
            else:
                merged_params = params

            # 在 boundary_map 里找匹配的边界
            boundary_name = self._find_matching_boundary(prop_name)

            if boundary_name is None:
                logger.warning(
                    f"Property '{prop_name}' from YAML not found in boundary map. "
                    f"Skipping."
                )
                continue

            # 校验参数
            self._validate_parameters(merged_params, bc_type, prop_name)

            # 创建边界条件实例
            try:
                bc_instance = create_boundary_condition(bc_type, **merged_params)
                bc_instance.validate()
                self._bc_instances[boundary_name] = bc_instance

                # 把参数存进边界映射
                self.boundary_map.parameters[boundary_name] = merged_params

                configured_boundaries.add(boundary_name)

                logger.info(
                    f"YAML-configured {boundary_name} as {bc_type} "
                    f"(from Property '{prop_name}') with {len(self.boundary_map.get_cell_indices(boundary_name))} cells"
                )
            except Exception as e:
                logger.error(f"Failed to configure boundary '{boundary_name}': {e}")
                raise

        # 第二步：自动配置剩余边界
        for boundary_name in self.boundary_map.boundary_names:
            if boundary_name in configured_boundaries:
                continue

            # 用自动识别确定 BC 类型
            prop_name = self.boundary_map.get_property_name(boundary_name)
            if prop_name:
                bc_type = self._type_mapper.map(prop_name)
            else:
                # 兜底：用 boundary_map 里已有的 bc_type
                bc_type = self.boundary_map.get_boundary_type(boundary_name)

            # 取默认参数
            if bc_type.lower() in defaults:
                default_params = defaults[bc_type.lower()]
            else:
                default_params = self._get_default_parameters(bc_type)

            # 创建边界条件实例
            try:
                bc_instance = create_boundary_condition(bc_type, **default_params)
                bc_instance.validate()
                self._bc_instances[boundary_name] = bc_instance

                # 把参数存进边界映射
                self.boundary_map.parameters[boundary_name] = default_params

                logger.info(
                    f"Auto-configured {boundary_name} as {bc_type} "
                    f"(fallback) with {len(self.boundary_map.get_cell_indices(boundary_name))} cells"
                )
            except Exception as e:
                logger.error(f"Failed to auto-configure boundary '{boundary_name}': {e}")
                raise

        logger.success(
            f"Hybrid configuration completed: {len(self._bc_instances)} boundaries configured"
        )

    def _find_matching_boundary(self, prop_name: str) -> Optional[str]:
        """为一个 property 名在 boundary_map 里找匹配的边界名。

        Args:
            prop_name: 来自 YAML 或 NAS 文件的 property 名

        Returns:
            Optional[str]: 匹配到的边界名，找不到则为 None
        """
        # 先尝试精确匹配（不区分大小写）
        prop_name_lower = prop_name.lower()

        for boundary_name in self.boundary_map.boundary_names:
            if boundary_name.lower() == prop_name_lower:
                return boundary_name

        # 再尝试部分匹配（property 名包含边界名，或反过来）
        for boundary_name in self.boundary_map.boundary_names:
            if prop_name_lower in boundary_name.lower() or boundary_name.lower() in prop_name_lower:
                return boundary_name

        return None

    def _validate_parameters(self, params: Dict[str, Any], bc_type: str, prop_name: str) -> None:
        """校验边界条件参数。

        Args:
            params: 参数字典
            bc_type: 边界条件类型
            prop_name: property 名（用于错误信息）

        Raises:
            ValueError: 参数无效时
        """
        # 校验速度
        if 'velocity' in params:
            self._param_validator.validate_velocity(params['velocity'])

        # 校验压力
        if 'pressure' in params:
            self._param_validator.validate_pressure(params['pressure'])

        # 校验湍流强度
        if 'turbulence_intensity' in params:
            self._param_validator.validate_turbulence_intensity(params['turbulence_intensity'])

        # 校验粗糙度高度
        if 'roughness_height' in params:
            self._param_validator.validate_roughness_height(params['roughness_height'])

    def _get_default_parameters(self, bc_type: str) -> Dict[str, Any]:
        """获取某边界条件类型的默认参数。

        Args:
            bc_type: 边界条件类型

        Returns:
            Dict[str, Any]: 默认参数字典
        """
        defaults = {
            'VELOCITY_INLET': {
                'velocity': [33.33, 0.0, 0.0],
                'turbulence_intensity': 0.05
            },
            'PRESSURE_OUTLET': {
                'pressure': 0.0
            },
            'WALL': {
                'wall_function': 'standard'
            },
            'SYMMETRY': {},
            'SLIP_WALL': {}
        }

        return defaults.get(bc_type, {})

    def update_boundary_params(self, boundary_name: str, **kwargs) -> None:
        """更新指定边界的参数。

        Args:
            boundary_name: 边界名称
            **kwargs: 要更新的参数键值对

        Raises:
            KeyError: 边界名称不存在
            ValueError: 参数值不合法

        Example:
            >>> bc_manager.update_boundary_params(
            ...     "CAR_SURFACE",
            ...     wall_function="enhanced",
            ...     roughness_height=0.0001
            ... )
        """
        if boundary_name not in self._bc_instances:
            raise KeyError(f"Boundary '{boundary_name}' not configured")

        # 取当前 BC 实例
        bc = self._bc_instances[boundary_name]
        bc_type = bc.get_type()

        # 更新 BC 实例的参数
        for key, value in kwargs.items():
            if hasattr(bc, key):
                setattr(bc, key, value)
            elif key in bc.params:
                bc.params[key] = value
            else:
                logger.warning(f"Parameter '{key}' not recognized for {bc_type}")

        # 更新边界映射里的参数
        if boundary_name not in self.boundary_map.parameters:
            self.boundary_map.parameters[boundary_name] = {}
        self.boundary_map.parameters[boundary_name].update(kwargs)

        # 重新校验
        try:
            bc.validate()
        except ValueError as e:
            raise ValueError(f"Invalid parameters after update: {e}")

        logger.info(f"Updated parameters for boundary '{boundary_name}': {kwargs}")

    def export_to_vtk(self, output_path: str) -> None:
        """导出边界分组到 VTK 文件。

        用于 ParaView 可视化验证边界划分正确性。

        Args:
            output_path: VTK 输出文件路径

        Example:
            >>> bc_manager.export_to_vtk("boundaries.vtk")
        """
        # TODO: 实现 VTK 导出
        logger.info(f"Exporting boundary visualization to VTK: {output_path}")
        # 待后处理模块补充这部分功能时再实现

    def export_to_json(self, output_path: str) -> dict:
        """导出边界统计信息到 JSON 文件。

        Args:
            output_path: JSON 输出文件路径

        Returns:
            dict: 边界统计信息字典

        Example:
            >>> stats = bc_manager.export_to_json("boundary_stats.json")
            >>> print(stats["total_boundaries"])
        """
        import json
        from pathlib import Path

        summary = self.get_summary()

        path = Path(output_path)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"Exported boundary statistics to JSON: {output_path}")

        return summary

    def generate_template(self, output_path: str) -> None:
        """生成 YAML 配置模板。

        基于当前自动识别结果生成可编辑的 YAML 模板。

        Args:
            output_path: YAML 模板输出路径

        Example:
            >>> bc_manager.generate_template("template.yaml")
            # 然后编辑 template.yaml，补充/修改参数
        """
        # 构建已识别边界的信息
        detected = {}
        for name in self.boundary_map.boundary_names:
            detected[name] = {
                'type': self.boundary_map.get_boundary_type(name),
                'cells': len(self.boundary_map.get_cell_indices(name)),
                'property_id': self.boundary_map.get_property_id(name),
                'property_name': self.boundary_map.get_property_name(name)
            }

        # 生成模板
        self._config_loader.generate_template(output_path, detected)

    def get_summary(self) -> dict:
        """获取边界配置摘要。

        Returns:
            dict: 包含边界统计信息的字典

        Example:
            >>> summary = bc_manager.get_summary()
            >>> for name, info in summary["boundaries"].items():
            ...     print(f"{name}: {info['type']} ({info['cell_count']} cells)")
        """
        summary = {
            'total_boundaries': self.boundary_map.boundary_count,
            'detection_mode': self.boundary_map.detection_mode,
            'config_source': self.boundary_map.config_source,
            'boundaries_with_bc': len(self._bc_instances),
            'boundaries_without_bc': [],
            'bc_details': {},
        }

        # 找出没有 BC 的边界
        for name in self.boundary_map.boundary_names:
            if name not in self._bc_instances:
                summary['boundaries_without_bc'].append(name)

        # 取每个 BC 的详情
        for name in self.boundary_map.boundary_names:
            cell_count = len(self.boundary_map.get_cell_indices(name))
            bc_type = self.boundary_map.get_boundary_type(name)
            params = self.boundary_map.get_parameters(name)

            summary['bc_details'][name] = {
                'type': bc_type,
                'cell_count': cell_count,
                'property_id': self.boundary_map.get_property_id(name),
                'property_name': self.boundary_map.get_property_name(name),
                'params': params,
            }

        return summary

    def add_bc(self, boundary_name: str, bc_type: Optional[str] = None, **kwargs) -> BaseBC:
        """给一个边界登记边界条件。

        Args:
            boundary_name: 边界名称（必须已存在于 boundary_map 中）
            bc_type: 边界条件类型（例如 'VELOCITY_INLET'、'WALL'）。
                    若为 None，则从 boundary_name 推断。
            **kwargs: 边界条件参数

        Returns:
            BaseBC: 创建的边界条件实例

        Raises:
            KeyError: boundary_map 中找不到 boundary_name 时
            ValueError: bc_type 无效时

        Example:
            >>> manager.add_bc("INLET", velocity_x=30.0, pressure=101325.0)
            >>> manager.add_bc("BODY", bc_type="WALL", wall_function='enhanced')
        """
        # 检查边界是否存在
        if not self.boundary_map.has_boundary(boundary_name):
            raise KeyError(
                f"Boundary '{boundary_name}' not found in boundary map. "
                f"Available boundaries: {self.boundary_map.boundary_names}"
            )

        # 未指定 bc_type 时从 boundary_name 推断
        if bc_type is None:
            prop_name = self.boundary_map.get_property_name(boundary_name)
            if prop_name:
                bc_type = self._type_mapper.map(prop_name)
            else:
                bc_type = self.boundary_map.get_boundary_type(boundary_name)

        # 校验 bc_type
        try:
            get_boundary_condition_class(bc_type)
        except KeyError:
            raise ValueError(
                f"Invalid boundary condition type: {bc_type}. "
                f"Use bc_type parameter to specify a valid type."
            )

        # 创建边界条件实例
        bc_instance = create_boundary_condition(bc_type, **kwargs)

        # 校验边界条件
        try:
            bc_instance.validate()
        except ValueError as e:
            raise ValueError(f"Invalid parameters for {boundary_name}: {e}")

        # 存储实例
        self._bc_instances[boundary_name] = bc_instance

        # 把参数存进边界映射
        self.boundary_map.parameters[boundary_name] = kwargs

        logger.info(f"Added {bc_type} BC to boundary '{boundary_name}'")

        return bc_instance

    def get_bc(self, boundary_name: str) -> BaseBC:
        """获取某个边界的边界条件实例。

        Args:
            boundary_name: 边界名称

        Returns:
            BaseBC: 边界条件实例

        Raises:
            KeyError: 该边界没有登记边界条件时
        """
        if boundary_name not in self._bc_instances:
            raise KeyError(
                f"No boundary condition assigned to '{boundary_name}'. "
                f"Use add_bc() or configure methods to assign one."
            )

        return self._bc_instances[boundary_name]

    def update_time_dependent_bcs(self, time: float) -> None:
        """更新随时间变化的边界条件。

        对于参数随时间变化的边界，根据当前仿真时间更新其参数。

        Args:
            time: 当前仿真时间
        """
        # TODO: 实现随时间变化的边界条件更新
        # 需要检查哪些 BC 有随时间变化的参数并相应更新
        logger.debug(f"Updating time-dependent BCs at t={time:.6f}s")
        pass

    def validate_all(self) -> bool:
        """校验全部边界条件。

        Returns:
            bool: 全部 BC 都有效则为 True

        Raises:
            ValueError: 任一 BC 无效时
        """
        for boundary_name, bc in self._bc_instances.items():
            try:
                bc.validate()
            except ValueError as e:
                raise ValueError(f"Invalid BC for '{boundary_name}': {e}")

        logger.info("All boundary conditions validated successfully")
        return True

    def list_boundaries(self) -> List[str]:
        """列出全部边界名称。

        Returns:
            List[str]: 边界名称列表
        """
        return self.boundary_map.boundary_names

    def list_assigned_bcs(self) -> List[str]:
        """列出已登记边界条件的边界。

        Returns:
            List[str]: 已登记 BC 的边界名称列表
        """
        return list(self._bc_instances.keys())

    def remove_bc(self, boundary_name: str) -> None:
        """移除某个边界的边界条件。

        Args:
            boundary_name: 边界名称

        Raises:
            KeyError: 该边界没有登记边界条件时
        """
        if boundary_name not in self._bc_instances:
            raise KeyError(f"No BC assigned to boundary '{boundary_name}'")

        del self._bc_instances[boundary_name]
        logger.info(f"Removed BC from boundary '{boundary_name}'")

    def clear_all(self) -> None:
        """移除全部边界条件。"""
        self._bc_instances.clear()
        logger.info("Cleared all boundary conditions")

    def __repr__(self) -> str:
        """字符串表示。"""
        return (
            f"BoundaryManager("
            f"boundaries={len(self.boundary_map.boundary_names)}, "
            f"assigned={len(self._bc_instances)}, "
            f"mode={self.boundary_map.detection_mode})"
        )

    def __len__(self) -> int:
        """返回已登记 BC 的数量。"""
        return len(self._bc_instances)
