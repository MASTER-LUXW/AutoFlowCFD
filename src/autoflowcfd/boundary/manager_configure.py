"""BoundaryManager 的配置方法混入类。

从 manager.py 拆出，控制单文件行数。提供 auto/manual/hybrid 三种配置模式
以及参数校验、默认值、更新等辅助方法。

BoundaryManager 通过继承本模块的 _BoundaryConfigMixin 获得这些方法。
"""

from typing import Dict, Any, Optional
from loguru import logger

from .conditions import BaseBC, create_boundary_condition


class _BoundaryConfigMixin:
    """BoundaryManager 配置方法混入。

    子类需要提供以下实例属性：
        boundary_map  - 网格的 BoundaryMap 对象
        _bc_instances - Dict[str, BaseBC] 已登记的 BC 实例
        _config_loader - YAMLConfigLoader 实例
        _type_mapper   - BoundaryTypeMapper 实例
        _param_validator - ParameterValidator 实例
    """

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
