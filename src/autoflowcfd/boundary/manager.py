"""Boundary condition manager.

This module provides the BoundaryManager class for managing and applying
boundary conditions in AutoFlowCFD simulations.

Key Components:
    - BoundaryManager: Main boundary condition manager
    
Example:
    >>> from autoflowcfd.boundary import BoundaryManager
    >>> from autoflowcfd.grid import BoundaryMap
    >>> 
    >>> # Create boundary map
    >>> bmap = BoundaryMap()
    >>> bmap.add_boundary("INLET", [0, 1, 2])
    >>> bmap.add_boundary("OUTLET", [3, 4, 5])
    >>> bmap.add_boundary("BODY", list(range(6, 100)))
    >>> 
    >>> # Create manager
    >>> bc_manager = BoundaryManager(bmap)
    >>> 
    >>> # Add boundary conditions
    >>> bc_manager.add_bc("INLET", velocity_x=30.0)
    >>> bc_manager.add_bc("OUTLET", pressure=101325.0)
    >>> bc_manager.add_bc("BODY", wall_function='enhanced')
    >>> 
    >>> # Apply to solution
    >>> bc_manager.apply_all(solution, time=0.0)
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
    """Boundary condition manager for CFD simulations (v2.0).
    
    Manages boundary condition instances and applies them to solution vectors
    based on boundary mappings from the grid. Supports three configuration modes:
    auto, manual, and hybrid.
    
    Attributes:
        boundary_map: Boundary mapping from grid data
        _bc_instances: Dictionary of boundary condition instances
        _config_loader: YAML configuration loader
        _type_mapper: Boundary type mapper for auto mode
        _param_validator: Parameter validator
        
    Example:
        >>> manager = BoundaryManager(boundary_map)
        >>> manager.auto_configure()  # Auto mode
        >>> manager.configure_from_yaml("config.yaml")  # Manual mode
        >>> manager.hybrid_configure("config.yaml")  # Hybrid mode
    """
    
    def __init__(self, boundary_map: Any):
        """Initialize boundary manager.
        
        Args:
            boundary_map: BoundaryMap object from grid data
            
        Raises:
            ValueError: If boundary_map is invalid
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
        """自动配置边界条件（Auto模式）
        
        Based on Properties Name keywords, automatically identifies boundary types
        and applies default parameters.
        
        Raises:
            BoundaryError: If parsing fails or no valid Properties found
            
        Example:
            >>> bc_manager = BoundaryManager(grid.boundaries)
            >>> bc_manager.auto_configure()
            >>> print(bc_manager.get_summary())
        """
        logger.info("Configuring boundaries in AUTO mode...")
        
        # Update boundary map detection mode
        self.boundary_map.detection_mode = "auto"
        
        # For each boundary, create BC instance with default parameters
        for boundary_name in self.boundary_map.boundary_names:
            bc_type = self.boundary_map.get_boundary_type(boundary_name)
            
            # Get default parameters based on BC type
            default_params = self._get_default_parameters(bc_type)
            
            # Create boundary condition instance
            try:
                bc_instance = create_boundary_condition(bc_type, **default_params)
                bc_instance.validate()
                self._bc_instances[boundary_name] = bc_instance
                
                # Store parameters in boundary map
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
        """从YAML配置文件配置边界条件（Manual模式）
        
        Completely relies on YAML configuration file. Properties not configured
        in YAML will raise an error.
        
        Args:
            config_path: Path to YAML configuration file
            
        Raises:
            FileNotFoundError: Configuration file not found
            ConfigurationError: YAML syntax error or incomplete configuration
            
        Example:
            >>> bc_manager.configure_from_yaml("boundary_config.yaml")
        """
        logger.info(f"Configuring boundaries in MANUAL mode from {config_path}...")
        
        # Load YAML configuration
        config = self._config_loader.load(config_path)
        
        # Verify mode is manual
        mode = self._config_loader.get_mode()
        if mode != 'manual':
            logger.warning(
                f"Configuration mode is '{mode}', but configure_from_yaml expects 'manual'. "
                f"Proceeding anyway."
            )
        
        # Update boundary map
        self.boundary_map.detection_mode = "manual"
        self.boundary_map.config_source = config_path
        
        # Get properties mapping from config
        properties_mapping = self._config_loader.get_properties_mapping()
        
        # Configure each property from YAML
        configured_boundaries = set()
        
        for prop_name, prop_config in properties_mapping.items():
            bc_type = prop_config['type']
            params = prop_config.get('parameters', {})
            
            # Find matching boundary in boundary_map
            boundary_name = self._find_matching_boundary(prop_name)
            
            if boundary_name is None:
                logger.warning(
                    f"Property '{prop_name}' from YAML not found in boundary map. "
                    f"Skipping."
                )
                continue
            
            # Validate parameters
            self._validate_parameters(params, bc_type, prop_name)
            
            # Create boundary condition instance
            try:
                bc_instance = create_boundary_condition(bc_type, **params)
                bc_instance.validate()
                self._bc_instances[boundary_name] = bc_instance
                
                # Store parameters in boundary map
                self.boundary_map.parameters[boundary_name] = params
                
                configured_boundaries.add(boundary_name)
                
                logger.info(
                    f"Configured {boundary_name} as {bc_type} "
                    f"(from Property '{prop_name}') with {len(self.boundary_map.get_cell_indices(boundary_name))} cells"
                )
            except Exception as e:
                logger.error(f"Failed to configure boundary '{boundary_name}': {e}")
                raise
        
        # Check if all boundaries are configured (manual mode requirement)
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
        """混合模式配置边界条件
        
        YAML configuration takes priority. Properties not configured in YAML
        are handled using auto-detection rules.
        
        Args:
            config_path: Path to YAML configuration file
            
        Raises:
            FileNotFoundError: Configuration file not found
            ConfigurationError: YAML syntax error
            
        Example:
            >>> bc_manager.hybrid_configure("partial_config.yaml")
        """
        logger.info(f"Configuring boundaries in HYBRID mode from {config_path}...")
        
        # Load YAML configuration
        config = self._config_loader.load(config_path)
        
        # Update boundary map
        self.boundary_map.detection_mode = "hybrid"
        self.boundary_map.config_source = config_path
        
        # Get properties mapping and defaults from config
        properties_mapping = self._config_loader.get_properties_mapping()
        defaults = self._config_loader.get_defaults()
        
        configured_boundaries = set()
        
        # Step 1: Configure boundaries from YAML (priority)
        for prop_name, prop_config in properties_mapping.items():
            bc_type = prop_config['type']
            params = prop_config.get('parameters', {})
            
            # Merge with defaults if available
            if bc_type.lower() in defaults:
                default_params = defaults[bc_type.lower()]
                merged_params = {**default_params, **params}
            else:
                merged_params = params
            
            # Find matching boundary in boundary_map
            boundary_name = self._find_matching_boundary(prop_name)
            
            if boundary_name is None:
                logger.warning(
                    f"Property '{prop_name}' from YAML not found in boundary map. "
                    f"Skipping."
                )
                continue
            
            # Validate parameters
            self._validate_parameters(merged_params, bc_type, prop_name)
            
            # Create boundary condition instance
            try:
                bc_instance = create_boundary_condition(bc_type, **merged_params)
                bc_instance.validate()
                self._bc_instances[boundary_name] = bc_instance
                
                # Store parameters in boundary map
                self.boundary_map.parameters[boundary_name] = merged_params
                
                configured_boundaries.add(boundary_name)
                
                logger.info(
                    f"YAML-configured {boundary_name} as {bc_type} "
                    f"(from Property '{prop_name}') with {len(self.boundary_map.get_cell_indices(boundary_name))} cells"
                )
            except Exception as e:
                logger.error(f"Failed to configure boundary '{boundary_name}': {e}")
                raise
        
        # Step 2: Auto-configure remaining boundaries
        for boundary_name in self.boundary_map.boundary_names:
            if boundary_name in configured_boundaries:
                continue
            
            # Use auto-detection to determine BC type
            prop_name = self.boundary_map.get_property_name(boundary_name)
            if prop_name:
                bc_type = self._type_mapper.map(prop_name)
            else:
                # Fallback: use existing bc_type from boundary_map
                bc_type = self.boundary_map.get_boundary_type(boundary_name)
            
            # Get default parameters
            if bc_type.lower() in defaults:
                default_params = defaults[bc_type.lower()]
            else:
                default_params = self._get_default_parameters(bc_type)
            
            # Create boundary condition instance
            try:
                bc_instance = create_boundary_condition(bc_type, **default_params)
                bc_instance.validate()
                self._bc_instances[boundary_name] = bc_instance
                
                # Store parameters in boundary map
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
        """Find matching boundary name in boundary_map for a property name
        
        Args:
            prop_name: Property name from YAML or NAS file
            
        Returns:
            Optional[str]: Matching boundary name, or None if not found
        """
        # Try exact match (case-insensitive)
        prop_name_lower = prop_name.lower()
        
        for boundary_name in self.boundary_map.boundary_names:
            if boundary_name.lower() == prop_name_lower:
                return boundary_name
        
        # Try partial match (property name contains boundary name or vice versa)
        for boundary_name in self.boundary_map.boundary_names:
            if prop_name_lower in boundary_name.lower() or boundary_name.lower() in prop_name_lower:
                return boundary_name
        
        return None
    
    def _validate_parameters(self, params: Dict[str, Any], bc_type: str, prop_name: str) -> None:
        """Validate boundary condition parameters
        
        Args:
            params: Parameters dictionary
            bc_type: Boundary condition type
            prop_name: Property name (for error messages)
            
        Raises:
            ValueError: If parameters are invalid
        """
        # Validate velocity
        if 'velocity' in params:
            self._param_validator.validate_velocity(params['velocity'])
        
        # Validate pressure
        if 'pressure' in params:
            self._param_validator.validate_pressure(params['pressure'])
        
        # Validate turbulence_intensity
        if 'turbulence_intensity' in params:
            self._param_validator.validate_turbulence_intensity(params['turbulence_intensity'])
        
        # Validate roughness_height
        if 'roughness_height' in params:
            self._param_validator.validate_roughness_height(params['roughness_height'])
    
    def _get_default_parameters(self, bc_type: str) -> Dict[str, Any]:
        """Get default parameters for a boundary condition type
        
        Args:
            bc_type: Boundary condition type
            
        Returns:
            Dict[str, Any]: Default parameters dictionary
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
        """更新指定边界的参数
        
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
        
        # Get current BC instance
        bc = self._bc_instances[boundary_name]
        bc_type = bc.get_type()
        
        # Update parameters in BC instance
        for key, value in kwargs.items():
            if hasattr(bc, key):
                setattr(bc, key, value)
            elif key in bc.params:
                bc.params[key] = value
            else:
                logger.warning(f"Parameter '{key}' not recognized for {bc_type}")
        
        # Update parameters in boundary map
        if boundary_name not in self.boundary_map.parameters:
            self.boundary_map.parameters[boundary_name] = {}
        self.boundary_map.parameters[boundary_name].update(kwargs)
        
        # Re-validate
        try:
            bc.validate()
        except ValueError as e:
            raise ValueError(f"Invalid parameters after update: {e}")
        
        logger.info(f"Updated parameters for boundary '{boundary_name}': {kwargs}")
    
    def export_to_vtk(self, output_path: str) -> None:
        """导出边界分组到VTK文件
        
        用于ParaView可视化验证边界划分正确性。
        
        Args:
            output_path: VTK输出文件路径
            
        Example:
            >>> bc_manager.export_to_vtk("boundaries.vtk")
        """
        # TODO: Implement VTK export
        logger.info(f"Exporting boundary visualization to VTK: {output_path}")
        # This will be implemented when we add postprocessing module
    
    def export_to_json(self, output_path: str) -> dict:
        """导出边界统计信息到JSON文件
        
        Args:
            output_path: JSON输出文件路径
            
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
        """生成YAML配置模板
        
        基于当前自动识别结果生成可编辑的YAML模板。
        
        Args:
            output_path: YAML模板输出路径
            
        Example:
            >>> bc_manager.generate_template("template.yaml")
            # 然后编辑template.yaml，补充/修改参数
        """
        # Build detected boundaries info
        detected = {}
        for name in self.boundary_map.boundary_names:
            detected[name] = {
                'type': self.boundary_map.get_boundary_type(name),
                'cells': len(self.boundary_map.get_cell_indices(name)),
                'property_id': self.boundary_map.get_property_id(name),
                'property_name': self.boundary_map.get_property_name(name)
            }
        
        # Generate template
        self._config_loader.generate_template(output_path, detected)
    
    def get_summary(self) -> dict:
        """获取边界配置摘要
        
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
        
        # Find boundaries without BC
        for name in self.boundary_map.boundary_names:
            if name not in self._bc_instances:
                summary['boundaries_without_bc'].append(name)
        
        # Get details for each BC
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
        """Add boundary condition to a boundary (legacy method).
        
        Args:
            boundary_name: Name of the boundary (must exist in boundary_map)
            bc_type: Boundary condition type (e.g., 'VELOCITY_INLET', 'WALL').
                    If None, inferred from boundary_name.
            **kwargs: Boundary condition parameters
            
        Returns:
            BaseBC: Created boundary condition instance
            
        Raises:
            KeyError: If boundary_name not found in boundary_map
            ValueError: If bc_type is invalid
            
        Example:
            >>> manager.add_bc("INLET", velocity_x=30.0, pressure=101325.0)
            >>> manager.add_bc("BODY", bc_type="WALL", wall_function='enhanced')
        """
        # Check if boundary exists
        if not self.boundary_map.has_boundary(boundary_name):
            raise KeyError(
                f"Boundary '{boundary_name}' not found in boundary map. "
                f"Available boundaries: {self.boundary_map.boundary_names}"
            )
        
        # Infer bc_type from boundary_name if not specified
        if bc_type is None:
            prop_name = self.boundary_map.get_property_name(boundary_name)
            if prop_name:
                bc_type = self._type_mapper.map(prop_name)
            else:
                bc_type = self.boundary_map.get_boundary_type(boundary_name)
        
        # Validate bc_type
        try:
            get_boundary_condition_class(bc_type)
        except KeyError:
            raise ValueError(
                f"Invalid boundary condition type: {bc_type}. "
                f"Use bc_type parameter to specify a valid type."
            )
        
        # Create boundary condition instance
        bc_instance = create_boundary_condition(bc_type, **kwargs)
        
        # Validate the boundary condition
        try:
            bc_instance.validate()
        except ValueError as e:
            raise ValueError(f"Invalid parameters for {boundary_name}: {e}")
        
        # Store the instance
        self._bc_instances[boundary_name] = bc_instance
        
        # Store parameters in boundary map
        self.boundary_map.parameters[boundary_name] = kwargs
        
        logger.info(f"Added {bc_type} BC to boundary '{boundary_name}'")
        
        return bc_instance
    
    def get_bc(self, boundary_name: str) -> BaseBC:
        """Get boundary condition instance for a boundary.
        
        Args:
            boundary_name: Name of the boundary
            
        Returns:
            BaseBC: Boundary condition instance
            
        Raises:
            KeyError: If no BC assigned to this boundary
        """
        if boundary_name not in self._bc_instances:
            raise KeyError(
                f"No boundary condition assigned to '{boundary_name}'. "
                f"Use add_bc() or configure methods to assign one."
            )
        
        return self._bc_instances[boundary_name]
    
    def apply_boundary(
        self,
        boundary_name: str,
        solution: Any,
        time: float = 0.0
    ) -> None:
        """Apply boundary condition to a specific boundary.
        
        Args:
            boundary_name: Name of the boundary
            solution: Solution vector
            time: Current simulation time
            
        Raises:
            KeyError: If no BC assigned to this boundary
        """
        if boundary_name not in self._bc_instances:
            raise KeyError(f"No BC assigned to boundary '{boundary_name}'")
        
        # Get boundary cells (for surface mesh, solution is stored per cell)
        boundary_cells = self.boundary_map.get_cell_indices(boundary_name)
        
        if boundary_cells.size == 0:
            logger.warning(f"Boundary '{boundary_name}' has no cells")
            return
        
        # Apply boundary condition
        bc = self._bc_instances[boundary_name]
        bc.apply(solution, boundary_cells, time)
    
    def apply_all(self, solution: Any, time: float = 0.0) -> None:
        """Apply all boundary conditions to solution.
        
        Args:
            solution: Solution vector
            time: Current simulation time
            
        Example:
            >>> manager.apply_all(solution, time=0.1)
        """
        if not self._bc_instances:
            return
            
        for boundary_name in self._bc_instances:
            try:
                self.apply_boundary(boundary_name, solution, time)
            except Exception as e:
                logger.error(
                    f"Failed to apply BC to boundary '{boundary_name}': {e}"
                )
                raise
    
    def update_time_dependent_bcs(self, time: float) -> None:
        """Update time-dependent boundary conditions.
        
        For boundaries with time-varying conditions, update their parameters
        based on current simulation time.
        
        Args:
            time: Current simulation time
        """
        # TODO: Implement time-dependent BC updates
        # This would involve checking if BCs have time-dependent parameters
        # and updating them accordingly
        logger.debug(f"Updating time-dependent BCs at t={time:.6f}s")
        pass
    
    def validate_all(self) -> bool:
        """Validate all boundary conditions.
        
        Returns:
            bool: True if all BCs are valid
            
        Raises:
            ValueError: If any BC is invalid
        """
        for boundary_name, bc in self._bc_instances.items():
            try:
                bc.validate()
            except ValueError as e:
                raise ValueError(f"Invalid BC for '{boundary_name}': {e}")
        
        logger.info("All boundary conditions validated successfully")
        return True
    
    def list_boundaries(self) -> List[str]:
        """List all boundary names.
        
        Returns:
            List[str]: List of boundary names
        """
        return self.boundary_map.boundary_names
    
    def list_assigned_bcs(self) -> List[str]:
        """List boundaries with assigned BCs.
        
        Returns:
            List[str]: List of boundary names with BCs
        """
        return list(self._bc_instances.keys())
    
    def remove_bc(self, boundary_name: str) -> None:
        """Remove boundary condition from a boundary.
        
        Args:
            boundary_name: Name of the boundary
            
        Raises:
            KeyError: If no BC assigned to this boundary
        """
        if boundary_name not in self._bc_instances:
            raise KeyError(f"No BC assigned to boundary '{boundary_name}'")
        
        del self._bc_instances[boundary_name]
        logger.info(f"Removed BC from boundary '{boundary_name}'")
    
    def clear_all(self) -> None:
        """Remove all boundary conditions."""
        self._bc_instances.clear()
        logger.info("Cleared all boundary conditions")
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"BoundaryManager("
            f"boundaries={len(self.boundary_map.boundary_names)}, "
            f"assigned={len(self._bc_instances)}, "
            f"mode={self.boundary_map.detection_mode})"
        )
    
    def __len__(self) -> int:
        """Return number of assigned BCs."""
        return len(self._bc_instances)