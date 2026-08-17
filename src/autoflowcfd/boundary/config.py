"""边界条件配置加载器。

本模块提供边界条件的 YAML 配置加载功能，
支持自动、手动和混合模式。

核心组件:
    - YAMLConfigLoader: 加载并验证 YAML 边界配置
    - BoundaryTypeMapper: 属性名到边界类型的映射
    - ParameterValidator: 边界条件参数验证

示例:
    >>> from autoflowcfd.boundary.config import YAMLConfigLoader
    >>> loader = YAMLConfigLoader()
    >>> config = loader.load("boundary_config.yaml")
    >>> print(config['properties_mapping'])
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
import yaml
from loguru import logger

# BoundaryTypeMapper 和 ParameterValidator 已拆分到 config_validators.py
from .config_validators import BoundaryTypeMapper, ParameterValidator  # noqa: F401


class ConfigurationError(Exception):
    """配置加载或验证错误。"""
    pass


class YAMLConfigLoader:
    """YAML配置文件加载器
    
    Loads and validates boundary condition configurations from YAML files.
    Supports three modes: auto, manual, hybrid.
    
    Attributes:
        config: Loaded configuration dictionary
        
    Example:
        >>> loader = YAMLConfigLoader()
        >>> config = loader.load("boundary_config.yaml")
        >>> print(config['mode'])  # 'hybrid'
    """
    
    def __init__(self):
        """Initialize configuration loader."""
        self.config: Optional[Dict[str, Any]] = None
    
    def load(self, config_path: str) -> Dict[str, Any]:
        """Load YAML configuration file
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            Dict[str, Any]: Configuration dictionary
            
        Raises:
            FileNotFoundError: Configuration file not found
            ConfigurationError: Invalid YAML syntax or structure
            
        Example:
            >>> loader = YAMLConfigLoader()
            >>> config = loader.load("config.yaml")
        """
        path = Path(config_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML syntax in {config_path}: {e}")
        
        # Validate configuration structure
        self._validate_config(config, config_path)
        
        self.config = config
        logger.info(f"Loaded boundary configuration from {config_path}")
        
        return config
    
    def _validate_config(self, config: Dict[str, Any], config_path: str) -> None:
        """Validate configuration structure and values
        
        Args:
            config: Configuration dictionary
            config_path: Path to configuration file (for error messages)
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        if not isinstance(config, dict):
            raise ConfigurationError(
                f"Configuration must be a dictionary, got {type(config).__name__}"
            )
        
        # Check required top-level keys
        if 'boundary_detection' not in config:
            raise ConfigurationError(
                f"Missing required key 'boundary_detection' in {config_path}"
            )
        
        detection_config = config['boundary_detection']
        
        if not isinstance(detection_config, dict):
            raise ConfigurationError(
                f"'boundary_detection' must be a dictionary in {config_path}"
            )
        
        # Validate mode
        if 'mode' not in detection_config:
            raise ConfigurationError(
                f"Missing required key 'mode' in boundary_detection section"
            )
        
        mode = detection_config['mode']
        valid_modes = ['auto', 'manual', 'hybrid']
        
        if mode not in valid_modes:
            raise ConfigurationError(
                f"Invalid mode '{mode}'. Must be one of {valid_modes}"
            )
        
        # For manual/hybrid modes, validate properties_mapping
        if mode in ['manual', 'hybrid']:
            if 'properties_mapping' not in config:
                raise ConfigurationError(
                    f"Missing required key 'properties_mapping' for {mode} mode"
                )
            
            self._validate_properties_mapping(
                config['properties_mapping'],
                config_path
            )
        
        # Validate defaults if present
        if 'defaults' in config:
            self._validate_defaults(config['defaults'], config_path)
    
    def _validate_properties_mapping(
        self,
        mapping: Dict[str, Any],
        config_path: str
    ) -> None:
        """Validate properties_mapping section
        
        Args:
            mapping: Properties mapping dictionary
            config_path: Path to configuration file
            
        Raises:
            ConfigurationError: If mapping is invalid
        """
        if not isinstance(mapping, dict):
            raise ConfigurationError(
                f"'properties_mapping' must be a dictionary in {config_path}"
            )
        
        valid_bc_types = [
            'VELOCITY_INLET',
            'PRESSURE_OUTLET',
            'WALL',
            'SYMMETRY',
            'SLIP_WALL',
            'PERIODIC',
        ]
        
        for prop_name, prop_config in mapping.items():
            if not isinstance(prop_config, dict):
                raise ConfigurationError(
                    f"Property '{prop_name}' configuration must be a dictionary"
                )
            
            if 'type' not in prop_config:
                raise ConfigurationError(
                    f"Property '{prop_name}' missing required 'type' field"
                )
            
            bc_type = prop_config['type']
            if bc_type not in valid_bc_types:
                raise ConfigurationError(
                    f"Property '{prop_name}' has invalid type '{bc_type}'. "
                    f"Must be one of {valid_bc_types}"
                )
            
            # Validate parameters if present
            if 'parameters' in prop_config:
                self._validate_boundary_parameters(
                    prop_config['parameters'],
                    bc_type,
                    prop_name
                )

            # PERIODIC 必须显式给出配对的另一侧组名和平移向量——这两项
            # 无法从几何/NAS 文件自动反推（见 grid/nas_io/nas_parser_boundary.py
            # 关键字表旁的说明），跟 'type' 一样是这个 bc_type 下的必填项。
            if bc_type == 'PERIODIC':
                params = prop_config.get('parameters', {})
                if 'paired_with' not in params or 'translation' not in params:
                    raise ConfigurationError(
                        f"Property '{prop_name}' has type 'PERIODIC' but is missing "
                        f"'paired_with' and/or 'translation' under 'parameters' - both "
                        f"are required (cannot be auto-detected from geometry)."
                    )
    
    def _validate_boundary_parameters(
        self,
        params: Dict[str, Any],
        bc_type: str,
        prop_name: str
    ) -> None:
        """Validate boundary condition parameters
        
        Args:
            params: Parameters dictionary
            bc_type: Boundary condition type
            prop_name: Property name (for error messages)
            
        Raises:
            ConfigurationError: If parameters are invalid
        """
        # Validate velocity parameter
        if 'velocity' in params:
            velocity = params['velocity']
            if not isinstance(velocity, (list, tuple)) or len(velocity) != 3:
                raise ConfigurationError(
                    f"Property '{prop_name}': 'velocity' must be a list of 3 floats"
                )
            try:
                velocity = [float(v) for v in velocity]
            except (TypeError, ValueError):
                raise ConfigurationError(
                    f"Property '{prop_name}': 'velocity' values must be numeric"
                )
        
        # Validate pressure parameter
        if 'pressure' in params:
            try:
                pressure = float(params['pressure'])
            except (TypeError, ValueError):
                raise ConfigurationError(
                    f"Property '{prop_name}': 'pressure' must be numeric"
                )
        
        # Validate turbulence_intensity
        if 'turbulence_intensity' in params:
            try:
                ti = float(params['turbulence_intensity'])
                if not (0.0 < ti < 1.0):
                    logger.warning(
                        f"Property '{prop_name}': turbulence_intensity={ti} "
                        f"is outside typical range (0.001-0.2)"
                    )
            except (TypeError, ValueError):
                raise ConfigurationError(
                    f"Property '{prop_name}': 'turbulence_intensity' must be numeric"
                )
        
        # Validate wall_function
        if 'wall_function' in params:
            valid_functions = ['standard', 'enhanced', 'moving_wall']
            if params['wall_function'] not in valid_functions:
                raise ConfigurationError(
                    f"Property '{prop_name}': 'wall_function' must be one of {valid_functions}"
                )
        
        # Validate roughness_height
        if 'roughness_height' in params:
            try:
                rh = float(params['roughness_height'])
                if rh < 0:
                    raise ConfigurationError(
                        f"Property '{prop_name}': 'roughness_height' must be non-negative"
                    )
            except (TypeError, ValueError):
                raise ConfigurationError(
                    f"Property '{prop_name}': 'roughness_height' must be numeric"
                )

        # Validate PERIODIC pairing parameters
        if 'paired_with' in params:
            if not isinstance(params['paired_with'], str) or not params['paired_with']:
                raise ConfigurationError(
                    f"Property '{prop_name}': 'paired_with' must be a non-empty boundary group name"
                )
        if 'translation' in params:
            translation = params['translation']
            if not isinstance(translation, (list, tuple)) or len(translation) != 3:
                raise ConfigurationError(
                    f"Property '{prop_name}': 'translation' must be a list of 3 floats"
                )
            try:
                [float(v) for v in translation]
            except (TypeError, ValueError):
                raise ConfigurationError(
                    f"Property '{prop_name}': 'translation' values must be numeric"
                )
    
    def _validate_defaults(
        self,
        defaults: Dict[str, Any],
        config_path: str
    ) -> None:
        """Validate defaults section
        
        Args:
            defaults: Defaults dictionary
            config_path: Path to configuration file
            
        Raises:
            ConfigurationError: If defaults are invalid
        """
        if not isinstance(defaults, dict):
            raise ConfigurationError(
                f"'defaults' must be a dictionary in {config_path}"
            )
        
        # Validate each default section
        for bc_type, params in defaults.items():
            if not isinstance(params, dict):
                raise ConfigurationError(
                    f"Default parameters for '{bc_type}' must be a dictionary"
                )
            
            # Reuse parameter validation logic
            self._validate_boundary_parameters(params, bc_type, f"default_{bc_type}")
    
    def get_mode(self) -> str:
        """Get configuration mode
        
        Returns:
            str: Configuration mode ('auto', 'manual', or 'hybrid')
            
        Raises:
            ConfigurationError: If configuration not loaded
        """
        if self.config is None:
            raise ConfigurationError("Configuration not loaded. Call load() first.")
        
        return self.config['boundary_detection']['mode']
    
    def get_properties_mapping(self) -> Dict[str, Any]:
        """Get properties mapping from configuration
        
        Returns:
            Dict[str, Any]: Properties mapping dictionary
            
        Raises:
            ConfigurationError: If configuration not loaded or mode is auto
        """
        if self.config is None:
            raise ConfigurationError("Configuration not loaded. Call load() first.")
        
        mode = self.get_mode()
        
        if mode == 'auto':
            raise ConfigurationError(
                "Auto mode does not have properties_mapping. "
                "Use manual or hybrid mode."
            )
        
        return self.config.get('properties_mapping', {})
    
    def get_defaults(self) -> Dict[str, Any]:
        """Get default parameters from configuration
        
        Returns:
            Dict[str, Any]: Default parameters dictionary
        """
        if self.config is None:
            raise ConfigurationError("Configuration not loaded. Call load() first.")
        
        return self.config.get('defaults', {})
    
    def generate_template(self, output_path: str, detected_boundaries: Dict[str, Any]) -> None:
        """Generate YAML configuration template based on detected boundaries
        
        Args:
            output_path: Path to output YAML file
            detected_boundaries: Dictionary of detected boundaries with their info
            
        Example:
            >>> detected = {
            ...     'INLET': {'type': 'VELOCITY_INLET', 'cells': 500},
            ...     'OUTLET': {'type': 'PRESSURE_OUTLET', 'cells': 500}
            ... }
            >>> loader.generate_template("template.yaml", detected)
        """
        template = {
            'boundary_detection': {
                'mode': 'hybrid'
            },
            'properties_mapping': {},
            'defaults': {
                'velocity_inlet': {
                    'velocity': [33.33, 0.0, 0.0],
                    'turbulence_intensity': 0.05
                },
                'pressure_outlet': {
                    'pressure': 0.0
                },
                'wall': {
                    'wall_function': 'standard'
                }
            }
        }
        
        # Add detected boundaries to template
        for prop_name, info in detected_boundaries.items():
            template['properties_mapping'][prop_name] = {
                'type': info['type'],
                'parameters': {}
            }
        
        # Write template to file
        path = Path(output_path)
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(template, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"Generated boundary configuration template: {output_path}")

