"""Unit tests for boundary condition module."""

import numpy as np
import pytest

from autoflowcfd.boundary import (
    BaseBC,
    BoundaryManager,
    InletBC,
    OutletBC,
    WallBC,
    GroundBC,
    FarfieldBC,
    SymmetryBC,
    BodyBC,
    register_boundary_condition,
    create_boundary_condition,
)
from autoflowcfd.grid.structures import BoundaryMap


def make_boundary_map() -> BoundaryMap:
    """Build a real BoundaryMap fixture matching its actual dataclass
    interface (groups/bc_types/property_ids/parameters), instead of a
    hand-maintained mock that has to be kept in sync by hand - a
    previous mock here only implemented a few of the methods
    BoundaryManager actually calls (get_cell_indices, parameters,
    detection_mode, boundary_count, config_source, ...), so most
    BoundaryManager tests failed with AttributeError against the mock
    rather than testing real behaviour."""
    groups = {
        "INLET": np.array([0, 1, 2], dtype=np.int32),
        "OUTLET": np.array([3, 4, 5], dtype=np.int32),
        "BODY": np.arange(6, 20, dtype=np.int32),
        "GROUND": np.array([20, 21, 22], dtype=np.int32),
        "FARFIELD": np.array([23, 24, 25], dtype=np.int32),
        "SYMMETRY": np.array([26, 27], dtype=np.int32),
    }
    # bc_types matches each boundary's own name (what add_bc() falls back
    # to when a boundary has no recorded property_name to run through
    # BoundaryTypeMapper) - matching what a NAS-derived BoundaryMap would
    # actually record for boundaries named this way.
    bc_types = {name: name for name in groups}
    return BoundaryMap(groups=groups, bc_types=bc_types)


class TestInletBC:
    """Test suite for InletBC."""

    def test_creation_with_defaults(self):
        """Test creating InletBC with default parameters."""
        inlet = InletBC()
        assert inlet.bc_type == "INLET"
        assert inlet.params['velocity_x'] == 30.0
        assert inlet.params['pressure'] == 101325.0

    def test_creation_with_custom_params(self):
        """Test creating InletBC with custom parameters."""
        inlet = InletBC(
            velocity_x=40.0,
            velocity_y=5.0,
            pressure=100000.0,
            turbulence_k=0.5
        )
        assert inlet.params['velocity_x'] == 40.0
        assert inlet.params['velocity_y'] == 5.0
        assert inlet.params['pressure'] == 100000.0
        assert inlet.params['turbulence_k'] == 0.5

    def test_validate_success(self):
        """Test validation with valid parameters."""
        inlet = InletBC()
        assert inlet.validate() is True

    def test_validate_negative_pressure(self):
        """Test validation fails with negative pressure."""
        with pytest.raises(ValueError, match="Pressure must be positive"):
            inlet = InletBC(pressure=-100.0)
            inlet.validate()

    def test_validate_negative_turbulence_k(self):
        """Test validation fails with negative turbulence k."""
        with pytest.raises(ValueError, match="Turbulence k must be non-negative"):
            inlet = InletBC(turbulence_k=-1.0)
            inlet.validate()

    def test_validate_zero_omega(self):
        """Test validation fails with zero omega."""
        with pytest.raises(ValueError, match="Turbulence omega must be positive"):
            inlet = InletBC(turbulence_omega=0.0)
            inlet.validate()

    def test_repr(self):
        """Test string representation."""
        inlet = InletBC()
        assert "InletBC" in repr(inlet)
        assert "INLET" in repr(inlet)


class TestOutletBC:
    """Test suite for OutletBC."""

    def test_creation_with_defaults(self):
        """Test creating OutletBC with default parameters."""
        outlet = OutletBC()
        assert outlet.bc_type == "OUTLET"
        assert outlet.params['pressure'] == 101325.0

    def test_validate_success(self):
        """Test validation with valid parameters."""
        outlet = OutletBC()
        assert outlet.validate() is True

    def test_validate_negative_pressure(self):
        """Test validation fails with negative pressure."""
        with pytest.raises(ValueError, match="Pressure must be positive"):
            outlet = OutletBC(pressure=-100.0)
            outlet.validate()


class TestWallBC:
    """Test suite for WallBC."""

    def test_creation_with_defaults(self):
        """Test creating WallBC with default parameters."""
        wall = WallBC()
        assert wall.bc_type == "WALL"
        assert wall.params['wall_function'] == 'standard'

    def test_creation_with_enhanced_wall_function(self):
        """Test creating WallBC with enhanced wall function."""
        wall = WallBC(wall_function='enhanced')
        assert wall.params['wall_function'] == 'enhanced'

    def test_invalid_wall_function(self):
        """Test creation fails with invalid wall function."""
        with pytest.raises(ValueError, match="Invalid wall function"):
            WallBC(wall_function='invalid')

    def test_validate_success(self):
        """Test validation with valid parameters."""
        wall = WallBC()
        assert wall.validate() is True

    def test_validate_negative_roughness(self):
        """Test validation fails with negative roughness."""
        with pytest.raises(ValueError, match="Roughness height must be non-negative"):
            wall = WallBC(roughness_height=-0.001)
            wall.validate()


class TestGroundBC:
    """Test suite for GroundBC."""

    def test_stationary_ground(self):
        """Test creating stationary ground."""
        ground = GroundBC(moving=False)
        assert ground.bc_type == "GROUND"
        assert ground.params['moving'] is False

    def test_moving_ground(self):
        """Test creating moving ground."""
        ground = GroundBC(moving=True, velocity_x=30.0)
        assert ground.params['moving'] is True
        assert ground.params['velocity_x'] == 30.0

    def test_validate_success(self):
        """Test validation with valid parameters."""
        ground = GroundBC()
        assert ground.validate() is True


class TestFarfieldBC:
    """Test suite for FarfieldBC."""

    def test_creation_with_defaults(self):
        """Test creating FarfieldBC with default parameters."""
        farfield = FarfieldBC()
        assert farfield.bc_type == "FARFIELD"
        assert farfield.params['pressure'] == 101325.0
        assert farfield.params['temperature'] == 288.15

    def test_validate_success(self):
        """Test validation with valid parameters."""
        farfield = FarfieldBC()
        assert farfield.validate() is True

    def test_validate_negative_pressure(self):
        """Test validation fails with negative pressure."""
        with pytest.raises(ValueError, match="Pressure must be positive"):
            farfield = FarfieldBC(pressure=-100.0)
            farfield.validate()

    def test_validate_negative_temperature(self):
        """Test validation fails with negative temperature."""
        with pytest.raises(ValueError, match="Temperature must be positive"):
            farfield = FarfieldBC(temperature=-10.0)
            farfield.validate()


class TestSymmetryBC:
    """Test suite for SymmetryBC."""

    def test_creation(self):
        """Test creating SymmetryBC."""
        symmetry = SymmetryBC()
        assert symmetry.bc_type == "SYMMETRY"

    def test_validate_success(self):
        """Test validation always succeeds."""
        symmetry = SymmetryBC()
        assert symmetry.validate() is True


class TestBodyBC:
    """Test suite for BodyBC."""

    def test_creation_with_defaults(self):
        """Test creating BodyBC with default parameters."""
        body = BodyBC()
        assert body.bc_type == "BODY"
        assert body.params['wall_function'] == 'enhanced'

    def test_inherits_from_wall(self):
        """Test that BodyBC inherits from WallBC."""
        body = BodyBC()
        assert isinstance(body, WallBC)


class TestBoundaryManager:
    """Test suite for BoundaryManager.

    BoundaryManager now only handles boundary-condition metadata
    registration (add_bc/get_bc/validate_all/...) - apply_all()/
    apply_boundary() were removed as dead code (see conditions.py's
    module docstring): the live solve path never called them, it
    reads registered metadata through a completely separate,
    vectorized implementation in core/bc_handler.py instead.
    """

    def setup_method(self):
        """Set up test fixtures."""
        self.bmap = make_boundary_map()
        self.manager = BoundaryManager(self.bmap)

    def test_initialization(self):
        """Test manager initialization."""
        assert len(self.manager.list_boundaries()) == 6
        assert len(self.manager.list_assigned_bcs()) == 0

    def test_add_inlet_bc(self):
        """Test adding inlet BC."""
        bc = self.manager.add_bc("INLET", velocity_x=30.0)
        assert isinstance(bc, InletBC)
        assert "INLET" in self.manager.list_assigned_bcs()

    def test_add_outlet_bc(self):
        """Test adding outlet BC."""
        bc = self.manager.add_bc("OUTLET", pressure=101325.0)
        assert isinstance(bc, OutletBC)

    def test_add_body_bc(self):
        """Test adding body BC."""
        bc = self.manager.add_bc("BODY", wall_function='enhanced')
        assert isinstance(bc, BodyBC)

    def test_add_bc_auto_type_inference(self):
        """Test automatic BC type inference from boundary name.

        make_boundary_map() records each boundary's own name as its
        bc_types entry (no NAS Properties-Name based auto-detection in
        this fixture) - add_bc() with no explicit bc_type falls back to
        that recorded type when the boundary has no property name.
        """
        bc = self.manager.add_bc("INLET", velocity_x=30.0)
        assert bc.get_type() == "INLET"

        bc = self.manager.add_bc("GROUND")
        assert bc.get_type() == "GROUND"

    def test_add_bc_explicit_type(self):
        """Test explicit BC type specification."""
        bc = self.manager.add_bc("INLET", bc_type="WALL")
        assert isinstance(bc, WallBC)

    def test_add_bc_nonexistent_boundary(self):
        """Test adding BC to nonexistent boundary raises error."""
        with pytest.raises(KeyError, match="not found in boundary map"):
            self.manager.add_bc("NONEXISTENT")

    def test_add_bc_invalid_type(self):
        """Test adding BC with invalid type raises error."""
        with pytest.raises(ValueError, match="Invalid boundary condition type"):
            self.manager.add_bc("INLET", bc_type="INVALID_TYPE")

    def test_get_bc(self):
        """Test getting BC instance."""
        self.manager.add_bc("INLET", velocity_x=30.0)
        bc = self.manager.get_bc("INLET")
        assert isinstance(bc, InletBC)

    def test_get_bc_not_assigned(self):
        """Test getting unassigned BC raises error."""
        with pytest.raises(KeyError, match="No boundary condition assigned"):
            self.manager.get_bc("INLET")

    def test_validate_all(self):
        """Test validating all BCs."""
        self.manager.add_bc("INLET", velocity_x=30.0)
        self.manager.add_bc("OUTLET", pressure=101325.0)

        assert self.manager.validate_all() is True

    def test_get_summary(self):
        """Test getting summary."""
        self.manager.add_bc("INLET", velocity_x=30.0)
        self.manager.add_bc("BODY")

        summary = self.manager.get_summary()
        assert summary['total_boundaries'] == 6
        assert summary['boundaries_with_bc'] == 2
        assert len(summary['boundaries_without_bc']) == 4
        assert "INLET" in summary['bc_details']
        assert "BODY" in summary['bc_details']

    def test_remove_bc(self):
        """Test removing BC."""
        self.manager.add_bc("INLET", velocity_x=30.0)
        assert "INLET" in self.manager.list_assigned_bcs()

        self.manager.remove_bc("INLET")
        assert "INLET" not in self.manager.list_assigned_bcs()

    def test_clear_all(self):
        """Test clearing all BCs."""
        self.manager.add_bc("INLET")
        self.manager.add_bc("OUTLET")

        self.manager.clear_all()
        assert len(self.manager.list_assigned_bcs()) == 0

    def test_len(self):
        """Test length operator."""
        assert len(self.manager) == 0

        self.manager.add_bc("INLET")
        assert len(self.manager) == 1

    def test_repr(self):
        """Test string representation."""
        self.manager.add_bc("INLET")
        repr_str = repr(self.manager)
        assert "BoundaryManager" in repr_str
        assert "boundaries=6" in repr_str
        assert "assigned=1" in repr_str


class TestBCRegistry:
    """Test suite for BC registry and factory."""

    def test_create_builtin_bc(self):
        """Test creating built-in BC via factory."""
        bc = create_boundary_condition("INLET", velocity_x=30.0)
        assert isinstance(bc, InletBC)

    def test_create_wall_bc(self):
        """Test creating wall BC via factory."""
        bc = create_boundary_condition("WALL", wall_function='standard')
        assert isinstance(bc, WallBC)

    def test_create_unknown_bc(self):
        """Test creating unknown BC raises error."""
        with pytest.raises(KeyError, match="Unknown boundary condition type"):
            create_boundary_condition("UNKNOWN_TYPE")

    def test_register_custom_bc(self):
        """Test registering custom BC.

        Custom BC classes must inherit from BaseBC (register_boundary_condition
        enforces this - see conditions.py) - only validate() is required
        now that apply() has been removed from BaseBC's abstract contract.
        """
        @register_boundary_condition("CUSTOM_TEST")
        class CustomTestBC(BaseBC):
            def __init__(self, **kwargs):
                super().__init__("CUSTOM_TEST", kwargs)

            def validate(self):
                return True

        bc = create_boundary_condition("CUSTOM_TEST")
        assert isinstance(bc, CustomTestBC)

    def test_register_invalid_class(self):
        """Test registering non-BC class raises error."""
        with pytest.raises(TypeError, match="must inherit from BaseBC"):
            @register_boundary_condition("INVALID")
            class NotABC:
                pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
