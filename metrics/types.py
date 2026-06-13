# metrics/types.py
# ==============================================================================
# Module: metrics.types
# Purpose: Data structures for metric computation results
#
# Defines dataclasses:
#   - MetricValue: Single metric value with components
#   - MetricResult: Complete metric computation result
#
# Author: Contact Analysis Team
# ==============================================================================

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

import numpy as np

from .. import ValidationError, DataTypeError

logger = logging.getLogger(__name__)


# ==============================================================================
# METRIC VALUE: Single metric value with components
# ==============================================================================

@dataclass
class MetricValue:
    """
    Single metric value with optional component breakdown.
    
    Represents a computed metric value, potentially with breakdown into
    components (e.g., x, y, z components of a distance vector).
    
    Attributes
    ----------
    value : float
        Scalar metric value
    components : Dict[str, float], optional
        Component breakdown (e.g., {'x': 1.0, 'y': 2.0, 'z': 3.0})
    uncertainty : float, optional
        Estimated uncertainty in value
    metadata : Dict[str, Any], optional
        Additional data about computation
    
    Raises
    ------
    DataTypeError
        If value or components are invalid
    """
    
    value: float
    components: Dict[str, float] = field(default_factory=dict)
    uncertainty: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate metric value after initialization."""
        # Validate value
        if not isinstance(self.value, (int, float)):
            raise DataTypeError(
                f"value must be numeric, got {type(self.value).__name__}",
                error_code="METRIC_VALUE_TYPE_ERROR"
            )
        
        if np.isnan(self.value) or np.isinf(self.value):
            raise DataTypeError(
                f"value contains NaN or Inf: {self.value}",
                error_code="METRIC_VALUE_NAN_INF"
            )
        
        # Validate components
        for comp_name, comp_val in self.components.items():
            if not isinstance(comp_val, (int, float)):
                raise DataTypeError(
                    f"component '{comp_name}' must be numeric, got {type(comp_val).__name__}",
                    error_code="METRIC_COMPONENT_TYPE_ERROR"
                )
            
            if np.isnan(comp_val) or np.isinf(comp_val):
                raise DataTypeError(
                    f"component '{comp_name}' contains NaN or Inf",
                    error_code="METRIC_COMPONENT_NAN_INF"
                )
        
        # Validate uncertainty
        if self.uncertainty is not None:
            if self.uncertainty < 0:
                raise ValidationError(
                    f"uncertainty must be non-negative, got {self.uncertainty}",
                    error_code="METRIC_NEGATIVE_UNCERTAINTY"
                )
        
        logger.debug(f"MetricValue created: {self.value:.6f}")
    
    @property
    def magnitude(self) -> float:
        """Magnitude of vector components (if they exist)."""
        if not self.components:
            return abs(self.value)
        
        return np.sqrt(sum(v**2 for v in self.components.values()))
    
    def __repr__(self) -> str:
        """String representation."""
        if self.components:
            return (
                f"MetricValue({self.value:.6f}, "
                f"components={list(self.components.keys())})"
            )
        else:
            return f"MetricValue({self.value:.6f})"


# ==============================================================================
# METRIC RESULT: Complete metric computation result
# ==============================================================================

@dataclass
class MetricResult:
    """
    Complete metric computation result.
    
    Contains results for a single pair of entities (particles, faces, vertices).
    
    Attributes
    ----------
    metric_name : str
        Name of metric (e.g., 'face_center_face_center')
    particle_A_id : int
        ID of first particle (or -1 if N/A)
    particle_B_id : int
        ID of second particle (or -1 if N/A)
    metric_values : Dict[str, MetricValue]
        Results: {component_name: MetricValue}
    is_valid : bool
        Whether metric computation succeeded
    error_message : Optional[str]
        Error message if computation failed
    frame_index : int, optional
        Frame number in trajectory
    metadata : Dict[str, Any], optional
        Additional metadata
    
    Raises
    ------
    ValidationError
        If particle IDs or metric_name invalid
    """
    
    metric_name: str
    particle_A_id: int
    particle_B_id: int
    metric_values: Dict[str, MetricValue]
    is_valid: bool
    error_message: Optional[str] = None
    frame_index: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate metric result after initialization."""
        # Validate metric_name
        if not isinstance(self.metric_name, str) or not self.metric_name:
            raise ValidationError(
                f"metric_name must be non-empty string",
                error_code="METRIC_RESULT_INVALID_NAME"
            )
        
        # Validate particle IDs
        if self.particle_A_id < -1 or self.particle_B_id < -1:
            raise ValidationError(
                f"Particle IDs must be >= -1",
                error_code="METRIC_RESULT_INVALID_PARTICLE_ID"
            )
        
        # Validate frame_index
        if self.frame_index < 0:
            raise ValidationError(
                f"frame_index must be non-negative",
                error_code="METRIC_RESULT_NEGATIVE_FRAME_INDEX"
            )
        
        # Validate metric_values
        for val_name, val in self.metric_values.items():
            if not isinstance(val, MetricValue):
                raise DataTypeError(
                    f"metric_values['{val_name}'] must be MetricValue",
                    error_code="METRIC_RESULT_VALUE_TYPE_ERROR"
                )
        
        # If invalid, require error message
        if not self.is_valid and not self.error_message:
            logger.warning("Invalid result without error message")
        
        logger.debug(
            f"MetricResult created: {self.metric_name} "
            f"({self.particle_A_id},{self.particle_B_id}), "
            f"valid={self.is_valid}"
        )
    
    @property
    def num_components(self) -> int:
        """Number of metric components."""
        return len(self.metric_values)
    
    def get_value(self, component: str = "value") -> Optional[float]:
        """
        Get metric value by component.
        
        Parameters
        ----------
        component : str, optional
            Component name (default: 'value')
        
        Returns
        -------
        float or None
            Value if available, None otherwise
        """
        
        if component not in self.metric_values:
            return None
        
        return self.metric_values[component].value
    
    def get_component(self, name: str) -> Optional[MetricValue]:
        """Get metric value object by name."""
        return self.metric_values.get(name)
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"MetricResult({self.metric_name}, "
            f"({self.particle_A_id},{self.particle_B_id}), "
            f"valid={self.is_valid})"
        )


if __name__ == "__main__":
    """Test when run directly."""
    print("\n" + "="*80)
    print("METRICS TYPES MODULE - TESTING")
    print("="*80 + "\n")
    
    print("[TEST 1] Creating MetricValue...")
    try:
        mv = MetricValue(
            value=3.5,
            components={'x': 1.0, 'y': 2.0, 'z': 2.5},
            uncertainty=0.01
        )
        
        print(f"✓ {mv}")
        print(f"  Magnitude: {mv.magnitude:.3f}\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")
    
    print("[TEST 2] Creating MetricResult...")
    try:
        result = MetricResult(
            metric_name="face_center_face_center",
            particle_A_id=0,
            particle_B_id=1,
            metric_values={
                "distance": MetricValue(value=3.5, components={'x': 1.0, 'y': 2.0, 'z': 2.5}),
                "contact_order": MetricValue(value=2.0)
            },
            is_valid=True,
            frame_index=0
        )
        
        print(f"✓ {result}")
        print(f"  Components: {result.num_components}")
        print(f"  Distance: {result.get_value('distance'):.3f}\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")
    
    print("[TEST 3] Invalid MetricValue (NaN)...")
    try:
        bad_mv = MetricValue(value=float('nan'))
    except DataTypeError as e:
        print(f"✓ Correctly caught: {e.error_code}\n")
    
    print("="*80)
    print("✓ All tests passed!")
    print("="*80 + "\n")
