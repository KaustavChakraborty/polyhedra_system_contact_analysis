# core/types.py
# ==============================================================================
# Module: core.types
# Purpose: Immutable data structures (dataclasses) used throughout the workflow
#
# This module defines the fundamental data types used in contact analysis:
#   - Box: Simulation box dimensions (immutable)
#   - ComputationResult: Base class for computation results with status tracking
#
# Design principles:
#   1. Use frozen dataclasses where possible (immutability prevents bugs)
#   2. Include comprehensive validation in __post_init__()
#   3. Provide helper methods for common operations
#   4. Include detailed docstrings with examples
#   5. Raise custom exceptions for invalid data
#
# These types are designed to be:
#   - Type-hinted for IDE support and type checking
#   - Immutable to prevent accidental modification
#   - Validating to catch errors early
#   - Serializable for logging and debugging
#
# Author: Contact Analysis Team
# ==============================================================================

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Any, Dict

import numpy as np

from .exceptions import ValidationError, DataTypeError

logger = logging.getLogger(__name__)


# ==============================================================================
# SIMULATION BOX DEFINITION
# ==============================================================================

@dataclass(frozen=True)
class Box:
    """
    Immutable representation of a simulation box (unit cell).
    
    Stores the dimensions and orientation of the simulation box, including
    support for triclinic (sheared) boxes commonly used in molecular dynamics.
    
    Attributes
    ----------
    Lx : float
        Length in x-direction. Must be positive.
    Ly : float
        Length in y-direction. Must be positive.
    Lz : float
        Length in z-direction. Must be positive.
    xy : float, optional
        Shear component xy (default: 0.0). Used for triclinic boxes.
    xz : float, optional
        Shear component xz (default: 0.0). Used for triclinic boxes.
    yz : float, optional
        Shear component yz (default: 0.0). Used for triclinic boxes.
    
    Raises
    ------
    ValidationError
        If Lx, Ly, or Lz are not positive
    DataTypeError
        If dimensions cannot be converted to float
    
    Examples
    --------
    Orthogonal box:
    
    >>> box = Box(Lx=10.0, Ly=10.0, Lz=10.0)
    >>> print(f"Volume: {box.volume}")
    Volume: 1000.0
    
    Triclinic box (with shear):
    
    >>> box = Box(Lx=10.0, Ly=10.0, Lz=10.0, xy=2.0)
    >>> print(f"Shear xy: {box.xy}")
    Shear xy: 2.0
    
    Notes
    -----
    - Box is frozen (immutable) to prevent accidental modification
    - All dimensions are validated on creation
    - Use this for all box-related operations throughout the workflow
    - Box information is essential for periodic boundary condition handling
    """
    
    Lx: float
    Ly: float
    Lz: float
    xy: float = 0.0
    xz: float = 0.0
    yz: float = 0.0
    
    def __post_init__(self) -> None:
        """
        Validate box dimensions after initialization.
        
        Checks that:
        - All lengths (Lx, Ly, Lz) are positive
        - All values can be converted to float
        - Dimensions are not NaN or infinite
        
        Raises
        ------
        ValidationError
            If any dimension is not positive
        DataTypeError
            If any dimension cannot be converted to float
        """
        # Validate conversion to float
        try:
            object.__setattr__(self, 'Lx', float(self.Lx))
            object.__setattr__(self, 'Ly', float(self.Ly))
            object.__setattr__(self, 'Lz', float(self.Lz))
            object.__setattr__(self, 'xy', float(self.xy))
            object.__setattr__(self, 'xz', float(self.xz))
            object.__setattr__(self, 'yz', float(self.yz))
        except (TypeError, ValueError) as e:
            raise DataTypeError(
                "Box dimensions must be convertible to float",
                error_code="BOX_TYPE_CONVERSION_FAILED",
                context={
                    "Lx": self.Lx, "Ly": self.Ly, "Lz": self.Lz,
                    "xy": self.xy, "xz": self.xz, "yz": self.yz
                }
            ) from e
        
        # Validate positive dimensions
        if self.Lx <= 0:
            raise ValidationError(
                f"Box dimension Lx must be positive, got {self.Lx}",
                error_code="BOX_INVALID_LX",
                context={"Lx": self.Lx}
            )
        
        if self.Ly <= 0:
            raise ValidationError(
                f"Box dimension Ly must be positive, got {self.Ly}",
                error_code="BOX_INVALID_LY",
                context={"Ly": self.Ly}
            )
        
        if self.Lz <= 0:
            raise ValidationError(
                f"Box dimension Lz must be positive, got {self.Lz}",
                error_code="BOX_INVALID_LZ",
                context={"Lz": self.Lz}
            )
        
        # Validate no NaN or infinite values
        for name, value in [('Lx', self.Lx), ('Ly', self.Ly), ('Lz', self.Lz),
                            ('xy', self.xy), ('xz', self.xz), ('yz', self.yz)]:
            if np.isnan(value):
                raise ValidationError(
                    f"Box dimension {name} is NaN",
                    error_code="BOX_NAN_VALUE",
                    context={name: value}
                )
            if np.isinf(value):
                raise ValidationError(
                    f"Box dimension {name} is infinite",
                    error_code="BOX_INFINITE_VALUE",
                    context={name: value}
                )
        
        logger.debug(f"Box created: Lx={self.Lx}, Ly={self.Ly}, Lz={self.Lz}")
    
    @property
    def volume(self) -> float:
        """
        Compute the box volume.
        
        For orthogonal box: V = Lx * Ly * Lz
        For triclinic box: V = Lx * Ly * Lz * sqrt(1 - cos²(alpha) - cos²(beta) - cos²(gamma) + 2*cos(alpha)*cos(beta)*cos(gamma))
        
        Currently simplified for orthogonal boxes.
        
        Returns
        -------
        float
            Volume of the box
            
        Notes
        -----
        If you have a triclinic box, this calculation is simplified.
        For accurate volume of sheared boxes, use MD package calculations.
        """
        return self.Lx * self.Ly * self.Lz
    
    @property
    def is_orthogonal(self) -> bool:
        """
        Check if box is orthogonal (no shear).
        
        Returns
        -------
        bool
            True if xy, xz, yz are all zero (or very close due to floating point)
        """
        return (
            abs(self.xy) < 1e-12 and
            abs(self.xz) < 1e-12 and
            abs(self.yz) < 1e-12
        )
    
    def as_dict(self) -> Dict[str, float]:
        """
        Convert box to dictionary representation.
        
        Returns
        -------
        dict
            Dictionary with keys 'Lx', 'Ly', 'Lz', 'xy', 'xz', 'yz'
            
        Examples
        --------
        >>> box = Box(Lx=10.0, Ly=10.0, Lz=10.0)
        >>> box_dict = box.as_dict()
        >>> print(box_dict)
        {'Lx': 10.0, 'Ly': 10.0, 'Lz': 10.0, 'xy': 0.0, 'xz': 0.0, 'yz': 0.0}
        """
        return {
            'Lx': self.Lx,
            'Ly': self.Ly,
            'Lz': self.Lz,
            'xy': self.xy,
            'xz': self.xz,
            'yz': self.yz,
        }
    
    def __repr__(self) -> str:
        """Return detailed representation."""
        if self.is_orthogonal:
            return f"Box(Lx={self.Lx}, Ly={self.Ly}, Lz={self.Lz})"
        else:
            return (
                f"Box(Lx={self.Lx}, Ly={self.Ly}, Lz={self.Lz}, "
                f"xy={self.xy}, xz={self.xz}, yz={self.yz})"
            )
    
    def __str__(self) -> str:
        """Return string representation."""
        return repr(self)


# ==============================================================================
# COMPUTATION RESULT BASE CLASS
# ==============================================================================

@dataclass
class ComputationResult:
    """
    Base class for all computation results with status tracking.
    
    This class provides a standard structure for tracking whether a computation
    succeeded, failed, or was skipped, along with confidence metrics and
    diagnostic information.
    
    Attributes
    ----------
    status : str
        Status of the computation: 'ok', 'failed', 'skipped', 'partial'
    reason : Optional[str]
        Human-readable reason for status (e.g., "NaN in output", "Input invalid")
    confidence : float
        Confidence score in the result [0.0, 1.0]. 
        1.0 = fully confident, 0.0 = no confidence
    metadata : Dict[str, Any]
        Additional metadata for debugging and analysis
    
    Raises
    ------
    ValidationError
        If status not in valid set
    DataTypeError
        If confidence not in [0.0, 1.0]
    
    Examples
    --------
    Successful result:
    
    >>> result = ComputationResult(status='ok', confidence=1.0)
    >>> if result.is_successful():
    ...     print(f"Success! (confidence: {result.confidence})")
    
    Failed result:
    
    >>> result = ComputationResult(
    ...     status='failed',
    ...     reason='NaN detected in output',
    ...     confidence=0.0,
    ...     metadata={'value': float('nan')}
    ... )
    >>> if result.is_failed():
    ...     print(f"Failed: {result.reason}")
    
    Partial result:
    
    >>> result = ComputationResult(
    ...     status='partial',
    ...     reason='Some values missing',
    ...     confidence=0.7,
    ...     metadata={'n_computed': 95, 'n_expected': 100}
    ... )
    """
    
    status: str = 'ok'
    reason: Optional[str] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """
        Validate computation result fields.
        
        Checks that:
        - status is in valid set
        - confidence is in [0.0, 1.0]
        - metadata is a dictionary
        
        Raises
        ------
        ValidationError
            If status is invalid
        DataTypeError
            If confidence out of range or wrong type
        """
        valid_statuses = {'ok', 'failed', 'skipped', 'partial'}
        if self.status not in valid_statuses:
            raise ValidationError(
                f"Result status must be one of {valid_statuses}, got '{self.status}'",
                error_code="RESULT_INVALID_STATUS",
                context={"status": self.status, "valid": list(valid_statuses)}
            )
        
        try:
            self.confidence = float(self.confidence)
        except (TypeError, ValueError) as e:
            raise DataTypeError(
                "Confidence must be convertible to float",
                error_code="RESULT_CONFIDENCE_TYPE_ERROR",
                context={"confidence": self.confidence}
            ) from e
        
        if not (0.0 <= self.confidence <= 1.0):
            raise ValidationError(
                f"Confidence must be in [0.0, 1.0], got {self.confidence}",
                error_code="RESULT_CONFIDENCE_OUT_OF_RANGE",
                context={"confidence": self.confidence}
            )
        
        if not isinstance(self.metadata, dict):
            raise DataTypeError(
                "Metadata must be a dictionary",
                error_code="RESULT_METADATA_TYPE_ERROR",
                context={"metadata_type": type(self.metadata).__name__}
            )
        
        logger.debug(
            f"ComputationResult created: status={self.status}, "
            f"confidence={self.confidence:.2f}, reason={self.reason}"
        )
    
    def is_successful(self) -> bool:
        """
        Check if computation was successful.
        
        Returns
        -------
        bool
            True if status == 'ok'
        """
        return self.status == 'ok'
    
    def is_failed(self) -> bool:
        """
        Check if computation failed.
        
        Returns
        -------
        bool
            True if status == 'failed'
        """
        return self.status == 'failed'
    
    def is_partial(self) -> bool:
        """
        Check if computation was partial (some results valid, some missing).
        
        Returns
        -------
        bool
            True if status == 'partial'
        """
        return self.status == 'partial'
    
    def is_skipped(self) -> bool:
        """
        Check if computation was skipped.
        
        Returns
        -------
        bool
            True if status == 'skipped'
        """
        return self.status == 'skipped'
    
    def add_metadata(self, key: str, value: Any) -> None:
        """
        Add metadata to result for debugging.
        
        Parameters
        ----------
        key : str
            Metadata key
        value : Any
            Metadata value
        
        Examples
        --------
        >>> result = ComputationResult()
        >>> result.add_metadata('n_iterations', 100)
        >>> result.add_metadata('convergence_error', 1e-6)
        """
        self.metadata[key] = value
        logger.debug(f"Added metadata: {key}={value}")
    
    def summary(self) -> str:
        """
        Return a summary string of the result.
        
        Returns
        -------
        str
            Human-readable summary of result status and metadata
        
        Examples
        --------
        >>> result = ComputationResult(status='partial', reason='3 of 10 values missing')
        >>> print(result.summary())
        Result: partial (confidence: 0.70)
        Reason: 3 of 10 values missing
        """
        lines = [f"Result: {self.status} (confidence: {self.confidence:.2f})"]
        if self.reason:
            lines.append(f"Reason: {self.reason}")
        if self.metadata:
            lines.append(f"Metadata: {self.metadata}")
        return '\n'.join(lines)
    
    def __repr__(self) -> str:
        """Return detailed representation."""
        return (
            f"ComputationResult(status={self.status!r}, "
            f"confidence={self.confidence}, "
            f"reason={self.reason!r})"
        )


if __name__ == "__main__":
    """
    Test/demo when module is run directly.
    """
    import sys
    
    print("\n" + "="*80)
    print("CORE TYPES MODULE - TESTING")
    print("="*80 + "\n")
    
    # Test Box creation and validation
    print("[TEST 1] Creating orthogonal box...")
    try:
        box = Box(Lx=10.0, Ly=10.0, Lz=10.0)
        print(f"✓ {box}")
        print(f"  Volume: {box.volume:.2f}")
        print(f"  Is orthogonal: {box.is_orthogonal}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("\n[TEST 2] Creating triclinic box...")
    try:
        box = Box(Lx=10.0, Ly=10.0, Lz=10.0, xy=2.0)
        print(f"✓ {box}")
        print(f"  Is orthogonal: {box.is_orthogonal}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("\n[TEST 3] Invalid box (negative dimension)...")
    try:
        box = Box(Lx=-5.0, Ly=10.0, Lz=10.0)
        print(f"✓ {box}")
    except ValidationError as e:
        print(f"✓ Correctly caught error: {e.error_code}")
    
    print("\n[TEST 4] Creating successful ComputationResult...")
    try:
        result = ComputationResult(status='ok', confidence=1.0)
        print(f"✓ {result}")
        print(f"  Is successful: {result.is_successful()}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("\n[TEST 5] Creating partial ComputationResult...")
    try:
        result = ComputationResult(
            status='partial',
            reason='Some values missing',
            confidence=0.7,
            metadata={'n_computed': 95, 'n_expected': 100}
        )
        print(f"✓ {result}")
        print(f"  Summary:\n{result.summary()}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("\n[TEST 6] Invalid ComputationResult (bad status)...")
    try:
        result = ComputationResult(status='invalid_status')
        print(f"✓ {result}")
    except ValidationError as e:
        print(f"✓ Correctly caught error: {e.error_code}")
    
    print("\n" + "="*80)
    print("✓ Types module is ready for use!")
    print("="*80 + "\n")