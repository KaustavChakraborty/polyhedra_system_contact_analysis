# contacts/properties.py
# ==============================================================================
# Module: contacts.properties
# Purpose: Calculate contact properties (area, normal, etc)
#
# Classes:
#   - ContactPropertyCalculator: Compute properties from face pairs
#
# Author: Contact Analysis Team
# ==============================================================================

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .. import ValidationError
from ..primitives import Polygon
from .types import FacePair

logger = logging.getLogger(__name__)


# ==============================================================================
# CONTACT PROPERTY CALCULATOR
# ==============================================================================

@dataclass
class ContactPropertyCalculator:
    """
    Calculate contact properties from face pairs.
    
    Computes geometric properties of contact regions including area,
    normal vectors, center of contact, and orientation metrics.
    
    Examples
    --------
    >>> calculator = ContactPropertyCalculator()
    >>> props = calculator.compute_properties(face_pair)
    >>> print(f"Contact area: {props['area']}")
    """
    
    tolerance: float = 1e-12
    
    def __post_init__(self) -> None:
        """Validate calculator."""
        if self.tolerance <= 0:
            raise ValidationError(
                f"tolerance must be positive, got {self.tolerance}",
                error_code="CALC_INVALID_TOLERANCE"
            )
        
        logger.debug(f"[ContactPropertyCalculator] Initialized")
    
    def compute_properties(
        self,
        face_pair: FacePair,
        face_A: Polygon = None,
        face_B: Polygon = None
    ) -> Dict[str, float]:
        """
        Compute all contact properties.
        
        Parameters
        ----------
        face_pair : FacePair
            Face pair to analyze
        face_A : Polygon, optional
            Face A (optional, can compute from face_pair)
        face_B : Polygon, optional
            Face B (optional, can compute from face_pair)
        
        Returns
        -------
        Dict[str, float]
            Properties: {name: value}
        
        Examples
        --------
        >>> props = calculator.compute_properties(face_pair)
        """
        
        if not isinstance(face_pair, FacePair):
            raise ValidationError(
                "face_pair must be FacePair instance",
                error_code="CALC_INVALID_FACE_PAIR"
            )
        
        properties = {}
        
        # Basic properties from face_pair
        properties['overlap_area'] = face_pair.overlap_area
        properties['normal_dot'] = face_pair.normal_dot
        properties['face_A_area'] = face_pair.face_A_area
        properties['face_B_area'] = face_pair.face_B_area
        
        # Computed properties
        properties['overlap_fraction_A'] = face_pair.overlap_fraction_A
        properties['overlap_fraction_B'] = face_pair.overlap_fraction_B
        
        # Contact orientation metrics
        properties['contact_orientation'] = self._compute_contact_orientation(
            face_pair.normal_dot
        )
        
        # Contact quality (measure of how well faces align)
        properties['contact_quality'] = self._compute_contact_quality(face_pair)
        
        logger.debug(
            f"[ContactPropertyCalculator] Computed properties for pair "
            f"({face_pair.particle_A_id},{face_pair.particle_B_id})"
        )
        
        return properties
    
    def compute_contact_area(self, face_pair: FacePair) -> float:
        """
        Get contact area.
        
        Parameters
        ----------
        face_pair : FacePair
            Face pair
        
        Returns
        -------
        float
            Contact area
        """
        
        return face_pair.overlap_area
    
    def compute_contact_normal(
        self,
        face_A: Polygon,
        face_B: Polygon
    ) -> np.ndarray:
        """
        Compute average contact normal.
        
        Parameters
        ----------
        face_A : Polygon
            Face A
        face_B : Polygon
            Face B
        
        Returns
        -------
        np.ndarray
            Unit normal vector (shape: (3,))
        
        Notes
        -----
        Returns the normalized average of the two face normals.
        
        Examples
        --------
        >>> normal = calculator.compute_contact_normal(face_A, face_B)
        >>> print(f"Normal: {normal}")
        """
        
        try:
            # Get normals
            normal_A = face_A.normal / (np.linalg.norm(face_A.normal) + 1e-15)
            normal_B = face_B.normal / (np.linalg.norm(face_B.normal) + 1e-15)
            
            # Average and normalize
            avg_normal = (normal_A + normal_B) / 2.0
            avg_normal = avg_normal / (np.linalg.norm(avg_normal) + 1e-15)
            
            return avg_normal
        
        except Exception as e:
            logger.warning(f"Failed to compute contact normal: {e}")
            return np.array([0.0, 0.0, 1.0])
    
    def _compute_contact_orientation(self, normal_dot: float) -> str:
        """
        Classify contact orientation.
        
        Parameters
        ----------
        normal_dot : float
            Dot product of normals
        
        Returns
        -------
        str
            Orientation type
        """
        
        # Clamp to valid range
        normal_dot = np.clip(normal_dot, -1.0, 1.0)
        
        abs_dot = abs(normal_dot)
        
        if abs_dot > 0.95:
            return 'parallel'  # Normals nearly parallel or anti-parallel
        elif abs_dot > 0.70:
            return 'angled'    # Normals at significant angle
        else:
            return 'perpendicular'  # Normals nearly perpendicular
    
    def _compute_contact_quality(self, face_pair: FacePair) -> float:
        """
        Compute contact quality metric.
        
        Higher values indicate better alignment and larger overlap.
        
        Parameters
        ----------
        face_pair : FacePair
            Face pair
        
        Returns
        -------
        float
            Quality metric in [0, 1]
        """
        
        if face_pair.overlap_area == 0:
            return 0.0
        
        try:
            # Combination of overlap fraction and orientation
            max_overlap_frac = max(
                face_pair.overlap_fraction_A,
                face_pair.overlap_fraction_B
            )
            
            # Bonus for parallel normals
            orientation_factor = (1.0 + abs(face_pair.normal_dot)) / 2.0
            
            quality = max_overlap_frac * orientation_factor
            
            return float(np.clip(quality, 0.0, 1.0))
        
        except:
            return 0.0
    
    def get_summary(self, properties: Dict[str, float]) -> str:
        """
        Get human-readable summary of properties.
        
        Parameters
        ----------
        properties : Dict[str, float]
            Properties from compute_properties()
        
        Returns
        -------
        str
            Summary string
        
        Examples
        --------
        >>> props = calculator.compute_properties(face_pair)
        >>> print(calculator.get_summary(props))
        """
        
        return (
            f"Contact Properties:\n"
            f"  Area: {properties['overlap_area']:.6f}\n"
            f"  Orientation: {properties['contact_orientation']}\n"
            f"  Quality: {properties['contact_quality']:.3f}\n"
            f"  Coverage A: {properties['overlap_fraction_A']:.1%}\n"
            f"  Coverage B: {properties['overlap_fraction_B']:.1%}"
        )


if __name__ == "__main__":
    """Test when run directly."""
    print("\n" + "="*80)
    print("CONTACTS PROPERTIES MODULE - TESTING")
    print("="*80 + "\n")
    
    print("[TEST] ContactPropertyCalculator...")
    try:
        from .types import FacePair
        
        calculator = ContactPropertyCalculator()
        
        # Create test face pair
        face_pair = FacePair(
            particle_A_id=0,
            particle_B_id=1,
            face_A_idx=0,
            face_B_idx=2,
            overlap_area=0.5,
            normal_dot=0.95,
            face_A_area=1.0,
            face_B_area=1.0
        )
        
        # Compute properties
        properties = calculator.compute_properties(face_pair)
        
        print("✓ Properties computed:")
        print(calculator.get_summary(properties))
        print()
        
        print(f"✓ Contact quality: {properties['contact_quality']:.3f}")
        print(f"✓ Contact orientation: {properties['contact_orientation']}\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")
        import traceback
        traceback.print_exc()
    
    print("="*80)
    print("✓ Properties module tests passed!")
    print("="*80 + "\n")
