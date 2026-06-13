# contacts/types.py
# ==============================================================================
# Module: contacts.types
# Purpose: Data structures for contact analysis results
#
# Defines dataclasses:
#   - FacePair: Pair of potentially overlapping faces
#   - ContactResult: Complete contact analysis between two particles
#
# Author: Contact Analysis Team
# ==============================================================================

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import numpy as np

from .. import ValidationError, DataTypeError

logger = logging.getLogger(__name__)


# ==============================================================================
# FACE PAIR: Two faces that may be in contact
# ==============================================================================

@dataclass
class FacePair:
    """
    Pair of faces that may be in contact.
    
    Represents potential overlap between one face from particle A and one face
    from particle B, including geometric and topological information.
    
    Attributes
    ----------
    particle_A_id : int
        ID of first particle
    particle_B_id : int
        ID of second particle
    face_A_idx : int
        Face index in particle A's shape
    face_B_idx : int
        Face index in particle B's shape
    overlap_area : float
        Area of overlap between the two faces (≥ 0)
    normal_dot : float
        Dot product of face normals (ranges: -1 to 1)
    face_A_area : float
        Total area of face A
    face_B_area : float
        Total area of face B
    face_A_type : Optional[str], optional
        Classification of face A (e.g., "flat", "edge", "vertex")
    face_B_type : Optional[str], optional
        Classification of face B
    
    Raises
    ------
    ValidationError
        If particle IDs or face indices invalid
    DataTypeError
        If areas or overlap area invalid
    """
    
    particle_A_id: int
    particle_B_id: int
    face_A_idx: int
    face_B_idx: int
    overlap_area: float
    normal_dot: float
    face_A_area: float
    face_B_area: float
    face_A_type: Optional[str] = None
    face_B_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate face pair after initialization."""
        # Validate particle IDs
        if self.particle_A_id < 0 or self.particle_B_id < 0:
            raise ValidationError(
                f"Particle IDs must be non-negative: A={self.particle_A_id}, B={self.particle_B_id}",
                error_code="FACEPAIR_INVALID_PARTICLE_ID"
            )
        
        if self.particle_A_id == self.particle_B_id:
            raise ValidationError(
                f"Cannot have face pair between same particle {self.particle_A_id}",
                error_code="FACEPAIR_SAME_PARTICLE"
            )
        
        # Validate face indices
        if self.face_A_idx < 0 or self.face_B_idx < 0:
            raise ValidationError(
                f"Face indices must be non-negative: A={self.face_A_idx}, B={self.face_B_idx}",
                error_code="FACEPAIR_INVALID_FACE_INDEX"
            )
        
        # Validate areas
        for val, name in [
            (self.overlap_area, "overlap_area"),
            (self.face_A_area, "face_A_area"),
            (self.face_B_area, "face_B_area")
        ]:
            if not isinstance(val, (int, float)):
                raise DataTypeError(
                    f"{name} must be numeric, got {type(val).__name__}",
                    error_code="FACEPAIR_AREA_TYPE_ERROR"
                )
            
            if val < 0:
                raise ValidationError(
                    f"{name} must be non-negative, got {val}",
                    error_code="FACEPAIR_NEGATIVE_AREA"
                )
            
            if np.isnan(val) or np.isinf(val):
                raise DataTypeError(
                    f"{name} contains NaN or Inf: {val}",
                    error_code="FACEPAIR_AREA_NAN_INF"
                )
        
        # Validate normal_dot
        if not isinstance(self.normal_dot, (int, float)):
            raise DataTypeError(
                f"normal_dot must be numeric, got {type(self.normal_dot).__name__}",
                error_code="FACEPAIR_NORMAL_DOT_TYPE_ERROR"
            )
        
        if not (-1.0 <= self.normal_dot <= 1.0):
            raise ValidationError(
                f"normal_dot must be in [-1, 1], got {self.normal_dot}",
                error_code="FACEPAIR_NORMAL_DOT_OUT_OF_RANGE"
            )
        
        # Check overlap consistency
        if self.overlap_area > 0:
            max_area = min(self.face_A_area, self.face_B_area)
            if self.overlap_area > max_area * 1.001:  # Allow 0.1% tolerance
                logger.warning(
                    f"Overlap area {self.overlap_area} exceeds min face area {max_area}"
                )
        
        logger.debug(
            f"FacePair created: ({self.particle_A_id},{self.particle_B_id}), "
            f"faces=({self.face_A_idx},{self.face_B_idx}), "
            f"overlap={self.overlap_area:.6f}"
        )
    
    @property
    def is_overlapping(self) -> bool:
        """Whether faces actually overlap."""
        return self.overlap_area > 0.0
    
    @property
    def overlap_fraction_A(self) -> float:
        """Fraction of face A covered by overlap."""
        if self.face_A_area == 0:
            return 0.0
        return self.overlap_area / self.face_A_area
    
    @property
    def overlap_fraction_B(self) -> float:
        """Fraction of face B covered by overlap."""
        if self.face_B_area == 0:
            return 0.0
        return self.overlap_area / self.face_B_area
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"FacePair(({self.particle_A_id},{self.particle_B_id}), "
            f"faces=({self.face_A_idx},{self.face_B_idx}), "
            f"overlap={self.overlap_area:.6f})"
        )


# ==============================================================================
# CONTACT RESULT: Complete contact analysis between two particles
# ==============================================================================

@dataclass
class ContactResult:
    """
    Complete contact analysis between two particles.
    
    Contains all information about contact between a pair of particles,
    including face pairs, distances, and classification metrics.
    
    Attributes
    ----------
    particle_A_id : int
        ID of first particle
    particle_B_id : int
        ID of second particle
    face_pairs : List[FacePair]
        All overlapping face pairs (may be empty if no contact)
    distances : Dict[str, Dict[str, float]]
        Distance metrics: {metric_name: {component: value}}
    contact_order : Dict[str, Dict[str, float]]
        Contact order metrics: {metric_name: {component: value}}
    overlap_count : int
        Number of overlapping face pairs
    frame_index : int
        Frame number in trajectory
    metadata : Dict[str, Any], optional
        Additional analysis metadata
    
    Raises
    ------
    ValidationError
        If particle IDs invalid or data inconsistent
    """
    
    particle_A_id: int
    particle_B_id: int
    face_pairs: List[FacePair]
    distances: Dict[str, Dict[str, float]]
    contact_order: Dict[str, Dict[str, float]]
    overlap_count: int
    frame_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate contact result after initialization."""
        # Validate particle IDs
        if self.particle_A_id < 0 or self.particle_B_id < 0:
            raise ValidationError(
                f"Particle IDs must be non-negative",
                error_code="CONTACT_INVALID_PARTICLE_ID"
            )
        
        if self.particle_A_id == self.particle_B_id:
            raise ValidationError(
                f"Cannot have contact between same particle",
                error_code="CONTACT_SAME_PARTICLE"
            )
        
        # Validate frame index
        if self.frame_index < 0:
            raise ValidationError(
                f"frame_index must be non-negative, got {self.frame_index}",
                error_code="CONTACT_NEGATIVE_FRAME_INDEX"
            )
        
        # Validate face pairs
        if not isinstance(self.face_pairs, list):
            raise DataTypeError(
                f"face_pairs must be list, got {type(self.face_pairs).__name__}",
                error_code="CONTACT_FACEPAIRS_TYPE_ERROR"
            )
        
        for i, fp in enumerate(self.face_pairs):
            if not isinstance(fp, FacePair):
                raise DataTypeError(
                    f"face_pairs[{i}] must be FacePair, got {type(fp).__name__}",
                    error_code="CONTACT_FACEPAIR_TYPE_ERROR"
                )
        
        # Validate overlap count
        if self.overlap_count < 0:
            raise ValidationError(
                f"overlap_count must be non-negative, got {self.overlap_count}",
                error_code="CONTACT_NEGATIVE_OVERLAP_COUNT"
            )
        
        # Check consistency between face_pairs and overlap_count
        actual_overlaps = sum(1 for fp in self.face_pairs if fp.is_overlapping)
        if actual_overlaps != self.overlap_count:
            logger.warning(
                f"Inconsistency: overlap_count={self.overlap_count}, "
                f"actual_overlapping_pairs={actual_overlaps}"
            )
        
        logger.debug(
            f"ContactResult created: ({self.particle_A_id},{self.particle_B_id}), "
            f"frame={self.frame_index}, overlaps={self.overlap_count}"
        )
    
    @property
    def has_contact(self) -> bool:
        """Whether particles are in contact (have overlapping faces)."""
        return self.overlap_count > 0
    
    @property
    def total_overlap_area(self) -> float:
        """Sum of all overlapping face areas."""
        return sum(fp.overlap_area for fp in self.face_pairs if fp.is_overlapping)
    
    @property
    def num_face_pairs(self) -> int:
        """Total number of face pairs analyzed."""
        return len(self.face_pairs)
    
    @property
    def contact_fraction(self) -> float:
        """Fraction of analyzed face pairs that overlap."""
        if len(self.face_pairs) == 0:
            return 0.0
        return self.overlap_count / len(self.face_pairs)
    
    def get_metric_value(self, metric_name: str, component: str = "value") -> Optional[float]:
        """
        Get distance metric value.
        
        Parameters
        ----------
        metric_name : str
            Name of metric
        component : str, optional
            Component name (default: "value")
        
        Returns
        -------
        float or None
            Metric value if available, None otherwise
        """
        
        if metric_name not in self.distances:
            return None
        
        if component not in self.distances[metric_name]:
            return None
        
        return self.distances[metric_name][component]
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ContactResult(({self.particle_A_id},{self.particle_B_id}), "
            f"frame={self.frame_index}, overlaps={self.overlap_count}/{self.num_face_pairs})"
        )


if __name__ == "__main__":
    """Test when run directly."""
    print("\n" + "="*80)
    print("CONTACTS TYPES MODULE - TESTING")
    print("="*80 + "\n")
    
    print("[TEST 1] Creating test FacePair...")
    try:
        fp = FacePair(
            particle_A_id=0,
            particle_B_id=1,
            face_A_idx=0,
            face_B_idx=2,
            overlap_area=0.5,
            normal_dot=0.95,
            face_A_area=1.0,
            face_B_area=1.0,
            face_A_type="flat",
            face_B_type="flat"
        )
        
        print(f"✓ {fp}")
        print(f"  Is overlapping: {fp.is_overlapping}")
        print(f"  Overlap fraction A: {fp.overlap_fraction_A:.1%}")
        print(f"  Overlap fraction B: {fp.overlap_fraction_B:.1%}\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")
    
    print("[TEST 2] Creating ContactResult...")
    try:
        face_pairs = [
            FacePair(0, 1, 0, 2, 0.5, 0.95, 1.0, 1.0),
            FacePair(0, 1, 1, 3, 0.3, 0.90, 1.0, 1.0),
        ]
        
        result = ContactResult(
            particle_A_id=0,
            particle_B_id=1,
            face_pairs=face_pairs,
            distances={"center_distance": {"value": 3.5}},
            contact_order={"contact_order": {"value": 2}},
            overlap_count=2,
            frame_index=0
        )
        
        print(f"✓ {result}")
        print(f"  Has contact: {result.has_contact}")
        print(f"  Total overlap area: {result.total_overlap_area:.3f}")
        print(f"  Contact fraction: {result.contact_fraction:.1%}\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")
    
    print("[TEST 3] Invalid FacePair (same particle)...")
    try:
        bad_fp = FacePair(0, 0, 0, 1, 0.5, 0.9, 1.0, 1.0)
    except ValidationError as e:
        print(f"✓ Correctly caught: {e.error_code}\n")
    
    print("="*80)
    print("✓ All tests passed!")
    print("="*80 + "\n")
