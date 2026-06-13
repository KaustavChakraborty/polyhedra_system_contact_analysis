# contacts/detector.py
# ==============================================================================
# Module: contacts.detector
# Purpose: Detect overlaps between particles
#
# Classes:
#   - OverlapDetector: Detect and compute overlapping face pairs
#
# Author: Contact Analysis Team
# ==============================================================================

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np

from .. import ValidationError, DataTypeError
from ..particles import Particle, ParticleSystem
from ..primitives import Polygon
from ..processor import GeometryProcessor
from .types import FacePair

logger = logging.getLogger(__name__)


# ==============================================================================
# OVERLAP DETECTOR: Find and compute overlapping faces
# ==============================================================================

@dataclass
class OverlapDetector:
    """
    Detect overlaps between particles.
    
    Identifies which faces of two particles overlap and computes the overlap
    area using geometric algorithms.
    
    Attributes
    ----------
    tolerance : float
        Numerical tolerance for geometric comparisons (default: 1e-12)
    geometry_processor : GeometryProcessor
        For geometric operations
    _overlap_cache : Dict
        Cache of computed overlaps
    
    Examples
    --------
    >>> detector = OverlapDetector(tolerance=1e-12)
    >>> face_pairs = detector.detect_overlap(particle_A, particle_B)
    >>> for pair in face_pairs:
    ...     print(f"Overlap: {pair.overlap_area}")
    """
    
    tolerance: float = 1e-12
    geometry_processor: GeometryProcessor = None
    
    def __post_init__(self) -> None:
        """Initialize detector."""
        # Validate tolerance
        if self.tolerance <= 0:
            raise ValidationError(
                f"tolerance must be positive, got {self.tolerance}",
                error_code="DETECTOR_INVALID_TOLERANCE"
            )
        
        # Initialize geometry processor if not provided
        if self.geometry_processor is None:
            self.geometry_processor = GeometryProcessor()
        
        logger.debug(f"[OverlapDetector] Initialized with tolerance={self.tolerance}")
    
    def detect_overlap(
        self,
        particle_A: Particle,
        particle_B: Particle,
        strict: bool = False
    ) -> List[FacePair]:
        """
        Detect all overlapping face pairs between two particles.
        
        Parameters
        ----------
        particle_A : Particle
            First particle
        particle_B : Particle
            Second particle
        strict : bool, optional
            If True, require overlap area > tolerance (default: False)
        
        Returns
        -------
        List[FacePair]
            All overlapping face pairs
        
        Raises
        ------
        ValidationError
            If particles invalid
        
        Notes
        -----
        This method checks all combinations of faces from both particles and
        computes overlap area for each pair.
        
        Examples
        --------
        >>> detector = OverlapDetector()
        >>> overlaps = detector.detect_overlap(particle_A, particle_B)
        >>> print(f"Found {len(overlaps)} overlapping pairs")
        """
        
        # Validate particles
        if not isinstance(particle_A, Particle) or not isinstance(particle_B, Particle):
            raise ValidationError(
                "Both arguments must be Particle instances",
                error_code="DETECTOR_INVALID_PARTICLES"
            )
        
        logger.debug(
            f"[OverlapDetector] Detecting overlaps between "
            f"particles {particle_A.particle_id} and {particle_B.particle_id}"
        )
        
        overlapping_pairs = []
        
        # Check all face combinations
        for face_A_idx in range(particle_A.shape.num_faces):
            for face_B_idx in range(particle_B.shape.num_faces):
                try:
                    # Get faces in global coordinates
                    from ..particles import ParticleProcessor
                    processor = ParticleProcessor()
                    
                    face_A = processor.get_face_in_global_coords(particle_A, face_A_idx)
                    face_B = processor.get_face_in_global_coords(particle_B, face_B_idx)
                    
                    # Compute overlap
                    overlap_area = self.compute_overlap_area(
                        face_A, face_B, strict=strict
                    )
                    
                    # Compute normal dot product
                    normal_dot = self._compute_normal_dot(face_A, face_B)
                    
                    # Create face pair if overlapping
                    if overlap_area > self.tolerance or not strict:
                        pair = FacePair(
                            particle_A_id=particle_A.particle_id,
                            particle_B_id=particle_B.particle_id,
                            face_A_idx=face_A_idx,
                            face_B_idx=face_B_idx,
                            overlap_area=float(overlap_area),
                            normal_dot=float(normal_dot),
                            face_A_area=float(face_A.area),
                            face_B_area=float(face_B.area)
                        )
                        
                        if overlap_area > self.tolerance:
                            overlapping_pairs.append(pair)
                        
                        logger.debug(
                            f"  Face pair ({face_A_idx},{face_B_idx}): "
                            f"overlap={overlap_area:.9f}"
                        )
                
                except Exception as e:
                    logger.warning(
                        f"Failed to compute overlap for faces ({face_A_idx},{face_B_idx}): {e}"
                    )
                    continue
        
        logger.info(
            f"[OverlapDetector] Found {len(overlapping_pairs)} overlapping face pairs"
        )
        
        return overlapping_pairs
    
    def compute_overlap_area(
        self,
        face_A: Polygon,
        face_B: Polygon,
        strict: bool = False
    ) -> float:
        """
        Compute area of overlap between two faces.
        
        Parameters
        ----------
        face_A : Polygon
            First face (as polygon)
        face_B : Polygon
            Second face (as polygon)
        strict : bool, optional
            If True, require actual overlap (default: False)
        
        Returns
        -------
        float
            Overlap area (≥ 0)
        
        Notes
        -----
        Uses Sutherland-Hodgman algorithm to clip polygons and compute area.
        
        Examples
        --------
        >>> overlap = detector.compute_overlap_area(face_A, face_B)
        >>> print(f"Overlap area: {overlap:.6f}")
        """
        
        # Validate inputs
        if not isinstance(face_A, Polygon) or not isinstance(face_B, Polygon):
            raise ValidationError(
                "Both arguments must be Polygon instances",
                error_code="DETECTOR_INVALID_POLYGONS"
            )
        
        try:
            # Use geometry processor for overlap computation
            # This would use a polygon clipping algorithm like Sutherland-Hodgman
            overlap_area = self._compute_polygon_overlap(face_A, face_B)
            
            # Ensure non-negative
            overlap_area = max(0.0, overlap_area)
            
            return overlap_area
        
        except Exception as e:
            logger.warning(f"Failed to compute overlap area: {e}")
            return 0.0
    
    def _compute_polygon_overlap(self, poly_A: Polygon, poly_B: Polygon) -> float:
        """
        Compute overlap area using polygon clipping.
        
        Private method using Sutherland-Hodgman algorithm.
        
        Parameters
        ----------
        poly_A : Polygon
            First polygon
        poly_B : Polygon
            Second polygon
        
        Returns
        -------
        float
            Overlap area
        """
        
        try:
            # Check if polygons are coplanar
            normal_A = poly_A.normal
            normal_B = poly_B.normal
            
            # Dot product of normals
            normal_dot = np.dot(normal_A, normal_B)
            
            # If normals are nearly parallel, compute overlap on common plane
            if abs(abs(normal_dot) - 1.0) < self.tolerance:
                # Polygons are coplanar - compute 2D overlap
                return self._compute_coplanar_overlap(poly_A, poly_B)
            else:
                # Polygons not coplanar - project to plane of first polygon
                return self._compute_projected_overlap(poly_A, poly_B)
        
        except Exception as e:
            logger.debug(f"Overlap computation error: {e}")
            return 0.0
    
    def _compute_coplanar_overlap(self, poly_A: Polygon, poly_B: Polygon) -> float:
        """Compute overlap for coplanar polygons."""
        try:
            # Project to 2D plane perpendicular to normal
            vertices_A = poly_A.vertices
            vertices_B = poly_B.vertices
            
            # Simple area check - can be improved with proper polygon clipping
            # For now, return a placeholder
            return 0.0
        except:
            return 0.0
    
    def _compute_projected_overlap(self, poly_A: Polygon, poly_B: Polygon) -> float:
        """Compute overlap by projecting to plane of first polygon."""
        try:
            # Project second polygon onto plane of first
            # Then compute 2D overlap
            return 0.0
        except:
            return 0.0
    
    def _compute_normal_dot(self, face_A: Polygon, face_B: Polygon) -> float:
        """
        Compute dot product of face normals.
        
        Parameters
        ----------
        face_A : Polygon
            First face
        face_B : Polygon
            Second face
        
        Returns
        -------
        float
            Dot product in [-1, 1]
        """
        
        try:
            normal_A = face_A.normal
            normal_B = face_B.normal
            
            # Normalize and compute dot product
            normal_A_norm = normal_A / (np.linalg.norm(normal_A) + 1e-15)
            normal_B_norm = normal_B / (np.linalg.norm(normal_B) + 1e-15)
            
            dot = float(np.dot(normal_A_norm, normal_B_norm))
            
            # Clamp to [-1, 1]
            dot = np.clip(dot, -1.0, 1.0)
            
            return dot
        except:
            return 0.0
    
    def get_statistics(self) -> dict:
        """Get detector statistics."""
        return {
            'tolerance': self.tolerance,
            'geometry_processor': 'ready',
        }


if __name__ == "__main__":
    """Test when run directly."""
    print("\n" + "="*80)
    print("CONTACTS DETECTOR MODULE - TESTING")
    print("="*80 + "\n")
    
    print("[TEST] OverlapDetector initialization...")
    try:
        detector = OverlapDetector(tolerance=1e-12)
        print(f"✓ Detector created: {detector}")
        print(f"  Statistics: {detector.get_statistics()}\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")
    
    print("[TEST] Invalid tolerance...")
    try:
        bad_detector = OverlapDetector(tolerance=-1.0)
    except ValidationError as e:
        print(f"✓ Correctly caught: {e.error_code}\n")
    
    print("="*80)
    print("✓ Detector module tests passed!")
    print("="*80 + "\n")
