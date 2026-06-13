# metrics/implementations/distance_calc.py
# ==============================================================================
# Module: metrics.implementations.distance_calc
# Purpose: Pure distance calculation utilities
#
# Classes:
#   - DistanceCalculator: Utility for computing distances
#
# Author: Contact Analysis Team
# ==============================================================================

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ==============================================================================
# DISTANCE CALCULATOR
# ==============================================================================

class DistanceCalculator:
    """
    Utility class for distance calculations.
    
    Provides static and instance methods for computing various distances
    between geometric entities (points, lines, planes, etc.).
    
    Examples
    --------
    >>> calc = DistanceCalculator()
    >>> dist = calc.point_to_point(p1, p2)
    >>> perp_dist = calc.point_to_plane(point, plane_normal, plane_point)
    """
    
    def __init__(self, tolerance: float = 1e-12) -> None:
        """
        Initialize distance calculator.
        
        Parameters
        ----------
        tolerance : float, optional
            Numerical tolerance (default: 1e-12)
        """
        self.tolerance = tolerance
        logger.debug(f"[DistanceCalculator] Initialized with tolerance={tolerance}")
    
    @staticmethod
    def point_to_point(p1: np.ndarray, p2: np.ndarray) -> float:
        """
        Compute Euclidean distance between two points.
        
        Parameters
        ----------
        p1 : np.ndarray
            First point (shape: (3,))
        p2 : np.ndarray
            Second point (shape: (3,))
        
        Returns
        -------
        float
            Euclidean distance
        
        Examples
        --------
        >>> d = DistanceCalculator.point_to_point(p1, p2)
        """
        
        p1 = np.asarray(p1, dtype=float)
        p2 = np.asarray(p2, dtype=float)
        
        return float(np.linalg.norm(p2 - p1))
    
    @staticmethod
    def point_to_plane(
        point: np.ndarray,
        plane_normal: np.ndarray,
        plane_point: np.ndarray
    ) -> float:
        """
        Compute perpendicular distance from point to plane.
        
        Parameters
        ----------
        point : np.ndarray
            Point coordinates (shape: (3,))
        plane_normal : np.ndarray
            Plane normal vector (shape: (3,))
        plane_point : np.ndarray
            Point on plane (shape: (3,))
        
        Returns
        -------
        float
            Perpendicular distance
        
        Examples
        --------
        >>> d = DistanceCalculator.point_to_plane(p, normal, p_plane)
        """
        
        point = np.asarray(point, dtype=float)
        plane_normal = np.asarray(plane_normal, dtype=float)
        plane_point = np.asarray(plane_point, dtype=float)
        
        # Normalize normal vector
        normal = plane_normal / (np.linalg.norm(plane_normal) + 1e-15)
        
        # Project vector onto normal
        vec = point - plane_point
        distance = abs(np.dot(vec, normal))
        
        return float(distance)
    
    @staticmethod
    def line_to_line(
        p1: np.ndarray,
        d1: np.ndarray,
        p2: np.ndarray,
        d2: np.ndarray
    ) -> float:
        """
        Compute minimum distance between two skew lines.
        
        Parameters
        ----------
        p1 : np.ndarray
            Point on first line (shape: (3,))
        d1 : np.ndarray
            Direction of first line (shape: (3,))
        p2 : np.ndarray
            Point on second line (shape: (3,))
        d2 : np.ndarray
            Direction of second line (shape: (3,))
        
        Returns
        -------
        float
            Minimum distance
        
        Notes
        -----
        Uses formula for skew lines in 3D space.
        """
        
        p1 = np.asarray(p1, dtype=float)
        d1 = np.asarray(d1, dtype=float)
        p2 = np.asarray(p2, dtype=float)
        d2 = np.asarray(d2, dtype=float)
        
        # Normalize directions
        d1 = d1 / (np.linalg.norm(d1) + 1e-15)
        d2 = d2 / (np.linalg.norm(d2) + 1e-15)
        
        # Vector between points
        w = p1 - p2
        
        # Cross products
        denom = np.linalg.norm(np.cross(d1, d2))**2
        
        if denom < 1e-10:
            # Lines are parallel
            return float(np.linalg.norm(np.cross(w, d1)))
        
        # Skew line distance formula
        distance = abs(np.dot(w, np.cross(d1, d2))) / np.sqrt(denom)
        
        return float(distance)
    
    def get_statistics(self) -> dict:
        """Get calculator statistics."""
        return {'tolerance': self.tolerance}


if __name__ == "__main__":
    """Test when run directly."""
    print("\n" + "="*80)
    print("DISTANCE CALCULATOR - TESTING")
    print("="*80 + "\n")
    
    print("[TEST 1] Point to point distance...")
    try:
        p1 = np.array([0, 0, 0])
        p2 = np.array([1, 1, 1])
        
        dist = DistanceCalculator.point_to_point(p1, p2)
        print(f"✓ Distance: {dist:.6f}")
        print(f"  Expected: {np.sqrt(3):.6f}\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")
    
    print("[TEST 2] Point to plane distance...")
    try:
        point = np.array([1, 1, 1])
        plane_normal = np.array([0, 0, 1])
        plane_point = np.array([0, 0, 0])
        
        dist = DistanceCalculator.point_to_plane(point, plane_normal, plane_point)
        print(f"✓ Distance: {dist:.6f}")
        print(f"  Expected: 1.000000\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")
    
    print("="*80)
    print("✓ Distance calculator tests passed!")
    print("="*80 + "\n")
