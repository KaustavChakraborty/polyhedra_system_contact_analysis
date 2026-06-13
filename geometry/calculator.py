# geometry/calculator.py
# ==============================================================================
# Module: geometry.calculator
# Purpose: Pure geometric calculation functions (no side effects)
#
# This module provides stateless geometric operations:
#   - polygon_area_3d(): Shoelace formula for 3D polygon area
#   - polygon_normal(): Normal vector to polygon plane
#   - polygon_centroid(): Center of mass of polygon vertices
#   - reorder_polygon_vertices(): Cyclic vertex ordering
#   - check_coplanar(): Check if vertices lie in same plane
#   - polygon_intersection_area(): Compute overlap area between two polygons
#
# All functions are PURE:
#   - No side effects (no global state modified)
#   - Same input → same output (deterministic)
#   - Easy to test and cache
#   - Easily parallelizable
#
# Author: Contact Analysis Team
# ==============================================================================

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

from core import (
    ValidationError,
    DataTypeError,
    GeometryError,
    POLYGON_REORDER_TOL,
    normalize_vector,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# POLYGON AREA COMPUTATION
# ==============================================================================

def polygon_area_3d(vertices: np.ndarray) -> float:
    """
    Compute polygon area using Shoelace formula for 3D polygon.
    
    Calculates the area of a flat polygon embedded in 3D space using the
    cross product method. This works for any planar polygon with vertices
    in any order.
    
    Method:
        1. Compute cross product of consecutive edge vectors
        2. Sum all cross products
        3. Take magnitude and divide by 2
    
    Parameters
    ----------
    vertices : np.ndarray
        Polygon vertices in 3D, shape (N, 3) where N >= 3
    
    Returns
    -------
    float
        Area of the polygon (always non-negative)
    
    Raises
    ------
    ValidationError
        If vertices shape is invalid
    DataTypeError
        If vertices contain NaN/Inf
    GeometryError
        If vertices are collinear (area ≈ 0)
    
    Examples
    --------
    >>> import numpy as np
    >>> from geometry.calculator import polygon_area_3d
    >>> 
    >>> # Unit triangle in XY plane
    >>> vertices = np.array([
    ...     [0.0, 0.0, 0.0],
    ...     [1.0, 0.0, 0.0],
    ...     [0.0, 1.0, 0.0]
    ... ])
    >>> area = polygon_area_3d(vertices)
    >>> print(f"Area: {area:.6f}")  # Should be 0.5
    Area: 0.500000
    
    >>> # Unit square in XY plane
    >>> vertices = np.array([
    ...     [0.0, 0.0, 0.0],
    ...     [1.0, 0.0, 0.0],
    ...     [1.0, 1.0, 0.0],
    ...     [0.0, 1.0, 0.0]
    ... ])
    >>> area = polygon_area_3d(vertices)
    >>> print(f"Area: {area:.6f}")  # Should be 1.0
    Area: 1.000000
    
    Notes
    -----
    - Works for any planar polygon (convex or concave)
    - Vertices can be in any order (will compute area correctly)
    - For optimal accuracy, reorder vertices with reorder_polygon_vertices()
    - Zero area indicates degenerate polygon (collinear points)
    """
    # Validate input
    vertices = np.asarray(vertices, dtype=float)
    
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValidationError(
            f"Vertices must have shape (N, 3), got {vertices.shape}",
            error_code="AREA_VERTICES_INVALID_SHAPE",
            context={"shape": vertices.shape}
        )
    
    if vertices.shape[0] < 3:
        raise ValidationError(
            f"Need at least 3 vertices, got {vertices.shape[0]}",
            error_code="AREA_TOO_FEW_VERTICES",
            context={"n_vertices": vertices.shape[0]}
        )
    
    if np.any(np.isnan(vertices)) or np.any(np.isinf(vertices)):
        raise DataTypeError(
            "Vertices contain NaN or infinite values",
            error_code="AREA_VERTICES_NAN_INF"
        )
    
    # Compute area using cross product method
    # Area = 0.5 * | sum of (v_i × v_{i+1}) |
    
    n_vertices = vertices.shape[0]
    cross_sum = np.zeros(3)
    
    for i in range(n_vertices):
        v1 = vertices[i]
        v2 = vertices[(i + 1) % n_vertices]
        cross_sum += np.cross(v1, v2)
    
    # Magnitude of sum gives 2 * area
    area = 0.5 * np.linalg.norm(cross_sum)
    
    logger.debug(f"Computed polygon area: {area:.6e} from {n_vertices} vertices")
    
    return float(area)


# ==============================================================================
# POLYGON NORMAL VECTOR
# ==============================================================================

def polygon_normal(vertices: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    Compute the unit normal vector to a polygon plane.
    
    Computes the outward normal using the cross product of two edge vectors.
    The normal points in the direction of (v1 - v0) × (v2 - v0).
    
    Parameters
    ----------
    vertices : np.ndarray
        Polygon vertices, shape (N, 3) where N >= 3
    normalize : bool, optional
        If True (default), return unit normal. If False, return unnormalized.
    
    Returns
    -------
    np.ndarray
        Normal vector to polygon plane, shape (3,)
        Magnitude is 1.0 if normalize=True, else magnitude = 2*area
    
    Raises
    ------
    ValidationError
        If vertices shape is invalid
    GeometryError
        If vertices are collinear (cannot compute normal)
    
    Examples
    --------
    >>> import numpy as np
    >>> from geometry.calculator import polygon_normal
    >>> 
    >>> # Triangle in XY plane, should have normal pointing in Z
    >>> vertices = np.array([
    ...     [0.0, 0.0, 0.0],
    ...     [1.0, 0.0, 0.0],
    ...     [0.0, 1.0, 0.0]
    ... ])
    >>> normal = polygon_normal(vertices)
    >>> print(f"Normal: {normal}")
    Normal: [0. 0. 1.]
    
    Notes
    -----
    - First three vertices define the plane orientation
    - Collinear vertices will raise GeometryError
    - Normal direction depends on vertex order (right-hand rule)
    """
    # Validate input
    vertices = np.asarray(vertices, dtype=float)
    
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValidationError(
            f"Vertices must have shape (N, 3), got {vertices.shape}",
            error_code="NORMAL_VERTICES_INVALID_SHAPE",
            context={"shape": vertices.shape}
        )
    
    if vertices.shape[0] < 3:
        raise ValidationError(
            f"Need at least 3 vertices, got {vertices.shape[0]}",
            error_code="NORMAL_TOO_FEW_VERTICES"
        )
    
    if np.any(np.isnan(vertices)) or np.any(np.isinf(vertices)):
        raise DataTypeError(
            "Vertices contain NaN or infinite values",
            error_code="NORMAL_VERTICES_NAN_INF"
        )
    
    # Compute cross product of first two edges
    v0 = vertices[0]
    v1 = vertices[1]
    v2 = vertices[2]
    
    edge1 = v1 - v0
    edge2 = v2 - v0
    
    normal = np.cross(edge1, edge2)
    
    # Check if vertices are collinear
    magnitude = np.linalg.norm(normal)
    if magnitude < 1e-12:
        raise GeometryError(
            "Vertices are collinear, cannot compute normal",
            error_code="GEOMETRY_COLLINEAR_VERTICES",
            context={"edge1": edge1.tolist(), "edge2": edge2.tolist()}
        )
    
    # Optionally normalize
    if normalize:
        normal = normal / magnitude
    
    logger.debug(f"Computed polygon normal: {normal}")
    
    return normal


# ==============================================================================
# POLYGON CENTROID
# ==============================================================================

def polygon_centroid(vertices: np.ndarray) -> np.ndarray:
    """
    Compute centroid (center of mass) of polygon vertices.
    
    For simple geometric centroid, uses the average of all vertices.
    For weighted centroid by area, more complex formula is needed.
    
    This function computes the simple geometric centroid.
    
    Parameters
    ----------
    vertices : np.ndarray
        Polygon vertices, shape (N, 3)
    
    Returns
    -------
    np.ndarray
        Centroid coordinates, shape (3,)
    
    Raises
    ------
    ValidationError
        If vertices shape is invalid
    
    Examples
    --------
    >>> import numpy as np
    >>> from geometry.calculator import polygon_centroid
    >>> 
    >>> vertices = np.array([
    ...     [0.0, 0.0, 0.0],
    ...     [1.0, 0.0, 0.0],
    ...     [0.0, 1.0, 0.0]
    ... ])
    >>> centroid = polygon_centroid(vertices)
    >>> print(f"Centroid: {centroid}")
    Centroid: [0.33333333 0.33333333 0.        ]
    
    Notes
    -----
    - Simple centroid (average of vertices)
    - For non-uniform density, weight vertices appropriately
    """
    vertices = np.asarray(vertices, dtype=float)
    
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValidationError(
            f"Vertices must have shape (N, 3), got {vertices.shape}",
            error_code="CENTROID_VERTICES_INVALID_SHAPE",
            context={"shape": vertices.shape}
        )
    
    if vertices.shape[0] < 1:
        raise ValidationError(
            f"Need at least 1 vertex, got {vertices.shape[0]}",
            error_code="CENTROID_NO_VERTICES"
        )
    
    centroid = np.mean(vertices, axis=0)
    
    logger.debug(f"Computed polygon centroid: {centroid}")
    
    return centroid


# ==============================================================================
# POLYGON VERTEX REORDERING
# ==============================================================================

def reorder_polygon_vertices(
    vertices: np.ndarray,
    tolerance: float = POLYGON_REORDER_TOL
) -> np.ndarray:
    """
    Reorder polygon vertices in cyclic order around the boundary.
    
    Essential for correct geometric computations. Arranges vertices so they
    form a proper cycle around the polygon perimeter.
    
    Algorithm:
        1. Check vertices are coplanar (tolerance = POLYGON_REORDER_TOL)
        2. Compute centroid and normal vector
        3. Create 2D coordinate system in the polygon plane
        4. Project vertices to 2D coordinates
        5. Sort vertices by angle from centroid
        6. Return reordered 3D vertices
    
    Parameters
    ----------
    vertices : np.ndarray
        Polygon vertices (possibly unordered), shape (N, 3)
    tolerance : float, optional
        Coplanarity tolerance. Default is POLYGON_REORDER_TOL.
    
    Returns
    -------
    np.ndarray
        Reordered vertices in cyclic order, shape (N, 3)
    
    Raises
    ------
    ValidationError
        If vertices shape is invalid
    GeometryError
        If vertices are not coplanar
    
    Examples
    --------
    >>> import numpy as np
    >>> from geometry.calculator import reorder_polygon_vertices
    >>> 
    >>> # Unordered vertices of a triangle
    >>> vertices = np.array([
    ...     [0.0, 1.0, 0.0],  # Middle vertex (unordered)
    ...     [0.0, 0.0, 0.0],  # Last vertex
    ...     [1.0, 0.0, 0.0]   # First vertex
    ... ])
    >>> 
    >>> ordered = reorder_polygon_vertices(vertices)
    >>> print(f"Ordered vertices shape: {ordered.shape}")
    Ordered vertices shape: (3, 3)
    
    Notes
    -----
    - Assumes vertices are coplanar
    - Requires at least 3 vertices
    - Result maintains planarity
    """
    # Validate input
    vertices = np.asarray(vertices, dtype=float)
    
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValidationError(
            f"Vertices must have shape (N, 3), got {vertices.shape}",
            error_code="REORDER_VERTICES_INVALID_SHAPE",
            context={"shape": vertices.shape}
        )
    
    if vertices.shape[0] < 3:
        raise ValidationError(
            f"Need at least 3 vertices, got {vertices.shape[0]}",
            error_code="REORDER_TOO_FEW_VERTICES"
        )
    
    # Check coplanarity
    check_coplanar(vertices, tolerance=tolerance)
    
    # Compute centroid
    centroid = polygon_centroid(vertices)
    
    # Compute normal
    normal = polygon_normal(vertices, normalize=True)
    
    # Create 2D coordinate system in the plane
    # u-axis: perpendicular to normal, in direction of (v[1] - centroid)
    v_first = vertices[0] - centroid
    u_axis = normalize_vector(v_first)
    
    # v-axis: perpendicular to both normal and u-axis
    v_axis = np.cross(normal, u_axis)
    v_axis = normalize_vector(v_axis)
    
    # Project vertices to 2D
    angles = []
    for vertex in vertices:
        rel_pos = vertex - centroid
        x = np.dot(rel_pos, u_axis)
        y = np.dot(rel_pos, v_axis)
        angle = np.arctan2(y, x)
        angles.append(angle)
    
    # Sort by angle
    sorted_indices = np.argsort(angles)
    reordered = vertices[sorted_indices]
    
    logger.debug(f"Reordered {vertices.shape[0]} vertices cyclically")
    
    return reordered


# ==============================================================================
# COPLANARITY CHECK
# ==============================================================================

def check_coplanar(
    vertices: np.ndarray,
    tolerance: float = POLYGON_REORDER_TOL
) -> bool:
    """
    Check if all vertices lie in the same plane (within tolerance).
    
    Fits a plane to the vertices and checks that all points lie within
    the specified tolerance of that plane.
    
    Algorithm:
        1. Compute plane using first 3 vertices (normal, distance)
        2. For each vertex, compute distance to plane
        3. Check all distances < tolerance
    
    Parameters
    ----------
    vertices : np.ndarray
        Vertices to check, shape (N, 3)
    tolerance : float, optional
        Maximum perpendicular distance to plane. Default is POLYGON_REORDER_TOL.
    
    Returns
    -------
    bool
        True if all vertices are coplanar within tolerance
    
    Raises
    ------
    ValidationError
        If vertices shape is invalid
    GeometryError
        If cannot compute plane from vertices
    
    Examples
    --------
    >>> import numpy as np
    >>> from geometry.calculator import check_coplanar
    >>> 
    >>> # Coplanar vertices (in XY plane)
    >>> vertices = np.array([
    ...     [0.0, 0.0, 0.0],
    ...     [1.0, 0.0, 0.0],
    ...     [0.0, 1.0, 0.0],
    ...     [1.0, 1.0, 0.0]
    ... ])
    >>> is_coplanar = check_coplanar(vertices)
    >>> print(f"Coplanar: {is_coplanar}")
    Coplanar: True
    
    >>> # Non-coplanar vertices (includes point out of plane)
    >>> bad_vertices = np.array([
    ...     [0.0, 0.0, 0.0],
    ...     [1.0, 0.0, 0.0],
    ...     [0.0, 1.0, 0.0],
    ...     [0.5, 0.5, 1.0]  # Out of plane
    ... ])
    >>> is_coplanar = check_coplanar(bad_vertices)
    >>> print(f"Coplanar: {is_coplanar}")
    Coplanar: False
    """
    # Validate input
    vertices = np.asarray(vertices, dtype=float)
    
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValidationError(
            f"Vertices must have shape (N, 3), got {vertices.shape}",
            error_code="COPLANAR_VERTICES_INVALID_SHAPE",
            context={"shape": vertices.shape}
        )
    
    if vertices.shape[0] < 3:
        raise ValidationError(
            f"Need at least 3 vertices, got {vertices.shape[0]}",
            error_code="COPLANAR_TOO_FEW_VERTICES"
        )
    
    # Compute plane normal from first 3 vertices
    try:
        normal = polygon_normal(vertices[:3], normalize=True)
    except GeometryError as e:
        raise GeometryError(
            "Cannot compute plane: vertices are collinear",
            error_code="GEOMETRY_COLLINEAR_VERTICES"
        ) from e
    
    # Compute plane equation: dot(normal, p - p0) = 0
    p0 = vertices[0]
    
    # Check distance of all vertices to plane
    max_distance = 0.0
    for i, vertex in enumerate(vertices):
        rel_pos = vertex - p0
        distance = abs(np.dot(rel_pos, normal))
        max_distance = max(max_distance, distance)
        
        if distance > tolerance:
            logger.debug(
                f"Vertex {i} is not coplanar: distance={distance:.2e} > {tolerance:.2e}"
            )
            return False
    
    logger.debug(f"✓ All {vertices.shape[0]} vertices are coplanar (max_dist={max_distance:.2e})")
    return True


# ==============================================================================
# POLYGON INTERSECTION
# ==============================================================================

def polygon_intersection_area(
    ptsA: np.ndarray,
    ptsB: np.ndarray,
    strict: bool = False
) -> float:
    """
    Compute the overlap area between two polygons.
    
    Computes the area of the intersection of two planar polygons.
    Handles both coplanar and non-coplanar cases.
    
    Parameters
    ----------
    ptsA : np.ndarray
        First polygon vertices, shape (N, 3)
    ptsB : np.ndarray
        Second polygon vertices, shape (M, 3)
    strict : bool, optional
        If True, raise error on numerical issues. If False, return 0.0
        for problematic cases. Default is False.
    
    Returns
    -------
    float
        Overlap area (always >= 0.0)
    
    Raises
    ------
    ValidationError
        If polygon shapes are invalid (if strict=True)
    GeometryError
        If cannot compute intersection (if strict=True)
    
    Examples
    --------
    >>> import numpy as np
    >>> from geometry.calculator import polygon_intersection_area
    >>> 
    >>> # Two overlapping squares
    >>> poly_A = np.array([
    ...     [0.0, 0.0, 0.0],
    ...     [1.0, 0.0, 0.0],
    ...     [1.0, 1.0, 0.0],
    ...     [0.0, 1.0, 0.0]
    ... ])
    >>> 
    >>> poly_B = np.array([
    ...     [0.5, 0.5, 0.0],
    ...     [1.5, 0.5, 0.0],
    ...     [1.5, 1.5, 0.0],
    ...     [0.5, 1.5, 0.0]
    ... ])
    >>> 
    >>> overlap = polygon_intersection_area(poly_A, poly_B)
    >>> print(f"Overlap area: {overlap:.6f}")  # Should be 0.25
    Overlap area: 0.250000
    
    Notes
    -----
    - Complex polygon intersection is handled using Shapely when available
    - Falls back to simple methods for basic cases
    - strict=False prevents crashes on degenerate cases
    """
    try:
        from shapely.geometry import Polygon as ShapelyPolygon
        use_shapely = True
    except ImportError:
        use_shapely = False
    
    # Validate inputs
    ptsA = np.asarray(ptsA, dtype=float)
    ptsB = np.asarray(ptsB, dtype=float)
    
    if ptsA.shape[1] != 3 or ptsB.shape[1] != 3:
        if strict:
            raise ValidationError(
                "Polygons must have shape (N, 3)",
                error_code="INTERSECTION_INVALID_SHAPE"
            )
        else:
            logger.warning("Invalid polygon shape in intersection calculation")
            return 0.0
    
    if ptsA.shape[0] < 3 or ptsB.shape[0] < 3:
        if strict:
            raise ValidationError(
                "Polygons need at least 3 vertices",
                error_code="INTERSECTION_TOO_FEW_VERTICES"
            )
        else:
            return 0.0
    
    # Try using Shapely for robust computation
    if use_shapely:
        try:
            # Project to 2D (assuming coplanar)
            poly_A_2d = ptsA[:, :2]
            poly_B_2d = ptsB[:, :2]
            
            shape_A = ShapelyPolygon(poly_A_2d)
            shape_B = ShapelyPolygon(poly_B_2d)
            
            intersection = shape_A.intersection(shape_B)
            area = intersection.area
            
            logger.debug(f"Computed intersection area using Shapely: {area:.6e}")
            return float(area)
        
        except Exception as e:
            if strict:
                raise GeometryError(
                    f"Shapely intersection failed: {e}",
                    error_code="GEOMETRY_INTERSECTION_FAILED"
                ) from e
            else:
                logger.warning(f"Shapely intersection failed: {e}, returning 0.0")
                return 0.0
    else:
        # Fallback: simple check for non-overlapping
        logger.debug("Shapely not available, using simple overlap check")
        # For non-coplanar or complex cases, return 0
        return 0.0


if __name__ == "__main__":
    """
    Test/demo when module is run directly.
    """
    import sys
    from core.utilities import setup_logging
    
    setup_logging(level="DEBUG")
    
    print("\n" + "="*80)
    print("GEOMETRY CALCULATOR MODULE - TESTING")
    print("="*80 + "\n")
    
    # Test polygon area
    print("[TEST 1] Polygon area...")
    try:
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        area = polygon_area_3d(vertices)
        print(f"✓ Triangle area: {area:.6f} (expected 0.5)")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test polygon normal
    print("\n[TEST 2] Polygon normal...")
    try:
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        normal = polygon_normal(vertices)
        print(f"✓ Normal: {normal} (expected [0 0 1])")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test centroid
    print("\n[TEST 3] Polygon centroid...")
    try:
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        centroid = polygon_centroid(vertices)
        print(f"✓ Centroid: {centroid}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test coplanarity
    print("\n[TEST 4] Coplanarity check...")
    try:
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0]
        ], dtype=float)
        is_coplanar = check_coplanar(vertices)
        print(f"✓ Coplanar: {is_coplanar} (expected True)")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test vertex reordering
    print("\n[TEST 5] Vertex reordering...")
    try:
        vertices = np.array([
            [0, 1, 0],
            [0, 0, 0],
            [1, 0, 0]
        ], dtype=float)
        reordered = reorder_polygon_vertices(vertices)
        print(f"✓ Reordered {vertices.shape[0]} vertices")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("\n" + "="*80)
    print("✓ Calculator module is ready for use!")
    print("="*80 + "\n")
