"""
Combined plane data for a three-dimensional polygon.

PLANE DATA OVERVIEW
===================
This module computes complete plane information from a polygon.

Creates 4 values describing the polygon's plane:
1. Centroid (center point)
2. Normal (direction perpendicular to plane)
3. In-plane basis vector u
4. In-plane basis vector v

PURPOSE
=======
Combines multiple geometric computations into single function.

Instead of calling 3 separate functions:
    centroid = polygon.mean()
    normal = find_unit_normal()
    u, v = plane_basis_from_normal()

Call one function:
    centroid, normal, u, v = plane_data(polygon, name)

More convenient and reduces redundant computation.

PLANE REPRESENTATION
====================
A plane is defined by:

Point on plane:
    centroid = arithmetic mean of all vertices

Normal direction:
    n = unit vector perpendicular to plane
    Points outward 

In-plane basis:
    u, v = orthonormal vectors in the plane
    span the plane (perpendicular to n)
    u cross v = n (right-hand rule)

This representation enables:
    3D => 2D projection (using centroid, u, v)
    Plane operations (using normal, centroid)
    Coordinate transformations

ALGORITHM
=========
Four computational steps:

1. Validate polygon
   as_3d_polygon(polygon, name)
   Ensures: shape (N, 3), N >= 3, all finite

2. Compute centroid
   centroid = polygon.mean(axis=0)

3. Compute unit normal
   normal = find_unit_normal(polygon, name)
   First non-collinear triplet cross product
   Normalized to unit length

4. Compute basis vectors
   u, v = plane_basis_from_normal(normal)
   Two orthonormal vectors in plane
   Perpendicular to normal

Returns: (centroid, normal, u, v)

RETURN VALUES EXPLAINED
=======================
Centroid: (3,) array
    Arithmetic mean of polygon vertices
    Position in 3D space
    Center of polygon

Normal: (3,) array
    Unit vector perpendicular to plane
    Length: exactly 1.0
    Orientation: right-hand rule (CCW vertices)

u: (3,) array
    First in-plane basis vector
    Unit length: exactly 1.0
    Perpendicular to normal

v: (3,) array
    Second in-plane basis vector
    Unit length: exactly 1.0
    Perpendicular to normal and u
"""

from __future__ import annotations


from .face_overlap_basis import plane_basis_from_normal
from .face_overlap_geometry import as_3d_polygon, find_unit_normal


def plane_data(polygon, name):
    """
    Compute centroid, unit normal, and in-plane basis for a polygon.

    FUNCTION PURPOSE
    ================
    Computes complete plane representation from polygon vertices.

    Parameters
    ----------
    polygon : array-like
        Three-dimensional polygon vertices.
        Shape: (N, 3) where N >= 3
        Each row: [x, y, z]
        
        Will be validated with as_3d_polygon().

    name : str
        Label for error messages and validation.
        Forwarded to all sub-functions.

    Returns
    -------
    centroid : ndarray, shape (3,)
    normal : ndarray, shape (3,)
        Unit normal vector.
    u : ndarray, shape (3,)
        First in-plane basis vector.

    v : ndarray, shape (3,)
        Second in-plane basis vector.
    """

    # =========================================================================
    # STEP 1: VALIDATE POLYGON
    # =========================================================================
    polygon = as_3d_polygon(polygon, name)

    # =========================================================================
    # STEP 2: COMPUTE CENTROID
    # =========================================================================
    # Arithmetic mean of all vertices
    centroid = polygon.mean(axis=0)

    # =========================================================================
    # STEP 3: COMPUTE UNIT NORMAL
    # =========================================================================
    # First non-collinear triplet cross product
    normal = find_unit_normal(polygon, name)

    # =========================================================================
    # STEP 4: COMPUTE IN-PLANE BASIS
    # =========================================================================
    # Two orthonormal vectors (u, v) in the plane
    u, v = plane_basis_from_normal(normal)

    # =========================================================================
    # STEP 5: RETURN RESULTS
    # =========================================================================
    # Return in exact order: centroid, normal, u, v
    return centroid, normal, u, v


# ============================================================================
# COMPATIBILITY ALIAS
# ============================================================================
# Retain original private naming for backward compatibility
_plane_data = plane_data

