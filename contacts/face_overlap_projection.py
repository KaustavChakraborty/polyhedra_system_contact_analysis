"""
Orthogonal projection of three-dimensional points onto a plane.

PROJECTION OVERVIEW
===================
This module projects 3D points onto a plane.

Converts 3D coordinates to closest points on a plane.

PURPOSE
=======
Face-overlap pipeline needs to:
1. Rotate neighbor face to be parallel to central
2. Project both faces onto central's plane
3. Work with projected 2D coordinates

This function handles step 2: projection.

MATHEMATICAL FOUNDATION
=======================
Orthogonal projection definition:

A plane is defined by:
    Point on plane: P0
    Normal direction: n (unit vector)
    
    Plane equation: (X - P0) \dot n = 0
    
Projection of point p onto plane:
    Find closest point on plane
    Use distance along normal direction
    
    Distance: d = (p - P0) \dot n
    Projected: p' = p - d*n
    
    Where n is unit normal (|n| = 1)

ALGORITHM
=========
Five computational steps:

1. Convert inputs to numpy float arrays

2. Normalize the plane normal
   Ensures |n| = 1.0
   Required for distance formula

3. Compute signed distances
   d = (points - P0) \dot n
   For all N points simultaneously

4. Compute projected points
   p' = p - d*n
   Subtract distance*normal from each point

5. Return projected points
   Same shape as input
   Points now on plane
"""

from __future__ import annotations

import numpy as np


def project_points_to_plane(points, plane_point, plane_normal):
    """
    Orthogonally project three-dimensional points onto a plane.

    Parameters
    ----------
    points : ndarray, shape (N, 3)
        Three-dimensional point coordinates.
        Each row: one point [x, y, z]
        N = number of points

    plane_point : ndarray, shape (3,)
        One point lying on the target plane.
        [x, y, z] coordinates

    plane_normal : ndarray, shape (3,)
        Normal vector of the target plane.
        [nx, ny, nz] direction

    Returns
    -------
    ndarray, shape (N, 3)
        Projected point coordinates.
        Same shape as input points.
        Each point moved to plane along normal.
    """

    # =========================================================================
    # STEP 1: CONVERT INPUTS TO NUMPY FLOAT ARRAYS
    # =========================================================================
    points = np.asarray(points, dtype=float)
    plane_point = np.asarray(plane_point, dtype=float)
    normal = np.asarray(plane_normal, dtype=float)

    # =========================================================================
    # STEP 2: NORMALIZE PLANE NORMAL TO UNIT LENGTH
    # =========================================================================
    normal = normal / np.linalg.norm(normal)

    # =========================================================================
    # STEP 3: COMPUTE SIGNED DISTANCES
    # =========================================================================
    signed_distances = (points - plane_point) @ normal

    # =========================================================================
    # STEP 4: COMPUTE PROJECTED POINTS
    # =========================================================================
    return points - signed_distances[:, None] * normal[None, :]


# Compatibility alias matching the original private helper.
_project_points_to_plane = project_points_to_plane