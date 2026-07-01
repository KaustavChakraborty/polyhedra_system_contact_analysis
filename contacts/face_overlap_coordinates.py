"""
Transformation of three-dimensional points to two-dimensional coordinates.

COORDINATE TRANSFORMATION OVERVIEW
===================================
This module projects 3D points onto 2D plane using orthonormal basis.

Converts points from 3D to local 2D coordinate system.

PURPOSE
=======
In face-overlap pipeline:

After projecting 3D faces to plane:
- Points are still in 3D coordinates
- Need 2D coordinates for polygon operations
- This function transforms 3D => 2D

Extracts coordinates in plane's local basis.

COORDINATE SYSTEMS
==================
Two coordinate systems involved:

Global (3D):
    Origin at arbitrary point
    Axes: x, y, z (global frame)
    Points: [X, Y, Z]

Local (2D):
    Origin at plane point
    Axes: u, v (plane basis vectors)
    Points: [u_coord, v_coord]

Transformation:
    3D point in global frame => 2D point in local frame
    Projects onto plane basis

PLANE REPRESENTATION
====================
Plane defined by:
    origin: O (point on plane)
    u_axis: u (unit vector in plane)
    v_axis: v (unit vector in plane, orthogonal to u)

Note:
    u and v are orthonormal (unit length, perpendicular)
    Origin O is on the plane

LOCAL COORDINATE COMPUTATION
============================
For point P in 3D global frame:

1. Translate to plane origin:
    P_relative = P - O

2. Project onto u-axis:
    u_coord = P_relative \dot u

3. Project onto v-axis:
    v_coord = P_relative \dot v

4. 2D coordinate:
    [u_coord, v_coord]

Result:
    3D point => 2D point
    Discards component along plane normal
    Keeps components in plane
"""

from __future__ import annotations

import numpy as np


def project_to_2d(points, origin, u, v):
    """
    Transform three-dimensional points to two-dimensional coordinates on a plane.

    Parameters
    ----------
    points : array-like, shape (N, 3)
        Three-dimensional point coordinates.
        Each row: one point [X, Y, Z]
        N = number of points

    origin : array-like, shape (3,)
        Reference point on the plane.
        [Ox, Oy, Oz]

    u_axis : array-like, shape (3,)
        Unit basis vector (in-plane).
        [ux, uy, uz]

    v_axis : array-like, shape (3,)
        Unit basis vector (in-plane).
        [vx, vy, vz]

    Returns
    -------
    ndarray, shape (N, 2)
        Two-dimensional coordinates in plane.
        Each row: one point [u_coord, v_coord]
    """

    # =========================================================================
    # STEP 1: CONVERT INPUTS TO NUMPY FLOAT ARRAYS
    # =========================================================================
    points = np.asarray(points, dtype=float)
    origin = np.asarray(origin, dtype=float)

    # =========================================================================
    # STEP 2: TRANSLATE POINTS TO PLANE ORIGIN
    # =========================================================================
    shifted = points - origin

    # =========================================================================
    # STEP 3: PROJECT ONTO U-AXIS and V-AXIS
    # =========================================================================
    return np.column_stack([shifted @ u, shifted @ v])


# Compatibility alias matching the original private helper.
_project_to_2d = project_to_2d