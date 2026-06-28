"""
Pure geometric helpers for polygonal particle faces.

FACE GEOMETRY OVERVIEW
======================
This module provides pure geometric calculations for polygonal faces.

Functions handle:
1. Computing face centers (centroids)
2. Computing face normals (outward directions)

COMPATIBILITY BEHAVIOR
======================
Important compatibility notes:

1. Face center is the ARITHMETIC MEAN of vertices
   center = mean(vertices) for all vertices in face
   Not centroid (would weight by area)
   Simple averaging approach

2. Faces with fewer than 3 vertices are REJECTED
   ValueError raised immediately
   Ensures valid polygons

3. Face normal computed from FIRST NON-COLLINEAR TRIPLET
   Not just first 3 vertices
   Searches systematically for valid triplet
   Uses first vertex as common origin (robust method)

4. Normal accepted only when norm > 1e-12
   Below this is considered numerical noise
   Threshold avoids numerical precision issues

5. Returns normalized unit normals (|n| = 1)
   Unit length ensures proper dot products
   Perpendicular to face plane

Return as numpy array

Uses double loop to find first valid triplet:
    - Systematic search
    - First found is returned (no random element)
    - Deterministic behavior
"""

from __future__ import annotations

import numpy as np


def face_centers(vertex_array, faces):
    """
    Compute the centre of every polygonal face.

    FUNCTION PURPOSE
    ================
    Computes the arithmetic mean (center) of vertices for each face.

    For a face with vertices v0, v1, ..., vn:
        center = (1/N) * sum(vi)

    This is the centroid of the vertex set (not weighted by area).

    Parameters
    ----------
    vertex_array : ndarray, shape (N_vertices, 3)
        Array of all particle vertices.


    faces : list of lists
        Face definitions (vertex indices).
        Each element is a face (list of indices).

    Returns
    -------
    ndarray, shape (N_faces, 3)
        Face centers (centroids).
        One row per face.
        Rows in same order as input faces.
        
        Each row: [center_x, center_y, center_z]
    """

    # =========================================================================
    # INITIALIZE RESULTS LIST
    # =========================================================================
    centers = []

    # =========================================================================
    # LOOP OVER FACES
    # =========================================================================
    for face_id, face in enumerate(faces):
        # Validate: face must have at least 3 vertices
        if len(face) < 3:
            raise ValueError(f"Face {face_id} has fewer than 3 vertices: {face}")
        # Extract vertices for this face
        # Compute mean (centroid) of face vertices
        centers.append(vertex_array[face].mean(axis=0))

    # =========================================================================
    # RETURN RESULTS
    # =========================================================================
    return np.asarray(centers, dtype=float)


def unit_face_normal(face_points, face_id=None):
    """
    Compute a robust unit normal for one polygonal face.

    FUNCTION PURPOSE
    ================
    Computes the outward normal direction of a polygonal face.

    For a planar polygon with vertices in counter-clockwise order
    (when viewed from outside):
        normal = (v1 - v0) * (v2 - v0)
        unit_normal = normal / |normal|

    Result is a unit vector (length 1.0) perpendicular to face plane.

    ROBUSTNESS STRATEGY
    ===================
    Instead of assuming first three vertices are non-collinear,
    this function searches for the FIRST VALID TRIPLET.

    Solution: Systematic search
        1. Fix first vertex p0 as common origin
        2. Try pairs of other vertices (a, b)
        3. Compute vectors v1 = pts[a] - p0, v2 = pts[b] - p0
        4. Compute cross product: n = v1 * v2
        5. Check magnitude: |n| > 1e-12
        6. Accept if valid, otherwise try next pair

    This always finds a valid triplet (unless all vertices collinear).

    Parameters
    ----------
    face_points : ndarray, shape (N, 3)
        Face-vertex coordinates.
        Each row is one vertex: [x, y, z]
        N >= 3 (at least 3 vertices required)

    face_id : int or str, optional
        Face identifier for error messages.
        Used when raising ValueError.
        Default: None (no ID in error message)

    Returns
    -------
    ndarray, shape (3,)
        Unit normal vector.
        Magnitude: exactly 1.0
        Direction: perpendicular to face plane
        
        For CCW vertices (viewed from outside):
        Normal points outward (away from polyhedron interior).
    """

    # =========================================================================
    # STEP 1: VALIDATE INPUT SHAPE
    # =========================================================================
    # Convert to numpy array for consistent handling
    face_points = np.asarray(face_points, dtype=float)

    # Check shape is exactly (N, 3)
    if face_points.ndim != 2 or face_points.shape[1] != 3:
        raise ValueError(f"face_points must have shape (N, 3), got {face_points.shape}.")

    # =========================================================================
    # STEP 2: VALIDATE VERTEX COUNT
    # =========================================================================
    # At least 3 vertices required to define a plane
    if face_points.shape[0] < 3:
        raise ValueError(f"Face {face_id} has fewer than 3 vertices.")

    # =========================================================================
    # STEP 3: SEARCH FOR FIRST VALID TRIPLET
    # =========================================================================
    # Set first vertex as common origin 
    p0 = face_points[0]

    # Nested loop: try all pairs of other vertices
    # a ranges from 1 to N-2 (ensures b > a)
    # b ranges from a+1 to N-1 (ensures distinct vertices)
    for a in range(1, face_points.shape[0] - 1):
        for b in range(a + 1, face_points.shape[0]):
            # Compute vectors from origin to two vertices
            # v1 = pts[a] - p0
            # v2 = pts[b] - p0
            v1 = face_points[a] - p0
            v2 = face_points[b] - p0
            # Compute cross product (perpendicular vector)
            # normal = v1 * v2
            normal = np.cross(v1, v2)
            # Check magnitude to ensure non-collinear
            # Threshold 1e-12 avoids numerical precision issues
            norm = np.linalg.norm(normal)
            # Accept first valid triplet
            if norm > 1.0e-12:
                # Return normalized unit vector
                # unit_normal = normal / |normal|
                return normal / norm

    # =========================================================================
    # STEP 4: NO VALID TRIPLET FOUND
    # =========================================================================
    raise ValueError(f"Could not compute valid normal for face {face_id}. All tested vertex triplets appear collinear.")