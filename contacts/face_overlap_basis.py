"""
Construction of an orthonormal in-plane basis from a plane normal.

PLANE BASIS OVERVIEW
====================
This module constructs two orthonormal basis vectors from a plane normal.

Given a plane normal (perpendicular direction),
computes two unit vectors in the plane (u, v).

PURPOSE
=======
Plane representation needs 3 orthonormal directions:
1. Normal (n): perpendicular to plane
2. u: in-plane direction 1
3. v: in-plane direction 2

This function computes u, v from n.

Uses in face overlap pipeline:
- Define 2D coordinate system in plane
- Project 3D points onto plane
- Transform between 3D and 2D coordinates

ALGORITHM STRATEGY
==================
Construct basis using cross products:

Step 1: Normalize input normal
    Ensure |n| = 1.0
    Handle unnormalized input

Step 2: Choose reference vector
    Start with x-axis: ref = (1, 0, 0)
    If n parallel to x-axis (|n dot x| > 0.9):
    Use y-axis instead: ref = (0, 1, 0)
    Ensures ref not parallel to n

Step 3: Compute first basis vector
    u = n * ref
    Perpendicular to both n and ref
    Normalize to unit length

Step 4: Compute second basis vector
    v = n * u
    Perpendicular to n and u
    Normalize to unit length

Result: orthonormal basis (u, v) in plane
"""

from __future__ import annotations

import numpy as np


def plane_basis_from_normal(normal, tol=1.0e-12):
    """
    Construct two orthonormal vectors spanning a plane.

    FUNCTION PURPOSE
    ================
    Creates coordinate system for a plane given its normal.

    Given normal direction (perpendicular to plane),
    returns two orthonormal basis vectors (u, v) in the plane.

    These vectors define a 2D coordinate system for the plane.

    ALGORITHM
    =========
    1. Normalize input normal to unit length
    2. Choose reference vector (x-axis or y-axis)
    3. Compute first basis vector: u = n cross ref
    4. Compute second basis vector: v = n cross u
    5. Normalize both basis vectors
    6. Return u, v

    Parameters
    ----------
    normal : array-like, shape (3,)
        Plane-normal vector.

    tol : float, optional
        Degeneracy tolerance.
        Minimum magnitude for valid vectors.

    Returns
    -------
    u : ndarray, shape (3,)
        First in-plane basis vector.
        Unit length: exactly 1.0
        Perpendicular to normal.

    v : ndarray, shape (3,)
        Second in-plane basis vector.
        Unit length: exactly 1.0
        Perpendicular to normal and u.
    """

    # =========================================================================
    # STEP 1: CONVERT AND NORMALIZE INPUT NORMAL
    # =========================================================================
    normal = np.asarray(normal, dtype=float)

    # Compute magnitude of input normal
    normal_norm = np.linalg.norm(normal)

    if normal_norm < tol:
        raise ValueError(
            "Cannot construct plane basis from zero normal."
        )

    # Normalize to unit length
    # n = normal / |normal|
    n = normal / normal_norm

    # =========================================================================
    # STEP 2: CHOOSE REFERENCE VECTOR
    # =========================================================================
    # Start with x-axis as reference
    reference = np.array([1.0, 0.0, 0.0])

    # Check if normal parallel to x-axis
    # Parallel means: |n dot x_axis| > 0.9 (nearly aligned/opposite)
    # If parallel: switch to y-axis (perpendicular to x-axis)
    if abs(np.dot(n, reference)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])

    # =========================================================================
    # STEP 3: COMPUTE FIRST IN-PLANE BASIS VECTOR
    # =========================================================================
    # u = n cross reference
    # This vector is perpendicular to both n and reference
    # Magnitude depends on angle between n and reference
    u = np.cross(n, reference)

    # Compute magnitude of u
    u_norm = np.linalg.norm(u)

    # Check if first basis vector is non-degenerate
    if u_norm < tol:
        raise ValueError(
            "Failed to construct first in-plane basis vector."
        )

    # Normalize to unit length
    u = u / u_norm

    # =========================================================================
    # STEP 4: COMPUTE SECOND IN-PLANE BASIS VECTOR
    # =========================================================================
    # v = n cross u
    # This vector is perpendicular to both n and u
    # Should always have good magnitude if u is good
    v = np.cross(n, u)

    # Compute magnitude of v
    v_norm = np.linalg.norm(v)

    # Check if second basis vector is non-degenerate
    if v_norm < tol:
        raise ValueError(
            "Failed to construct second in-plane basis vector."
        )

    # Normalize to unit length
    v = v / v_norm

    # =========================================================================
    # STEP 5: RETURN ORTHONORMAL BASIS
    # =========================================================================
    # Return (u, v) in this exact order
    # Both unit vectors perpendicular to n and to each other
    # u cross v = n (right-handed)
    return u, v


# ============================================================================
# COMPATIBILITY ALIAS
# ============================================================================
# Retain original private naming for backward compatibility
_plane_basis_from_normal = plane_basis_from_normal