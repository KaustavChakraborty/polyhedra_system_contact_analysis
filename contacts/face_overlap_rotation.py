"""
Minimum rotation matrix between two vectors.

ROTATION OVERVIEW
=================
This module computes rotation matrix between two 3D vectors.

Given vectors a and b, finds rotation R such that R @ a = b.
Chooses minimum rotation (smallest angle).

PURPOSE
=======
In face-overlap pipeline:

Two face normals need to be aligned via rotation
This function computes the rotation matrix

Used by: minimum_rotation_to_parallelize_normals()
Which passes normalized normals to this function

MATHEMATICAL FOUNDATION
=======================

Finding minimum rotation:

Given unit vectors a, b:
    Want to find R that rotates a toward b
    Minimize rotation angle

Angle θ between vectors:
    cos(θ) = a · b
    θ = arccos(a · b)

Rotation axis:
    Axis of rotation = a cross b (perpendicular to both)
    Direction where rotation happens

Algorithm handles three cases:

Case 1: Already aligned (a = b, dot = 1)
    No rotation needed
    Return identity matrix I

Case 2: Anti-parallel (a = -b, dot = -1)
    Need 180° rotation around perpendicular axis
    Special formula (simpler than general case)

Case 3: General case (all other angles)
    Use Rodrigues' rotation formula
    Combines axis and angle into matrix

CASE 1: ALREADY ALIGNED
=======================
Condition: a dot b > 1 - tolerance

Meaning:
    Dot product = 1 (maximum alignment)
    Angle = 0 deg
    Vectors point same direction

Action:
    Return identity matrix I (no rotation)
    R @ a = a (unchanged)

CASE 2: ANTI-PARALLEL
=====================
Condition: a \dot b < -1 + tolerance

Meaning:
    Dot product = -1 (maximum misalignment)
    Angle = 180°
    Vectors point opposite directions

Problem:
    Want to rotate a by 180° to align with b
    Infinite rotation axes perpendicular to a
    Which axis to choose?

Solution (stability):
    Choose axis perpendicular to a
    But avoid collinearity with a
    Use reference vector (x-axis or y-axis)
    If reference parallel to a: switch reference

Algorithm:
    1. Choose reference vector
       Start with [1, 0, 0]
       If parallel to a: use [0, 1, 0]
    2. Compute axis: axis = a cross reference
    3. Normalize axis
    4. Use 180° rotation formula: R = -I + 2 * axis * axis^T

Why this formula?
    For θ = 180°:
    cos(180°) = -1
    sin(180°) = 0
    Rodrigues simplifies to: R = -I + 2 * n * n^T
    Where n = normalized axis

Intuition:
    Reflection through plane perpendicular to axis
    Then flip sign
    Net effect: 180° rotation

CASE 3: GENERAL CASE
====================
Condition: all other angles (not aligned, not anti-parallel)

Algorithm: Rodrigues' rotation formula

Given:
    Unit vectors a, b
    Angle \theta where cos(\thaeta) = a \dot b
    Axis n = (a cross b) / |a cross b|

Rodrigues formula:
    R = I + sin(\theta) * [n]_x + (1 - cos(\theta)) * [n]_x^2
    
Where [n]_x is skew-symmetric cross-product matrix:
    [n]_x = [[0, -nz, ny],
             [nz, 0, -nx],
             [-ny, nx, 0]]
"""

from __future__ import annotations
import numpy as np


def rotation_matrix_from_vectors(a, b, tol=1.0e-12):
    """
    Return the minimum rotation matrix mapping vector ``a`` to vector ``b``.

    FUNCTION PURPOSE
    ================
    Computes rotation matrix that rotates vector a to align with vector b.

    Returns smallest rotation (minimum angle).
    Handles three cases: aligned, anti-parallel, general.

    ALGORITHM
    =========
    1. Normalize both input vectors
    2. Compute dot product (cosine of angle)
    3. Branch based on dot product:
       Case 1: Already aligned (dot = 1) => identity
       Case 2: Anti-parallel (dot = -1) => 180° rotation
       Case 3: General case => Rodrigues formula
    4. Return rotation matrix

    Parameters
    ----------
    a : array-like, shape (3,)
        Source vector to rotate.
        [ax, ay, az]

    b : array-like, shape (3,)
        Target vector to rotate toward.
        [bx, by, bz]

    tol : float, optional
        Numerical tolerance for degeneracy checks.
        
        Default: 1.0e-12
    """

    # =========================================================================
    # STEP 1: CONVERT INPUTS TO NUMPY FLOAT ARRAYS
    # =========================================================================
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    # =========================================================================
    # STEP 2: COMPUTE VECTOR MAGNITUDES
    # =========================================================================
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)

    # =========================================================================
    # STEP 3: VALIDATE NON-ZERO VECTORS
    # =========================================================================
    if a_norm < tol or b_norm < tol: raise ValueError("Cannot build rotation from zero vector.")

    # =========================================================================
    # STEP 4: NORMALIZE VECTORS TO UNIT LENGTH
    # =========================================================================
    a = a / a_norm; b = b / b_norm

    # =========================================================================
    # STEP 5: COMPUTE DOT PRODUCT (COSINE OF ANGLE)
    # =========================================================================
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))

    # =========================================================================
    # CASE 1: ALREADY ALIGNED
    # =========================================================================
    if dot > 1.0 - tol: return np.eye(3)

    # =========================================================================
    # CASE 2: ANTI-PARALLEL (180 deg ROTATION)
    # =========================================================================
    if dot < -1.0 + tol:
        # Need perpendicular axis for 180° rotation
        # Choose reference vector (x-axis or y-axis)
        reference = np.array([1.0, 0.0, 0.0])
        # If reference parallel to a: switch to y-axis
        # This ensures cross product is non-zero
        if abs(np.dot(a, reference)) > 0.9: reference = np.array([0.0, 1.0, 0.0])

        # Compute perpendicular axis: a cross reference
        axis = np.cross(a, reference)

        # Normalize axis to unit length
        axis = axis / np.linalg.norm(axis)

        # 180 deg rotation formula: R = -I + 2 * axis * axis^T
        # This rotates around axis by 180 deg
        return -np.eye(3) + 2.0 * np.outer(axis, axis)

    # =========================================================================
    # CASE 3: GENERAL CASE (RODRIGUES' ROTATION FORMULA)
    # =========================================================================
    # Compute rotation axis: a cross b
    # Cross product is perpendicular to both vectors
    # Direction indicates rotation axis
    axis = np.cross(a, b)

    # Compute magnitude of cross product
    # |a cross b| = |a| |b| sin(theta) = sin(theta) for unit vectors
    sine_theta = np.linalg.norm(axis)

    if sine_theta < tol: return np.eye(3)

    # Normalize axis to unit length
    axis = axis / sine_theta

    # =========================================================================
    # STEP 6: BUILD SKEW-SYMMETRIC CROSS-PRODUCT MATRIX
    # =========================================================================
    cross_matrix = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]], dtype=float)

    # =========================================================================
    # STEP 7: COMPUTE ROTATION ANGLE
    # =========================================================================
    # theta = arccos(a \dot b)
    theta = np.arccos(dot)

    # =========================================================================
    # STEP 8: APPLY RODRIGUES' ROTATION FORMULA
    # =========================================================================
    rotation = np.eye(3) + np.sin(theta) * cross_matrix + (1.0 - np.cos(theta)) * (cross_matrix @ cross_matrix)

    # =========================================================================
    # STEP 9: RETURN ROTATION MATRIX
    # =========================================================================
    return rotation


# Compatibility alias matching the original private helper.
_rotation_matrix_from_vectors = rotation_matrix_from_vectors

