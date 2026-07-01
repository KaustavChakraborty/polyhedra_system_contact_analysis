"""
Minimum rotation required to make two plane normals parallel.

PARALLELIZATION OVERVIEW
=========================
This module finds rotation to make two normals parallel.

Given two plane normals, computes the smallest rotation
that aligns one with the other.

PURPOSE
=======
In face-overlap pipeline:

Two faces have different orientations (different normals)
Need to rotate one face to be parallel to the other

This function computes the rotation matrix.

Key insight: "parallel" means same direction OR opposite direction
Algorithm chooses direction to minimize rotation angle.

ROTATIONS EXPLAINED
===================
Types of rotations (qualitatively):

Dot product > 0 (acute angle < 90°):
    Normals point generally same direction
    Angle 0° to 90°
    Small rotation needed

Dot product = 0 (right angle):
    Normals perpendicular
    Angle exactly 90°
    Medium rotation needed

Dot product < 0 (obtuse angle > 90°):
    Normals point generally opposite
    Angle 90° to 180°
    Large rotation needed

Minimization:
    If angle > 90°: use opposite target
    Reduces to < 90°

Result: always rotate by smallest angle (< 90°).

ALGORITHM FLOW
==============
1. Convert inputs to numpy arrays
2. Normalize both normals
3. Compute dot product
4. Choose target:
   If dot >= 0: target = n_reference
   If dot < 0: target = -n_reference
5. Compute rotation to align n_from with target
6. Return rotation matrix

Step 5 delegated to rotation_matrix_from_vectors().

INTEGRATION WITH ROTATION_MATRIX_FROM_VECTORS
============================================
This function delegates final rotation to:
    rotation_matrix_from_vectors(n_from, target)

That function:
    Computes axis of rotation (n_from cross target)
    Computes angle from dot product
    Creates rotation matrix (Rodrigues formula or similar)
    Returns 3x3 rotation matrix

This function just chooses:
    Which target (n_reference or -n_reference)
    Based on dot product sign

"""

from __future__ import annotations

import numpy as np

# Import rotation computation function
from .face_overlap_rotation import rotation_matrix_from_vectors


def minimum_rotation_to_parallelize_normals(n_from, n_reference):
    """
    Build the minimum rotation making ``n_from`` parallel to ``n_reference``.

    FUNCTION PURPOSE
    ================
    Computes rotation matrix to align two plane normals.

    Given n_from (source normal) and n_reference (target normal),
    finds minimum rotation that makes n_from parallel to n_reference.

    "Parallel" means same or opposite direction (colinear).
    Minimizes rotation angle by choosing appropriate target direction.

    ALGORITHM
    =========
    1. Normalize both input normals
    2. Compute dot product
    3. Choose target direction:
       - If dot >= 0: use n_reference (same side)
       - If dot < 0: use -n_reference (flip to same side)
    4. Compute rotation from n_from to target
    5. Return rotation matrix

    Parameters
    ----------
    n_from : array-like, shape (3,)
        Normal vector to rotate.
        [nx, ny, nz] direction
        

    n_reference : array-like, shape (3,)
        Reference normal defining target orientation.
        [nx, ny, nz] direction

    Returns
    -------
    ndarray, shape (3, 3)
        Rotation matrix returned by rotation_matrix_from_vectors().
    """

    # =========================================================================
    # STEP 1: CONVERT INPUTS TO NUMPY FLOAT ARRAYS
    # =========================================================================
    n_from = np.asarray(n_from, dtype=float)
    n_reference = np.asarray(n_reference, dtype=float)

    # =========================================================================
    # STEP 2: NORMALIZE BOTH NORMALS TO UNIT LENGTH
    # =========================================================================
    n_from = n_from / np.linalg.norm(n_from)
    n_reference = n_reference / np.linalg.norm(n_reference)

    # =========================================================================
    # STEP 3: COMPUTE DOT PRODUCT AND CHOOSE TARGET
    # =========================================================================
    # Dot product tells us alignment
    # Positive: same direction
    # Negative: opposite direction
    # Zero: perpendicular (edge case)
    dot = float(np.dot(n_from, n_reference))


    # Choose target direction to minimize rotation
    target = n_reference if dot >= 0.0 else -n_reference


    # =========================================================================
    # STEP 4: COMPUTE ROTATION TO ALIGN n_from WITH TARGET
    # =========================================================================
    # Delegate to rotation_matrix_from_vectors()
    # That function:
    #   Computes axis of rotation
    #   Computes angle from dot product
    #   Creates rotation matrix
    return rotation_matrix_from_vectors(n_from, target)


# Compatibility alias matching the original private helper.
_minimum_rotation_to_parallelize_normals = minimum_rotation_to_parallelize_normals