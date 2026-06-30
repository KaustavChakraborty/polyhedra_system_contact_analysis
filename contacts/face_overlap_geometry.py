"""
Basic 3D polygon geometry for projected face-overlap calculations.

FACE OVERLAP GEOMETRY OVERVIEW
==============================
This module provides basic 3D polygon geometry validation and normal computation.

Used in the face-overlap pipeline to validate and analyze polygon data.

PURPOSE
=======
Two core functions:

1. as_3d_polygon()
   Validates and converts polygon vertex arrays
   Ensures proper format: (N, 3) with N >= 3
   Checks for NaN/inf values

2. find_unit_normal()
   Computes robust unit normal for a 3D polygon
   Uses first non-collinear triplet
   Returns unit vector perpendicular to plane

VALIDATION APPROACH
===================
Both functions use defensive validation:

Input checking:
    - Shape must be (N, 3) where N >= 3
    - All values must be finite (no NaN/inf)
    - At least 3 vertices required

Error messages:
    - Include polygon name for context
    - Describe what's wrong (shape, count, values)
    - Suggest what's expected

ALGORITHM DETAILS
=================

as_3d_polygon() Algorithm
---------------------------
Input: polygon (array-like), name (str)
Output: numpy array (N, 3)

1. Convert to numpy array
   np.asarray(polygon, dtype=float)
   Ensures numpy array with float dtype

2. Validate shape
   Check ndim == 2 and shape[1] == 3
   Raises ValueError if wrong shape

3. Validate vertex count
   Check shape[0] >= 3
   Raises ValueError if too few

4. Validate all finite
   np.all(np.isfinite(polygon))
   Rejects any NaN or inf values
   Raises ValueError if found

5. Return validated array
   Same array (not copied)
   Ready for downstream use
"""

from __future__ import annotations

import numpy as np


def as_3d_polygon(polygon, name):
    """
    Convert an input into a validated three-dimensional polygon array.

    FUNCTION PURPOSE
    ================
    Validates and standardizes polygon vertex arrays.

    Ensures proper format for downstream geometric computation:
    - Shape (N, 3) with N >= 3
    - All values finite (no NaN/inf)
    - Converted to numpy float array

    Each check raises ValueError with descriptive message.

    Parameters
    ----------
    polygon : array-like
        Polygon vertex coordinates.
        Can be: list, tuple, numpy array, etc.
        Expected shape: (N, 3) where N >= 3
        Each row: [x, y, z] coordinates
        
        Examples:
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
            np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])

    name : str
        Polygon identifier for error messages.
        Used to describe which polygon failed validation.

    Returns
    -------
    ndarray, shape (N, 3)
        Validated floating-point vertex array.
    """

    # =========================================================================
    # STEP 1: CONVERT TO NUMPY FLOAT ARRAY
    # =========================================================================
    polygon = np.asarray(polygon, dtype=float)

    # =========================================================================
    # STEP 2: VALIDATE SHAPE
    # =========================================================================
    # Must be 2D array with exactly 3 columns
    if polygon.ndim != 2 or polygon.shape[1] != 3: raise ValueError(f"{name} must have shape (N, 3), got {polygon.shape}.")
    
    # =========================================================================
    # STEP 3: VALIDATE VERTEX COUNT
    # =========================================================================
    # Must have at least 3 vertices
    if polygon.shape[0] < 3: raise ValueError(f"{name} must contain at least 3 vertices.")
    
    # =========================================================================
    # STEP 4: VALIDATE ALL VALUES FINITE
    # =========================================================================
    # Check no NaN (not-a-number) or inf (infinity) values
    if not np.all(np.isfinite(polygon)): raise ValueError(f"{name} contains NaN or infinite values.")


    # =========================================================================
    # STEP 5: RETURN VALIDATED ARRAY
    # =========================================================================
    return polygon



def find_unit_normal(polygon, name, tol=1.0e-12):
    """
    Compute the reference robust unit normal for a 3D polygon.

    The first polygon vertex is used as the origin. The remaining vertices are
    searched in their existing order until the first non-collinear triplet is
    found.

    Parameters
    ----------
    polygon
        Polygon vertices.

    name
        Label used in validation messages.

    tol
        Degeneracy threshold. A normal is accepted only when its norm is
        strictly greater than this value.

    Returns
    -------
    numpy.ndarray
        Unit normal vector with shape ``(3,)``.

    Raises
    ------
    ValueError
        If the polygon is invalid or all tested triplets are collinear.
    """

    # =========================================================================
    # STEP 1: VALIDATE POLYGON
    # =========================================================================
    polygon = as_3d_polygon(polygon, name)

    # =========================================================================
    # STEP 2: SET TRIPLET ORIGIN
    # =========================================================================
    # Use first vertex as common origin for all triplets
    p0 = polygon[0]

    # =========================================================================
    # STEP 3: SEARCH FOR FIRST VALID TRIPLET
    # =========================================================================
    # Double loop over remaining vertices
    # a and b are indices of second and third vertices
    for a in range(1, len(polygon) - 1):
        for b in range(a + 1, len(polygon)):
            # Compute vectors from origin to two vertices
            v1 = polygon[a] - p0
            v2 = polygon[b] - p0
            # Compute cross product (perpendicular vector)
            normal = np.cross(v1, v2)
            # Compute magnitude (length of cross product)
            norm = np.linalg.norm(normal)
            # Check if magnitude exceeds threshold
            # Valid triplet: norm > tolerance
            # Invalid (collinear): norm <= tolerance
            if norm > tol: return normal / norm

    # =========================================================================
    # STEP 4: NO VALID TRIPLET FOUND
    # =========================================================================
    raise ValueError(f"{name}: could not compute normal; vertices are collinear.")


# Compatibility aliases matching the original helper names.
_as_3d_polygon = as_3d_polygon
_find_unit_normal = find_unit_normal
