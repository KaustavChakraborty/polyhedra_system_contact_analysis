"""
Tolerance-bin deduplication of two-dimensional polygon points.

DEDUPLICATION OVERVIEW
======================
This module removes duplicate vertices from 2D point sets.

Eliminates nearly duplicate points within tolerance using bin-based approach.

PURPOSE
=======
In face-overlap pipeline:

After projecting 3D faces to 2D:
- Points may contain duplicates (numerical precision)
- Nearly-collinear vertices
- Duplicates cause problems in polygon operations

This function removes duplicates using tolerance bins.

DUPLICATE DEFINITION
====================
Two points are duplicates if:

They map to the same rounded integer bin:
    key = tuple(np.round(point / tol).astype(int))

Distance metric is implicit:
    Two points in same bin if within tol distance
    Not explicit Euclidean distance check
    Bin-based approach (more efficient)

Tolerance (default 1e-10):
    Grid spacing for bins
    Points closer than tol map to same bin
    Handles numerical precision

ALGORITHM
=========
Four steps:

1. Convert to numpy array

2. Validate input shape
   Must be exactly (N, 2)

3. Bin-based deduplication
   Initialize empty set (seen bins)
   Initialize empty list (output points)
   For each point:
     - Compute bin key: round(point / tol) as integers
     - If key not seen: add to output and mark seen
     - If key seen: skip (duplicate)

4. Return deduplicated points
   As numpy array (N_unique, 2)
   Preserves order
   Removes duplicates
"""

from __future__ import annotations

import numpy as np


def deduplicate_points_2d(points, tol=1.0e-10):
    """Remove nearly duplicate 2D points while preserving first occurrence.

    FUNCTION PURPOSE
    ================
    Eliminates near-duplicate vertices from 2D point set.

    Parameters
    ----------
    points
        Two-dimensional point coordinates.
    tol
        Width used to construct rounded integer bins.

    Returns
    -------
    numpy.ndarray
        First point retained from every unique rounded bin.

    Raises
    ------
    ValueError
        If input shape is not (N, 2).
    """

    # Convert input to numpy float array
    points = np.asarray(points, dtype=float)

    # Validate input shape: must be exactly (N, 2)
    if points.ndim != 2 or points.shape[1] != 2: raise ValueError(f"2D polygon must have shape (N, 2), got {points.shape}.")

    # Initialize tracking structures
    # seen: set of bin keys already encountered
    # output: list of unique points (in order of first occurrence)
    seen, output = set(), []

    # Iterate through all input points
    for point in points:
        # Compute bin key for this point
        # Divide by tolerance to scale, round to nearest integer, convert to tuple (hashable)
        # Two points in same bin are considered duplicates
        key = tuple(np.round(point / tol).astype(int))
        # Check if this bin has been seen before
        if key not in seen:
            # First occurrence of this bin: add to output and mark as seen
            seen.add(key)
            output.append(point)

    # Convert accumulated output list to numpy array (N_unique, 2)
    return np.asarray(output, dtype=float)


# Compatibility alias matching the original private helper.
_deduplicate_points_2d = deduplicate_points_2d