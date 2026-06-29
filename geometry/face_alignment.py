"""
Normal alignment between two selected polygonal faces.

FACE ALIGNMENT OVERVIEW
=======================
This module computes how well aligned two faces are.

Measures the angle between face normals using dot product.

PURPOSE
=======
Determine if two faces are:
- Parallel (same direction): dot = +1
- Perpendicular: dot = 0
- Antiparallel (opposite direction): dot = -1

MATHEMATICAL FOUNDATION
=======================
Dot product of unit vectors:

    n_A dot n_B = |n_A| |n_B| cos(\theta)

For unit vectors (|n| = 1):
    n_A dot n_B = cos(\theta)

Where \theta = angle between vectors
"""

from __future__ import annotations

import numpy as np

# Import the robust normal computation function
from .face_geometry import unit_face_normal


def selected_face_normal_alignment(central_face_points, neighbor_face_points, central_face_id=None, neighbor_face_id=None):
    """Compute the signed normal alignment of two selected faces.

    Parameters
    ----------
    central_face_points
        Vertex coordinates of the selected central-particle face.
    neighbor_face_points
        Vertex coordinates of the selected neighbour-particle face.
    central_face_id
        Identifier passed to ``unit_face_normal`` for the central face.
    neighbor_face_id
        Identifier passed to ``unit_face_normal`` for the neighbour face.

    Returns
    -------
    float
        Clipped signed dot product of the two unit face normals.
    """

    # =========================================================================
    # STEP 1: COMPUTE NORMAL FOR CENTRAL FACE
    # =========================================================================
    # Robust computation of unit normal using first non-collinear triplet
    central_normal = unit_face_normal(central_face_points, face_id=central_face_id)

    # =========================================================================
    # STEP 2: COMPUTE NORMAL FOR NEIGHBOR FACE
    # =========================================================================
    # Same process for neighbor face
    neighbor_normal = unit_face_normal(neighbor_face_points, face_id=neighbor_face_id)

    # =========================================================================
    # STEP 3: COMPUTE SIGNED DOT PRODUCT
    # =========================================================================
    # Measure alignment of the two normals
    normal_dot = float(np.dot(central_normal, neighbor_normal))

    # =========================================================================
    # STEP 4: CLIP TO VALID RANGE
    # =========================================================================
    normal_dot = float(np.clip(normal_dot, -1.0, 1.0))

    # =========================================================================
    # STEP 5: RETURN RESULT
    # =========================================================================
    return normal_dot