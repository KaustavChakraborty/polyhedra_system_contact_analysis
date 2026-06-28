"""
Face-centre distance matrices and closest-pair selection.

FACE PAIR GEOMETRY OVERVIEW
============================
This module finds the closest face pair between two particles.

Uses face-center distances to find which faces are nearest to each other.

ALGORITHM
=========
1. Compute distance between EVERY face pair
   Central faces * Neighbor faces = distance matrix

2. Find face pair with MINIMUM distance
   Flatten matrix, find argmin, convert back to indices

3. Return closest pair and distance

TIE-BREAKING
============
If multiple face pairs have same distance:

Example:
    d[0, 0] = 1.5
    d[2, 1] = 1.5  (same!)
    d[1, 2] = 2.0

argmin returns first occurrence (C-order):
    Flattens to: [1.5, ..., ..., 2.0, 1.5, ...]
    C-order: d[0,0]=index0, d[0,1]=index1, ..., d[1,0]=index(Nn), ..., d[2,1]=index(2*Nn+1)
    argmin = 0 (first 1.5)
    divmod(0, Nn) = (0, 0)
    Returns d[0, 0]
"""

from __future__ import annotations

import numpy as np


def face_center_distance_matrix(central_centres, neighbour_centres):
    """Compute all central-face to neighbour-face centre distances.

    Parameters
    ----------
    central_centres
        Central-particle face centres with shape ``(Nc, 3)``.
    neighbour_centres
        Neighbour-particle face centres with shape ``(Nn, 3)``.

    Returns
    -------
    numpy.ndarray
        Distance matrix with shape ``(Nc, Nn)``.
    """

    
    diff = central_centres[:, None, :] - neighbour_centres[None, :, :]
    return np.linalg.norm(diff, axis=2)


def select_closest_face_center_pair(distance_matrix, face_count, neighbor_id):
    """
    Select the closest face pair using the reference flattened argmin.

    FUNCTION PURPOSE
    ================
    Finds face pair with minimum distance.

    Given distance matrix, returns:
    - Indices of closest face pair
    - Distance value
    - Complete distance row (for later use)

    Parameters
    ----------
    distance_matrix : ndarray, shape (Nc, Nn)
        Face-center distance matrix.
        Each element is distance between one face pair.

    face_count : int
        Number of faces (Nc dimension).
        Used as divisor in divmod() for index conversion.
        
        Note: face_count should equal len(faces)
        This is the number of CENTRAL faces (matrix rows)

    neighbor_id : int
        Neighbor identifier (for error messages).
        Used only when raising ValueError.

    Returns
    -------
    central_face : int
        Row index of closest central face.
        Range: 0 to face_count-1

    neighbor_face : int
        Column index of closest neighbor face.
        Range: 0 to Nn-1

    minimum_distance : float
        Distance between closest pair.
        Scalar value (non-negative).

    selected_row : ndarray, shape (Nn,)
        Distance row for selected central face.
        All distances from this central face to all neighbor faces.
        Used for diagnostics or later computation.
    """

    # =========================================================================
    # STEP 1: VALIDATE INPUT
    # =========================================================================
    if not np.all(np.isfinite(distance_matrix)):
        raise ValueError(f"Non-finite values detected in face-distance matrix for neighbor {neighbor_id}.")

    # =========================================================================
    # STEP 2: FIND MINIMUM DISTANCE
    # =========================================================================
    # Flatten matrix and find index of minimum value
    flat_index = int(np.argmin(distance_matrix))

    # =========================================================================
    # STEP 3: CONVERT FLAT INDEX TO 2D INDICES
    # =========================================================================
    # Use divmod to recover (row, column) from linear index
    # divmod(flat_index, face_count) returns:
    # central_face = flat_index // face_count (quotient)
    # neighbor_face = flat_index % face_count (remainder)
    central_face, neighbor_face = divmod(flat_index, face_count)

    # =========================================================================
    # STEP 4: GET DISTANCE VALUE
    # =========================================================================
    # Retrieve the actual distance from the matrix
    # distance_matrix[central_face, neighbor_face]
    minimum_distance = float(distance_matrix[central_face, neighbor_face])

    # =========================================================================
    # STEP 5: GET DISTANCE ROW
    # =========================================================================
    # Return the complete distance row for selected central face
    # This is the distance from selected central face to ALL neighbor faces
    selected_row = distance_matrix[central_face]

    # =========================================================================
    # STEP 6: RETURN RESULTS
    # =========================================================================
    # Return all computed values
    return int(central_face), int(neighbor_face), minimum_distance, selected_row
