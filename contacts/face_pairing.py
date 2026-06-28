"""
Closest-face-pair calculation for one neighbour particle.

FACE PAIRING OVERVIEW
=====================
This module computes the closest face pair between central and neighbor particles.

Orchestrates all geometry helpers to find which faces are nearest.

WORKFLOW
========
High-level algorithm:

1. Validate neighbor vertex count
2. Compute neighbor face centers
3. Compute distance matrix (all face pairs)
4. Find face pair with minimum distance
5. Compute normal alignment for selected faces
6. Assemble result record

Return comprehensive record with all information.

ARCHITECTURE
============
This module COMPOSES lower-level geometry helpers:

    face_centers()                    => neighbor face centers
    face_center_distance_matrix()     => distance matrix
    select_closest_face_center_pair() => closest pair indices
    selected_face_normal_alignment()  => normal dot product
    build_face_pair_record()          => result dict

Each helper is independent and tested.
This module just orchestrates them.

DIAGNOSTIC OUTPUT
=================
Optional safety_print callback for diagnostics.

If safety_print is None:
    No output (production mode)

If safety_print is provided:
    Called with diagnostic messages at each step:
    1. Vertex shape validation
    2. Neighbor face centers shape
    3. Distance matrix shape
    4. Selected face pair and distance
    5. Normal alignment value

Useful for debugging individual neighbor computations.

USAGE IN WORKFLOW
=================
Called from:
    closest_face_pairs_from_decomposition()
    
Usage pattern:

    central_centres = face_centers(central_vertices, faces)
    
    for neighbor_id in neighbors:
        try:
            record = compute_neighbor_face_pair(
                neighbor_id,
                neighbor_vertices[neighbor_id],
                body_vertex_count,
                faces,
                central_vertices,
                central_centres,
                safety_print=verbose_print,
            )
            face_pair_info[neighbor_id] = record
        except Exception as exc:
            if verbose:
                print(f"Warning: {exc}")
            # Continue with next neighbor

RESULT RECORD FORMAT
====================
Returned dictionary:

{
    "central_face": int,           # Central face index
    "neighbor_face": int,          # Neighbor face index
    "distance": float,             # Face center distance
    "row_to_neighbour": list,      # All distances from central face
    "normal_dot": float,           # Normal alignment (-1 to +1)
}

ALGORITHM WALKTHROUGH
=====================
Step-by-step execution for one neighbor:

Input:
    neighbor_id = 5
    neighbor_vertices = [[x1,y1,z1], [x2,y2,z2], ...]
    body_vertex_count = 8
    faces = [[0,1,2], [1,2,3], ...]
    central_vertices = [[x1,y1,z1], ...]
    central_centres = [[cx1,cy1,cz1], ...]
    safety_print = print (or None)

STEP 1: Convert neighbor vertices to numpy array
    neighbor_vertices_array = np.asarray(neighbor_vertices, dtype=float)
    Shape should be (8, 3)

STEP 2: Print diagnostic (if safety_print provided)
    "Neighbor 5 global vertices shape: (8, 3)"

STEP 3: Validate vertex array shape
    Check: ndim == 2 and shape[1] == 3
    Raise ValueError if not

STEP 4: Validate vertex count
    Check: shape[0] == body_vertex_count
    Raise ValueError if not (e.g., 7 instead of 8)

STEP 5: Compute neighbor face centers
    neighbour_centres = face_centers(neighbor_vertices_array, faces)
    Result shape: (6, 3) for 6 faces

STEP 6: Print diagnostic
    "Neighbor 5: neighbor face centers shape = (6, 3)"

STEP 7: Compute distance matrix
    distances = face_center_distance_matrix(central_centres, neighbour_centres)
    Result shape: (6, 6) for 6 central, 6 neighbor faces

STEP 8: Print diagnostic
    "Neighbor 5: face-distance matrix shape = (6, 6)"

STEP 9: Select closest face pair
    c_face, n_face, min_dist, row = select_closest_face_center_pair(...)
    Checks for non-finite values
    Raises ValueError if NaN/inf detected

STEP 10: Print diagnostic
    "Neighbor 5: closest pair = central_face 0, neighbor_face 2, distance = 0.15"

STEP 11: Compute normal alignment
    normal_dot = selected_face_normal_alignment(
        central_vertices[faces[0]],  # Central face 0 vertices
        neighbor_vertices_array[faces[2]],  # Neighbor face 2 vertices
        central_face_id=0,
        neighbor_face_id=2,
    )
    Result: scalar float in [-1, 1]

STEP 12: Print diagnostic
    "Neighbor 5: normal_dot = 0.125000"

STEP 13: Build result record
    record = build_face_pair_record(
        central_face=0,
        neighbor_face=2,
        minimum_distance=0.15,
        selected_row=row,
        normal_dot=0.125,
    )
    Result: dict with 5 keys

STEP 14: Return result
    return record

Caller receives:
{
    "central_face": 0,
    "neighbor_face": 2,
    "distance": 0.15,
    "row_to_neighbour": [0.15, ..., 0.82, ...],
    "normal_dot": 0.125,
}
"""

from __future__ import annotations

import numpy as np

from geometry import (
    build_face_pair_record,
    face_center_distance_matrix,
    face_centers,
    select_closest_face_center_pair,
    selected_face_normal_alignment,
)


def compute_neighbor_face_pair(neighbor_id, neighbor_vertices, body_vertex_count, faces, central_vertices, central_centres, safety_print=None):
    """
    Compute the closest face-pair record for one neighbour.

    FUNCTION PURPOSE
    ================
    Orchestrates complete face-pair computation for one neighbor.

    Given neighbor vertex data and precomputed central face centers,
    computes which faces are closest and returns comprehensive record.

    WORKFLOW
    ========
    1. Convert and validate neighbor vertices
    2. Compute neighbor face centers
    3. Compute distance matrix (all face pairs)
    4. Find face pair with minimum distance
    5. Compute normal alignment for selected faces
    6. Assemble and return result record

    PARAMETERS EXPLAINED
    ====================
    neighbor_id
        Unique identifier for this neighbor particle

    neighbor_vertices
        Global vertex coordinates of neighbor particle

    body_vertex_count
        Expected number of vertices 

    faces
        Face connectivity 

    central_vertices
        Global vertex coordinates of central particle

    central_centres
        PRECOMPUTED face centers of central particle

    safety_print
        Optional diagnostic output callback

    Returns
    -------
    dict
        Comprehensive face-pair result record
        
        Keys:
            "central_face": int
                Index of closest face in central particle
            "neighbor_face": int
                Index of closest face in neighbor particle
            "distance": float
                Euclidean distance between face centers
            "row_to_neighbour": list
                Distances from selected central face to all neighbor faces
            "normal_dot": float
                Dot product of face normals (range: -1 to 1)
    """

    # =========================================================================
    # STEP 1: CONVERT NEIGHBOR VERTICES TO NUMPY ARRAY
    # =========================================================================
    neighbor_vertices_array = np.asarray(neighbor_vertices, dtype=float)

    # =========================================================================
    # STEP 2: DIAGNOSTIC OUTPUT - VERTEX SHAPE
    # =========================================================================
    # Print diagnostic message if safety_print provided
    if safety_print is not None:
        safety_print(f"Neighbor {neighbor_id} global vertices shape: {neighbor_vertices_array.shape}")

    # =========================================================================
    # STEP 3: VALIDATE VERTEX ARRAY SHAPE
    # =========================================================================
    # Check shape is exactly (N, 3)
    if neighbor_vertices_array.ndim != 2 or neighbor_vertices_array.shape[1] != 3:
        raise ValueError(f"verts_global[{neighbor_id}] must have shape (N_vertices, 3), got {neighbor_vertices_array.shape}.")

    # =========================================================================
    # STEP 4: VALIDATE VERTEX COUNT
    # =========================================================================
    # Check that actual vertex count matches expected count
    if neighbor_vertices_array.shape[0] != body_vertex_count:
        raise ValueError(f"Neighbor {neighbor_id} vertex count mismatch. Expected {body_vertex_count}, got {neighbor_vertices_array.shape[0]}.")

    # =========================================================================
    # STEP 5: COMPUTE NEIGHBOR FACE CENTERS
    # =========================================================================
    # Compute centroid of each face
    neighbour_centres = face_centers(neighbor_vertices_array, faces)

    # =========================================================================
    # STEP 6: DIAGNOSTIC OUTPUT - FACE CENTERS
    # =========================================================================
    if safety_print is not None:
        safety_print(f"Neighbor {neighbor_id}: neighbor face centers shape = {neighbour_centres.shape}")

    # =========================================================================
    # STEP 7: COMPUTE FACE-CENTER DISTANCE MATRIX
    # =========================================================================
    # Compute all pairwise distances: central faces * neighbor faces
    # Returns (num_central_faces, num_neighbor_faces) matrix
    # distances[i, j] = distance from central face i to neighbor face j
    distances = face_center_distance_matrix(central_centres, neighbour_centres)

    # =========================================================================
    # STEP 8: DIAGNOSTIC OUTPUT - DISTANCE MATRIX
    # =========================================================================
    if safety_print is not None:
        safety_print(f"Neighbor {neighbor_id}: face-distance matrix shape = {distances.shape}")

    # =========================================================================
    # STEP 9: SELECT CLOSEST FACE PAIR
    # =========================================================================
    # Find face pair with minimum distance
    # Returns indices and distance value
    central_face, neighbor_face, minimum_distance, selected_row = select_closest_face_center_pair(distances, face_count=len(faces), neighbor_id=neighbor_id)

    # =========================================================================
    # STEP 10: DIAGNOSTIC OUTPUT - SELECTED PAIR
    # =========================================================================
    if safety_print is not None:
        safety_print(f"Neighbor {neighbor_id}: closest pair = central_face {central_face}, neighbor_face {neighbor_face}, distance = {minimum_distance:.6e}")

    # =========================================================================
    # STEP 11: COMPUTE FACE NORMAL ALIGNMENT
    # =========================================================================
    # Extract vertices for the two selected faces
    # central_vertices[faces[central_face]] = vertices of central face
    # neighbor_vertices_array[faces[neighbor_face]] = vertices of neighbor face
    #
    # Compute unit normals and dot product
    normal_dot = selected_face_normal_alignment(central_vertices[faces[central_face]], neighbor_vertices_array[faces[neighbor_face]], central_face_id=central_face, neighbor_face_id=neighbor_face)

    # =========================================================================
    # STEP 12: DIAGNOSTIC OUTPUT - NORMAL ALIGNMENT
    # =========================================================================
    if safety_print is not None:
        safety_print(f"Neighbor {neighbor_id}: normal_dot = {normal_dot:.6f}")

    # =========================================================================
    # STEP 13: BUILD AND RETURN RESULT RECORD
    # =========================================================================
    # Assemble all computed values into result dictionary
    # Dictionary format:
    # {
    #     "central_face": int,
    #     "neighbor_face": int,
    #     "distance": float,
    #     "row_to_neighbour": list,
    #     "normal_dot": float,
    # }
    return build_face_pair_record(central_face=central_face, neighbor_face=neighbor_face, minimum_distance=minimum_distance, selected_row=selected_row, normal_dot=normal_dot)