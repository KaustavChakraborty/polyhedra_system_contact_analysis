"""
Outer closest-face-pair loop for the face-centre method.

CLOSEST FACE PAIRS MODULE OVERVIEW
===================================
This module finds the closest face pair between a central particle and each
of its neighbors. It is the entry point for face-based contact detection.

MODULE PURPOSE
==============
Given a central particle and its neighbors, with global vertex coordinates
and polygonal face definitions, this module:

1. Validates geometry (vertices, faces, edges)
2. Computes face centers for the central particle
3. For each neighbor: finds closest face pair using face-center method
4. Returns face pair records and topology

FACE-CENTER METHOD
==================
The face-center method finds the closest face pair by:

1. Compute center of each face (average of vertices)
2. For each (central_face, neighbor_face) pair:
   - Compute distance between face centers
   - Track pair with minimum distance
3. Return closest pair for this neighbor

CLOSEST FACE PAIR RECORD FORMAT
================================
For each neighbor, face pair record contains:

    {
        'central_face': int,        # Face index in central particle
        'neighbor_face': int,       # Face index in neighbor particle
        'distance': float,          # Distance between face centers
        'normal_dot': float,        # Dot product of face normals
        (additional fields from compute_neighbor_face_pair)
    }

VALIDATION WORKFLOW
===================
Input validation order (matches reference):

1. Validate central_id (must be integer)
2. Validate neighbors (must be iterable)
3. Check empty neighbor list (return early if true)
4. Validate verts_global (not None, has central and all neighbors)
5. Validate body_vertices (shape, size)
6. Validate face/edge counts (match expected from config)
7. Validate central particle geometry (shape, size)
8. Compute central face centers

Each validation has specific error message format (reference-compatible).

OUTPUT STRUCTURE
================
Returns tuple: (results, faces)

results: dict
    Mapping neighbor_id => face_pair_record
    Example:
        {
            1: {'central_face': 0, 'neighbor_face': 2, 'distance': 1.5, ...},
            2: {'central_face': 1, 'neighbor_face': 0, 'distance': 2.1, ...},
        }


INTEGRATION POINTS
==================
This function is called from:

1. workflow/contact_stage
   Part of frame analysis pipeline
   Finds closest face pairs for all neighbors
   
2. Contact analysis main loop
   For each central particle
   Gets results dict and continues

Called after:
- Global vertex coordinates computed (via compute_global_vertices)
- Face topology loaded (from shape geometry)

Called before:
- Face overlap calculation
- Contact distance computation
"""



from __future__ import annotations

import numpy as np

from geometry import face_centers

from .face_pairing import compute_neighbor_face_pair


def closest_face_pairs_from_decomposition(i0, neighbors, verts_global, body_vertices, faces, edges, expected_num_faces, expected_num_edges, verbose=False, rank="NA"):
    """
    Calculate closest face pairs using precomputed polygonal topology.

    FUNCTION PURPOSE
    ================
    This function finds the closest face pair between a central particle
    and each of its neighbors, using a precomputed face-topology decomposition.

    INPUT REQUIREMENTS
    ==================
    All inputs must be validated and prepared before calling this function:

    i0: int
        Central particle index 

    neighbors: list of int
        Neighbor particle indices 
        Can be empty (early return with empty result)

    verts_global: dict
        Mapping particle_id => global_vertex_coordinates
        Must contain i0 and all neighbors

    body_vertices: array
        Template body-frame vertices (shape (V, 3))
        V must match actual vertex count for all particles

    faces: list of list
        Face definitions (each face = list of vertex indices)
        Same topology for all particles (same shape type)

    edges: list of tuple
        Edge definitions (each edge = tuple of 2 vertex indices)
        Derived from face topology

    expected_num_faces: int
        Face count from configuration (param_file.json)
        Used to validate detected faces

    expected_num_edges: int
        Edge count from configuration (param_file.json)
        Used to validate detected edges

    verbose: bool, default False
        Enable diagnostic output with detailed workflow steps
        Useful for debugging geometry/topology issues

    rank: str, default "NA"
        MPI rank identifier for verbose output
        Used in diagnostic messages

    PROCESSING WORKFLOW
    ===================
    1. Validate inputs (central_id, neighbors, verts_global)
    2. Validate geometry (body_vertices, faces, edges)
    3. Validate counts (faces, edges match expected)
    4. Compute central particle face centers
    5. For each neighbor:
       - Call compute_neighbor_face_pair()
       - Store result or log warning
    6. Return results dict and faces

    Returns
    -------
    results, faces
        results: dict
            Mapping neighbor_id => face_pair_record
            May be empty if no neighbors
            May be partial if some neighbors fail
            
        faces: list
            The faces parameter (returned unchanged)
            Caller provided, caller uses return value
    """


    # =====================================================================
    # HELPER FUNCTION: Conditional printing with diagnostic format
    # =====================================================================
    def safety_print(message):
        if verbose:
            print(f"[closest_face_pairs | rank {rank} | i0={i0}] {message}")

    # =====================================================================
    # STEP 1: DIAGNOSTIC OUTPUT - Starting
    # =====================================================================

    safety_print("Starting closest-face-pair detection.")
    safety_print("Method selected: face_centers")
    safety_print(f"Number of neighbors received: {len(neighbors) if neighbors is not None else 'None'}")

    # =====================================================================
    # STEP 2: VALIDATE CENTRAL PARTICLE ID
    # =====================================================================
    # Central particle ID must be integer (handles numpy types)
    if not isinstance(i0, (int, np.integer)):
        raise ValueError(f"i0 must be an integer, got {type(i0)}.")

    i0 = int(i0)

    # =====================================================================
    # STEP 3: VALIDATE AND PROCESS NEIGHBOR LIST
    # =====================================================================
    # Neighbors must be iterable; convert each to int

    if neighbors is None:
        raise ValueError("neighbors cannot be None.")

    # Convert each neighbor ID to int (handles numpy types)
    neighbors = [int(j) for j in neighbors]

    # EARLY EXIT: Empty neighbor list
    # Return immediately without geometry validation
    if len(neighbors) == 0:
        safety_print("No neighbors found. Returning empty result.")
        return {}, []

    # =====================================================================
    # STEP 4: VALIDATE GLOBAL VERTICES DICTIONARY
    # =====================================================================
    # verts_global must exist and contain all particles
    if verts_global is None:
        raise ValueError("verts_global cannot be None.")

    # Check central particle in verts_global
    if i0 not in verts_global:
        raise ValueError(f"Central particle {i0} is missing from verts_global.")

    # Check all neighbors in verts_global
    missing_neighbors = [j for j in neighbors if j not in verts_global]
    if missing_neighbors:
        raise ValueError(f"The following neighbors are missing from verts_global: {missing_neighbors}")

    # =====================================================================
    # STEP 5: VALIDATE BODY VERTICES TEMPLATE
    # =====================================================================
    # Body vertices must have correct shape and size

    body_vertices_array = np.asarray(body_vertices, dtype=float)
    safety_print(f"Body vertices shape: {body_vertices_array.shape}")

    # Check 2D array with 3 columns (N, 3)
    if body_vertices_array.ndim != 2 or body_vertices_array.shape[1] != 3:
        raise ValueError(f"self.vertices must have shape (N_vertices, 3), got {body_vertices_array.shape}.")

    # Check minimum vertices for 3D polyhedron
    if body_vertices_array.shape[0] < 4:
        raise ValueError("At least 4 vertices are required to define a 3D polyhedron.")

    # =====================================================================
    # STEP 6: VALIDATE FACE AND EDGE TOPOLOGY COUNTS
    # =====================================================================
    # Detected counts must match expected counts from configuration

    detected_faces = len(faces)
    detected_edges = len(edges)

    safety_print(f"Number of detected faces: {detected_faces}")
    safety_print(f"Expected number of faces from JSON: {expected_num_faces}")
    safety_print(f"Number of detected edges: {detected_edges}")
    safety_print(f"Expected number of edges from JSON: {expected_num_edges}")

    # Check faces detected
    if detected_faces == 0:
        raise ValueError("No faces were detected for this particle shape.")

    # Check face count match
    if detected_faces != expected_num_faces:
        raise ValueError(f"Mismatch in detected face count. Expected {expected_num_faces} faces from param JSON, but convex decomposition detected {detected_faces} faces.")

    # Check edge count match
    if detected_edges != expected_num_edges:
        raise ValueError(f"Mismatch in detected edge count. Expected {expected_num_edges} edges from param JSON, but convex decomposition detected {detected_edges} edges.")

    # Log face vertex counts
    face_sizes = [len(face) for face in faces]
    safety_print(f"Face vertex counts: {face_sizes}")

    # =====================================================================
    # STEP 7: VALIDATE CENTRAL PARTICLE GEOMETRY
    # =====================================================================
    # Central particle vertices must have correct shape and match body vertex count

    central_vertices = np.asarray(verts_global[i0], dtype=float)
    safety_print(f"Central particle global vertices shape: {central_vertices.shape}")

    # Check 2D array with 3 columns (V, 3)
    if central_vertices.ndim != 2 or central_vertices.shape[1] != 3:
        raise ValueError(f"verts_global[{i0}] must have shape (N_vertices, 3), got {central_vertices.shape}.")

    # Check vertex count match (same as body template)
    if central_vertices.shape[0] != body_vertices_array.shape[0]:
        raise ValueError(f"Central particle vertex count mismatch. Expected {body_vertices_array.shape[0]}, got {central_vertices.shape[0]}.")

    # =====================================================================
    # STEP 8: COMPUTE CENTRAL PARTICLE FACE CENTERS
    # =====================================================================
    # Face centers used to find closest face pairs
    try:
        central_centres = face_centers(central_vertices, faces)
    except Exception as exc:
        raise RuntimeError(f"Failed to compute face centers for central particle {i0}.") from exc

    safety_print(f"Central face centers shape: {central_centres.shape}")

    # =====================================================================
    # STEP 9: NEIGHBOR-WISE CLOSEST-FACE-PAIR DETECTION
    # =====================================================================
    # For each neighbor: compute closest face pair 

    results = {}

    for neighbor_id in neighbors:
        safety_print(f"Processing neighbor {neighbor_id}.")

        try:
            # Call face pairing function for this neighbor
            results[neighbor_id] = compute_neighbor_face_pair(
                neighbor_id=neighbor_id,
                neighbor_vertices=verts_global[neighbor_id],
                body_vertex_count=body_vertices_array.shape[0],
                faces=faces,
                central_vertices=central_vertices,
                central_centres=central_centres,
                safety_print=safety_print,
            )
        except Exception as exc:
            # Log warning but continue to next neighbor
            safety_print(f"WARNING: Failed to process neighbor {neighbor_id}. Reason: {exc}")
            continue

    # =====================================================================
    # STEP 10: FINAL DIAGNOSTIC SUMMARY
    # =====================================================================

    safety_print(f"Finished closest-face-pair detection. Successful neighbors: {len(results)} / {len(neighbors)}")

    # Verbose output table of results
    if verbose:
        for neighbor_id, record in results.items():
            print(
                f"Neighbour {neighbor_id}: "
                f"central_face={record['central_face']}  "
                f"nearest_face={record['neighbor_face']}  "
                f"d_min={record['distance']:.6f}   "
                f"n·n'={record['normal_dot']:.3f}"
            )

    # =====================================================================
    # STEP 11: RETURN RESULTS AND TOPOLOGY
    # =====================================================================

    return results, faces
