"""
Periodic particle displacement and global vertex transformations.

TRANSFORMS MODULE OVERVIEW
===========================
This module handles coordinate system transformations for particles in
periodic boundary condition (PBC) simulations.

Core operations:
1. Compute wrapped displacements (with PBC)
2. Transform vertices from body frame to global coordinates (with rotation)

COORDINATE TRANSFORMATION WORKFLOW
===================================

Transformation from body frame to global frame:

    For central particle i:
        global_v = R(q_i) * body_v + pos_i
    
    For neighbor particle j (with PBC):
        local_pos_j = box.wrap(pos_j - pos_i)
        image_pos_j = pos_i + local_pos_j  (neighbor in nearest image)
        global_v = R(q_j) * body_v + image_pos_j

Where:
- R(q) is rotation by quaternion q
- body_v is vertex in body frame
- pos_i is central particle position
- q_j is neighbor orientation
"""

from __future__ import annotations

import numpy as np
import rowan


def compute_wrapped_neighbor_displacements(box, positions, central_id, neighbors):
    """
    Compute wrapped displacement vectors from a central particle to neighbours.

    OVERVIEW
    ========
    This function computes the displacement from a central particle to each
    of its neighbors, with periodic boundary condition wrapping. The result
    is the nearest periodic image displacement for each neighbor.

    VECTORIZATION
    ==============
    The function uses NumPy broadcasting to compute all displacements at once.
    This is much faster than looping.

    Parameters
    ----------
    box : freud.box.Box
        Simulation box with periodic boundary conditions.
        Must have a wrap() method that takes (M, 3) array and returns
        wrapped displacements in range (-L/2, L/2) for each dimension.

    positions : array-like, shape (N, 3)
        Global positions of all particles.

    central_id : int
        Index of the central particle.

    neighbors : iterable of int
        Indices of neighbor particles.

    Returns
    -------
    numpy.ndarray
        Wrapped displacement vectors with shape (M, 3) where M = len(neighbors).
        
        Result[i] = displacement from central particle to neighbor i
                  = wrapped(positions[neighbors[i]] - positions[central_id])
        
        Each displacement is in range (-L/2, L/2) for each box dimension.
    """

    # STEP 1: Compute raw displacements from central to neighbors
    displacement = (
        positions[neighbors]
        - positions[central_id]
    )

    # STEP 2: Wrap displacements using periodic boundary conditions
    local_positions = box.wrap(
        displacement
    )

    # STEP 3: Return wrapped displacements
    return local_positions


def compute_global_vertices(body_vertices, central_id, positions, orientations, neighbors, local_positions):
    """
    Transform central and neighbouring particle vertices to global coordinates.

    OVERVIEW
    ========
    This function transforms all particle vertices from body-frame coordinates to global coordinates. It accounts for:
    1. Particle position (translation)
    2. Particle orientation (rotation via quaternion)
    3. Periodic boundary conditions (via pre-wrapped displacements)

    TRANSFORMATION STEPS
    ====================
    For central particle i:
        1. Rotate body vertices by quaternion q_i
        2. Translate by position pos_i
        Result: global_verts_i = R(q_i) * body_verts + pos_i

    For each neighbor j (with PBC):
        1. Rotate body vertices by quaternion q_j
        2. Translate by effective position (pos_i + local_pos_j)
           where local_pos_j is the wrapped displacement
        Result: global_verts_j = R(q_j) * body_verts + (pos_i + local_pos_j)

    Why this works:
    - local_positions are pre-wrapped (via compute_wrapped_neighbor_displacements)
    - Adding to central position places neighbor in nearest periodic image
    - Rotation applied in body frame (before translation)
    - Result is all vertices in common global coordinate system

    DATA FLOW
    =========
    Inputs:
    - body_vertices: (V, 3) reference shape
    - central_id: which particle is central
    - positions: (N, 3) global positions
    - orientations: (N, 4) quaternions [q_w, q_x, q_y, q_z]
    - neighbors: (M,) indices of neighbor particles
    - local_positions: (M, 3) wrapped displacements

    Processing:
    - For central particle: rotate then translate
    - For each neighbor: rotate then translate by wrapped position

    Output:
    - Dictionary mapping particle_id => global_vertices
    - global_vertices: (V, 3) array of transformed vertices

    OUTPUT STRUCTURE
    ================
    Returns dictionary (not 2D array) for flexibility:

        {
            central_id: [[v0_x, v0_y, v0_z], [v1_x, v1_y, v1_z], ...],
            neighbor_1: [[v0_x, v0_y, v0_z], [v1_x, v1_y, v1_z], ...],
            neighbor_2: [[v0_x, v0_y, v0_z], [v1_x, v1_y, v1_z], ...],
            ...
        }

    Format preserved for downstream compatibility:
    - Keys are particle IDs (int)
    - Values are lists of lists (V, 3)
    - Each vertex represented as [x, y, z]

    Parameters
    ----------
    body_vertices : array-like, shape (V, 3)
        Common body-frame template vertices.
        Same for all particles of same type.
        Typically centered at origin (or body-frame center).

    central_id : int
        Index of the central particle.
        Position: positions[central_id]
        Orientation: orientations[central_id]

    positions : array-like, shape (N, 3)
        Global positions of all particles (N particles).
        Must be 2D array with 3 columns (x, y, z).

    orientations : array-like, shape (N, 4)
        Particle orientation quaternions in Rowan/HOOMD format.
        Format: [q_w, q_x, q_y, q_z].

    neighbors : iterable of int
        Indices of neighbor particles.
        Each must be in range [0, N).

    local_positions : array-like, shape (M, 3)
        Wrapped displacement vectors for each neighbor.
        Usually from compute_wrapped_neighbor_displacements().

    Returns
    -------
    dict
        Mapping particle_id => global_vertices
        
        global_vertices format:
        - List of lists (M, 3) => list(ndarray.tolist())
        - Each vertex: [x, y, z] (3 scalars)
        - Example:
            {
                0: [[1.0, 2.0, 3.0], [1.1, 2.1, 3.1], ...],
                1: [[0.8, 1.9, 2.9], [0.9, 2.0, 3.0], ...],
                ...
            }
    """

    # STEP 1: Convert body vertices to numpy array for efficient computation
    body_verts = np.array(body_vertices)
    # STEP 2: Get central particle position
    pos0 = positions[central_id]
    # STEP 3: Initialize output dictionary
    verts_global = {}

    # STEP 4: Transform central particle vertices
    try:
        # Rotate body vertices by central particle's quaternion
        rotated = rowan.rotate(orientations[central_id], body_verts)
        # Translate to global position and convert to list structure
        verts_global[central_id] = (rotated + pos0).tolist()
    except Exception as exc:
        # Wrap any rotation/translation error with context
        raise RuntimeError(f"Failed to rotate/translate central particle i0={central_id}.") from exc

    # STEP 5: Transform neighbor particle vertices
    # Each neighbor has potentially different orientation and position
    for index, neighbor_id in enumerate(neighbors):
        try:
            # Get neighbor's quaternion
            quaternion = orientations[neighbor_id]
            # Get wrapped displacement to this neighbor
            displacement = np.array(local_positions[index])
            # Compute neighbor's global position 
            # pos_central + wrapped_displacement
            global_neighbor_position = pos0 + displacement
            # Rotate body vertices by neighbor's quaternion
            rotated_neighbor = rowan.rotate(quaternion, body_verts)
            # Translate to neighbor's global position and convert to list
            verts_global[neighbor_id] = (rotated_neighbor + global_neighbor_position).tolist()
        except Exception as exc:
            # Wrap any rotation/translation error with full context
            raise RuntimeError(f"Failed to rotate/translate neighbor particle j={neighbor_id} around central particle i0={central_id}.") from exc

    # STEP 6: Return complete global vertex mapping
    return verts_global