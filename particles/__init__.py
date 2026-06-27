"""
Particle-related utilities for the modular contact-analysis project.

PARTICLES MODULE OVERVIEW
=========================
This module provides utility functions for handling particle geometry,
transformations, and vertex calculations. It bridges particle frame
definitions with global coordinate systems.

MODULE PURPOSE
==============
The particles module handles:

1. Shape Scaling
   Scale particle vertices based on volume
   Used when particles have different sizes

2. Coordinate Transformations
   Convert between body-frame and global coordinates
   Apply periodic boundary conditions (PBC)

3. Vertex Management
   Track vertex coordinates across particles
   Handle rotations and translations

MODULE COMPONENTS
=================
The module consists of 3 core submodules:

1. scaling.py
   Purpose: Volume-based vertex scaling
   Main function: calculate_the_vertices()
   
   Scales reference particle vertices to new target volume.
   Used for: non-uniformly sized particle populations
   
   Algorithm: cubic scaling in 3D
       scale_factor = (V_target / V_reference)^(1/3)
       scaled_vertices = reference_vertices * scale_factor

2. transforms.py
   Purpose: Coordinate transformations with PBC
   Main functions:
       - compute_wrapped_neighbor_displacements()
       - compute_global_vertices()
   
   Handles:
   - Periodic boundary condition wrapping
   - Quaternion-based rotations
   - Translation to global coordinates

3. closest_face_pairs.py (in contacts module)
   Purpose: Find closest face pairs between particles
   Main function: closest_face_pairs_from_decomposition()
   
   Outputs: Face pair records and topology

DATA FLOW THROUGH MODULE
========================
Typical workflow:

1. Shape Preparation Stage
   - Reference vertices loaded (body-frame)
   - Volume specified in config
   
2. Contact Analysis Stage
   Per frame:
       a. Load particle positions and orientations
       
       b. Scale vertices if needed (calculate_the_vertices)
          reference_vertices => scaled_vertices
       
       c. Compute wrapped displacements (compute_wrapped_neighbor_displacements)
          (neighbor_pos - central_pos) with PBC
          => local displacements
       
       d. Transform to global coordinates (compute_global_vertices)
          scaled_vertices + rotation + translation
          => global_vertices for central and all neighbors
       
       e. Find closest face pairs (closest_face_pairs_from_decomposition)
          global_vertices + faces + edges
          => face pair records

3. Contact Metrics Stage
   - Global vertices used for distance calculations
   - Face pairs used for overlap calculations

COORDINATE SYSTEMS
===================
coordinate systems are used:

1. Body Frame
   - Reference particle definition
   - Local to particle
   - Typically centered at origin
   - Particle stored in shape JSON files

2. Global Frame
   - World coordinates with periodic boundary conditions
   - Used for distance calculations
   - Handles wrapping across periodic boundaries

PERIODIC BOUNDARY CONDITIONS (PBC)
==================================

Algorithm:
    1. Compute raw displacement: disp = pos_neighbor - pos_central
    2. Wrap using simulation box: disp_wrapped = box.wrap(disp)
    3. Result is displacement to nearest image
    4. Neighbor in nearest image placed at: pos_central + disp_wrapped

This ensures contact calculations use nearest particle images.

VOLUME SCALING
==============
When particles have different volumes, vertices are scaled:

    scale_factor = (V_target / V_reference)^(1/3)
    scaled_vertex = body_vertex * scale_factor

This preserves shape while changing size.

Common use cases:
- Polydisperse systems (different sized particles)
- Non-unit particle sizes
- Adjusting particle volume to match simulation parameters

INTEGRATION WITH OTHER MODULES
===============================
This module is used by:

1. workflow.contact_stage
   Creates ContactDistanceMetrics with global_vertices
   
2. contacts.closest_face_pairs
   Finds closest face pairs using global_vertices
   
3. Main analysis workflow
   Orchestrates vertex transformations across frames

VALIDATION STRATEGY
===================
Inputs validated at function boundaries:
- Array shapes checked
- Element counts verified
- Required keys in dicts checked
- Types converted/validated

Errors caught early to prevent silent corruption of results.

PUBLIC API
==========
Exported functions (in __all__):

1. calculate_the_vertices(shape_vertices_basic, particle_volume, upcoming_particle_volume)
   Scale vertices from one volume to another
   Returns: scaled vertices (same structure as input)

2. compute_wrapped_neighbor_displacements(box, positions, central_id, neighbors)
   Compute displacements with PBC wrapping
   Returns: (M, 3) array of wrapped displacements

3. compute_global_vertices(body_vertices, central_id, positions, orientations, neighbors, local_positions)
   Transform body-frame vertices to global coordinates
   Returns: dict {particle_id: global_vertices}

USAGE EXAMPLES
==============
Example 1: Scale vertices for different volume

    from particles import calculate_the_vertices
    
    body_verts = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
    ref_volume = 1.0
    target_volume = 8.0  # 2x larger
    
    scaled = calculate_the_vertices(body_verts, ref_volume, target_volume)
    # Each coordinate scaled by (8/1)^(1/3) = 2

Example 2: Transform vertices to global coordinates

    from particles import compute_global_vertices, compute_wrapped_neighbor_displacements
    
    # Get wrapped displacements
    disp = compute_wrapped_neighbor_displacements(
        box=box,
        positions=positions,
        central_id=0,
        neighbors=[1, 2, 3]
    )
    
    # Transform to global coordinates
    verts = compute_global_vertices(
        body_vertices=body_vertices,
        central_id=0,
        positions=positions,
        orientations=orientations,
        neighbors=[1, 2, 3],
        local_positions=disp
    )
    
    # Access results
    central_vertices = verts[0]  # Central particle
    neighbor_1_vertices = verts[1]  # First neighbor
"""

from .scaling import (
    calculate_the_vertices,
)
from .transforms import (
    compute_global_vertices,
    compute_wrapped_neighbor_displacements,
)


__all__ = [
    "calculate_the_vertices",
    "compute_wrapped_neighbor_displacements",
    "compute_global_vertices",
]