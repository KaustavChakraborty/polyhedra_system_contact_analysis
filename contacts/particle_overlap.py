"""
Central-particle overlap diagnostics.

PARTICLE OVERLAP DIAGNOSTICS OVERVIEW
=====================================
This module provides diagnostic checking for polyhedron interpenetration.

It applies the complete polyhedron-overlap check between one central
particle and each particle in its ordered neighbour list.

PURPOSE
=======
Detects if particle polyhedra interpenetrate or just touch.

This is a DIAGNOSTIC CHECK only:
   - Not used for contact force calculations
   - Not used for filtering neighbors
   - Helps identify bad input geometry
   - Printed as diagnostic warnings if found

USE CASE
========
After particles are placed in simulation:
1. Central particle is specified
2. Its neighbors are listed
3. For each neighbor: check if polyhedra overlap
4. Report any overlaps (indicates geometry issue)

COMPATIBILITY BEHAVIOR
======================
Important compatibility notes:

1. Neighbour order is PRESERVED
   - Results returned in same order as input
   - If neighbors = [1, 3, 2], results in same order

2. Duplicate neighbour IDs are PRESERVED
   - If neighbor appears twice, checked twice
   - Both results included

3. Neighbour IDs are RETURNED UNCHANGED
   - Same type as input (int or numpy type)
   - No conversion or filtering

4. Overlap results are RETURNED UNCHANGED
   - Whatever overlap checker returns
   - No modification or interpretation

5. No filtering, sorting, or deduplication
   - All neighbors processed as-is
   - No special handling

6. Exceptions from overlap checker PROPAGATE
   - Errors not caught
   - Caller handles exceptions

7. Calculation is DIAGNOSTIC ONLY
   - Results not used for contact decisions
   - Just for reporting potential issues

WORKFLOW
========
For each neighbor:
1. Get particle A (central) vertices
2. Get particle B (neighbor) vertices
3. Call check_overlap(A, B) to check for interpenetration
4. Append (neighbor_id, overlap_result) to results list

Output: list of (neighbor_id, overlap_result) tuples

ALGORITHM
=========
    central_verts = verts_local[i0]
    
    for each neighbor in neighbors:
        neighbor_verts = verts_local[neighbor]
        overlap = check_overlap(
            central_verts,
            neighbor_verts,
            faces,
            edges
        )
        append (neighbor, overlap) to results

PERFORMANCE
===========
Time complexity per function call:
    O(M * V^3) where:
    M = number of neighbors
    V = number of vertices per particle
    
    Each check_overlap call is O(V^3) in worst case
    (SAT algorithm: V^2 axes, V projections each)

Typical case (M=10, V=8):
    10 * 512 = 5120 operations
    Time: ~1 millisecond per particle

Not a bottleneck in workflow.

NUMERICAL CONSIDERATIONS
========================
Overlap detection uses multiple fallback methods:

1. FCL (Fast Collision Library) - if available
   Fastest method, uses spatial partitioning
   
2. Boolean intersection - uses OpenSCAD
   Computes actual intersection volume
   Slower but more robust
   
3. SAT (Separating Axis Theorem) - final fallback
   Checks if separating axis exists
   Robust handwritten implementation
   Always works (never fails)

Therefore: check_overlap NEVER fails
Result is always bool or raises exception

RETURN FORMAT
=============
Returns list of tuples: [(neighbor_id, overlap_result), ...]

Each tuple contains:
    neighbor_id: int
        The neighbor particle ID
        Same as input (preserved order and type)
    
    overlap_result: bool or dict
        Result from check_overlap()
        True = overlap/touch detected
        False = no overlap
        May be dict if diagnostic info included

USAGE EXAMPLES
==============
Example 1: Basic diagnostic check

    from contacts import check_particle_overlaps
    
    overlaps = check_particle_overlaps(
        i0=0,  # Central particle
        neighbors=[1, 2, 3],  # Check these neighbors
        verts_local=verts_global,  # Global vertices
        faces=faces,  # Face topology
        edges=edges,  # Edge topology
    )
    
    # overlaps = [(1, False), (2, False), (3, True)]
    # neighbor 3 overlaps with central particle 0

Example 2: Handling results

    overlaps = check_particle_overlaps(...)
    
    for neighbor_id, overlap in overlaps:
        if overlap:
            print(f"WARNING: Particles {central_id} and {neighbor_id} overlap!")

Example 3: With duplicates

    overlaps = check_particle_overlaps(
        i0=0,
        neighbors=[1, 2, 1],  # Neighbor 1 appears twice
        verts_local=verts_global,
        faces=faces,
        edges=edges,
    )
    
    # Both occurrences of neighbor 1 checked
    # overlaps = [(1, False), (2, False), (1, False)]

INTEGRATION
===========
Called from:
    analyze_particle_contacts() in workflow/contact_stage.py
    
Used to:
    Detect bad input geometry
    Print diagnostic warnings
    Help debug position/orientation issues

When called:
    After particles placed in simulation
    Before contact analysis
    If params["check_particle_overlaps"] = True

LIMITATIONS
===========
1. Diagnostic only
   - Not used for contact filtering
   - Not used for physics calculations

2. Expensive computation
   - O(V^3) for each neighbor
   - May be slow for large faces

3. Boolean method limitations
   - Requires valid meshes
   - Some degenerate cases fail
   - Falls back to SAT

4. SAT limitations
   - Only checks face normals
   - Edge-cross-edge axes not checked
   - May miss some edge-touch cases

DEBUGGING TIPS
==============
1. Enable verbose output
   Reports which neighbors overlap

2. Inspect results
   Look for unexpected overlaps

3. Check vertex coordinates
   Verify particles positioned correctly

4. Check mesh validity
   Use trimesh.Trimesh.is_valid property

KNOWN ISSUES
============
1. Large faces slow SAT
   If faces have many vertices, SAT slow
   Solution: reduce vertex count

2. Degenerate meshes
   Some mesh configurations fail FCL
   Falls back to SAT (works)

3. Numerical precision
   Very close (nearly touching) may be detected as overlapping
   Design feature (errs on side of caution)

"""

from __future__ import annotations

import numpy as np

from .overlap import check_overlap


def check_particle_overlaps(i0, neighbors, verts_local, faces, edges):
    """
    Check overlap between a central particle and all listed neighbours.

    FUNCTION PURPOSE
    ================
    Diagnostic check to detect if particle polyhedra interpenetrate.

    For a central particle, checks if its polyhedron overlaps with each
    neighbor's polyhedron. Returns results for all neighbors.

    DIAGNOSTIC ONLY
    ===============
    This function:
    - Detects interpenetrating geometry (bad input)
    - Helps identify misplaced or misoriented particles
    - Reports issues for debugging
    - Does NOT affect contact calculations
    - Is NOT used for filtering neighbors
    - Is NOT used for physics

    WORKFLOW
    ========
    1. Get central particle vertices: verts_local[i0]
    2. For EACH neighbor:
       a. Get neighbor vertices: verts_local[neighbor]
       b. Call check_overlap() to detect interpenetration
       c. Append (neighbor_id, overlap_result) to results
    3. Return list of results

    COMPATIBILITY
    ==============
    Behavior preserved from reference:
    - Neighbor order preserved (results in same order as input)
    - Duplicate IDs preserved (if neighbor appears twice, checked twice)
    - IDs returned unchanged (same type as input)
    - Results returned unchanged (no filtering/modification)
    - Exceptions propagate (not caught)

    Parameters
    ----------
    i0 : int
        Central-particle index (0-based particle ID).
        
        Example: 0 for first particle, 1 for second, etc.

    neighbors : iterable of int
        Ordered iterable of neighbour-particle indices.
        Can be list, array, tuple, or any iterable.
        
        Order preserved in output (results in same order).
        Duplicates preserved (if neighbor appears twice).
        
        Example: [1, 2, 3] or [1, 3, 2] or [1, 1, 2]

    verts_local : dict
        Dictionary mapping particle IDs to their global vertex coordinates.
        
        Format: {particle_id: (V, 3) array}
        
        Where:
            particle_id: int matching i0 or neighbors
            (V, 3) array: V vertices, 3 coordinates each
        
        Name "verts_local" retained from reference.
        Actually contains GLOBAL coordinates (with PBC applied).
        
        Example:
            {
                0: [[x0, y0, z0], [x1, y1, z1], ...],
                1: [[x0, y0, z0], [x1, y1, z1], ...],
                2: [[x0, y0, z0], [x1, y1, z1], ...],
            }

    faces : list of lists
        Polyhedron face connectivity.
        
        Each face is a list of vertex indices.
        Same topology for all particles (shared shape).
        
        Example:
            [[0, 1, 2], [1, 2, 3], [0, 1, 3], ...]
        
        Each inner list specifies which vertices form a face.
        Vertices must be CCW when viewed from outside.

    edges : list of tuples
        Polyhedron edge connectivity.
        
        Each edge is a tuple of two vertex indices.
        Same topology for all particles.
        
        Example:
            [(0, 1), (1, 2), (2, 0), (0, 3), ...]
        
        Note: edges parameter accepted for compatibility
              but NOT USED in this function
              (check_overlap uses faces only)

    Returns
    -------
    list of tuples
        Ordered list of (neighbor_id, overlap_result) tuples.
        
        Format: [(neighbor_1, result_1), (neighbor_2, result_2), ...]
        
        Where:
            neighbor_id: int
                Same as input (order and type preserved)
            
            overlap_result: bool or dict
                Overlap detection result from check_overlap()
                
                True: polyhedra overlap or touch
                False: polyhedra separate (no contact)
                
                May be dict if diagnostic info included

        Length: same as len(neighbors)
        Order: same as input neighbors order
        
        Example return:
            [(1, False), (2, False), (3, True)]
        means:
            - Neighbor 1: no overlap
            - Neighbor 2: no overlap
            - Neighbor 3: overlaps with central

    Notes
    -----
    - This is a DIAGNOSTIC function only
    - Results not used for contact physics
    - Neighbor order ALWAYS preserved
    - Duplicates ALWAYS included (if in input)
    - Exceptions from check_overlap PROPAGATE
    - No error handling or filtering

    Examples
    --------
    Basic usage:

        >>> from contacts import check_particle_overlaps
        >>> 
        >>> # Particle 0 with 3 neighbors
        >>> overlaps = check_particle_overlaps(
        ...     i0=0,
        ...     neighbors=[1, 2, 3],
        ...     verts_local=verts_dict,
        ...     faces=faces_list,
        ...     edges=edges_list,
        ... )
        >>> print(overlaps)
        [(1, False), (2, False), (3, False)]

    Check for overlaps:

        >>> overlaps = check_particle_overlaps(...)
        >>> 
        >>> for neighbor_id, overlap in overlaps:
        ...     if overlap:
        ...         print(f"WARNING: {central} overlaps with {neighbor_id}")

    Handle duplicates (neighbors can appear multiple times):

        >>> overlaps = check_particle_overlaps(
        ...     i0=0,
        ...     neighbors=[1, 2, 1],  # Neighbor 1 twice
        ...     verts_local=verts_dict,
        ...     faces=faces_list,
        ...     edges=edges_list,
        ... )
        >>> print(overlaps)
        [(1, False), (2, False), (1, False)]
        >>> # Neighbor 1 appears twice in results

    See Also
    --------
    check_overlap : check overlap between one pair
    report_particle_overlap_diagnostic : print diagnostic report
    """

    # =========================================================================
    # STEP 1: GET CENTRAL PARTICLE VERTICES
    # =========================================================================
    # Convert to numpy array for efficient computation
    # verts_local[i0] returns (V, 3) array of vertex coordinates
    verts_A = np.array(verts_local[i0])

    # =========================================================================
    # STEP 2: INITIALIZE RESULTS LIST
    # =========================================================================
    # Will accumulate (neighbor_id, overlap_result) tuples
    # Preserves order and duplicates (one entry per neighbor)
    overlaps = []

    # =========================================================================
    # STEP 3: LOOP OVER NEIGHBORS
    # =========================================================================
    # Check overlap with each neighbor
    # Order preserved (results in same order as input neighbors)
    # Duplicates preserved (if same neighbor appears twice, checked twice)
    for j in neighbors:
        # Get neighbor vertices
        # Convert to numpy array for efficient computation
        verts_B = np.array(verts_local[j])

        # Call overlap checker for this neighbor pair
        # This function handles all fallback logic:
        # 1. Try FCL (Fast Collision Library)
        # 2. Try Boolean intersection
        # 3. Fall back to SAT (always works)
        overlap = check_overlap(verts_A, verts_B, faces, edges)

        # Append (neighbor_id, overlap_result) to results
        # Preserves neighbor order and ID type
        overlaps.append((j, overlap))

    # =========================================================================
    # STEP 4: RETURN RESULTS
    # =========================================================================
    # Return list of results in same order as input neighbors
    # Each result: (neighbor_id, overlap_bool)
    return overlaps
