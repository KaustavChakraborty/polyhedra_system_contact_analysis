"""
Particle-contact and neighbor-list utilities.

CONTACTS MODULE OVERVIEW
========================
This module provides comprehensive contact detection and analysis utilities
for particles in periodic simulations. It includes neighbor-list construction,
overlap detection, and face-based contact analysis.

MODULE PURPOSE
==============
The contacts module handles:

1. Neighbor List Management
   - Build neighbor lists from distance cutoffs
   - Convert between list formats
   - Compute coordination numbers

2. Overlap Detection (Diagnostic)
   - Check if particle polyhedra overlap
   - Multiple fallback methods (FCL, Boolean, SAT)
   - Face normal SAT as final fallback

3. Face-Based Contact Analysis
   - Find closest face pairs between particles
   - Compute face overlap areas
   - Classify face types
   - Calculate contact metrics

MODULE ORGANIZATION
===================
The module is organized into logical submodules:

NEIGHBOR LIST CONSTRUCTION
──────────────────────────
neighbor_list.py
    - build_neighbor_list_from_cutoffs()
    - coordination_number_from_rdf()
    - nearest_neighbor_list_calc_func()

Builds neighbor lists based on distance cutoffs.
Used for identifying potential contacts.

NEIGHBOR PAIR MANAGEMENT
─────────────────────────
neighbor_pairs.py
    - build_neighbor_map()
    - neighbor_list_to_pairs()

Converts between neighbor list formats.
Creates neighbor_map: {particle_id → [neighbor_ids]}

POLYHEDRON OVERLAP DETECTION (DIAGNOSTIC)
──────────────────────────────────────────
overlap.py, overlap_sat.py, particle_overlap.py
    - check_overlap(): single pair overlap check
    - check_overlap_sat(): SAT fallback
    - check_particle_overlaps(): all neighbors

Checks if polyhedra interpenetrate or touch.
Uses multiple fallback methods for robustness.

FACE PAIR DETECTION
───────────────────
closest_face_pairs.py
    - closest_face_pairs_from_decomposition()

Finds closest face pair between particles.
Uses face-center method for efficiency.

FACE OVERLAP ANALYSIS (COMPLEX PIPELINE)
─────────────────────────────────────────
Multiple submodules for computing face overlap:

1. Geometry (face_overlap_geometry.py)
   - as_3d_polygon(): ensure 3D polygon format
   - find_unit_normal(): compute face normal

2. Basis (face_overlap_basis.py)
   - plane_basis_from_normal(): 2D basis in plane

3. Plane Data (face_overlap_plane.py)
   - plane_data(): plane equation from vertices

4. Rotation (face_overlap_rotation.py)
   - rotation_matrix_from_vectors(): rotation matrix

5. Parallelization (face_overlap_parallelization.py)
   - minimum_rotation_to_parallelize_normals(): align normals

6. Projection (face_overlap_projection.py)
   - project_points_to_plane(): project to plane

7. 2D Coordinates (face_overlap_coordinates.py)
   - project_to_2d(): convert to 2D in plane

8. Deduplication (face_overlap_deduplication.py)
   - deduplicate_points_2d(): remove duplicate points

9. Ordering (face_overlap_ordering.py)
   - order_points_ccw(): counter-clockwise order

10. Area (face_overlap_area.py)
    - polygon_area_2d(): compute 2D polygon area

11. Polygon Validation (face_overlap_polygon.py)
    - make_valid_polygon(): fix degenerate polygons

12. Results (face_overlap_results.py)
    - zero_overlap_result(): return 0 area
    - successful_overlap_result(): return computed area
    - failed_neighbour_overlap_result(): error result

13. Intersection (face_overlap_intersection.py)
    - polygon_intersection_area(): intersection computation

14. Single Pair (face_overlap.py)
    - max_overlap_face_area(): overlap between two faces

15. All Neighbors (all_neighbour_face_overlap.py)
    - max_overlap_all_neighbours(): compute for all neighbors

16. Printing (face_overlap_printing.py)
    - print_max_overlap_areas(): formatted output

COMPLETE WORKFLOW
=================
Typical contact analysis workflow:

1. Build Neighbor List
   - Input: particle positions, cutoff distance
   - Output: neighbor_list (for each particle: nearby particles)

2. Convert to Neighbor Map
   - Input: neighbor_list
   - Output: neighbor_map {particle_id → [neighbors]}

3. For Each Central Particle:
   a. Check Particle Overlaps (diagnostic)
      - Input: global vertices, faces
      - Detects interpenetrating geometry
      - Prints warnings if found
   
   b. Find Closest Face Pairs
      - Input: global vertices, neighbor list
      - Output: which faces touch each neighbor
   
   c. Compute Face Overlap Areas
      - Input: closest face pairs
      - Complex geometric pipeline:
        * Extract face geometry
        * Compute intersection polygon
        * Calculate area
      - Output: overlap area for each pair
   
   d. Compute Contact Metrics
      - Input: face overlap areas + geometry
      - Compute distance metrics
      - Calculate contact indices
      - Output: comprehensive contact records

FUNCTION CATEGORIES
===================

IMPORTED FUNCTIONS
──────────────────
(from neighbor_list module)
    build_neighbor_list_from_cutoffs
        Build neighbor list from distance cutoff
    coordination_number_from_rdf
        Extract coordination from RDF
    nearest_neighbor_list_calc_func
        Compute nearest neighbor list

(from neighbor_pairs module)
    build_neighbor_map
        Convert to neighbor_map format
    neighbor_list_to_pairs
        Extract neighbor pairs

(from overlap module)
    check_overlap
        Single pair overlap check (with fallbacks)

(from overlap_reporting module)
    collect_overlapping_neighbors
        Find neighbors with overlaps
    report_particle_overlap_diagnostic
        Print overlap diagnostics

(from overlap_sat module)
    check_overlap_sat
        SAT-based overlap check
    _check_overlap_sat
        Private version (compatibility)

(from particle_overlap module)
    check_particle_overlaps
        Check overlaps for all neighbors

(from face_pairing module)
    compute_neighbor_face_pair
        Compute closest face pair

(from closest_face_pairs module)
    closest_face_pairs_from_decomposition
        Find closest pairs for all neighbors

FACE OVERLAP PIPELINE FUNCTIONS
────────────────────────────────
Geometry functions:
    as_3d_polygon, _as_3d_polygon
    find_unit_normal, _find_unit_normal

Basis functions:
    plane_basis_from_normal, _plane_basis_from_normal

Plane functions:
    plane_data, _plane_data

Rotation functions:
    rotation_matrix_from_vectors, _rotation_matrix_from_vectors

Parallelization functions:
    minimum_rotation_to_parallelize_normals
    _minimum_rotation_to_parallelize_normals

Projection functions:
    project_points_to_plane, _project_points_to_plane

2D coordinate functions:
    project_to_2d, _project_to_2d

Deduplication functions:
    deduplicate_points_2d, _deduplicate_points_2d

Ordering functions:
    order_points_ccw, _order_points_ccw

Area functions:
    polygon_area_2d, _polygon_area_2d

Polygon validation functions:
    make_valid_polygon, _make_valid_polygon

Result functions:
    zero_overlap_result, _zero_result
    successful_overlap_result
    failed_neighbour_overlap_result

High-level functions:
    polygon_intersection_area
    max_overlap_face_area
    max_overlap_all_neighbours
    print_max_overlap_areas

PUBLIC API
==========
Exported functions (in __all__):
    [60+ functions covering all aspects]

Main user-facing functions:
    build_neighbor_list_from_cutoffs()
    check_particle_overlaps()
    closest_face_pairs_from_decomposition()
    max_overlap_all_neighbours()
    print_max_overlap_areas()

INTEGRATION WITH OTHER MODULES
===============================
Used by:
1. workflow.contact_stage
   - Calls check_particle_overlaps() for diagnostics
   - Calls closest_face_pairs_from_decomposition()
   - Calls max_overlap_all_neighbours()

2. particles module
   - Uses neighbor lists for particle selection

3. geometry module
   - Provides face/edge topology

4. metrics module
   - Uses face geometry from overlap calculations

INPUT DATA TYPES
================
Typical inputs:

positions: (N, 3) array
    Particle center-of-mass positions

neighbors: list or dict
    Neighbor relationships {id → [neighbor_ids]}

verts_global: dict {particle_id → (V, 3) array}
    Global vertex coordinates

faces: list of lists
    Face connectivity (vertex indices)

edges: list of tuples
    Edge connectivity (vertex index pairs)

OUTPUT DATA TYPES
=================
Typical outputs:

overlap_result: dict or bool
    Overlap detection result

face_pair_info: dict {neighbor_id → record}
    Closest face pair information

overlap_area: float
    Area of intersection polygon

contact_record: dict
    Comprehensive contact information

NUMERICAL ROBUSTNESS
====================
Key robustness features:

1. Fallback methods for overlap detection
   - FCL (Fast Collision Library)
   - Boolean intersection (OpenSCAD)
   - SAT (Separating Axis Theorem)

2. Polygon handling
   - Validation and cleanup
   - Deduplication of points
   - Handling of collinear vertices

3. Numerical tolerances
   - Small area thresholding
   - Projection accuracy
   - Normal vector normalization

ERROR HANDLING
==============
Strategy:

1. Graceful degradation
   - Fallback methods used automatically
   - No errors propagate to user

2. Diagnostic reporting
   - Overlap issues printed with diagnostics
   - Failures tracked and reported

3. Exception handling
   - FCL failures caught and handled
   - Boolean failures trigger SAT fallback
   - SAT as final fallback (always works)

PERFORMANCE NOTES
=================
Complexity analysis:

Neighbor list: O(N log N) for N particles
Overlap detection: O(V^3) for V vertices (SAT fallback)
Face overlap: O(F * V^2) for F faces, V vertices
Total for one frame: O(N * M * V^2) for N particles, M neighbors

Optimization opportunities:
- Cache normals between frames
- Use spatial hashing for neighbor lists
- Parallelize over particles
- GPU acceleration for overlap detection

USAGE EXAMPLES
==============
Example 1: Build neighbor list

    from contacts import build_neighbor_list_from_cutoffs
    
    neighbor_list = build_neighbor_list_from_cutoffs(
        positions=positions,
        r_max=5.0,
        box=box,
    )

Example 2: Check particle overlaps (diagnostic)

    from contacts import check_particle_overlaps
    
    overlaps = check_particle_overlaps(
        i0=0,
        neighbors=[1, 2, 3],
        verts_local=verts_global,
        faces=faces,
        edges=edges,
    )

Example 3: Find closest face pairs

    from contacts import closest_face_pairs_from_decomposition
    
    face_pairs, faces = closest_face_pairs_from_decomposition(
        i0=0,
        neighbors=[1, 2],
        verts_global=verts_global,
        body_vertices=body_vertices,
        faces=faces,
        edges=edges,
        expected_num_faces=6,
        expected_num_edges=12,
    )

Example 4: Compute face overlaps

    from contacts import max_overlap_all_neighbours
    
    overlaps = max_overlap_all_neighbours(
        central_id=0,
        face_pair_info=face_pairs,
        verts_global=verts_global,
        faces=faces,
    )

DEBUGGING UTILITIES
===================
Available for debugging:

check_particle_overlaps()
    Diagnostic overlap checking
    Helps identify bad input geometry

print_max_overlap_areas()
    Formatted output of overlap results
    Useful for visual inspection

Verbose flags
    Most functions support verbose=True
    Enables detailed diagnostic output

KNOWN LIMITATIONS
=================
1. SAT only checks face-normal axes
   - Edge-cross-edge axes not included
   - May miss some edge-to-edge contacts
   - But works for typical cases

2. Overlap detection is diagnostic only
   - Not used for contact force calculation
   - Just confirms particles don't interpenetrate

3. Face overlap computation is expensive
   - O(V^2) per face pair
   - May be bottleneck for large faces

FUTURE IMPROVEMENTS
===================
Potential enhancements:

1. GPU-accelerated overlap detection
2. Spatial hashing for neighbor finding
3. Incremental overlap computation
4. Contact persistence over frames
5. Edge-cross-edge SAT axes
6. Voronoi-based neighbor lists
"""


# ============================================================================
# IMPORTS FROM SUBMODULES
# ============================================================================

# Neighbor list construction
from .neighbor_list import (
    build_neighbor_list_from_cutoffs,
    coordination_number_from_rdf,
    nearest_neighbor_list_calc_func,
)

# Neighbor pair management
from .neighbor_pairs import (
    build_neighbor_map,
    neighbor_list_to_pairs,
)

# Polyhedron overlap detection
from .overlap import (
    check_overlap,
)

# Overlap diagnostics and reporting
from .overlap_reporting import (
    collect_overlapping_neighbors,
    report_particle_overlap_diagnostic,
)

# SAT-based overlap fallback
from .overlap_sat import (
    _check_overlap_sat,
    check_overlap_sat,
)

# Particle overlap checking
from .particle_overlap import (
    check_particle_overlaps,
)

# Face pair detection
from .face_pairing import (
    compute_neighbor_face_pair,
)

# Closest face pairs for all neighbors
from .closest_face_pairs import (
    closest_face_pairs_from_decomposition,
)

# Face overlap pipeline: Geometry
from .face_overlap_geometry import (
    _as_3d_polygon,
    _find_unit_normal,
    as_3d_polygon,
    find_unit_normal,
)

# Face overlap pipeline: Plane basis
from .face_overlap_basis import (
    _plane_basis_from_normal,
    plane_basis_from_normal,
)

# Face overlap pipeline: Plane data
from .face_overlap_plane import (
    _plane_data,
    plane_data,
)

# Face overlap pipeline: Rotation matrices
from .face_overlap_rotation import (
    _rotation_matrix_from_vectors,
    rotation_matrix_from_vectors,
)

# Face overlap pipeline: Parallel alignment
from .face_overlap_parallelization import (
    _minimum_rotation_to_parallelize_normals,
    minimum_rotation_to_parallelize_normals,
)

# Face overlap pipeline: Projection to plane
from .face_overlap_projection import (
    _project_points_to_plane,
    project_points_to_plane,
)

# Face overlap pipeline: 2D coordinate conversion
from .face_overlap_coordinates import (
    _project_to_2d,
    project_to_2d,
)

# Face overlap pipeline: Point deduplication
from .face_overlap_deduplication import (
    _deduplicate_points_2d,
    deduplicate_points_2d,
)

# Face overlap pipeline: Counter-clockwise ordering
from .face_overlap_ordering import (
    _order_points_ccw,
    order_points_ccw,
)

# Face overlap pipeline: 2D area calculation
from .face_overlap_area import (
    _polygon_area_2d,
    polygon_area_2d,
)

# Face overlap pipeline: Polygon validation
from .face_overlap_polygon import (
    _make_valid_polygon,
    make_valid_polygon,
)

# Face overlap pipeline: Result structures
from .face_overlap_results import (
    _zero_result,
    failed_neighbour_overlap_result,
    successful_overlap_result,
    zero_overlap_result,
)

# Face overlap pipeline: Polygon intersection
from .face_overlap_intersection import (
    polygon_intersection_area,
)

# Face overlap for single pair
from .face_overlap import (
    max_overlap_face_area,
)

# Face overlap for all neighbors
from .all_neighbour_face_overlap import (
    max_overlap_all_neighbours,
)

# Formatted output utilities
from .face_overlap_printing import (
    print_max_overlap_areas,
)


# ============================================================================
# PUBLIC API
# ============================================================================

__all__ = [
    # Neighbor list functions
    "coordination_number_from_rdf",
    "build_neighbor_list_from_cutoffs",
    "nearest_neighbor_list_calc_func",

    # Neighbor pair functions
    "neighbor_list_to_pairs",
    "build_neighbor_map",

    # Overlap checking functions
    "check_overlap_sat",
    "_check_overlap_sat",
    "check_overlap",
    "check_particle_overlaps",

    # Overlap reporting functions
    "collect_overlapping_neighbors",
    "report_particle_overlap_diagnostic",

    # Face pairing functions
    "compute_neighbor_face_pair",
    "closest_face_pairs_from_decomposition",

    # Face overlap geometry pipeline
    "as_3d_polygon",
    "find_unit_normal",
    "_as_3d_polygon",
    "_find_unit_normal",
    "plane_basis_from_normal",
    "_plane_basis_from_normal",
    "plane_data",
    "_plane_data",
    "rotation_matrix_from_vectors",
    "_rotation_matrix_from_vectors",
    "minimum_rotation_to_parallelize_normals",
    "_minimum_rotation_to_parallelize_normals",
    "project_points_to_plane",
    "_project_points_to_plane",
    "project_to_2d",
    "_project_to_2d",
    "deduplicate_points_2d",
    "_deduplicate_points_2d",
    "order_points_ccw",
    "_order_points_ccw",
    "polygon_area_2d",
    "_polygon_area_2d",
    "make_valid_polygon",
    "_make_valid_polygon",

    # Face overlap result functions
    "zero_overlap_result",
    "_zero_result",
    "successful_overlap_result",
    "polygon_intersection_area",
    "max_overlap_face_area",
    "failed_neighbour_overlap_result",

    # High-level overlap functions
    "max_overlap_all_neighbours",
    "print_max_overlap_areas",
]