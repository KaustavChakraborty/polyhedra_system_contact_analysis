"""
Reference-style polygonal face categorization.

Overview
--------
This module assembles verified geometry components into a complete pipeline
for categorizing the faces of a polyhedron. It takes a set of vertices and
produces a sorted list of faces with their geometric properties.

What This Module Does
---------------------
The module performs high-level shape categorization by assembling and
coordinating several lower-level geometry operations:

    1. Convex Decomposition
       Compute the convex hull of the vertex set

    2. Topology Matching
       Test multiple tolerance values to find a decomposition matching
       the expected number of edges and faces

    3. Face-Record Construction
       Build structured records for each face

    4. Deterministic Sorting
       Sort faces in a reproducible order for consistent results

Tolerance Iteration Strategy
----------------------------
The module uses a robust tolerance-iteration approach to handle numerical
precision issues:

    - Floating-point rounding errors can cause convex hull algorithms
      to produce slightly different topologies
    - Different tolerance values (epsilon) produce different results
    - The module tries multiple tolerances until one produces the
      expected topology
    - This makes the algorithm robust to small variations in input      

Example:
    For a cube with 12 edges and 6 faces:
        - Try tolerance 1e-10: produces 12 edges, 6 faces ✓ MATCH!
        - (Other tolerances might produce 11 or 13 edges if we tested them)

Return Value Flexibility
------------------------
This module provides two interfaces:

    1. categorize_polyhedron_faces()
       Returns a rich FaceCategorizationResult with full details:
           - face_vertices: Sorted vertex lists
           - face_areas: Precomputed areas
           - decomposition: Full decomposition object
           - tolerance: The tolerance that worked

    2. poly_face_categorization_func()
       Returns a simple tuple (face_vertices, face_areas) for
       compatibility with existing code

Error Handling
--------------
The module validates inputs and raises descriptive exceptions:

    ValueError
        - Invalid vertex array shape or values
        - Invalid edge/face count parameters
        - No tolerance produces expected topology

    RuntimeError
        - Convex decomposition fails for all tolerances

These exceptions propagate to the caller, which is responsible for
handling them appropriately.

Caching Opportunities

    # First time: compute
    geometry = categorize_polyhedron_faces(vertices, 12, 6)
    
    # Later: reuse
    geometry = load_from_cache(shape_signature(vertices))

The tolerance field in the result can be used as a key for caching.
"""

from __future__ import annotations

from collections import namedtuple

import numpy as np

from .convex_decomposition import convexDecomposition
from .face_records import get_sorted_face_vertices_and_areas
from .topology import decomposition_tolerances


# Define the return type as a named tuple
# Contains all information about the categorized shape
FaceCategorizationResult = namedtuple(
    "FaceCategorizationResult",
    [
        "face_vertices",
        "face_areas",
        "decomposition",
        "tolerance",
    ],
)


def categorize_polyhedron_faces(vertices, num_edges, num_faces, verbose=True):
    """
    Identify and sort the polygonal faces of a convex polyhedron.

    This is the main entry point for shape categorization. It performs
    convex decomposition, topology matching, and face sorting to produce
    a complete geometric description of the polyhedron.

    Parameters
    ----------
    vertices
        Polyhedron vertex coordinates. Accepted formats:
            - Numpy array of shape (N, 3) with dtype float
            - List of lists: [[x0, y0, z0], [x1, y1, z1], ...]
            - Other array-like structures compatible with np.asarray()

    num_edges : int
        Used for topology matching to verify the decomposition is correct.

    num_faces : int
        Expected number of polygonal faces in the polyhedron.
        Also used for topology matching.

    verbose : bool, default True
        If True, print a categorization summary when a matching decomposition
        is found. The output shows:
            - Tolerance value used
            - Number of faces detected
            - Number of edges detected
            - Expected counts for verification
        
        Set to False to suppress this output 

    Returns
    -------
    FaceCategorizationResult
        Named tuple containing the complete categorization:

        face_vertices : list of list
            Sorted vertex lists for each face. Format:
            [[v0, v1, v2, ...], [v3, v4, v5, ...], ...]
            Each inner list contains indices into the vertices array.
            The faces are sorted in a deterministic, reproducible order.

        face_areas : list of float
            Surface area of each face, in the same order as face_vertices.
            Computed using standard 3D polygon area formula.

        decomposition : ConvexDecomposition
            The complete convex decomposition object containing:
                - vertices: The input vertices
                - edges: All unique edges
                - faces: All faces with connectivity info
            Can be inspected for debugging or further processing.

        tolerance : float
            The numerical tolerance value (epsilon) that produced the
            expected topology. Useful for:
                - Understanding numerical precision requirements
                - Reproducibility and caching
                - Debugging topology issues

    RuntimeError
        Raised if convex decomposition fails internally:
        "Convex decomposition failed for all tested tolerance values."
        Cause: The convex hull algorithm failed for all tested tolerances

    Example
    -------
    Categorize a cube:

        import numpy as np
        from geometry import categorize_polyhedron_faces

        # Cube vertices
        vertices = [
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # bottom
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]   # top
        ]

        result = categorize_polyhedron_faces(
            vertices=vertices,
            num_edges=12,
            num_faces=6,
            verbose=True,
        )

        print(f"Found {len(result.face_vertices)} faces")
        print(f"Face areas: {result.face_areas}")
        print(f"Tolerance used: {result.tolerance}")

    """

    # Step 1: Convert and validate the vertices input
    # Try to convert to numpy array; raise ValueError if conversion fails
    try:
        vertices = np.asarray(
            vertices,
            dtype=float,
        )
    except Exception as exc:
        raise ValueError(
            "Invalid vertices input. Expected a list or array "
            "of 3D coordinates."
        ) from exc

    # Verify vertices have the correct shape: (N, 3)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(
            f"Invalid vertices shape: expected (N, 3), "
            f"got {vertices.shape}."
        )

    # Verify we have at least 4 vertices (minimum for a tetrahedron)
    if vertices.shape[0] < 4:
        raise ValueError(
            "At least 4 vertices are required to construct "
            "a 3D convex polyhedron."
        )

    # Verify all vertex values are finite (no NaN or inf)
    if not np.all(np.isfinite(vertices)):
        raise ValueError(
            "Vertices contain NaN or infinite values."
        )

    # Preserve the reference strict integer validation.
    if not isinstance(num_edges, int) or num_edges <= 0:
        raise ValueError(
            f"num_edges must be a positive integer, got {num_edges}."
        )

    # Step 2: Validate the expected topology parameters
    # Preserve strict integer validation (no floats or booleans)
    if not isinstance(num_faces, int) or num_faces <= 0:
        raise ValueError(
            f"num_faces must be a positive integer, got {num_faces}."
        )

    # Step 3: Tolerance iteration loop
    # Try different tolerance values until one produces the expected topology
    convex_decomposition = None
    matched_tolerance = None
    last_error = None

    # Iterate through candidate tolerance values
    # These are generated by decomposition_tolerances() and typically span
    # from very tight (1e-10) to loose (1e-6)
    for tolerance in decomposition_tolerances():
        try:
            # Attempt convex decomposition with this tolerance
            # The tolerance affects how "flat" faces are allowed to be
            candidate_decomposition = convexDecomposition(vertices, tolerance)

            # Extract the topology from the decomposition
            candidate_edges = candidate_decomposition.edges
            candidate_faces = candidate_decomposition.faces

        except Exception as exc:
            # If decomposition fails, save the error and try next tolerance
            last_error = exc
            continue

        # Check if this decomposition matches the expected topology
        # Both edges AND faces must match exactly
        if (
            len(candidate_edges) == num_edges
            and len(candidate_faces) == num_faces
        ):
            # Found a match! Save the decomposition and tolerance
            convex_decomposition = candidate_decomposition
            matched_tolerance = tolerance

            # Print categorization summary if verbose mode is on
            if verbose:
                print("\n[Face Categorization]")
                print(
                    f"  Tolerance used      : "
                    f"{tolerance:.1e}"
                )
                print(
                    f"  Faces detected      : "
                    f"{len(candidate_faces)}"
                )
                print(
                    f"  Edges detected      : "
                    f"{len(candidate_edges)}"
                )
                print(
                    f"  Expected faces      : "
                    f"{num_faces}"
                )
                print(
                    f"  Expected edges      : "
                    f"{num_edges}"
                )
            # Exit loop; we found a match
            break

    # Step 4: Handle case where no tolerance produced expected topology
    if convex_decomposition is None:
        # Choose appropriate error message based on what went wrong
        if last_error is not None:
            raise RuntimeError(
                "Convex decomposition failed for all tested "
                "tolerance values."
            ) from last_error

        # Some decompositions succeeded but none matched topology
        raise ValueError(
            "Could not find a convex decomposition matching "
            "the expected topology. "
            f"Expected num_edges={num_edges}, "
            f"num_faces={num_faces}."
        )

    # Step 5: Extract faces from the matched decomposition
    faces = convex_decomposition.faces

    # Sanity check: ensure we got faces (should never fail if decomposition succeeded)
    if len(faces) == 0:
        raise ValueError(
            "Convex decomposition returned zero faces."
        )

    # Step 6: Construct and sort face records
    face_vertices_sorted, face_areas_sorted = (
        get_sorted_face_vertices_and_areas(
            vertices,
            faces,
        )
    )

    # Step 7: Build and return the result
    # Package all information into a named tuple for structured access
    return FaceCategorizationResult(
        face_vertices=face_vertices_sorted,
        face_areas=face_areas_sorted,
        decomposition=convex_decomposition,
        tolerance=matched_tolerance,
    )


def poly_face_categorization_func(
    vertices,
    num_edges,
    num_faces,
    verbose=True,
):
    """
    Compatibility wrapper preserving the reference two-value return.

    Returns
    -------
    face_vertices_sorted, face_areas_sorted
    """
    result = categorize_polyhedron_faces(
        vertices=vertices,
        num_edges=num_edges,
        num_faces=num_faces,
        verbose=verbose,
    )

    return (
        result.face_vertices,
        result.face_areas,
    )
