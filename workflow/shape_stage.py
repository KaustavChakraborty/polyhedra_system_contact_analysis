"""
Shape-preparation stage for the contact-analysis workflow.

Overview
--------
This stage loads and categorizes the reference particle shape geometry,
preparing it for use in the contact-analysis workflow.

What This Stage Does
--------------------
This stage performs the following operations:

    1. Loads the reference particle vertices from the shape JSON file
    2. Performs convex decomposition to identify faces and edges
    3. Matches the decomposition topology to expected face/edge counts
    4. Constructs and sorts the polygonal face records
    5. Computes face areas and geometric properties
    6. Returns the prepared geometry needed by later analysis stages

Topology Matching
-----------------
The shape categorization logic iterates through multiple tolerance values
to find a convex decomposition that matches the expected topology:
    - Expected number of edges (num_edges)
    - Expected number of faces (num_faces)

Return Value Structure
----------------------
The ShapePreparationResult contains:
    - vertices : Loaded 3D vertex coordinates
    - face_vertices : Sorted vertex lists for each face
    - face_areas : Precomputed area for each face
    - decomposition : The matched convex decomposition object
    - tolerance : The numerical tolerance that produced the match
"""

from __future__ import annotations

from collections import namedtuple

from dataio import load_shape_vertices
from geometry import categorize_polyhedron_faces


# Define the return type as a named tuple
ShapePreparationResult = namedtuple(
    "ShapePreparationResult",
    [
        "vertices",       # Original vertex coordinates
        "face_vertices",  # Sorted vertex lists for each face
        "face_areas",     # Precomputed area for each face
        "decomposition",  # The matched convex decomposition object
        "tolerance",      # Numerical tolerance used for matching
    ],
)


def prepare_shape_geometry(shape_file, num_edges: int, num_faces: int, verbose: bool = True) -> ShapePreparationResult:
    """
    Load and categorize the reference particle shape.

    This function executes the shape-preparation stage of the workflow:

        1. Load particle vertices from JSON file
        2. Categorize faces and edges via convex decomposition
        3. Validate topology matches expected counts
        4. Sort face records deterministically
        5. Return structured geometry data

    The resulting geometry is used by all later analysis stages to compute
    contact metrics, overlaps, and other particle-interaction properties.

    Parameters
    ----------
    shape_file : str or Path
        Path to the particle-shape JSON file. The file should contain
        vertex data in the standard format (typically under key "8_vertices").

        File Format:
            {
                "8_vertices": [
                    [x0, y0, z0],
                    [x1, y1, z1],
                    ...
                ]
            }

    num_edges : int
        Expected number of polyhedron edges. Used as a check to ensure
        the loaded shape matches the expected topology.

    num_faces : int
        Expected number of polygonal faces. Also used to validate the
        loaded shape matches the expected topology.

    verbose : bool, default True
        If True, print face-categorization summary to stdout when a
        matching decomposition is found. Useful for debugging and
        confirming geometry is loaded correctly. Set to False to suppress
        output.

    Returns
    -------
    ShapePreparationResult
        Named tuple containing prepared geometry data:

        vertices : list of list
            The loaded vertex coordinates, shape (N, 3) where N is the
            number of vertices. These are the original vertices from the
            JSON file, used as reference for all geometric calculations.

        face_vertices : list of list
            Vertex indices for each face, sorted deterministically.
            Format: [[v0, v1, v2, ...], [v3, v4, v5, ...], ...]
            Each inner list is a face, containing indices into vertices.

        face_areas : list of float
            Precomputed area for each face, in the same order as
            face_vertices. Used later for contact-area calculations.

        decomposition : ConvexDecomposition
            The matched convex decomposition object containing detailed
            topology information (edges, faces, connectivity, etc.).
            Can be cached and inspected for debugging.

        tolerance : float
            The numerical tolerance that produced the expected topology.
            Used for reproducibility and for understanding numerical
            precision requirements.


    Example
    -------
    Prepare a cubic particle geometry:

        try:
            shape_geometry = prepare_shape_geometry(
                shape_file="shape_Cube.json",
                num_edges=12,
                num_faces=6,
                verbose=True,
            )
            print(f"Loaded {len(shape_geometry.vertices)} vertices")
            print(f"Decomposed into {len(shape_geometry.face_vertices)} faces")
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            print(f"Failed to prepare shape: {exc}")
            sys.exit(2)

        # The result is now ready for use in contact analysis
        result = run_contact_analysis(
            shape_geometry=shape_geometry,
            ...
        )
    """

    # Step 1: Load the vertex coordinates from the shape JSON file
    vertices = load_shape_vertices(shape_file)

    # Step 2: Categorize the polyhedron faces
    # This performs:
    #   - Convex decomposition with tolerance iteration
    #   - Topology matching against num_edges and num_faces
    #   - Face sorting and area computation
    # Errors (ValueError, RuntimeError) propagate with descriptive messages
    categorization = categorize_polyhedron_faces(vertices=vertices, num_edges=num_edges, num_faces=num_faces, verbose=verbose)

    # Step 3: Build and return the result
    # Package all the geometry data into the ShapePreparationResult namedtuple
    # This makes all the data easily accessible with named attributes
    return ShapePreparationResult(vertices=vertices, face_vertices=categorization.face_vertices,
        face_areas=categorization.face_areas, decomposition=categorization.decomposition,
        tolerance=categorization.tolerance,
    )
