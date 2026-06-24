"""
Convex-hull and convex-decomposition utilities.

Overview
--------
This module performs the core geometric operation of converting a point set
into a convex polyhedron representation with faces and edges. It handles
the delicate task of merging coplanar triangular facets into polygonal faces.

Core Challenge
--------------
The scipy ConvexHull algorithm produces triangular facets by default:

    Input: 8 vertices of a cube
    Output: 12 triangular facets (2 triangles per face)

But we want polygonal faces:

    Input: 8 vertices of a cube
    Output: 6 square faces

This module solves this by:
    1. Computing the convex hull as triangular facets
    2. Identifying triangles that belong to the same plane
    3. Merging coplanar triangles into a single polygonal face
    4. Ordering vertices of each polygon consistently

Tolerance and Coplanarity
-------------------------
The key challenge is determining which triangles belong to the same face.
Floating-point precision makes exact plane matching impossible, so we use
a tolerance parameter:

    Two plane equations (a, b, c, d) are coplanar if:
        ||plane1 - plane2|| < tolerance

    - Tight tolerance (1e-12): Strict coplanarity check
      Only triangles on nearly identical planes merge
    - Loose tolerance (1e-4): Relaxed coplanarity check
      Triangles on slightly different planes may merge

The tolerance directly affects the number of resulting faces.

Algorithm Overview
------------------
convexHull() performs these steps:

    1. Compute convex hull using scipy (triangular facets)
    2. Build a distance tree of plane equations
    3. Find pairs of coplanar triangles using tree queries
    4. Build connectivity graph of coplanar triangles
    5. Identify connected components (triangle groups for each face)
    6. For each group, order vertices consistently
    7. Return hull vertices and ordered polygonal faces

convexDecomposition() adds edges:

    1. Call convexHull() to get faces
    2. Extract unique edges from face perimeters
    3. Return ConvexDecomposition with vertices, edges, faces

Data Structures
---------------
ConvexDecomposition (namedtuple)
    Contains the complete geometric decomposition:
        - vertices : np.ndarray of shape (N, 3)
        - edges : set of (i, j) edge pairs
        - faces : list of np.ndarray face-vertex indices

Plane Equations
    scipy.spatial.ConvexHull.equations is shape (F, 4) where each row is
    (a, b, c, d) representing the plane equation: ax + by + cz + d = 0
"""

from __future__ import annotations

from collections import defaultdict, namedtuple

import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull, cKDTree

# Define the return type as a named tuple for immutability and clarity
ConvexDecomposition = namedtuple(
    "ConvexDecomposition",
    [
        "vertices",    # Hull vertex coordinates
        "edges",       # Unique edge pairs
        "faces",       # Ordered polygonal faces
    ],
)


def convexHull(vertices, tol):
    """
    Compute the 3D convex hull and merge coplanar triangular facets.

    This function takes a point set and produces a convex polyhedron
    represented as polygonal faces (not triangular facets). It handles
    the key challenge of merging coplanar triangles into larger polygons.

    Parameters
    ----------
    vertices
        Sequence of three-dimensional vertex coordinates. Accepted formats:
            - NumPy array: np.array([[x0, y0, z0], ...])
            - Nested lists: [[x0, y0, z0], ...]
            - Any array-like convertible to numpy

        Should have at least 4 non-colinear points. Fewer points raise
        ValueError from scipy.spatial.ConvexHull.

    tol : float
        Tolerance for identifying coplanar triangular facets.
        Two plane equations are considered coplanar if they differ by
        less than this tolerance.

        Typical values:
            - 1e-12 : Very tight; most triangles stay separate
            - 1e-6 : Moderate; most coplanar triangles merge
            - 1e-4 : Loose; aggressively merges triangles

        The tolerance directly affects the number of output faces:
            - Tight tolerance => more faces (less merging)
            - Loose tolerance => fewer faces (more merging)

    Returns
    -------
    tuple of (hull_points, polygonal_faces)
        hull_points : np.ndarray
            Vertex coordinates of the convex hull, shape (N, 3).
            Same as ConvexHull.points.

        polygonal_faces : list of np.ndarray
            Ordered polygonal faces. Each element is a NumPy array of
            vertex indices for one face, in counterclockwise order when
            viewed from outside the polyhedron.

    Notes
    -----
    - Vertex ordering within each face is counterclockwise from outside
    - Face ordering is arbitrary (order of connected components)
    - The algorithm is robust to floating-point precision issues
    - Plane equations use format: ax + by + cz + d = 0 with unit normals
    """

    # Step 1: Compute convex hull using scipy
    # This produces triangular facets and plane equations
    hull = ConvexHull(vertices)

    # Step 2: Build spatial index of plane equations
    # Each plane is represented by its equation coefficients (a, b, c, d)
    distance_tree = cKDTree(hull.equations)

    # Step 3: Find pairs of coplanar triangles
    # query_pairs returns all pairs of planes within distance 'tol'
    # This identifies which triangular facets belong to the same polygonal face
    triangle_pairs = distance_tree.query_pairs(tol)

    # Step 4: Build connectivity graph of coplanar triangles
    connectivity = np.zeros(
        (
            len(hull.simplices),
            len(hull.simplices),
        ),
        dtype=np.int32,
    )

    # Mark pairs of coplanar triangles as connected
    for i, j in triangle_pairs:
        connectivity[i, j] = 1
        connectivity[j, i] = 1

    # Step 5: Find connected components
    # Each component is a group of coplanar triangles that form one polygonal face
    # connected_components returns component labels for each triangle
    _, join_target = connected_components(connectivity, directed=False)

    # Step 6: Group triangles by their face component
    # Build dictionaries mapping component ID to triangle indices and plane normals
    faces = defaultdict(list)    # Maps component ID => list of triangle indices
    normals = defaultdict(list)  # Maps component ID => plane normal vector

    # Assign each triangle to its component and save the normal
    for index, target in enumerate(join_target):
        faces[target].append(index)                   # Triangle 'index' belongs to component 'target'
        normals[target] = hull.equations[index][:3]   # Save the plane normal

    # Step 7: Extract vertices for each merged face
    # For each component, collect all unique vertices from the triangles in that component
    face_vertices = [
        set(hull.simplices[faces[face_index]].flat)
        for face_index in sorted(faces)
    ]

    # Step 8: Get normals for each face (just one normal per face, from any triangle)
    face_normals = [
        normals[face_index]
        for face_index in sorted(faces)
    ]

    # Step 9: Build ordered polygonal faces
    polygonal_faces = []

    # Process each face (component) with its vertices and normal
    for normal, face_indices in zip(face_normals, face_vertices):
        # Convert vertex index set to array
        face = np.array(list(face_indices), dtype=np.uint32)

        # Get the 3D coordinates of the face vertices
        points = hull.points[face]
        # Step 10: Compute face center
        center = np.mean(points, axis=0)

        # Step 11: Construct an in-plane basis for ordering vertices

        # First basis vector: direction from center to first vertex
        plane_a = points[0] - center
        # Normalize to unit length
        plane_a /= np.sqrt(np.sum(plane_a ** 2))

        # Second basis vector: perpendicular to both the normal and plane_a
        # plane_b = normal * plane_a (cross product)
        plane_b = np.cross(normal, plane_a)

        # Step 12: Compute angles for each vertex around the center
        # Express each vertex as (distance_a, distance_b) in the plane basis
        displacements = points - center[np.newaxis, :]

        # Compute projections onto the two basis vectors
        # This gives us 2D coordinates for each vertex in the plane
        angles = np.arctan2(displacements.dot(plane_b), displacements.dot(plane_a))

        # Step 13: Sort vertices by angle (counterclockwise order)
        sort_indices = np.argsort(angles)
        face = face[sort_indices]

        # Step 14: Add the ordered face to the result list
        polygonal_faces.append(face)

    return hull.points, polygonal_faces


def convexDecomposition(vertices, tol):
    """
    Decompose a convex polyhedron into vertices, edges, and polygonal faces.

    This is the main entry point for convex decomposition. It computes the
    convex hull, merges coplanar triangles, and extracts the unique edges.

    Processing Pipeline
    -------------------
    1. Call convexHull() to compute polygonal faces
    2. Extract unique edges from face perimeters
    3. Return ConvexDecomposition with all components

    Edge Extraction
    ---------------
    Edges are extracted by walking around each face's perimeter:
        - For each face: [v0, v1, v2, ..., vn]
        - Edges: (v0,v1), (v1,v2), ..., (vn,v0)
        - Normalize to undirected: min(i,j), max(i,j)
        - Use a set to collect unique edges (no duplicates)

    Parameters
    ----------
    vertices
        Point set for convex hull computation (see convexHull for details)

    tol : float
        Coplanar-facet merging tolerance (see convexHull for details)

    Returns
    -------
    ConvexDecomposition
        Named tuple containing the complete decomposition:

        vertices : np.ndarray
            Hull vertex coordinates, shape (N, 3)

        edges : set
            Unique undirected edge-index pairs: {(i, j), ...}
            Each edge is stored with i < j (canonical form)

        faces : list of np.ndarray
            Ordered polygonal face-vertex indices
    """

    # Step 1: Compute convex hull and merge coplanar triangles
    hull_vertices, faces = convexHull(vertices, tol)

    # Step 2: Extract unique edges from face perimeters
    edges = set()

    # Walk around each face's perimeter to extract edges
    for face in faces:
        # For face [v0, v1, v2, ..., vn], compute edges:
        # (v0,v1), (v1,v2), ..., (vn,v0)
        for i, j in zip(
            face,
            np.roll(face, -1),    # Shift by 1: [v1, v2, ..., vn, v0]
        ):
            edges.add(
                (
                    min(i, j),
                    max(i, j),
                )
            )

    # Step 3: Build and return the ConvexDecomposition
    return ConvexDecomposition(hull_vertices, edges, faces)


convex_hull = convexHull
convex_decomposition = convexDecomposition
