"""
Geometry utilities for the modular contact-analysis project.

Overview
--------
This module provides low-level geometric computations and data structures
used throughout the contact-analysis workflow. It handles:

    - Convex decomposition and convex hull computation
    - Polygonal face classification and properties
    - Face record construction and sorting
    - Topology matching and validation
    - Distance calculations between geometric objects
    - Normal vector and alignment computations

Module Organization
-------------------
The geometry package is organized into specialized submodules:

    characteristic_length.py
        Compute characteristic length scales (sigma) from shape geometry

    convex_decomposition.py
        Core convex hull and decomposition algorithms

    face_classification.py
        Classify and categorize polygonal faces

    face_geometry.py
        Properties of individual faces (centers, normals)

    face_pair_geometry.py
        Properties of face pairs (distances, angles)

    face_records.py
        Construction and management of face data structures

    face_alignment.py
        Compute alignment between face pairs

    face_pair_record.py
        Data structures for face-pair information

    polygons.py
        Low-level polygon operations (area, vertex ordering)

    shape_categorization.py
        High-level shape categorization pipeline

    topology.py
        Topology matching and validation

----------
The geometry module exposes the following public functions and types:

    Polygon Operations:
        polygon_area_3d()
            Compute area of a 3D polygon
        reorder_polygon_vertices()
            Reorder vertices into canonical form

    Face Classification:
        classify_polygon_face()
            Determine face type and properties
        canonical_pair_type()
            Get canonical type for face pairs

    Convex Decomposition:
        ConvexDecomposition (class)
            Data structure for decomposition results
        convexHull()
            Compute convex hull (C++ interface)
        convexDecomposition()
            Decompose shape (C++ interface)
        convex_hull()
            Wrapper function for convex hull
        convex_decomposition()
            Wrapper function for decomposition

    Topology Handling:
        decomposition_tolerances()
            Generate tolerance values for testing
        get_shape_decomposition_matching_topology()
            Find decomposition matching expected topology

    Face Records:
        shape_signature()
            Compute shape signature for caching
        build_face_records()
            Construct face record structures
        sort_face_records()
            Sort face records deterministically
        get_sorted_face_vertices_and_areas()
            Get sorted vertices and areas for faces

    Shape Categorization:
        FaceCategorizationResult (namedtuple)
            Result of face categorization
        categorize_polyhedron_faces()
            Main entry point for shape categorization
        poly_face_categorization_func()
            Compatibility wrapper

    Characteristic Lengths:
        compute_sigma()
            Compute characteristic length
        compute_shape_sigma()
            Compute shape-specific length

    Face Geometry:
        face_centers()
            Compute center of each face
        unit_face_normal()
            Compute unit normal for a face

    Face-Pair Geometry:
        face_center_distance_matrix()
            Compute distances between face centers
        select_closest_face_center_pair()
            Find closest pair of face centers

    Face Alignment:
        selected_face_normal_alignment()
            Compute alignment between face normals

    Face-Pair Records:
        build_face_pair_record()
            Construct record for a face pair

Typical Usage Pattern
---------------------
High-level users typically only interact with shape_categorization:

    from geometry import categorize_polyhedron_faces

    result = categorize_polyhedron_faces(
        vertices=vertex_array,
        num_edges=12,
        num_faces=6,
        verbose=True,
    )

    faces = result.face_vertices
    areas = result.face_areas

Low-level users can access individual operations as needed:

    from geometry import (
        polygon_area_3d,
        face_centers,
        face_center_distance_matrix,
    )

    area = polygon_area_3d(vertices, face_indices)
    centers = face_centers(vertices, faces)
    distances = face_center_distance_matrix(centers)
"""


from .characteristic_length import (
    compute_shape_sigma,
    compute_sigma,
)

from .convex_decomposition import (
    ConvexDecomposition,
    convexDecomposition,
    convexHull,
    convex_decomposition,
    convex_hull,
)
from .face_classification import (
    canonical_pair_type,
    classify_polygon_face,
)
from .face_records import (
    build_face_records,
    get_sorted_face_vertices_and_areas,
    shape_signature,
    sort_face_records,
)
from .polygons import (
    polygon_area_3d,
    reorder_polygon_vertices,
)
from .topology import (
    decomposition_tolerances,
    get_shape_decomposition_matching_topology,
)

from .shape_categorization import (
    FaceCategorizationResult,
    categorize_polyhedron_faces,
    poly_face_categorization_func,
)


from .face_geometry import (
    face_centers,
    unit_face_normal,
)

from .face_pair_geometry import (
    face_center_distance_matrix,
    select_closest_face_center_pair,
)

from .face_alignment import (
    selected_face_normal_alignment,
)

from .face_pair_record import (
    build_face_pair_record,
)



__all__ = [
    "polygon_area_3d",
    "reorder_polygon_vertices",
    "classify_polygon_face",
    "canonical_pair_type",
    "ConvexDecomposition",
    "convexHull",
    "convexDecomposition",
    "convex_hull",
    "convex_decomposition",
    "decomposition_tolerances",
    "get_shape_decomposition_matching_topology",
    "shape_signature",
    "build_face_records",
    "sort_face_records",
    "get_sorted_face_vertices_and_areas",
    "FaceCategorizationResult",
    "categorize_polyhedron_faces",
    "poly_face_categorization_func",
    "compute_sigma",
    "compute_shape_sigma",
    "face_centers",
    "unit_face_normal",
    "face_center_distance_matrix",
    "select_closest_face_center_pair",
    "selected_face_normal_alignment",
    "build_face_pair_record",
]