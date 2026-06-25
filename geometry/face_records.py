"""
Construction and deterministic sorting of polygonal face records.

Overview
--------
This module handles the extraction and organization of face data from a
convex decomposition. It takes a set of polygonal faces and produces:

    - Face vertices (coordinates for each face)
    - Face areas (precomputed surface areas)
    - Face signatures (geometric fingerprints for uniqueness)
    - Deterministic sorting order (reproducible results)

Module Organization
-------------------

    shape_signature()
        Compute a geometric fingerprint for one face

    build_face_records()
        Extract vertices, area, and signature for all faces

    sort_face_records()
        Sort faces by vertex count and area deterministically

    get_sorted_face_vertices_and_areas()
        High-level convenience function orchestrating all steps

Sorting Strategy
----------------

    Level 1 (Primary): Sort by number of vertices (descending)
        - Faces with more vertices come first
        - Example: Hexagon (6 vertices) before Square (4 vertices)

    Level 2 (Secondary): Sort by area (descending)
        - Among faces with same vertex count, larger faces come first
        - Example: Large square before small square

This sorting order is deterministic and reproducible, ensuring consistent
results across multiple runs.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple

import compas.geometry
import numpy as np

# Type alias for convenience; a face record is a dictionary
FaceRecord = Dict[str, Any]


def shape_signature(face_points) -> Tuple[int, tuple, float]:
    """
    Compute the reference geometric signature of one polygonal face.

    The signature has the form:

        (
            number_of_vertices,
            sorted_rounded_edge_lengths,
            rounded_area,
        )

    Edge lengths and area are rounded to six decimal places, matching the
    reference implementation.

    Parameters
    ----------
    face_points
        Polygon vertices with shape ``(N, 3)``.

    Returns
    -------
    tuple
        ``(number_of_vertices, edge_lengths, area)``

    Raises
    ------
    ValueError
        If the face points do not have shape ``(N, 3)`` or contain fewer than
        three vertices.
    """
    points = np.asarray(
        face_points,
        dtype=float,
    )

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"Face points must have shape (N, 3), got {points.shape}."
        )

    if points.shape[0] < 3:
        raise ValueError(
            "A polygonal face must contain at least 3 vertices."
        )

    edge_lengths = np.linalg.norm(
        points - np.roll(points, -1, axis=0),
        axis=1,
    )

    sorted_edge_lengths = np.sort(edge_lengths)

    area = compas.geometry.area_polygon(
        [
            np.asarray(point, dtype=float)
            for point in points
        ]
    )

    return (
        points.shape[0],
        tuple(np.round(sorted_edge_lengths, 6)),
        round(float(area), 6),
    )


def build_face_records(vertices, faces) -> List[FaceRecord]:
    """
    Build vertices, area, and signature records for all polygonal faces.

    This function processes the output of convex decomposition and creates
    structured records for each face containing all relevant geometric data.

    Processing Steps
    ----------------
    For each face in the decomposition:
        1. Extract vertex coordinates from the vertex array
        2. Convert to nested list format for storage
        3. Compute polygon area using COMPAS geometry
        4. Compute geometric signature using shape_signature()
        5. Package into a FaceRecord dictionary
        6. Collect all records into a list

    Parameters
    ----------
    vertices
        Complete polyhedron vertex array with shape (N, 3) where N is the
        number of vertices. Accepted formats:

    faces
        Iterable of face connectivity sequences. Each face is a sequence
        (list, tuple, array, etc.) of integer indices into the vertices array.

    Returns
    -------
    list of dict
        List of FaceRecord dictionaries, one per input face. Each record
        contains:

        "vertices" : list of list
            Vertex coordinates for this face: [[x0, y0, z0], [x1, y1, z1], ...]
            Coordinates are in the same order as extracted from the input vertices array.

        "area" : float
            Computed polygon area using COMPAS geometry.area_polygon().

        "signature" : tuple
            Result from shape_signature(); includes vertex count, sorted edge
            lengths, and rounded area.
    """

    # Step 1: Convert vertices to NumPy float array for uniform handling
    vertices = np.asarray(vertices, dtype=float)

    # Step 2: Initialize empty list to accumulate face records
    face_records: List[FaceRecord] = []

    # Step 3: Process each face
    for face_id, face in enumerate(faces):
        try:
            # Step 3a: Extract vertex coordinates for this face
            # Map each vertex index to its coordinates from the vertices array
            face_vertices_array = [np.asarray(vertices[index], dtype=float) for index in face]

            # Step 3b: Convert from NumPy arrays to nested lists
            face_vertices_list = [vertex.tolist() for vertex in face_vertices_array]

            # Step 3c: Compute polygon area using COMPAS geometry
            face_area = compas.geometry.area_polygon(face_vertices_array)

            # Step 3d: Compute geometric signature for this face
            face_geometry_signature = shape_signature(face_vertices_list)

        except IndexError as exc:
            raise IndexError(
                f"Face {face_id} contains a vertex index outside "
                "the valid range."
            ) from exc

        except Exception as exc:
            raise RuntimeError(
                f"Failed to process face {face_id} during "
                "face categorization."
            ) from exc

        # Step 3e: Create and store the face record
        face_records.append(
            {
                "vertices": face_vertices_list,
                "area": float(face_area),
                "signature": face_geometry_signature,
            }
        )

    # Step 4: Validate we produced at least one record
    if len(face_records) == 0:
        raise ValueError("No valid polygonal faces were extracted.")

    return face_records


def sort_face_records(face_records: List[FaceRecord]) -> List[FaceRecord]:
    """
    Sorting Strategy
    ----------------
    The sorting uses Python's stable sort to maintain a specific order:

        Step 1 : Sort by area descending
            - Larger areas come first

        Step 2 : Stable sort by vertex count descending
            - Faces with more vertices come first
            - Since the sort is stable, faces with the same vertex count
              retain their relative order from Step 1
            - This means within same vertex count, larger areas come first

    Parameters
    ----------
    face_records : list of dict
        Face records as produced by build_face_records(). Each record
        contains:
            - "vertices" : list of coordinates
            - "area" : float area value
            - "signature" : tuple with (vertex_count, edges, area)

        The input list is modified in-place and also returned.

    Returns
    -------
    list of dict
        The sorted face_records list (same object as input, modified in-place).
    """

    # Step 1: Sort by area descending (larger areas come first)
    face_records.sort(key=lambda record: record["area"], reverse=True)

    # Step 2: Stable sort by number of vertices descending (more vertices come first)
    # Since this is a stable sort, faces with the same vertex count maintain
    # their relative order from Step 1, which sorts them by area descending
    face_records.sort(key=lambda record: len(record["vertices"]), reverse=True)

    return face_records


def get_sorted_face_vertices_and_areas(vertices, faces):
    """
    Build and sort face records, then return the reference two-list output.

    This is a high-level convenience function that orchestrates the complete
    face-processing pipeline and returns results in the format expected by
    the reference workflow.

    Processing Pipeline
    -------------------
    The function performs these steps in sequence:

        1. build_face_records()
           Extract vertices, area, and signature for each face
        2. sort_face_records()
           Sort faces deterministically by vertex count and area
        3. Extract and return
           Separate vertices and areas into two parallel lists

    Return Format
    -------------
    The function returns two parallel lists:

        face_vertices_sorted
            Nested list of vertex coordinates, one per face
            [[v0, v1, v2, v3], [v4, v5, v6], ...]

        face_areas_sorted
            Float areas in the same order as face_vertices_sorted
            [1.0, 1.5, 1.0, ...]

    Parameters
    ----------
    vertices
        Polyhedron vertex array, shape (N, 3)

    faces
        Iterable of face vertex indices

    Returns
    -------
    tuple of (face_vertices_sorted, face_areas_sorted)
        face_vertices_sorted : list of list
            Sorted vertex lists for each face

        face_areas_sorted : list of float
            Face areas in the same order as face_vertices_sorted

    Example
    -------
    Use this as the main entry point:

        from geometry import get_sorted_face_vertices_and_areas

        vertices = [... vertex array ...]
        faces = [... face connectivity ...]

        face_vertices, face_areas = get_sorted_face_vertices_and_areas(
            vertices, faces
        )

        # Now can use the sorted data
        for vertices, area in zip(face_vertices, face_areas):
            print(f"Face with area {area} has {len(vertices)} vertices")

    """

    # Step 1: Build face records with all geometric data
    face_records = build_face_records(vertices, faces)

    # Step 2: Sort records deterministically
    sort_face_records(face_records)

    # Step 3: Extract just the vertices list from each record
    face_vertices_sorted = [
        record["vertices"]
        for record in face_records
    ]

    # Step 4: Extract just the area list from each record
    face_areas_sorted = [
        record["area"]
        for record in face_records
    ]

    # Step 5: Return the two parallel lists
    return (face_vertices_sorted, face_areas_sorted)
