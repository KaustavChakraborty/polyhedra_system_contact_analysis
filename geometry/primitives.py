# geometry/primitives.py
# ==============================================================================
# Module: geometry.primitives
# Purpose: Immutable data structures for geometric objects
#
# This module defines the fundamental geometric types:
#   - Polygon: 2D polygon in 3D space (N vertices, defined by indices)
#   - ConvexShape: 3D convex polyhedron (vertices, faces, edges)
#
# Design principles:
#   - Frozen dataclasses (immutable) prevent bugs
#   - Properties compute derived values (area, centroid, normal)
#   - Validation in __post_init__
#   - Minimal data, maximum safety
#
# Author: Contact Analysis Team
# ==============================================================================

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from core import ValidationError, DataTypeError, GeometryError

logger = logging.getLogger(__name__)


# ==============================================================================
# POLYGON: 2D POLYGON IN 3D SPACE
# ==============================================================================

@dataclass(frozen=True)
class Polygon:
    """
    Immutable 2D polygon embedded in 3D space.
    
    Represents a flat polygon defined by a list of vertices. The polygon is
    defined in a parent vertex array using indices. All geometric properties
    (area, centroid, normal) are computed lazily.
    
    Attributes
    ----------
    vertices : np.ndarray
        Actual 3D vertex coordinates, shape (N, 3)
    face_indices : List[int]
        Indices into parent vertex array that define this face
    
    Raises
    ------
    ValidationError
        If vertices shape is invalid
    DataTypeError
        If vertices contain NaN/Inf
    
    Examples
    --------
    >>> import numpy as np
    >>> from geometry.primitives import Polygon
    >>> 
    >>> # Define a triangle
    >>> verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    >>> indices = [0, 1, 2]
    >>> poly = Polygon(vertices=verts, face_indices=indices)
    >>> 
    >>> # Compute properties
    >>> print(f"Area: {poly.area}")
    >>> print(f"Centroid: {poly.centroid}")
    >>> print(f"Normal: {poly.normal}")
    
    Notes
    -----
    - Vertices must be coplanar (checked during geometric operations)
    - Vertices are stored in the order given (not reordered)
    - Use geometry.calculator.reorder_polygon_vertices() if ordering matters
    - Properties are computed on-demand but should be cached by caller
    """
    
    vertices: np.ndarray       # Shape: (N, 3)
    face_indices: List[int]    # Indices into parent array
    
    def __post_init__(self) -> None:
        """
        Validate polygon vertices after initialization.
        
        Checks that:
        - vertices is 2D array with shape (N, 3)
        - N >= 3 (at least a triangle)
        - No NaN or Inf values
        - vertices are numeric and convertible to float
        """
        # Validate shape
        vertices = np.asarray(self.vertices, dtype=float)
        
        if vertices.ndim != 2:
            raise ValidationError(
                f"Polygon vertices must be 2D array, got shape {vertices.shape}",
                error_code="POLYGON_VERTICES_INVALID_NDIM",
                context={"shape": vertices.shape, "ndim": vertices.ndim}
            )
        
        if vertices.shape[1] != 3:
            raise ValidationError(
                f"Polygon vertices must have 3 coordinates (x,y,z), got {vertices.shape[1]}",
                error_code="POLYGON_VERTICES_INVALID_COORDS",
                context={"shape": vertices.shape, "coords": vertices.shape[1]}
            )
        
        if vertices.shape[0] < 3:
            raise ValidationError(
                f"Polygon needs at least 3 vertices, got {vertices.shape[0]}",
                error_code="POLYGON_TOO_FEW_VERTICES",
                context={"n_vertices": vertices.shape[0]}
            )
        
        # Validate no NaN/Inf
        if np.any(np.isnan(vertices)):
            raise DataTypeError(
                "Polygon vertices contain NaN values",
                error_code="POLYGON_VERTICES_NAN"
            )
        
        if np.any(np.isinf(vertices)):
            raise DataTypeError(
                "Polygon vertices contain infinite values",
                error_code="POLYGON_VERTICES_INF"
            )
        
        # Store as float array (in case input was int)
        object.__setattr__(self, 'vertices', vertices)
        
        logger.debug(f"Polygon created: {vertices.shape[0]} vertices, indices {self.face_indices}")
    
    @property
    def num_vertices(self) -> int:
        """Return number of vertices in polygon."""
        return len(self.vertices)
    
    @property
    def area(self) -> float:
        """
        Compute polygon area using Shoelace formula for 3D polygon.
        
        Returns
        -------
        float
            Area of the polygon (always non-negative)
        """
        # Import here to avoid circular dependency
        from .calculator import polygon_area_3d
        return polygon_area_3d(self.vertices)
    
    @property
    def centroid(self) -> np.ndarray:
        """
        Compute centroid (center of mass) of polygon vertices.
        
        Returns
        -------
        np.ndarray
            Centroid coordinates, shape (3,)
        """
        from .calculator import polygon_centroid
        return polygon_centroid(self.vertices)
    
    @property
    def normal(self) -> np.ndarray:
        """
        Compute normal vector to the polygon plane.
        
        Returns
        -------
        np.ndarray
            Outward normal vector, shape (3,), unit length
        """
        from .calculator import polygon_normal
        return polygon_normal(self.vertices)
    
    def __repr__(self) -> str:
        """Return detailed representation."""
        return (
            f"Polygon(n_vertices={self.num_vertices}, "
            f"indices={self.face_indices[:5]}{'...' if len(self.face_indices) > 5 else ''})"
        )


# ==============================================================================
# CONVEX SHAPE: 3D CONVEX POLYHEDRON
# ==============================================================================

@dataclass(frozen=True)
class ConvexShape:
    """
    Immutable 3D convex polyhedron with faces and edges.
    
    Represents a convex polyhedron defined by its vertices, faces, and edges.
    This is the fundamental geometry representation for particle shapes.
    
    Attributes
    ----------
    vertices : np.ndarray
        All vertex coordinates, shape (V, 3)
    faces : List[List[int]]
        Face definitions as lists of vertex indices
        Each face must have >= 3 vertices
    edges : List[Tuple[int, int]]
        Edge definitions as (vertex_i, vertex_j) index pairs
    num_faces : int
        Number of faces (for quick access)
    num_edges : int
        Number of edges (for quick access)
    
    Raises
    ------
    ValidationError
        If any field is invalid
    
    Examples
    --------
    >>> import numpy as np
    >>> from geometry.primitives import ConvexShape
    >>> 
    >>> # Create a tetrahedron
    >>> vertices = np.array([
    ...     [0, 0, 0],
    ...     [1, 0, 0],
    ...     [0, 1, 0],
    ...     [0, 0, 1]
    ... ], dtype=float)
    >>> 
    >>> faces = [
    ...     [0, 1, 2],  # bottom
    ...     [0, 1, 3],  # side 1
    ...     [0, 2, 3],  # side 2
    ...     [1, 2, 3]   # side 3
    ... ]
    >>> 
    >>> edges = [
    ...     (0, 1), (0, 2), (0, 3),
    ...     (1, 2), (1, 3), (2, 3)
    ... ]
    >>> 
    >>> shape = ConvexShape(
    ...     vertices=vertices,
    ...     faces=faces,
    ...     edges=edges,
    ...     num_faces=4,
    ...     num_edges=6
    ... )
    
    Notes
    -----
    - All faces must be convex and properly oriented
    - Vertices must form a convex hull (no concavities)
    - Edges connect adjacent vertices in faces
    - This is typically created by convex_decomposition()
    """
    
    vertices: np.ndarray           # Shape: (V, 3)
    faces: List[List[int]]         # Each face is a list of vertex indices
    edges: List[Tuple[int, int]]   # Each edge is (v_i, v_j)
    num_faces: int                 # Number of faces
    num_edges: int                 # Number of edges
    
    def __post_init__(self) -> None:
        """
        Validate ConvexShape after initialization.
        
        Checks that:
        - vertices is (V, 3) array
        - num_faces and num_edges match actual counts
        - Face indices are in valid range
        """
        # Validate vertices
        vertices = np.asarray(self.vertices, dtype=float)
        
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValidationError(
                f"Shape vertices must be (V, 3), got {vertices.shape}",
                error_code="SHAPE_VERTICES_INVALID_SHAPE",
                context={"shape": vertices.shape}
            )
        
        if np.any(np.isnan(vertices)) or np.any(np.isinf(vertices)):
            raise DataTypeError(
                "Shape vertices contain NaN or Inf",
                error_code="SHAPE_VERTICES_NAN_INF"
            )
        
        object.__setattr__(self, 'vertices', vertices)
        
        # Validate num_faces
        if self.num_faces != len(self.faces):
            raise ValidationError(
                f"num_faces mismatch: {self.num_faces} vs {len(self.faces)}",
                error_code="SHAPE_NUM_FACES_MISMATCH",
                context={"stated": self.num_faces, "actual": len(self.faces)}
            )
        
        # Validate num_edges
        if self.num_edges != len(self.edges):
            raise ValidationError(
                f"num_edges mismatch: {self.num_edges} vs {len(self.edges)}",
                error_code="SHAPE_NUM_EDGES_MISMATCH",
                context={"stated": self.num_edges, "actual": len(self.edges)}
            )
        
        # Validate face indices are in range
        n_vertices = vertices.shape[0]
        for face_idx, face in enumerate(self.faces):
            if len(face) < 3:
                raise ValidationError(
                    f"Face {face_idx} has < 3 vertices",
                    error_code="SHAPE_FACE_TOO_SMALL",
                    context={"face_idx": face_idx, "n_vertices": len(face)}
                )
            
            for vert_idx, v in enumerate(face):
                try:
                    v_int = int(v)
                except (TypeError, ValueError):
                    raise ValidationError(
                        f"Face {face_idx}, vertex {vert_idx}: index not int",
                        error_code="SHAPE_FACE_INDEX_TYPE"
                    )
                
                if not (0 <= v_int < n_vertices):
                    raise ValidationError(
                        f"Face {face_idx}: index {v_int} out of range [0, {n_vertices})",
                        error_code="SHAPE_FACE_INDEX_RANGE",
                        context={"face_idx": face_idx, "index": v_int, "max": n_vertices}
                    )
        
        # Validate edge indices
        for edge_idx, (v1, v2) in enumerate(self.edges):
            for v in [v1, v2]:
                if not (0 <= v < n_vertices):
                    raise ValidationError(
                        f"Edge {edge_idx}: index {v} out of range",
                        error_code="SHAPE_EDGE_INDEX_RANGE"
                    )
        
        logger.debug(
            f"ConvexShape created: {vertices.shape[0]} vertices, "
            f"{len(self.faces)} faces, {len(self.edges)} edges"
        )
    
    def get_face_vertices(self, face_idx: int) -> np.ndarray:
        """
        Get vertices for a specific face.
        
        Parameters
        ----------
        face_idx : int
            Index of the face
        
        Returns
        -------
        np.ndarray
            Vertices of the face, shape (N, 3)
        
        Raises
        ------
        ValidationError
            If face_idx is out of range
        """
        if not (0 <= face_idx < len(self.faces)):
            raise ValidationError(
                f"Face index {face_idx} out of range [0, {len(self.faces)})",
                error_code="SHAPE_FACE_INDEX_OUT_OF_RANGE"
            )
        
        face_indices = self.faces[face_idx]
        return self.vertices[face_indices]
    
    def get_edge_vertices(self, edge_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the two vertices of an edge.
        
        Parameters
        ----------
        edge_idx : int
            Index of the edge
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Two vertices (v1, v2) of the edge
        
        Raises
        ------
        ValidationError
            If edge_idx is out of range
        """
        if not (0 <= edge_idx < len(self.edges)):
            raise ValidationError(
                f"Edge index {edge_idx} out of range [0, {len(self.edges)})",
                error_code="SHAPE_EDGE_INDEX_OUT_OF_RANGE"
            )
        
        v1_idx, v2_idx = self.edges[edge_idx]
        return self.vertices[v1_idx], self.vertices[v2_idx]
    
    def __repr__(self) -> str:
        """Return detailed representation."""
        return (
            f"ConvexShape(n_vertices={self.vertices.shape[0]}, "
            f"n_faces={self.num_faces}, n_edges={self.num_edges})"
        )


if __name__ == "__main__":
    """
    Test/demo when module is run directly.
    """
    import sys
    from core.utilities import setup_logging
    
    setup_logging(level="DEBUG")
    
    print("\n" + "="*80)
    print("GEOMETRY PRIMITIVES MODULE - TESTING")
    print("="*80 + "\n")
    
    # Test Polygon
    print("[TEST 1] Creating Polygon...")
    try:
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ])
        poly = Polygon(vertices=vertices, face_indices=[0, 1, 2])
        print(f"✓ {poly}")
        print(f"  Area: {poly.area:.6f}")
        print(f"  Centroid: {poly.centroid}")
        print(f"  Normal: {poly.normal}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test ConvexShape
    print("\n[TEST 2] Creating ConvexShape (tetrahedron)...")
    try:
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        faces = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
        edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        
        shape = ConvexShape(
            vertices=vertices,
            faces=faces,
            edges=edges,
            num_faces=4,
            num_edges=6
        )
        print(f"✓ {shape}")
        print(f"  Face 0 vertices:\n{shape.get_face_vertices(0)}")
        print(f"  Edge 0 vertices: {shape.get_edge_vertices(0)}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test invalid polygon
    print("\n[TEST 3] Invalid polygon (too few vertices)...")
    try:
        bad_verts = np.array([[0, 0, 0], [1, 0, 0]])  # Only 2 vertices
        poly = Polygon(vertices=bad_verts, face_indices=[0, 1])
    except ValidationError as e:
        print(f"✓ Correctly caught error: {e.error_code}")
    
    print("\n" + "="*80)
    print("✓ Primitives module is ready for use!")
    print("="*80 + "\n")
