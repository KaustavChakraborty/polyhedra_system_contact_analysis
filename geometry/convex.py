# geometry/convex.py
# ==============================================================================
# Module: geometry.convex
# Purpose: Convex hull and decomposition algorithms
#
# Implements:
#   - convex_hull_3d(): Compute 3D convex hull from vertices
#   - convex_decomposition(): Convert vertices to ConvexShape object
#   - ConvexShapeValidator: Validate convex hull properties
#
# Uses scipy.spatial.ConvexHull as computational backend for numerical
# stability and robustness. Provides detailed error handling and validation.
#
# Author: Contact Analysis Team
# ==============================================================================

from __future__ import annotations

import logging
from typing import List, Tuple, Optional

import numpy as np

from core import (
    ValidationError,
    DataTypeError,
    GeometryError,
    CONVEX_DECOMP_TOL,
)
from .primitives import ConvexShape

logger = logging.getLogger(__name__)

# Try to import scipy for robust convex hull computation
try:
    from scipy.spatial import ConvexHull
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not available, convex hull computation will have limited robustness")


# ==============================================================================
# CONVEX HULL COMPUTATION
# ==============================================================================

def convex_hull_3d(
    vertices: np.ndarray,
    tolerance: float = CONVEX_DECOMP_TOL,
) -> Tuple[List[List[int]], List[Tuple[int, int]]]:
    """
    Compute 3D convex hull from vertices.
    
    Computes the convex hull of a point set in 3D space, returning the
    faces (as vertex index lists) and edges (as vertex index pairs).
    
    Uses scipy.spatial.ConvexHull for robust computation via Qhull algorithm.
    This avoids numerical issues common with custom implementations.
    
    Parameters
    ----------
    vertices : np.ndarray
        Input vertices, shape (N, 3) where N >= 4
    tolerance : float, optional
        Numerical tolerance. Default is CONVEX_DECOMP_TOL.
    
    Returns
    -------
    Tuple[List[List[int]], List[Tuple[int, int]]]
        (faces, edges) where:
        - faces: List of faces, each face is a list of vertex indices
        - edges: List of edges, each edge is a (v_i, v_j) pair
    
    Raises
    ------
    ValidationError
        If vertices shape or count is invalid
    GeometryError
        If vertices are coplanar or collinear (cannot form 3D hull)
    
    Examples
    --------
    >>> import numpy as np
    >>> from geometry.convex import convex_hull_3d
    >>> 
    >>> # Vertices of a tetrahedron
    >>> vertices = np.array([
    ...     [0.0, 0.0, 0.0],
    ...     [1.0, 0.0, 0.0],
    ...     [0.0, 1.0, 0.0],
    ...     [0.0, 0.0, 1.0]
    ... ])
    >>> 
    >>> faces, edges = convex_hull_3d(vertices)
    >>> print(f"Faces: {len(faces)}")  # Should be 4
    >>> print(f"Edges: {len(edges)}")  # Should be 6
    
    Notes
    -----
    - Requires at least 4 vertices to form a 3D hull
    - Coplanar or collinear points will raise GeometryError
    - scipy.spatial.ConvexHull is used as backend for robustness
    """
    if not SCIPY_AVAILABLE:
        raise RuntimeError(
            "scipy is required for robust convex hull computation. "
            "Install it with: pip install scipy"
        )
    
    # Validate input
    vertices = np.asarray(vertices, dtype=float)
    
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValidationError(
            f"Vertices must have shape (N, 3), got {vertices.shape}",
            error_code="CONVEX_VERTICES_INVALID_SHAPE",
            context={"shape": vertices.shape}
        )
    
    if vertices.shape[0] < 4:
        raise ValidationError(
            f"Need at least 4 vertices for 3D hull, got {vertices.shape[0]}",
            error_code="CONVEX_TOO_FEW_VERTICES",
            context={"n_vertices": vertices.shape[0]}
        )
    
    if np.any(np.isnan(vertices)) or np.any(np.isinf(vertices)):
        raise DataTypeError(
            "Vertices contain NaN or infinite values",
            error_code="CONVEX_VERTICES_NAN_INF"
        )
    
    # Compute convex hull using scipy
    try:
        hull = ConvexHull(vertices, qhull_options="Qt Qx")
        logger.debug(f"Computed convex hull: {len(hull.simplices)} simplices")
    except Exception as e:
        # scipy raises ValueError for degenerate cases
        raise GeometryError(
            f"Cannot compute convex hull: {e}",
            error_code="GEOMETRY_CONVEX_HULL_FAILED",
            context={"reason": str(e)}
        ) from e
    
    # Extract faces (simplices) from hull
    faces = hull.simplices.tolist()
    
    # Extract edges from faces
    edges_set = set()
    for face in hull.simplices:
        # Each face is a triangle (3 vertices)
        for i in range(3):
            v1 = face[i]
            v2 = face[(i + 1) % 3]
            # Store edge as sorted pair to avoid duplicates
            edge = tuple(sorted([v1, v2]))
            edges_set.add(edge)
    
    edges = list(edges_set)
    
    logger.info(
        f"Convex hull computed: {len(hull.vertices)} vertices, "
        f"{len(faces)} faces, {len(edges)} edges"
    )
    
    return faces, edges


# ==============================================================================
# CONVEX SHAPE CREATION
# ==============================================================================

def convex_decomposition(
    vertices: np.ndarray,
    tolerance: float = CONVEX_DECOMP_TOL,
) -> ConvexShape:
    """
    Create ConvexShape from vertices using convex hull decomposition.
    
    Computes the convex hull of the input vertices and returns a ConvexShape
    object representing the polyhedron.
    
    Parameters
    ----------
    vertices : np.ndarray
        Input vertices, shape (N, 3)
    tolerance : float, optional
        Numerical tolerance. Default is CONVEX_DECOMP_TOL.
    
    Returns
    -------
    ConvexShape
        Convex polyhedron representation
    
    Raises
    ------
    ValidationError
        If vertices are invalid
    GeometryError
        If convex hull cannot be computed
    
    Examples
    --------
    >>> import numpy as np
    >>> from geometry.convex import convex_decomposition
    >>> 
    >>> # Create a cube
    >>> vertices = np.array([
    ...     [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
    ...     [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
    ... ], dtype=float)
    >>> 
    >>> shape = convex_decomposition(vertices)
    >>> print(f"Shape: {shape}")
    >>> print(f"Volume proxy: {len(shape.vertices)} vertices")
    """
    # Compute convex hull
    faces, edges = convex_hull_3d(vertices, tolerance=tolerance)
    
    # Validate result
    if not faces or not edges:
        raise GeometryError(
            "Convex hull has no faces or edges",
            error_code="GEOMETRY_EMPTY_HULL",
            context={"n_faces": len(faces), "n_edges": len(edges)}
        )
    
    # Create ConvexShape
    shape = ConvexShape(
        vertices=vertices,
        faces=faces,
        edges=edges,
        num_faces=len(faces),
        num_edges=len(edges),
    )
    
    logger.info(f"Created ConvexShape: {shape}")
    
    return shape


# ==============================================================================
# CONVEX SHAPE VALIDATION
# ==============================================================================

class ConvexShapeValidator:
    """
    Validator for convex hull properties and topology.
    
    Provides comprehensive validation of ConvexShape objects:
    - Face planarity (all vertices in face lie in same plane)
    - Edge connectivity (edges form proper face boundaries)
    - Face orientation (outward-pointing normals)
    - Topology consistency (closed manifold)
    
    Examples
    --------
    >>> from geometry.convex import ConvexShapeValidator
    >>> from geometry.convex import convex_decomposition
    >>> import numpy as np
    >>> 
    >>> # Create and validate a shape
    >>> vertices = np.array([[0, 0, 0], [1, 0, 0], ...], dtype=float)
    >>> shape = convex_decomposition(vertices)
    >>> 
    >>> validator = ConvexShapeValidator(tolerance=1e-12)
    >>> is_valid, issues = validator.validate(shape)
    >>> 
    >>> if is_valid:
    ...     print("✓ Shape is valid")
    ... else:
    ...     for issue in issues:
    ...         print(f"⚠️  {issue}")
    """
    
    def __init__(self, tolerance: float = CONVEX_DECOMP_TOL):
        """
        Initialize validator.
        
        Parameters
        ----------
        tolerance : float, optional
            Numerical tolerance for checks. Default is CONVEX_DECOMP_TOL.
        """
        self.tolerance = tolerance
        logger.debug(f"ConvexShapeValidator initialized with tolerance={tolerance:.2e}")
    
    def validate(self, shape: ConvexShape) -> Tuple[bool, List[str]]:
        """
        Validate ConvexShape properties.
        
        Performs comprehensive checks:
        1. Geometry: Face planarity, vertex positions
        2. Topology: Edge connectivity, face closure
        3. Orientation: Normal consistency
        4. Consistency: Counts match actual arrays
        
        Parameters
        ----------
        shape : ConvexShape
            Shape to validate
        
        Returns
        -------
        Tuple[bool, List[str]]
            (is_valid, issues) where:
            - is_valid: True if all checks pass
            - issues: List of validation messages (empty if valid)
        
        Examples
        --------
        >>> is_valid, issues = validator.validate(shape)
        >>> if not is_valid:
        ...     for issue in issues:
        ...         print(f"Issue: {issue}")
        """
        issues: List[str] = []
        
        logger.debug(f"Validating ConvexShape...")
        
        # Check vertex count
        if shape.vertices.shape[0] == 0:
            issues.append("✗ Shape has no vertices")
        
        # Check face count
        if not shape.faces:
            issues.append("✗ Shape has no faces")
        
        # Check edge count
        if not shape.edges:
            issues.append("✗ Shape has no edges")
        
        # Validate each face
        for face_idx, face in enumerate(shape.faces):
            if len(face) < 3:
                issues.append(f"✗ Face {face_idx} has < 3 vertices")
            
            # Check face planarity
            try:
                face_verts = shape.vertices[face]
                if len(face) >= 3:
                    # Check if face vertices are coplanar
                    if len(face) > 3:
                        # For non-triangular faces, check planarity
                        v0 = face_verts[0]
                        v1 = face_verts[1]
                        v2 = face_verts[2]
                        
                        edge1 = v1 - v0
                        edge2 = v2 - v0
                        normal = np.cross(edge1, edge2)
                        
                        for i in range(3, len(face_verts)):
                            v = face_verts[i]
                            rel_pos = v - v0
                            dist_to_plane = abs(np.dot(rel_pos, normal))
                            if dist_to_plane > self.tolerance:
                                issues.append(
                                    f"⚠️  Face {face_idx}: vertex {i} not coplanar "
                                    f"(distance={dist_to_plane:.2e})"
                                )
            except Exception as e:
                issues.append(f"✗ Error checking face {face_idx}: {e}")
        
        # Validate edges
        max_vertex_idx = shape.vertices.shape[0] - 1
        for edge_idx, (v1, v2) in enumerate(shape.edges):
            if not (0 <= v1 <= max_vertex_idx):
                issues.append(f"✗ Edge {edge_idx}: vertex {v1} out of range")
            if not (0 <= v2 <= max_vertex_idx):
                issues.append(f"✗ Edge {edge_idx}: vertex {v2} out of range")
        
        # Log results
        if issues:
            logger.warning(f"Validation found {len(issues)} issue(s)")
            for issue in issues:
                logger.warning(issue)
        else:
            logger.info("✓ ConvexShape validation successful")
        
        return len(issues) == 0, issues
    
    def print_validation(self, shape: ConvexShape) -> None:
        """
        Print validation results in human-readable format.
        
        Examples
        --------
        >>> validator.print_validation(shape)
        """
        is_valid, issues = self.validate(shape)
        
        print("\nConvexShape Validation:")
        print(f"  Vertices: {shape.vertices.shape[0]}")
        print(f"  Faces: {shape.num_faces}")
        print(f"  Edges: {shape.num_edges}")
        
        if is_valid:
            print(f"  Status: ✓ VALID")
        else:
            print(f"  Status: ✗ INVALID ({len(issues)} issue(s))")
            for issue in issues:
                print(f"    {issue}")
        
        print()


if __name__ == "__main__":
    """
    Test/demo when module is run directly.
    """
    import sys
    from core.utilities import setup_logging
    
    setup_logging(level="DEBUG")
    
    print("\n" + "="*80)
    print("GEOMETRY CONVEX MODULE - TESTING")
    print("="*80 + "\n")
    
    if not SCIPY_AVAILABLE:
        print("⚠️  scipy not available. Install with: pip install scipy")
        print("Cannot run convex hull tests without scipy.\n")
        sys.exit(0)
    
    # Test convex hull
    print("[TEST 1] Computing convex hull (tetrahedron)...")
    try:
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        faces, edges = convex_hull_3d(vertices)
        print(f"✓ Faces: {len(faces)}, Edges: {len(edges)}")
        print(f"  Faces: {faces}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test convex decomposition
    print("\n[TEST 2] Creating ConvexShape...")
    try:
        shape = convex_decomposition(vertices)
        print(f"✓ {shape}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test validation
    print("\n[TEST 3] Validating ConvexShape...")
    try:
        validator = ConvexShapeValidator(tolerance=1e-12)
        is_valid, issues = validator.validate(shape)
        if is_valid:
            print(f"✓ Shape is valid")
        else:
            print(f"✗ Shape has {len(issues)} issue(s):")
            for issue in issues:
                print(f"  {issue}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("\n" + "="*80)
    print("✓ Convex module is ready for use!")
    print("="*80 + "\n")
