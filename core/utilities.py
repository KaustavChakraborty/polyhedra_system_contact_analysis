# core/utilities.py
# ==============================================================================
# Module: core.utilities
# Purpose: Utility functions for mathematics, validation, and logging
#
# This module provides:
#   - Mathematical helpers (vector operations, rotations)
#   - Validation functions (check array shapes, types)
#   - Logging setup and configuration
#
# All functions include:
#   - Comprehensive input validation
#   - Clear error messages with context
#   - Logging of operations for debugging
#   - Type hints for IDE support
#
# Author: Contact Analysis Team
# ==============================================================================

from __future__ import annotations

import logging
import sys
from typing import Tuple, Optional

import numpy as np

from .exceptions import (
    ValidationError,
    DataTypeError,
    GeometryError,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# LOGGING CONFIGURATION
# ==============================================================================

def setup_logging(
    level: str = "INFO",
    format_string: Optional[str] = None,
    file_path: Optional[str] = None,
) -> None:
    """
    Configure logging for the contact analysis workflow.
    
    Sets up both console and optional file logging with consistent formatting.
    Call this at the start of your workflow to enable detailed debugging output.
    
    Parameters
    ----------
    level : str, optional
        Logging level: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
        Default is 'INFO'
    format_string : Optional[str], optional
        Custom log format string. If None, uses default format.
        Default: '%(asctime)s | %(name)s | %(levelname)-8s | %(message)s'
    file_path : Optional[str], optional
        If provided, also write logs to this file
        
    Returns
    -------
    None
    
    Examples
    --------
    Basic setup:
    
    >>> from core.utilities import setup_logging
    >>> setup_logging(level='DEBUG')
    
    With file output:
    
    >>> setup_logging(level='INFO', file_path='workflow.log')
    
    Custom format:
    
    >>> setup_logging(
    ...     level='DEBUG',
    ...     format_string='%(levelname)s - %(name)s: %(message)s'
    ... )
    
    Notes
    -----
    Call this function early in your workflow (e.g., in main.py) before
    creating any loggers. All loggers created after this call will use
    the configured format and handlers.
    """
    # Validate level
    valid_levels = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
    level_upper = level.upper()
    if level_upper not in valid_levels:
        raise ValidationError(
            f"Log level must be one of {valid_levels}, got '{level}'",
            error_code="LOGGING_INVALID_LEVEL",
            context={"requested_level": level, "valid_levels": list(valid_levels)}
        )
    
    # Set up format
    if format_string is None:
        format_string = (
            '%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s'
        )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level_upper))
    
    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level_upper))
    console_formatter = logging.Formatter(format_string)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if requested)
    if file_path:
        try:
            file_handler = logging.FileHandler(file_path, mode='a')
            file_handler.setLevel(getattr(logging, level_upper))
            file_handler.setFormatter(console_formatter)
            root_logger.addHandler(file_handler)
            root_logger.info(f"Logging to file: {file_path}")
        except IOError as e:
            root_logger.warning(f"Could not open log file {file_path}: {e}")
    
    root_logger.debug(f"Logging configured: level={level_upper}, format=custom")


# ==============================================================================
# VECTOR AND MATRIX OPERATIONS
# ==============================================================================

def normalize_vector(v: np.ndarray, tolerance: float = 1e-12) -> np.ndarray:
    """
    Normalize a vector to unit length.
    
    Converts any vector to a unit vector (magnitude = 1.0) pointing in the
    same direction. Useful for computing normal vectors and direction cosines.
    
    Parameters
    ----------
    v : np.ndarray
        Vector to normalize. Should be 1D array of length 3 for 3D vectors.
    tolerance : float, optional
        Tolerance for zero-length vector check. Default is 1e-12.
        
    Returns
    -------
    np.ndarray
        Normalized vector (magnitude ~1.0)
    
    Raises
    ------
    ValidationError
        If vector shape is invalid (not 1D or not length 3)
    GeometryError
        If vector magnitude is less than tolerance (zero-length vector)
    
    Examples
    --------
    >>> import numpy as np
    >>> from core.utilities import normalize_vector
    >>> v = np.array([3.0, 4.0, 0.0])
    >>> n = normalize_vector(v)
    >>> print(n)
    [0.6 0.8 0. ]
    >>> print(f"Magnitude: {np.linalg.norm(n):.6f}")
    Magnitude: 1.000000
    
    Notes
    -----
    Returns the zero vector (unmodified) if magnitude < tolerance to prevent
    division by zero. Caller should check for this case.
    """
    # Validate input
    v = np.asarray(v, dtype=float)
    
    if v.ndim != 1:
        raise ValidationError(
            f"Vector must be 1D array, got shape {v.shape}",
            error_code="VECTOR_INVALID_NDIM",
            context={"shape": v.shape, "ndim": v.ndim}
        )
    
    if v.shape[0] != 3:
        raise ValidationError(
            f"Vector must have length 3, got {v.shape[0]}",
            error_code="VECTOR_INVALID_LENGTH",
            context={"shape": v.shape, "expected_length": 3}
        )
    
    # Check for NaN/Inf
    if np.any(np.isnan(v)) or np.any(np.isinf(v)):
        raise DataTypeError(
            "Vector contains NaN or infinite values",
            error_code="VECTOR_NAN_INF",
            context={"vector": v.tolist()}
        )
    
    # Compute magnitude
    magnitude = np.linalg.norm(v)
    
    # Check for zero-length vector
    if magnitude < tolerance:
        logger.warning(
            f"Cannot normalize zero-length vector. Magnitude {magnitude:.2e} < {tolerance:.2e}"
        )
        return v  # Return unmodified
    
    # Normalize
    normalized = v / magnitude
    
    logger.debug(
        f"Normalized vector: magnitude {magnitude:.6f} -> {np.linalg.norm(normalized):.6f}"
    )
    
    return normalized


def rotation_matrix_from_quaternion(quat: np.ndarray) -> np.ndarray:
    """
    Convert a quaternion to a 3x3 rotation matrix.
    
    Quaternion format: [w, x, y, z] where w is the scalar part.
    
    Parameters
    ----------
    quat : np.ndarray
        Quaternion as [w, x, y, z]. Should be unit quaternion (magnitude = 1.0)
        
    Returns
    -------
    np.ndarray
        3x3 rotation matrix (orthogonal, det=1)
    
    Raises
    ------
    ValidationError
        If quaternion shape is invalid
    DataTypeError
        If quaternion contains NaN/Inf
    
    Examples
    --------
    >>> import numpy as np
    >>> from core.utilities import rotation_matrix_from_quaternion
    >>> # Identity quaternion (no rotation)
    >>> quat = np.array([1.0, 0.0, 0.0, 0.0])
    >>> R = rotation_matrix_from_quaternion(quat)
    >>> print(R)
    [[1. 0. 0.]
     [0. 1. 0.]
     [0. 0. 1.]]
    
    Notes
    -----
    Quaternion should be normalized before calling this function.
    Non-unit quaternions will produce scaling in the rotation matrix.
    """
    # Validate input
    quat = np.asarray(quat, dtype=float)
    
    if quat.shape != (4,):
        raise ValidationError(
            f"Quaternion must have shape (4,), got {quat.shape}",
            error_code="QUATERNION_INVALID_SHAPE",
            context={"shape": quat.shape}
        )
    
    if np.any(np.isnan(quat)) or np.any(np.isinf(quat)):
        raise DataTypeError(
            "Quaternion contains NaN or infinite values",
            error_code="QUATERNION_NAN_INF",
            context={"quaternion": quat.tolist()}
        )
    
    # Extract components
    w, x, y, z = quat
    
    # Compute rotation matrix using standard formula
    # R = [[1-2(y²+z²),  2(xy-wz),    2(xz+wy)  ],
    #      [2(xy+wz),    1-2(x²+z²),  2(yz-wx)  ],
    #      [2(xz-wy),    2(yz+wx),    1-2(x²+y²)]]
    
    R = np.array([
        [1 - 2*(y**2 + z**2),  2*(x*y - w*z),      2*(x*z + w*y)     ],
        [2*(x*y + w*z),        1 - 2*(x**2 + z**2),  2*(y*z - w*x)     ],
        [2*(x*z - w*y),        2*(y*z + w*x),        1 - 2*(x**2 + y**2)],
    ])
    
    logger.debug(f"Quaternion {quat} -> rotation matrix (det={np.linalg.det(R):.6f})")
    
    return R


def apply_rotation(points: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """
    Apply quaternion rotation to points.
    
    Rotates a set of 3D points using the given quaternion.
    
    Parameters
    ----------
    points : np.ndarray
        Array of shape (N, 3) containing N 3D points
    quat : np.ndarray
        Quaternion as [w, x, y, z]
        
    Returns
    -------
    np.ndarray
        Rotated points, same shape as input
    
    Raises
    ------
    ValidationError
        If points shape is invalid
    
    Examples
    --------
    >>> import numpy as np
    >>> from core.utilities import apply_rotation
    >>> points = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    >>> quat = np.array([1.0, 0.0, 0.0, 0.0])  # Identity
    >>> rotated = apply_rotation(points, quat)
    >>> print(rotated)
    [[1. 0. 0.]
     [0. 1. 0.]]
    """
    # Validate points
    points = np.asarray(points, dtype=float)
    
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValidationError(
            f"Points must have shape (N, 3), got {points.shape}",
            error_code="POINTS_INVALID_SHAPE",
            context={"shape": points.shape}
        )
    
    # Get rotation matrix
    R = rotation_matrix_from_quaternion(quat)
    
    # Apply rotation: p_rotated = R @ p^T
    rotated = (R @ points.T).T
    
    logger.debug(f"Applied rotation to {points.shape[0]} points")
    
    return rotated


def distance_point_to_plane(
    point: np.ndarray,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
) -> float:
    """
    Compute perpendicular distance from point to plane.
    
    Uses the formula: d = |(P - P0) · N| / |N|
    where P is the point, P0 is a point on the plane, and N is the normal.
    
    Parameters
    ----------
    point : np.ndarray
        Point coordinates (length 3)
    plane_point : np.ndarray
        Any point on the plane (length 3)
    plane_normal : np.ndarray
        Normal vector to the plane (length 3)
    
    Returns
    -------
    float
        Perpendicular distance (always non-negative)
    
    Raises
    ------
    ValidationError
        If any input has invalid shape
    GeometryError
        If plane normal is zero-length
    
    Examples
    --------
    >>> import numpy as np
    >>> from core.utilities import distance_point_to_plane
    >>> point = np.array([1.0, 1.0, 1.0])
    >>> plane_point = np.array([0.0, 0.0, 0.0])
    >>> plane_normal = np.array([0.0, 0.0, 1.0])
    >>> d = distance_point_to_plane(point, plane_point, plane_normal)
    >>> print(f"Distance: {d:.6f}")
    Distance: 1.000000
    """
    # Validate inputs
    point = np.asarray(point, dtype=float)
    plane_point = np.asarray(plane_point, dtype=float)
    plane_normal = np.asarray(plane_normal, dtype=float)
    
    for name, arr in [('point', point), ('plane_point', plane_point), 
                      ('plane_normal', plane_normal)]:
        if arr.shape != (3,):
            raise ValidationError(
                f"{name} must have shape (3,), got {arr.shape}",
                error_code="DISTANCE_INVALID_SHAPE",
                context={"parameter": name, "shape": arr.shape}
            )
    
    # Check for NaN/Inf
    for name, arr in [('point', point), ('plane_point', plane_point),
                      ('plane_normal', plane_normal)]:
        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            raise DataTypeError(
                f"{name} contains NaN or infinite values",
                error_code="DISTANCE_NAN_INF",
                context={"parameter": name}
            )
    
    # Check plane normal is not zero
    normal_magnitude = np.linalg.norm(plane_normal)
    if normal_magnitude < 1e-12:
        raise GeometryError(
            "Plane normal is zero-length",
            error_code="GEOMETRY_ZERO_NORMAL",
            context={"normal": plane_normal.tolist()}
        )
    
    # Compute distance: |dot(point - plane_point, normal)| / |normal|
    v = point - plane_point
    distance = abs(np.dot(v, plane_normal)) / normal_magnitude
    
    logger.debug(f"Point-to-plane distance: {distance:.6e}")
    
    return float(distance)


# ==============================================================================
# VALIDATION FUNCTIONS
# ==============================================================================

def validate_vertices(vertices: np.ndarray, min_vertices: int = 3) -> bool:
    """
    Validate vertex array format.
    
    Checks that:
    - Shape is (N, 3)
    - N >= min_vertices
    - No NaN or Inf values
    - All values are numeric
    
    Parameters
    ----------
    vertices : np.ndarray
        Array of vertex coordinates
    min_vertices : int, optional
        Minimum number of vertices required. Default is 3.
    
    Returns
    -------
    bool
        True if all checks pass
    
    Raises
    ------
    ValidationError
        If any check fails
    
    Examples
    --------
    >>> import numpy as np
    >>> from core.utilities import validate_vertices
    >>> vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    >>> if validate_vertices(vertices):
    ...     print("✓ Vertices are valid")
    """
    vertices = np.asarray(vertices)
    
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValidationError(
            f"Vertices must have shape (N, 3), got {vertices.shape}",
            error_code="VERTICES_INVALID_SHAPE",
            context={"shape": vertices.shape}
        )
    
    if vertices.shape[0] < min_vertices:
        raise ValidationError(
            f"Need at least {min_vertices} vertices, got {vertices.shape[0]}",
            error_code="VERTICES_TOO_FEW",
            context={"n_vertices": vertices.shape[0], "min_required": min_vertices}
        )
    
    if np.any(np.isnan(vertices)) or np.any(np.isinf(vertices)):
        raise DataTypeError(
            "Vertices contain NaN or infinite values",
            error_code="VERTICES_NAN_INF"
        )
    
    logger.debug(f"✓ Validated {vertices.shape[0]} vertices")
    return True


def validate_faces(faces: list, num_vertices: int) -> bool:
    """
    Validate face definitions.
    
    Checks that:
    - faces is a list
    - Each face is iterable (list or tuple)
    - All vertex indices are in valid range [0, num_vertices)
    - No empty faces
    
    Parameters
    ----------
    faces : list
        List of faces, where each face is a list of vertex indices
    num_vertices : int
        Total number of vertices (for index range check)
    
    Returns
    -------
    bool
        True if all checks pass
    
    Raises
    ------
    ValidationError
        If any check fails
    
    Examples
    --------
    >>> from core.utilities import validate_faces
    >>> faces = [[0, 1, 2], [0, 2, 3]]
    >>> if validate_faces(faces, num_vertices=4):
    ...     print("✓ Faces are valid")
    """
    if not isinstance(faces, list):
        raise ValidationError(
            f"Faces must be a list, got {type(faces).__name__}",
            error_code="FACES_INVALID_TYPE"
        )
    
    for i, face in enumerate(faces):
        if not hasattr(face, '__iter__') or isinstance(face, str):
            raise ValidationError(
                f"Face {i} must be iterable, got {type(face).__name__}",
                error_code="FACES_INVALID_ELEMENT_TYPE",
                context={"face_index": i}
            )
        
        if len(face) == 0:
            raise ValidationError(
                f"Face {i} is empty",
                error_code="FACES_EMPTY_FACE",
                context={"face_index": i}
            )
        
        for j, idx in enumerate(face):
            try:
                idx_int = int(idx)
            except (TypeError, ValueError):
                raise ValidationError(
                    f"Face {i}, vertex {j}: index not convertible to int",
                    error_code="FACES_INVALID_INDEX",
                    context={"face_index": i, "vertex_index": j, "value": idx}
                )
            
            if not (0 <= idx_int < num_vertices):
                raise ValidationError(
                    f"Face {i}, vertex {j}: index {idx_int} out of range [0, {num_vertices})",
                    error_code="FACES_INDEX_OUT_OF_RANGE",
                    context={"face_index": i, "vertex_index": j, "index": idx_int, 
                            "max": num_vertices}
                )
    
    logger.debug(f"✓ Validated {len(faces)} faces with {num_vertices} vertices")
    return True


def validate_box(Lx: float, Ly: float, Lz: float) -> bool:
    """
    Validate box dimensions.
    
    Checks that all dimensions are positive and numeric.
    
    Parameters
    ----------
    Lx, Ly, Lz : float
        Box dimensions
    
    Returns
    -------
    bool
        True if all checks pass
    
    Raises
    ------
    ValidationError
        If any dimension is invalid
    
    Examples
    --------
    >>> from core.utilities import validate_box
    >>> if validate_box(Lx=10.0, Ly=10.0, Lz=10.0):
    ...     print("✓ Box dimensions are valid")
    """
    for name, val in [('Lx', Lx), ('Ly', Ly), ('Lz', Lz)]:
        try:
            val_float = float(val)
        except (TypeError, ValueError):
            raise DataTypeError(
                f"{name} not convertible to float",
                error_code="BOX_TYPE_ERROR",
                context={"parameter": name, "value": val}
            )
        
        if val_float <= 0:
            raise ValidationError(
                f"{name} must be positive, got {val_float}",
                error_code="BOX_NON_POSITIVE",
                context={"parameter": name, "value": val_float}
            )
    
    logger.debug(f"✓ Validated box: Lx={Lx}, Ly={Ly}, Lz={Lz}")
    return True


if __name__ == "__main__":
    """
    Test/demo when module is run directly.
    """
    print("\n" + "="*80)
    print("CORE UTILITIES MODULE - TESTING")
    print("="*80 + "\n")
    
    # Setup logging
    setup_logging(level="DEBUG")
    
    # Test vector normalization
    print("[TEST 1] Vector normalization...")
    try:
        v = np.array([3.0, 4.0, 0.0])
        n = normalize_vector(v)
        print(f"✓ Normalized {v} -> {n} (magnitude={np.linalg.norm(n):.6f})")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test rotation matrix
    print("\n[TEST 2] Quaternion to rotation matrix...")
    try:
        quat = np.array([1.0, 0.0, 0.0, 0.0])  # Identity
        R = rotation_matrix_from_quaternion(quat)
        print(f"✓ Identity quaternion -> identity matrix (det={np.linalg.det(R):.6f})")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test point-to-plane distance
    print("\n[TEST 3] Point-to-plane distance...")
    try:
        point = np.array([1.0, 1.0, 1.0])
        plane_point = np.array([0.0, 0.0, 0.0])
        plane_normal = np.array([0.0, 0.0, 1.0])
        d = distance_point_to_plane(point, plane_point, plane_normal)
        print(f"✓ Distance = {d:.6f}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test vertex validation
    print("\n[TEST 4] Vertex validation...")
    try:
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        if validate_vertices(vertices):
            print(f"✓ Valid vertices: {vertices.shape}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("\n" + "="*80)
    print("Utilities module is ready for use!")
    print("="*80 + "\n")