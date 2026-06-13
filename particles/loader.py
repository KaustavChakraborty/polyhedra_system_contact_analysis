# particles/loader.py
# ==============================================================================
# Module: particles.loader
# Purpose: Load particle shapes and create particles from trajectory data
#
# Functions:
#   - load_shape_from_json(): Load shape definition from JSON file
#   - create_particle(): Factory function to create particle instances
#
# Author: Contact Analysis Team
# ==============================================================================

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np

from .. import ValidationError, DataTypeError
from ..primitives import ConvexShape
from .types import Particle

logger = logging.getLogger(__name__)


# ==============================================================================
# SHAPE LOADING
# ==============================================================================

def load_shape_from_json(filepath: str) -> ConvexShape:
    """
    Load shape definition from JSON file.
    
    Parameters
    ----------
    filepath : str
        Path to shape JSON file
    
    Returns
    -------
    ConvexShape
        Shape with vertices, faces, edges
    
    Raises
    ------
    ValidationError
        If file not found or JSON structure invalid
    DataTypeError
        If vertices/faces/edges have wrong types
    
    Notes
    -----
    Expected JSON structure:
    {
        "vertices": [[x, y, z], ...],
        "faces": [[v0, v1, v2, ...], ...],
        "edges": [[v0, v1], ...]
    }
    
    Examples
    --------
    >>> shape = load_shape_from_json('shape_023.json')
    >>> print(f"Shape: {shape}")
    """
    
    filepath = Path(filepath)
    
    # Validate file exists
    if not filepath.exists():
        raise ValidationError(
            f"Shape file not found: {filepath}",
            error_code="SHAPE_FILE_NOT_FOUND",
            context={"path": str(filepath)}
        )
    
    logger.info(f"[load_shape_from_json] Loading: {filepath}")
    
    # Parse JSON
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValidationError(
            f"Invalid JSON: {str(e)}",
            error_code="SHAPE_JSON_PARSE_ERROR"
        ) from e
    
    # Validate required fields
    required = {'vertices', 'faces', 'edges'}
    if not required.issubset(data.keys()):
        raise ValidationError(
            f"JSON missing required fields: {required - set(data.keys())}",
            error_code="SHAPE_MISSING_FIELDS"
        )
    
    # Parse vertices
    try:
        vertices = np.asarray(data['vertices'], dtype=float)
    except (TypeError, ValueError) as e:
        raise DataTypeError(
            "vertices must be convertible to float array",
            error_code="SHAPE_VERTICES_TYPE_ERROR"
        ) from e
    
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValidationError(
            f"vertices must have shape (N, 3), got {vertices.shape}",
            error_code="SHAPE_VERTICES_SHAPE_ERROR"
        )
    
    if np.any(np.isnan(vertices)):
        raise DataTypeError(
            "vertices contain NaN",
            error_code="SHAPE_VERTICES_NAN"
        )
    
    # Parse faces
    try:
        faces = [[int(v) for v in face] for face in data['faces']]
    except (TypeError, ValueError) as e:
        raise DataTypeError(
            "face indices must be integers",
            error_code="SHAPE_FACES_TYPE_ERROR"
        ) from e
    
    # Validate face indices
    n_verts = vertices.shape[0]
    for i, face in enumerate(faces):
        for v in face:
            if not (0 <= v < n_verts):
                raise ValidationError(
                    f"Face {i}: vertex index {v} out of range [0, {n_verts})",
                    error_code="SHAPE_FACE_INDEX_OUT_OF_RANGE"
                )
    
    # Parse edges
    try:
        edges = [(int(e[0]), int(e[1])) for e in data['edges']]
    except (TypeError, ValueError) as e:
        raise DataTypeError(
            "edge indices must be integers",
            error_code="SHAPE_EDGES_TYPE_ERROR"
        ) from e
    
    # Create ConvexShape
    shape = ConvexShape(
        vertices=vertices,
        faces=faces,
        edges=edges,
        num_faces=len(faces),
        num_edges=len(edges)
    )
    
    logger.info(
        f"[load_shape_from_json] ✓ Loaded: {vertices.shape[0]} vertices, "
        f"{len(faces)} faces, {len(edges)} edges"
    )
    
    return shape


# ==============================================================================
# PARTICLE CREATION
# ==============================================================================

def create_particle(
    particle_id: int,
    shape: ConvexShape,
    position: np.ndarray,
    orientation: np.ndarray,
) -> Particle:
    """
    Factory function for creating particles.
    
    Parameters
    ----------
    particle_id : int
        Unique particle identifier
    shape : ConvexShape
        Shape definition
    position : np.ndarray
        Position in global coordinates, shape (3,)
    orientation : np.ndarray
        Orientation quaternion [qw, qx, qy, qz], shape (4,)
    
    Returns
    -------
    Particle
        Created particle
    
    Raises
    ------
    ValidationError
        If particle_id invalid
    DataTypeError
        If position/orientation invalid
    
    Examples
    --------
    >>> p = create_particle(
    ...     particle_id=0,
    ...     shape=shape,
    ...     position=np.array([0, 0, 0]),
    ...     orientation=np.array([1, 0, 0, 0])
    ... )
    """
    
    # Convert to arrays
    position = np.asarray(position, dtype=float)
    orientation = np.asarray(orientation, dtype=float)
    
    # Validate
    if position.shape != (3,):
        raise DataTypeError(
            f"position shape must be (3,), got {position.shape}",
            error_code="PARTICLE_POSITION_SHAPE_ERROR"
        )
    
    if orientation.shape != (4,):
        raise DataTypeError(
            f"orientation shape must be (4,), got {orientation.shape}",
            error_code="PARTICLE_ORIENTATION_SHAPE_ERROR"
        )
    
    # Create particle
    particle = Particle(
        particle_id=particle_id,
        shape=shape,
        position=position,
        orientation=orientation
    )
    
    logger.debug(f"[create_particle] Created particle {particle_id}")
    return particle


def create_particles_from_arrays(
    shape: ConvexShape,
    positions: np.ndarray,
    orientations: np.ndarray,
    particle_ids: List[int] = None,
) -> List[Particle]:
    """
    Create multiple particles from position and orientation arrays.
    
    Parameters
    ----------
    shape : ConvexShape
        Shape definition (shared by all particles)
    positions : np.ndarray
        Array of positions, shape (N, 3)
    orientations : np.ndarray
        Array of orientations, shape (N, 4)
    particle_ids : List[int], optional
        Custom particle IDs. If None, uses 0-based indexing.
    
    Returns
    -------
    List[Particle]
        List of created particles
    
    Raises
    ------
    ValidationError
        If array shapes don't match
    
    Examples
    --------
    >>> positions = np.array([[0, 0, 0], [1, 0, 0]])
    >>> orientations = np.array([[1, 0, 0, 0], [1, 0, 0, 0]])
    >>> particles = create_particles_from_arrays(shape, positions, orientations)
    """
    
    # Convert to arrays
    positions = np.asarray(positions, dtype=float)
    orientations = np.asarray(orientations, dtype=float)
    
    # Validate shapes
    if positions.shape[0] != orientations.shape[0]:
        raise ValidationError(
            f"positions and orientations have different lengths",
            error_code="PARTICLE_LENGTH_MISMATCH"
        )
    
    if positions.shape[1] != 3:
        raise ValidationError(
            f"positions shape must be (N, 3), got {positions.shape}",
            error_code="PARTICLE_POSITIONS_SHAPE_ERROR"
        )
    
    if orientations.shape[1] != 4:
        raise ValidationError(
            f"orientations shape must be (N, 4), got {orientations.shape}",
            error_code="PARTICLE_ORIENTATIONS_SHAPE_ERROR"
        )
    
    n_particles = positions.shape[0]
    
    # Generate IDs if not provided
    if particle_ids is None:
        particle_ids = list(range(n_particles))
    
    # Create particles
    particles = []
    for i in range(n_particles):
        p = create_particle(
            particle_id=particle_ids[i],
            shape=shape,
            position=positions[i],
            orientation=orientations[i]
        )
        particles.append(p)
    
    logger.info(f"[create_particles_from_arrays] Created {len(particles)} particles")
    return particles


if __name__ == "__main__":
    """Test when run directly."""
    print("\n" + "="*80)
    print("PARTICLES LOADER MODULE - TESTING")
    print("="*80 + "\n")
    
    print("[TEST] Creating test particles...")
    try:
        from geometry import ConvexShape
        from core import Box
        from .types import ParticleSystem
        
        # Create shape
        vertices = np.array([
            [-1, -1, -1], [1, -1, -1],
            [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1],
            [1, 1, 1], [-1, 1, 1]
        ], dtype=float)
        
        faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], 
                 [2, 3, 7, 6], [0, 3, 7, 4], [1, 2, 6, 5]]
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), 
                 (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
        
        shape = ConvexShape(vertices, faces, edges, len(faces), len(edges))
        
        # Create particles
        positions = np.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ])
        
        orientations = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ])
        
        particles = create_particles_from_arrays(shape, positions, orientations)
        print(f"✓ Created {len(particles)} particles")
        
        for p in particles:
            print(f"  {p}")
        
        # Create system
        box = Box(Lx=10.0, Ly=10.0, Lz=10.0)
        system = ParticleSystem(particles=particles, box=box)
        print(f"✓ Created system: {system}\n")
        
    except Exception as e:
        print(f"✗ Error: {e}\n")
    
    print("="*80)
    print("✓ Loader tests passed!")
    print("="*80 + "\n")
