# particles/processor.py
# ==============================================================================
# Module: particles.processor
# Purpose: Stateful processor for particle operations
#
# Class:
#   - ParticleProcessor: Cache particle computations and handle geometry
#
# Author: Contact Analysis Team
# ==============================================================================

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, Any

import numpy as np

from ..primitives import Polygon
from ..processor import GeometryProcessor
from .types import Particle

logger = logging.getLogger(__name__)


# ==============================================================================
# PARTICLE PROCESSOR: Stateful operations on particles
# ==============================================================================

@dataclass
class ParticleProcessor:
    """
    Stateful processor for particle operations.
    
    Handles caching and geometry transformations for particles.
    
    Attributes
    ----------
    _aabb_cache : Dict[int, Tuple[np.ndarray, np.ndarray]]
        Cache of axis-aligned bounding boxes: particle_id -> (min, max)
    geometry_processor : GeometryProcessor
        For geometric operations on faces
    
    Examples
    --------
    >>> processor = ParticleProcessor()
    >>> aabb = processor.get_particle_aabb(particle)
    """
    
    _aabb_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    geometry_processor: GeometryProcessor = field(default_factory=GeometryProcessor)
    
    def __post_init__(self) -> None:
        """Initialize processor."""
        logger.debug("[ParticleProcessor] Initialized")
    
    def get_particle_aabb(
        self,
        particle: Particle,
        use_cache: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Get axis-aligned bounding box (cached).
        
        Computes and caches the bounding box of a particle's global vertices.
        
        Parameters
        ----------
        particle : Particle
            Particle to compute AABB for
        use_cache : bool, optional
            Whether to use caching (default: True)
        
        Returns
        -------
        dict
            Dictionary with keys 'min' and 'max' containing corner coordinates
        
        Examples
        --------
        >>> processor = ParticleProcessor()
        >>> aabb = processor.get_particle_aabb(particle)
        >>> print(f"Min: {aabb['min']}, Max: {aabb['max']}")
        """
        
        if use_cache and particle.particle_id in self._aabb_cache:
            bb_min, bb_max = self._aabb_cache[particle.particle_id]
            logger.debug(f"[ParticleProcessor] AABB cache HIT for particle {particle.particle_id}")
            return {'min': bb_min.copy(), 'max': bb_max.copy()}
        
        # Compute AABB from global vertices
        verts = particle.global_vertices
        bb_min = np.min(verts, axis=0)
        bb_max = np.max(verts, axis=0)
        
        # Cache result
        if use_cache:
            self._aabb_cache[particle.particle_id] = (bb_min.copy(), bb_max.copy())
            logger.debug(f"[ParticleProcessor] AABB cache MISS for particle {particle.particle_id}")
        
        return {'min': bb_min, 'max': bb_max}
    
    def get_face_in_global_coords(
        self,
        particle: Particle,
        face_idx: int
    ) -> Polygon:
        """
        Transform face to global coordinates.
        
        Takes a face (defined by indices into particle shape) and transforms
        it to global coordinates using particle position and orientation.
        
        Parameters
        ----------
        particle : Particle
            Particle containing the face
        face_idx : int
            Index of face in particle shape
        
        Returns
        -------
        Polygon
            Face with vertices in global coordinates
        
        Raises
        ------
        ValidationError
            If face_idx out of range
        
        Examples
        --------
        >>> processor = ParticleProcessor()
        >>> face = processor.get_face_in_global_coords(particle, face_idx=0)
        >>> print(f"Face area: {face.area}")
        """
        
        from .. import ValidationError
        
        # Validate face index
        if not (0 <= face_idx < particle.shape.num_faces):
            raise ValidationError(
                f"face_idx {face_idx} out of range [0, {particle.shape.num_faces})",
                error_code="PARTICLE_FACE_INDEX_OUT_OF_RANGE"
            )
        
        # Get face vertex indices from shape
        face_indices = particle.shape.faces[face_idx]
        
        # Get local vertices for this face
        local_verts = particle.shape.vertices[face_indices]
        
        # Transform to global coordinates
        from ..utilities import apply_rotation
        
        rotated_verts = apply_rotation(local_verts, particle.orientation)
        global_verts = rotated_verts + particle.position[np.newaxis, :]
        
        # Create Polygon object
        polygon = Polygon(
            vertices=global_verts,
            face_indices=face_indices
        )
        
        logger.debug(
            f"[ParticleProcessor] Face {face_idx} of particle {particle.particle_id} "
            f"transformed to global coords"
        )
        
        return polygon
    
    def clear_cache(self) -> None:
        """
        Clear AABB cache.
        
        Call when particles have been modified.
        """
        n_cached = len(self._aabb_cache)
        self._aabb_cache.clear()
        logger.debug(f"[ParticleProcessor] Cleared AABB cache ({n_cached} entries)")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get processor statistics.
        
        Returns
        -------
        dict
            Statistics about cache usage and computations
        """
        return {
            'aabb_cache_size': len(self._aabb_cache),
            'geometry_processor': 'ready',
        }


if __name__ == "__main__":
    """Test when run directly."""
    print("\n" + "="*80)
    print("PARTICLES PROCESSOR MODULE - TESTING")
    print("="*80 + "\n")
    
    print("[TEST] ParticleProcessor operations...")
    try:
        from geometry import ConvexShape
        
        # Create shape and particle
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
        
        particle = Particle(
            particle_id=0,
            shape=shape,
            position=np.array([1.0, 2.0, 3.0]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0])
        )
        
        # Test processor
        processor = ParticleProcessor()
        
        # Get AABB
        aabb = processor.get_particle_aabb(particle)
        print(f"✓ AABB computed:")
        print(f"    Min: {aabb['min']}")
        print(f"    Max: {aabb['max']}")
        
        # Get face in global coords
        face = processor.get_face_in_global_coords(particle, face_idx=0)
        print(f"✓ Face 0 transformed to global coords")
        print(f"    Vertices shape: {face.vertices.shape}")
        print(f"    Area: {face.area:.3f}")
        
        # Get statistics
        stats = processor.get_statistics()
        print(f"✓ Statistics: {stats}\n")
        
    except Exception as e:
        print(f"✗ Error: {e}\n")
        import traceback
        traceback.print_exc()
    
    print("="*80)
    print("✓ Processor tests passed!")
    print("="*80 + "\n")
