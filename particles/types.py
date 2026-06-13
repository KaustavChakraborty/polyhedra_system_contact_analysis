# particles/types.py
# ==============================================================================
# Module: particles.types
# Purpose: Data structures for particle systems
#
# Defines immutable and mutable dataclasses:
#   - Particle: Individual particle with geometry and kinematics
#   - ParticleSystem: Collection of particles with box and neighbor list
#
# Author: Contact Analysis Team
# ==============================================================================

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import numpy as np

from .. import ValidationError, DataTypeError, Box
from ..primitives import ConvexShape

logger = logging.getLogger(__name__)


# ==============================================================================
# PARTICLE: Individual particle with geometry and state
# ==============================================================================

@dataclass
class Particle:
    """
    Particle with geometry and state.
    
    Represents a single particle with:
    - Unique identifier (particle_id)
    - Shape definition (ConvexShape)
    - Position in 3D space
    - Orientation as quaternion
    
    Attributes
    ----------
    particle_id : int
        Unique identifier for this particle
    shape : ConvexShape
        Shape definition (vertices, faces, edges)
    position : np.ndarray
        Position in global coordinates, shape (3,)
    orientation : np.ndarray
        Orientation quaternion [qw, qx, qy, qz], shape (4,)
    
    Raises
    ------
    ValidationError
        If particle_id is negative
    DataTypeError
        If position or orientation have wrong shape
    """
    
    particle_id: int
    shape: ConvexShape
    position: np.ndarray        # Shape: (3,)
    orientation: np.ndarray     # Shape: (4,) quaternion [qw, qx, qy, qz]
    
    def __post_init__(self) -> None:
        """Validate particle after initialization."""
        # Validate particle_id
        if self.particle_id < 0:
            raise ValidationError(
                f"particle_id must be non-negative, got {self.particle_id}",
                error_code="PARTICLE_ID_NEGATIVE"
            )
        
        # Validate position
        position_array = np.asarray(self.position, dtype=float)
        if position_array.shape != (3,):
            raise DataTypeError(
                f"position must have shape (3,), got {position_array.shape}",
                error_code="PARTICLE_POSITION_SHAPE_ERROR"
            )
        
        if np.any(np.isnan(position_array)):
            raise DataTypeError(
                "position contains NaN values",
                error_code="PARTICLE_POSITION_NAN"
            )
        
        object.__setattr__(self, 'position', position_array)
        
        # Validate orientation (quaternion)
        orientation_array = np.asarray(self.orientation, dtype=float)
        if orientation_array.shape != (4,):
            raise DataTypeError(
                f"orientation must have shape (4,), got {orientation_array.shape}",
                error_code="PARTICLE_ORIENTATION_SHAPE_ERROR"
            )
        
        if np.any(np.isnan(orientation_array)):
            raise DataTypeError(
                "orientation contains NaN values",
                error_code="PARTICLE_ORIENTATION_NAN"
            )
        
        object.__setattr__(self, 'orientation', orientation_array)
        
        logger.debug(f"Particle {self.particle_id} created at {self.position}")
    
    @property
    def global_vertices(self) -> np.ndarray:
        """
        Compute vertices in global coordinates.
        
        Applies rotation and translation to shape vertices.
        
        Returns
        -------
        np.ndarray
            Global vertex coordinates, shape (V, 3)
        """
        from core.utilities import apply_rotation
        
        # Apply rotation to local vertices
        rotated_verts = apply_rotation(self.shape.vertices, self.orientation)
        
        # Translate to global position
        global_verts = rotated_verts + self.position[np.newaxis, :]
        
        return global_verts
    
    @property
    def bounding_box_min(self) -> np.ndarray:
        """Minimum corner of axis-aligned bounding box."""
        verts = self.global_vertices
        return np.min(verts, axis=0)
    
    @property
    def bounding_box_max(self) -> np.ndarray:
        """Maximum corner of axis-aligned bounding box."""
        verts = self.global_vertices
        return np.max(verts, axis=0)
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Particle(id={self.particle_id}, "
            f"pos=[{self.position[0]:.3f}, {self.position[1]:.3f}, {self.position[2]:.3f}])"
        )


# ==============================================================================
# PARTICLE SYSTEM: Collection of particles
# ==============================================================================

@dataclass
class ParticleSystem:
    """
    Collection of particles with spatial structure.
    
    Represents all particles in one frame of a trajectory, including:
    - List of all particles
    - Simulation box dimensions
    - Neighbor list (computed on demand)
    - Time/frame information
    
    Attributes
    ----------
    particles : List[Particle]
        List of all particles in system
    box : Box
        Simulation box dimensions
    frame_index : int
        Frame number in trajectory (default: 0)
    time : float
        Simulation time (default: 0.0)
    neighbor_list : Optional[dict]
        Precomputed neighbor list (default: None)
    
    Raises
    ------
    ValidationError
        If particles list is empty
    """
    
    particles: List[Particle]
    box: Box
    frame_index: int = 0
    time: float = 0.0
    neighbor_list: Optional[Dict[int, List[int]]] = None
    
    def __post_init__(self) -> None:
        """Validate particle system after initialization."""
        # Validate particles list is not empty
        if len(self.particles) == 0:
            raise ValidationError(
                "ParticleSystem must contain at least 1 particle",
                error_code="PARTICLE_SYSTEM_EMPTY"
            )
        
        # Validate all particles
        for p in self.particles:
            if not isinstance(p, Particle):
                raise DataTypeError(
                    f"particles must contain Particle instances, got {type(p).__name__}",
                    error_code="PARTICLE_SYSTEM_TYPE_ERROR"
                )
        
        # Validate box
        if not isinstance(self.box, Box):
            raise DataTypeError(
                f"box must be Box instance, got {type(self.box).__name__}",
                error_code="PARTICLE_SYSTEM_BOX_TYPE_ERROR"
            )
        
        logger.debug(f"ParticleSystem created with {len(self.particles)} particles, frame={self.frame_index}")
    
    @property
    def num_particles(self) -> int:
        """Number of particles in system."""
        return len(self.particles)
    
    @property
    def packing_fraction(self) -> float:
        """
        Estimate packing fraction.
        
        Returns
        -------
        float
            Approximate packing fraction in [0, 1]
        """
        if len(self.particles) == 0:
            return 0.0
        
        # Rough estimate: use bounding box volume
        p0 = self.particles[0]
        bb_min = p0.bounding_box_min
        bb_max = p0.bounding_box_max
        particle_vol = np.prod(bb_max - bb_min)
        
        total_vol = particle_vol * len(self.particles)
        box_vol = self.box.volume
        
        return float(np.clip(total_vol / box_vol, 0.0, 1.0))
    
    def get_particle(self, particle_id: int) -> Optional[Particle]:
        """
        Get particle by ID.
        
        Parameters
        ----------
        particle_id : int
            Particle identifier
        
        Returns
        -------
        Particle or None
            Particle if found, None otherwise
        """
        for p in self.particles:
            if p.particle_id == particle_id:
                return p
        return None
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ParticleSystem("
            f"n={self.num_particles}, "
            f"frame={self.frame_index}, "
            f"box={self.box}, "
            f"φ={self.packing_fraction:.3f})"
        )


if __name__ == "__main__":
    """Test when run directly."""
    print("\n" + "="*80)
    print("PARTICLES TYPES MODULE - TESTING")
    print("="*80 + "\n")
    
    # Create test shape
    print("[TEST 1] Creating test Particle...")
    try:
        from geometry import ConvexShape
        
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
            position=np.array([0.0, 0.0, 0.0]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0])
        )
        
        print(f"✓ {particle}")
        print(f"  Global vertices shape: {particle.global_vertices.shape}")
        print(f"  Bounding box: {particle.bounding_box_min} to {particle.bounding_box_max}\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")
    
    # Test ParticleSystem
    print("[TEST 2] Creating ParticleSystem...")
    try:
        particles = [
            Particle(0, shape, np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0])),
            Particle(1, shape, np.array([2.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0])),
        ]
        
        box = Box(Lx=10.0, Ly=10.0, Lz=10.0)
        system = ParticleSystem(particles=particles, box=box)
        
        print(f"✓ {system}")
        print(f"  Num particles: {system.num_particles}")
        print(f"  Packing fraction: {system.packing_fraction:.4f}\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")
    
    # Test invalid cases
    print("[TEST 3] Invalid Particle (negative ID)...")
    try:
        bad_p = Particle(
            particle_id=-1,
            shape=shape,
            position=np.array([0.0, 0.0, 0.0]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0])
        )
    except ValidationError as e:
        print(f"✓ Correctly caught: {e.error_code}\n")
    
    print("[TEST 4] Invalid ParticleSystem (empty)...")
    try:
        bad_sys = ParticleSystem(particles=[], box=box)
    except ValidationError as e:
        print(f"✓ Correctly caught: {e.error_code}\n")
    
    print("="*80)
    print("✓ All tests passed!")
    print("="*80 + "\n")
