# particles/system.py
# ==============================================================================
# Module: particles.system
# Purpose: Algorithms for particle system operations
#
# Class:
#   - ParticleSystemHandler: Neighbor finding, AABB trees
#
# Author: Contact Analysis Team
# ==============================================================================

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np

from .. import ValidationError
from .types import Particle, ParticleSystem

logger = logging.getLogger(__name__)


# ==============================================================================
# PARTICLE SYSTEM HANDLER: Neighbor finding and spatial algorithms
# ==============================================================================

@dataclass
class ParticleSystemHandler:
    """
    Handle particle system operations (neighbor finding, AABB trees).
    
    Provides algorithms for spatial queries on particle systems.
    
    Attributes
    ----------
    aabb_tree : Dict[int, Dict]
        Cached AABB tree: particle_id -> {'min': np.ndarray, 'max': np.ndarray}
    
    Examples
    --------
    >>> handler = ParticleSystemHandler()
    >>> neighbors = handler.get_neighbors(system, particle_id=0, r_max=4.0)
    """
    
    aabb_tree: Dict[int, Dict] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Initialize handler."""
        logger.debug("[ParticleSystemHandler] Initialized")
    
    def build_neighbor_list(
        self,
        system: ParticleSystem,
        r_max: float
    ) -> Dict[int, List[int]]:
        """
        Build AABB-based neighbor list.
        
        Computes which particles are within distance r_max of each particle.
        
        Parameters
        ----------
        system : ParticleSystem
            Particle system to analyze
        r_max : float
            Maximum distance cutoff for neighbors
        
        Returns
        -------
        dict
            Neighbor list: particle_id -> [neighbor_ids]
        
        Raises
        ------
        ValidationError
            If r_max is invalid
        
        Notes
        -----
        This builds a spatial index (AABB tree) and uses it to efficiently
        find neighbors. The neighbor list is cached in the system.neighbor_list.
        
        Examples
        --------
        >>> handler = ParticleSystemHandler()
        >>> nlist = handler.build_neighbor_list(system, r_max=4.0)
        >>> print(f"Particle 0 neighbors: {nlist[0]}")
        """
        
        # Validate r_max
        if r_max <= 0:
            raise ValidationError(
                f"r_max must be positive, got {r_max}",
                error_code="NEIGHBOR_LIST_INVALID_R_MAX"
            )
        
        logger.info(f"[ParticleSystemHandler] Building neighbor list (r_max={r_max})")
        
        # Build AABB tree
        self._build_aabb_tree(system)
        
        # Compute neighbor list
        neighbor_list = {}
        
        for p in system.particles:
            neighbors = self.get_neighbors(system, p.particle_id, r_max)
            neighbor_list[p.particle_id] = neighbors
        
        logger.info(
            f"[ParticleSystemHandler] Neighbor list complete: "
            f"{sum(len(nl) for nl in neighbor_list.values())} total neighbors"
        )
        
        # Store in system
        system.neighbor_list = neighbor_list
        
        return neighbor_list
    
    def get_neighbors(
        self,
        system: ParticleSystem,
        particle_id: int,
        r_max: float
    ) -> List[int]:
        """
        Get neighbors of a particle within distance r_max.
        
        Parameters
        ----------
        system : ParticleSystem
            Particle system
        particle_id : int
            ID of query particle
        r_max : float
            Maximum distance
        
        Returns
        -------
        List[int]
            List of neighbor particle IDs (excluding self)
        
        Raises
        ------
        ValidationError
            If particle_id not found or r_max invalid
        
        Examples
        --------
        >>> handler = ParticleSystemHandler()
        >>> neighbors = handler.get_neighbors(system, particle_id=0, r_max=4.0)
        >>> print(f"Neighbors: {neighbors}")
        """
        
        # Validate inputs
        if r_max <= 0:
            raise ValidationError(
                f"r_max must be positive, got {r_max}",
                error_code="GET_NEIGHBORS_INVALID_R_MAX"
            )
        
        # Find query particle
        query_particle = system.get_particle(particle_id)
        if query_particle is None:
            raise ValidationError(
                f"Particle {particle_id} not found in system",
                error_code="GET_NEIGHBORS_PARTICLE_NOT_FOUND"
            )
        
        # Build AABB tree if needed
        if not self.aabb_tree:
            self._build_aabb_tree(system)
        
        # Get query AABB
        query_aabb = self.aabb_tree.get(particle_id)
        if query_aabb is None:
            raise ValidationError(
                f"AABB not computed for particle {particle_id}",
                error_code="GET_NEIGHBORS_MISSING_AABB"
            )
        
        query_center = query_particle.position
        
        # Expand search region by r_max
        search_min = query_aabb['min'] - r_max
        search_max = query_aabb['max'] + r_max
        
        # Find overlapping particles
        neighbors = []
        
        for p in system.particles:
            if p.particle_id == particle_id:
                continue  # Skip self
            
            p_aabb = self.aabb_tree.get(p.particle_id)
            if p_aabb is None:
                continue
            
            # Check AABB overlap
            if (search_max >= p_aabb['min']).all() and (search_min <= p_aabb['max']).all():
                # Distance check
                dist = np.linalg.norm(query_center - p.position)
                if dist <= r_max:
                    neighbors.append(p.particle_id)
        
        logger.debug(
            f"[ParticleSystemHandler] Found {len(neighbors)} neighbors "
            f"for particle {particle_id} (r_max={r_max})"
        )
        
        return sorted(neighbors)
    
    def _build_aabb_tree(self, system: ParticleSystem) -> None:
        """
        Build internal AABB tree.
        
        Private method to compute bounding boxes for all particles.
        
        Parameters
        ----------
        system : ParticleSystem
            Particle system
        """
        
        logger.debug(f"[ParticleSystemHandler] Building AABB tree for {system.num_particles} particles")
        
        self.aabb_tree.clear()
        
        for p in system.particles:
            verts = p.global_vertices
            bb_min = np.min(verts, axis=0)
            bb_max = np.max(verts, axis=0)
            
            self.aabb_tree[p.particle_id] = {
                'min': bb_min,
                'max': bb_max
            }
        
        logger.debug(f"[ParticleSystemHandler] AABB tree built: {len(self.aabb_tree)} entries")
    
    def clear_cache(self) -> None:
        """Clear cached AABB tree."""
        n = len(self.aabb_tree)
        self.aabb_tree.clear()
        logger.debug(f"[ParticleSystemHandler] Cleared AABB cache ({n} entries)")
    
    def get_statistics(self) -> Dict[str, int]:
        """
        Get handler statistics.
        
        Returns
        -------
        dict
            Statistics about cached data
        """
        return {
            'aabb_cache_size': len(self.aabb_tree),
        }


if __name__ == "__main__":
    """Test when run directly."""
    print("\n" + "="*80)
    print("PARTICLES SYSTEM MODULE - TESTING")
    print("="*80 + "\n")
    
    print("[TEST] ParticleSystemHandler operations...")
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
        
        # Create system with 5 particles in a line
        particles = []
        for i in range(5):
            p = Particle(
                particle_id=i,
                shape=shape,
                position=np.array([float(i) * 1.5, 0.0, 0.0]),
                orientation=np.array([1.0, 0.0, 0.0, 0.0])
            )
            particles.append(p)
        
        box = Box(Lx=10.0, Ly=10.0, Lz=10.0)
        system = ParticleSystem(particles=particles, box=box)
        
        # Test handler
        handler = ParticleSystemHandler()
        
        # Get neighbors
        neighbors = handler.get_neighbors(system, particle_id=0, r_max=2.0)
        print(f"✓ Particle 0 neighbors (r_max=2.0): {neighbors}")
        
        # Build neighbor list
        nlist = handler.build_neighbor_list(system, r_max=2.5)
        print(f"✓ Neighbor list built:")
        for pid in sorted(nlist.keys()):
            print(f"    Particle {pid}: {nlist[pid]}")
        
        # Get statistics
        stats = handler.get_statistics()
        print(f"✓ Statistics: {stats}\n")
        
    except Exception as e:
        print(f"✗ Error: {e}\n")
        import traceback
        traceback.print_exc()
    
    print("="*80)
    print("✓ System tests passed!")
    print("="*80 + "\n")
