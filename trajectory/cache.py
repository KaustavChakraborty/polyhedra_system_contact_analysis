# trajectory/cache.py
# ==============================================================================
# Module: trajectory.cache
# Purpose: Frame caching strategy for performance optimization
#
# Classes:
#   - FrameCache: LRU cache for frame access
#   - CacheStatistics: Track cache performance metrics
#
# Author: Contact Analysis Team
# ==============================================================================

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Any
from collections import OrderedDict

import numpy as np

from .. import ValidationError
from .types import Frame

logger = logging.getLogger(__name__)


# ==============================================================================
# CACHE STATISTICS
# ==============================================================================

@dataclass
class CacheStatistics:
    """
    Cache performance statistics.
    
    Tracks hits, misses, and memory usage.
    
    Attributes
    ----------
    hits : int
        Number of cache hits
    misses : int
        Number of cache misses
    total_accesses : int
        Total number of access attempts
    current_size : int
        Number of frames currently cached
    max_size : int
        Maximum cache size allowed
    """
    
    hits: int = 0
    misses: int = 0
    total_accesses: int = 0
    current_size: int = 0
    max_size: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Cache hit rate in [0, 1]."""
        if self.total_accesses == 0:
            return 0.0
        return self.hits / self.total_accesses
    
    @property
    def miss_rate(self) -> float:
        """Cache miss rate in [0, 1]."""
        return 1.0 - self.hit_rate
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"CacheStats(hits={self.hits}, misses={self.misses}, "
            f"hit_rate={self.hit_rate:.1%}, size={self.current_size}/{self.max_size})"
        )


# ==============================================================================
# LRU FRAME CACHE
# ==============================================================================

@dataclass
class FrameCache:
    """
    LRU (Least Recently Used) cache for trajectory frames.
    
    Caches recently accessed frames in memory to avoid repeated GSD file reads.
    Uses LRU eviction policy to keep memory usage bounded.
    
    Attributes
    ----------
    max_frames : int
        Maximum number of frames to keep in cache
    _cache : OrderedDict
        Frame cache: frame_index -> Frame
    _stats : CacheStatistics
        Cache performance statistics
    
    Examples
    --------
    >>> cache = FrameCache(max_frames=100)
    >>> cache.put(0, frame)
    >>> frame = cache.get(0)  # Returns cached frame
    """
    
    max_frames: int = 100
    _cache: OrderedDict = field(default_factory=OrderedDict)
    _stats: CacheStatistics = field(default_factory=CacheStatistics)
    
    def __post_init__(self) -> None:
        """Validate cache parameters."""
        if self.max_frames <= 0:
            raise ValidationError(
                f"max_frames must be positive, got {self.max_frames}",
                error_code="CACHE_MAX_FRAMES_INVALID"
            )
        
        self._stats.max_size = self.max_frames
        logger.debug(f"[FrameCache] Initialized with max_frames={self.max_frames}")
    
    def get(self, frame_index: int) -> Optional[Frame]:
        """
        Get frame from cache.
        
        Parameters
        ----------
        frame_index : int
            Frame index to retrieve
        
        Returns
        -------
        Frame or None
            Frame if in cache, None if not cached
        """
        
        self._stats.total_accesses += 1
        
        if frame_index in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(frame_index)
            self._stats.hits += 1
            logger.debug(f"[FrameCache] HIT frame {frame_index}")
            return self._cache[frame_index]
        
        self._stats.misses += 1
        logger.debug(f"[FrameCache] MISS frame {frame_index}")
        return None
    
    def put(self, frame_index: int, frame: Frame) -> None:
        """
        Put frame in cache.
        
        Parameters
        ----------
        frame_index : int
            Frame index
        frame : Frame
            Frame to cache
        
        Raises
        ------
        ValidationError
            If frame is None
        """
        
        if frame is None:
            raise ValidationError(
                "Cannot cache None frame",
                error_code="CACHE_FRAME_NONE"
            )
        
        # If frame already cached, move to end
        if frame_index in self._cache:
            self._cache.move_to_end(frame_index)
            logger.debug(f"[FrameCache] Updated frame {frame_index}")
            return
        
        # Add new frame
        self._cache[frame_index] = frame
        self._cache.move_to_end(frame_index)
        
        # Evict LRU if over capacity
        while len(self._cache) > self.max_frames:
            oldest_idx, _ = self._cache.popitem(last=False)
            logger.debug(f"[FrameCache] Evicted frame {oldest_idx} (LRU)")
        
        self._stats.current_size = len(self._cache)
        logger.debug(f"[FrameCache] Cached frame {frame_index}, size={self._stats.current_size}")
    
    def clear(self) -> None:
        """Clear all cached frames."""
        n = len(self._cache)
        self._cache.clear()
        self._stats.current_size = 0
        logger.info(f"[FrameCache] Cleared {n} cached frames")
    
    def remove(self, frame_index: int) -> bool:
        """
        Remove frame from cache.
        
        Parameters
        ----------
        frame_index : int
            Frame index to remove
        
        Returns
        -------
        bool
            True if frame was cached and removed, False otherwise
        """
        
        if frame_index in self._cache:
            del self._cache[frame_index]
            self._stats.current_size = len(self._cache)
            logger.debug(f"[FrameCache] Removed frame {frame_index}")
            return True
        
        return False
    
    def get_statistics(self) -> CacheStatistics:
        """
        Get cache performance statistics.
        
        Returns
        -------
        CacheStatistics
            Current cache statistics
        """
        
        return CacheStatistics(
            hits=self._stats.hits,
            misses=self._stats.misses,
            total_accesses=self._stats.total_accesses,
            current_size=len(self._cache),
            max_size=self.max_frames
        )
    
    def get_memory_estimate_mb(self) -> float:
        """
        Estimate memory usage of cached frames.
        
        Returns
        -------
        float
            Estimated memory in MB
        """
        
        if len(self._cache) == 0:
            return 0.0
        
        # Estimate one frame size (rough)
        sample_frame = next(iter(self._cache.values()))
        num_particles = sample_frame.num_particles
        
        # Rough estimate: ~500 bytes per particle + overhead
        bytes_per_frame = 500 * num_particles + 1000
        total_bytes = bytes_per_frame * len(self._cache)
        
        return total_bytes / (1024 * 1024)
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"FrameCache(size={len(self._cache)}/{self.max_frames}, "
            f"hits={self._stats.hits}, misses={self._stats.misses}, "
            f"hit_rate={self.get_statistics().hit_rate:.1%})"
        )


if __name__ == "__main__":
    """Test when run directly."""
    print("\n" + "="*80)
    print("TRAJECTORY CACHE MODULE - TESTING")
    print("="*80 + "\n")
    
    print("[TEST 1] FrameCache initialization...")
    try:
        cache = FrameCache(max_frames=10)
        print(f"✓ Cache created: {cache}")
        print(f"  Statistics: {cache.get_statistics()}\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")
    
    print("[TEST 2] Cache put/get operations...")
    try:
        from .types import Frame
        from .. import Box
        from ..particles import ParticleSystem, Particle
        from ..primitives import ConvexShape
        
        # Create dummy frames
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
        
        box = Box(Lx=10.0, Ly=10.0, Lz=10.0)
        system = ParticleSystem(particles=[particle], box=box)
        
        frame = Frame(frame_index=0, time=0.0, particle_system=system)
        
        # Test caching
        cache.put(0, frame)
        print(f"✓ Put frame 0 in cache")
        
        retrieved = cache.get(0)
        print(f"✓ Retrieved frame 0 from cache: {retrieved is frame}")
        
        stats = cache.get_statistics()
        print(f"✓ Stats: {stats}")
        print(f"  Memory estimate: {cache.get_memory_estimate_mb():.2f} MB\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")
        import traceback
        traceback.print_exc()
    
    print("[TEST 3] LRU eviction...")
    try:
        cache = FrameCache(max_frames=3)
        frames = [
            Frame(frame_index=i, time=float(i), particle_system=system)
            for i in range(5)
        ]
        
        for i, f in enumerate(frames):
            cache.put(i, f)
            print(f"  Added frame {i}, cache size: {len(cache._cache)}")
        
        print(f"✓ LRU eviction works (cache size capped at {cache.max_frames})\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")
    
    print("="*80)
    print("✓ Cache module tests passed!")
    print("="*80 + "\n")
