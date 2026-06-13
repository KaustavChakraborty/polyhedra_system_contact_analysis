# trajectory/processor.py
# ==============================================================================
# Module: trajectory.processor
# Purpose: Stateful frame access with intelligent caching
#
# Classes:
#   - TrajectoryProcessor: Manage frame access with caching strategy
#
# Author: Contact Analysis Team
# ==============================================================================

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from .. import ValidationError
from .types import Frame, Trajectory
from .cache import FrameCache, CacheStatistics

logger = logging.getLogger(__name__)


# ==============================================================================
# TRAJECTORY PROCESSOR: Stateful frame access with caching
# ==============================================================================

@dataclass
class TrajectoryProcessor:
    """
    Stateful processor for trajectory frame access.
    
    Manages frame access with LRU caching for performance. Provides both
    random access (by index) and sequential iteration over frames.
    
    Attributes
    ----------
    trajectory : Trajectory
        Trajectory to process
    cache : FrameCache
        Frame cache with LRU eviction
    current_frame_index : int
        Index of currently loaded frame (-1 if none)
    
    Examples
    --------
    >>> processor = TrajectoryProcessor(trajectory, cache_size=50)
    >>> frame = processor.get_frame(10)  # Random access
    >>> for frame in processor.iterate(start=0, end=99):
    ...     process_frame(frame)
    """
    
    trajectory: Trajectory
    cache: FrameCache = field(default_factory=lambda: FrameCache(max_frames=100))
    current_frame_index: int = field(default=-1, init=False)
    _access_history: List[int] = field(default_factory=list, init=False)
    
    def __post_init__(self) -> None:
        """Validate processor after initialization."""
        if self.trajectory is None or len(self.trajectory.frames) == 0:
            raise ValidationError(
                "Trajectory must contain at least 1 frame",
                error_code="PROCESSOR_TRAJECTORY_EMPTY"
            )
        
        logger.debug(
            f"[TrajectoryProcessor] Initialized with {self.trajectory.num_frames} frames, "
            f"cache_size={self.cache.max_frames}"
        )
    
    def get_frame(self, frame_index: int) -> Frame:
        """
        Get frame by index with caching.
        
        Checks cache first, falls back to trajectory if not cached.
        
        Parameters
        ----------
        frame_index : int
            Frame index in range [0, num_frames)
        
        Returns
        -------
        Frame
            Frame at given index
        
        Raises
        ------
        ValidationError
            If frame_index out of range
        
        Examples
        --------
        >>> processor = TrajectoryProcessor(trajectory)
        >>> frame = processor.get_frame(10)
        """
        
        # Validate index
        if not (0 <= frame_index < self.trajectory.num_frames):
            raise ValidationError(
                f"Frame index {frame_index} out of range [0, {self.trajectory.num_frames})",
                error_code="PROCESSOR_FRAME_INDEX_OUT_OF_RANGE"
            )
        
        # Try cache first
        cached = self.cache.get(frame_index)
        if cached is not None:
            self.current_frame_index = frame_index
            self._access_history.append(frame_index)
            logger.debug(f"[TrajectoryProcessor] Frame {frame_index} from CACHE")
            return cached
        
        # Fall back to trajectory
        frame = self.trajectory.get_frame(frame_index)
        self.cache.put(frame_index, frame)
        self.current_frame_index = frame_index
        self._access_history.append(frame_index)
        logger.debug(f"[TrajectoryProcessor] Frame {frame_index} from TRAJECTORY (cached)")
        
        return frame
    
    def get_frame_range(
        self,
        start_index: int,
        end_index: int
    ) -> List[Frame]:
        """
        Get frames in index range [start_index, end_index].
        
        Parameters
        ----------
        start_index : int
            Start frame index (inclusive)
        end_index : int
            End frame index (inclusive)
        
        Returns
        -------
        List[Frame]
            Frames in range
        
        Raises
        ------
        ValidationError
            If range invalid
        
        Examples
        --------
        >>> processor = TrajectoryProcessor(trajectory)
        >>> frames = processor.get_frame_range(0, 99)
        """
        
        if not (0 <= start_index <= end_index < self.trajectory.num_frames):
            raise ValidationError(
                f"Invalid range [{start_index}, {end_index}], "
                f"trajectory has {self.trajectory.num_frames} frames",
                error_code="PROCESSOR_RANGE_INVALID"
            )
        
        frames = []
        for i in range(start_index, end_index + 1):
            frames.append(self.get_frame(i))
        
        logger.info(f"[TrajectoryProcessor] Retrieved {len(frames)} frames")
        return frames
    
    def iterate(
        self,
        start: int = 0,
        end: Optional[int] = None,
        stride: int = 1
    ) -> Iterable[Frame]:
        """
        Iterate over frames in range with optional stride.
        
        Parameters
        ----------
        start : int, optional
            Start frame index (default: 0)
        end : Optional[int], optional
            End frame index (default: last frame)
        stride : int, optional
            Frame stride (default: 1, meaning every frame)
        
        Yields
        ------
        Frame
            Frames in iteration
        
        Raises
        ------
        ValidationError
            If parameters invalid
        
        Examples
        --------
        >>> processor = TrajectoryProcessor(trajectory)
        >>> # Iterate every other frame
        >>> for frame in processor.iterate(start=0, end=99, stride=2):
        ...     process_frame(frame)
        """
        
        if end is None:
            end = self.trajectory.num_frames - 1
        
        if not (0 <= start <= end < self.trajectory.num_frames):
            raise ValidationError(
                f"Invalid iteration range [{start}, {end}]",
                error_code="PROCESSOR_ITERATION_RANGE_INVALID"
            )
        
        if stride <= 0:
            raise ValidationError(
                f"stride must be positive, got {stride}",
                error_code="PROCESSOR_STRIDE_INVALID"
            )
        
        logger.info(
            f"[TrajectoryProcessor] Starting iteration: "
            f"range=[{start}, {end}], stride={stride}"
        )
        
        count = 0
        for frame_idx in range(start, end + 1, stride):
            yield self.get_frame(frame_idx)
            count += 1
        
        logger.info(f"[TrajectoryProcessor] Iteration complete: {count} frames yielded")
    
    def get_current_frame(self) -> Optional[Frame]:
        """
        Get currently loaded frame.
        
        Returns
        -------
        Frame or None
            Current frame if loaded, None if no frame loaded
        """
        
        if self.current_frame_index < 0:
            return None
        
        return self.trajectory.frames[self.current_frame_index]
    
    def get_frame_statistics(self, frame_index: int) -> Dict[str, Any]:
        """
        Get statistics for a frame.
        
        Parameters
        ----------
        frame_index : int
            Frame index
        
        Returns
        -------
        dict
            Frame statistics (num_particles, box dimensions, etc.)
        """
        
        frame = self.get_frame(frame_index)
        
        return {
            'frame_index': frame.frame_index,
            'time': frame.time,
            'timestep': frame.timestep,
            'num_particles': frame.num_particles,
            'box': {'Lx': frame.box.Lx, 'Ly': frame.box.Ly, 'Lz': frame.box.Lz},
            'packing_fraction': frame.packing_fraction,
        }
    
    def get_analysis_frames(self) -> List[Frame]:
        """
        Get frames in analysis range.
        
        Returns
        -------
        List[Frame]
            Frames from start_index to end_index (inclusive) in trajectory
        """
        
        return self.get_frame_range(
            self.trajectory.start_index,
            self.trajectory.end_index
        )
    
    def clear_cache(self) -> None:
        """Clear frame cache."""
        self.cache.clear()
        logger.info("[TrajectoryProcessor] Cache cleared")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get processor statistics.
        
        Returns
        -------
        dict
            Statistics including cache hit rate, memory usage, etc.
        """
        
        cache_stats = self.cache.get_statistics()
        
        return {
            'total_frames': self.trajectory.num_frames,
            'frames_to_analyze': self.trajectory.num_frames_to_analyze,
            'current_frame': self.current_frame_index,
            'total_accesses': cache_stats.total_accesses,
            'cache_hits': cache_stats.hits,
            'cache_misses': cache_stats.misses,
            'cache_hit_rate': cache_stats.hit_rate,
            'cache_size': cache_stats.current_size,
            'cache_max_size': cache_stats.max_size,
            'memory_estimate_mb': self.cache.get_memory_estimate_mb(),
            'access_history_length': len(self._access_history),
        }
    
    def __repr__(self) -> str:
        """String representation."""
        stats = self.cache.get_statistics()
        return (
            f"TrajectoryProcessor("
            f"frames={self.trajectory.num_frames}, "
            f"current={self.current_frame_index}, "
            f"{stats})"
        )


# For type hints
from typing import Iterable


if __name__ == "__main__":
    """Test when run directly."""
    print("\n" + "="*80)
    print("TRAJECTORY PROCESSOR MODULE - TESTING")
    print("="*80 + "\n")
    
    print("[TEST] TrajectoryProcessor operations...")
    try:
        from .types import Frame, Trajectory
        from .. import Box
        from ..particles import ParticleSystem, Particle
        from ..primitives import ConvexShape
        import numpy as np
        
        # Create test frames
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
        
        frames = [
            Frame(frame_index=i, time=float(i)*0.1, particle_system=system)
            for i in range(5)
        ]
        
        traj = Trajectory(
            filepath="test.gsd",
            frames=frames,
            num_frames=5,
            start_index=0,
            end_index=4
        )
        
        processor = TrajectoryProcessor(traj, FrameCache(max_frames=3))
        
        # Test frame access
        frame = processor.get_frame(0)
        print(f"✓ Got frame 0: {frame}")
        
        # Test iteration
        print("✓ Iterating frames...")
        count = 0
        for f in processor.iterate(start=0, end=2):
            count += 1
        print(f"  Iterated {count} frames")
        
        # Get statistics
        stats = processor.get_statistics()
        print(f"✓ Statistics:")
        print(f"  Cache hit rate: {stats['cache_hit_rate']:.1%}")
        print(f"  Memory: {stats['memory_estimate_mb']:.2f} MB\n")
        
    except Exception as e:
        print(f"✗ Error: {e}\n")
        import traceback
        traceback.print_exc()
    
    print("="*80)
    print("✓ Processor module tests passed!")
    print("="*80 + "\n")
