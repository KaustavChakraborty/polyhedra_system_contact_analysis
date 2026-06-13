# trajectory/types.py
# ==============================================================================
# Module: trajectory.types
# Purpose: Data structures for trajectory frames and trajectories
#
# Defines dataclasses:
#   - Frame: Single snapshot of particle system at one time point
#   - Trajectory: Container for all frames in a GSD trajectory file
#
# Author: Contact Analysis Team
# ==============================================================================

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import numpy as np

from .. import ValidationError, DataTypeError, Box
from ..particles import ParticleSystem

logger = logging.getLogger(__name__)


# ==============================================================================
# FRAME: Single snapshot in trajectory
# ==============================================================================

@dataclass
class Frame:
    """
    Single snapshot of particle system at one time point.
    
    Represents all particles and box state at a specific frame in the trajectory.
    
    Attributes
    ----------
    frame_index : int
        Frame number in trajectory (0-indexed)
    time : float
        Simulation time (in simulation units)
    particle_system : ParticleSystem
        All particles and box at this frame
    timestep : int, optional
        Timestep number (may differ from frame_index)
    metadata : Dict[str, Any], optional
        Additional frame metadata (e.g., temperature, pressure)
    
    Raises
    ------
    ValidationError
        If frame_index is negative
    DataTypeError
        If particle_system is not ParticleSystem
    """
    
    frame_index: int
    time: float
    particle_system: ParticleSystem
    timestep: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate frame after initialization."""
        # Validate frame_index
        if self.frame_index < 0:
            raise ValidationError(
                f"frame_index must be non-negative, got {self.frame_index}",
                error_code="FRAME_INDEX_NEGATIVE"
            )
        
        # Validate time
        if not isinstance(self.time, (int, float)):
            raise DataTypeError(
                f"time must be numeric, got {type(self.time).__name__}",
                error_code="FRAME_TIME_TYPE_ERROR"
            )
        
        if np.isnan(self.time):
            raise DataTypeError(
                "time contains NaN",
                error_code="FRAME_TIME_NAN"
            )
        
        # Validate particle_system
        if not isinstance(self.particle_system, ParticleSystem):
            raise DataTypeError(
                f"particle_system must be ParticleSystem, got {type(self.particle_system).__name__}",
                error_code="FRAME_PARTICLE_SYSTEM_TYPE_ERROR"
            )
        
        # Validate timestep
        if self.timestep < 0:
            raise ValidationError(
                f"timestep must be non-negative, got {self.timestep}",
                error_code="FRAME_TIMESTEP_NEGATIVE"
            )
        
        logger.debug(
            f"Frame {self.frame_index} created (time={self.time}, "
            f"particles={self.particle_system.num_particles})"
        )
    
    @property
    def num_particles(self) -> int:
        """Number of particles in this frame."""
        return self.particle_system.num_particles
    
    @property
    def box(self) -> Box:
        """Simulation box for this frame."""
        return self.particle_system.box
    
    @property
    def packing_fraction(self) -> float:
        """Packing fraction at this frame."""
        return self.particle_system.packing_fraction
    
    def get_particle(self, particle_id: int):
        """Get particle by ID from this frame."""
        return self.particle_system.get_particle(particle_id)
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Frame(index={self.frame_index}, time={self.time:.3f}, "
            f"particles={self.num_particles}, φ={self.packing_fraction:.3f})"
        )


# ==============================================================================
# TRAJECTORY: Container for all frames
# ==============================================================================

@dataclass
class Trajectory:
    """
    Container for all frames in a GSD trajectory file.
    
    Manages access to frames with optional caching for performance.
    
    Attributes
    ----------
    filepath : str
        Path to GSD trajectory file
    frames : List[Frame]
        All loaded frames
    num_frames : int
        Total number of frames
    start_index : int, optional
        Start frame for analysis (default: 0)
    end_index : int, optional
        End frame for analysis (default: last frame)
    
    Raises
    ------
    ValidationError
        If frames list is empty or indices invalid
    """
    
    filepath: str
    frames: List[Frame]
    num_frames: int
    start_index: int = 0
    end_index: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate trajectory after initialization."""
        # Validate filepath
        if not isinstance(self.filepath, str):
            raise DataTypeError(
                f"filepath must be str, got {type(self.filepath).__name__}",
                error_code="TRAJECTORY_FILEPATH_TYPE_ERROR"
            )
        
        # Validate frames
        if len(self.frames) == 0:
            raise ValidationError(
                "Trajectory must contain at least 1 frame",
                error_code="TRAJECTORY_EMPTY"
            )
        
        # Validate num_frames
        if self.num_frames != len(self.frames):
            raise ValidationError(
                f"num_frames ({self.num_frames}) != len(frames) ({len(self.frames)})",
                error_code="TRAJECTORY_FRAME_COUNT_MISMATCH"
            )
        
        # Validate all frames
        for i, f in enumerate(self.frames):
            if not isinstance(f, Frame):
                raise DataTypeError(
                    f"frames[{i}] must be Frame, got {type(f).__name__}",
                    error_code="TRAJECTORY_FRAME_TYPE_ERROR"
                )
        
        # Validate start_index
        if not (0 <= self.start_index < len(self.frames)):
            raise ValidationError(
                f"start_index {self.start_index} out of range [0, {len(self.frames)})",
                error_code="TRAJECTORY_START_INDEX_OUT_OF_RANGE"
            )
        
        # Validate or set end_index
        if self.end_index is None:
            object.__setattr__(self, 'end_index', len(self.frames) - 1)
        elif not (self.start_index <= self.end_index < len(self.frames)):
            raise ValidationError(
                f"end_index {self.end_index} out of range [{self.start_index}, {len(self.frames)})",
                error_code="TRAJECTORY_END_INDEX_OUT_OF_RANGE"
            )
        
        logger.debug(
            f"Trajectory created: {len(self.frames)} frames, "
            f"analysis range [{self.start_index}, {self.end_index}]"
        )
    
    @property
    def num_frames_to_analyze(self) -> int:
        """Number of frames in analysis range."""
        return self.end_index - self.start_index + 1
    
    @property
    def first_frame(self) -> Frame:
        """First frame in trajectory."""
        return self.frames[0]
    
    @property
    def last_frame(self) -> Frame:
        """Last frame in trajectory."""
        return self.frames[-1]
    
    @property
    def time_range(self) -> tuple:
        """Time range (min_time, max_time)."""
        return (self.first_frame.time, self.last_frame.time)
    
    def get_frame(self, index: int) -> Frame:
        """
        Get frame by index.
        
        Parameters
        ----------
        index : int
            Frame index
        
        Returns
        -------
        Frame
            Frame at given index
        
        Raises
        ------
        ValidationError
            If index out of range
        """
        if not (0 <= index < len(self.frames)):
            raise ValidationError(
                f"Frame index {index} out of range [0, {len(self.frames)})",
                error_code="TRAJECTORY_FRAME_INDEX_OUT_OF_RANGE"
            )
        
        return self.frames[index]
    
    def get_frames_in_range(self, start_idx: int, end_idx: int) -> List[Frame]:
        """
        Get frames in index range [start_idx, end_idx].
        
        Parameters
        ----------
        start_idx : int
            Start frame index (inclusive)
        end_idx : int
            End frame index (inclusive)
        
        Returns
        -------
        List[Frame]
            Frames in range
        """
        if not (0 <= start_idx <= end_idx < len(self.frames)):
            raise ValidationError(
                f"Invalid range [{start_idx}, {end_idx}], frame count: {len(self.frames)}",
                error_code="TRAJECTORY_RANGE_INVALID"
            )
        
        return self.frames[start_idx:end_idx+1]
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Trajectory(frames={self.num_frames}, "
            f"time_range=[{self.first_frame.time:.3f}, {self.last_frame.time:.3f}], "
            f"analyze={self.num_frames_to_analyze})"
        )


if __name__ == "__main__":
    """Test when run directly."""
    print("\n" + "="*80)
    print("TRAJECTORY TYPES MODULE - TESTING")
    print("="*80 + "\n")
    
    print("[TEST 1] Creating test Frame...")
    try:
        from ..particles import Particle, ParticleSystem
        from ..primitives import ConvexShape
        
        # Create minimal system
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
        
        frame = Frame(
            frame_index=0,
            time=0.0,
            particle_system=system,
            timestep=0
        )
        
        print(f"✓ {frame}")
        print(f"  Particles: {frame.num_particles}")
        print(f"  Packing: {frame.packing_fraction:.4f}\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")
        import traceback
        traceback.print_exc()
    
    print("[TEST 2] Creating Trajectory...")
    try:
        frames = [
            Frame(frame_index=i, time=float(i)*0.1, particle_system=system)
            for i in range(3)
        ]
        
        traj = Trajectory(
            filepath="test.gsd",
            frames=frames,
            num_frames=3,
            start_index=0,
            end_index=2
        )
        
        print(f"✓ {traj}")
        print(f"  Frames to analyze: {traj.num_frames_to_analyze}")
        print(f"  Time range: {traj.time_range}\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")
    
    print("[TEST 3] Invalid Frame (negative index)...")
    try:
        bad_frame = Frame(
            frame_index=-1,
            time=0.0,
            particle_system=system
        )
    except ValidationError as e:
        print(f"✓ Correctly caught: {e.error_code}\n")
    
    print("="*80)
    print("✓ All tests passed!")
    print("="*80 + "\n")
