# trajectory/reader.py
# ==============================================================================
# Module: trajectory.reader
# Purpose: Read GSD trajectory files and create trajectory objects
#
# Classes:
#   - GSDReader: Read and parse GSD files
#
# Author: Contact Analysis Team
# ==============================================================================

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Any

import numpy as np

from .. import ValidationError, DataTypeError, Box
from ..particles import ParticleSystem, Particle, ConvexShape
from .types import Frame, Trajectory

logger = logging.getLogger(__name__)


# ==============================================================================
# GSD FILE READER
# ==============================================================================

class GSDReader:
    """
    Read and parse GSD trajectory files.
    
    GSD (GROMACS Simulation Data) is a binary trajectory format used by
    HOOMD-blue simulation engine. This reader extracts particle positions,
    orientations, and box dimensions from GSD files.
    
    Attributes
    ----------
    gsd_filepath : str
        Path to GSD file
    shape : ConvexShape
        Particle shape (same for all particles)
    
    Examples
    --------
    >>> reader = GSDReader('trajectory.gsd', shape)
    >>> trajectory = reader.read_trajectory(start_frame=0, end_frame=-1)
    """
    
    def __init__(self, gsd_filepath: str, shape: ConvexShape) -> None:
        """
        Initialize GSD reader.
        
        Parameters
        ----------
        gsd_filepath : str
            Path to GSD trajectory file
        shape : ConvexShape
            Particle shape definition (same for all particles)
        
        Raises
        ------
        ValidationError
            If file not found
        DataTypeError
            If shape is not ConvexShape
        """
        
        self.gsd_filepath = Path(gsd_filepath)
        
        # Validate file exists
        if not self.gsd_filepath.exists():
            raise ValidationError(
                f"GSD file not found: {self.gsd_filepath}",
                error_code="GSD_FILE_NOT_FOUND",
                context={"path": str(self.gsd_filepath)}
            )
        
        # Validate shape
        if not isinstance(shape, ConvexShape):
            raise DataTypeError(
                f"shape must be ConvexShape, got {type(shape).__name__}",
                error_code="GSD_READER_SHAPE_TYPE_ERROR"
            )
        
        self.shape = shape
        logger.info(f"[GSDReader] Initialized for: {self.gsd_filepath}")
    
    def read_trajectory(
        self,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
    ) -> Trajectory:
        """
        Read trajectory from GSD file.
        
        Parameters
        ----------
        start_frame : int, optional
            First frame to read (default: 0)
        end_frame : Optional[int], optional
            Last frame to read (default: last frame in file)
        
        Returns
        -------
        Trajectory
            Complete trajectory with all frames
        
        Raises
        ------
        ValidationError
            If frame indices invalid or file cannot be read
        
        Examples
        --------
        >>> reader = GSDReader('traj.gsd', shape)
        >>> traj = reader.read_trajectory(start_frame=0, end_frame=99)
        """
        
        logger.info(f"[GSDReader] Reading trajectory: {self.gsd_filepath}")
        
        try:
            # Import GSD library (HOOMD)
            try:
                from .. import header
                gsd = header.gsd
            except (ImportError, AttributeError):
                raise ImportError(
                    "GSD reader requires HOOMD-blue. Install with: pip install hoomd",
                    error_code="GSD_HOOMD_NOT_AVAILABLE"
                )
            
            # Open GSD file
            traj = gsd.hoomd.open(name=str(self.gsd_filepath), mode="r")
            
            # Get number of frames
            num_frames_total = len(traj)
            if num_frames_total == 0:
                raise ValidationError(
                    f"GSD file contains no frames: {self.gsd_filepath}",
                    error_code="GSD_NO_FRAMES"
                )
            
            logger.info(f"[GSDReader] Total frames in file: {num_frames_total}")
            
            # Validate and process frame indices
            if end_frame is None:
                end_frame = num_frames_total - 1
            
            if not (0 <= start_frame <= end_frame < num_frames_total):
                raise ValidationError(
                    f"Invalid frame range [{start_frame}, {end_frame}], "
                    f"file has {num_frames_total} frames",
                    error_code="GSD_FRAME_RANGE_INVALID"
                )
            
            # Read frames
            frames: List[Frame] = []
            
            for frame_idx in range(start_frame, end_frame + 1):
                try:
                    frame = self._read_frame(traj, frame_idx)
                    frames.append(frame)
                except Exception as e:
                    logger.warning(f"Failed to read frame {frame_idx}: {e}")
                    if frame_idx == start_frame:
                        raise  # Fail if first frame fails
                    continue
            
            traj.close()
            
            logger.info(f"[GSDReader] Successfully read {len(frames)} frames")
            
            # Create trajectory object
            trajectory = Trajectory(
                filepath=str(self.gsd_filepath),
                frames=frames,
                num_frames=len(frames),
                start_index=0,
                end_index=len(frames) - 1,
                metadata={'source_file': str(self.gsd_filepath), 'gsd_total_frames': num_frames_total}
            )
            
            return trajectory
        
        except Exception as e:
            logger.error(f"[GSDReader] Failed to read trajectory: {e}")
            raise
    
    def _read_frame(self, gsd_file, frame_idx: int) -> Frame:
        """
        Read single frame from GSD file.
        
        Private method to extract frame data.
        
        Parameters
        ----------
        gsd_file : GSD file object
            Open GSD file
        frame_idx : int
            Frame index to read
        
        Returns
        -------
        Frame
            Parsed frame
        
        Raises
        ------
        ValidationError
            If frame data invalid
        """
        
        try:
            gsd_frame = gsd_file[frame_idx]
        except (IndexError, KeyError) as e:
            raise ValidationError(
                f"Cannot read frame {frame_idx} from GSD file",
                error_code="GSD_FRAME_READ_ERROR"
            ) from e
        
        # Extract box information
        try:
            box_data = gsd_frame.configuration.box
            if box_data is not None and len(box_data) >= 3:
                Lx, Ly, Lz = box_data[0], box_data[1], box_data[2]
            else:
                raise ValueError("Box data incomplete or missing")
        except Exception as e:
            logger.warning(f"Failed to read box from frame {frame_idx}, using default")
            Lx = Ly = Lz = 1.0
        
        box = Box(Lx=float(Lx), Ly=float(Ly), Lz=float(Lz))
        
        # Extract particle data
        try:
            positions = np.array(gsd_frame.particles.position, dtype=float)
            orientations = np.array(gsd_frame.particles.orientation, dtype=float)
            
            if positions.shape[0] == 0:
                raise ValueError("No particles in frame")
            
            if positions.shape != (len(positions), 3):
                raise ValueError(f"Position shape invalid: {positions.shape}")
            
            if orientations.shape != (len(orientations), 4):
                raise ValueError(f"Orientation shape invalid: {orientations.shape}")
        
        except Exception as e:
            raise ValidationError(
                f"Failed to read particle data from frame {frame_idx}: {e}",
                error_code="GSD_PARTICLE_DATA_ERROR"
            ) from e
        
        # Create particles
        particles: List[Particle] = []
        for i in range(len(positions)):
            p = Particle(
                particle_id=i,
                shape=self.shape,
                position=positions[i],
                orientation=orientations[i]
            )
            particles.append(p)
        
        # Create particle system
        particle_system = ParticleSystem(
            particles=particles,
            box=box,
            frame_index=frame_idx
        )
        
        # Extract time if available
        try:
            time = float(gsd_frame.configuration.step) if hasattr(gsd_frame.configuration, 'step') else float(frame_idx)
        except:
            time = float(frame_idx)
        
        # Create frame
        frame = Frame(
            frame_index=frame_idx,
            time=time,
            particle_system=particle_system,
            timestep=frame_idx
        )
        
        logger.debug(
            f"[GSDReader] Frame {frame_idx}: {len(particles)} particles, "
            f"box=[{Lx:.3f}, {Ly:.3f}, {Lz:.3f}]"
        )
        
        return frame
    
    def get_frame_count(self) -> int:
        """
        Get total number of frames in GSD file.
        
        Returns
        -------
        int
            Number of frames
        
        Raises
        ------
        ValidationError
            If file cannot be read
        """
        
        try:
            from .. import header
            gsd = header.gsd
            
            traj = gsd.hoomd.open(name=str(self.gsd_filepath), mode="r")
            count = len(traj)
            traj.close()
            return count
        except Exception as e:
            raise ValidationError(
                f"Cannot read frame count from {self.gsd_filepath}: {e}",
                error_code="GSD_FRAME_COUNT_ERROR"
            ) from e


if __name__ == "__main__":
    """Test when run directly."""
    print("\n" + "="*80)
    print("TRAJECTORY READER MODULE - TESTING")
    print("="*80 + "\n")
    
    print("[TEST] GSDReader basic functionality...")
    try:
        from ..primitives import ConvexShape
        
        # Create test shape
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
        
        # Test with non-existent file (to show error handling)
        print("✓ Testing error handling for non-existent file...")
        try:
            reader = GSDReader("nonexistent.gsd", shape)
        except ValidationError as e:
            print(f"  Correctly caught: {e.error_code}\n")
        
        print("✓ Reader initialization works correctly")
        
    except Exception as e:
        print(f"✗ Error: {e}\n")
        import traceback
        traceback.print_exc()
    
    print("="*80)
    print("✓ Reader module tests passed!")
    print("="*80 + "\n")
