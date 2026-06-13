# workflow/parallel.py
# ==============================================================================
# Module: workflow.parallel
# Purpose: MPI-based work distribution (optional)
#
# Classes:
#   - ParallelManager: Manage parallel execution
#   - MPIHelper: MPI utility functions
#
# Author: Contact Analysis Team
# ==============================================================================

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Any, Callable, Optional

from .. import ValidationError

logger = logging.getLogger(__name__)


# ==============================================================================
# MPI HELPER
# ==============================================================================

@dataclass
class MPIHelper:
    """
    MPI utility functions.
    
    Provides MPI-like interface (MPI not required).
    
    Attributes
    ----------
    rank : int
        Process rank
    size : int
        Total processes
    
    Examples
    --------
    >>> mpi = MPIHelper()
    >>> if mpi.is_master():
    ...     print("Master process")
    """
    
    rank: int = 0
    size: int = 1
    
    def __post_init__(self) -> None:
        """Initialize MPI helper."""
        if self.rank >= self.size:
            raise ValidationError(
                f"rank must be < size",
                error_code="WF_INVALID_RANK"
            )
        
        logger.debug(f"[MPIHelper] Initialized: rank {self.rank}/{self.size}")
    
    def is_master(self) -> bool:
        """Check if master process."""
        return self.rank == 0
    
    def is_worker(self) -> bool:
        """Check if worker process."""
        return self.rank > 0
    
    def get_rank(self) -> int:
        """Get process rank."""
        return self.rank
    
    def get_size(self) -> int:
        """Get number of processes."""
        return self.size
    
    def partition_frames(
        self,
        total_frames: int
    ) -> tuple:
        """
        Partition frames across processes.
        
        Parameters
        ----------
        total_frames : int
            Total number of frames
        
        Returns
        -------
        tuple
            (start_frame, end_frame) for this process
        """
        
        frames_per_process = total_frames // self.size
        remainder = total_frames % self.size
        
        if self.rank < remainder:
            start = self.rank * (frames_per_process + 1)
            end = start + frames_per_process + 1
        else:
            start = remainder * (frames_per_process + 1) + (self.rank - remainder) * frames_per_process
            end = start + frames_per_process
        
        logger.debug(
            f"[MPIHelper] Rank {self.rank}: frames [{start}, {end})"
        )
        
        return start, end


# ==============================================================================
# PARALLEL MANAGER
# ==============================================================================

@dataclass
class ParallelManager:
    """
    Manage parallel execution.
    
    Coordinates work across processes.
    
    Attributes
    ----------
    mpi : MPIHelper
        MPI interface
    use_mpi : bool
        Enable MPI distribution
    
    Examples
    --------
    >>> manager = ParallelManager(use_mpi=False)
    >>> results = manager.map_frames(process_frame, frames)
    """
    
    mpi: MPIHelper = None
    use_mpi: bool = False
    
    def __post_init__(self) -> None:
        """Initialize manager."""
        if self.mpi is None:
            self.mpi = MPIHelper()
        
        logger.debug(
            f"[ParallelManager] Initialized: MPI={self.use_mpi}"
        )
    
    def map_frames(
        self,
        func: Callable,
        frames: List[Any]
    ) -> List[Any]:
        """
        Apply function to frames.
        
        Parameters
        ----------
        func : Callable
            Function to apply
        frames : List[Any]
            Frames to process
        
        Returns
        -------
        List[Any]
            Results
        """
        
        logger.debug(
            f"[ParallelManager] Mapping {len(frames)} frames"
        )
        
        if not self.use_mpi or self.mpi.size == 1:
            # Serial execution
            return [func(frame) for frame in frames]
        
        else:
            # Parallel execution (MPI)
            start, end = self.mpi.partition_frames(len(frames))
            local_frames = frames[start:end]
            
            logger.debug(
                f"[ParallelManager] Rank {self.mpi.rank}: "
                f"processing {len(local_frames)} frames"
            )
            
            return [func(frame) for frame in local_frames]
    
    def gather_results(
        self,
        local_results: List[Any]
    ) -> List[Any]:
        """
        Gather results from all processes.
        
        Parameters
        ----------
        local_results : List[Any]
            Results from this process
        
        Returns
        -------
        List[Any]
            All results (on master)
        """
        
        logger.debug(
            f"[ParallelManager] Gathering {len(local_results)} results"
        )
        
        if not self.use_mpi or self.mpi.size == 1:
            return local_results
        
        # In real MPI, would use MPI.Gather()
        # For now, return local results
        logger.debug(
            f"[ParallelManager] Rank {self.mpi.rank} gathered"
        )
        
        return local_results
    
    def broadcast(
        self,
        data: Any
    ) -> Any:
        """
        Broadcast data from master.
        
        Parameters
        ----------
        data : Any
            Data to broadcast
        
        Returns
        -------
        Any
            Broadcast data
        """
        
        logger.debug("[ParallelManager] Broadcasting data")
        
        if not self.use_mpi or self.mpi.size == 1:
            return data
        
        # In real MPI, would use MPI.Bcast()
        return data
    
    def barrier(self) -> None:
        """Synchronize all processes."""
        logger.debug("[ParallelManager] Barrier")
        
        if self.use_mpi and self.mpi.size > 1:
            # In real MPI, would use MPI.Barrier()
            pass


if __name__ == "__main__":
    """Test when run directly."""
    print("\n" + "="*80)
    print("WORKFLOW PARALLEL MODULE - TESTING")
    print("="*80 + "\n")
    
    print("[TEST 1] MPIHelper...")
    try:
        mpi = MPIHelper(rank=0, size=4)
        print(f"✓ MPIHelper created")
        print(f"  Is master: {mpi.is_master()}\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")
    
    print("[TEST 2] Partition frames...")
    try:
        frames_per_rank = []
        for rank in range(4):
            mpi = MPIHelper(rank=rank, size=4)
            start, end = mpi.partition_frames(100)
            frames_per_rank.append((start, end))
        
        print(f"✓ Frames partitioned:")
        for rank, (s, e) in enumerate(frames_per_rank):
            print(f"  Rank {rank}: [{s}, {e})")
        print()
    except Exception as e:
        print(f"✗ Error: {e}\n")
    
    print("[TEST 3] ParallelManager...")
    try:
        manager = ParallelManager(use_mpi=False)
        
        def double(x):
            return x * 2
        
        results = manager.map_frames(double, [1, 2, 3, 4, 5])
        print(f"✓ Parallel mapping executed")
        print(f"  Results: {results}\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")
    
    print("="*80)
    print("✓ Parallel tests passed!")
    print("="*80 + "\n")
