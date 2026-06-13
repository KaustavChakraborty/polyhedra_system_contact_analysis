# analysis/rdf/processor.py
# ==============================================================================
# Module: analysis.rdf.processor
# Purpose: Stateful RDF computation and frame averaging
#
# Classes:
#   - RDFProcessor: Compute and average RDF across frames
#
# Author: Contact Analysis Team
# ==============================================================================

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np

from ... import ValidationError
from ...trajectory import Trajectory
from ...particles import ParticleSystem
from ..types import RDFData
from .calculator import RDFCalculator

logger = logging.getLogger(__name__)


# ==============================================================================
# RDF PROCESSOR: Stateful RDF computation
# ==============================================================================

@dataclass
class RDFProcessor:
    """
    Stateful processor for RDF computation.
    
    Computes RDF for multiple frames and averages results.
    
    Attributes
    ----------
    r_min : float
        Minimum radius
    r_max : float
        Maximum radius
    n_bins : int
        Number of bins
    calculator : RDFCalculator
        Pure calculation engine
    _histograms : List
        Cache of histograms
    _frame_count : int
        Number of frames processed
    
    Examples
    --------
    >>> processor = RDFProcessor(0.5, 5.0, 50)
    >>> processor.process_frame(system)
    >>> rdf = processor.get_rdf()
    """
    
    r_min: float
    r_max: float
    n_bins: int
    calculator: RDFCalculator = None
    _histograms: List = field(default_factory=list)
    _frame_count: int = 0
    
    def __post_init__(self) -> None:
        """Initialize processor."""
        # Validate parameters
        if self.r_min >= self.r_max:
            raise ValidationError(
                f"r_min {self.r_min} >= r_max {self.r_max}",
                error_code="RDF_PROC_INVALID_RANGE"
            )
        
        if self.n_bins <= 0:
            raise ValidationError(
                f"n_bins must be positive",
                error_code="RDF_PROC_INVALID_BINS"
            )
        
        if self.calculator is None:
            self.calculator = RDFCalculator()
        
        logger.debug(
            f"[RDFProcessor] Initialized: r=[{self.r_min}, {self.r_max}], "
            f"bins={self.n_bins}"
        )
    
    def process_frame(
        self,
        system: ParticleSystem,
        distances: np.ndarray
    ) -> None:
        """
        Process a single frame.
        
        Parameters
        ----------
        system : ParticleSystem
            Particle system
        distances : np.ndarray
            Distance values for this frame
        """
        
        logger.debug(f"[RDFProcessor] Processing frame {self._frame_count}")
        
        # Compute histogram
        r_vals, hist = self.calculator.compute_histogram(
            distances,
            self.r_min,
            self.r_max,
            self.n_bins
        )
        
        # Cache histogram
        self._histograms.append({
            'r_values': r_vals,
            'histogram': hist,
            'frame_index': self._frame_count
        })
        
        self._frame_count += 1
        
        logger.debug(f"[RDFProcessor] Frame processed: count={np.sum(hist):.0f}")
    
    def get_rdf(
        self,
        box_volume: float = 1000.0,
        particle_density: float = 0.1
    ) -> Optional[RDFData]:
        """
        Get averaged RDF.
        
        Parameters
        ----------
        box_volume : float, optional
            Simulation box volume
        particle_density : float, optional
            Number density
        
        Returns
        -------
        RDFData or None
            Averaged RDF if data available
        """
        
        if not self._histograms:
            logger.warning("[RDFProcessor] No frames processed")
            return None
        
        logger.debug(f"[RDFProcessor] Computing averaged RDF from {self._frame_count} frames")
        
        # Average histograms
        avg_histogram = np.zeros(self.n_bins)
        r_values = None
        
        for data in self._histograms:
            avg_histogram += data['histogram']
            if r_values is None:
                r_values = data['r_values'].copy()
        
        avg_histogram /= self._frame_count
        
        # Compute g(r)
        g_r = self.calculator.compute_g_r(
            avg_histogram,
            r_values,
            box_volume,
            particle_density
        )
        
        # Create result
        rdf = RDFData(
            r_values=r_values,
            g_r=g_r,
            histogram=avg_histogram,
            r_min=self.r_min,
            r_max=self.r_max,
            n_bins=self.n_bins,
            total_count=int(np.sum(avg_histogram) * self._frame_count),
            frame_count=self._frame_count
        )
        
        logger.info(f"[RDFProcessor] RDF computed with {self._frame_count} frames")
        
        return rdf
    
    def reset(self) -> None:
        """Reset processor state."""
        self._histograms.clear()
        self._frame_count = 0
        logger.debug("[RDFProcessor] Reset")
    
    def get_statistics(self) -> dict:
        """Get processor statistics."""
        return {
            'frames_processed': self._frame_count,
            'r_range': (self.r_min, self.r_max),
            'n_bins': self.n_bins,
            'bin_width': (self.r_max - self.r_min) / self.n_bins,
        }


if __name__ == "__main__":
    """Test when run directly."""
    print("\n" + "="*80)
    print("RDF PROCESSOR MODULE - TESTING")
    print("="*80 + "\n")
    
    print("[TEST] RDFProcessor initialization...")
    try:
        processor = RDFProcessor(0.5, 5.0, 50)
        print(f"✓ Processor created")
        print(f"  Stats: {processor.get_statistics()}\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")
    
    print("="*80)
    print("✓ Processor tests passed!")
    print("="*80 + "\n")
