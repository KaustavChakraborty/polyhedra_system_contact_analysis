# analysis/rdf/calculator.py
# ==============================================================================
# Module: analysis.rdf.calculator
# Purpose: Pure RDF calculations - histogram, binning, g(r)
#
# Classes:
#   - RDFCalculator: Pure calculations for RDF
#
# Author: Contact Analysis Team
# ==============================================================================

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from ... import ValidationError, DataTypeError

logger = logging.getLogger(__name__)


# ==============================================================================
# RDF CALCULATOR: Pure RDF computations
# ==============================================================================

@dataclass
class RDFCalculator:
    """
    Pure RDF calculations.
    
    Performs histogramming, binning, and g(r) computation.
    No state maintained between calls.
    
    Attributes
    ----------
    tolerance : float
        Numerical tolerance
    
    Examples
    --------
    >>> calc = RDFCalculator()
    >>> r_vals, hist = calc.compute_histogram(distances, 0.5, 5.0, 50)
    """
    
    tolerance: float = 1e-12
    
    def __post_init__(self) -> None:
        """Validate calculator."""
        if self.tolerance <= 0:
            raise ValidationError(
                f"tolerance must be positive",
                error_code="RDF_CALC_INVALID_TOLERANCE"
            )
        
        logger.debug("[RDFCalculator] Initialized")
    
    def compute_histogram(
        self,
        distances: np.ndarray,
        r_min: float,
        r_max: float,
        n_bins: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute distance histogram.
        
        Parameters
        ----------
        distances : np.ndarray
            Distance values (shape: (n,))
        r_min : float
            Minimum radius
        r_max : float
            Maximum radius
        n_bins : int
            Number of bins
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (r_values, histogram)
        
        Examples
        --------
        >>> r_vals, hist = calc.compute_histogram(distances, 0.5, 5.0, 50)
        """
        
        # Validate inputs
        distances = np.asarray(distances, dtype=float)
        
        if r_min >= r_max:
            raise ValidationError(
                f"r_min {r_min} >= r_max {r_max}",
                error_code="RDF_CALC_INVALID_RANGE"
            )
        
        if n_bins <= 0:
            raise ValidationError(
                f"n_bins must be positive, got {n_bins}",
                error_code="RDF_CALC_INVALID_BINS"
            )
        
        logger.debug(
            f"[RDFCalculator] Computing histogram: "
            f"r=[{r_min:.3f}, {r_max:.3f}], bins={n_bins}"
        )
        
        # Create bins
        r_values = np.linspace(r_min, r_max, n_bins)
        bin_edges = np.linspace(r_min, r_max + self.tolerance, n_bins + 1)
        
        # Compute histogram
        histogram, _ = np.histogram(distances, bins=bin_edges)
        
        logger.debug(f"[RDFCalculator] Histogram computed: total={np.sum(histogram)}")
        
        return r_values, histogram.astype(float)
    
    def compute_g_r(
        self,
        histogram: np.ndarray,
        r_values: np.ndarray,
        box_volume: float,
        particle_density: float
    ) -> np.ndarray:
        """
        Compute radial distribution function g(r).
        
        Parameters
        ----------
        histogram : np.ndarray
            Bin counts (shape: (n_bins,))
        r_values : np.ndarray
            Radial distances (shape: (n_bins,))
        box_volume : float
            Simulation box volume
        particle_density : float
            Number density of particles
        
        Returns
        -------
        np.ndarray
            g(r) values (shape: (n_bins,))
        
        Notes
        -----
        g(r) = (histogram * volume) / (4π * r² * Δr * N * ρ)
        """
        
        if box_volume <= 0:
            raise ValidationError(
                f"box_volume must be positive",
                error_code="RDF_CALC_INVALID_VOLUME"
            )
        
        if particle_density <= 0:
            raise ValidationError(
                f"particle_density must be positive",
                error_code="RDF_CALC_INVALID_DENSITY"
            )
        
        logger.debug("[RDFCalculator] Computing g(r)")
        
        # Bin width
        dr = r_values[1] - r_values[0] if len(r_values) > 1 else 1.0
        
        # Shell volume
        shell_volume = 4 * np.pi * r_values**2 * dr
        
        # Avoid division by zero
        shell_volume = np.maximum(shell_volume, self.tolerance)
        
        # g(r) = (count * volume) / (shell_volume * particle_density)
        g_r = (histogram * box_volume) / (shell_volume * particle_density)
        
        # Handle zero counts
        g_r = np.where(histogram > 0, g_r, 0.0)
        
        logger.debug(f"[RDFCalculator] g(r) computed: max={np.max(g_r):.3f}")
        
        return g_r
    
    def smooth_histogram(
        self,
        histogram: np.ndarray,
        window_size: int = 3
    ) -> np.ndarray:
        """
        Smooth histogram using moving average.
        
        Parameters
        ----------
        histogram : np.ndarray
            Original histogram
        window_size : int, optional
            Size of smoothing window (default: 3)
        
        Returns
        -------
        np.ndarray
            Smoothed histogram
        """
        
        if window_size < 1:
            raise ValidationError(
                f"window_size must be positive",
                error_code="RDF_CALC_INVALID_WINDOW"
            )
        
        if window_size == 1:
            return histogram.copy()
        
        # Moving average
        smoothed = np.convolve(
            histogram,
            np.ones(window_size) / window_size,
            mode='same'
        )
        
        return smoothed
    
    def compute_peak_location(
        self,
        r_values: np.ndarray,
        g_r: np.ndarray
    ) -> Tuple[float, float]:
        """
        Find location and value of RDF peak.
        
        Parameters
        ----------
        r_values : np.ndarray
            Radial values
        g_r : np.ndarray
            RDF values
        
        Returns
        -------
        Tuple[float, float]
            (r_peak, g_peak)
        """
        
        if len(g_r) == 0:
            return 0.0, 0.0
        
        # Find maximum
        idx_max = np.argmax(g_r)
        r_peak = r_values[idx_max]
        g_peak = g_r[idx_max]
        
        return float(r_peak), float(g_peak)


if __name__ == "__main__":
    """Test when run directly."""
    print("\n" + "="*80)
    print("RDF CALCULATOR MODULE - TESTING")
    print("="*80 + "\n")
    
    print("[TEST 1] RDFCalculator histogram...")
    try:
        calc = RDFCalculator()
        
        # Create test distances
        distances = np.random.uniform(0.5, 5.0, 1000)
        
        r_vals, hist = calc.compute_histogram(distances, 0.5, 5.0, 50)
        
        print(f"✓ Histogram computed:")
        print(f"  Bins: {len(r_vals)}")
        print(f"  Total count: {np.sum(hist):.0f}\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")
    
    print("[TEST 2] g(r) computation...")
    try:
        g_r = calc.compute_g_r(hist, r_vals, 1000.0, 0.1)
        
        print(f"✓ g(r) computed:")
        print(f"  Max g(r): {np.max(g_r):.3f}")
        print(f"  Mean g(r): {np.mean(g_r):.3f}\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")
    
    print("="*80)
    print("✓ Calculator tests passed!")
    print("="*80 + "\n")
