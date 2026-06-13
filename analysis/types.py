# analysis/types.py
# ==============================================================================
# Module: analysis.types
# Purpose: Data structures for analysis results
#
# Defines dataclasses:
#   - RDFData: Radial distribution function data
#   - StatsResult: Statistical analysis result
#
# Author: Contact Analysis Team
# ==============================================================================

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any

import numpy as np

from .. import ValidationError, DataTypeError

logger = logging.getLogger(__name__)


# ==============================================================================
# RDF DATA: Radial distribution function
# ==============================================================================

@dataclass
class RDFData:
    """
    Radial distribution function (RDF) data.
    
    Contains histogram and computed g(r) for particles or contacts.
    
    Attributes
    ----------
    r_values : np.ndarray
        Radial distances (shape: (n_bins,))
    g_r : np.ndarray
        RDF values g(r) at each radius (shape: (n_bins,))
    histogram : np.ndarray
        Bin counts (shape: (n_bins,))
    r_min : float
        Minimum radius
    r_max : float
        Maximum radius
    n_bins : int
        Number of bins
    total_count : int
        Total count across all bins
    frame_count : int
        Number of frames analyzed
    metadata : Dict[str, Any], optional
        Additional data
    
    Raises
    ------
    ValidationError
        If data invalid
    """
    
    r_values: np.ndarray
    g_r: np.ndarray
    histogram: np.ndarray
    r_min: float
    r_max: float
    n_bins: int
    total_count: int
    frame_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate RDF data."""
        # Validate array shapes
        if len(self.r_values) != self.n_bins:
            raise ValidationError(
                f"r_values length {len(self.r_values)} != n_bins {self.n_bins}",
                error_code="RDF_SHAPE_MISMATCH"
            )
        
        if len(self.g_r) != self.n_bins:
            raise ValidationError(
                f"g_r length {len(self.g_r)} != n_bins {self.n_bins}",
                error_code="RDF_SHAPE_MISMATCH"
            )
        
        if len(self.histogram) != self.n_bins:
            raise ValidationError(
                f"histogram length {len(self.histogram)} != n_bins {self.n_bins}",
                error_code="RDF_SHAPE_MISMATCH"
            )
        
        # Validate radius range
        if self.r_min >= self.r_max:
            raise ValidationError(
                f"r_min {self.r_min} >= r_max {self.r_max}",
                error_code="RDF_INVALID_RANGE"
            )
        
        # Validate counts
        if self.total_count < 0:
            raise ValidationError(
                f"total_count must be non-negative, got {self.total_count}",
                error_code="RDF_NEGATIVE_COUNT"
            )
        
        if self.frame_count <= 0:
            raise ValidationError(
                f"frame_count must be positive, got {self.frame_count}",
                error_code="RDF_INVALID_FRAME_COUNT"
            )
        
        # Validate values
        for val in self.g_r:
            if np.isnan(val) or np.isinf(val):
                raise DataTypeError(
                    f"g_r contains NaN or Inf",
                    error_code="RDF_VALUE_NAN_INF"
                )
        
        logger.debug(
            f"RDFData created: r=[{self.r_min:.3f}, {self.r_max:.3f}], "
            f"bins={self.n_bins}, frames={self.frame_count}"
        )
    
    @property
    def bin_width(self) -> float:
        """Width of each bin."""
        return (self.r_max - self.r_min) / self.n_bins
    
    @property
    def average_count_per_bin(self) -> float:
        """Average count per bin."""
        if self.n_bins == 0:
            return 0.0
        return self.total_count / self.n_bins
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"RDFData(r=[{self.r_min:.3f}, {self.r_max:.3f}], "
            f"bins={self.n_bins}, frames={self.frame_count})"
        )


# ==============================================================================
# STATS RESULT: Statistical analysis result
# ==============================================================================

@dataclass
class StatsResult:
    """
    Statistical analysis result.
    
    Contains statistics and aggregated results.
    
    Attributes
    ----------
    metric_name : str
        Name of analyzed metric
    mean : float
        Mean value
    std : float
        Standard deviation
    min : float
        Minimum value
    max : float
        Maximum value
    median : float
        Median value
    values : List[float]
        All values analyzed
    n_samples : int
        Number of samples
    frame_count : int
        Number of frames
    metadata : Dict[str, Any], optional
        Additional data
    
    Raises
    ------
    ValidationError
        If data invalid
    """
    
    metric_name: str
    mean: float
    std: float
    min: float
    max: float
    median: float
    values: List[float]
    n_samples: int
    frame_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate stats result."""
        if not isinstance(self.metric_name, str) or not self.metric_name:
            raise ValidationError(
                f"metric_name must be non-empty string",
                error_code="STATS_INVALID_NAME"
            )
        
        if self.n_samples < 0:
            raise ValidationError(
                f"n_samples must be non-negative",
                error_code="STATS_NEGATIVE_SAMPLES"
            )
        
        if self.frame_count <= 0:
            raise ValidationError(
                f"frame_count must be positive",
                error_code="STATS_INVALID_FRAME_COUNT"
            )
        
        # Check for NaN/Inf
        for val in [self.mean, self.std, self.min, self.max, self.median]:
            if np.isnan(val) or np.isinf(val):
                raise DataTypeError(
                    f"Statistics contain NaN or Inf",
                    error_code="STATS_VALUE_NAN_INF"
                )
        
        logger.debug(
            f"StatsResult created: {self.metric_name}, "
            f"mean={self.mean:.6f}, std={self.std:.6f}"
        )
    
    @property
    def cv(self) -> float:
        """Coefficient of variation (std/mean)."""
        if self.mean == 0:
            return 0.0
        return abs(self.std / self.mean)
    
    @property
    def range(self) -> float:
        """Range (max - min)."""
        return self.max - self.min
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"StatsResult({self.metric_name}, "
            f"mean={self.mean:.6f}±{self.std:.6f})"
        )


if __name__ == "__main__":
    """Test when run directly."""
    print("\n" + "="*80)
    print("ANALYSIS TYPES MODULE - TESTING")
    print("="*80 + "\n")
    
    print("[TEST 1] Creating RDFData...")
    try:
        r_vals = np.linspace(0.5, 5.0, 50)
        g_r_vals = np.random.random(50) + 0.5
        hist_vals = np.random.randint(0, 100, 50)
        
        rdf = RDFData(
            r_values=r_vals,
            g_r=g_r_vals,
            histogram=hist_vals,
            r_min=0.5,
            r_max=5.0,
            n_bins=50,
            total_count=sum(hist_vals),
            frame_count=10
        )
        
        print(f"✓ {rdf}")
        print(f"  Bin width: {rdf.bin_width:.4f}")
        print(f"  Avg count/bin: {rdf.average_count_per_bin:.1f}\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")
    
    print("[TEST 2] Creating StatsResult...")
    try:
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = StatsResult(
            metric_name="test_metric",
            mean=3.0,
            std=1.4,
            min=1.0,
            max=5.0,
            median=3.0,
            values=values,
            n_samples=5,
            frame_count=10
        )
        
        print(f"✓ {result}")
        print(f"  CV: {result.cv:.3f}")
        print(f"  Range: {result.range:.1f}\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")
    
    print("="*80)
    print("✓ All tests passed!")
    print("="*80 + "\n")
