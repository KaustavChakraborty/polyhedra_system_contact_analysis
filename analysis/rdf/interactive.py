# analysis/rdf/interactive.py
# ==============================================================================
# Module: analysis.rdf.interactive
# Purpose: User interaction for RDF parameter adjustment
#
# Classes:
#   - RDFInteractive: Interactive parameter adjustment
#
# Author: Contact Analysis Team
# ==============================================================================

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from ... import ValidationError

logger = logging.getLogger(__name__)


# ==============================================================================
# RDF INTERACTIVE: User parameter interaction
# ==============================================================================

@dataclass
class RDFInteractive:
    """
    Interactive RDF parameter adjustment.
    
    Allows users to dynamically adjust r_min, r_max, and n_bins.
    
    Attributes
    ----------
    r_min : float
        Minimum radius
    r_max : float
        Maximum radius
    n_bins : int
        Number of bins
    
    Examples
    --------
    >>> interactive = RDFInteractive(0.5, 5.0, 50)
    >>> interactive.set_r_min(0.3)
    >>> interactive.set_n_bins(100)
    """
    
    r_min: float = 0.5
    r_max: float = 5.0
    n_bins: int = 50
    
    def __post_init__(self) -> None:
        """Validate parameters."""
        self._validate_range()
        logger.debug("[RDFInteractive] Initialized")
    
    def _validate_range(self) -> None:
        """Validate r_min and r_max."""
        if self.r_min >= self.r_max:
            raise ValidationError(
                f"r_min {self.r_min} >= r_max {self.r_max}",
                error_code="RDF_INTER_INVALID_RANGE"
            )
    
    def set_r_min(self, r_min: float) -> None:
        """
        Set minimum radius.
        
        Parameters
        ----------
        r_min : float
            New minimum radius
        
        Raises
        ------
        ValidationError
            If r_min invalid
        """
        
        if r_min <= 0:
            raise ValidationError(
                f"r_min must be positive, got {r_min}",
                error_code="RDF_INTER_INVALID_R_MIN"
            )
        
        if r_min >= self.r_max:
            raise ValidationError(
                f"r_min {r_min} >= r_max {self.r_max}",
                error_code="RDF_INTER_INVALID_RANGE"
            )
        
        self.r_min = r_min
        logger.info(f"[RDFInteractive] r_min set to {r_min:.3f}")
    
    def set_r_max(self, r_max: float) -> None:
        """
        Set maximum radius.
        
        Parameters
        ----------
        r_max : float
            New maximum radius
        
        Raises
        ------
        ValidationError
            If r_max invalid
        """
        
        if r_max <= 0:
            raise ValidationError(
                f"r_max must be positive, got {r_max}",
                error_code="RDF_INTER_INVALID_R_MAX"
            )
        
        if self.r_min >= r_max:
            raise ValidationError(
                f"r_min {self.r_min} >= r_max {r_max}",
                error_code="RDF_INTER_INVALID_RANGE"
            )
        
        self.r_max = r_max
        logger.info(f"[RDFInteractive] r_max set to {r_max:.3f}")
    
    def set_n_bins(self, n_bins: int) -> None:
        """
        Set number of bins.
        
        Parameters
        ----------
        n_bins : int
            New number of bins
        
        Raises
        ------
        ValidationError
            If n_bins invalid
        """
        
        if n_bins <= 0:
            raise ValidationError(
                f"n_bins must be positive, got {n_bins}",
                error_code="RDF_INTER_INVALID_BINS"
            )
        
        self.n_bins = n_bins
        logger.info(f"[RDFInteractive] n_bins set to {n_bins}")
    
    def set_range(self, r_min: float, r_max: float) -> None:
        """
        Set radius range.
        
        Parameters
        ----------
        r_min : float
            Minimum radius
        r_max : float
            Maximum radius
        
        Raises
        ------
        ValidationError
            If range invalid
        """
        
        if r_min >= r_max:
            raise ValidationError(
                f"r_min {r_min} >= r_max {r_max}",
                error_code="RDF_INTER_INVALID_RANGE"
            )
        
        self.r_min = r_min
        self.r_max = r_max
        logger.info(f"[RDFInteractive] Range set to [{r_min:.3f}, {r_max:.3f}]")
    
    @property
    def bin_width(self) -> float:
        """Get current bin width."""
        return (self.r_max - self.r_min) / self.n_bins
    
    def get_parameters(self) -> dict:
        """
        Get current parameters.
        
        Returns
        -------
        dict
            Current parameters
        """
        
        return {
            'r_min': self.r_min,
            'r_max': self.r_max,
            'n_bins': self.n_bins,
            'bin_width': self.bin_width,
        }
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"RDFInteractive(r=[{self.r_min:.3f}, {self.r_max:.3f}], "
            f"bins={self.n_bins})"
        )


if __name__ == "__main__":
    """Test when run directly."""
    print("\n" + "="*80)
    print("RDF INTERACTIVE MODULE - TESTING")
    print("="*80 + "\n")
    
    print("[TEST 1] RDFInteractive initialization...")
    try:
        interactive = RDFInteractive()
        print(f"✓ {interactive}")
        print(f"  Parameters: {interactive.get_parameters()}\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")
    
    print("[TEST 2] Adjust parameters...")
    try:
        interactive.set_r_min(0.3)
        interactive.set_r_max(6.0)
        interactive.set_n_bins(100)
        
        print(f"✓ Parameters updated:")
        print(f"  {interactive}")
        print(f"  Bin width: {interactive.bin_width:.4f}\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")
    
    print("[TEST 3] Invalid range...")
    try:
        interactive.set_r_min(10.0)
    except ValidationError as e:
        print(f"✓ Correctly caught: {e.error_code}\n")
    
    print("="*80)
    print("✓ Interactive tests passed!")
    print("="*80 + "\n")
