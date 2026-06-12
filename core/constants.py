# core/constants.py
# ==============================================================================
# Module: core.constants
# Purpose: Central repository for all physical constants, tolerances, and defaults
#
# This module defines all constant values used throughout the contact analysis
# workflow. By centralizing these values, they can be easily modified for
# different simulation scenarios without changing source code.
#
# Constants are organized into categories:
#   - Geometric tolerances: For numerical comparisons in 3D space
#   - Physical defaults: RDF, neighbor finding, etc.
#   - Computational: Component names, metric registry
#
# Note: All tolerance values use scientific notation for clarity and are 
# empirically tested for numerical stability.
#
# Author: Contact Analysis Team
# ==============================================================================

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


# ==============================================================================
# GEOMETRIC TOLERANCES
# ==============================================================================
# These values control numerical precision in geometric calculations.
# They should be tuned based on the simulation box size and particle dimensions.

TOLERANCE_QUATERNION: float = 2
"""
Tolerance for quaternion inverse calculation.

Type: float
Default: 2
Purpose: Controls numerical precision when computing quaternion inverses for
         particle orientation transformations. Higher values are more lenient.
Usage: Used in orientation-based geometric transformations.
Notes: This is a scale factor, not an absolute tolerance. Adjust if you see
       numerical instability in rotations.
"""

POLYGON_REORDER_TOL: float = 1e-12
"""
Tolerance for polygon vertex reordering algorithm.

Type: float
Default: 1e-12 (0.000000000001)
Purpose: Controls precision when determining if vertices are coplanar and
         in correct cyclic order around the polygon boundary.
Usage: Applied in geometry.processor.reorder_polygon_vertices()
Notes: Very tight tolerance ensures vertices maintain proper ordering for
       area and overlap calculations. Decrease if getting "non-coplanar" errors.
       Do NOT increase above 1e-10 without careful validation.
"""

CONVEX_DECOMP_TOL: float = 1e-12
"""
Tolerance for convex hull/decomposition algorithm.

Type: float
Default: 1e-12
Purpose: Precision threshold for determining if faces are valid in convex
         decomposition. Affects which vertices are included in convex hull.
Usage: Applied in geometry.convex.convex_decomposition()
Notes: Similar to POLYGON_REORDER_TOL. Keep at 1e-12 for numerical stability.
       Very tight to ensure correct topology of the convex shape.
"""

MIN_OVERLAP_AREA: float = 1e-12
"""
Minimum overlap area to consider particles in contact.

Type: float
Default: 1e-12
Purpose: Threshold below which overlaps are ignored as numerical noise.
         Prevents spurious contact detection from floating-point rounding.
Usage: Applied in contacts.detector.detect_overlap()
Notes: Particles with overlap area < this value are NOT considered in contact.
       Increase if seeing false positives, decrease if missing real contacts.
       Should be ~(minimum element size)^2.
"""

# ==============================================================================
# RADIAL DISTRIBUTION FUNCTION (RDF) DEFAULTS
# ==============================================================================
# These control the computation of g(r) and coordination numbers.

RDF_DEFAULT_BINS: int = 200
"""
Default number of bins for RDF histogram.

Type: int
Default: 200
Purpose: Resolution of the radial distribution function histogram.
         More bins = finer resolution but requires larger ensemble.
Usage: analysis.rdf.processor.RDFProcessor(bins=RDF_DEFAULT_BINS)
Notes: Memory usage ∝ bins. 200 is good balance for 1000+ particles.
       Increase to 500 if you have large trajectories and need fine detail.
       Decrease to 100 if computing RDF for small systems.
"""

RDF_DEFAULT_R_MAX: float = 4.0
"""
Default maximum distance for RDF calculation.

Type: float
Default: 4.0
Purpose: Limits RDF computation to distances up to r_max. Farther particles
         contribute to g(r)=1.0 baseline, not detailed structure.
Usage: analysis.rdf.processor.RDFProcessor(r_max=RDF_DEFAULT_R_MAX)
Notes: Should be ~1/2 to 2/3 of the box size to capture full structure
       without including periodic images. User can override interactively.
"""

DEFAULT_NEIGHBOR_R_MAX: float = 4.0
"""
Default cutoff distance for neighbor list construction.

Type: float
Default: 4.0
Purpose: Only particles within this distance are considered neighbors in
         AABBQuery. Reduces computation cost for sparse systems.
Usage: particles.system.ParticleSystemHandler.build_neighbor_list(r_max=...)
Notes: Should match or exceed the interaction cutoff in your simulation.
       Particles beyond this distance are ignored in contact analysis.
"""

# ==============================================================================
# COMPUTATIONAL DEFAULTS
# ==============================================================================
# These define what metrics are computed and how results are organized.

COMPONENTS: List[str] = [
    "C_min",   # Minimum contact order parameter
    "C_avg",   # Average contact order parameter
    "C_max",   # Maximum contact order parameter
    "C_eff",   # Effective contact order parameter
]
"""
List of contact order parameter components.

Type: List[str]
Default: ["C_min", "C_avg", "C_max", "C_eff"]
Purpose: Defines which contact order components are computed for each metric.
Usage: Used in analysis.order_parameters to compute all components.
Notes: DO NOT reorder these. Code assumes this specific order.
       Modify only if adding new order parameter types.
"""

AVAILABLE_METRICS: Dict[str, str] = {}
"""
Registry of available distance metrics.

Type: Dict[str, str]
Default: {} (populated at runtime from metric_definitions.json)
Purpose: Maps metric index to metric name.
         Loaded from metric_definitions.json at workflow startup.
Usage: metrics.base.MetricRegistry uses this for validation.
Notes: This dictionary is populated dynamically, not hardcoded.
       See metric_definitions.json for the authoritative list.
Example:
    AVAILABLE_METRICS = {
        "0": "face_center_face_center",
        "1": "face_center_to_face_perp",
        ...
    }
"""

# ==============================================================================
# VALIDATION AND TESTING UTILITIES
# ==============================================================================

def validate_constants() -> Tuple[bool, List[str]]:
    """
    Validate that all constants have correct types and reasonable values.
    
    This function performs sanity checks on all constant values to catch
    configuration errors early. Useful for debugging if constants are
    accidentally modified.
    
    Returns
    -------
    Tuple[bool, List[str]]
        (is_valid, issues) where:
        - is_valid: True if all validation checks pass
        - issues: List of validation messages (empty if valid)
        
    Raises
    ------
    None (returns results instead of raising)
    
    Examples
    --------
    >>> from core.constants import validate_constants
    >>> valid, issues = validate_constants()
    >>> if not valid:
    ...     for issue in issues:
    ...         print(f"⚠️  {issue}")
    >>> else:
    ...     print("✓ All constants are valid")
    
    Notes
    -----
    This function is called automatically when the core module is imported.
    Can also be called manually for debugging.
    """
    issues: List[str] = []
    
    # Validate numeric tolerances are positive
    if TOLERANCE_QUATERNION <= 0:
        issues.append(f"⚠️  TOLERANCE_QUATERNION ({TOLERANCE_QUATERNION}) must be positive")
    
    if POLYGON_REORDER_TOL <= 0:
        issues.append(f"⚠️  POLYGON_REORDER_TOL ({POLYGON_REORDER_TOL}) must be positive")
    elif POLYGON_REORDER_TOL > 1e-6:
        issues.append(f"⚠️  POLYGON_REORDER_TOL ({POLYGON_REORDER_TOL}) may be too loose for stability")
    
    if CONVEX_DECOMP_TOL <= 0:
        issues.append(f"⚠️  CONVEX_DECOMP_TOL ({CONVEX_DECOMP_TOL}) must be positive")
    elif CONVEX_DECOMP_TOL > 1e-6:
        issues.append(f"⚠️  CONVEX_DECOMP_TOL ({CONVEX_DECOMP_TOL}) may be too loose for stability")
    
    if MIN_OVERLAP_AREA <= 0:
        issues.append(f"⚠️  MIN_OVERLAP_AREA ({MIN_OVERLAP_AREA}) must be positive")
    
    # Validate RDF parameters
    if RDF_DEFAULT_BINS <= 0:
        issues.append(f"⚠️  RDF_DEFAULT_BINS ({RDF_DEFAULT_BINS}) must be positive")
    elif RDF_DEFAULT_BINS < 50:
        issues.append(f"⚠️  RDF_DEFAULT_BINS ({RDF_DEFAULT_BINS}) may be too coarse (<50)")
    elif RDF_DEFAULT_BINS > 1000:
        issues.append(f"⚠️  RDF_DEFAULT_BINS ({RDF_DEFAULT_BINS}) may be too fine (>1000) and slow")
    
    if RDF_DEFAULT_R_MAX <= 0:
        issues.append(f"⚠️  RDF_DEFAULT_R_MAX ({RDF_DEFAULT_R_MAX}) must be positive")
    
    if DEFAULT_NEIGHBOR_R_MAX <= 0:
        issues.append(f"⚠️  DEFAULT_NEIGHBOR_R_MAX ({DEFAULT_NEIGHBOR_R_MAX}) must be positive")
    
    # Validate component names
    if not COMPONENTS:
        issues.append("⚠️  COMPONENTS list is empty!")
    elif len(COMPONENTS) != len(set(COMPONENTS)):
        issues.append(f"⚠️  COMPONENTS contains duplicates: {COMPONENTS}")
    
    # Check for expected components
    expected_components = {"C_min", "C_avg", "C_max", "C_eff"}
    actual_components = set(COMPONENTS)
    if not expected_components.issubset(actual_components):
        missing = expected_components - actual_components
        issues.append(f"⚠️  Missing expected components: {missing}")
    
    is_valid = len(issues) == 0
    
    if is_valid:
        logger.info("✓ All constants validation checks passed")
    else:
        logger.warning(f"Constants validation found {len(issues)} issue(s)")
        for issue in issues:
            logger.warning(issue)
    
    return is_valid, issues


def print_constants_summary() -> None:
    """
    Print a human-readable summary of all constants.
    
    Useful for debugging and understanding the current configuration.
    Also performs validation and reports any issues found.
    
    Returns
    -------
    None
    
    Examples
    --------
    >>> from core.constants import print_constants_summary
    >>> print_constants_summary()
    
    Notes
    -----
    Call this at the start of a run to confirm configuration is correct.
    """
    print("\n" + "="*80)
    print("CORE CONSTANTS CONFIGURATION")
    print("="*80)
    
    print("\n[GEOMETRIC TOLERANCES]")
    print(f"  TOLERANCE_QUATERNION        : {TOLERANCE_QUATERNION}")
    print(f"  POLYGON_REORDER_TOL         : {POLYGON_REORDER_TOL:.2e}")
    print(f"  CONVEX_DECOMP_TOL           : {CONVEX_DECOMP_TOL:.2e}")
    print(f"  MIN_OVERLAP_AREA            : {MIN_OVERLAP_AREA:.2e}")
    
    print("\n[RDF DEFAULTS]")
    print(f"  RDF_DEFAULT_BINS            : {RDF_DEFAULT_BINS}")
    print(f"  RDF_DEFAULT_R_MAX           : {RDF_DEFAULT_R_MAX}")
    print(f"  DEFAULT_NEIGHBOR_R_MAX      : {DEFAULT_NEIGHBOR_R_MAX}")
    
    print("\n[COMPUTATIONAL DEFAULTS]")
    print(f"  COMPONENTS                  : {COMPONENTS}")
    print(f"  AVAILABLE_METRICS           : {len(AVAILABLE_METRICS)} metrics loaded")
    if AVAILABLE_METRICS:
        for idx, name in sorted(AVAILABLE_METRICS.items()):
            print(f"    [{idx}] {name}")
    else:
        print("    (No metrics loaded - will be loaded at runtime from metric_definitions.json)")
    
    # Validate
    print("\n[VALIDATION]")
    is_valid, issues = validate_constants()
    if is_valid:
        print("  ✓ All constants are valid")
    else:
        print(f"  ⚠️  {len(issues)} validation issue(s) found:")
        for issue in issues:
            print(f"    {issue}")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    """
    Test/demo when module is run directly.
    """
    import sys
    
    # Print configuration summary
    print_constants_summary()
    
    # Run validation
    is_valid, issues = validate_constants()
    
    if is_valid:
        print("✓ Constants module is ready for use!")
        sys.exit(0)
    else:
        print(f"✗ Constants validation failed with {len(issues)} issue(s)")
        sys.exit(1)