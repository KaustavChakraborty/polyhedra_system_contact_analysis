"""
Canonical runtime registry for selectable distance metrics.

METRIC REGISTRY OVERVIEW
========================
This module maintains the authoritative mapping between public metric names
and their private implementation methods in ContactDistanceMetrics class.

The dictionary order is critical: the index position of each metric in
METRIC_REGISTRY determines how integer-based metric selection works.

IMPORTANCE OF ORDERING
======================
The registry order MUST be preserved because:

1. Historical Compatibility
   The old workflow converts integer indices to metric names using this exact order:
   
       index 0 → face_center_face_center
       index 1 → face_center_to_face_perp
       index 2 → vertex_vertex
       ... etc
   
   Changing the order would change scientific results.

2. Configuration Files
   metric_definitions.json stores metric indices:
   
       "0": "face_center_face_center",
       "1": "face_center_to_face_perp",
       ... etc
   
   These indices MUST map to the correct metrics via registry order.

3. MPI Communication
   Integer indices are more compact for MPI broadcasts than metric names.
   The registry order ensures all ranks use the same index→name mapping.

REORDERING CONSEQUENCES
=======================
DO NOT REORDER unless you also:

1. Update param_file.json metric indices
2. Update metric_definitions.json indices
3. Recompute all historical analyses with old index mappings
4. Verify the old workflow still produces identical results

METRIC IMPLEMENTATIONS
======================
Each registry entry maps a public name to a private method:

Public name                 => Private method in ContactDistanceMetrics
─────────────────────────────────────────────────────────────────────
face_center_face_center     => _face_center_face_center()
face_center_to_face_perp    => _face_center_to_face_perp()
vertex_vertex               => _vertex_vertex()
vertex_edge_mp              => _vertex_edge_mp()
edge_mp_edge_mp             => _edge_mp_edge_mp()
vertex_to_edge_perp         => _vertex_to_edge_perp()
edge_midpoint_to_edge_perp  => _edge_midpoint_to_edge_perp()
vertex_to_face_perp         => _vertex_to_face_perp()
edge_midpoint_to_face_perp  => _edge_midpoint_to_face_perp()
zero                        => _zero()

REGISTRY STRUCTURE
==================
The METRIC_REGISTRY dictionary has:

- Keys: public metric names (used in param_file.json)
- Values: private method names (prefixed with underscore)
- Order: insertion order (Python 3.7+)
"""

from __future__ import annotations

from typing import Dict, List


# ============================================================================
# METRIC REGISTRY - CANONICAL ORDERING
# ============================================================================
# CRITICAL: This dictionary order determines metric index mappings.
# DO NOT REORDER without understanding the consequences (see module docstring).
# ============================================================================
METRIC_REGISTRY = {
    # Index 0: Distance between face centers
    "face_center_face_center": "_face_center_face_center",
    
    # Index 1: Face center A to plane B (perpendicular)
    "face_center_to_face_perp": "_face_center_to_face_perp",
    
    # Index 2: All vertex pairs between faces
    "vertex_vertex": "_vertex_vertex",
    
    # Index 3: Vertices A to edge midpoints B
    "vertex_edge_mp": "_vertex_edge_mp",
    
    # Index 4: Edge midpoints A to edge midpoints B
    "edge_mp_edge_mp": "_edge_mp_edge_mp",
    
    # Index 5: Vertices A perpendicular to edges B
    "vertex_to_edge_perp": "_vertex_to_edge_perp",
    
    # Index 6: Edge midpoints A perpendicular to edges B
    "edge_midpoint_to_edge_perp": "_edge_midpoint_to_edge_perp",
    
    # Index 7: Vertices A perpendicular to plane B
    "vertex_to_face_perp": "_vertex_to_face_perp",
    
    # Index 8: Edge midpoints A perpendicular to plane B
    "edge_midpoint_to_face_perp": "_edge_midpoint_to_face_perp",
    
    # Index 9: Zero-distance placeholder
    "zero": "_zero",
}


def get_metric_registry() -> Dict[str, str]:
    """
    Return a copy of the runtime metric registry.

    Returning a copy prevents callers from accidentally modifying the
    canonical registry.
    """
    return dict(METRIC_REGISTRY)

#used
def get_available_metric_names() -> List[str]:
    """
    Return metric names in the exact runtime order 

    PURPOSE
    =======
    This function extracts the public metric names from METRIC_REGISTRY
    in their canonical order. This ordering is essential for index-based
    metric selection where integer indices map to metric names by position.

    ORDERING GUARANTEE
    ==================
    The returned list preserves the insertion order of METRIC_REGISTRY:
    
        Index 0 always => first metric name
        Index 1 always => second metric name
        ... etc
    
    This ordering must never change.

    Parameters
    ----------
    (none)

    Returns
    -------
    list of str
        Metric names in exact registry order:

    Notes
    -----
    - Returns fresh list each call (safe to modify)
    - Length always equals number of registered metrics
    - Order is immutable (hardcoded in METRIC_REGISTRY)
    - Used for: index-based metric selection in param_file.json

    Examples
    --------

    Convert integer indices to metric names:

        metrics = get_available_metric_names()
        selected_indices = [0, 2, 5]
        selected_names = [metrics[i] for i in selected_indices]
        # selected_names = [
        #     "face_center_face_center",
        #     "vertex_vertex",
        #     "vertex_to_edge_perp"
        # ]

    Validate metric selection from param file:

        metrics = get_available_metric_names()
        param_indices = [0, 1, 2]
        for idx in param_indices:
            if idx < 0 or idx >= len(metrics):
                raise ValueError(f"Invalid metric index: {idx}")
    """

    # Extract and return keys in insertion order
    return list(METRIC_REGISTRY.keys())
