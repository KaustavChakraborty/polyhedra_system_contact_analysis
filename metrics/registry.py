"""
Canonical runtime registry for selectable distance metrics.

Important
---------
The dictionary order below exactly preserves the insertion order of
ContactDistanceMetrics.METRIC_REGISTRY in the reference all_distances.py.

The old workflow converts integer indices into metric names using this order.
Therefore, reordering this dictionary would change scientific behavior.
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
