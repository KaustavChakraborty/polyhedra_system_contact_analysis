"""
Distance-metric registry, selection, and calculation tools.

METRICS MODULE OVERVIEW
=======================
This module provides a complete distance-metric system for the contact-analysis
workflow. It enables:

1. Distance metric computation between two polygonal faces
2. Configurable metric selection (individual or all)
3. Metric registry management and introspection
4. Documentation validation against runtime configuration

ARCHITECTURE
============
The metrics module is organized into 4 core components:

    distance_metrics.py
        ├── ContactDistanceMetrics class
        │   ├── Metric computation (10+ metric implementations)
        │   ├── Geometric utilities (normals, distances, edges)
        │   ├── Validation and error handling
        │   └── Lazy computation and caching
        │
        ├── Metric registry: public names => private methods
        ├── Input validation: points arrays, face indices
        └── Output formatting: min/max/avg/eff statistics

    registry.py
        ├── METRIC_REGISTRY: canonical runtime ordering
        ├── get_metric_registry(): returns registry copy
        └── get_available_metric_names(): returns metric list

    selection.py
        ├── resolve_metric_selection(): indices => names
        ├── find_documentation_mismatches(): compare JSON vs runtime
        └── find_undocumented_runtime_metrics(): identify gaps

    __init__.py
        ├── Public API: exports all classes and functions
        └── Module documentation and usage guidance

PUBLIC API
==========
Classes:
    ContactDistanceMetrics
        Main class for computing distance metrics between faces.
        Usage:
            metric = ContactDistanceMetrics(
                pointsA, faceA,
                pointsB, faceB,
                debug=False,
                strict=True,
                edge_distance_mode="infinite_line"
            )
            result = metric.compute("all")
        
        Output: dict {metric_name: {min, max, avg, eff}}

Functions (from registry):
    get_metric_registry()
        Returns copy of METRIC_REGISTRY dict.
        Used for: querying available metrics

    get_available_metric_names()
        Returns list of metric names in registry order.
        Used for: index-based metric selection

Functions (from selection):
    resolve_metric_selection(indices)
        Converts metric indices to metric names.
        Input: "all" or [0, 1, 2, ...]
        Output: ["face_center_face_center", ...]

    find_documentation_mismatches(metric_definitions)
        Compares JSON definitions with runtime registry.
        Used for: validation and debugging

    find_undocumented_runtime_metrics(metric_definitions)
        Identifies metrics in runtime but not JSON.
        Used for: documentation completeness checks

AVAILABLE METRICS
=================
10 distance metrics are available (in registry order):

1. face_center_face_center

2. face_center_to_face_perp

3. vertex_vertex

4. vertex_edge_mp

5. edge_mp_edge_mp

6. vertex_to_edge_perp

7. edge_midpoint_to_edge_perp

8. vertex_to_face_perp

9. edge_midpoint_to_face_perp

10. zero


METRIC SELECTION
================
Metrics are selected via resolve_metric_selection():

    # All metrics
    selected = resolve_metric_selection("all")
    # Returns: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    # Specific metrics by index
    selected = resolve_metric_selection([0, 2, 5])
    # Returns: ["face_center_face_center", "vertex_vertex", "vertex_to_edge_perp"]

    # Single metric
    selected = resolve_metric_selection([0])
    # Returns: ["face_center_face_center"]

METRIC OUTPUT FORMAT
====================
Each metric returns a dictionary with these keys:

    min     : minimum distance value across all pairs
    max     : maximum distance value across all pairs
    avg     : average distance value
    eff     : effectiveness measure (max - min) / avg

For scalar metrics (face_center_face_center, face_center_to_face_perp):
    value   : the scalar distance value
    min, max, avg, eff : same as value

USAGE WORKFLOW
==============
Typical usage in contact analysis:

    from metrics import ContactDistanceMetrics, resolve_metric_selection

    # Step 1: Resolve metric selection
    selected_metrics = resolve_metric_selection([0, 2, 5])

    # Step 2: Create metric calculator for one face pair
    metric = ContactDistanceMetrics(
        pointsA=particle_A_vertices,
        faceA=[0, 1, 2, 3],  # face A vertex indices
        pointsB=particle_B_vertices,
        faceB=[0, 1, 2],     # face B vertex indices
        debug=False,
        strict=True,
    )

    # Step 3: Compute selected metrics
    distances = metric.compute(selected_metrics)

    # Step 4: Access results
    for metric_name, metric_data in distances.items():
        print(f"{metric_name}:")
        print(f"  min: {metric_data['min']:.6f}")
        print(f"  max: {metric_data['max']:.6f}")
        print(f"  avg: {metric_data['avg']:.6f}")
        print(f"  eff: {metric_data['eff']:.6f}")

ERROR HANDLING
==============
Two error modes are available:

    strict=True (default)
        - Failed metrics raise RuntimeError
        - Workflow stops at first metric failure
        - Use for: development and debugging

    strict=False
        - Failed metrics return NaN-valued dicts
        - Workflow continues to next metric
        - Use for: production (resilience to bad geometry)

Invalid inputs raise ValueError immediately regardless of strict mode:
    - Invalid array shapes
    - Invalid face indices
    - Degenerate faces (< 3 vertices)

EDGE DISTANCE MODES
===================
For vertex_to_edge_perp and related metrics:

    edge_distance_mode="infinite_line" (default)
        - Distance to infinite line through edge
        - Preserves original workflow behavior
        - Can give very large distances for perpendiculars

    edge_distance_mode="segment"
        - Distance to finite edge segment
        - Uses clamped projection
        - More geometrically intuitive

DEBUG MODE
==========
Enable debug=True for detailed diagnostic output:

    metric = ContactDistanceMetrics(
        pointsA, faceA, pointsB, faceB,
        debug=True  # Prints detailed info during computation
    )

Output includes:
    - Array shapes and values
    - Face indices and vertex counts
    - Computed face centers and normals
    - Step-by-step metric calculations

Caching:
    - Face normals computed lazily and cached
    - Edge arrays computed once and cached
    - Face centers computed once and cached

VALIDATION AND QUALITY
======================
The module provides diagnostic functions:

    find_documentation_mismatches(metric_definitions)
        Identifies where JSON metric_definitions.json differs from runtime.
        Useful for: detecting configuration changes

    find_undocumented_runtime_metrics(metric_definitions)
        Finds metrics in registry but not in JSON.
        Useful for: ensuring complete documentation
"""


# ============================================================================
# IMPORTS
# ============================================================================
# Import the main metric computation class from distance_metrics module
from .distance_metrics import ContactDistanceMetrics

# Import metric registry and registry functions from registry module
from .registry import (
    METRIC_REGISTRY,                # Canonical registry dict
    get_available_metric_names,     # Function to get metric list
    get_metric_registry,            # Function to get registry copy
)
from .selection import (
    find_documentation_mismatches,        # Compare JSON vs runtime metrics
    find_undocumented_runtime_metrics,    # Find undocumented metrics
    resolve_metric_selection,             # Resolve indices to metric names
)


# ============================================================================
# PUBLIC API
# ============================================================================
# Define what is exported when 'from metrics import *' is used
__all__ = [
    "ContactDistanceMetrics",
    "METRIC_REGISTRY",
    "get_available_metric_names",
    "get_metric_registry",
    "resolve_metric_selection",
    "find_documentation_mismatches",
    "find_undocumented_runtime_metrics",
]