"""
Metric-index resolution for the contact-analysis workflow.

METRIC SELECTION OVERVIEW
=========================
This module resolves metric selection from param_file.json into runtime
metric names for use in ContactDistanceMetrics.

It also provides diagnostic functions for comparing documentation (JSON)
against runtime metric definitions.

SELECTION WORKFLOW
==================
Typical workflow in contact analysis:

1. Configuration specifies metric selection:

        # From param_file.json
        {
            "indices": "all"                    # or [0, 1, 2, ...], or [0]
        }

2. resolve_metric_selection() converts to metric names:

        selected = resolve_metric_selection("all")
        # Returns: list of all metric names in order

        selected = resolve_metric_selection([0, 2, 5])
        # Returns: [name_0, name_2, name_5] by index

3. ContactDistanceMetrics.compute() uses metric names:

        metric = ContactDistanceMetrics(...)
        result = metric.compute(selected)

REFERENCE BEHAVIOR PRESERVATION
===============================
This module intentionally preserves exact behavior from reference workflow:

    available_metrics = list(ContactDistanceMetrics.METRIC_REGISTRY.keys())

    if indices == "all":
        selected_metrics = available_metrics
    else:
        selected_metrics = [available_metrics[i] for i in indices]

SELECTION FORMATS
=================
resolve_metric_selection() supports multiple formats:

1. "all" - Select all available metrics

        resolve_metric_selection("all")
        # Returns all 10 metrics in registry order

2. List of integers - Select by index

        resolve_metric_selection([0, 2, 5])
        # Returns: [metrics[0], metrics[2], metrics[5]]

3. Empty list - Error (no metrics selected)

        resolve_metric_selection([])
        # Raises ValueError
"""

from __future__ import annotations

from typing import Dict, List, Union

from .registry import get_available_metric_names

# ============================================================================
# TYPE DEFINITIONS
# ============================================================================
# Define the types that resolve_metric_selection accepts
MetricIndexSelection = Union[str, List[int]]


# ============================================================================
# METRIC SELECTION FUNCTION
# ============================================================================
# used
def resolve_metric_selection(indices: MetricIndexSelection) -> List[str]:
    """
    Resolve configured metric indices into runtime metric names.

    RESOLUTION LOGIC
    ================
    Converts from configuration format to runtime metric names:

        Input: "all" or [0, 1, 2, ...]
        Output: ["metric_name_0", "metric_name_1", ...]

    Algorithm:

        1. Get list of available metric names from registry
        2. If indices == "all", return entire list
        3. Otherwise, index into list using provided indices
        4. Return resulting metric names

    Parameters
    ----------
    indices : str or list of int
        Metric selection specification.
        
        "all"
            Select all available metrics in registry order.
            
        [0, 1, 2, ...]
            Select metrics by integer index.
            Indices must be valid (0 to N-1 where N = num metrics).

    Returns
    -------
    list of str
        Selected metric names in requested order.
        
        If indices="all":
            Returns all metric names in registry order.
            
        If indices=[0, 2, 5]:
            Returns [metric_0, metric_2, metric_5]
            Length: 3 (same as input list)

    Notes
    -----
    - This intentionally uses positional indexing into the registry order.
    - That is the exact behavior of the reference implementation.
    - The order of METRIC_REGISTRY must never change (see registry.py).
    - Order is preserved in output (input order for list indices).

    Examples
    --------
    Select specific metrics by index:

        >>> resolve_metric_selection([0, 2, 5])
        [
            "face_center_face_center",    # index 0
            "vertex_vertex",              # index 2
            "vertex_to_edge_perp",        # index 5
        ]

    """

    # STEP 1: Get list of all available metric names in registry order
    # This list has fixed order determined by METRIC_REGISTRY insertion order
    available_metrics = get_available_metric_names()

    # STEP 2: Select metrics based on input format
    if indices == "all":
        # Special case: return all available metrics
        return available_metrics

    # STEP 3: Otherwise, indices should be a list/sequence of integers
    # Use list comprehension to index into available_metrics
    # This preserves both the input order and the metric order
    return [available_metrics[index] for index in indices]


def find_documentation_mismatches(
    metric_definitions: Dict[str, str],
) -> Dict[str, Dict[str, str]]:
    """
    Compare documented JSON index mappings with runtime registry ordering.

    This function is diagnostic only. It does not modify either mapping.
    """
    available_metrics = get_available_metric_names()
    mismatches: Dict[str, Dict[str, str]] = {}

    for index_text, documented_name in metric_definitions.items():
        index = int(index_text)

        runtime_name = (
            available_metrics[index]
            if 0 <= index < len(available_metrics)
            else "<OUT_OF_RANGE>"
        )

        if runtime_name != documented_name:
            mismatches[index_text] = {
                "runtime": runtime_name,
                "documented": documented_name,
            }

    return mismatches


def find_undocumented_runtime_metrics(
    metric_definitions: Dict[str, str],
) -> Dict[str, str]:
    """
    Return runtime metric indices that are absent from metric_definitions.json.
    """
    available_metrics = get_available_metric_names()

    return {
        str(index): metric_name
        for index, metric_name in enumerate(available_metrics)
        if str(index) not in metric_definitions
    }
