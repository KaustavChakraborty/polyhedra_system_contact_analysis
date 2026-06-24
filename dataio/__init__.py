"""
Input/output helpers for project data files.

Overview
--------
This module provides utilities for reading input files needed by the
contact-analysis workflow:

    - Shape definition files (JSON)
    - Trajectory files (GSD/POS format)

    
"""

from .shape_reader import (
    load_shape_data,
    load_shape_vertices,
)
from .trajectory_reader import (
    get_trajectory_frame_count,
    load_trajectory_frame,
)


__all__ = [
    "load_shape_data",
    "load_shape_vertices",
    "get_trajectory_frame_count",
    "load_trajectory_frame",
]