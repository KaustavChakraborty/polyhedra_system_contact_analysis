# metrics/implementations/__init__.py
# ==============================================================================
# Module: metrics.implementations
# Purpose: Concrete metric implementations
#
# Exports:
#   - FaceCenterMetric
#   - VertexBasedMetrics
#   - EdgeBasedMetrics
#   - DistanceCalculator
#
# Author: Contact Analysis Team
# ==============================================================================

from .face_center import FaceCenterMetric
from .vertex_based import VertexToFacePerpMetric, VertexVertexMetric
from .edge_based import EdgeMidpointMetric
from .distance_calc import DistanceCalculator

__all__ = [
    'FaceCenterMetric',
    'VertexToFacePerpMetric',
    'VertexVertexMetric',
    'EdgeMidpointMetric',
    'DistanceCalculator',
]

__version__ = '1.0.0'
__author__ = 'Contact Analysis Team'
