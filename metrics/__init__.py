# metrics/__init__.py
# ==============================================================================
# Module: metrics
# Purpose: Distance metrics domain - metric computation and analysis
#
# Exports:
#   - MetricValue, MetricResult (types)
#   - MetricBase, MetricRegistry (base classes)
#   - FaceCenterMetric, VertexToFacePerpMetric, VertexVertexMetric, 
#     EdgeMidpointMetric (implementations)
#   - DistanceCalculator (utilities)
#   - MetricValidator (validation)
#   - MetricProcessor (processing)
#
# Author: Contact Analysis Team
# ==============================================================================

from .types import MetricValue, MetricResult
from .base import MetricBase, MetricRegistry
from .implementations import (
    FaceCenterMetric,
    VertexToFacePerpMetric,
    VertexVertexMetric,
    EdgeMidpointMetric,
    DistanceCalculator,
)
from .validator import MetricValidator
from .processor import MetricProcessor

__all__ = [
    # Data types
    'MetricValue',
    'MetricResult',
    
    # Base classes
    'MetricBase',
    'MetricRegistry',
    
    # Implementations
    'FaceCenterMetric',
    'VertexToFacePerpMetric',
    'VertexVertexMetric',
    'EdgeMidpointMetric',
    
    # Utilities
    'DistanceCalculator',
    
    # Validation
    'MetricValidator',
    
    # Processing
    'MetricProcessor',
]

__version__ = '1.0.0'
__author__ = 'Contact Analysis Team'
