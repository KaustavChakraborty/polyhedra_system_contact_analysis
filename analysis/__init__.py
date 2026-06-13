# analysis/__init__.py
# ==============================================================================
# Module: analysis
# Purpose: Analysis and aggregation domain - statistics, RDF, order parameters
#
# Exports:
#   - RDFData, StatsResult (types)
#   - RDFCalculator, RDFProcessor, RDFInteractive (rdf sub-domain)
#   - OrderParameterCalculator (order parameters)
#   - StatisticsAggregator (statistics)
#   - AnalysisAggregator (coordinator)
#
# Author: Contact Analysis Team
# ==============================================================================

from .types import RDFData, StatsResult
from .rdf import RDFCalculator, RDFProcessor, RDFInteractive
from .order_parameters import OrderParameterCalculator
from .statistics import StatisticsAggregator
from .aggregator import AnalysisAggregator

__all__ = [
    # Data types
    'RDFData',
    'StatsResult',
    
    # RDF sub-domain
    'RDFCalculator',
    'RDFProcessor',
    'RDFInteractive',
    
    # Analysis modules
    'OrderParameterCalculator',
    'StatisticsAggregator',
    'AnalysisAggregator',
]

__version__ = '1.0.0'
__author__ = 'Contact Analysis Team'
