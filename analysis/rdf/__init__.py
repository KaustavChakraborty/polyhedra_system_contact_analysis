# analysis/rdf/__init__.py
# ==============================================================================
# Module: analysis.rdf
# Purpose: Radial distribution function computation
#
# Exports:
#   - RDFCalculator
#   - RDFProcessor
#   - RDFInteractive
#
# Author: Contact Analysis Team
# ==============================================================================

from .calculator import RDFCalculator
from .processor import RDFProcessor
from .interactive import RDFInteractive

__all__ = [
    'RDFCalculator',
    'RDFProcessor',
    'RDFInteractive',
]

__version__ = '1.0.0'
__author__ = 'Contact Analysis Team'
