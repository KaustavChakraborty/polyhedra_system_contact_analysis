# contacts/__init__.py
# ==============================================================================
# Module: contacts
# Purpose: Contact analysis domain - detection, classification, processing
#
# Exports:
#   - FacePair, ContactResult (types)
#   - OverlapDetector (detector)
#   - FaceClassifier (classifier)
#   - ContactPropertyCalculator (properties)
#   - ContactProcessor (processor)
#
# Author: Contact Analysis Team
# ==============================================================================

from .types import FacePair, ContactResult
from .detector import OverlapDetector
from .classifier import FaceClassifier
from .properties import ContactPropertyCalculator
from .processor import ContactProcessor

__all__ = [
    # Data types
    'FacePair',
    'ContactResult',
    
    # Algorithms
    'OverlapDetector',
    'FaceClassifier',
    
    # Calculations
    'ContactPropertyCalculator',
    
    # Processing
    'ContactProcessor',
]

__version__ = '1.0.0'
__author__ = 'Contact Analysis Team'
