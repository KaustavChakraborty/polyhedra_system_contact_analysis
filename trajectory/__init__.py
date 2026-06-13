# trajectory/__init__.py
# ==============================================================================
# Module: trajectory
# Purpose: Trajectory file handling domain - types, reading, processing, caching
#
# Exports:
#   - Frame, Trajectory (types)
#   - GSDReader (reader)
#   - FrameCache, CacheStatistics (cache)
#   - TrajectoryProcessor (processor)
#
# Author: Contact Analysis Team
# ==============================================================================

from .types import Frame, Trajectory
from .reader import GSDReader
from .cache import FrameCache, CacheStatistics
from .processor import TrajectoryProcessor

__all__ = [
    # Data types
    'Frame',
    'Trajectory',
    
    # File reading
    'GSDReader',
    
    # Caching
    'FrameCache',
    'CacheStatistics',
    
    # Processing
    'TrajectoryProcessor',
]

__version__ = '1.0.0'
__author__ = 'Contact Analysis Team'
